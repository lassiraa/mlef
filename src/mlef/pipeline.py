"""Declarative script contracts and static, non-executing pipeline validation."""

from __future__ import annotations

import ast
import heapq
import json
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Self
from uuid import uuid4

from pydantic import Field, JsonValue, field_validator, model_validator

from .artifacts import ArtifactType, safe_path
from .records import Contract

if TYPE_CHECKING:
    from .sdk import Artifact, ArtifactSource, LocalContext, StepResult

Name = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")]
Status = Literal["verified", "invalid", "deferred"]


class Input(Contract):
    name: Name
    type: ArtifactType
    format: str | None = Field(default=None, min_length=1)
    required: bool = True


class Output(Contract):
    name: Name
    type: ArtifactType
    format: str = Field(min_length=1)
    required: bool = True


class Step(Contract):
    schema_version: Literal[1] = 1
    name: Name
    inputs: list[Input] = Field(default_factory=list)
    outputs: list[Output] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_names(self) -> Self:
        for declarations in (self.inputs, self.outputs):
            names = [declaration.name for declaration in declarations]
            if len(set(names)) != len(names):
                raise ValueError("Input and output names must each be unique within a step.")
        return self

    def local(
        self,
        directory: Path,
        *,
        inputs: dict[str, ArtifactSource] | None = None,
        run_id: str | None = None,
        parameters: dict[str, JsonValue] | None = None,
    ) -> LocalContext:
        from .sdk import LocalContext

        return LocalContext(
            directory=directory,
            step=self,
            inputs=inputs or {},
            run_id=run_id if run_id is not None else str(uuid4()),
            parameters=parameters or {},
        ).prepare()

    def save(self, artifacts: list[Artifact], *, context: LocalContext) -> StepResult:
        return context.save(self, artifacts)

    def context(self) -> LocalContext:
        """Open the explicit context assigned to this script by the pipeline runner."""
        from .sdk import managed_context

        return managed_context(self)


class PipelineStep(Contract):
    name: Name
    script: str
    contract: str | None = None
    bindings: dict[Name, str] = Field(default_factory=dict)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    cache: bool = False
    cache_version: str | None = Field(default=None, min_length=1)

    _paths = field_validator("script", "contract")(
        lambda value: safe_path(value) if value is not None else None
    )


class PipelineDefinition(Contract):
    schema_version: Literal[1] = 1
    steps: list[PipelineStep] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_steps(self) -> Self:
        names = [step.name for step in self.steps]
        if len(set(names)) != len(names):
            raise ValueError("Pipeline step names must be unique.")
        return self


class ValidationCheck(Contract):
    status: Status
    code: str
    message: str
    step: str | None = None


class ValidationReport(Contract):
    schema_version: Literal[1] = 1
    status: Status
    definition: PipelineDefinition | None = None
    contracts: dict[str, Step] = Field(default_factory=dict)
    order: list[str] = Field(default_factory=list)
    checks: list[ValidationCheck] = Field(default_factory=list)


class DeferredContractError(ValueError):
    pass


def _file(root: Path, relative: str, suffix: str) -> Path:
    safe_path(relative)
    path = root / relative
    if any(part.lower() in {".git", ".mlef"} for part in Path(relative).parts):
        raise ValueError("Pipeline sources cannot use Git internals or run storage.")
    if not path.resolve().is_relative_to(root.resolve()) or any(
        parent.is_symlink() for parent in [path, *path.parents] if parent.is_relative_to(root)
    ):
        raise ValueError("Pipeline sources must remain inside the project without symlinks.")
    if path.suffix != suffix or not path.is_file():
        raise ValueError(f"Expected an existing {suffix} file: {relative}")
    return path


def _literal(node: ast.expr) -> JsonValue:
    if isinstance(node, ast.Constant) and isinstance(
        node.value, str | int | float | bool | type(None)
    ):
        return node.value
    if isinstance(node, ast.List):
        return [_literal(element) for element in node.elts]
    if isinstance(node, ast.Dict):
        result: dict[str, JsonValue] = {}
        for key, value in zip(node.keys, node.values, strict=True):
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                raise DeferredContractError("Only literal string dictionary keys are supported.")
            if key.value in result:
                raise ValueError("Duplicate literal dictionary key.")
            result[key.value] = _literal(value)
        return result
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "ArtifactType"
        and node.attr in ArtifactType.__members__
    ):
        return ArtifactType[node.attr].value
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"Step", "Input", "Output"}
        and not node.args
    ):
        arguments: dict[str, JsonValue] = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                raise DeferredContractError(
                    "Expanded constructor arguments require an explicit contract."
                )
            if keyword.arg in arguments:
                raise ValueError("Duplicate constructor keyword.")
            arguments[keyword.arg] = _literal(keyword.value)
        return arguments
    raise DeferredContractError("Computed declarations require an explicit TOML contract file.")


def _contract(script: Path) -> Step:
    tree = ast.parse(script.read_text(encoding="utf-8-sig"), filename=str(script))
    declarations = [
        statement
        for statement in tree.body
        if isinstance(statement, ast.Assign | ast.AnnAssign)
        and any(
            isinstance(target, ast.Name) and target.id == "step"
            for target in (
                statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            )
        )
    ]
    if len(declarations) != 1:
        raise DeferredContractError(
            "Declare exactly one top-level step = Step(...), or use a contract file."
        )
    declaration = declarations[0]
    value = declaration.value
    if not (
        isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "Step"
    ) or (isinstance(declaration, ast.Assign) and len(declaration.targets) != 1):
        raise DeferredContractError("Use a literal step = Step(...), or an explicit contract file.")
    return Step.model_validate_json(json.dumps(_literal(value), allow_nan=False))


def _order(dependencies: dict[str, set[str]]) -> list[str]:
    pending = {name: set(parents) for name, parents in dependencies.items()}
    ready = [name for name, parents in pending.items() if not parents]
    heapq.heapify(ready)
    order = []
    while ready:
        name = heapq.heappop(ready)
        order.append(name)
        for child, parents in pending.items():
            if name in parents:
                parents.remove(name)
                if not parents:
                    heapq.heappush(ready, child)
    if len(order) != len(pending):
        raise ValueError("Pipeline dependencies contain a cycle.")
    return order


def validate(root: Path, definition: Path = Path("mlef.toml")) -> ValidationReport:
    """Read contracts and wiring only; never import scripts or inspect payload bytes."""
    checks: list[ValidationCheck] = []
    contracts: dict[str, Step] = {}

    def check(status: Status, code: str, message: str, step: str | None = None) -> None:
        checks.append(ValidationCheck(status=status, code=code, message=message, step=step))

    try:
        path = _file(root, definition.as_posix(), ".toml")
        pipeline = PipelineDefinition.model_validate_json(
            json.dumps(tomllib.loads(path.read_text(encoding="utf-8")), allow_nan=False)
        )
    except (OSError, ValueError, TypeError, RecursionError) as error:
        check("invalid", "definition", str(error))
        return ValidationReport(status="invalid", checks=checks)
    check("verified", "definition", "Pipeline definition is structurally valid.")
    dependencies: dict[str, set[str]] = {step.name: set() for step in pipeline.steps}
    for entry in pipeline.steps:
        try:
            script = _file(root, entry.script, ".py")
            if entry.contract is not None:
                path = _file(root, entry.contract, ".toml")
                contract = Step.model_validate_json(
                    json.dumps(tomllib.loads(path.read_text(encoding="utf-8")), allow_nan=False)
                )
                ast.parse(script.read_text(encoding="utf-8-sig"), filename=str(script))
            else:
                contract = _contract(script)
            if contract.name != entry.name:
                raise ValueError("Declared step name does not match pipeline wiring.")
            contracts[entry.name] = contract
            check("verified", "contract", "Step contract is valid.", entry.name)
        except DeferredContractError as error:
            check("deferred", "contract", str(error), entry.name)
        except (OSError, ValueError, TypeError, SyntaxError, RecursionError) as error:
            check("invalid", "contract", str(error), entry.name)
    for entry in pipeline.steps:
        contract = contracts.get(entry.name)
        inputs = {item.name: item for item in contract.inputs} if contract is not None else {}
        if contract is not None:
            for name in entry.bindings.keys() - inputs.keys():
                check("invalid", "binding", f"Undeclared input: {name}", entry.name)
            for item in contract.inputs:
                if item.required and item.name not in entry.bindings:
                    check(
                        "invalid", "binding", f"Required input is unbound: {item.name}", entry.name
                    )
        for name, reference in entry.bindings.items():
            parts = reference.split(".")
            if len(parts) != 2 or parts[0] not in dependencies:
                check(
                    "invalid",
                    "reference",
                    f"Unknown step.output reference: {reference}",
                    entry.name,
                )
                continue
            producer, output_name = parts
            dependencies[entry.name].add(producer)
            upstream = contracts.get(producer)
            if upstream is None:
                check(
                    "deferred", "binding", f"Unverified producer contract: {producer}", entry.name
                )
                continue
            output = next((item for item in upstream.outputs if item.name == output_name), None)
            if output is None:
                check("invalid", "reference", f"Unknown output: {reference}", entry.name)
                continue
            if contract is None or name not in inputs:
                continue
            target = inputs[name]
            if target.type != output.type or (
                target.format is not None and target.format != output.format
            ):
                check("invalid", "compatibility", f"Incompatible binding: {name}", entry.name)
            elif target.required and not output.required:
                check(
                    "invalid", "compatibility", "Required input uses optional output.", entry.name
                )
            else:
                check("verified", "binding", f"Compatible binding: {name}", entry.name)
    order: list[str] = []
    try:
        order = _order(dependencies)
        check("verified", "graph", "Dependency graph is acyclic and deterministically ordered.")
    except ValueError as error:
        check("invalid", "graph", str(error))
    status: Status = (
        "invalid"
        if any(item.status == "invalid" for item in checks)
        else "deferred"
        if any(item.status == "deferred" for item in checks)
        else "verified"
    )
    return ValidationReport(
        status=status, definition=pipeline, contracts=contracts, order=order, checks=checks
    )
