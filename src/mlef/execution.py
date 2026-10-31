"""Sequential captured script execution inside the existing supervised container."""

import json
import os
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from .artifacts import safe_path
from .environment import SdkWheel, file_hash
from .pipeline import Name, PipelineDefinition, Step, validate
from .provenance import extract_source
from .records import Contract, read_json, run_id
from .sdk import (
    ArtifactSource,
    LocalContext,
    StepResult,
    _checked,
    _write_result,
    finalize_managed,
    read_step,
)


class StepAttempt(Contract):
    step: Name
    attempt: Literal[1] = 1
    script_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    state: Literal["pending", "running", "complete", "failed", "cancelled", "skipped"] = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    error: str | None = None


class PipelineRecord(Contract):
    schema_version: Literal[1] = 1
    run_id: str
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    sdk_wheel: SdkWheel
    definition_path: str
    definition: PipelineDefinition
    contracts: dict[Name, Step]
    order: list[Name]
    attempts: dict[Name, StepAttempt]
    state: Literal["preparing", "running", "complete", "failed", "cancelled"] = "preparing"
    error: str | None = None

    _run_id = field_validator("run_id")(run_id)
    _definition_path = field_validator("definition_path")(safe_path)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        names = {entry.name for entry in self.definition.steps}
        if (
            self.contracts.keys() != names
            or self.attempts.keys() != names
            or set(self.order) != names
            or len(self.order) != len(names)
        ):
            raise ValueError("Pipeline contracts, order and attempts must cover every step once.")
        for name in names:
            attempt = self.attempts[name]
            if self.contracts[name].name != name or attempt.step != name:
                raise ValueError("Pipeline step identities disagree.")
            if attempt.state == "complete" and attempt.exit_code != 0:
                raise ValueError("A completed attempt requires a successful process exit.")
        positions = {name: position for position, name in enumerate(self.order)}
        for entry in self.definition.steps:
            inputs = {item.name: item for item in self.contracts[entry.name].inputs}
            if entry.bindings.keys() - inputs.keys() or any(
                item.required and item.name not in entry.bindings for item in inputs.values()
            ):
                raise ValueError("Pipeline bindings must match required declared inputs.")
            for binding, reference in entry.bindings.items():
                parts = reference.split(".")
                if len(parts) != 2 or parts[0] not in positions:
                    raise ValueError("Pipeline bindings require known producer outputs.")
                producer, output = parts
                outputs = {item.name: item for item in self.contracts[producer].outputs}
                if output not in outputs or positions[producer] >= positions[entry.name]:
                    raise ValueError("Pipeline order must place producers before consumers.")
                source, target = outputs[output], inputs[binding]
                if (
                    source.type != target.type
                    or (target.format is not None and source.format != target.format)
                    or (target.required and not source.required)
                ):
                    raise ValueError("Pipeline input/output declarations are incompatible.")
        if self.state == "complete" and any(
            attempt.state != "complete" for attempt in self.attempts.values()
        ):
            raise ValueError("A complete pipeline requires every step to complete.")
        if (self.state in {"failed", "cancelled"}) != (self.error is not None):
            raise ValueError("Failed or cancelled pipelines require an error.")
        return self


def attempt_directory(directory: Path, name: str) -> Path:
    return _checked(directory.absolute() / "steps" / name / "1")


def _save(directory: Path, record: PipelineRecord) -> None:
    from .runner import save

    record = PipelineRecord.model_validate_json(record.model_dump_json())
    directory = _checked(directory.absolute())
    _checked(directory / "pipeline.tmp")
    save(_checked(directory / "pipeline.json"), record)


def prepare_pipeline(
    workspace: Path,
    definition: Path,
    directory: Path,
    identifier: str,
    source_sha256: str,
    sdk: SdkWheel,
) -> PipelineRecord:
    report = validate(workspace, definition)
    if report.status != "verified" or report.definition is None:
        raise ValueError("Captured pipeline validation failed: " + report.model_dump_json())
    record = PipelineRecord(
        run_id=identifier,
        source_sha256=source_sha256,
        sdk_wheel=sdk,
        definition_path=definition.as_posix(),
        definition=report.definition,
        contracts=report.contracts,
        order=report.order,
        attempts={
            entry.name: StepAttempt(
                step=entry.name, script_sha256=file_hash(workspace / entry.script)
            )
            for entry in report.definition.steps
        },
    )
    _checked(directory.absolute()).mkdir(parents=True, exist_ok=False)
    _save(directory, record)
    return record


def _context(directory: Path, record: PipelineRecord, name: str) -> LocalContext:
    entry = next(entry for entry in record.definition.steps if entry.name == name)
    inputs = {}
    for binding, reference in entry.bindings.items():
        producer, output = reference.split(".")
        if record.attempts[producer].state != "complete":
            raise ValueError("Pipeline inputs require a completed producer attempt.")
        if output not in read_step(attempt_directory(directory, producer)).artifacts:
            continue
        inputs[binding] = ArtifactSource(
            directory=attempt_directory(directory, producer), name=output
        )
    return LocalContext(
        directory=attempt_directory(directory, name),
        step=record.contracts[name],
        run_id=record.run_id,
        inputs=inputs,
    )


def read_pipeline(directory: Path, *, verify: bool = False) -> PipelineRecord:
    directory = _checked(directory.absolute())
    record = PipelineRecord.model_validate_json(
        json.dumps(read_json(_checked(directory / "pipeline.json")), allow_nan=False)
    )
    if verify:
        for name in record.order:
            if record.attempts[name].state != "complete":
                continue
            result = read_step(attempt_directory(directory, name))
            context = _context(directory, record, name)
            expected = {
                binding: read_step(source.directory).artifacts[source.name].producer
                for binding, source in context.inputs.items()
            }
            if (
                result.step != context.step
                or result.run_id != record.run_id
                or result.inputs != expected
            ):
                raise ValueError("Completed step does not match its captured contract or wiring.")
    return record


def verify_source_archive(archive: Path, record: PipelineRecord) -> None:
    archive = _checked(archive.absolute())
    if file_hash(archive) != record.source_sha256:
        raise ValueError("Captured source archive bytes changed.")
    with tempfile.TemporaryDirectory(prefix="mlef-verify-") as temporary:
        workspace = Path(temporary)
        extract_source(archive, workspace)
        report = validate(workspace, Path(record.definition_path))
        if (
            report.status != "verified"
            or report.definition != record.definition
            or report.contracts != record.contracts
            or report.order != record.order
        ):
            raise ValueError("Pipeline record disagrees with the captured source contracts.")
        for entry in record.definition.steps:
            if file_hash(workspace / entry.script) != record.attempts[entry.name].script_sha256:
                raise ValueError("Step script identity disagrees with the captured source.")


def interrupt_pipeline(directory: Path, message: str, *, cancelled: bool = False) -> None:
    from .runner import now

    if not (directory / "pipeline.json").exists():
        return
    record = read_pipeline(directory)
    if record.state in {"failed", "cancelled"}:
        return
    for attempt in record.attempts.values():
        if attempt.state == "running":
            attempt.state = "cancelled" if cancelled else "failed"
            attempt.error = message
            attempt.finished_at = now()
            path = attempt_directory(directory, attempt.step)
            if (path / "step.json").exists():
                try:
                    result = StepResult.model_validate_json(
                        json.dumps(read_json(_checked(path / "step.json")), allow_nan=False)
                    )
                    result = result.model_copy(update={"state": "failed", "error": message})
                    _write_result(path, result)
                except (OSError, ValueError):
                    pass
        elif attempt.state == "pending":
            attempt.state = "skipped"
            attempt.finished_at = now()
            attempt.error = "Upstream execution did not complete."
    record = record.model_copy(
        update={"state": "cancelled" if cancelled else "failed", "error": message}
    )
    _save(directory, record)


def execute_pipeline(workspace: Path, directory: Path) -> int:
    from .runner import now, save

    workspace, directory = workspace.absolute(), directory.absolute()
    record = read_pipeline(directory)
    if record.state != "preparing":
        raise ValueError("A captured pipeline can only start once.")
    record.state = "running"
    _save(directory, record)
    try:
        contexts = _checked(directory / "contexts")
        contexts.mkdir(exist_ok=False)
        entries = {entry.name: entry for entry in record.definition.steps}
        for name in record.order:
            attempt = record.attempts[name]
            attempt.state, attempt.started_at = "running", now()
            _save(directory, record)
            script = workspace / entries[name].script
            if file_hash(script) != attempt.script_sha256:
                raise ValueError(f"Captured script changed before execution: {name}")
            context = _context(directory, record, name)
            filename = contexts / f"{name}.json"
            save(_checked(filename), context)
            print(f"Step {name}: running", flush=True)
            process = subprocess.run(
                [sys.executable, "-I", "-m", "mlef.execution", "--step", str(script)],
                cwd=workspace,
                env={**os.environ, "MLEF_STEP_CONTEXT": str(filename)},
                check=False,
            )
            attempt.exit_code = process.returncode
            _save(directory, record)
            if process.returncode:
                raise RuntimeError(f"Step {name} exited with code {process.returncode}.")
            finalize_managed(context)
            attempt.state, attempt.finished_at = "complete", now()
            _save(directory, record)
            print(f"Step {name}: complete", flush=True)
        record.state = "complete"
        _save(directory, record)
        read_pipeline(directory, verify=True)
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        interrupt_pipeline(directory, str(error))
        print(f"mlef: {error}", file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        interrupt_pipeline(directory, "Pipeline interrupted.", cancelled=True)
        return 130


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--step":
        script = Path(sys.argv[2]).absolute()
        sys.argv = [str(script)]
        sys.path[:0] = [str(script.parent), str(Path.cwd())]
        runpy.run_path(str(script), run_name="__main__")
    elif len(sys.argv) == 2:
        raise SystemExit(execute_pipeline(Path.cwd(), Path(sys.argv[1])))
    else:
        raise SystemExit("Use mlef run to execute a captured pipeline.")


if __name__ == "__main__":
    main()
