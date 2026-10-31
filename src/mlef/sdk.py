"""Local script artifact persistence with explicit contexts and verified handoff."""

import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from types import TracebackType
from typing import Literal, Self
from uuid import uuid4

from pydantic import ConfigDict, Field, PrivateAttr, field_validator, model_validator

from .artifacts import (
    ArtifactFile,
    ArtifactManifest,
    ArtifactRef,
    ArtifactType,
    GenericMetadata,
    Metadata,
    safe_path,
)
from .locking import file_lock
from .pipeline import Name, Step
from .records import Contract, read_json
from .records import run_id as validate_run_id


class Artifact(Contract):
    """A file or directory in a declared output's managed directory."""

    name: Name
    type: ArtifactType
    format: str = Field(min_length=1)
    path: Path
    metadata: Metadata = Field(default_factory=GenericMetadata, discriminator="kind")
    serializer: Callable[[Path], None] | None = Field(default=None, exclude=True)


class StepResult(Contract):
    schema_version: Literal[1] = 1
    run_id: str
    step: Step
    state: Literal["running", "complete", "failed"]
    inputs: dict[Name, ArtifactRef] = Field(default_factory=dict)
    artifacts: dict[Name, ArtifactManifest] = Field(default_factory=dict)
    error: str | None = None

    _run_id = field_validator("run_id")(validate_run_id)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        inputs = {item.name: item for item in self.step.inputs}
        outputs = {item.name: item for item in self.step.outputs}
        if self.inputs.keys() - inputs.keys() or self.artifacts.keys() - outputs.keys():
            raise ValueError("Step record contains undeclared inputs or outputs.")
        if (self.state == "failed") != (self.error is not None):
            raise ValueError("Only failed step records require an error.")
        if self.state == "complete" and (
            any(item.required and item.name not in self.inputs for item in self.step.inputs)
            or any(item.required and item.name not in self.artifacts for item in self.step.outputs)
        ):
            raise ValueError("Complete step records require every required input and output.")
        for name, manifest in self.artifacts.items():
            output = outputs[name]
            if manifest.type == ArtifactType.FILE and len(manifest.files) != 1:
                raise ValueError("File artifacts require exactly one inventory file.")
            if (
                manifest.name != name
                or manifest.type != output.type
                or manifest.format != output.format
            ):
                raise ValueError("Saved artifact does not match its output declaration.")
            if manifest.producer is None or (
                manifest.producer.run_id != self.run_id or manifest.producer.step != self.step.name
            ):
                raise ValueError("Saved artifact producer does not match its step record.")
        return self


def _checked(path: Path) -> Path:
    for component in [path, *path.parents]:
        try:
            attributes = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(attributes.st_mode) or getattr(attributes, "st_file_attributes", 0) & 0x400:
            raise ValueError("Artifact paths must not contain symlinks or reparse points.")
        if stat.S_ISREG(attributes.st_mode) and attributes.st_nlink != 1:
            raise ValueError("Artifact paths must not contain hard links.")
        if not (stat.S_ISREG(attributes.st_mode) or stat.S_ISDIR(attributes.st_mode)):
            raise ValueError("Artifact paths require regular files or directories.")
    return path


def _within(base: Path, path: Path) -> Path:
    candidate = path.absolute()
    if ".." in candidate.parts or not candidate.is_relative_to(base):
        raise ValueError("Artifact path escapes its managed output directory.")
    return _checked(candidate)


def _inventory(directory: Path) -> list[ArtifactFile]:
    _checked(directory)
    if not directory.is_dir():
        raise ValueError("Artifact directory is missing.")
    inventory = []
    for path in sorted(
        directory.rglob("*"), key=lambda item: item.relative_to(directory).as_posix()
    ):
        _checked(path)
        attributes = path.stat()
        if stat.S_ISDIR(attributes.st_mode):
            continue
        if not stat.S_ISREG(attributes.st_mode) or attributes.st_nlink != 1:
            raise ValueError("Artifacts require regular files without hard links.")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        inventory.append(
            ArtifactFile(
                path=path.relative_to(directory).as_posix(),
                size_bytes=attributes.st_size,
                sha256=digest,
            )
        )
    return inventory


def _verify(directory: Path, manifest: ArtifactManifest) -> None:
    if _inventory(directory) != sorted(manifest.files, key=lambda item: item.path):
        raise ValueError(f"Artifact bytes or inventory changed: {manifest.name}")


def _output_names(directory: Path, names: set[str]) -> None:
    outputs = _checked(directory / "outputs")
    if outputs.exists() and not outputs.is_dir():
        raise ValueError("Output storage must be a directory.")
    actual = {path.name for path in outputs.iterdir()} if outputs.is_dir() else set()
    if actual != names:
        raise ValueError("Output directories must exactly match saved artifact names.")


def _write_result(directory: Path, result: StepResult) -> None:
    _checked(directory)
    text = result.model_dump_json(indent=2) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=".step-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        Path(temporary).replace(_checked(directory / "step.json"))
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_step(directory: Path) -> StepResult:
    """Accept only complete records whose output inventories still match actual bytes."""
    directory = _checked(directory.absolute())
    result = StepResult.model_validate_json(
        json.dumps(read_json(_checked(directory / "step.json")), allow_nan=False)
    )
    if result.state != "complete":
        raise ValueError("Failed or incomplete steps cannot supply reusable artifacts.")
    _output_names(directory, set(result.artifacts))
    for name, manifest in result.artifacts.items():
        _verify(directory / "outputs" / name, manifest)
    return result


class ArtifactSource(Contract):
    """A named artifact in a completed local attempt, verified when consumed."""

    directory: Path
    name: Name

    _directory = field_validator("directory")(lambda value: value.absolute())


def load_artifacts(directory: Path) -> dict[str, ArtifactSource]:
    result = read_step(directory)
    return {
        name: ArtifactSource(directory=directory.absolute(), name=name) for name in result.artifacts
    }


class LocalContext(Contract):
    """One explicit, single-writer attempt. Existing directories are never overwritten."""

    model_config = ConfigDict(frozen=True)

    directory: Path
    step: Step
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    inputs: dict[Name, ArtifactSource] = Field(default_factory=dict)
    _prepared: bool = PrivateAttr(default=False)
    _finished: bool = PrivateAttr(default=False)
    _entered: bool = PrivateAttr(default=False)
    _saved: bool = PrivateAttr(default=False)
    _contract: Step | None = PrivateAttr(default=None)
    _bound: dict[str, ArtifactManifest] = PrivateAttr(default_factory=dict)
    _sources: dict[str, ArtifactSource] = PrivateAttr(default_factory=dict)
    _artifacts: dict[str, ArtifactManifest] = PrivateAttr(default_factory=dict)

    _run_id = field_validator("run_id")(validate_run_id)
    _directory = field_validator("directory")(lambda value: value.absolute())

    def _record(
        self, state: Literal["running", "complete", "failed"], error: str | None = None
    ) -> StepResult:
        assert self._contract is not None
        result = StepResult(
            run_id=self.run_id,
            step=self._contract,
            state=state,
            inputs={name: self.input_reference(name) for name in self._bound},
            artifacts=self._artifacts,
            error=error,
        )
        result = StepResult.model_validate_json(result.model_dump_json())
        _write_result(self.directory.absolute(), result)
        return result

    def _fail(self, error: BaseException) -> None:
        self._finished = True
        with suppress(OSError, ValueError):
            self._record("failed", f"{type(error).__name__}: {error}")

    def prepare(self) -> Self:
        if self._prepared:
            return self
        self._contract = Step.model_validate_json(self.step.model_dump_json())
        directory = _checked(self.directory.absolute())
        directory.mkdir(parents=True, exist_ok=False)
        self._prepared = True
        try:
            self._record("running")
            declarations = {item.name: item for item in self._contract.inputs}
            if self.inputs.keys() - declarations.keys() or any(
                item.required and item.name not in self.inputs for item in self._contract.inputs
            ):
                raise ValueError("Input bindings must satisfy declared required inputs only.")
            for name, source in self.inputs.items():
                source = source.model_copy(deep=True)
                record = read_step(source.directory)
                if source.name not in record.artifacts:
                    raise ValueError(f"Unknown source artifact: {source.name}")
                manifest = record.artifacts[source.name]
                declaration = declarations[name]
                if manifest.type != declaration.type or (
                    declaration.format is not None and manifest.format != declaration.format
                ):
                    raise ValueError(f"Incompatible input artifact: {name}")
                target = directory / "inputs" / name
                shutil.copytree(source.directory / "outputs" / source.name, target, symlinks=True)
                _verify(target, manifest)
                for path in target.rglob("*"):
                    if path.is_file():
                        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
                self._bound[name] = manifest
                self._sources[name] = source
            self._record("running")
        except BaseException as error:
            self._fail(error)
            raise
        return self

    def input_reference(self, name: str) -> ArtifactRef:
        if name not in self._bound or self._bound[name].producer is None:
            raise ValueError(f"Unknown bound input: {name}")
        reference = self._bound[name].producer
        assert reference is not None
        return reference.model_copy(deep=True)

    def input_path(self, name: str, relative: str | None = None) -> Path:
        if not self._prepared or name not in self._bound:
            raise ValueError(f"Unknown bound input: {name}")
        directory = self.directory.absolute() / "inputs" / name
        _verify(directory, self._bound[name])
        path = (
            directory if relative is None else _within(directory, directory / safe_path(relative))
        )
        if not path.exists():
            raise ValueError(f"Unknown input payload path: {relative}")
        return path

    def output_path(self, name: str, relative: str | None = None) -> Path:
        self.prepare()
        if self._finished or self._contract is None:
            raise ValueError("This attempt is already finalized.")
        if name not in {item.name for item in self._contract.outputs}:
            raise ValueError(f"Undeclared output: {name}")
        directory = _checked(self.directory.absolute() / "outputs" / name)
        target = directory if relative is None else directory / safe_path(relative)
        _within(directory, target)
        (target if relative is None else target.parent).mkdir(parents=True, exist_ok=True)
        return target

    def _verify_inputs(self) -> None:
        for name, manifest in self._bound.items():
            _verify(self.directory.absolute() / "inputs" / name, manifest)
            source = self._sources[name]
            current = read_step(source.directory).artifacts.get(source.name)
            if current != manifest:
                raise ValueError(f"Source input changed: {name}")

    def save(self, step: Step, artifacts: list[Artifact]) -> StepResult:
        self.prepare()
        with file_lock(_checked(self.directory.absolute() / "active.lock")):
            if self._finished:
                raise ValueError("This attempt is already finalized.")
            try:
                assert self._contract is not None
                if step.model_dump_json() != self._contract.model_dump_json():
                    raise ValueError("Step declaration changed after opening its context.")
                names = [artifact.name for artifact in artifacts]
                declarations = {item.name: item for item in self._contract.outputs}
                if len(set(names)) != len(names) or set(names) - declarations.keys():
                    raise ValueError("Artifacts must name distinct declared outputs.")
                if any(item.required and item.name not in names for item in self._contract.outputs):
                    raise ValueError("Missing required output artifact.")
                self._verify_inputs()
                for artifact in artifacts:
                    declaration = declarations[artifact.name]
                    if artifact.type != declaration.type or artifact.format != declaration.format:
                        raise ValueError("Artifact type or format does not match its declaration.")
                    root = self.directory.absolute() / "outputs" / artifact.name
                    path = _within(root, artifact.path)
                    if artifact.serializer is not None:
                        if path.exists():
                            raise ValueError("Serializers cannot overwrite existing payloads.")
                        path.parent.mkdir(parents=True, exist_ok=True)
                        artifact.serializer(path)
                    _checked(path)
                    if not path.exists() or not (path.is_file() or path.is_dir()):
                        raise ValueError("Artifact payload is missing or is not a file/directory.")
                    if artifact.type == ArtifactType.FILE and not path.is_file():
                        raise ValueError("File artifacts require a file payload.")
                    if artifact.type == ArtifactType.DIRECTORY and not path.is_dir():
                        raise ValueError("Directory artifacts require a directory payload.")
                    inventory = _inventory(root)
                    if path.is_file() and any(root / item.path != path for item in inventory):
                        raise ValueError(
                            "A file artifact must not contain additional payload files."
                        )
                    if path.is_dir() and any(
                        not (root / item.path).is_relative_to(path) for item in inventory
                    ):
                        raise ValueError(
                            "Artifact contains files outside its declared payload directory."
                        )
                    manifest = ArtifactManifest(
                        name=artifact.name,
                        type=artifact.type,
                        format=artifact.format,
                        files=inventory,
                        metadata=artifact.metadata,
                    )
                    manifest.producer = ArtifactRef(
                        run_id=self.run_id,
                        step=self._contract.name,
                        output=artifact.name,
                        sha256=manifest.content_sha256(),
                    )
                    self._artifacts[artifact.name] = ArtifactManifest.model_validate_json(
                        manifest.model_dump_json()
                    )
                _output_names(self.directory.absolute(), set(names))
                self._verify_inputs()
                for name, manifest in self._artifacts.items():
                    _verify(self.directory.absolute() / "outputs" / name, manifest)
                result = self._record("running" if self._entered else "complete")
                self._finished = True
                self._saved = True
            except BaseException as error:
                self._fail(error)
                raise
        return result if self._entered else read_step(self.directory)

    def __enter__(self) -> Self:
        self.prepare()
        if self._entered or self._finished:
            raise ValueError("This attempt is already active or finalized.")
        self._entered = True
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        if error is not None:
            self._fail(error)
        elif not self._saved:
            error = ValueError("Step context exited without saving its outputs.")
            self._fail(error)
            raise error
        else:
            try:
                assert self._contract is not None
                if self.step.model_dump_json() != self._contract.model_dump_json():
                    raise ValueError("Step declaration changed after opening its context.")
                self._verify_inputs()
                _output_names(self.directory.absolute(), set(self._artifacts))
                for name, manifest in self._artifacts.items():
                    _verify(self.directory.absolute() / "outputs" / name, manifest)
                self._record("complete")
                read_step(self.directory)
            except BaseException as error:
                self._fail(error)
                raise
        self._entered = False
        return False
