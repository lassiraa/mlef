"""Local model versions reference verified artifacts without deserializing them."""

import hashlib
import json
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from . import storage
from .artifacts import ArtifactManifest, ArtifactType
from .execution import attempt_directory
from .locking import file_lock, run_lock
from .pipeline import Name
from .records import Contract, RunRecord, read_json, run_id
from .sdk import _checked, _verify, read_step


class ModelArtifact(Contract):
    location_run_id: str
    step: Name
    output: Name
    manifest: ArtifactManifest

    _run_id = field_validator("location_run_id")(run_id)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        producer = self.manifest.producer
        if (
            self.manifest.type != ArtifactType.MODEL
            or producer is None
            or producer.step != self.step
            or producer.output != self.output
            or self.manifest.name != self.output
        ):
            raise ValueError("Model artifact requires matching typed producer evidence.")
        return self


class ModelVersion(Contract):
    schema_version: Literal[1] = 1
    name: Name
    version: str = Field(pattern=r"^[a-f0-9]{64}$")
    registered_at: str
    artifact: ModelArtifact
    captured_run: RunRecord


class ModelListing(Contract):
    schema_version: Literal[1] = 1
    artifacts: list[ModelArtifact] = Field(default_factory=list)
    versions: list[ModelVersion] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)


def _identity(artifact: ModelArtifact) -> str:
    assert artifact.manifest.producer is not None
    return hashlib.sha256(artifact.manifest.producer.model_dump_json().encode()).hexdigest()


def _catalog(root: Path) -> Path:
    if not (root / ".mlef/runs").is_dir():
        raise storage.QueryError(
            "not_initialized", "Project is not initialized; run mlef init first."
        )
    return _checked(root / ".mlef/models")


def _name(value: str) -> str:
    # Reuse the existing portable step/output naming contract for model names.
    from pydantic import TypeAdapter

    return TypeAdapter(Name).validate_python(value)


def discover(root: Path, identifier: str) -> list[ModelArtifact]:
    from .api import pipeline_status

    pipeline = pipeline_status(identifier, root=root, verify=True)
    directory = storage.artifact_path(root, identifier, "pipeline/pipeline.json").parent
    result = []
    for name in pipeline.order:
        if pipeline.attempts[name].state != "complete":
            continue
        saved = read_step(attempt_directory(directory, name))
        result.extend(
            ModelArtifact(location_run_id=identifier, step=name, output=output, manifest=manifest)
            for output, manifest in saved.artifacts.items()
            if manifest.type == ArtifactType.MODEL
        )
    return result


def _read(path: Path) -> ModelVersion:
    version = ModelVersion.model_validate_json(
        json.dumps(read_json(_checked(path)), allow_nan=False)
    )
    if (
        version.name != path.parent.name
        or version.version != path.stem
        or version.version != _identity(version.artifact)
        or version.captured_run.id != version.artifact.location_run_id
    ):
        raise ValueError("Model version identity does not match its catalog location/evidence.")
    return version


def list_models(root: Path, identifier: str | None = None) -> ModelListing:
    catalog = _catalog(root)
    if identifier is not None:
        return ModelListing(artifacts=discover(root, identifier))
    result = ModelListing()
    if catalog.exists():
        for path in sorted(catalog.glob("*/*.json")):
            try:
                result.versions.append(_read(path))
            except (OSError, ValueError) as error:
                result.diagnostics.append(f"{path.parent.name}/{path.name}: {error}")
    return result


def register(root: Path, identifier: str, step: str, output: str, name: str) -> ModelVersion:
    from .runner import now, save

    name = _name(name)
    run_directory = _checked(storage.directory(root, identifier))
    catalog = _catalog(root)
    with run_lock(run_directory):
        parent = storage.inspect(root, identifier)
        if parent.state not in {"succeeded", "failed", "cancelled"}:
            raise ValueError("Model registration requires a terminal run.")
        artifact = next(
            (
                item
                for item in discover(root, identifier)
                if item.step == step and item.output == output
            ),
            None,
        )
        if artifact is None:
            raise ValueError("No completed typed model output matches the selected step/output.")
        record = ModelVersion(
            name=name,
            version=_identity(artifact),
            registered_at=now(),
            artifact=artifact,
            captured_run=parent,
        )
        catalog.mkdir(exist_ok=True)
        with file_lock(_checked(catalog / "catalog.lock")):
            destination = _checked(catalog / name)
            destination.mkdir(exist_ok=True)
            path = _checked(destination / f"{record.version}.json")
            if path.exists():
                existing = _read(path)
                if existing.artifact.manifest != artifact.manifest:
                    raise ValueError(
                        "Existing model version conflicts with verified artifact evidence."
                    )
                return existing
            _checked(path.with_suffix(".tmp"))
            save(path, record)
            return record


def show(root: Path, selector: str) -> ModelVersion:
    name, separator, version = selector.partition("@")
    if (
        not separator
        or len(version) != 64
        or any(char not in "0123456789abcdef" for char in version)
    ):
        raise ValueError("Expected MODEL@VERSION with the complete version hash.")
    path = _catalog(root) / _name(name) / f"{version}.json"
    if not _checked(path).exists():
        raise storage.QueryError("unknown_model", f"Unknown model version: {selector}")
    try:
        return _read(path)
    except (OSError, ValueError) as error:
        raise storage.QueryError("invalid_record", f"Cannot read model version: {error}") from error


def get(root: Path, selector: str, destination: Path) -> Path:
    record = show(root, selector)
    artifact = record.artifact
    with run_lock(_checked(storage.directory(root, artifact.location_run_id))):
        current = next(
            (
                item
                for item in discover(root, artifact.location_run_id)
                if item.step == artifact.step and item.output == artifact.output
            ),
            None,
        )
        if current != artifact:
            raise ValueError("Saved model no longer matches its registered artifact.")
        copied = storage.retrieve(
            root,
            artifact.location_run_id,
            f"pipeline/steps/{artifact.step}/1/outputs/{artifact.output}",
            destination,
        )
        _verify(copied, artifact.manifest)
        return copied
