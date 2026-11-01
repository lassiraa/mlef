"""Sequential evaluation batches over frozen source and exact model versions."""

from __future__ import annotations

import json
import shutil
import tempfile
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from pydantic import Field, JsonValue, field_validator

from .artifacts import ArtifactRef, ArtifactType
from .cache import digest, source_identity
from .environment import file_hash
from .events import Delivery
from .inspection import MetricArtifact
from .locking import file_lock
from .metrics import Scalar, Undefined
from .models import ModelVersion
from .models import show as model_version
from .pipeline import ExternalInput, Name, validate
from .provenance import archive_source, capture, extract_source
from .records import Contract, read_json, run_id
from .runner import now, relative_path, save, source_files
from .sdk import _checked
from .sdk_distribution import capture_sdk
from .snapshots import SourceSnapshot
from .storage import QueryError
from .supervision import Request, supervise

if TYPE_CHECKING:
    from .api import RunOptions


class EvaluationSpec(Contract):
    schema_version: Literal[1] = 1
    script: str
    contract: str | None = None
    step: Name = "evaluate"
    model_input: Name = "model"
    bindings: dict[Name, ExternalInput] = Field(default_factory=dict)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    includes: list[str] = Field(default_factory=list)


class EvaluationItem(Contract):
    run_id: str
    model: ModelVersion
    state: Literal["pending", "running", "succeeded", "failed", "cancelled"] = "pending"
    metrics: list[MetricArtifact] = Field(default_factory=list)
    metric_artifacts: list[ArtifactRef] = Field(default_factory=list)
    error: str | None = None

    _run_id = field_validator("run_id")(run_id)


class EvaluationContext(Contract):
    schema_version: Literal[1] = 1
    batch_id: str
    protocol_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    model: ModelVersion
    spec: EvaluationSpec

    _batch_id = field_validator("batch_id")(run_id)


class EvaluationBatch(Contract):
    schema_version: Literal[1] = 1
    id: str
    protocol_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    spec: EvaluationSpec
    created_at: str
    finished_at: str | None = None
    state: Literal["preparing", "running", "complete", "partial", "failed", "cancelled"] = (
        "preparing"
    )
    evaluations: list[EvaluationItem] = Field(min_length=1, max_length=100)
    error: str | None = None

    _id = field_validator("id")(run_id)


class EvaluationListing(Contract):
    schema_version: Literal[1] = 1
    batches: list[EvaluationBatch] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)
    total: int = 0
    next_offset: int | None = None


def _directory(root: Path, identifier: str) -> Path:
    run_id(identifier)
    return _checked(root / ".mlef/evaluations" / identifier)


def show(root: Path, identifier: str) -> EvaluationBatch:
    path = _directory(root, identifier) / "batch.json"
    if not _checked(path).exists():
        raise QueryError("unknown_evaluation", f"Unknown evaluation batch: {identifier}")
    try:
        record = EvaluationBatch.model_validate_json(json.dumps(read_json(path), allow_nan=False))
        if record.id != identifier:
            raise ValueError("Evaluation batch ID disagrees with its storage directory.")
        if len({item.run_id for item in record.evaluations}) != len(record.evaluations):
            raise ValueError("Evaluation child IDs must be distinct.")
        if record.state == "complete" and any(
            item.state != "succeeded" for item in record.evaluations
        ):
            raise ValueError("Complete batches require successful child evaluations.")
        if (
            record.state in {"complete", "partial", "failed", "cancelled"}
            and record.finished_at is None
        ):
            raise ValueError("Terminal evaluation batches require a finish timestamp.")
        return record
    except (OSError, ValueError) as error:
        raise QueryError("invalid_record", f"Cannot read evaluation batch: {error}") from error


def listing(
    root: Path, model: str | None = None, *, limit: int = 50, offset: int = 0
) -> EvaluationListing:
    if type(limit) is not int or not 1 <= limit <= 1000 or type(offset) is not int or offset < 0:
        raise ValueError("limit must be 1 through 1000 and offset must be nonnegative integers.")
    result = EvaluationListing()
    directory = _checked(root / ".mlef/evaluations")
    if directory.exists():
        for path in sorted(directory.iterdir(), key=lambda path: path.name):
            try:
                record = show(root, path.name)
                if model is None or any(
                    f"{item.model.name}@{item.model.version}" == model
                    for item in record.evaluations
                ):
                    result.batches.append(record)
            except (OSError, ValueError) as error:
                result.diagnostics.append(f"{path.name}: {error}")
    result.batches.sort(key=lambda record: (record.created_at, record.id), reverse=True)
    result.total = len(result.batches)
    end = offset + limit
    result.next_offset = end if end < result.total else None
    result.batches = result.batches[offset:end]
    return result


def _toml(value: JsonValue) -> str:
    if value is None:
        raise ValueError("TOML evaluation parameters cannot contain null.")
    if isinstance(value, dict):
        return (
            "{ "
            + ", ".join(f"{json.dumps(key)} = {_toml(item)}" for key, item in value.items())
            + " }"
        )
    if isinstance(value, list):
        return "[" + ", ".join(_toml(item) for item in value) + "]"
    return json.dumps(value, allow_nan=False)


def _definition(spec: EvaluationSpec, model: ModelVersion) -> str:
    producer = model.artifact.manifest.producer
    assert producer is not None
    bindings = {
        **spec.bindings,
        spec.model_input: ExternalInput(run_id=model.artifact.location_run_id, artifact=producer),
    }
    lines = [
        "schema_version = 2",
        "[[steps]]",
        f"name = {json.dumps(spec.step)}",
        f"script = {json.dumps(spec.script)}",
    ]
    if spec.contract is not None:
        lines.append(f"contract = {json.dumps(spec.contract)}")
    serialized_bindings = _toml(
        {name: value.model_dump(mode="json") for name, value in bindings.items()}
    )
    lines.extend(
        [
            f"parameters = {_toml(spec.parameters)}",
            f"bindings = {serialized_bindings}",
        ]
    )
    return "\n".join(lines) + "\n"


def evaluate(
    root: Path, definition: Path, selectors: list[str], options: RunOptions
) -> EvaluationBatch:
    from . import api

    if not (root / ".mlef/runs").is_dir():
        raise ValueError("Project is not initialized; run mlef init first.")
    if not 1 <= len(selectors) <= 100 or len(set(selectors)) != len(selectors):
        raise ValueError("Choose 1 through 100 distinct model versions.")
    relative_path(root, definition)
    spec = EvaluationSpec.model_validate(
        tomllib.loads(_checked(root / definition).read_text(encoding="utf-8"))
    )
    if spec.model_input in spec.bindings:
        raise ValueError("The model input is assigned by the candidate selection.")
    versions = [model_version(root, selector) for selector in selectors]
    if len({version.version for version in versions}) != len(versions):
        raise ValueError("Candidate selectors must resolve to distinct model producers.")
    files = source_files(
        root,
        Path(spec.script),
        [
            definition,
            *map(Path, spec.includes),
            *([Path(spec.contract)] if spec.contract is not None else []),
        ],
    )
    identifier = str(uuid4())
    directory = _directory(root, identifier)
    directory.mkdir(parents=True)
    base = directory / "base"
    base.mkdir()
    git, inputs = capture(root, base, files)
    sdk = capture_sdk(base / "sdk")
    batch = EvaluationBatch(
        id=identifier,
        protocol_id=digest(
            {
                "source": source_identity(base / "source.tar.gz"),
                "sdk": sdk.sha256,
                "spec": spec.model_dump(mode="json"),
            }
        ),
        spec=spec,
        created_at=now(),
        evaluations=[EvaluationItem(run_id=str(uuid4()), model=model) for model in versions],
    )
    path = directory / "batch.json"
    with file_lock(_checked(directory / "active.lock")):
        save(path, batch)
        try:
            for item in batch.evaluations:
                # ponytail: copy source/SDK per model; deduplicate if storage becomes material.
                target = directory / item.run_id
                target.mkdir()
                with tempfile.TemporaryDirectory(prefix="evaluation-source-") as temporary:
                    workspace = Path(temporary)
                    extract_source(base / "source.tar.gz", workspace)
                    generated = Path(f"mlef-evaluation-{identifier}.toml")
                    (workspace / generated).write_text(
                        _definition(spec, item.model), encoding="utf-8"
                    )
                    report = validate(workspace, generated)
                    if (
                        report.status != "verified"
                        or report.contracts[spec.step].kind != "evaluate"
                        or not any(
                            item.name == spec.model_input
                            and item.type == ArtifactType.MODEL
                            and item.required
                            for item in report.contracts[spec.step].inputs
                        )
                    ):
                        raise ValueError(
                            "Evaluation requires a typed evaluate step with a required model input "
                            f"named {spec.model_input}: " + report.model_dump_json()
                        )
                    candidate_inputs = inputs.model_copy(
                        deep=True,
                        update={
                            "source_sha256": archive_source(
                                workspace, sorted([*files, generated]), target / "source.tar.gz"
                            ),
                            "included_untracked": [
                                *inputs.included_untracked,
                                generated.as_posix(),
                            ],
                        },
                    )
                shutil.copyfile(base / "changes.patch", target / "changes.patch")
                shutil.copytree(base / "sdk", target / "sdk")
                save(
                    target / "snapshot.json",
                    SourceSnapshot(
                        git=git,
                        inputs=candidate_inputs,
                        sdk=sdk,
                        evaluation=EvaluationContext(
                            batch_id=identifier,
                            protocol_id=batch.protocol_id,
                            model=item.model,
                            spec=spec,
                        ).model_dump(mode="json"),
                        patch_sha256=file_hash(target / "changes.patch"),
                    ),
                )
            batch.state = "running"
            save(path, batch)
            for item in batch.evaluations:
                item.state = "running"
                save(path, batch)
                delivery = Delivery(None, 65536)
                try:
                    request = Request(
                        root=str(root),
                        script=f"mlef-evaluation-{identifier}.toml",
                        arguments=[],
                        includes=[],
                        outputs=[],
                        experiment=None,
                        identifier=item.run_id,
                        token=str(uuid4()),
                        quiet=True,
                        source_snapshot=str(directory / item.run_id),
                        timeout=options.timeout_seconds,
                        limits=options,
                        max_log_bytes=options.max_log_bytes,
                        max_runner_log_bytes=options.max_runner_log_bytes,
                    )
                    code = supervise(request, delivery)
                    if code == 130:
                        item.state = "cancelled"
                    else:
                        saved = api.status(item.run_id, root=root)
                        if code != 0 or saved.state != "succeeded":
                            raise ValueError(
                                saved.error.message
                                if saved.error
                                else f"Evaluation exited with code {code}."
                            )
                        evidence = api.run_metrics(item.run_id, root=root, step=spec.step).artifacts
                        if not evidence:
                            raise ValueError("Evaluation produced no structured metrics.")
                        if any(
                            artifact.metrics.model != item.model.artifact.manifest.producer
                            for artifact in evidence
                        ):
                            raise ValueError(
                                "Evaluation metrics identify a different selected model."
                            )
                        item.metric_artifacts = [artifact.producer for artifact in evidence]
                        for artifact in evidence:
                            scalars = {
                                name: value
                                for name, value in artifact.metrics.values.items()
                                if isinstance(value, Scalar | Undefined)
                            }
                            if scalars:
                                summary = artifact.metrics.model_copy(update={"values": scalars})
                                item.metrics.append(
                                    artifact.model_copy(update={"metrics": summary})
                                )
                        item.state = "succeeded"
                except (api.MlefError, OSError, ValueError) as error:
                    item.state, item.error = "failed", str(error)
                    if delivery.diagnostics:
                        item.error += " " + delivery.diagnostics.decode("utf-8", errors="replace")
                save(path, batch)
                if item.state == "cancelled":
                    break
            if any(item.state == "cancelled" for item in batch.evaluations):
                batch.state = "cancelled"
            elif all(item.state == "succeeded" for item in batch.evaluations):
                batch.state = "complete"
            elif any(item.state == "succeeded" for item in batch.evaluations):
                batch.state = "partial"
            else:
                batch.state = "failed"
        except KeyboardInterrupt:
            batch.state = "cancelled"
            for item in batch.evaluations:
                if item.state == "running":
                    item.state = "cancelled"
        except (OSError, ValueError) as error:
            batch.state, batch.error = "failed", str(error)
            raise
        finally:
            batch.finished_at = now()
            save(path, batch)
    return batch
