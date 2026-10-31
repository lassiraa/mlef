"""Offline metric discovery from existing saved evidence."""

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from .artifacts import ArtifactRef, ArtifactType
from .evidence import Result, read_result
from .execution import attempt_directory, read_pipeline
from .metrics import Metrics
from .records import Contract, json_object
from .sdk import _checked, read_step
from .storage import QueryError, artifact_path, inspect


class MetricArtifact(Contract):
    step: str
    output: str
    producer: ArtifactRef
    metrics: Metrics


class RunMetrics(Contract):
    schema_version: Literal[1] = 1
    run_id: str
    root: Result | None = None
    artifacts: list[MetricArtifact] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)


def run_metrics(
    root: Path,
    identifier: str,
    *,
    step: str | None = None,
    scope: str | None = None,
    metric: str | None = None,
    max_bytes: int = 32 * 1024 * 1024,
) -> RunMetrics:
    """Verify completed pipeline outputs and inspect JSON without loading model code."""
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer.")
    parent = inspect(root, identifier)
    result = RunMetrics(run_id=identifier)
    if step is None and scope is None:
        result.root = read_result(root, identifier)
        if metric is not None:
            result.root.metrics = {
                name: value for name, value in result.root.metrics.items() if name == metric
            }
    try:
        path = artifact_path(root, identifier, "pipeline/pipeline.json")
    except QueryError as error:
        if error.category != "unknown_artifact":
            raise
        if step is not None:
            raise QueryError("unknown_step", "Run has no captured pipeline steps.") from error
        return result
    if parent.runtime is not None and parent.runtime.cleanup_pending:
        raise QueryError("cleanup_pending", "Owned execution resources need recovery.")
    try:
        pipeline = read_pipeline(path.parent, verify=True)
        if (
            pipeline.run_id != identifier
            or parent.inputs is None
            or pipeline.source_sha256 != parent.inputs.source_sha256
        ):
            raise ValueError("Pipeline does not match its captured run.")
        if step is not None and step not in pipeline.attempts:
            raise QueryError("unknown_step", f"Unknown pipeline step: {step}")
        for name in pipeline.order:
            if step is not None and name != step:
                continue
            if pipeline.attempts[name].state != "complete":
                result.diagnostics.append(
                    f"{name}: {pipeline.attempts[name].state}; no completed metrics."
                )
                continue
            directory = attempt_directory(path.parent, name)
            saved = read_step(directory)
            for output, manifest in saved.artifacts.items():
                if manifest.type != ArtifactType.METRICS:
                    continue
                if manifest.format != "application/json" or len(manifest.files) != 1:
                    result.diagnostics.append(
                        f"{name}.{output}: requires one application/json payload."
                    )
                    continue
                entry = manifest.files[0]
                if entry.size_bytes > max_bytes:
                    raise ValueError(f"{name}.{output}: metric payload exceeds max_bytes.")
                payload = _checked(directory / "outputs" / output / entry.path)
                with payload.open("rb") as stream:
                    content = stream.read(entry.size_bytes + 1)
                if (
                    len(content) != entry.size_bytes
                    or hashlib.sha256(content).hexdigest() != entry.sha256
                ):
                    raise ValueError(f"{name}.{output}: metric bytes changed after verification.")
                values = Metrics.model_validate(json.loads(content, object_pairs_hook=json_object))
                if scope is not None and values.scope != scope:
                    continue
                if metric is not None:
                    if metric not in values.values:
                        continue
                    values.values = {metric: values.values[metric]}
                assert manifest.producer is not None
                result.artifacts.append(
                    MetricArtifact(
                        step=name,
                        output=output,
                        producer=manifest.producer,
                        metrics=values,
                    )
                )
    except QueryError:
        raise
    except (OSError, ValueError, KeyError, RecursionError) as error:
        raise QueryError("invalid_record", f"Cannot inspect pipeline metrics: {error}") from error
    return result
