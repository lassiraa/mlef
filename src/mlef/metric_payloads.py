"""Explicit, bounded JSON materialization of immutable external metric references."""

import hashlib
import json
from typing import Annotated

from pydantic import Field, TypeAdapter

from .metrics import Distribution, ExternalMetric, Metric, MetricTable, Series
from .records import json_object
from .sdk import ArtifactSource, _checked, read_step

Payload = Annotated[Series | MetricTable | Distribution, Field(discriminator="kind")]


def resolve_metric(
    metric: ExternalMetric, source: ArtifactSource, *, max_bytes: int = 32 * 1024 * 1024
) -> Series | MetricTable | Distribution:
    """Verify the exact producer/file and reject malformed or contradictory metric payloads."""
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("Metric payload max_bytes must be a positive integer.")
    metric = ExternalMetric.model_validate_json(metric.model_dump_json())
    record = read_step(source.directory)
    manifest = record.artifacts.get(source.name)
    if manifest is None or manifest.producer != metric.artifact:
        raise ValueError("External metric requires its exact declared producer artifact.")
    if manifest.format != metric.format:
        raise ValueError("External metric payload format disagrees with its artifact.")
    entry = next((entry for entry in manifest.files if entry.path == metric.path), None)
    if entry is None:
        raise ValueError("External metric path is not in the artifact inventory.")
    if entry.size_bytes > max_bytes:
        raise ValueError("External metric payload exceeds max_bytes.")
    path = _checked(source.directory.absolute() / "outputs" / source.name / metric.path)
    with path.open("rb") as stream:
        content = stream.read(entry.size_bytes + 1)
    if len(content) > max_bytes:
        raise ValueError("External metric payload exceeds max_bytes.")
    if len(content) != entry.size_bytes or hashlib.sha256(content).hexdigest() != entry.sha256:
        raise ValueError("External metric bytes changed after artifact verification.")
    try:
        values = json.loads(content, object_pairs_hook=json_object)
        json.dumps(values, allow_nan=False)
        result = TypeAdapter(Payload).validate_python(values)
    except RecursionError as error:
        raise ValueError("Metric payload JSON nesting exceeds the supported depth.") from error
    if result.kind != metric.representation or any(
        getattr(result, name) != getattr(metric, name) for name in Metric.model_fields
    ):
        raise ValueError("External metric semantics disagree with its referenced payload.")
    return result
