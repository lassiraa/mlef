"""Verified scalar evaluation comparison; no implicit selection or significance claims."""

import json
import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import Field

from .artifacts import ArtifactRef
from .cache import digest, environment_identity, source_identity
from .comparison import Compatibility
from .evaluations import EvaluationContext, _definition, show
from .inspection import MetricArtifact
from .metrics import Interval, Scalar, Undefined
from .models import _identity
from .pipeline import PipelineDefinition
from .records import Contract, json_object, run_id
from .sdk import _checked
from .storage import QueryError, directory, inspect


class EvaluationRow(Contract):
    run_id: str
    model: str | None = None
    state: str = "unknown"
    value: float | None = None
    representation: str | None = None
    scope: str | None = None
    output: str | None = None
    metric_artifact: ArtifactRef | None = None
    dataset: ArtifactRef | None = None
    split: ArtifactRef | None = None
    sample_count: int | None = None
    interval: Interval | None = None
    direction: Literal["maximize", "minimize", "target", "none"] | None = None
    target: float | None = None
    delta: float | None = None
    relative_delta: float | None = None
    compatibility: Compatibility = "unknown"
    reasons: list[str] = Field(default_factory=list)
    observation: Literal["improved", "regressed", "unchanged", "unavailable"] = "unavailable"
    eligible: bool = False
    rank: int | None = None
    duplicate_of: str | None = None


class EvaluationComparison(Contract):
    schema_version: Literal[1] = 1
    baseline_id: str
    metric: str
    scope: str | None = None
    output: str | None = None
    rows: list[EvaluationRow]
    limitations: list[str] = Field(
        default_factory=lambda: [
            "Compatibility checks verified declarations, not actual row use or leakage freedom.",
            "Score differences and supplied intervals do not establish statistical significance.",
            "Runtime measurements are not normalized for hardware or measurement conditions.",
            "Repeated model producers and reused evidence are not independent model candidates.",
        ]
    )


@dataclass
class _Evidence:
    row: EvaluationRow
    context: EvaluationContext | None = None
    artifact: MetricArtifact | None = None
    environment: str | None = None
    missing: list[str] = field(default_factory=list)


def _load(
    root: Path, identifier: str, metric: str, scope: str | None, output: str | None
) -> _Evidence:
    from . import api

    result = _Evidence(EvaluationRow(run_id=identifier))
    try:
        parent = inspect(root, identifier)
        result.row.state = parent.state
        location = directory(root, identifier)
        context_path = _checked(location / "evaluation.json")
        with context_path.open("rb") as stream:
            content = stream.read(1024 * 1024 + 1)
        if len(content) > 1024 * 1024:
            raise ValueError("Evaluation context exceeds 1 MiB.")
        context = EvaluationContext.model_validate_json(
            json.dumps(json.loads(content, object_pairs_hook=json_object), allow_nan=False)
        )
        pipeline = api.pipeline_status(identifier, root=root, verify=True)
        expected = PipelineDefinition.model_validate_json(
            json.dumps(tomllib.loads(_definition(context.spec, context.model)), allow_nan=False)
        )
        if (
            pipeline.definition != expected
            or pipeline.contracts[context.spec.step].kind != "evaluate"
            or pipeline.order != [context.spec.step]
            or pipeline.definition_path != f"mlef-evaluation-{context.batch_id}.toml"
            or context.model.version != _identity(context.model.artifact)
        ):
            raise ValueError("Evaluation context disagrees with its captured model/pipeline.")
        identity = digest(
            {
                "source": source_identity(
                    location / "source.tar.gz", exclude={pipeline.definition_path}
                ),
                "sdk": pipeline.sdk_wheel.sha256,
                "spec": context.spec.model_dump(mode="json"),
            }
        )
        if identity != context.protocol_id:
            raise ValueError("Evaluation recipe identity does not match captured source/SDK.")
        evidence = api.run_metrics(
            identifier, root=root, step=context.spec.step, scope=scope, metric=metric
        ).artifacts
        matches = [
            item
            for item in evidence
            if metric in item.metrics.values and (output is None or item.output == output)
        ]
        if len(matches) > 1:
            raise QueryError(
                "ambiguous_metric", "Metric name is ambiguous; select --scope and --output."
            )
        result.context = context
        result.row.model = f"{context.model.name}@{context.model.version}"
        if matches:
            result.artifact = matches[0]
            result.row.scope = result.artifact.metrics.scope
            result.row.output = result.artifact.output
            result.row.metric_artifact = result.artifact.producer
            result.row.dataset = result.artifact.metrics.dataset
            result.row.split = result.artifact.metrics.split
            if result.artifact.metrics.model != context.model.artifact.manifest.producer:
                raise ValueError("Metric evidence identifies a different model version.")
            value = result.artifact.metrics.values[metric]
            result.row.representation = value.kind
            if isinstance(value, Scalar):
                result.row.value = value.value
                result.row.direction = value.direction
                result.row.target = value.target
                result.row.sample_count = value.sample_count
                result.row.interval = value.interval
            else:
                result.missing.append(
                    f"Metric is undefined: {value.reason}"
                    if isinstance(value, Undefined)
                    else "Selected metric is not an available scalar."
                )
        else:
            result.missing.append("Selected metric evidence is unavailable.")
        if parent.state != "succeeded":
            result.missing.append("Evaluation run did not succeed.")
        try:
            result.environment = environment_identity(location)
        except (OSError, ValueError) as error:
            result.missing.append(f"Complete environment evidence is unavailable: {error}")
    except QueryError as error:
        if error.category == "ambiguous_metric":
            raise
        result.missing.append(str(error))
    except (api.MlefError, OSError, ValueError, RecursionError) as error:
        result.context = None
        result.missing.append(f"Evaluation evidence could not be verified: {error}")
    return result


def _score(row: EvaluationRow) -> float:
    assert row.value is not None
    if row.direction == "target":
        assert row.target is not None
        return abs(row.value - row.target)
    return -row.value if row.direction == "maximize" else row.value


def _rank(rows: list[EvaluationRow]) -> None:
    for row in rows:
        row.rank = None
    ranked = sorted(
        (row for row in rows if row.eligible), key=lambda row: (_score(row), row.run_id)
    )
    previous_score: float | None = None
    rank = 0
    for position, row in enumerate(ranked, 1):
        score = _score(row)
        if previous_score is None or score != previous_score:
            rank = position
        row.rank, previous_score = rank, score


def compare(
    root: Path,
    identifiers: list[str],
    *,
    baseline: str,
    metric: str,
    scope: str | None = None,
    output: str | None = None,
) -> EvaluationComparison:
    if (
        not 1 <= len(identifiers) <= 100
        or len(set(identifiers)) != len(identifiers)
        or baseline not in identifiers
        or not metric.strip()
    ):
        raise ValueError(
            "Choose 1 through 100 distinct evaluation IDs, an included baseline and a metric."
        )
    for identifier in identifiers:
        run_id(identifier)
    observations = {
        identifier: _load(root, identifier, metric, scope, output) for identifier in identifiers
    }
    first = observations[baseline]
    seen_models: dict[str, str] = {}
    seen_evidence: dict[str, str] = {}
    for identifier in [
        baseline,
        *(identifier for identifier in identifiers if identifier != baseline),
    ]:
        current = observations[identifier]
        row = current.row
        conflicts: list[str] = []
        missing = (
            [*(f"Baseline: {reason}" for reason in first.missing), *current.missing]
            if identifier != baseline
            else list(first.missing)
        )
        if first.context is None or current.context is None:
            missing.append("Verified evaluation recipe identity is unavailable.")
        elif first.context.protocol_id != current.context.protocol_id:
            conflicts.append("Evaluation recipe identities differ.")
        if (
            first.environment is not None
            and current.environment is not None
            and first.environment != current.environment
        ):
            conflicts.append("Captured environments or resource controls differ.")
        if first.artifact is not None and current.artifact is not None:
            left, right = first.artifact.metrics, current.artifact.metrics
            if left.dataset is None or right.dataset is None:
                missing.append("Dataset identity is unavailable.")
            elif left.dataset != right.dataset:
                conflicts.append("Dataset identities differ.")
            if left.split != right.split:
                conflicts.append("Split identities differ.")
            if isinstance(left.values[metric], Scalar) and isinstance(right.values[metric], Scalar):
                conflicts.extend(left.semantic_differences(right))
                if row.sample_count is None or first.row.sample_count is None:
                    missing.append("Metric sample counts are unavailable.")
        if first.row.value is not None and row.value is not None:
            if first.row.direction == "target" and not math.isfinite(_score(first.row)):
                missing.append("Baseline target distance exceeds finite numeric representation.")
            delta = row.value - first.row.value
            row.delta = delta if math.isfinite(delta) else None
            if row.delta is not None and first.row.value != 0:
                relative = row.delta / abs(first.row.value)
                row.relative_delta = relative if math.isfinite(relative) else None
        row.compatibility = (
            "incompatible" if conflicts else "unknown" if missing else "declared-compatible"
        )
        row.reasons = list(dict.fromkeys([*conflicts, *missing]))
        if first.row.value is not None and row.value is not None and row.delta is None:
            row.reasons.append("Delta exceeds finite numeric representation.")
        if (
            first.row.value is not None
            and first.row.value != 0
            and row.delta is not None
            and row.relative_delta is None
        ):
            row.reasons.append("Relative delta exceeds finite numeric representation.")
        row.eligible = (
            row.compatibility == "declared-compatible"
            and row.direction in {"maximize", "minimize", "target"}
            and row.value is not None
        )
        if row.eligible and not math.isfinite(_score(row)):
            row.eligible = False
            row.reasons.append("Target distance exceeds finite numeric representation.")
        if row.compatibility == "declared-compatible" and row.direction == "none":
            row.reasons.append("Metric has no declared ordering.")
        if row.eligible:
            if math.isfinite(_score(first.row)):
                a, b = _score(first.row), _score(row)
                row.observation = "unchanged" if a == b else "improved" if b < a else "regressed"
            assert current.context is not None
            assert current.artifact is not None
            model_identity = current.context.model.version
            evidence_identity = current.artifact.producer.model_dump_json()
            duplicate = seen_models.get(model_identity) or seen_evidence.get(evidence_identity)
            if duplicate is not None:
                row.duplicate_of, row.eligible = duplicate, False
                row.observation = "unavailable"
                row.reasons.append(
                    "Repeated model producer or reused metric evidence; excluded from ranking."
                )
            else:
                seen_models[model_identity] = identifier
                seen_evidence[evidence_identity] = identifier
    _rank([item.row for item in observations.values()])
    return EvaluationComparison(
        baseline_id=baseline,
        metric=metric,
        scope=scope,
        output=output,
        rows=[observations[identifier].row for identifier in identifiers],
    )


def leaderboard(
    root: Path,
    batch_id: str,
    *,
    baseline: str,
    metric: str,
    scope: str | None = None,
    output: str | None = None,
) -> EvaluationComparison:
    batch = show(root, batch_id)
    result = compare(
        root,
        [item.run_id for item in batch.evaluations],
        baseline=baseline,
        metric=metric,
        scope=scope,
        output=output,
    )
    states = {item.run_id: item.state for item in batch.evaluations}
    for row in result.rows:
        if row.state == "unknown":
            row.state = states[row.run_id]
        if states[baseline] != "succeeded" or states[row.run_id] != "succeeded":
            row.eligible = False
            row.observation = "unavailable"
            if row.compatibility == "declared-compatible":
                row.compatibility = "unknown"
            row.reasons.append("Saved batch did not mark the evaluation/baseline successful.")
    _rank(result.rows)
    result.rows.sort(key=lambda row: (row.rank is None, row.rank or 0, row.run_id))
    return result
