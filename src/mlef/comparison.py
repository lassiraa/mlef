"""Compare saved declarations and aggregate observations, never infer significance."""

import json
import math
from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue, TypeAdapter

from .environment import Environment
from .evidence import Result, read_result
from .records import Contract, RunRecord, read_json
from .storage import QueryError, directory, inspect

Compatibility = Literal["declared-compatible", "incompatible", "unknown"]


class ValueChange(Contract):
    baseline_present: bool
    candidate_present: bool
    baseline: JsonValue = None
    candidate: JsonValue = None


class MetricComparison(Contract):
    name: str
    baseline: int | float | None
    candidate: int | float | None
    delta: int | float | None = None
    relative_delta: float | None = None
    direction: Literal["maximize", "minimize"] | None = None
    compatibility: Compatibility
    reasons: list[str]
    observation: Literal["improved", "regressed", "unchanged", "unavailable"] = "unavailable"


class Comparison(Contract):
    schema_version: Literal[1] = 1
    baseline_id: str
    candidate_id: str
    compatibility: Compatibility
    metrics: list[MetricComparison]
    parameters: dict[str, ValueChange]
    provenance: dict[str, ValueChange]
    baseline_seconds: float | None
    candidate_seconds: float | None
    baseline_experiment: JsonValue
    candidate_experiment: JsonValue
    artifacts: dict[str, list[str]]
    limitations: list[str] = Field(
        default_factory=lambda: [
            "Compatibility checks user declarations, not actual split use or leakage prevention.",
            "Aggregate deltas do not establish statistical significance or causal attribution.",
            "Failed or unavailable evidence cannot support improvement claims.",
        ]
    )


def changes(
    baseline: dict[str, JsonValue], candidate: dict[str, JsonValue]
) -> dict[str, ValueChange]:
    return {
        name: ValueChange(
            baseline_present=name in baseline,
            candidate_present=name in candidate,
            baseline=baseline.get(name),
            candidate=candidate.get(name),
        )
        for name in sorted(baseline.keys() | candidate.keys())
        if (name in baseline) != (name in candidate)
        or json.dumps(baseline.get(name), sort_keys=True)
        != json.dumps(candidate.get(name), sort_keys=True)
    }


def environment_values(root: Path, identifier: str) -> JsonValue:
    path = directory(root, identifier) / "environment.json"
    if not path.exists():
        return None
    try:
        if path.is_symlink():
            raise ValueError("Environment metadata must not be a symbolic link.")
        values = Environment.model_validate(read_json(path)).model_dump(mode="json")
        return TypeAdapter(JsonValue).validate_python(values)
    except (OSError, ValueError) as error:
        raise QueryError(
            "invalid_environment", f"Cannot read environment for {identifier}: {error}"
        ) from error


def metric_comparison(
    name: str, baseline: RunRecord, candidate: RunRecord, first: Result, second: Result
) -> MetricComparison:
    conflicts: list[str] = []
    missing: list[str] = []

    def declaration(field: str, left: object, right: object) -> None:
        if (
            left is None
            or right is None
            or (isinstance(left, str) and not left.strip())
            or (isinstance(right, str) and not right.strip())
        ):
            missing.append(f"Missing {field} declaration.")
        elif left != right:
            conflicts.append(f"Conflicting {field} declarations.")

    left, right = baseline.experiment, candidate.experiment
    declaration(
        "dataset fingerprint",
        left.dataset.fingerprint if left and left.dataset else None,
        right.dataset.fingerprint if right and right.dataset else None,
    )
    le, re = left.evaluation if left else None, right.evaluation if right else None
    for field in ["protocol_id", "split_fingerprint", "seeds"]:
        declaration(f"evaluation {field}", getattr(le, field, None), getattr(re, field, None))
    declaration(
        f"{name} definition",
        le.metric_definitions.get(name) if le else None,
        re.metric_definitions.get(name) if re else None,
    )
    ld, rd = (
        left.metric_directions.get(name) if left else None,
        right.metric_directions.get(name) if right else None,
    )
    declaration(f"{name} direction", ld, rd)
    a, b = first.metrics.get(name), second.metrics.get(name)
    if a is None or b is None:
        missing.append("Metric is unavailable in one or both runs.")
    if first.evidence_status != "available" or second.evidence_status != "available":
        missing.append("Normalized evidence is missing or invalid.")
    if baseline.state != "succeeded" or candidate.state != "succeeded":
        missing.append("Both runs must have succeeded to support an improvement claim.")
    compatibility: Compatibility = (
        "incompatible" if conflicts else "unknown" if missing else "declared-compatible"
    )
    row = MetricComparison(
        name=name,
        baseline=a,
        candidate=b,
        compatibility=compatibility,
        reasons=conflicts + missing,
        direction=ld if ld == rd else None,
    )
    if a is not None and b is not None:
        try:
            delta = b - a
            if isinstance(delta, float) and not math.isfinite(delta):
                raise OverflowError
            row.delta = delta
            if a == 0:
                row.reasons.append("Relative delta is unavailable for a zero baseline.")
            else:
                relative = delta / abs(a)
                if not math.isfinite(relative):
                    raise OverflowError
                row.relative_delta = relative
        except OverflowError:
            row.reasons.append("A delta exceeded finite numeric representation.")
        if compatibility == "declared-compatible":
            row.observation = (
                "unchanged"
                if a == b
                else ("improved" if (b > a) == (ld == "maximize") else "regressed")
            )
    return row


def compare(root: Path, baseline_id: str, candidate_id: str) -> Comparison:
    baseline, candidate = inspect(root, baseline_id), inspect(root, candidate_id)
    first, second = read_result(root, baseline_id), read_result(root, candidate_id)
    rows = [
        metric_comparison(name, baseline, candidate, first, second)
        for name in sorted(first.metrics.keys() | second.metrics.keys())
    ]
    compatibility: Compatibility = "unknown"
    if any(row.compatibility == "incompatible" for row in rows):
        compatibility = "incompatible"
    elif rows and all(row.compatibility == "declared-compatible" for row in rows):
        compatibility = "declared-compatible"
    left, right = baseline.experiment, candidate.experiment
    comparison = Comparison(
        baseline_id=baseline_id,
        candidate_id=candidate_id,
        compatibility=compatibility,
        metrics=rows,
        parameters=changes(left.parameters if left else {}, right.parameters if right else {}),
        provenance=changes(
            {
                "git": baseline.git.model_dump(mode="json") if baseline.git else None,
                "inputs": baseline.inputs.model_dump(mode="json") if baseline.inputs else None,
                "environment": environment_values(root, baseline_id),
            },
            {
                "git": candidate.git.model_dump(mode="json") if candidate.git else None,
                "inputs": candidate.inputs.model_dump(mode="json") if candidate.inputs else None,
                "environment": environment_values(root, candidate_id),
            },
        ),
        baseline_seconds=baseline.execution_seconds,
        candidate_seconds=candidate.execution_seconds,
        baseline_experiment=left.model_dump(mode="json") if left else None,
        candidate_experiment=right.model_dump(mode="json") if right else None,
        artifacts={"baseline": first.artifacts, "candidate": second.artifacts},
    )
    if not (first.metrics.keys() & second.metrics.keys()):
        comparison.compatibility = "unknown"
        comparison.limitations.append("No shared metrics are available to compare.")
    for side, result in [("baseline", first), ("candidate", second)]:
        for diagnostic in result.diagnostics:
            comparison.limitations.append(f"{side} {diagnostic.code}: {diagnostic.message}")
    for side, record in [("baseline", baseline), ("candidate", candidate)]:
        if record.git is None or record.inputs is None:
            comparison.limitations.append(f"{side} source/Git provenance is unavailable.")
    return comparison
