"""Bounded, descriptive agent feedback over saved exploration evidence."""

from pathlib import Path
from typing import Literal

from .records import Contract, read_json
from .storage import QueryError, artifact_path
from .tabular_comparison import (
    MAX_EVIDENCE_BYTES,
    TabularComparison,
    compare_tabular,
    load_evaluation,
)
from .tabular_runtime import Profile


class ErrorSlice(Contract):
    name: str
    count: int
    baseline_mae: float
    candidate_mae: float
    delta: float


class Diagnostics(Contract):
    schema_version: Literal[1] = 1
    comparison: TabularComparison
    error_measure: Literal["absolute probability error", "absolute target error", "unavailable"]
    minimum_count: int
    slices: list[ErrorSlice]
    omitted_slices: int
    limitation: str = "Exploratory overlapping slices do not establish causal or held-out gains."


def read_profile(root: Path, identifier: str) -> Profile:
    evaluation, _ = load_evaluation(root, identifier)
    path = artifact_path(root, identifier, "profile.json")
    try:
        if not path.is_file() or path.stat().st_size > MAX_EVIDENCE_BYTES:
            raise ValueError("Profile exceeds the inspection limit.")
        profile = Profile.model_validate(read_json(path))
        if profile.protocol_id != evaluation.protocol.fingerprint():
            raise ValueError("Profile belongs to a different evaluation protocol.")
        return profile
    except (ValueError, OSError) as error:
        raise QueryError("invalid_tabular_profile", str(error)) from error


def diagnose(
    root: Path, baseline: str, candidate: str, *, minimum_count: int = 20, limit: int = 20
) -> Diagnostics:
    if type(minimum_count) is not int or not 2 <= minimum_count <= 1_000_000:
        raise QueryError("arguments", "Minimum slice count must be between 2 and 1000000.")
    comparison = compare_tabular(root, baseline, candidate, limit=limit)
    result = Diagnostics(
        comparison=comparison,
        error_measure="unavailable",
        minimum_count=minimum_count,
        slices=[],
        omitted_slices=0,
    )
    if comparison.compatibility != "matched":
        return result
    first, left = load_evaluation(root, baseline)
    _, right = load_evaluation(root, candidate)
    binary = first.protocol.task == "binary"
    result.error_measure = "absolute probability error" if binary else "absolute target error"
    candidate_rows = {(row.fold, row.row_id): row for row in right}
    groups = {"all": left}
    if binary:
        groups.update(
            {f"target_{value}": [row for row in left if row.truth == value] for value in (0, 1)}
        )
        for index in range(4):
            groups[f"baseline_probability_band_{index}"] = [
                row
                for row in left
                if index / 4 <= row.prediction <= (index + 1) / 4
                and (index == 0 or row.prediction > index / 4)
            ]
    else:
        ordered = sorted(left, key=lambda row: (row.prediction, row.fold, row.row_id))
        for index in range(4):
            groups[f"baseline_prediction_quartile_{index}"] = ordered[
                index * len(ordered) // 4 : (index + 1) * len(ordered) // 4
            ]
    eligible: list[ErrorSlice] = []
    for name, rows in groups.items():
        if len(rows) < minimum_count:
            continue
        a = sum(abs(row.truth - row.prediction) for row in rows) / len(rows)
        b = sum(
            abs(row.truth - candidate_rows[row.fold, row.row_id].prediction) for row in rows
        ) / len(rows)
        eligible.append(
            ErrorSlice(name=name, count=len(rows), baseline_mae=a, candidate_mae=b, delta=b - a)
        )
    result.slices = eligible[:limit]
    result.omitted_slices = len(groups) - len(result.slices)
    return result
