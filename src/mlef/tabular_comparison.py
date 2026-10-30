"""Offline comparisons of actual saved fold evidence; no ML imports or model loading."""

import csv
from pathlib import Path
from typing import Literal

from pydantic import Field

from .records import Contract, read_json
from .storage import QueryError, artifact_path, inspect
from .tabular_runtime import Evaluation, Prediction, digest, measure, parse_evaluation

MAX_EVIDENCE_BYTES = 32 * 1024 * 1024


class FoldComparison(Contract):
    id: str
    baseline: float
    candidate: float
    delta: float


class TabularComparison(Contract):
    schema_version: Literal[1] = 1
    baseline_id: str
    candidate_id: str
    compatibility: Literal["matched", "incompatible", "unknown"]
    reasons: list[str] = Field(default_factory=list)
    metric: str | None = None
    direction: Literal["maximize", "minimize"] | None = None
    baseline_mean: float | None = None
    candidate_mean: float | None = None
    delta: float | None = None
    observation: Literal["improved", "regressed", "unchanged", "unavailable"] = "unavailable"
    baseline_seconds: float | None = None
    candidate_seconds: float | None = None
    folds: list[FoldComparison] = Field(default_factory=list)
    omitted_folds: int = 0
    artifacts: dict[str, str] = Field(default_factory=dict)
    limitations: list[str] = Field(
        default_factory=lambda: [
            "Saved evidence is checked for consistency, not independently attested execution.",
            "Fold deltas are descriptive; overlapping CV folds are not independent samples.",
            "Adaptive validation gains do not establish final held-out improvement.",
        ]
    )


def load_evaluation(root: Path, identifier: str) -> tuple[Evaluation, list[Prediction]]:
    inspect(root, identifier)
    try:
        path = artifact_path(root, identifier, "evaluation.json")
        predictions = artifact_path(root, identifier, "predictions.csv")
        if any(
            not item.is_file() or item.stat().st_size > MAX_EVIDENCE_BYTES
            for item in [path, predictions]
        ):
            raise ValueError("Tabular evidence must be regular files of at most 32 MiB each.")
        result = parse_evaluation(read_json(path))
        protocol = result.protocol
        expected = {split.id: split for split in protocol.splits}
        seen: set[str] = set()
        successful: dict[str, set[str]] = {}
        for fold in result.folds:
            if fold.id not in expected or fold.id in seen:
                raise ValueError("Unknown or duplicate fold results.")
            seen.add(fold.id)
            split = expected[fold.id]
            if fold.fit is not None:
                fit = fold.fit
                if (
                    len(set(fit.train)) != len(fit.train)
                    or len(set(fit.validation)) != len(fit.validation)
                    or set(fit.train) & set(fit.validation)
                    or set(fit.train) | set(fit.validation) != set(split.train)
                ):
                    raise ValueError(
                        "Inner fitting partitions do not partition outer training rows."
                    )
            if fold.train_count != len(split.train) or fold.score_count != len(split.score):
                raise ValueError("Fold counts do not match actual assignments.")
            if fold.state == "succeeded":
                if fold.score is None or fold.error is not None or not 0 <= fold.score <= 1:
                    raise ValueError("Invalid successful-fold score/diagnostic.")
                successful[fold.id] = set(split.score)
            elif fold.score is not None or fold.error is None:
                raise ValueError("Failed folds cannot supply scores.")
        complete = len(successful) == len(expected)
        if result.complete != complete:
            raise ValueError("Completion contradicts saved fold outcomes.")
        if digest(predictions) != result.predictions_sha256:
            raise ValueError("Prediction bytes do not match their saved fingerprint.")
        rows: list[Prediction] = []
        observed: dict[str, set[str]] = {key: set() for key in successful}
        with predictions.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(Prediction.model_fields):
                raise ValueError("Unexpected prediction columns.")
            for values in reader:
                prediction = Prediction(
                    fold=values["fold"],
                    row_id=values["row_id"],
                    truth=float(values["truth"]),
                    prediction=float(values["prediction"]),
                )
                if (
                    prediction.fold not in successful
                    or prediction.row_id not in successful[prediction.fold]
                    or prediction.row_id in observed[prediction.fold]
                    or prediction.truth not in (0, 1)
                    or not 0 <= prediction.prediction <= 1
                ):
                    raise ValueError("Invalid or duplicate row-aligned prediction.")
                observed[prediction.fold].add(prediction.row_id)
                rows.append(prediction)
        if observed != successful:
            raise ValueError("Predictions do not cover all successful scoring assignments.")
        for fold in result.folds:
            if fold.id in successful:
                scored = [row for row in rows if row.fold == fold.id]
                recomputed = measure(
                    [row.truth for row in scored], [row.prediction for row in scored]
                )
                if fold.score is None or abs(recomputed - fold.score) > 1e-10:
                    raise ValueError("Reported score differs from saved predictions.")
        return result, rows
    except QueryError:
        raise
    except (ValueError, OSError, KeyError, TypeError) as error:
        raise QueryError("invalid_tabular_evidence", str(error)) from error


def compare_tabular(
    root: Path, baseline: str, candidate: str, *, limit: int = 20
) -> TabularComparison:
    if type(limit) is not int or not 1 <= limit <= 100:
        raise QueryError("arguments", "Fold summary limit must be between 1 and 100.")
    first_record, second_record = inspect(root, baseline), inspect(root, candidate)
    result = TabularComparison(
        baseline_id=baseline, candidate_id=candidate, compatibility="unknown"
    )
    try:
        first, left_predictions = load_evaluation(root, baseline)
        second, right_predictions = load_evaluation(root, candidate)
    except QueryError as error:
        if error.category == "unknown_artifact":
            result.reasons.append(
                "Tabular evidence is missing; generic run data is still available."
            )
            return result
        raise
    result.artifacts = {
        "baseline": f".mlef/runs/{baseline}/artifacts/evaluation.json",
        "candidate": f".mlef/runs/{candidate}/artifacts/evaluation.json",
    }
    if first.protocol.fingerprint() != second.protocol.fingerprint():
        result.compatibility = "incompatible"
        result.reasons.append("Dataset, actual folds or scoring protocol differ.")
        return result
    if (
        not first.complete
        or not second.complete
        or any(record.state != "succeeded" for record in [first_record, second_record])
    ):
        result.reasons.append("Both runs and every required fold must succeed.")
        return result
    labels = {(row.fold, row.row_id): row.truth for row in left_predictions}
    if labels != {(row.fold, row.row_id): row.truth for row in right_predictions}:
        result.compatibility = "incompatible"
        result.reasons.append("Saved scoring labels differ.")
        return result
    left = {fold.id: fold for fold in first.folds}
    right = {fold.id: fold for fold in second.folds}
    pairs: list[FoldComparison] = []
    for split in first.protocol.splits:
        a, b = left[split.id].score, right[split.id].score
        assert a is not None
        assert b is not None
        pairs.append(FoldComparison(id=split.id, baseline=a, candidate=b, delta=b - a))
    result.compatibility = "matched"
    result.metric = first.protocol.metric + "_mean"
    result.direction = first.protocol.direction
    result.baseline_mean = sum(pair.baseline for pair in pairs) / len(pairs)
    result.candidate_mean = sum(pair.candidate for pair in pairs) / len(pairs)
    result.delta = result.candidate_mean - result.baseline_mean
    improvement = result.delta
    result.observation = (
        "improved" if improvement > 0 else "regressed" if improvement < 0 else "unchanged"
    )
    result.baseline_seconds = sum(fold.seconds for fold in first.folds)
    result.candidate_seconds = sum(fold.seconds for fold in second.folds)
    result.folds, result.omitted_folds = pairs[:limit], max(0, len(pairs) - limit)
    return result
