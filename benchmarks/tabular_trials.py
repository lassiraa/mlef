"""Prepare isolated inputs and score frozen tabular submissions outside agent access."""

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

from mlef.benchmark import Holdout, HoldoutScore, TabularTrial
from mlef.locking import file_lock
from mlef.records import JsonValue, read_json
from mlef.tabular_runtime import digest, measure, read_rows
from mlef.tabular_runtime import prepare as prepare_protocol

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/sklearn"
TASKS = ["classification-mixed", "classification-missing", "regression-mixed", "regression-missing"]


def prepare_tabular(destination: Path, task: str, seed: int) -> None:
    from sklearn.model_selection import train_test_split

    if task not in TASKS:
        raise ValueError(f"Choose one of: {TASKS}")
    binary = task.startswith("classification")
    source = EXAMPLE / ("data.csv" if binary else "regression.csv")
    rows = read_rows(source, "id")
    if task.endswith("missing"):
        feature = "texture" if binary else "tenure"
        for index, row in enumerate(rows.values()):
            if index % 4 == 0:
                row[feature] = ""
    keys = list(rows)
    exploration, heldout = train_test_split(
        keys,
        test_size=0.2,
        random_state=seed,
        stratify=[row["target"] for row in rows.values()] if binary else None,
    )
    destination.mkdir()
    workspace, private = destination / "agent", destination / "private"
    workspace.mkdir()
    private.mkdir()
    columns = list(next(iter(rows.values())))
    for name, selected, fields in [
        ("exploration.csv", set(exploration), columns),
        ("submission.csv", set(heldout), [name for name in columns if name != "target"]),
    ]:
        with (workspace / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(
                {key: row[key] for key in fields}
                for identifier, row in rows.items()
                if identifier in selected
            )
    protocol = prepare_protocol(
        workspace / "exploration.csv",
        identifier="id",
        target="target",
        numeric=["radius", "texture", "smoothness"] if binary else ["amount", "age", "tenure"],
        categorical=["radius_band"] if binary else ["region"],
        seed=seed,
        task="binary" if binary else "regression",
    )
    (workspace / "protocol.json").write_text(protocol.model_dump_json(indent=2), encoding="utf-8")
    fingerprint = hashlib.sha256(
        (protocol.fingerprint() + digest(workspace / "submission.csv")).encode()
    ).hexdigest()
    origin = {
        "source": "Wisconsin Diagnostic; Wolberg et al. (1993); DOI 10.24432/C5DW2B"
        if binary
        else "MLEF synthetic regression; seed 1729",
        "license": "CC BY 4.0" if binary else "Apache-2.0",
        "modifications": "Renumbered IDs, derived band, deterministic missingness; see README."
        if binary
        else "Synthetic numeric/categorical nonlinear response; optional extra missingness.",
    }
    manifest = {
        "schema_version": 1,
        "task": task,
        "task_fingerprint": fingerprint,
        "protocol_id": protocol.fingerprint(),
        "origin": origin,
        "instructions": (
            "Explore only supplied data; freeze one finalist. Final labels and scoring stay "
            "with the coordinator. A directory name is not an access boundary."
        ),
    }
    (workspace / "task.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    holdout = Holdout(
        task=protocol.task,
        metric=protocol.metric,
        direction=protocol.direction,
        task_fingerprint=fingerprint,
        protocol_id=protocol.fingerprint(),
        labels={
            key: float(row["target"] == protocol.positive_label) if binary else float(row["target"])
            for key, row in rows.items()
            if key in set(heldout)
        },
    )
    (private / "labels.json").write_text(holdout.model_dump_json(indent=2), encoding="utf-8")
    for name in ["model.py", "pyproject.toml", "uv.lock"]:
        shutil.copyfile(EXAMPLE / name, workspace / name)
    shutil.copyfile(Path(__file__).with_name("tabular_reference.py"), workspace / "reference.py")


def score_tabular(labels: Path, predictions: Path, candidate: Path) -> HoldoutScore:
    holdout = Holdout.model_validate(read_json(labels))
    submitted: dict[str, float] = {}
    with predictions.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["id", "prediction"]:
            raise ValueError("Submit exactly the columns id,prediction.")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Prediction rows must have exactly two columns.")
            identifier, value = row["id"], float(row["prediction"])
            if (
                identifier in submitted
                or not math.isfinite(value)
                or (holdout.task == "binary" and not 0 <= value <= 1)
            ):
                raise ValueError("Predictions must be unique finite valid scores.")
            submitted[identifier] = value
    if submitted.keys() != holdout.labels.keys():
        raise ValueError("Submit exactly one prediction per held-out ID.")
    predicted_hash, candidate_hash = digest(predictions), digest(candidate)
    ledger = labels.with_suffix(".score.json")
    with file_lock(labels.with_suffix(".score.lock")):
        if ledger.exists():
            original = HoldoutScore.model_validate(read_json(ledger))
            if (original.prediction_sha256, original.candidate_sha256, original.labels_sha256) != (
                predicted_hash,
                candidate_hash,
                digest(labels),
            ):
                raise ValueError(
                    "Final holdout is already scored; a different finalist is refused."
                )
            return original
        result = HoldoutScore(
            task_fingerprint=holdout.task_fingerprint,
            protocol_id=holdout.protocol_id,
            metric=holdout.metric,
            direction=holdout.direction,
            score=measure(
                list(holdout.labels.values()),
                [submitted[key] for key in holdout.labels],
                holdout.metric,
            ),
            prediction_sha256=predicted_hash,
            candidate_sha256=candidate_hash,
            labels_sha256=digest(labels),
            rows=len(submitted),
        )
        with ledger.open("x", encoding="utf-8") as stream:
            stream.write(result.model_dump_json(indent=2))
        return result


def pair_tabular(control: TabularTrial, treatment: TabularTrial) -> dict[str, JsonValue]:
    if control.condition != "control" or treatment.condition != "mlef":
        raise ValueError("Expected control followed by mlef.")
    for name in [
        "task_fingerprint",
        "protocol_id",
        "task",
        "metric",
        "direction",
        "agent",
        "model",
        "agent_settings",
        "budget_seconds",
        "seed",
        "repetition",
    ]:
        left, right = getattr(control, name), getattr(treatment, name)
        if name == "agent_settings":
            left, right = json.dumps(left, sort_keys=True), json.dumps(right, sort_keys=True)
        if left != right:
            raise ValueError(f"Unmatched trial field: {name}")
    delta = None
    if control.heldout_score is not None and treatment.heldout_score is not None:
        delta = treatment.heldout_score - control.heldout_score
    return {
        "schema_version": 2,
        "metric": control.metric,
        "direction": control.direction,
        "control_state": control.state,
        "mlef_state": treatment.state,
        "heldout_delta": delta,
        "improvement": None
        if delta is None
        else delta
        if control.direction == "maximize"
        else -delta,
        "elapsed_seconds_delta": treatment.elapsed_seconds - control.elapsed_seconds,
        "tool_output_bytes_delta": treatment.tool_output_bytes - control.tool_output_bytes,
        "intervention_count_delta": treatment.intervention_count - control.intervention_count,
        "valid_experiments_delta": treatment.valid_experiments - control.valid_experiments,
        "valid_comparisons_delta": treatment.valid_comparisons - control.valid_comparisons,
        "control_budget_exceeded": control.elapsed_seconds > control.budget_seconds,
        "mlef_budget_exceeded": treatment.elapsed_seconds > treatment.budget_seconds,
        "limitation": (
            "Report all matched repetitions, including failures. "
            "This observation alone does not establish uplift."
        ),
    }
