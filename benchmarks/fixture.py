"""Prepare matched trial data and score held-out predictions outside the agent workspace."""

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue, TypeAdapter

from mlef.benchmark import HoldoutScore, TabularTrial
from mlef.records import Contract, read_json


class Trial(Contract):
    schema_version: Literal[1] = 1
    task_fingerprint: str
    condition: Literal["control", "mlef"]
    agent: str
    model: str
    budget_seconds: int = Field(gt=0)
    seed: int
    repetition: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)
    tool_output_bytes: int = Field(ge=0)
    intervention_count: int = Field(ge=0)
    valid_comparisons: int = Field(ge=0)
    time_to_first_valid_gain_seconds: float | None = Field(default=None, ge=0)
    heldout_accuracy: float = Field(ge=0, le=1)


def prepare(destination: Path, seed: int) -> None:
    destination.mkdir()  # Never overwrite an existing trial.
    workspace, private = destination / "agent", destination / "private"
    workspace.mkdir()
    private.mkdir()
    rng = random.Random(seed)
    labels = {}
    for split, count in [("train", 40), ("validation", 20), ("test", 20)]:
        with (workspace / f"{split}.csv").open("w", encoding="utf-8", newline="") as stream:
            fields = ["id", "x1", "x2"] + ([] if split == "test" else ["target"])
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for index in range(count):
                x1, x2 = rng.randint(-10, 10), rng.randint(-10, 10)
                identifier = f"{split}-{index}"
                target = int(x1 + 2 * x2 > 0)
                row: dict[str, str | int] = {"id": identifier, "x1": x1, "x2": x2}
                if split == "test":
                    labels[identifier] = target
                else:
                    row["target"] = target
                writer.writerow(row)
    fingerprint = hashlib.sha256(
        b"".join(
            (workspace / name).read_bytes() for name in ["train.csv", "validation.csv", "test.csv"]
        )
    ).hexdigest()
    (workspace / "task.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_fingerprint": fingerprint,
                "target": "target",
                "metric": "accuracy",
                "direction": "maximize",
                "seed": seed,
                "instructions": "Train, validate, then submit test predictions by ID.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (private / "labels.json").write_text(json.dumps(labels), encoding="utf-8")


def score(private_labels: Path, predictions: Path) -> float:
    adapter = TypeAdapter(dict[str, int])
    labels = adapter.validate_python(read_json(private_labels), strict=True)
    submitted = adapter.validate_python(read_json(predictions), strict=True)
    if (
        not labels
        or labels.keys() != submitted.keys()
        or any(value not in (0, 1) for value in submitted.values())
    ):
        raise ValueError("Submit exactly one binary prediction for each held-out ID.")
    return sum(submitted[key] == value for key, value in labels.items()) / len(labels)


def read_trial(path: Path) -> Trial | TabularTrial:
    values = read_json(path)
    if isinstance(values, dict) and values.get("schema_version") == 2:
        return TabularTrial.model_validate(values)
    return Trial.model_validate(values)


def record(path: Path, destination: Path, score_ledger: Path | None = None) -> None:
    trial = read_trial(path)
    if isinstance(trial, TabularTrial):
        if trial.state == "completed":
            if score_ledger is None:
                raise ValueError("Completed v2 trials require the coordinator's score ledger.")
            score = HoldoutScore.model_validate(read_json(score_ledger))
            for field in [
                "task_fingerprint",
                "protocol_id",
                "metric",
                "direction",
                "prediction_sha256",
                "candidate_sha256",
            ]:
                if getattr(trial, field) != getattr(score, field):
                    raise ValueError(f"Trial differs from frozen score ledger: {field}")
            if trial.heldout_score != score.score:
                raise ValueError("Trial held-out score differs from its ledger.")
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(trial.model_dump_json(indent=2) + "\n")
        return
    if trial.elapsed_seconds > trial.budget_seconds:
        raise ValueError("Trial exceeded its matched wall-clock budget.")
    first = trial.time_to_first_valid_gain_seconds
    if first is not None and (first > trial.elapsed_seconds or trial.valid_comparisons == 0):
        raise ValueError("First valid gain requires a valid comparison within the measured trial.")
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(trial.model_dump_json(indent=2) + "\n")


def pair(control_path: Path, mlef_path: Path) -> dict[str, JsonValue]:
    control, treatment = read_trial(control_path), read_trial(mlef_path)
    if isinstance(control, TabularTrial) and isinstance(treatment, TabularTrial):
        from tabular_trials import pair_tabular

        return pair_tabular(control, treatment)
    if not isinstance(control, Trial) or not isinstance(treatment, Trial):
        raise ValueError("Cannot pair legacy accuracy trials with v2 tabular trials.")
    if control.condition != "control" or treatment.condition != "mlef":
        raise ValueError("Expected a control trial followed by a mlef trial.")
    for name in ["task_fingerprint", "agent", "model", "budget_seconds", "seed", "repetition"]:
        if getattr(control, name) != getattr(treatment, name):
            raise ValueError(f"Unmatched trial field: {name}")
    return {
        "schema_version": 1,
        "repetition": control.repetition,
        "heldout_accuracy_delta": treatment.heldout_accuracy - control.heldout_accuracy,
        "elapsed_seconds_delta": treatment.elapsed_seconds - control.elapsed_seconds,
        "tool_output_bytes_delta": treatment.tool_output_bytes - control.tool_output_bytes,
        "intervention_count_delta": treatment.intervention_count - control.intervention_count,
        "limitation": "One paired observation does not establish product uplift or significance.",
    }


def main() -> None:
    from tabular_trials import TASKS, prepare_tabular, report_trials, score_tabular

    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("prepare")
    generate.add_argument("destination", type=Path)
    generate.add_argument("--seed", type=int, default=31415)
    evaluate = commands.add_parser("score")
    evaluate.add_argument("labels", type=Path)
    evaluate.add_argument("predictions", type=Path)
    capture = commands.add_parser("record")
    capture.add_argument("input", type=Path)
    capture.add_argument("output", type=Path)
    capture.add_argument("--score-ledger", type=Path)
    matched = commands.add_parser("pair")
    matched.add_argument("control", type=Path)
    matched.add_argument("mlef", type=Path)
    tabular = commands.add_parser("prepare-tabular")
    tabular.add_argument("destination", type=Path)
    tabular.add_argument("--task", choices=TASKS, required=True)
    tabular.add_argument("--seed", type=int, default=17)
    scoring = commands.add_parser("score-tabular")
    scoring.add_argument("labels", type=Path)
    scoring.add_argument("predictions", type=Path)
    scoring.add_argument("--candidate", type=Path, required=True)
    reporting = commands.add_parser("report")
    reporting.add_argument("directory", type=Path)
    reporting.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.destination, args.seed)
    elif args.command == "score":
        print(json.dumps({"heldout_accuracy": score(args.labels, args.predictions)}))
    elif args.command == "record":
        record(args.input, args.output, args.score_ledger)
    elif args.command == "prepare-tabular":
        prepare_tabular(args.destination, args.task, args.seed)
    elif args.command == "score-tabular":
        print(score_tabular(args.labels, args.predictions, args.candidate).model_dump_json())
    elif args.command == "report":
        print(json.dumps(report_trials(args.directory, args.limit)))
    else:
        print(json.dumps(pair(args.control, args.mlef)))


if __name__ == "__main__":
    main()
