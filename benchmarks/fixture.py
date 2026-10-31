"""Prepare matched tabular trials and score held-out predictions outside agent access."""

import argparse
import json
from pathlib import Path

from pydantic import JsonValue

from mlef.benchmark import HoldoutScore, TabularTrial
from mlef.records import read_json


def read_trial(path: Path) -> TabularTrial:
    return TabularTrial.model_validate(read_json(path))


def record(path: Path, destination: Path, score_ledger: Path | None = None) -> None:
    trial = read_trial(path)
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


def pair(control_path: Path, mlef_path: Path) -> dict[str, JsonValue]:
    from tabular_trials import pair_tabular

    return pair_tabular(read_trial(control_path), read_trial(mlef_path))


def main() -> None:
    from tabular_trials import TASKS, prepare_tabular, report_trials, score_tabular

    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
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
    if args.command == "record":
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
