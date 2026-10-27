"""Supervised toy classification: majority baseline versus a learned version feature."""

import argparse
import csv
import json
import os
from pathlib import Path

from packaging.version import Version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="store_true")
    args = parser.parse_args()
    with Path("data.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    majority = int(sum(int(row["target"]) for row in train) > len(train) / 2)
    thresholds = sorted({Version(row["version"]) for row in train})
    threshold = min(
        thresholds,
        key=lambda value: sum(
            int(Version(row["version"]) >= value) != int(row["target"]) for row in train
        ),
    )
    predictions = [
        int(Version(row["version"]) >= threshold) if args.candidate else majority
        for row in validation
    ]
    accuracy = sum(
        value == int(row["target"]) for value, row in zip(predictions, validation, strict=True)
    ) / len(validation)
    output = Path(os.environ["MLEF_OUTPUT_DIR"])
    (output / "metrics.json").write_text(json.dumps({"accuracy": accuracy}), encoding="utf-8")
    (output / "predictions.json").write_text(json.dumps(predictions), encoding="utf-8")
    print(f"Validation accuracy: {accuracy}")


if __name__ == "__main__":
    main()
