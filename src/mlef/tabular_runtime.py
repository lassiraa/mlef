"""Portable evaluation helper. Export this source into an experiment; ML imports are lazy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import time
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    field_validator,
    model_validator,
)

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    @field_validator("schema_version", mode="before", check_fields=False)
    @classmethod
    def version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Schema version must be an integer.")
        return value


class Split(Contract):
    id: str = Field(min_length=1)
    train: list[str] = Field(min_length=1)
    score: list[str] = Field(min_length=1)


class Protocol(Contract):
    schema_version: Literal[1] = 1
    task: Literal["binary"] = "binary"
    metric: Literal["roc_auc"] = "roc_auc"
    direction: Literal["maximize"] = "maximize"
    dataset_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    id_column: str
    target: str
    numeric: list[str]
    categorical: list[str] = Field(default_factory=list)
    positive_label: str = "1"
    seed: int = 17
    kind: Literal["stratified", "group", "time"] = "stratified"
    group_column: str | None = None
    time_column: str | None = None
    rows: list[str] = Field(min_length=2)
    splits: list[Split] = Field(min_length=2, max_length=100)

    @model_validator(mode="after")
    def assignments(self) -> Protocol:
        features = self.numeric + self.categorical
        if (
            not features
            or len(set(features)) != len(features)
            or self.target in features
            or self.id_column in features
            or self.target == self.id_column
        ):
            raise ValueError("Declare unique features excluding row ID and target.")
        universe = set(self.rows)
        if len(universe) != len(self.rows) or any(not row for row in self.rows):
            raise ValueError("Row IDs must be nonempty and unique.")
        if len({split.id for split in self.splits}) != len(self.splits):
            raise ValueError("Fold IDs must be unique.")
        scored: list[str] = []
        for split in self.splits:
            train, score = set(split.train), set(split.score)
            if (
                len(train) != len(split.train)
                or len(score) != len(split.score)
                or train & score
                or not (train | score) <= universe
            ):
                raise ValueError("Invalid, overlapping or duplicate fold assignments.")
            if self.kind != "time" and train | score != universe:
                raise ValueError("Each CV fold must partition all rows.")
            scored.extend(split.score)
        counts = Counter(scored)
        if any(count != 1 for count in counts.values()):
            raise ValueError("A row may be scored only once across these folds.")
        if self.kind != "time" and set(scored) != universe:
            raise ValueError("CV scoring assignments must cover every row exactly once.")
        if self.kind == "group" and not bool(self.group_column):
            raise ValueError("Grouped evaluation requires a group column.")
        if self.kind == "time" and not bool(self.time_column):
            raise ValueError("Temporal evaluation requires a time column.")
        return self

    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class Fold(Contract):
    id: str
    state: Literal["succeeded", "failed"]
    score: float | None = None
    seconds: float = Field(ge=0)
    train_count: int = Field(gt=0)
    score_count: int = Field(gt=0)
    error: str | None = None


class Prediction(Contract):
    fold: str
    row_id: str
    truth: float
    prediction: float


class Evaluation(Contract):
    schema_version: Literal[1] = 1
    protocol: Protocol
    complete: bool
    folds: list[Fold]
    predictions_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_object(path: Path) -> dict[str, JsonValue]:
    def unique(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(f"Duplicate JSON key: {name}")
            result[name] = value
        return result

    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object.")
    json.dumps(data, allow_nan=False)
    return TypeAdapter(dict[str, JsonValue]).validate_python(data, strict=True)


def read_rows(path: Path, identifier: str) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("CSV columns must be unique.")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            key = row.get(identifier)
            if (
                not bool(key)
                or key in rows
                or any(value is None for value in row.values())
                or None in row
            ):
                raise ValueError("CSV rows require unique nonempty IDs and consistent columns.")
            assert key is not None
            rows[key] = row
        if not rows:
            raise ValueError("Dataset is empty.")
        return rows


def validate_data(path: Path, protocol: Protocol) -> dict[str, dict[str, str]]:
    protocol = Protocol.model_validate(protocol.model_dump())
    if digest(path) != protocol.dataset_sha256:
        raise ValueError("Dataset content changed; prepare a new protocol.")
    rows = read_rows(path, protocol.id_column)
    if list(rows) != protocol.rows:
        raise ValueError("Dataset row identity/order differs from the protocol.")
    labels = {row.get(protocol.target) for row in rows.values()}
    if len(labels) != 2 or protocol.positive_label not in labels or "" in labels or None in labels:
        raise ValueError("Binary evaluation requires two labels and the declared positive class.")
    for row in rows.values():
        for feature in protocol.numeric + protocol.categorical:
            if feature not in row:
                raise ValueError(f"Unknown feature: {feature}")
        for feature in protocol.numeric:
            if row[feature] and not math.isfinite(float(row[feature])):
                raise ValueError("Numeric features must be finite or empty.")
    for split in protocol.splits:
        for partition in [split.train, split.score]:
            if {rows[key][protocol.target] for key in partition} != labels:
                raise ValueError("Each training/scoring partition must contain both classes.")
        if protocol.kind == "group":
            column = protocol.group_column
            assert column is not None
            if any(not bool(row.get(column)) for row in rows.values()) or {
                rows[key][column] for key in split.train
            } & {rows[key][column] for key in split.score}:
                raise ValueError("Groups must be present and disjoint across each split.")
        if protocol.kind == "time":
            column = protocol.time_column
            assert column is not None
            timestamps = {key: datetime.fromisoformat(row[column]) for key, row in rows.items()}
            if any(value.utcoffset() is None for value in timestamps.values()):
                raise ValueError("Temporal splits require timezone-aware ISO-8601 timestamps.")
            if max(timestamps[key] for key in split.train) >= min(
                timestamps[key] for key in split.score
            ):
                raise ValueError("Training timestamps must precede scoring timestamps.")
    return rows


def prepare(
    path: Path,
    *,
    identifier: str,
    target: str,
    numeric: list[str],
    categorical: list[str] | None = None,
    positive_label: str = "1",
    folds: int = 3,
    seed: int = 17,
) -> Protocol:
    from sklearn.model_selection import StratifiedKFold

    rows = read_rows(path, identifier)
    keys = list(rows)
    labels = [row[target] for row in rows.values()]
    if not 2 <= folds <= min(Counter(labels).values()):
        raise ValueError("Fold count exceeds the smallest class or is less than two.")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    protocol = Protocol(
        dataset_sha256=digest(path),
        id_column=identifier,
        target=target,
        numeric=numeric,
        categorical=categorical or [],
        positive_label=positive_label,
        seed=seed,
        rows=keys,
        splits=[
            Split(id=str(index), train=[keys[i] for i in train], score=[keys[i] for i in score])
            for index, (train, score) in enumerate(splitter.split(keys, labels))
        ],
    )
    validate_data(path, protocol)
    return protocol


def save(path: Path, value: BaseModel | dict[str, float]) -> None:
    temporary = path.with_suffix(".tmp")
    text = value.model_dump_json(indent=2) if isinstance(value, BaseModel) else json.dumps(value)
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(path)


def evaluate(
    path: Path,
    protocol: Protocol,
    factory: Callable[[int], BaseEstimator],
    output: Path,
) -> Evaluation:
    import numpy as np
    import pandas as pd
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score

    rows = validate_data(path, protocol)
    output.mkdir(parents=True, exist_ok=True)
    if any(
        (output / name).exists() for name in ["evaluation.json", "predictions.csv", "metrics.json"]
    ):
        raise ValueError("Evaluation destinations already exist.")
    frame = pd.DataFrame.from_dict(rows, orient="index")
    features = protocol.numeric + protocol.categorical
    for column in protocol.numeric:
        frame[column] = pd.to_numeric(frame[column].replace("", float("nan")), errors="raise")
    for column in protocol.categorical:
        frame[column] = frame[column].replace("", np.nan)
    target = (frame[protocol.target] == protocol.positive_label).astype(int)
    predictions: list[Prediction] = []
    results: list[Fold] = []
    parameters: dict[str, JsonValue] = {}
    evaluation: Evaluation | None = None
    for split in protocol.splits:
        began = time.monotonic()
        value: float | None = None
        error: str | None = None
        try:
            estimator = clone(factory(protocol.seed))
            parameters = {key: str(value)[:500] for key, value in estimator.get_params().items()}
            estimator.fit(frame.loc[split.train, features], target.loc[split.train])
            classes = list(estimator.classes_)
            scores = estimator.predict_proba(frame.loc[split.score, features])[:, classes.index(1)]
            if (
                len(scores) != len(split.score)
                or not np.isfinite(scores).all()
                or ((scores < 0) | (scores > 1)).any()
            ):
                raise ValueError("Expected one finite positive-class probability per scoring row.")
            value = float(roc_auc_score(target.loc[split.score], scores))
            predictions.extend(
                Prediction(
                    fold=split.id, row_id=key, truth=float(target.loc[key]), prediction=float(p)
                )
                for key, p in zip(split.score, scores, strict=True)
            )
        except Exception as failure:
            error = f"{type(failure).__name__}: {failure}"[:2000]
        results.append(
            Fold(
                id=split.id,
                state="failed" if error is not None else "succeeded",
                score=value,
                seconds=time.monotonic() - began,
                train_count=len(split.train),
                score_count=len(split.score),
                error=error,
            )
        )
        with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(Prediction.model_fields))
            writer.writeheader()
            writer.writerows(prediction.model_dump() for prediction in predictions)
        evaluation = Evaluation(
            protocol=protocol,
            complete=len(results) == len(protocol.splits)
            and all(f.state == "succeeded" for f in results),
            folds=results,
            predictions_sha256=digest(output / "predictions.csv"),
            parameters=parameters,
        )
        save(output / "evaluation.json", evaluation)
    assert evaluation is not None
    if evaluation.complete:
        save(
            output / "metrics.json",
            {"roc_auc_mean": sum(f.score or 0 for f in results) / len(results)},
        )
    return evaluation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ordinary Python estimators on saved folds."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("prepare")
    initialize.add_argument("--id-column", default="id")
    initialize.add_argument("--target", default="target")
    initialize.add_argument("--numeric", nargs="+", required=True)
    initialize.add_argument("--categorical", nargs="*", default=[])
    initialize.add_argument("--positive-label", default="1")
    initialize.add_argument("--folds", type=int, default=3)
    initialize.add_argument("--seed", type=int, default=17)
    execute = commands.add_parser("evaluate")
    execute.add_argument("--protocol", type=Path, default=Path("protocol.json"))
    execute.add_argument("--factory", default="model:make_estimator")
    for command in [initialize, execute]:
        command.add_argument("--data", type=Path, default=Path("data.csv"))
        command.add_argument(
            "--output", type=Path, default=Path(os.environ.get("MLEF_OUTPUT_DIR", "."))
        )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        protocol = prepare(
            args.data,
            identifier=args.id_column,
            target=args.target,
            numeric=args.numeric,
            categorical=args.categorical,
            positive_label=args.positive_label,
            folds=args.folds,
            seed=args.seed,
        )
        with (args.output / "protocol.json").open("x", encoding="utf-8") as stream:
            stream.write(protocol.model_dump_json(indent=2) + "\n")
    else:
        module, name = args.factory.split(":", 1)
        factory = getattr(importlib.import_module(module), name)
        protocol = Protocol.model_validate(read_object(args.protocol))
        result = evaluate(args.data, protocol, factory, args.output)
        raise SystemExit(0 if result.complete else 1)


if __name__ == "__main__":
    main()
