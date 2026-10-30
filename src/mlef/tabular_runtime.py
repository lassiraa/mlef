"""Portable evaluation helper. Export this source into an experiment; ML imports are lazy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import statistics
import time
from collections import Counter
from collections.abc import Callable
from contextlib import suppress
from datetime import datetime
from itertools import groupby
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
    from pandas import DataFrame, Series
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


class Fit(Contract):
    train: list[str]
    validation: list[str] = Field(default_factory=list)
    best_iteration: int | None = None


class Fold(Contract):
    id: str
    state: Literal["succeeded", "failed"]
    score: float | None = None
    seconds: float = Field(ge=0)
    train_count: int = Field(gt=0)
    score_count: int = Field(gt=0)
    error: str | None = None
    fit: Fit | None = None


class Prediction(Contract):
    fold: str
    row_id: str
    truth: float
    prediction: float


class Evaluation(Contract):
    schema_version: Literal[2] = 2
    protocol: Protocol
    complete: bool
    folds: list[Fold]
    predictions_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    model_artifact: str | None = None


def parse_evaluation(values: JsonValue) -> Evaluation:
    if (
        isinstance(values, dict)
        and type(values.get("schema_version")) is int
        and values.get("schema_version") == 1
    ):
        values = {**values, "schema_version": 2}
    return Evaluation.model_validate(values)


class ModelRecipe(Contract):
    schema_version: Literal[1] = 1
    protocol_id: str
    features: list[str]
    id_column: str
    numeric: list[str]
    categorical: list[str]
    positive_label: str
    fit_policy: str
    versions: dict[str, str]
    parameters: dict[str, JsonValue]


def feature_frame(
    rows: dict[str, dict[str, str]], numeric: list[str], categorical: list[str]
) -> DataFrame:
    import numpy as np
    import pandas as pd

    frame = pd.DataFrame.from_dict(rows, orient="index")
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column].replace("", float("nan")), errors="raise")
    for column in categorical:
        frame[column] = frame[column].replace("", np.nan)
    return frame


def predict(data: Path, model: Path, recipe_path: Path, output: Path) -> None:
    """Explicitly load a trusted model; never invoked by offline inspection."""
    import joblib

    recipe = ModelRecipe.model_validate(read_object(recipe_path))
    frame = feature_frame(read_rows(data, recipe.id_column), recipe.numeric, recipe.categorical)
    estimator = joblib.load(model)
    scores = estimator.predict_proba(frame[recipe.features])[:, list(estimator.classes_).index(1)]
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([recipe.id_column, "positive_probability"])
        writer.writerows(zip(frame.index, scores, strict=True))


def fit_estimator(
    estimator: BaseEstimator,
    frame: DataFrame,
    target: Series,
    split: Split,
    protocol: Protocol,
) -> Fit:
    from sklearn.base import clone
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    from sklearn.pipeline import Pipeline

    features = protocol.numeric + protocol.categorical
    model = estimator[-1] if isinstance(estimator, Pipeline) else estimator
    if model.get_params().get("early_stopping_rounds") is None:
        fit_function = getattr(estimator, "fit")  # noqa: B009 - BaseEstimator does not declare fit
        fit_function(frame.loc[split.train, features], target.loc[split.train])
        return Fit(train=split.train)
    if not type(model).__module__.startswith("xgboost"):
        raise ValueError("Explicit early stopping is supported for XGBoost estimators only.")
    if protocol.kind == "time":
        column = protocol.time_column
        assert column is not None
        ordered = sorted(
            split.train, key=lambda key: datetime.fromisoformat(str(frame.loc[key, column]))
        )
        cut = max(1, int(len(ordered) * 0.8))
        train, validation = ordered[:cut], ordered[cut:]
    elif protocol.kind == "group":
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=protocol.seed)
        train_indices, val_indices = next(
            splitter.split(split.train, groups=frame.loc[split.train, protocol.group_column])
        )
        train = [split.train[index] for index in train_indices]
        validation = [split.train[index] for index in val_indices]
    else:
        train, validation = train_test_split(
            split.train, test_size=0.2, random_state=protocol.seed, stratify=target.loc[split.train]
        )
    if any(set(target.loc[partition]) != {0, 1} for partition in [train, validation]):
        raise ValueError("Early-stopping partitions must contain both classes.")
    x_train, x_validation = frame.loc[train, features], frame.loc[validation, features]
    if isinstance(estimator, Pipeline):
        preprocess = clone(estimator[:-1])
        x_train = preprocess.fit_transform(x_train, target.loc[train])
        x_validation = preprocess.transform(x_validation)
        estimator.steps = [*preprocess.steps, estimator.steps[-1]]
    model_fit = getattr(model, "fit")  # noqa: B009 - optional estimator API
    model_fit(
        x_train, target.loc[train], eval_set=[(x_validation, target.loc[validation])], verbose=False
    )
    return Fit(
        train=train,
        validation=validation,
        best_iteration=int(getattr(model, "best_iteration")),  # noqa: B009 - optional XGBoost API
    )


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def measure(truth: list[float], predictions: list[float]) -> float:
    """ROC-AUC by average ranks, also usable by offline readers without sklearn."""
    positives = sum(truth)
    negatives = len(truth) - positives
    if not positives or not negatives or len(truth) != len(predictions):
        raise ValueError("ROC-AUC needs aligned predictions and both classes.")
    rank, positive_ranks = 1, 0.0
    for _, group in groupby(sorted(zip(predictions, truth, strict=True)), key=lambda pair: pair[0]):
        labels = [label for _, label in group]
        positive_ranks += sum(labels) * (rank + (len(labels) - 1) / 2)
        rank += len(labels)
    return (positive_ranks - positives * (positives + 1) / 2) / (positives * negatives)


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
    *,
    refit: bool = False,
) -> Evaluation:
    import numpy as np
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score

    rows = validate_data(path, protocol)
    output.mkdir(parents=True, exist_ok=True)
    if any(
        (output / name).exists() for name in ["evaluation.json", "predictions.csv", "metrics.json"]
    ):
        raise ValueError("Evaluation destinations already exist.")
    frame = feature_frame(rows, protocol.numeric, protocol.categorical)
    features = protocol.numeric + protocol.categorical
    target = (frame[protocol.target] == protocol.positive_label).astype(int)
    predictions: list[Prediction] = []
    results: list[Fold] = []
    parameters: dict[str, JsonValue] = {}
    evaluation: Evaluation | None = None
    for split in protocol.splits:
        began = time.monotonic()
        value: float | None = None
        error: str | None = None
        fit: Fit | None = None
        try:
            estimator = clone(factory(protocol.seed))
            parameters = {key: str(value)[:500] for key, value in estimator.get_params().items()}
            fit = fit_estimator(estimator, frame, target, split, protocol)
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
                fit=fit,
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
        if refit:
            import joblib

            selected = clone(factory(protocol.seed))
            from sklearn.pipeline import Pipeline

            model = selected[-1] if isinstance(selected, Pipeline) else selected
            iterations = [
                fold.fit.best_iteration + 1
                for fold in results
                if fold.fit is not None and fold.fit.best_iteration is not None
            ]
            if iterations:
                model.set_params(
                    n_estimators=int(statistics.median(iterations)),
                    early_stopping_rounds=None,
                    callbacks=None,
                )
            selected.fit(frame[features], target)
            joblib.dump(selected, output / "model.joblib")
            evaluation.model_artifact = "model.joblib"
            save(output / "evaluation.json", evaluation)
            versions = {
                name: importlib.metadata.version(name)
                for name in ["scikit-learn", "pandas", "numpy"]
            }
            for package in ["xgboost-cpu", "xgboost"]:
                with suppress(importlib.metadata.PackageNotFoundError):
                    versions[package] = importlib.metadata.version(package)
            save(
                output / "model-recipe.json",
                ModelRecipe(
                    protocol_id=protocol.fingerprint(),
                    features=features,
                    id_column=protocol.id_column,
                    numeric=protocol.numeric,
                    categorical=protocol.categorical,
                    positive_label=protocol.positive_label,
                    fit_policy="full exploration data; fixed median CV tree count"
                    if iterations
                    else "full exploration data",
                    versions=versions,
                    parameters={
                        key: str(value)[:500] for key, value in selected.get_params().items()
                    },
                ),
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
    execute.add_argument("--refit", action="store_true")
    inference = commands.add_parser("predict", help="Load only trusted model artifacts.")
    inference.add_argument("--model", type=Path, default=Path("model.joblib"))
    inference.add_argument("--recipe", type=Path, default=Path("model-recipe.json"))
    for command in [initialize, execute, inference]:
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
    elif args.command == "predict":
        predict(args.data, args.model, args.recipe, args.output / "inference.csv")
    else:
        module, name = args.factory.split(":", 1)
        factory = getattr(importlib.import_module(module), name)
        protocol = Protocol.model_validate(read_object(args.protocol))
        result = evaluate(args.data, protocol, factory, args.output, refit=args.refit)
        raise SystemExit(0 if result.complete else 1)


if __name__ == "__main__":
    main()
