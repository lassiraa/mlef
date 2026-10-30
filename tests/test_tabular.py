import csv
import os
from pathlib import Path
from typing import Self

import pytest
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from typing_extensions import override

from mlef import api, tabular
from mlef.tabular_runtime import evaluate, prepare, read_object, validate_data

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/sklearn"


@pytest.fixture
def data(tmp_path: Path) -> Path:
    path = tmp_path / "data.csv"
    path.write_text(
        "id,x,target,group,date\n"
        + "".join(
            f"{index},{index % 7},{index % 2},{index // 2},2026-01-{index + 1:02d}T00:00:00Z\n"
            for index in range(24)
        )
    )
    return path


def protocol_for(path: Path) -> tabular.Protocol:
    return prepare(path, identifier="id", target="target", numeric=["x"])


def linear(seed: int) -> BaseEstimator:
    return Pipeline([("impute", SimpleImputer()), ("model", LogisticRegression(random_state=seed))])


def test_reuses_folds_and_aligned_predictions(data: Path, tmp_path: Path) -> None:
    protocol = protocol_for(data)
    first = evaluate(data, protocol, linear, tmp_path / "first")
    second = evaluate(data, protocol, linear, tmp_path / "second")
    assert first.complete
    assert [fold.score for fold in first.folds] == [fold.score for fold in second.folds]
    assert protocol.fingerprint() == protocol_for(data).fingerprint()
    with (tmp_path / "first/predictions.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    expected = [(split.id, key) for split in protocol.splits for key in split.score]
    assert [(row["fold"], row["row_id"]) for row in rows] == expected
    assert all(float(row["truth"]) == int(row["row_id"]) % 2 for row in rows)


def test_preprocessing_sees_training_rows_only(data: Path, tmp_path: Path) -> None:
    protocol = protocol_for(data)
    seen: list[set[str]] = []

    class RecordingImputer(SimpleImputer):
        @override
        def fit(self, X, y=None) -> Self:  # noqa: ANN001 - sklearn override signature
            seen.append(set(X.index))
            super().fit(X, y)
            return self

    def factory(seed: int) -> BaseEstimator:
        return Pipeline([("impute", RecordingImputer()), ("model", LogisticRegression())])

    result = evaluate(data, protocol, factory, tmp_path / "out")
    assert result.complete
    assert seen == [set(split.train) for split in protocol.splits]


@pytest.mark.parametrize("change", ["overlap", "unknown", "duplicate", "coverage", "features"])
def test_refuses_invalid_protocols(data: Path, change: str) -> None:
    values = protocol_for(data).model_dump()
    protocol = protocol_for(data)
    if change == "overlap":
        protocol.splits[0].score.append(protocol.splits[0].train[0])
    elif change == "unknown":
        protocol.splits[0].score[0] = "missing"
    elif change == "duplicate":
        protocol.rows.append(protocol.rows[0])
    elif change == "coverage":
        protocol.splits.pop()
    else:
        protocol.numeric.append("target")
    assert values != protocol.model_dump()
    with pytest.raises(ValueError, match="validation error"):
        validate_data(data, protocol)


def test_changed_data_and_class_coverage_are_refused(data: Path) -> None:
    protocol = protocol_for(data)
    data.write_text(data.read_text().replace("0,0,0,", "0,0,1,"))
    with pytest.raises(ValueError, match="changed"):
        validate_data(data, protocol)
    protocol = protocol_for(data)
    keys = protocol.rows
    positives = [key for key in keys if int(key) % 2 == 1 or key == "0"]
    protocol.splits[0].score = positives
    protocol.splits[0].train = [key for key in keys if key not in positives]
    # A structurally valid two-fold partition still needs both classes in every partition.
    protocol.splits = [
        protocol.splits[0],
        tabular.Split(id="other", train=positives, score=protocol.splits[0].train),
    ]
    with pytest.raises(ValueError, match="both classes"):
        validate_data(data, protocol)


def test_group_and_temporal_constraints(data: Path) -> None:
    protocol = protocol_for(data)
    keys = protocol.rows
    protocol.kind = "group"
    protocol.group_column = "group"
    protocol.splits = [
        tabular.Split(id="a", train=keys[:12], score=keys[12:]),
        tabular.Split(id="b", train=keys[12:], score=keys[:12]),
    ]
    validate_data(data, protocol)
    protocol.group_column = "target"
    with pytest.raises(ValueError, match="Groups"):
        validate_data(data, protocol)
    protocol.kind = "time"
    protocol.time_column = "date"
    protocol.splits = [
        tabular.Split(id="a", train=keys[:8], score=keys[8:16]),
        tabular.Split(id="b", train=keys[:16], score=keys[16:]),
    ]
    validate_data(data, protocol)
    protocol.splits[0].train = keys[16:]
    with pytest.raises(ValueError, match="precede"):
        validate_data(data, protocol)


def test_fold_failure_preserves_partial_evidence(data: Path, tmp_path: Path) -> None:
    count = 0

    def factory(seed: int) -> BaseEstimator:
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("intentional fold failure")
        return linear(seed)

    output = tmp_path / "out"
    result = evaluate(data, protocol_for(data), factory, output)
    assert not result.complete
    assert result.folds[1].state == "failed"
    assert (
        len(tabular.Evaluation.model_validate(read_object(output / "evaluation.json")).folds) == 3
    )
    assert (output / "predictions.csv").stat().st_size > 100
    assert not (output / "metrics.json").exists()
    with pytest.raises(ValueError, match="already exist"):
        evaluate(data, protocol_for(data), linear, output)


@pytest.mark.skipif(os.environ.get("MLEF_TEST_DOCKER") != "1", reason="Set MLEF_TEST_DOCKER=1")
def test_real_sklearn_workflow_and_failure(project: Path) -> None:
    api.init()
    for name in ["pyproject.toml", "uv.lock", "data.csv", "model.py"]:
        (project / name).write_bytes((EXAMPLE / name).read_bytes())
    tabular.export_helper("evaluate.py")
    (project / "model.py").write_text(
        "import importlib.util\nassert importlib.util.find_spec('mlef') is None\n"
        + (project / "model.py").read_text()
    )
    options = api.RunOptions(timeout_seconds=240, max_log_bytes=65536)
    prepared = api.run(
        "evaluate.py",
        includes=["data.csv"],
        options=options,
        arguments=[
            "prepare",
            "--numeric",
            "radius",
            "texture",
            "smoothness",
            "--categorical",
            "radius_band",
        ],
    )
    assert prepared.state == "succeeded", prepared.model_dump_json()
    api.get(prepared.id, "protocol.json", output="protocol.json")
    result = api.run(
        "evaluate.py",
        includes=["data.csv", "model.py", "protocol.json"],
        arguments=["evaluate"],
        options=options,
    )
    assert result.state == "succeeded", result.model_dump_json()
    assert result.result.metrics["roc_auc_mean"] > 0.85
    (project / "failure.py").write_text(
        "def make_estimator(seed):\n raise RuntimeError('fold failure')\n"
    )
    failed = api.run(
        "evaluate.py",
        includes=["data.csv", "failure.py", "protocol.json"],
        arguments=["evaluate", "--factory", "failure:make_estimator"],
        options=options,
    )
    assert failed.state == "failed"
    assert "evaluation.json" in api.artifacts(failed.id)
