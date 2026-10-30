import csv
import importlib.util
import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Self
from uuid import uuid4

import pytest
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from typing_extensions import override

from mlef import api, tabular
from mlef.records import RunRecord
from mlef.runner import save
from mlef.storage import QueryError
from mlef.tabular_comparison import compare_tabular
from mlef.tabular_runtime import digest, evaluate, predict, prepare, read_object, validate_data

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
    selected = api.run(
        "evaluate.py",
        includes=["data.csv", "model.py", "protocol.json"],
        arguments=["evaluate", "--factory", "model:make_xgboost", "--refit"],
        options=options,
    )
    assert selected.state == "succeeded", selected.model_dump_json()
    assert api.compare_tabular(result.id, selected.id).compatibility == "matched"
    for artifact in ["model.joblib", "model-recipe.json"]:
        api.get(selected.id, artifact, output=artifact)
    inferred = api.run(
        "evaluate.py",
        includes=["data.csv", "model.joblib", "model-recipe.json"],
        arguments=["predict"],
        options=options,
    )
    assert inferred.state == "succeeded", inferred.model_dump_json()
    assert "inference.csv" in api.artifacts(inferred.id)
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


def stored(project: Path, data: Path) -> str:
    api.init()
    identifier = str(uuid4())
    directory = project / ".mlef/runs" / identifier
    evaluate(data, protocol_for(data), linear, directory / "artifacts")
    save(
        directory / "run.json",
        RunRecord(
            id=identifier,
            state="succeeded",
            command=["python", "evaluate.py"],
            created_at="2026-10-30T00:00:00Z",
            finished_at="2026-10-30T00:00:01Z",
            execution_exit_code=0,
        ),
    )
    return identifier


def test_tabular_comparison_matches_cli_and_api(
    project: Path, data: Path, cli: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    first, second = stored(project, data), stored(project, data)
    result = api.compare_tabular(first, second, limit=1)
    assert result.compatibility == "matched"
    assert result.observation == "unchanged"
    assert result.omitted_folds == 2
    assert result.model_dump(mode="json") == json.loads(
        cli("tabular", "compare", first, second, "--limit", "1", "--json").stdout
    )


@pytest.mark.parametrize("damage", ["protocol", "prediction", "score", "failed_run", "missing"])
def test_tabular_comparison_refuses_unreliable_winners(
    project: Path, data: Path, damage: str
) -> None:
    first, second = stored(project, data), stored(project, data)
    directory = project / ".mlef/runs" / second
    path = directory / "artifacts/evaluation.json"
    evidence = tabular.Evaluation.model_validate(read_object(path))
    if damage == "protocol":
        evidence.protocol.seed += 1
    elif damage == "prediction":
        predictions = directory / "artifacts/predictions.csv"
        with predictions.open() as stream:
            lines = stream.readlines()
        predictions.write_text("".join([*lines, lines[1]]))
        evidence.predictions_sha256 = digest(predictions)
    elif damage == "score":
        evidence.folds[0].score = 0.999
    elif damage == "failed_run":
        record = RunRecord.model_validate(read_object(directory / "run.json"))
        record.state = "failed"
        save(directory / "run.json", record)
    else:
        path.unlink()
    if damage != "missing":
        save(path, evidence)
    if damage in {"prediction", "score"}:
        with pytest.raises(QueryError, match=r"duplicate|score differs"):
            compare_tabular(project, first, second)
    else:
        result = compare_tabular(project, first, second)
        assert result.observation == "unavailable"
        assert result.compatibility == ("incompatible" if damage == "protocol" else "unknown")


def example_factory(name: str) -> Callable[[int], BaseEstimator]:
    spec = importlib.util.spec_from_file_location("example_model", EXAMPLE / "model.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def factory(seed: int) -> BaseEstimator:
        estimator = getattr(module, name)(seed)
        assert isinstance(estimator, BaseEstimator)
        return estimator

    return factory


def test_xgboost_inner_splits_and_refit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("xgboost")
    import joblib
    import numpy as np

    from mlef.tabular_runtime import feature_frame, read_rows

    fitted: list[set[str]] = []
    original_fit = SimpleImputer.fit

    def observed_fit(self: SimpleImputer, features, y=None) -> SimpleImputer:  # noqa: ANN001 - sklearn callback
        fitted.append(set(features.index))
        original_fit(self, features, y)
        return self

    monkeypatch.setattr(SimpleImputer, "fit", observed_fit)

    data = EXAMPLE / "data.csv"
    protocol = prepare(
        data,
        identifier="id",
        target="target",
        numeric=["radius", "texture", "smoothness"],
        categorical=["radius_band"],
    )
    output = tmp_path / "evaluation"
    result = evaluate(data, protocol, example_factory("make_xgboost"), output, refit=True)
    assert result.complete
    assert result.model_artifact == "model.joblib"
    for split, fold in zip(protocol.splits, result.folds, strict=True):
        assert fold.fit is not None
        assert fold.fit.best_iteration is not None
        assert set(fold.fit.train) | set(fold.fit.validation) == set(split.train)
        assert not (set(fold.fit.train) | set(fold.fit.validation)) & set(split.score)
    for index, fold in enumerate(result.folds):
        assert fold.fit is not None
        assert fitted[index * 2 : index * 2 + 2] == [set(fold.fit.train), set(fold.fit.train)]
    frame = feature_frame(read_rows(data, "id"), protocol.numeric, protocol.categorical)
    loaded = joblib.load(output / "model.joblib")
    expected = loaded.predict_proba(frame[protocol.numeric + protocol.categorical])[:, 1]
    predict(data, output / "model.joblib", output / "model-recipe.json", tmp_path / "predicted.csv")
    with (tmp_path / "predicted.csv").open() as stream:
        actual = [float(row["positive_probability"]) for row in csv.DictReader(stream)]
    np.testing.assert_allclose(actual, expected)
    versions = read_object(output / "model-recipe.json")["versions"]
    assert isinstance(versions, dict)
    assert "xgboost-cpu" in versions
