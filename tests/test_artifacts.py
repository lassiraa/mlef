"""Integrity and evaluation contracts are validated without execution or ML imports."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mlef.artifacts import (
    ArtifactFile,
    ArtifactManifest,
    ArtifactRef,
    ArtifactType,
    Column,
    DatasetMetadata,
    ModelMetadata,
    PredictionsMetadata,
    SplitMetadata,
)
from mlef.metrics import (
    Curve,
    Distribution,
    Interval,
    Matrix,
    Metrics,
    MetricTable,
    Scalar,
    Series,
    Undefined,
)


def file(path: str = "data.csv", digest: str = "a" * 64) -> ArtifactFile:
    return ArtifactFile(path=path, size_bytes=10, sha256=digest)


def reference(output: str = "raw", digest: str = "b" * 64) -> ArtifactRef:
    return ArtifactRef(
        run_id="00000000-0000-4000-8000-000000000001", step="ingest", output=output, sha256=digest
    )


def test_manifest_identity_and_json_round_trip() -> None:
    original = ArtifactManifest(
        name="bundle",
        type=ArtifactType.DIRECTORY,
        format="files",
        files=[file("b.csv"), file("a.csv")],
    )
    renamed = ArtifactManifest(
        name="copy",
        type=ArtifactType.DIRECTORY,
        format="files",
        files=list(reversed(original.files)),
    )
    assert original.content_sha256() == renamed.content_sha256()
    renamed.producer = reference("copy", renamed.content_sha256())
    assert original.content_sha256() == renamed.content_sha256()
    assert ArtifactManifest.model_validate_json(renamed.model_dump_json()) == renamed
    assert original.size_bytes == 20
    changed = ArtifactManifest(
        name="bundle",
        type=ArtifactType.DIRECTORY,
        format="files",
        files=[file("b.csv"), file("a.csv", "c" * 64)],
    )
    assert original.content_sha256() != changed.content_sha256()


@pytest.mark.parametrize(
    "path",
    ["", "/tmp/data", "../data", "a/../b", "a//b", "./x", "C:/data", "a\\b", "a/", "a\x00b"],
)
def test_inventory_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(ValidationError, match="normalized"):
        file(path)


@pytest.mark.parametrize("paths", [["a", "a"], ["a", "a/b"]])
def test_inventory_rejects_duplicate_and_conflicting_paths(paths: list[str]) -> None:
    with pytest.raises(ValidationError):
        ArtifactManifest(
            name="bundle",
            type=ArtifactType.DIRECTORY,
            format="files",
            files=[file(path) for path in paths],
        )


def test_manifest_identity_includes_semantics_and_inventory() -> None:
    original = ArtifactManifest(
        name="weights", type=ArtifactType.FILE, format="binary", files=[file("weights.bin")]
    )
    changes = [
        {"format": "pickle"},
        {"type": ArtifactType.CHECKPOINT},
        {"files": [file("renamed.bin")]},
        {"files": [ArtifactFile(path="weights.bin", size_bytes=11, sha256="a" * 64)]},
    ]
    for change in changes:
        changed = ArtifactManifest.model_validate(original.model_dump() | change)
        assert changed.content_sha256() != original.content_sha256()


def test_dataset_requires_complete_valid_identity() -> None:
    metadata = DatasetMetadata(
        rows=100,
        columns=[
            Column(name="id", dtype="string", role="row_id"),
            Column(name="feature", dtype="float64"),
        ],
        row_id=["id"],
    )
    manifest = ArtifactManifest(
        name="raw",
        type=ArtifactType.DATASET,
        format="csv",
        files=[file()],
        metadata=metadata,
    )
    assert ArtifactManifest.model_validate_json(manifest.model_dump_json()) == manifest
    with pytest.raises(ValidationError, match="existing"):
        DatasetMetadata(rows=1, columns=metadata.columns, row_id=["missing"])
    with pytest.raises(ValidationError, match="unique"):
        DatasetMetadata(rows=1, columns=metadata.columns * 2, row_id=["id"])
    with pytest.raises(ValidationError, match="specialized"):
        ArtifactManifest(name="raw", type=ArtifactType.DATASET, format="csv", files=[file()])
    with pytest.raises(ValidationError, match="disagree"):
        ArtifactManifest(
            name="raw",
            type=ArtifactType.MODEL,
            format="csv",
            files=[file()],
            metadata=metadata,
        )


def test_split_assignments_must_be_in_inventory() -> None:
    metadata = SplitMetadata(
        dataset=reference(),
        strategy="group",
        assignments_file="assignments.csv",
    )
    with pytest.raises(ValidationError, match="assignments"):
        ArtifactManifest(
            name="folds",
            type=ArtifactType.SPLIT,
            format="csv",
            files=[file()],
            metadata=metadata,
        )
    ArtifactManifest(
        name="folds",
        type=ArtifactType.SPLIT,
        format="csv",
        files=[file("assignments.csv")],
        metadata=metadata,
    )


def test_prediction_metadata_and_model_contract() -> None:
    prediction = PredictionsMetadata(
        dataset=reference(),
        model=reference("model"),
        task="binary",
        row_id=["id"],
        value_columns=["score"],
        class_order=["0", "1"],
        scope="validation",
    )
    assert prediction.class_order == ["0", "1"]
    with pytest.raises(ValidationError, match="distinct"):
        PredictionsMetadata(
            dataset=reference(),
            model=reference("model"),
            task="binary",
            row_id=["id"],
            value_columns=["id"],
            scope="validation",
        )
    model = ModelMetadata(framework="sklearn", serialization="joblib")
    ArtifactManifest(
        name="model",
        type=ArtifactType.MODEL,
        format="joblib",
        files=[file("model.joblib")],
        metadata=model,
    )


def test_wrong_producer_digest_rejected() -> None:
    with pytest.raises(ValidationError, match="Producer"):
        ArtifactManifest(
            name="raw",
            type=ArtifactType.FILE,
            format="csv",
            files=[file()],
            producer=reference(),
        )


def test_metric_representations_round_trip(tmp_path: Path) -> None:
    evidence = Metrics(
        scope="validation",
        dataset=reference(),
        values={
            "accuracy": Scalar(
                definition="fraction_correct",
                direction="maximize",
                value=0.9,
                sample_count=100,
                interval=Interval(lower=0.8, upper=0.95, confidence=0.95, method="bootstrap"),
            ),
            "loss": Series(
                definition="mean_loss",
                index="epoch",
                x=[1, 2],
                y=[0.7, 0.4],
                direction="minimize",
            ),
            "roc": Curve(
                definition="roc",
                curve_type="roc",
                x_label="fpr",
                y_label="tpr",
                x=[0, 0.2, 1],
                y=[0, 0.8, 1],
            ),
            "confusion": Matrix(
                definition="confusion_counts",
                row_axis="truth",
                column_axis="prediction",
                row_labels=["0", "1"],
                column_labels=["0", "1"],
                values=[[90, 10], [8, 92]],
            ),
            "by_class": MetricTable(
                definition="class_precision",
                columns=["class", "precision"],
                rows=[{"class": "positive", "precision": 0.8}],
            ),
            "latency": Distribution(
                definition="request_latency",
                unit="ms",
                samples=[10, 12, 18],
                weights=[1, 1, 2],
            ),
            "auc": Undefined(definition="roc_auc", reason="Only one class in this slice."),
        },
    )
    path = tmp_path / "metrics.json"
    path.write_text(evidence.model_dump_json(), encoding="utf-8")
    assert Metrics.model_validate_json(path.read_text(encoding="utf-8")) == evidence
    assert json.loads(path.read_text())["values"]["auc"]["kind"] == "undefined"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_metrics_reject_nonfinite_values(value: float) -> None:
    with pytest.raises(ValidationError):
        Scalar(definition="score", value=value)
    with pytest.raises(ValidationError):
        Distribution(definition="samples", samples=[value])
    with pytest.raises(ValidationError):
        MetricTable(definition="scores", columns=["value"], rows=[{"value": value}])
    with pytest.raises(ValidationError):
        Scalar(definition="score", value=1, parameters={"nested": [value]})


def test_metric_shape_and_semantic_errors() -> None:
    with pytest.raises(ValidationError, match="lengths"):
        Series(definition="loss", index="epoch", x=[1, 2], y=[0.5])
    with pytest.raises(ValidationError, match="Thresholds"):
        Curve(
            definition="roc",
            curve_type="roc",
            x_label="fpr",
            y_label="tpr",
            x=[0, 1],
            y=[0, 1],
            thresholds=[0.5],
        )
    with pytest.raises(ValidationError, match="shape"):
        Matrix(
            definition="confusion",
            row_axis="truth",
            column_axis="prediction",
            row_labels=["a"],
            column_labels=["a", "b"],
            values=[[1]],
        )
    with pytest.raises(ValidationError, match="columns"):
        MetricTable(definition="by_class", columns=["class", "score"], rows=[{"score": 0.1}])
    with pytest.raises(ValidationError, match="Weights"):
        Distribution(definition="latency", samples=[1], weights=[0])
    with pytest.raises(ValidationError, match="lower"):
        Interval(lower=2, upper=1, confidence=0.95, method="bootstrap")
    with pytest.raises(ValidationError, match="Target"):
        Scalar(definition="calibration", value=0.1, direction="target")
    with pytest.raises(ValidationError, match="Target"):
        Scalar(definition="accuracy", value=0.1, direction="maximize", target=1)
    assert (
        Scalar(
            definition="bias",
            value=0.1,
            direction="target",
            target=0,
        ).target
        == 0
    )


def test_unknown_fields_versions_and_empty_metric_names_rejected() -> None:
    with pytest.raises(ValidationError):
        Metrics.model_validate({"schema_version": 2, "scope": "test", "values": {}})
    with pytest.raises(ValidationError, match="empty"):
        Metrics(scope="test", values={" ": Scalar(definition="score", value=1)})
    with pytest.raises(ValidationError):
        Scalar.model_validate({"definition": "score", "value": 1, "typo": True})
