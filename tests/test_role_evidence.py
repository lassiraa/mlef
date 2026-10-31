import json
from pathlib import Path

import pytest

from mlef import (
    Artifact,
    ArtifactType,
    Input,
    Metrics,
    Output,
    Scalar,
    Step,
    load_artifacts,
    read_step,
)
from mlef.artifacts import Column, DatasetMetadata, ModelMetadata, PredictionsMetadata
from mlef.sdk import StepResult, _inventory, _write_result, finalize_managed


def sources(
    directory: Path, *, wrong_training: bool = False, wrong_model: bool = False
) -> tuple[Path, Path, Path]:
    data = Step(name="data", outputs=[Output(name="data", type=ArtifactType.DATASET, format="csv")])
    with data.local(directory / "data") as context:
        path = context.output_path("data", "data.csv")
        path.write_text("id,value\n1,2\n")
        data.save(
            [
                Artifact(
                    name="data",
                    type=ArtifactType.DATASET,
                    format="csv",
                    path=path,
                    metadata=DatasetMetadata(
                        rows=1, columns=[Column(name="id", dtype="int")], row_id=["id"]
                    ),
                )
            ],
            context=context,
        )
    dataset = read_step(directory / "data").artifacts["data"].producer
    assert dataset is not None
    train = Step(
        schema_version=2,
        kind="train",
        name="train",
        inputs=[Input(name="data", type=ArtifactType.DATASET)],
        outputs=[Output(name="model", type=ArtifactType.MODEL, format="json")],
    )
    with train.local(directory / "train", inputs=load_artifacts(directory / "data")) as context:
        path = context.output_path("model", "model.json")
        path.write_text("{}")
        train.save(
            [
                Artifact(
                    name="model",
                    type=ArtifactType.MODEL,
                    format="json",
                    path=path,
                    metadata=ModelMetadata(
                        framework="constant",
                        serialization="json",
                        training_dataset=dataset.model_copy(update={"sha256": "f" * 64})
                        if wrong_training
                        else dataset,
                    ),
                )
            ],
            context=context,
        )
    model = read_step(directory / "train").artifacts["model"].producer
    assert model is not None
    predict = Step(
        schema_version=2,
        kind="predict",
        name="predict",
        inputs=[
            Input(name="model", type=ArtifactType.MODEL),
            Input(name="data", type=ArtifactType.DATASET),
        ],
        outputs=[Output(name="predictions", type=ArtifactType.PREDICTIONS, format="csv")],
    )
    with predict.local(
        directory / "predict",
        inputs={
            **load_artifacts(directory / "data"),
            **load_artifacts(directory / "train"),
        },
    ) as context:
        path = context.output_path("predictions", "predictions.csv")
        path.write_text("id,value\n1,2\n")
        predict.save(
            [
                Artifact(
                    name="predictions",
                    type=ArtifactType.PREDICTIONS,
                    format="csv",
                    path=path,
                    metadata=PredictionsMetadata(
                        dataset=dataset,
                        model=model.model_copy(update={"sha256": "f" * 64})
                        if wrong_model
                        else model,
                        task="regression",
                        row_id=["id"],
                        value_columns=["value"],
                        scope="test",
                    ),
                )
            ],
            context=context,
        )
    return directory / "data", directory / "train", directory / "predict"


@pytest.mark.parametrize("mode", ["model", "predictions"])
@pytest.mark.parametrize(
    "conflict", [None, "model", "dataset", "split", "predictions", "scope", "malformed"]
)
def test_evaluation_evidence_matches_bound_inputs(
    tmp_path: Path,
    mode: str,
    conflict: str | None,
) -> None:
    data, train, predict = sources(tmp_path)
    dataset = read_step(data).artifacts["data"].producer
    model = read_step(train).artifacts["model"].producer
    prediction = read_step(predict).artifacts["predictions"].producer
    assert dataset is not None
    assert model is not None
    assert prediction is not None
    metrics = Metrics(
        model=model,
        dataset=dataset,
        scope="test",
        predictions=prediction if mode == "predictions" else None,
        values={"mae": Scalar(definition="mae", value=0, direction="minimize")},
    )
    if conflict == "scope":
        metrics.scope = "validation"
    elif conflict is not None and conflict != "malformed":
        setattr(metrics, conflict, model.model_copy(update={"sha256": "f" * 64}))
    step = Step(
        schema_version=2,
        kind="evaluate",
        name="evaluate",
        inputs=[
            Input(name="data", type=ArtifactType.DATASET),
            Input(
                name=mode,
                type=ArtifactType.MODEL if mode == "model" else ArtifactType.PREDICTIONS,
            ),
        ],
        outputs=[Output(name="metrics", type=ArtifactType.METRICS, format="application/json")],
    )

    def execute() -> None:
        with step.local(
            tmp_path / "evaluate",
            inputs={
                **load_artifacts(data),
                **load_artifacts(train if mode == "model" else predict),
            },
        ) as context:
            path = context.output_path("metrics", "metrics.json")
            path.write_text("{}" if conflict == "malformed" else metrics.model_dump_json())
            step.save(
                [
                    Artifact(
                        name="metrics",
                        type=ArtifactType.METRICS,
                        format="application/json",
                        path=path,
                    )
                ],
                context=context,
            )

    if conflict is None or (conflict == "scope" and mode == "model"):
        execute()
        assert read_step(tmp_path / "evaluate").state == "complete"
    else:
        with pytest.raises(
            ValueError, match="validation error" if conflict == "malformed" else "bound"
        ):
            execute()
        assert json.loads((tmp_path / "evaluate/step.json").read_text())["state"] == "failed"


@pytest.mark.parametrize("conflict", ["training", "prediction"])
def test_training_and_prediction_lineage(tmp_path: Path, conflict: str) -> None:
    with pytest.raises(ValueError, match="bound"):
        sources(
            tmp_path, wrong_training=conflict == "training", wrong_model=conflict == "prediction"
        )


def test_managed_publication_rechecks_evaluation_lineage(tmp_path: Path) -> None:
    data, train, _ = sources(tmp_path)
    dataset = read_step(data).artifacts["data"].producer
    model = read_step(train).artifacts["model"].producer
    assert model is not None
    step = Step(
        schema_version=2,
        kind="evaluate",
        name="evaluate",
        inputs=[
            Input(name="data", type=ArtifactType.DATASET),
            Input(name="model", type=ArtifactType.MODEL),
        ],
        outputs=[Output(name="metrics", type=ArtifactType.METRICS, format="application/json")],
    )
    context = step.local(
        tmp_path / "evaluate", inputs={**load_artifacts(data), **load_artifacts(train)}
    )
    context._managed = True
    metrics = Metrics(
        model=model,
        dataset=dataset,
        scope="test",
        values={"mae": Scalar(definition="mae", value=0)},
    )
    with context:
        path = context.output_path("metrics", "metrics.json")
        path.write_text(metrics.model_dump_json())
        step.save(
            [
                Artifact(
                    name="metrics", type=ArtifactType.METRICS, format="application/json", path=path
                )
            ],
            context=context,
        )
    # Simulate a script changing its pending output and inventory after save.
    metrics.model = model.model_copy(update={"sha256": "f" * 64})
    path.write_text(metrics.model_dump_json())
    pending = StepResult.model_validate_json((context.directory / "step.json").read_text())
    manifest = pending.artifacts["metrics"].model_copy(update={"files": _inventory(path.parent)})
    assert manifest.producer is not None
    manifest = manifest.model_copy(
        update={
            "producer": manifest.producer.model_copy(update={"sha256": manifest.content_sha256()})
        }
    )
    pending.artifacts["metrics"] = manifest
    _write_result(context.directory, pending)
    with pytest.raises(ValueError, match="bound model"):
        finalize_managed(context)
