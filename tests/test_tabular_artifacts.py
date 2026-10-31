"""Shared tabular artifacts can be checked and consumed without any ML imports."""

import csv
import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any
from uuid import uuid4

import pytest

from mlef import (
    Artifact,
    ArtifactType,
    Input,
    Output,
    Step,
    api,
    load_artifacts,
    read_step,
    tabular,
)
from mlef.artifacts import ArtifactManifest, ArtifactRef, Column, DatasetMetadata, DatasetProfile
from mlef.metrics import Metrics
from mlef.records import RunRecord
from mlef.runner import save
from mlef.sdk import _inventory, _write_result
from mlef.storage import QueryError
from mlef.tabular_artifacts import read_artifacts
from mlef.tabular_comparison import compare_tabular
from mlef.tabular_runtime import Evaluation, Fold, Protocol, Split, digest, measure, metric_evidence


@pytest.fixture(params=["binary", "regression"])
def saved(project: Path, request: pytest.FixtureRequest) -> tuple[str, Path]:
    api.init(root=project, scaffold=False)
    data = project / "public.csv"
    task = request.param
    targets = [
        str(index * 1.5) if task == "regression" else ("yes" if index % 2 else "no")
        for index in range(12)
    ]
    data.write_text(
        "id,x,target\n"
        + "".join(f"row-{index},{index},{target}\n" for index, target in enumerate(targets))
    )
    keys = [f"row-{index}" for index in range(12)]
    protocol = Protocol.model_validate(
        {
            "task": task,
            "metric": "rmse" if task == "regression" else "roc_auc",
            "direction": "minimize" if task == "regression" else "maximize",
            "dataset_sha256": digest(data),
            "id_column": "id",
            "target": "target",
            "numeric": ["x"],
            "positive_label": "yes",
            "rows": keys,
            "splits": [
                Split(
                    id=str(index),
                    train=[key for key in keys if key not in keys[index * 4 : index * 4 + 4]],
                    score=keys[index * 4 : index * 4 + 4],
                ).model_dump()
                for index in range(3)
            ],
        }
    )
    identifier = str(uuid4())
    directory = project / ".mlef/runs" / identifier
    output = directory / "artifacts"
    output.mkdir(parents=True)
    folds = []
    with (output / "predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["fold", "row_id", "truth", "prediction"])
        for split in protocol.splits:
            truth = [
                float(targets[keys.index(key)])
                if task == "regression"
                else float(targets[keys.index(key)] == "yes")
                for key in split.score
            ]
            predictions = [
                value + 0.5 if task == "regression" else 0.2 + value * 0.6 for value in truth
            ]
            writer.writerows(
                (split.id, key, label, prediction)
                for key, label, prediction in zip(split.score, truth, predictions, strict=True)
            )
            folds.append(
                Fold(
                    id=split.id,
                    state="succeeded",
                    score=measure(truth, predictions, protocol.metric),
                    seconds=0.1,
                    train_count=len(split.train),
                    score_count=len(split.score),
                )
            )
    evidence = Evaluation(
        protocol=protocol,
        complete=True,
        folds=folds,
        predictions_sha256=digest(output / "predictions.csv"),
        parameters={"estimator": "fixture"},
        metric_evidence=metric_evidence(protocol, folds),
    )
    save(output / "evaluation.json", evidence)
    save(
        directory / "run.json",
        RunRecord(
            id=identifier,
            state="succeeded",
            command=["evaluate"],
            created_at="2026-10-31T00:00:00Z",
            finished_at="2026-10-31T00:00:01Z",
            execution_exit_code=0,
        ),
    )
    return identifier, data


def test_exports_verified_self_contained_sdk_artifacts(
    project: Path, saved: tuple[str, Path], tmp_path: Path
) -> None:
    identifier, data = saved
    destination = tmp_path / "bundle"
    bundle = api.export_tabular_artifacts(identifier, data, destination, root=project, profile=True)
    assert set(bundle.artifacts) == {"dataset", "split", "model", "predictions", "metrics"}
    dataset = read_step(destination / "dataset").artifacts["dataset"]
    assert isinstance(dataset.metadata, DatasetMetadata)
    assert dataset.metadata.profile is not None
    assert dataset.metadata.profile.rows_examined == 12
    assert dataset.metadata.dataset_sha256 == digest(data)
    assert dataset.metadata.row_ids_sha256 is not None
    predictions = read_step(destination / "predictions").artifacts["predictions"]
    assert predictions.metadata.kind == "predictions"
    evidence = Evaluation.model_validate_json(
        (project / ".mlef/runs" / identifier / "artifacts/evaluation.json").read_text()
    )
    assert predictions.metadata.class_order == (
        ["no", "yes"] if evidence.protocol.task == "binary" else None
    )
    assert predictions.metadata.model == bundle.artifacts["model"]
    metrics = Metrics.model_validate_json(
        (destination / "metrics/outputs/metrics/metrics.json").read_text()
    )
    assert metrics.dataset == bundle.artifacts["dataset"]
    assert metrics.predictions == bundle.artifacts["predictions"]
    (project / ".mlef/runs" / identifier).rename(project / "moved-source")
    data.unlink()
    assert read_artifacts(destination) == bundle
    with pytest.raises(api.MlefError):
        api.export_tabular_artifacts(identifier, data, destination, root=project)
    step = Step(
        name="consume",
        inputs=[Input(name="predictions", type=ArtifactType.PREDICTIONS)],
        outputs=[Output(name="report", type=ArtifactType.TEXT, format="text/plain")],
    )
    with step.local(
        tmp_path / "consumer", inputs=load_artifacts(destination / "predictions")
    ) as context:
        assert context.input_reference("predictions") == bundle.artifacts["predictions"]
        path = context.output_path("report", "report.txt")
        path.write_text("consumed without loading an estimator")
        step.save(
            [Artifact(name="report", type=ArtifactType.TEXT, format="text/plain", path=path)],
            context=context,
        )
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; from pathlib import Path; "
            "from mlef.tabular import read_artifacts; read_artifacts(Path(sys.argv[1])); "
            "assert not {'numpy','pandas','sklearn','xgboost','joblib'} & sys.modules.keys()",
            str(destination),
        ],
        check=True,
    )


def rewrite_artifact(destination: Path, name: str, change: str) -> None:
    result = read_step(destination / name)
    manifest = result.artifacts[name]
    manifest.producer = None
    if change == "class_order":
        assert manifest.metadata.kind == "predictions"
        manifest.metadata.class_order = ["yes", "no"]
    elif change == "scope":
        assert manifest.metadata.kind == "predictions"
        manifest.metadata.scope = "training"
    elif change == "model_link":
        assert manifest.metadata.kind == "predictions"
        manifest.metadata.model = manifest.metadata.dataset
    elif change == "assignments":
        (destination / name / "outputs" / name / "assignments.csv").write_text(
            "fold,row_id,role\n0,unknown,score\n"
        )
        manifest.files = _inventory(destination / name / "outputs" / name)
    manifest.producer = ArtifactRef(
        run_id=result.run_id, step=result.step.name, output=name, sha256=manifest.content_sha256()
    )
    _write_result(destination / name, result)
    bundle_path = destination / "bundle.json"
    bundle = tabular.TabularBundle.model_validate_json(bundle_path.read_text())
    bundle.artifacts[name] = manifest.producer
    if name == "predictions":
        metrics_result = read_step(destination / "metrics")
        metrics_result.inputs[name] = manifest.producer
        metrics_path = destination / "metrics/outputs/metrics/metrics.json"
        metrics = Metrics.model_validate_json(metrics_path.read_text())
        metrics.predictions = manifest.producer
        metrics_path.write_text(metrics.model_dump_json())
        saved_metrics = metrics_result.artifacts["metrics"]
        saved_metrics.producer = None
        saved_metrics.files = _inventory(metrics_path.parent)
        saved_metrics.producer = ArtifactRef(
            run_id=result.run_id,
            step="tabular_metrics",
            output="metrics",
            sha256=saved_metrics.content_sha256(),
        )
        _write_result(destination / "metrics", metrics_result)
        bundle.artifacts["metrics"] = saved_metrics.producer
    bundle_path.write_text(bundle.model_dump_json())


@pytest.mark.parametrize("damage", ["bytes", "scope", "model_link", "assignments", "class_order"])
def test_refuses_corrupted_payloads_and_semantic_lineage(
    project: Path, saved: tuple[str, Path], tmp_path: Path, damage: str
) -> None:
    identifier, data = saved
    destination = tmp_path / "bundle"
    api.export_tabular_artifacts(identifier, data, destination, root=project)
    if damage == "bytes":
        (destination / "dataset/outputs/dataset/data.csv").write_text("changed")
    else:
        rewrite_artifact(destination, "split" if damage == "assignments" else "predictions", damage)
    with pytest.raises(ValueError, match=r"changed|lineage|metadata|assignments"):
        read_artifacts(destination)


@pytest.mark.parametrize("attribute", ["definition", "unit", "aggregation", "scope", "version"])
def test_comparison_uses_scoring_semantics_not_only_metric_names(
    project: Path, saved: tuple[str, Path], attribute: str
) -> None:
    identifier, _ = saved
    path = project / ".mlef/runs" / identifier / "artifacts/evaluation.json"
    original = path.read_bytes()
    other = str(uuid4())
    shutil.copytree(path.parents[1], path.parents[2] / other)
    parent_path = path.parents[2] / other / "run.json"
    parent = json.loads(parent_path.read_text())
    parent["id"] = other
    parent_path.write_text(json.dumps(parent))
    evidence = json.loads(original)
    metrics = evidence["metric_evidence"]
    if attribute == "scope":
        metrics[attribute] = "training"
    else:
        next(iter(metrics["values"].values()))[attribute] = "different"
    path.write_text(json.dumps(evidence))
    assert compare_tabular(project, identifier, other).compatibility == "incompatible"
    path.write_bytes(original)
    evidence["metric_evidence"] = None
    path.write_text(json.dumps(evidence))
    with pytest.raises(QueryError, match="validation error"):
        compare_tabular(project, identifier, other)


def test_dataset_content_identity_includes_all_semantic_fields() -> None:
    original = {
        "schema_version": 1,
        "type": "dataset",
        "format": "csv",
        "files": [{"path": "data.csv", "size_bytes": 1, "sha256": "a" * 64}],
        "metadata": {
            "kind": "dataset",
            "rows": 1,
            "columns": [{"name": "id", "dtype": "str", "role": "row_id", "nullable": False}],
            "row_id": ["id"],
            "statistics": "omitted",
            "dataset_sha256": None,
            "row_ids_sha256": None,
            "profile": None,
        },
    }
    identity = hashlib.sha256(
        json.dumps(original, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    original["name"] = "data"
    manifest = ArtifactManifest.model_validate_json(json.dumps(original))
    assert manifest.content_sha256() == identity


def test_cli_exports_evidence_without_mutating_the_run(
    project: Path,
    saved: tuple[str, Path],
    tmp_path: Path,
    cli: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    identifier, data = saved
    path = project / ".mlef/runs" / identifier / "artifacts/evaluation.json"
    original = path.read_bytes()
    destination = tmp_path / "cli-export"
    result = cli(
        "tabular",
        "export",
        identifier,
        "--data",
        str(data),
        "--output",
        str(destination),
        "--profile",
    )
    assert json.loads(result.stdout) == read_artifacts(destination).model_dump(mode="json")
    assert path.read_bytes() == original


def test_profiles_are_bounded_and_sample_provenance_is_required() -> None:
    with pytest.raises(ValueError, match="seed"):
        DatasetProfile(method="sampled", rows_examined=1, columns=[], omitted_columns=1)
    profile = DatasetProfile(
        method="sampled",
        rows_examined=1,
        columns=[],
        omitted_columns=1,
        sample_seed=17,
        sample_sha256="b" * 64,
    )
    DatasetMetadata(
        rows=2,
        columns=[Column(name="id", dtype="str")],
        row_id=["id"],
        statistics="sampled",
        profile=profile,
    )
    with pytest.raises(ValueError, match="column count"):
        DatasetMetadata(
            rows=0,
            columns=[Column(name="id", dtype="str")],
            row_id=["id"],
            statistics="sampled",
            profile=profile,
        )


def test_export_rejects_false_truth_even_with_consistent_scores(
    project: Path, saved: tuple[str, Path], tmp_path: Path
) -> None:
    identifier, data = saved
    output = project / ".mlef/runs" / identifier / "artifacts"
    evidence = Evaluation.model_validate_json((output / "evaluation.json").read_text())
    path = output / "predictions.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for field in ("truth", "prediction"):
            value = float(row[field])
            row[field] = str(1 - value if evidence.protocol.task == "binary" else value + 2)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    evidence.predictions_sha256 = digest(path)
    save(output / "evaluation.json", evidence)
    with pytest.raises(api.MlefError, match="explicitly supplied public dataset"):
        api.export_tabular_artifacts(identifier, data, tmp_path / "refused", root=project)
    assert not (tmp_path / "refused").exists()


def test_export_never_reads_private_labels_and_refuses_overwrite(
    project: Path, saved: tuple[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identifier, data = saved
    private = project / "private"
    private.mkdir()
    (private / "labels.csv").write_text("PRIVATE_SENTINEL")
    original_open = Path.open

    def public_only(
        path: Path,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> IO[Any]:
        if path.is_relative_to(private):
            pytest.fail("An offline export tried to read private labels")
        return original_open(path, mode, buffering, encoding, errors, newline)

    monkeypatch.setattr(Path, "open", public_only)
    destination = tmp_path / "bundle"
    bundle = api.export_tabular_artifacts(identifier, data, destination, root=project)
    assert read_artifacts(destination) == bundle
    with pytest.raises(api.MlefError, match="exists"):
        api.export_tabular_artifacts(identifier, data, destination, root=project)
    for path in destination.rglob("*"):
        if path.is_file():
            assert b"PRIVATE_SENTINEL" not in path.read_bytes()
