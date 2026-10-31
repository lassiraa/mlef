"""Verified, self-contained SDK exports of public tabular evaluation evidence."""

import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, JsonValue, field_validator, model_validator

from .artifacts import (
    ArtifactRef,
    ArtifactType,
    Column,
    ColumnStatistics,
    DatasetMetadata,
    DatasetProfile,
    GenericMetadata,
    Metadata,
    ModelMetadata,
    PredictionsMetadata,
    SplitMetadata,
)
from .locking import run_lock
from .metrics import Metrics
from .pipeline import Input, Output, Step
from .records import Contract, run_id
from .sdk import Artifact, ArtifactSource, _checked, read_step
from .storage import artifact_path, directory, inspect
from .tabular_comparison import (
    MAX_EVIDENCE_BYTES,
    evaluation_metrics,
    load_evaluation,
    read_evaluation,
)
from .tabular_runtime import Evaluation, Fit, Prediction, Protocol, validate_data

KINDS = {
    "dataset": ArtifactType.DATASET,
    "split": ArtifactType.SPLIT,
    "model": ArtifactType.MODEL,
    "predictions": ArtifactType.PREDICTIONS,
    "metrics": ArtifactType.METRICS,
}
DEPENDENCIES = {
    "dataset": [],
    "split": ["dataset"],
    "model": ["dataset", "split"],
    "predictions": ["dataset", "split", "model"],
    "metrics": ["dataset", "split", "model", "predictions"],
}


class CvRecipe(Contract):
    schema_version: Literal[1] = 1
    role: Literal["cross_validation_recipe"] = "cross_validation_recipe"
    protocol_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    evaluation_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    parameters: dict[str, JsonValue]
    fits: dict[str, Fit | None]


class TabularBundle(Contract):
    schema_version: Literal[1] = 1
    source_run: str
    evaluation_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    artifacts: dict[str, ArtifactRef]

    _source_run = field_validator("source_run")(run_id)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if set(self.artifacts) != set(KINDS) or any(
            reference.run_id != self.source_run
            or reference.step != "tabular_" + name
            or reference.output != name
            for name, reference in self.artifacts.items()
        ):
            raise ValueError("Tabular bundle artifacts must identify all five original producers.")
        return self


def _bytes(path: Path) -> bytes:
    path = _checked(path.absolute())
    if not path.is_file() or path.stat().st_size > MAX_EVIDENCE_BYTES:
        raise ValueError("Tabular payloads must be regular files of at most 32 MiB.")
    return path.read_bytes()


def _json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()


def _dataset(path: Path, protocol: Protocol, profile: bool) -> DatasetMetadata:
    _bytes(path)
    rows = validate_data(path, protocol)
    names = list(next(iter(rows.values())))
    roles = dict.fromkeys(protocol.numeric + protocol.categorical, "feature")
    roles.update({protocol.id_column: "row_id", protocol.target: "target"})
    if protocol.group_column is not None:
        roles[protocol.group_column] = "group"
    if protocol.time_column is not None:
        roles[protocol.time_column] = "time"
    columns = [
        Column.model_validate(
            {
                "name": name,
                "dtype": "float64"
                if name in protocol.numeric
                or (name == protocol.target and protocol.task == "regression")
                else "string",
                "role": roles.get(name, "auxiliary"),
                "nullable": any(row[name] == "" for row in rows.values()),
            }
        )
        for name in names
    ]
    summary = None
    if profile:
        summary = DatasetProfile(
            method="exact",
            rows_examined=len(rows),
            omitted_columns=max(0, len(names) - 20),
            columns=[
                ColumnStatistics(
                    name=name[:200],
                    name_sha256=hashlib.sha256(name.encode()).hexdigest(),
                    missing=sum(row[name] == "" for row in rows.values()),
                    distinct=len({row[name] for row in rows.values()} - {""}),
                )
                for name in names[:20]
            ],
        )
    return DatasetMetadata(
        rows=len(rows),
        columns=columns,
        row_id=[protocol.id_column],
        dataset_sha256=protocol.dataset_sha256,
        row_ids_sha256=hashlib.sha256(
            json.dumps(sorted(rows), separators=(",", ":")).encode()
        ).hexdigest(),
        statistics="exact" if profile else "omitted",
        profile=summary,
    )


def _assignments(protocol: Protocol) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["fold", "row_id", "role"])
    for split in protocol.splits:
        for role, keys in (("train", split.train), ("score", split.score)):
            writer.writerows((split.id, key, role) for key in keys)
    return stream.getvalue().encode()


def _recipe(result: Evaluation, evaluation_sha256: str) -> bytes:
    recipe = CvRecipe(
        protocol_id=result.protocol.fingerprint(),
        parameters=result.parameters,
        fits={fold.id: fold.fit for fold in result.folds},
        evaluation_sha256=evaluation_sha256,
    )
    return _json(recipe.model_dump(mode="json"))


def _predictions_metadata(
    result: Evaluation, data: Path, references: dict[str, ArtifactRef]
) -> PredictionsMetadata:
    protocol = result.protocol
    labels = {row[protocol.target] for row in validate_data(data, protocol).values()}
    classes = (
        [
            next(label for label in labels if label != protocol.positive_label),
            protocol.positive_label,
        ]
        if protocol.task == "binary"
        else None
    )
    return PredictionsMetadata(
        dataset=references["dataset"],
        split=references["split"],
        model=references["model"],
        task=protocol.task,
        scope="out_of_fold",
        row_id=["row_id"],
        value_columns=["prediction"],
        class_order=classes,
    )


def _metrics(result: Evaluation, references: dict[str, ArtifactRef]) -> Metrics:
    metrics = evaluation_metrics(result)
    if metrics.scope != "out_of_fold":
        raise ValueError("Typed tabular exports require out-of-fold metric scope.")
    return metrics.model_copy(
        update={name: references[name] for name in ("dataset", "split", "model", "predictions")}
    )


def _step(name: str) -> Step:
    return Step(
        name="tabular_" + name,
        inputs=[Input(name=key, type=KINDS[key]) for key in DEPENDENCIES[name]],
        outputs=[Output(name=name, type=KINDS[name], format="mlef.tabular-" + name + "-v1")],
    )


def _truth(result: Evaluation, rows: list[Prediction], data: Path) -> None:
    raw = validate_data(data, result.protocol)
    for row in rows:
        truth = (
            float(raw[row.row_id][result.protocol.target])
            if result.protocol.task == "regression"
            else float(raw[row.row_id][result.protocol.target] == result.protocol.positive_label)
        )
        if not math.isclose(truth, row.truth, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                "Prediction truth differs from the explicitly supplied public dataset."
            )


def _publish(
    destination: Path,
    identifier: str,
    name: str,
    metadata: Metadata,
    payloads: dict[str, bytes],
    sources: dict[str, ArtifactSource],
) -> ArtifactSource:
    inputs = {key: sources[key] for key in DEPENDENCIES[name]}
    step = _step(name)
    with step.local(destination / name, run_id=identifier, inputs=inputs) as context:
        for filename, payload in payloads.items():
            if len(payload) > MAX_EVIDENCE_BYTES:
                raise ValueError("Tabular export payload exceeds 32 MiB.")
            context.output_path(name, filename).write_bytes(payload)
        step.save(
            [
                Artifact(
                    name=name,
                    type=KINDS[name],
                    format=step.outputs[0].format,
                    path=context.output_path(name),
                    metadata=metadata,
                )
            ],
            context=context,
        )
    return ArtifactSource(directory=destination / name, name=name)


def export_artifacts(
    root: Path, identifier: str, data: Path, destination: Path, *, profile: bool = False
) -> TabularBundle:
    root, data, destination = root.absolute(), data.absolute(), destination.absolute()
    with run_lock(_checked(directory(root, identifier))):
        parent = inspect(root, identifier)
        if parent.state != "succeeded" or (
            parent.runtime is not None and parent.runtime.cleanup_pending
        ):
            raise ValueError("Tabular export requires a successful run with completed cleanup.")
        result, rows = load_evaluation(root, identifier)
        if not result.complete:
            raise ValueError("Tabular export requires every scoring fold to succeed.")
        evaluation = _bytes(artifact_path(root, identifier, "evaluation.json"))
        predictions = _bytes(artifact_path(root, identifier, "predictions.csv"))
        metadata = _dataset(data, result.protocol, profile)
        _truth(result, rows, data)
        _checked(destination).mkdir(parents=True, exist_ok=False)
        (destination / "evaluation.json").write_bytes(evaluation)
        evaluation_hash = hashlib.sha256(evaluation).hexdigest()
        sources: dict[str, ArtifactSource] = {}
        references: dict[str, ArtifactRef] = {}
        for name in KINDS:
            details: Metadata
            if name == "dataset":
                details, payloads = metadata, {"data.csv": _bytes(data)}
            elif name == "split":
                details = SplitMetadata(
                    dataset=references["dataset"],
                    strategy=result.protocol.kind,
                    seed=result.protocol.seed,
                    assignments_file="assignments.csv",
                )
                payloads = {"assignments.csv": _assignments(result.protocol)}
            elif name == "model":
                details = ModelMetadata(
                    framework="scikit-learn",
                    serialization="mlef.tabular-cv-recipe-v1",
                    training_dataset=references["dataset"],
                )
                payloads = {"recipe.json": _recipe(result, evaluation_hash)}
            elif name == "predictions":
                details = _predictions_metadata(result, data, references)
                payloads = {"predictions.csv": predictions}
            else:
                details = GenericMetadata()
                payloads = {
                    "metrics.json": _metrics(result, references).model_dump_json(indent=2).encode()
                }
            sources[name] = _publish(destination, identifier, name, details, payloads, sources)
            producer = read_step(sources[name].directory).artifacts[name].producer
            assert producer is not None
            references[name] = producer
        bundle = TabularBundle(
            source_run=identifier, evaluation_sha256=evaluation_hash, artifacts=references
        )
        (destination / "bundle.json").write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
        try:
            return read_artifacts(destination)
        except (OSError, ValueError):
            (destination / "bundle.json").unlink(missing_ok=True)
            raise


def read_artifacts(destination: Path) -> TabularBundle:
    destination = _checked(destination.absolute())
    bundle = TabularBundle.model_validate_json(_bytes(destination / "bundle.json"))
    payloads = {name: destination / name / "outputs" / name for name in KINDS}
    for name, reference in bundle.artifacts.items():
        result = read_step(destination / name)
        if (
            result.run_id != bundle.source_run
            or result.step != _step(name)
            or set(result.artifacts) != {name}
            or result.artifacts[name].type != KINDS[name]
            or result.artifacts[name].producer != reference
            or result.inputs != {key: bundle.artifacts[key] for key in DEPENDENCIES[name]}
        ):
            raise ValueError("Tabular artifact producer or dependency lineage disagrees.")
    evaluation = destination / "evaluation.json"
    if hashlib.sha256(_bytes(evaluation)).hexdigest() != bundle.evaluation_sha256:
        raise ValueError("Exported evaluation bytes changed.")
    result, rows = read_evaluation(evaluation, payloads["predictions"] / "predictions.csv")
    if not result.complete:
        raise ValueError("Tabular bundles require a complete evaluation.")
    data = payloads["dataset"] / "data.csv"
    manifest = read_step(destination / "dataset").artifacts["dataset"]
    if not isinstance(manifest.metadata, DatasetMetadata) or manifest.metadata != _dataset(
        data, result.protocol, manifest.metadata.profile is not None
    ):
        raise ValueError("Dataset metadata does not match public source bytes and row identities.")
    _truth(result, rows, data)
    expected_metadata: dict[str, Metadata] = {
        "split": SplitMetadata(
            dataset=bundle.artifacts["dataset"],
            strategy=result.protocol.kind,
            seed=result.protocol.seed,
            assignments_file="assignments.csv",
        ),
        "model": ModelMetadata(
            framework="scikit-learn",
            serialization="mlef.tabular-cv-recipe-v1",
            training_dataset=bundle.artifacts["dataset"],
        ),
        "predictions": _predictions_metadata(result, data, bundle.artifacts),
        "metrics": GenericMetadata(),
    }
    for name, metadata in expected_metadata.items():
        if read_step(destination / name).artifacts[name].metadata != metadata:
            raise ValueError(
                "Tabular metadata has incompatible scope, class ordering or model links."
            )
    if _bytes(payloads["split"] / "assignments.csv") != _assignments(result.protocol) or _bytes(
        payloads["model"] / "recipe.json"
    ) != _recipe(result, bundle.evaluation_sha256):
        raise ValueError("Split assignments or CV model recipe disagree with saved evaluation.")
    metrics = Metrics.model_validate_json(_bytes(payloads["metrics"] / "metrics.json"))
    if metrics != _metrics(result, bundle.artifacts):
        raise ValueError("Structured metrics disagree with saved evaluation and artifact lineage.")
    return bundle
