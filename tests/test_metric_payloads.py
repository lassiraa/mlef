"""Metric references are inert until an exact, verified artifact is explicitly supplied."""

import json
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from mlef import (
    Artifact,
    ArtifactRef,
    ArtifactSource,
    ArtifactType,
    ExternalMetric,
    Metrics,
    Output,
    Step,
    StepResult,
    load_artifacts,
    read_step,
    resolve_metric,
)
from mlef.metrics import Distribution, Metric, MetricTable, Series


def series() -> Series:
    return Series(
        definition="mean_squared_error",
        version="2",
        unit="squared_units",
        direction="minimize",
        aggregation="mean",
        sample_count=200,
        parameters={"reduction": "mean"},
        dimensions={"split": "validation"},
        index="epoch",
        x=[1, 2],
        y=[0.5, 0.25],
    )


def reference(
    payload: Series | MetricTable | Distribution, producer: ArtifactRef
) -> ExternalMetric:
    return ExternalMetric.model_validate(
        {
            **{name: getattr(payload, name) for name in Metric.model_fields},
            "representation": payload.kind,
            "artifact": producer,
            "path": "metric.json",
        }
    )


def save(
    directory: Path, content: str, media_format: str = "application/json"
) -> tuple[ArtifactSource, ArtifactRef]:
    step = Step(
        name="producer",
        outputs=[Output(name="payload", type=ArtifactType.FILE, format=media_format)],
    )
    with step.local(directory) as context:
        path = context.output_path("payload", "metric.json")
        path.write_text(content, encoding="utf-8")
        step.save(
            [Artifact(name="payload", type=ArtifactType.FILE, format=media_format, path=path)],
            context=context,
        )
    source = load_artifacts(directory)["payload"]
    producer = read_step(directory).artifacts["payload"].producer
    assert producer is not None
    return source, producer


@pytest.mark.parametrize(
    "payload",
    [
        series(),
        MetricTable(
            definition="class_scores",
            columns=["class", "score"],
            rows=[{"class": "a", "score": 0.9}],
        ),
        Distribution(definition="latency", unit="ms", samples=[1, 2, 3], weights=[1, 2, 1]),
    ],
)
def test_external_representations_roundtrip_and_resolve(
    tmp_path: Path, payload: Series | MetricTable | Distribution
) -> None:
    source, producer = save(tmp_path / "producer", payload.model_dump_json())
    external = reference(payload, producer)
    metrics = Metrics(scope="validation", values={"large": external})
    assert Metrics.model_validate_json(metrics.model_dump_json()) == metrics
    assert resolve_metric(external, source) == payload
    assert external.model_dump().keys().isdisjoint({"x", "y", "rows", "samples", "weights"})
    assert resolve_metric(external, source).model_dump() == payload.model_dump()


def test_reference_inspection_does_not_open_files_or_rank_unresolved_payloads(
    tmp_path: Path,
) -> None:
    source, producer = save(tmp_path / "producer", series().model_dump_json())
    external = reference(series(), producer)
    source.directory.rename(tmp_path / "unavailable")
    evidence = Metrics(scope="validation", values={"loss": external})
    assert Metrics.model_validate_json(evidence.model_dump_json()) == evidence
    assert evidence.semantic_differences(evidence) == [
        "External metric payload requires resolution: loss"
    ]
    inline = Metrics(scope="validation", values={"loss": series()})
    assert "requires resolution" in " ".join(evidence.semantic_differences(inline))
    with pytest.raises((OSError, ValueError)):
        resolve_metric(external, source)


@pytest.mark.parametrize(
    "path", ["../metric.json", "/metric.json", "C:/metric.json", "folder\\metric.json", "a//b"]
)
def test_metric_references_require_safe_relative_paths(tmp_path: Path, path: str) -> None:
    _, producer = save(tmp_path / "producer", series().model_dump_json())
    external = reference(series(), producer)
    with pytest.raises(ValidationError):
        ExternalMetric.model_validate(external.model_dump() | {"path": path})


@pytest.mark.parametrize("change", ["run", "step", "output", "hash", "path", "format"])
def test_resolution_requires_exact_identity_inventory_and_format(
    tmp_path: Path, change: str
) -> None:
    source, producer = save(
        tmp_path / "producer",
        series().model_dump_json(),
        media_format="text/plain" if change == "format" else "application/json",
    )
    external = reference(series(), producer)
    if change in {"run", "step", "output", "hash"}:
        field, value = {
            "run": ("run_id", str(uuid4())),
            "step": ("step", "other"),
            "output": ("output", "other"),
            "hash": ("sha256", "0" * 64),
        }[change]
        external.artifact = ArtifactRef.model_validate(producer.model_dump() | {field: value})
    elif change == "path":
        external.path = "other.json"
    with pytest.raises(ValueError, match=r"exact|inventory|format"):
        resolve_metric(external, source)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("definition", "other"),
        ("version", "3"),
        ("unit", "other"),
        ("aggregation", "sum"),
        ("sample_count", 2),
        ("parameters", {"reduction": "sum"}),
        ("dimensions", {"split": "test"}),
        ("representation", "table"),
    ],
)
def test_resolution_refuses_contradictory_semantics(
    tmp_path: Path, field: str, value: object
) -> None:
    source, producer = save(tmp_path / "producer", series().model_dump_json())
    external = ExternalMetric.model_validate(
        reference(series(), producer).model_dump() | {field: value}
    )
    with pytest.raises(ValueError, match="semantics"):
        resolve_metric(external, source)


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '{"kind":"series","definition":"loss","index":"epoch","x":[1],"y":[]}',
        '{"kind":"series","definition":"loss","index":"epoch","x":[1],"y":[NaN]}',
        '{"kind":"series","definition":"loss","index":"epoch","x":[1],"y":[1e999]}',
        '{"kind":"series","kind":"series","definition":"loss","index":"epoch","x":[1],"y":[1]}',
        '{"kind":"external"}',
        '{"kind":"scalar","definition":"loss","value":1}',
        '{"kind":"distribution","definition":"latency","samples":[1],"weights":[-1]}',
        '{"kind":"table","definition":"scores","columns":["score"],"rows":[{"other":1}]}',
        pytest.param("[" * 1100 + "0" + "]" * 1100, id="excessive-nesting"),
    ],
)
def test_resolution_validates_json_and_payload_structure(tmp_path: Path, content: str) -> None:
    source, producer = save(tmp_path / "producer", content)
    with pytest.raises(ValueError, match=r"JSON|validation|Expecting"):
        resolve_metric(reference(series(), producer), source)


def test_resolution_verifies_bytes_and_bounded_materialization(tmp_path: Path) -> None:
    payload = series()
    source, producer = save(tmp_path / "producer", payload.model_dump_json())
    external = reference(payload, producer)
    path = source.directory / "outputs/payload/metric.json"
    size = path.stat().st_size
    with pytest.raises(ValueError, match="max_bytes"):
        resolve_metric(external, source, max_bytes=size - 1)
    assert resolve_metric(external, source, max_bytes=size) == payload
    for limit in [True, 0, -1]:
        with pytest.raises(ValueError, match="positive integer"):
            resolve_metric(external, source, max_bytes=limit)
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        resolve_metric(external, source)


def test_resolution_rechecks_content_after_manifest_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, producer = save(tmp_path / "producer", series().model_dump_json())

    def changed(directory: Path) -> StepResult:
        record = read_step(directory)
        (directory / "outputs/payload/metric.json").write_text("changed")
        return record

    monkeypatch.setattr("mlef.metric_payloads.read_step", changed)
    with pytest.raises(ValueError, match="bytes changed"):
        resolve_metric(reference(series(), producer), source)


def test_resolution_rejects_incomplete_attempts(tmp_path: Path) -> None:
    source, producer = save(tmp_path / "producer", series().model_dump_json())
    path = source.directory / "step.json"
    record = json.loads(path.read_text())
    record["state"] = "running"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="incomplete"):
        resolve_metric(reference(series(), producer), source)
