import json
import os
from pathlib import Path
from uuid import uuid4

import pytest

from mlef.cli import init, main
from mlef.comparison import compare
from mlef.evidence import normalize
from mlef.records import Dataset, Evaluation, Experiment, RunRecord
from mlef.runner import save


def specification() -> Experiment:
    return Experiment(
        hypothesis="Test a declared change",
        parameters={"feature": False},
        dataset=Dataset(fingerprint="data-v1"),
        evaluation=Evaluation(
            protocol_id="holdout-v1",
            split_fingerprint="split-v1",
            seeds={},
            metric_definitions={"score": "accuracy-v1"},
        ),
        metric_directions={"score": "maximize"},
    )


def saved(
    project: Path,
    metrics: str | None,
    experiment: Experiment | None = None,
    state: str = "succeeded",
) -> str:
    identifier = str(uuid4())
    directory = project / ".mlef/runs" / identifier
    (directory / "artifacts").mkdir(parents=True)
    record = RunRecord.model_validate(
        {
            "id": identifier,
            "state": state,
            "created_at": "2026-01-01T00:00:00Z",
            "started_at": "2026-01-01T00:00:01Z",
            "finished_at": "2026-01-01T00:00:03Z",
            "command": ["python", "train.py"],
            "execution_exit_code": 0 if state == "succeeded" else 7,
            "execution_seconds": 2.0,
            "experiment": experiment,
        }
    )
    save(directory / "run.json", record)
    if metrics is not None:
        (directory / "artifacts/metrics.json").write_text(metrics)
    save(directory / "result.json", normalize(project, identifier))
    return identifier


@pytest.mark.parametrize(
    "metrics",
    [
        '{"score": true}',
        '{"score": NaN}',
        '{"score": 1e999}',
        '{"score": {"nested": 1}}',
        '{"score":1,"score":2}',
        "[]",
        "bad",
    ],
)
def test_invalid_optional_metrics_remain_retrievable(project: Path, metrics: str) -> None:
    identifier = saved(project, metrics)
    result = normalize(project, identifier)
    assert result.evidence_status == "invalid"
    assert result.metrics == {}
    assert result.diagnostics[0].code == "metrics_invalid"
    assert (project / ".mlef/runs" / identifier / "artifacts/metrics.json").read_text() == metrics
    assert (
        json.loads((project / ".mlef/runs" / identifier / "run.json").read_text())["state"]
        == "succeeded"
    )


def test_comparison_reports_deltas_parameters_and_lineage(project: Path) -> None:
    first = saved(project, '{"score":0.5}', specification())
    candidate = specification()
    candidate.parent_run_id = first
    candidate.parameters["feature"] = True
    candidate.changed_variables = {"feature": "enabled"}
    second = saved(project, '{"score":0.75}', candidate)
    comparison = compare(project, first, second)
    assert comparison.compatibility == "declared-compatible"
    (metric,) = comparison.metrics
    assert metric.delta == 0.25
    assert metric.relative_delta == 0.5
    assert metric.observation == "improved"
    assert comparison.parameters["feature"].baseline is False
    assert comparison.parameters["feature"].candidate is True
    assert comparison.candidate_experiment == candidate.model_dump(mode="json")
    assert comparison.baseline_seconds == 2


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ("split", "incompatible"),
        ("definition", "incompatible"),
        ("direction", "incompatible"),
        ("seeds", "incompatible"),
        ("missing_dataset", "unknown"),
        ("missing_seeds", "unknown"),
        ("failed", "unknown"),
        ("conflict_and_missing", "incompatible"),
    ],
)
def test_compatibility_never_invents_a_winner(project: Path, change: str, expected: str) -> None:
    first = saved(project, '{"score":0.5}', specification())
    candidate = specification()
    assert candidate.evaluation is not None
    if change in {"split", "conflict_and_missing"}:
        candidate.evaluation.split_fingerprint = "different-split"
    if change in {"missing_dataset", "conflict_and_missing"}:
        candidate.dataset = None
    if change == "definition":
        candidate.evaluation.metric_definitions["score"] = "different-definition"
    if change == "direction":
        candidate.metric_directions["score"] = "minimize"
    if change == "seeds":
        candidate.evaluation.seeds = {"split": 7}
    if change == "missing_seeds":
        candidate.evaluation.seeds = None
    second = saved(
        project, '{"score":0.9}', candidate, "failed" if change == "failed" else "succeeded"
    )
    comparison = compare(project, first, second)
    assert comparison.compatibility == expected
    assert comparison.metrics[0].observation == "unavailable"


def test_missing_legacy_and_zero_baseline(project: Path) -> None:
    first = saved(project, '{"score":0}', specification())
    second = saved(project, '{"score":0.5}', specification())
    row = compare(project, first, second).metrics[0]
    assert row.relative_delta is None
    assert any("zero baseline" in reason for reason in row.reasons)
    (project / ".mlef/runs" / second / "result.json").unlink()
    assert compare(project, first, second).compatibility == "unknown"
    third = saved(project, None)
    assert compare(project, third, third).compatibility == "unknown"


@pytest.mark.parametrize(
    "content",
    [
        '{"unknown": 1}',
        '{"schema_version":true}',
        '{"seeds":{"seed":true}}',
        '{"parameters":{"x":NaN}}',
        '{"parameters":{"x":1e999}}',
    ],
)
def test_invalid_experiments_do_not_allocate_runs(project: Path, content: str) -> None:
    init(project)
    (project / "train.py").write_text("print('unused')")
    (project / "experiment.json").write_text(content)
    assert main(["run", "train.py", "--experiment", "experiment.json"]) == 2
    assert list((project / ".mlef/runs").iterdir()) == []


def test_nonexistent_parent_and_nonfinite_model_are_rejected(project: Path) -> None:
    init(project)
    (project / "train.py").write_text("print('unused')")
    (project / "experiment.json").write_text(json.dumps({"parent_run_id": str(uuid4())}))
    assert main(["run", "train.py", "--experiment", "experiment.json"]) == 2
    with pytest.raises(ValueError, match="finite"):
        Experiment(parameters={"nested": [float("nan")]})


def test_json_comparison_and_bounded_summary(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    metrics = json.dumps({f"metric_{i}": i for i in range(25)})
    first, second = saved(project, metrics), saved(project, metrics)
    assert main(["compare", first, second, "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)["metrics"]) == 25
    assert main(["compare", first, second]) == 0
    assert "5 metrics omitted" in capsys.readouterr().out


def test_named_pipe_metrics_are_invalid_without_reading(project: Path) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("POSIX named pipes are unavailable on this host")
    identifier = saved(project, None)
    os.mkfifo(project / ".mlef/runs" / identifier / "artifacts/metrics.json")
    result = normalize(project, identifier)
    assert result.evidence_status == "invalid"
    assert "regular file" in result.diagnostics[0].message
