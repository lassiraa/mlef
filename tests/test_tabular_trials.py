import csv
import importlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from types import ModuleType

import pytest

from mlef import api
from mlef.benchmark import Holdout, TabularTrial
from mlef.records import read_json

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"


@pytest.fixture
def benchmark(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    return importlib.import_module("tabular_trials")


@pytest.mark.parametrize(
    "task",
    ["classification-mixed", "classification-missing", "regression-mixed", "regression-missing"],
)
def test_prepared_tasks_repeat_and_hide_final_labels(
    tmp_path: Path, benchmark: ModuleType, task: str
) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    benchmark.prepare_tabular(first, task, 17)
    benchmark.prepare_tabular(second, task, 17)
    for name in [
        "exploration.csv",
        "submission.csv",
        "protocol.json",
        "task.json",
        "model.py",
        "reference.py",
        "pyproject.toml",
        "uv.lock",
    ]:
        assert (first / "agent" / name).read_bytes() == (second / "agent" / name).read_bytes()
    with (first / "agent/submission.csv").open() as stream:
        assert "target" not in next(csv.reader(stream))
    assert not list((first / "agent").rglob("labels*"))
    holdout = Holdout.model_validate(read_json(first / "private/labels.json"))
    assert len(holdout.labels) > 20


def perfect_submission(destination: Path, holdout: Holdout) -> Path:
    path = destination / "predictions.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "prediction"])
        writer.writerows(holdout.labels.items())
    return path


@pytest.mark.parametrize("task", ["classification-mixed", "regression-mixed"])
def test_frozen_score_ledger_and_trial_recording(
    tmp_path: Path, benchmark: ModuleType, task: str
) -> None:
    fixture = importlib.import_module("fixture")
    destination = tmp_path / "trial"
    benchmark.prepare_tabular(destination, task, 17)
    labels = destination / "private/labels.json"
    holdout = Holdout.model_validate(read_json(labels))
    predictions = perfect_submission(tmp_path, holdout)
    candidate = destination / "agent/model.py"
    score = benchmark.score_tabular(labels, predictions, candidate)
    assert score.score == (1 if holdout.task == "binary" else 0)
    assert benchmark.score_tabular(labels, predictions, candidate) == score
    trial = TabularTrial(
        task_fingerprint=holdout.task_fingerprint,
        protocol_id=holdout.protocol_id,
        task=holdout.task,
        metric=holdout.metric,
        direction=holdout.direction,
        condition="control",
        agent="fixture-agent",
        model="fixture-model",
        budget_seconds=60,
        seed=17,
        repetition=0,
        state="completed",
        elapsed_seconds=2.0,
        tool_output_bytes=100,
        intervention_count=0,
        valid_experiments=1,
        valid_comparisons=1,
        time_to_first_valid_gain_seconds=1.0,
        heldout_score=score.score,
        prediction_sha256=score.prediction_sha256,
        candidate_sha256=score.candidate_sha256,
    )
    input_path = tmp_path / "trial.json"
    input_path.write_text(trial.model_dump_json())
    with pytest.raises(ValueError, match="score ledger"):
        fixture.record(input_path, tmp_path / "invalid.json")
    fixture.record(input_path, tmp_path / "control.json", labels.with_suffix(".score.json"))
    values = trial.model_dump()
    values["condition"] = "mlef"
    input_path.write_text(TabularTrial.model_validate(values).model_dump_json())
    fixture.record(input_path, tmp_path / "mlef.json", labels.with_suffix(".score.json"))
    assert fixture.pair(tmp_path / "control.json", tmp_path / "mlef.json")["improvement"] == 0
    values.update(state="failed", failure="timeout", heldout_score=None, elapsed_seconds=61.0)
    input_path.write_text(TabularTrial.model_validate(values).model_dump_json())
    fixture.record(input_path, tmp_path / "failed.json")
    pair = fixture.pair(tmp_path / "control.json", tmp_path / "failed.json")
    assert pair["improvement"] is None
    assert pair["mlef_budget_exceeded"] is True
    candidate.write_text(candidate.read_text() + "\n# different finalist\n")
    with pytest.raises(ValueError, match="already scored"):
        benchmark.score_tabular(labels, predictions, candidate)


@pytest.mark.parametrize("damage", ["missing", "duplicate", "nan", "extra_column"])
def test_invalid_submissions_do_not_consume_final_evaluation(
    tmp_path: Path, benchmark: ModuleType, damage: str
) -> None:
    destination = tmp_path / "trial"
    benchmark.prepare_tabular(destination, "classification-mixed", 17)
    labels = destination / "private/labels.json"
    holdout = Holdout.model_validate(read_json(labels))
    predictions = perfect_submission(tmp_path, holdout)
    lines = predictions.read_text().splitlines()
    if damage == "missing":
        lines.pop()
    elif damage == "duplicate":
        lines.append(lines[1])
    elif damage == "nan":
        lines[1] = lines[1].split(",")[0] + ",NaN"
    else:
        lines[1] += ",extra"
    predictions.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match=r"exactly|unique finite"):
        benchmark.score_tabular(labels, predictions, destination / "agent/model.py")
    assert not labels.with_suffix(".score.json").exists()


@pytest.mark.parametrize("task", ["classification-mixed", "regression-mixed"])
def test_conventional_search_reference_dry_run(
    tmp_path: Path, benchmark: ModuleType, task: str
) -> None:
    pytest.importorskip("xgboost")
    destination = tmp_path / "trial"
    benchmark.prepare_tabular(destination, task, 17)
    workspace = destination / "agent"
    result = subprocess.run(
        [sys.executable, "reference.py"], cwd=workspace, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    score = benchmark.score_tabular(
        destination / "private/labels.json",
        workspace / "reference-output/predictions.csv",
        workspace / "reference-output/model.joblib",
    )
    assert score.rows > 20
    reference = json.loads((workspace / "reference-output/reference.json").read_text())
    assert reference["search_candidates"] == 4
    assert reference["folds"] == 3


@pytest.mark.skipif(os.environ.get("MLEF_TEST_DOCKER") != "1", reason="Set MLEF_TEST_DOCKER=1")
def test_reference_docker_inputs_exclude_private_labels(
    tmp_path: Path, benchmark: ModuleType
) -> None:
    destination = tmp_path / "trial"
    benchmark.prepare_tabular(destination, "classification-mixed", 17)
    workspace = destination / "agent"
    subprocess.run(["git", "init", "--quiet", str(workspace)], check=True)
    api.init(root=workspace)
    result = api.run(
        "reference.py",
        root=workspace,
        includes=["exploration.csv", "submission.csv", "protocol.json", "model.py", "task.json"],
        options=api.RunOptions(timeout_seconds=240, max_log_bytes=65536),
    )
    assert result.state == "succeeded", result.model_dump_json()
    with tarfile.open(workspace / ".mlef/runs" / result.id / "source.tar.gz") as archive:
        assert not any(
            "private" in member.name or "labels.json" in member.name
            for member in archive.getmembers()
        )
    for artifact in ["predictions.csv", "model.joblib"]:
        api.get(result.id, artifact, root=workspace, output=tmp_path / artifact)
    score = benchmark.score_tabular(
        destination / "private/labels.json", tmp_path / "predictions.csv", tmp_path / "model.joblib"
    )
    assert score.metric == "roc_auc"


@pytest.mark.parametrize("task", ["binary", "regression"])
def test_matched_trial_direction_settings_and_budget(benchmark: ModuleType, task: str) -> None:
    values = {
        "task_fingerprint": "a" * 64,
        "protocol_id": "b" * 64,
        "task": task,
        "metric": "roc_auc" if task == "binary" else "rmse",
        "direction": "maximize" if task == "binary" else "minimize",
        "condition": "control",
        "agent": "fixture",
        "model": "fixture",
        "budget_seconds": 60,
        "seed": 17,
        "repetition": 0,
        "state": "completed",
        "elapsed_seconds": 2.0,
        "tool_output_bytes": 100,
        "intervention_count": 0,
        "valid_experiments": 1,
        "valid_comparisons": 1,
        "heldout_score": 0.6 if task == "binary" else 2.0,
        "prediction_sha256": "c" * 64,
        "candidate_sha256": "d" * 64,
        "agent_settings": {"temperature": False},
    }
    control = TabularTrial.model_validate(values)
    treatment = TabularTrial.model_validate(
        {**values, "condition": "mlef", "heldout_score": 0.7 if task == "binary" else 1.0}
    )
    assert benchmark.pair_tabular(control, treatment)["improvement"] == pytest.approx(
        0.1 if task == "binary" else 1.0
    )
    with pytest.raises(ValueError, match="Unmatched trial"):
        benchmark.pair_tabular(
            control,
            TabularTrial.model_validate({**treatment.model_dump(), "protocol_id": "e" * 64}),
        )
    with pytest.raises(ValueError, match="agent_settings"):
        benchmark.pair_tabular(
            control,
            TabularTrial.model_validate(
                {**treatment.model_dump(), "agent_settings": {"temperature": 0}}
            ),
        )
    with pytest.raises(ValueError, match="exceeded"):
        TabularTrial.model_validate({**values, "elapsed_seconds": 61.0})
