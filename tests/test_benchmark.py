import csv
import importlib.util
import json
from pathlib import Path

import pytest


def test_fixture_hides_labels_and_repeats_data(tmp_path: Path) -> None:
    path = Path(__file__).resolve().parents[1] / "benchmarks/fixture.py"
    specification = importlib.util.spec_from_file_location("fixture", path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    first, second = tmp_path / "first", tmp_path / "second"
    module.prepare(first, 31415)
    module.prepare(second, 31415)
    for name in ["train.csv", "validation.csv", "test.csv", "task.json"]:
        assert (first / "agent" / name).read_bytes() == (second / "agent" / name).read_bytes()
    with (first / "agent/test.csv").open(newline="") as stream:
        assert "target" not in next(csv.reader(stream))
    labels = first / "private/labels.json"
    predictions = tmp_path / "predictions.json"
    predictions.write_bytes(labels.read_bytes())
    assert module.score(labels, predictions) == 1
    predictions.write_text(json.dumps({"unknown": 0}))
    with pytest.raises(ValueError, match="each held-out ID"):
        module.score(labels, predictions)
    fingerprint = json.loads((first / "agent/task.json").read_text())["task_fingerprint"]
    observation = {
        "task_fingerprint": fingerprint,
        "condition": "control",
        "agent": "fixed-agent",
        "model": "fixed-model",
        "budget_seconds": 60,
        "seed": 31415,
        "repetition": 0,
        "elapsed_seconds": 30.0,
        "tool_output_bytes": 1000,
        "intervention_count": 0,
        "valid_comparisons": 1,
        "time_to_first_valid_gain_seconds": 20.0,
        "heldout_accuracy": 0.8,
    }
    input_path, control, treatment = (
        tmp_path / "input.json",
        tmp_path / "control.json",
        tmp_path / "mlef.json",
    )
    input_path.write_text(json.dumps(observation))
    module.record(input_path, control)
    observation["condition"] = "mlef"
    input_path.write_text(json.dumps(observation))
    module.record(input_path, treatment)
    assert module.pair(control, treatment)["heldout_accuracy_delta"] == 0
    observation["model"] = "different-model"
    treatment.write_text(json.dumps(observation))
    with pytest.raises(ValueError, match="Unmatched trial field"):
        module.pair(control, treatment)
    observation["elapsed_seconds"] = 100.0
    input_path.write_text(json.dumps(observation))
    with pytest.raises(ValueError, match="exceeded"):
        module.record(input_path, tmp_path / "invalid.json")
