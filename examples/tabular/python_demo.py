"""Run from this example's Git root with mlef installed; requires Docker."""

import hashlib
import json
from pathlib import Path

from mlef import api


def main() -> None:
    api.init()
    fingerprint = hashlib.sha256(Path("data.csv").read_bytes()).hexdigest()
    baseline = json.loads(Path("baseline.json").read_text())
    candidate = json.loads(Path("candidate.json").read_text())
    baseline["dataset"]["fingerprint"] = candidate["dataset"]["fingerprint"] = fingerprint
    Path("baseline.json").write_text(json.dumps(baseline), encoding="utf-8")
    options = api.RunOptions(timeout_seconds=180, max_log_bytes=65536)
    first = api.run("train.py", includes=["data.csv"], experiment="baseline.json", options=options)
    if first.state != "succeeded":
        raise RuntimeError(first.model_dump_json())
    candidate["parent_run_id"] = first.id
    Path("candidate.json").write_text(json.dumps(candidate), encoding="utf-8")
    second = api.run(
        "train.py",
        arguments=["--candidate"],
        includes=["data.csv"],
        experiment="candidate.json",
        options=options,
    )
    if second.state != "succeeded":
        raise RuntimeError(second.model_dump_json())
    print(api.compare(first.id, second.id).model_dump_json(indent=2))
    api.get(second.id, "metrics.json", output="python-candidate-metrics.json")


if __name__ == "__main__":
    main()
