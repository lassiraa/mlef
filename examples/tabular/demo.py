"""Run this from the copied example's Git root using the installed mlef command."""

import hashlib
import json
import subprocess
from pathlib import Path


def invoke(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["mlef", *arguments], check=True, text=True, capture_output=True)


def main() -> None:
    invoke("init")
    fingerprint = hashlib.sha256(Path("data.csv").read_bytes()).hexdigest()
    baseline = json.loads(Path("baseline.json").read_text())
    candidate = json.loads(Path("candidate.json").read_text())
    baseline["dataset"]["fingerprint"] = candidate["dataset"]["fingerprint"] = fingerprint
    Path("baseline.json").write_text(json.dumps(baseline), encoding="utf-8")
    first = invoke("run", "train.py", "--include", "data.csv", "--experiment", "baseline.json")
    baseline_id = first.stdout.splitlines()[0].removeprefix("Run ID: ")
    candidate["parent_run_id"] = baseline_id
    Path("candidate.json").write_text(json.dumps(candidate), encoding="utf-8")
    second = invoke(
        "run",
        "train.py",
        "--include",
        "data.csv",
        "--experiment",
        "candidate.json",
        "--",
        "--candidate",
    )
    candidate_id = second.stdout.splitlines()[0].removeprefix("Run ID: ")
    print(invoke("compare", baseline_id, candidate_id, "--json").stdout)
    invoke("get", candidate_id, "metrics.json", "--output", "candidate-metrics.json")


if __name__ == "__main__":
    main()
