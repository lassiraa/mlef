"""Opt-in checks against a real Docker engine; no mock container execution."""

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import BinaryIO

import pytest
from pydantic import BaseModel

from mlef import runner
from mlef.cli import init
from mlef.comparison import compare
from mlef.environment import file_hash
from mlef.provenance import extract_source
from mlef.records import read_run
from mlef.storage import list_runs

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/train.py"
pytestmark = pytest.mark.skipif(
    os.environ.get("MLEF_TEST_DOCKER") != "1", reason="Set MLEF_TEST_DOCKER=1"
)


def test_tabular_feedback_loop(
    project: Path, cli: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    example = EXAMPLE.parent / "tabular"
    for name in [
        "pyproject.toml",
        "uv.lock",
        "train.py",
        "data.csv",
        "baseline.json",
        "candidate.json",
    ]:
        (project / name).write_bytes((example / name).read_bytes())
    baseline_spec = json.loads((project / "baseline.json").read_text())
    candidate_spec = json.loads((project / "candidate.json").read_text())
    baseline_spec["dataset"]["fingerprint"] = candidate_spec["dataset"]["fingerprint"] = file_hash(
        project / "data.csv"
    )
    (project / "baseline.json").write_text(json.dumps(baseline_spec))
    cli("init")
    cli("run", "train.py", "--include", "data.csv", "--experiment", "baseline.json")
    (baseline,) = list_runs(project)[0]
    candidate_spec["parent_run_id"] = baseline.id
    (project / "candidate.json").write_text(json.dumps(candidate_spec))
    cli(
        "run",
        "train.py",
        "--include",
        "data.csv",
        "--experiment",
        "candidate.json",
        "--",
        "--candidate",
    )
    candidate = list_runs(project)[0][0]
    comparison = json.loads(cli("compare", baseline.id, candidate.id, "--json").stdout)
    assert comparison["compatibility"] == "declared-compatible"
    assert comparison["metrics"][0]["baseline"] == 0.5
    assert comparison["metrics"][0]["candidate"] == 1
    assert comparison["metrics"][0]["delta"] == 0.5
    assert comparison["candidate_experiment"]["parent_run_id"] == baseline.id
    assert (
        json.loads(cli("status", candidate.id, "--json").stdout)["result"]["evidence_status"]
        == "available"
    )
    assert len(json.loads(cli("list", "--json").stdout)) == 2
    cli("get", candidate.id, "metrics.json", "--output", "retrieved.json")
    assert json.loads((project / "retrieved.json").read_text())["accuracy"] == 1
    candidate_spec["evaluation"]["split_fingerprint"] = "different-split"
    (project / "candidate.json").write_text(json.dumps(candidate_spec))
    cli(
        "run",
        "train.py",
        "--include",
        "data.csv",
        "--experiment",
        "candidate.json",
        "--",
        "--candidate",
    )
    mismatch = list_runs(project)[0][0]
    assert compare(project, baseline.id, mismatch.id).compatibility == "incompatible"


def test_independent_runs_keep_outputs_separate(
    project: Path, cli: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    cli("init")
    (project / "train.py").write_bytes(EXAMPLE.read_bytes())
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(cli, "run", "train.py", "--", "--accuracy", value)
            for value in ["0.2", "0.8"]
        ]
        for future in futures:
            future.result()
    records = list_runs(project)[0]
    assert len(records) == 2
    scores = set()
    for record in records:
        assert record.state == "succeeded"
        output = project / ".mlef/runs" / record.id / "artifacts/metrics.json"
        scores.add(json.loads(output.read_text())["accuracy"])
    assert scores == {0.2, 0.8}


def test_invalid_metrics_preserve_success(project: Path) -> None:
    init(project)
    (project / "train.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "Path(os.environ['MLEF_OUTPUT_DIR'], 'metrics.json').write_text('{\"score\":true}')\n"
    )
    assert runner.run(project, Path("train.py"), [], [], []) == 0
    (directory,) = (project / ".mlef/runs").iterdir()
    assert read_run(directory / "run.json").state == "succeeded"
    assert json.loads((directory / "result.json").read_text())["evidence_status"] == "invalid"


def test_required_result_write_failure_fails_run(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init(project)
    (project / "train.py").write_bytes(EXAMPLE.read_bytes())
    original_save = runner.save

    def fail_result(path: Path, data: dict[str, object] | BaseModel) -> None:
        if path.name == "result.json":
            raise OSError("simulated full disk for required result")
        original_save(path, data)

    monkeypatch.setattr(runner, "save", fail_result)
    assert runner.run(project, Path("train.py"), [], [], []) == 1
    (directory,) = (project / ".mlef/runs").iterdir()
    record = read_run(directory / "run.json")
    assert record.state == "failed"
    assert record.execution_exit_code == 0
    assert record.error is not None
    assert record.error.category == "storage"


def test_locked_environment_and_manual_repeat(project: Path) -> None:
    example = EXAMPLE.parent / "locked"
    for name in ["pyproject.toml", "uv.lock", "train.py"]:
        (project / name).write_bytes((example / name).read_bytes())
    lock = (project / "uv.lock").read_bytes()
    init(project)
    assert runner.run(project, Path("train.py"), [], [], []) == 0
    (directory,) = (project / ".mlef/runs").iterdir()
    metadata = json.loads((directory / "run.json").read_text())
    environment = json.loads((directory / "environment.json").read_text())
    assert environment["container"]["packages"]["packaging"] == "25.0"
    assert environment["container"]["python"].startswith("3.12.")
    assert environment["container"]["os_release"]["ID"]
    assert environment["container"]["uv_version"].split()[:2] == ["uv", "0.13.6"]
    assert environment["base_image"]["digests"]
    assert environment["uv_image"]["digests"]
    assert environment["execution_image"]["id"].startswith("sha256:")
    assert environment["warnings"] == metadata["warnings"] == []
    assert metadata["inputs"]["dependency_hashes"] == {
        name: file_hash(project / name) for name in ["pyproject.toml", "uv.lock"]
    }
    assert metadata["inputs"]["build_recipe_sha256"] == file_hash(directory / "Dockerfile")
    assert (project / "uv.lock").read_bytes() == lock
    (project / "train.py").write_text("raise RuntimeError('edited after original run')")
    (project / "pyproject.toml").write_text("invalid metadata")
    repeat = project.parent / "repeat"
    repeat.mkdir()
    extract_source(directory / "source.tar.gz", repeat / "workspace")
    (repeat / "Dockerfile").write_bytes((directory / "Dockerfile").read_bytes())
    outputs = repeat / "outputs"
    outputs.mkdir()
    image = f"mlef-repeat:{metadata['id']}"
    with (repeat / "runner.log").open("ab") as log:
        try:
            runner.docker(["build", "--tag", image, str(repeat)], log)
            runner.docker(
                [
                    "run",
                    "--rm",
                    "--env",
                    "MLEF_OUTPUT_DIR=/outputs",
                    "--volume",
                    f"{outputs}:/outputs",
                    image,
                    *metadata["command"],
                ],
                log,
            )
        finally:
            runner.docker(["image", "rm", image], log, check=False)
    assert (outputs / "metrics.json").read_bytes() == (
        directory / "artifacts/metrics.json"
    ).read_bytes()
    assert (repeat / "workspace/uv.lock").read_bytes() == lock


def test_stale_lock_fails_without_resolving_or_running(project: Path) -> None:
    example = EXAMPLE.parent / "locked"
    for name in ["pyproject.toml", "uv.lock", "train.py"]:
        (project / name).write_bytes((example / name).read_bytes())
    lock = (project / "uv.lock").read_bytes()
    manifest = project / "pyproject.toml"
    manifest.write_text(manifest.read_text().replace("packaging==25.0", "packaging==24.2"))
    init(project)
    assert runner.run(project, Path("train.py"), [], [], []) == 1
    (directory,) = (project / ".mlef/runs").iterdir()
    metadata = json.loads((directory / "run.json").read_text())
    environment = json.loads((directory / "environment.json").read_text())
    assert metadata["state"] == "failed"
    assert metadata["started_at"] is None
    assert metadata["execution_exit_code"] is None
    assert environment["container"] is None
    assert environment["warnings"]
    assert (project / "uv.lock").read_bytes() == lock
    assert (directory / "Dockerfile").exists()
    assert "lockfile" in (directory / "runner.log").read_text().lower()


def test_nonroot_container_can_write_workspace_and_artifacts(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init(project)
    (project / "train.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "Path('results').mkdir()\n"
        "Path('results/value.txt').write_text('workspace')\n"
        "Path(os.environ['MLEF_OUTPUT_DIR'], 'value.txt').write_text('artifact')\n"
    )
    uid = os.getuid() if hasattr(os, "getuid") else 1000
    gid = os.getgid() if hasattr(os, "getgid") else 1000
    monkeypatch.setattr(runner.os, "getuid", lambda: uid, raising=False)
    monkeypatch.setattr(runner.os, "getgid", lambda: gid, raising=False)
    assert runner.run(project, Path("train.py"), [], [], [Path("results")]) == 0
    (directory,) = (project / ".mlef/runs").iterdir()
    assert (directory / "artifacts/results/value.txt").read_text() == "workspace"
    assert (directory / "artifacts/value.txt").read_text() == "artifact"


def test_success_failure_arguments_and_declared_outputs(
    project: Path, cli: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    cli("init")
    (project / "train.py").write_bytes(EXAMPLE.read_bytes())
    success = cli("run", "train.py", "--", "--accuracy", "0.95")
    assert "Training complete" in success.stdout
    failure = cli("run", "train.py", "--", "--fail", expected=1)
    assert "Requested example failure" in failure.stderr
    directories = list((project / ".mlef/runs").iterdir())
    assert len(directories) == 2
    runs = {json.loads((path / "run.json").read_text())["state"]: path for path in directories}
    assert set(runs) == {"succeeded", "failed"}
    for state, directory in runs.items():
        metadata = json.loads((directory / "run.json").read_text())
        assert metadata["execution_exit_code"] == (0 if state == "succeeded" else 7)
        assert metadata["started_at"]
        assert metadata["finished_at"]
        assert "Training complete" in (directory / "stdout.log").read_text()
        assert (directory / "artifacts/model.bin").exists()
    metrics = json.loads((runs["succeeded"] / "artifacts/metrics.json").read_text())
    assert metrics["accuracy"] == 0.95
    (project / "train.py").write_text(
        "from pathlib import Path\n"
        "Path('results').mkdir()\n"
        "Path('results/value.txt').write_text('collected')\n"
    )
    cli("run", "train.py", "--output", "results")
    collected = list((project / ".mlef/runs").glob("*/artifacts/results/value.txt"))
    assert len(collected) == 1
    assert collected[0].read_text() == "collected"
    assert not (project / "results").exists()

    # Missing required outputs must not silently turn a run into a success.
    cli("run", "train.py", "--output", "missing", expected=1)
    metadata = [
        json.loads(path.read_text()) for path in (project / ".mlef/runs").glob("*/run.json")
    ]
    collection_errors = [
        item for item in metadata if (item.get("error") or {}).get("category") == "collection"
    ]
    assert len(collection_errors) == 1
    assert collection_errors[0]["execution_exit_code"] == 0


def test_interrupt_handler_with_real_container(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Inject KeyboardInterrupt at the wait boundary to cover the handler on Windows too.
    init(project)
    (project / "wait.py").write_text(
        "import os, time\nfrom pathlib import Path\n"
        "Path(os.environ['MLEF_OUTPUT_DIR'], 'partial.txt').write_text('saved')\n"
        "print('ready', flush=True)\ntime.sleep(120)\n"
    )
    real_stream = runner.stream_process

    def interrupted_stream(
        process: subprocess.Popen[bytes], run_dir: Path, name: str, log: BinaryIO
    ) -> int:
        original_wait = process.wait
        interrupted = False

        def wait(timeout: float | None = None) -> int:
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                deadline = time.monotonic() + 30
                while not list((project / ".mlef/runs").glob("*/artifacts/partial.txt")):
                    if time.monotonic() > deadline:
                        raise RuntimeError("Container did not produce a partial output")
                    time.sleep(0.1)
                raise KeyboardInterrupt
            return original_wait(timeout)

        monkeypatch.setattr(process, "wait", wait)
        return real_stream(process, run_dir, name, log)

    monkeypatch.setattr(runner, "stream_process", interrupted_stream)
    assert runner.run(project, Path("wait.py"), [], [], []) == 130
    (directory,) = (project / ".mlef/runs").iterdir()
    metadata = json.loads((directory / "run.json").read_text())
    assert metadata["state"] == "cancelled"
    assert metadata["finished_at"]
    assert (directory / "artifacts/partial.txt").read_text() == "saved"
    assert "ready" in (directory / "stdout.log").read_text()


@pytest.mark.skipif(sys.platform == "win32", reason="SIGINT delivery checked on POSIX CI")
def test_cancellation_preserves_partial_outputs(
    project: Path, cli: Callable[..., subprocess.CompletedProcess[str]]
) -> None:
    cli("init")
    (project / "wait.py").write_text(
        "import os, time\nfrom pathlib import Path\n"
        "Path(os.environ['MLEF_OUTPUT_DIR'], 'partial.txt').write_text('saved')\n"
        "print('ready', flush=True)\ntime.sleep(120)\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "mlef", "run", "wait.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 120
        while not list((project / ".mlef/runs").glob("*/artifacts/partial.txt")):
            if process.poll() is not None:
                pytest.fail(str(process.communicate()))
            if time.monotonic() > deadline:
                pytest.fail("Container never wrote its partial output")
            time.sleep(0.2)
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 130, stdout + stderr
        (directory,) = (project / ".mlef/runs").iterdir()
        metadata = json.loads((directory / "run.json").read_text())
        assert metadata["state"] == "cancelled"
        assert (directory / "artifacts/partial.txt").read_text() == "saved"
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
