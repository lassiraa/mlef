"""Opt-in checks against a real Docker engine; no mock container execution."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from mlef import runner
from mlef.cli import init

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/train.py"
pytestmark = pytest.mark.skipif(
    os.environ.get("MLEF_TEST_DOCKER") != "1", reason="Set MLEF_TEST_DOCKER=1"
)


def test_nonroot_container_can_write_workspace_and_artifacts(project, monkeypatch):
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


def test_success_failure_arguments_and_declared_outputs(project, cli):
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


def test_interrupt_handler_with_real_container(project, monkeypatch):
    # Inject KeyboardInterrupt at the wait boundary to cover the handler on Windows too.
    init(project)
    (project / "wait.py").write_text(
        "import os, time\nfrom pathlib import Path\n"
        "Path(os.environ['MLEF_OUTPUT_DIR'], 'partial.txt').write_text('saved')\n"
        "print('ready', flush=True)\ntime.sleep(120)\n"
    )
    real_popen = subprocess.Popen

    def popen(command, *args, **kwargs):
        process = real_popen(command, *args, **kwargs)
        if command[:3] == ["docker", "start", "--attach"]:
            original_wait = process.wait
            interrupted = False

            def wait(timeout=None):
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
        return process

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    assert runner.run(project, Path("wait.py"), [], [], []) == 130
    (directory,) = (project / ".mlef/runs").iterdir()
    metadata = json.loads((directory / "run.json").read_text())
    assert metadata["state"] == "cancelled"
    assert metadata["finished_at"]
    assert (directory / "artifacts/partial.txt").read_text() == "saved"
    assert "ready" in (directory / "stdout.log").read_text()


@pytest.mark.skipif(sys.platform == "win32", reason="SIGINT delivery checked on POSIX CI")
def test_cancellation_preserves_partial_outputs(project, cli):
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
