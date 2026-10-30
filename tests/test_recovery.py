import json
import os
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import pytest

from mlef import recovery, supervision
from mlef.cli import init
from mlef.locking import run_lock, supervisor_lock
from mlef.records import RunRecord
from mlef.runner import save
from mlef.storage import QueryError


def abandoned(project: Path) -> Path:
    init(project)
    identifier = str(uuid4())
    directory = project / ".mlef/runs" / identifier
    (directory / "artifacts").mkdir(parents=True)
    save(
        directory / "run.json",
        RunRecord(
            id=identifier,
            state="running",
            created_at="2026-01-01T00:00:00Z",
            command=["python", "x.py"],
            runtime=supervision.identity(project, identifier, str(uuid4())),
        ),
    )
    return directory


@pytest.mark.parametrize("owner", ["worker", "supervisor"])
def test_recovery_refuses_active_owners(project: Path, owner: str) -> None:
    directory = abandoned(project)
    lock = run_lock(directory) if owner == "worker" else supervisor_lock(project, directory.name)
    with lock, pytest.raises(QueryError, match="active"):
        recovery.recover(project, directory.name)


@pytest.mark.parametrize("stalled", [False, True])
def test_offline_engine_does_not_change_metadata(
    project: Path, monkeypatch: pytest.MonkeyPatch, stalled: bool
) -> None:
    directory = abandoned(project)
    original = (directory / "run.json").read_bytes()

    def offline(*args: object) -> str:
        if stalled:
            raise subprocess.TimeoutExpired(["docker", "info"], 30)
        raise RuntimeError("engine offline")

    monkeypatch.setattr(recovery, "query", offline)
    with pytest.raises(QueryError, match="unavailable"):
        recovery.recover(project, directory.name)
    assert (directory / "run.json").read_bytes() == original


def test_cleanup_refuses_foreign_labels(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = abandoned(project)
    record = supervision.read_run(directory / "run.json")
    commands: list[list[str]] = []

    def docker(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(arguments)
        response = json.dumps(
            [
                {
                    "Id": "foreign",
                    "Config": {"Labels": {"mlef.owner": "other"}},
                    "State": {"Running": True},
                }
            ]
        )
        return subprocess.CompletedProcess(arguments, 0, response, "")

    monkeypatch.setattr(supervision.subprocess, "run", docker)
    assert not supervision.cleanup(project, record, directory)
    assert all(arguments[1] == "inspect" for arguments in commands)


def test_missing_resources_never_become_success(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = abandoned(project)
    monkeypatch.setattr(recovery, "query", lambda *args: "server")

    def missing(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            arguments,
            1 if arguments[1] == "inspect" else 0,
            "server",
            "No such object" if arguments[1] == "inspect" else "",
        )

    monkeypatch.setattr(supervision.subprocess, "run", missing)
    record = recovery.recover(project, directory.name)
    assert record.state == "failed"
    assert record.execution_exit_code is None
    original = (directory / "run.json").read_bytes()
    assert recovery.recover(project, directory.name) == record
    assert (directory / "run.json").read_bytes() == original


@pytest.mark.skipif(os.environ.get("MLEF_TEST_DOCKER") != "1", reason="Set MLEF_TEST_DOCKER=1")
def test_real_abandoned_container_recovery(project: Path) -> None:
    init(project)
    (project / "wait.py").write_text(
        "import os,time\nfrom pathlib import Path\n"
        "Path('results').mkdir()\nPath('results/value').write_text('workspace')\n"
        "Path(os.environ['MLEF_OUTPUT_DIR'], 'partial').write_text('kept')\n"
        "time.sleep(120)\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "mlef", "run", "wait.py", "--output", "results"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        start_new_session=os.name != "nt",
    )
    try:
        deadline = time.monotonic() + 75
        while not list((project / ".mlef/runs").glob("*/artifacts/partial")):
            if process.poll() is not None:
                pytest.fail(str(process.communicate()))
            if time.monotonic() > deadline:
                pytest.fail("Container did not produce partial evidence")
            time.sleep(0.1)
        (directory,) = (project / ".mlef/runs").iterdir()
        with pytest.raises(QueryError, match="active"):
            recovery.recover(project, directory.name)
        supervision.kill_tree(process)
        record = recovery.recover(project, directory.name)
        assert record.state == "failed"
        assert record.runtime is not None
        assert not record.runtime.cleanup_pending
        assert (directory / "artifacts/partial").read_text() == "kept"
        assert (directory / "artifacts/results/value").read_text() == "workspace"
        assert recovery.recover(project, directory.name) == record
    finally:
        if process.poll() is None:
            supervision.kill_tree(process)
        process.communicate(timeout=5)
