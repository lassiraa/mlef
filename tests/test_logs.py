import os
import subprocess
import sys
from pathlib import Path

import pytest

from mlef.cli import init, main
from mlef.processes import BoundedLog, LogLimitError, command
from mlef.records import read_run
from mlef.runner import run


def test_native_command_logs_are_capped_without_deadlock(tmp_path: Path) -> None:
    path = tmp_path / "runner.log"
    with BoundedLog(str(path), 1000) as log, pytest.raises(LogLimitError):
        command(
            [sys.executable, "-c", "import os,time; os.write(2,b'x'*200000); time.sleep(60)"],
            log,
        )
    assert path.stat().st_size == 1000


@pytest.mark.parametrize("flag", ["--max-log-bytes", "--max-runner-log-bytes"])
def test_bad_log_limit_does_not_allocate(project: Path, flag: str) -> None:
    init(project)
    (project / "train.py").write_text("pass")
    assert main(["run", "train.py", flag, "0"]) == 2
    assert list((project / ".mlef/runs").iterdir()) == []


@pytest.mark.skipif(os.environ.get("MLEF_TEST_DOCKER") != "1", reason="Set MLEF_TEST_DOCKER=1")
@pytest.mark.parametrize("runner_budget", [False, True])
def test_real_log_flood_preserves_bounded_evidence(project: Path, runner_budget: bool) -> None:
    init(project)
    (project / "train.py").write_text(
        "import os, threading\n"
        "def flood(fd):\n while True: os.write(fd, 'é'.encode()*5000)\n"
        "threading.Thread(target=flood,args=(1,)).start()\nflood(2)\n",
        encoding="utf-8",
    )
    assert (
        run(
            project,
            Path("train.py"),
            [],
            [],
            [],
            timeout=30,
            max_runner_log_bytes=64 if runner_budget else None,
            max_log_bytes=None if runner_budget else 1001,
        )
        == 1
    )
    (directory,) = (project / ".mlef/runs").iterdir()
    record = read_run(directory / "run.json")
    assert record.error is not None
    assert record.error.category == "output_limit"
    if runner_budget:
        assert (directory / "runner.log").stat().st_size <= 64
    else:
        assert (
            sum((directory / name).stat().st_size for name in ["stdout.log", "stderr.log"]) <= 1001
        )
    result = subprocess.run(
        ["docker", "inspect", f"mlef-{record.id}"], capture_output=True, check=False
    )
    assert result.returncode != 0
