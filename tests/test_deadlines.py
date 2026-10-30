import json
import os
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import pytest

from mlef import supervision
from mlef.cli import init, main
from mlef.records import RunRecord, read_run
from mlef.runner import BASE_IMAGE, run, save


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_bad_deadline_does_not_allocate(project: Path, value: str) -> None:
    init(project)
    (project / "train.py").write_text("pass")
    assert main(["run", "train.py", "--timeout", value]) == 2
    assert list((project / ".mlef/runs").iterdir()) == []


@pytest.mark.parametrize(
    "phase", ["source_capture", "image_pull", "image_build", "environment_probe"]
)
def test_stalled_worker_is_bounded_and_preserves_stage(
    project: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    init(project)
    identifier, token = str(uuid4()), str(uuid4())
    run_dir = project / ".mlef/runs" / identifier
    run_dir.mkdir()
    record = RunRecord(
        id=identifier,
        state="preparing",
        created_at="2026-01-01T00:00:00Z",
        command=["python", "train.py"],
        stage=phase,
        runtime=supervision.identity(project, identifier, token),
    )
    save(run_dir / "run.json", record)
    real_popen = subprocess.Popen

    def stalled(path: Path) -> subprocess.Popen[bytes]:
        return real_popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            start_new_session=os.name != "nt",
        )

    monkeypatch.setattr(supervision, "start_worker", stalled)
    monkeypatch.setattr(supervision, "cleanup", lambda *args: True)
    began = time.monotonic()
    request = supervision.Request(
        root=str(project),
        script="train.py",
        arguments=[],
        includes=[],
        outputs=[],
        experiment=None,
        timeout=0.1,
        identifier=identifier,
        token=token,
    )
    assert supervision.supervise(request) == 1
    assert time.monotonic() - began < 8
    saved = read_run(run_dir / "run.json")
    assert saved.state == "failed"
    assert saved.error is not None
    assert saved.error.category == "timeout"
    assert phase in saved.error.message


def test_version_two_reads_unknown_controls(project: Path) -> None:
    path = project / "run.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "id": str(uuid4()),
                "state": "running",
                "created_at": "2026-01-01T00:00:00Z",
                "command": ["python", "x.py"],
            }
        )
    )
    record = read_run(path)
    assert record.schema_version == 3
    assert record.controls is None
    assert record.runtime is None


@pytest.mark.skipif(os.environ.get("MLEF_TEST_DOCKER") != "1", reason="Set MLEF_TEST_DOCKER=1")
def test_real_deadline_keeps_partial_outputs(project: Path) -> None:
    init(project)
    subprocess.run(["docker", "pull", BASE_IMAGE], check=True, capture_output=True, timeout=120)
    (project / "wait.py").write_text(
        "import os, time\nfrom pathlib import Path\n"
        "Path(os.environ['MLEF_OUTPUT_DIR'], 'partial.txt').write_text('kept')\n"
        "print('ready', flush=True)\ntime.sleep(120)\n"
    )
    began = time.monotonic()
    assert run(project, Path("wait.py"), [], [], [], timeout=15) == 1
    assert time.monotonic() - began < 38
    (directory,) = (project / ".mlef/runs").iterdir()
    record = read_run(directory / "run.json")
    assert record.error is not None
    assert record.error.category == "timeout"
    assert record.stage == "execution"
    assert (directory / "artifacts/partial.txt").read_text() == "kept"
    inspection = subprocess.run(
        ["docker", "inspect", f"mlef-{record.id}"], capture_output=True, check=False
    )
    assert inspection.returncode != 0
