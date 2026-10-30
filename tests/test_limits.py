import os
from pathlib import Path

import pytest

from mlef.cli import init, main
from mlef.limits import Limits, memory_bytes
from mlef.records import read_run
from mlef.runner import run


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--cpus", "0"),
        ("--cpus", "1e-12"),
        ("--cpus", "nan"),
        ("--memory", "-1"),
        ("--memory", "unlimited"),
    ],
)
def test_invalid_controls_do_not_allocate(project: Path, flag: str, value: str) -> None:
    init(project)
    (project / "train.py").write_text("pass")
    assert main(["run", "train.py", flag, value]) == 2
    assert list((project / ".mlef/runs").iterdir()) == []


def test_requests_fail_when_not_enforced() -> None:
    limits = Limits(cpus=0.5, memory_bytes=64 * 1024**2, network="none")
    assert memory_bytes("64m") == 64 * 1024**2
    with pytest.raises(RuntimeError, match="quota enforcement"):
        limits.capabilities({})
    with pytest.raises(RuntimeError, match="CPU limit"):
        limits.applied({"NanoCpus": 0})


@pytest.mark.skipif(os.environ.get("MLEF_TEST_DOCKER") != "1", reason="Set MLEF_TEST_DOCKER=1")
@pytest.mark.parametrize("scenario", ["controls", "oom", "exit137"])
def test_real_runtime_limits(project: Path, scenario: str) -> None:
    init(project)
    source = "print('controlled')\n"
    if scenario == "oom":
        source = (
            "import os\nfrom pathlib import Path\n"
            "Path(os.environ['MLEF_OUTPUT_DIR'], 'partial').write_text('kept')\n"
            "a=[]\nwhile True: a.append(bytearray(8*1024*1024))\n"
        )
    elif scenario == "exit137":
        source = "raise SystemExit(137)\n"
    (project / "train.py").write_text(source)
    assert run(project, Path("train.py"), [], [], [], cpus=0.5, memory="64m", network="none") == (
        0 if scenario == "controls" else 1
    )
    (directory,) = (project / ".mlef/runs").iterdir()
    record = read_run(directory / "run.json")
    assert record.controls is not None
    assert record.controls.applied["NanoCpus"] == 500_000_000
    assert record.controls.applied["Memory"] == 64 * 1024**2
    assert record.controls.applied["NetworkMode"] == "none"
    if scenario != "controls":
        assert record.error is not None
        assert record.error.category == ("oom" if scenario == "oom" else "execution")
    if scenario == "oom":
        assert (directory / "artifacts/partial").read_text() == "kept"
