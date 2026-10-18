import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "mlef test project"
    root.mkdir()
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    monkeypatch.chdir(root)
    return root


@pytest.fixture
def cli() -> Callable[..., subprocess.CompletedProcess[str]]:
    def invoke(*arguments: str, expected: int = 0) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [sys.executable, "-m", "mlef", *arguments],
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert result.returncode == expected, result.stdout + result.stderr
        return result

    return invoke
