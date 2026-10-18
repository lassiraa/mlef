import subprocess
import sys

import pytest


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "mlef test project"
    root.mkdir()
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    monkeypatch.chdir(root)
    return root


@pytest.fixture
def cli():
    def invoke(*arguments, expected=0):
        result = subprocess.run(
            [sys.executable, "-m", "mlef", *arguments],
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert result.returncode == expected, result.stdout + result.stderr
        return result

    return invoke
