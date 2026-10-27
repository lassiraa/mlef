import json
import subprocess
from pathlib import Path
from typing import BinaryIO
from unittest.mock import Mock

import pytest

from mlef.cli import main
from mlef.runner import relative_path, run, source_files


def test_init_is_idempotent_and_preserves_ignore_file(project: Path) -> None:
    (project / ".gitignore").write_text("existing-entry", encoding="utf-8")
    assert main(["init"]) == 0
    assert main(["init"]) == 0
    assert (project / ".gitignore").read_text() == "existing-entry\n/.mlef/\n"
    assert (project / ".mlef/runs").is_dir()
    result = subprocess.run(
        ["git", "check-ignore", ".mlef/runs/test"], capture_output=True, check=False
    )
    assert result.returncode == 0


def test_arguments_after_separator_are_forwarded_verbatim(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    execute = Mock(return_value=0)
    monkeypatch.setattr("mlef.cli.run", execute)
    assert main(["run", "train.py", "--", "--output", "a b", "$(text)"]) == 0
    execute.assert_called_once_with(
        project, Path("train.py"), ["--output", "a b", "$(text)"], [], []
    )


def test_nested_directory_is_a_usage_error(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (project / "nested").mkdir()
    monkeypatch.chdir(project / "nested")
    assert main(["init"]) == 2


def test_missing_project_is_a_usage_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["init"]) == 2


def test_bad_script_does_not_create_a_run(project: Path) -> None:
    main(["init"])
    assert main(["run", "missing.py"]) == 2
    assert list((project / ".mlef/runs").iterdir()) == []


def test_source_selection_and_path_boundaries(project: Path) -> None:
    (project / "train.py").write_text("print('hello')")
    (project / "helper.py").write_text("value = 1")
    (project / "ignored.py").write_text("secret = 1")
    subprocess.run(["git", "add", "helper.py"], check=True)
    selected = source_files(project, Path("train.py"), [])
    assert set(selected) == {Path("train.py"), Path("helper.py")}
    for path in [Path("../escape"), project / "absolute", Path(".git/config"), Path(".mlef")]:
        with pytest.raises(ValueError, match=r"project-relative|Git internals"):
            relative_path(project, path)
    (project / ".env").write_text("SECRET=never-copy")
    with pytest.raises(ValueError, match="Excluded credential"):
        source_files(project, Path("train.py"), [Path(".env")])


def test_missing_docker_preserves_failed_metadata_and_logs(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main(["init"])
    (project / "train.py").write_text("print('hello')")
    monkeypatch.setattr("mlef.runner.shutil.which", lambda name: None)
    assert run(project, Path("train.py"), [], [], []) == 1
    (directory,) = (project / ".mlef/runs").iterdir()
    metadata = json.loads((directory / "run.json").read_text())
    assert metadata["state"] == "failed"
    assert metadata["execution_exit_code"] is None
    assert metadata["finished_at"] is not None
    assert "Docker executable not found" in metadata["error"]["message"]
    for name in ["stdout.log", "stderr.log", "runner.log"]:
        assert (directory / name).exists()


def test_build_failure_is_recorded(
    project: Path, monkeypatch: pytest.MonkeyPatch, docker_images: None
) -> None:
    main(["init"])
    (project / "train.py").write_text("print('hello')")

    def fail_build(
        arguments: list[str], log: BinaryIO, *, check: bool = True
    ) -> subprocess.CompletedProcess[bytes]:
        if arguments[0] == "build":
            raise RuntimeError("Image build failed")
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr("mlef.runner.shutil.which", lambda name: "docker")
    monkeypatch.setattr("mlef.runner.docker", fail_build)
    assert run(project, Path("train.py"), [], [], []) == 1
    (directory,) = (project / ".mlef/runs").iterdir()
    metadata = json.loads((directory / "run.json").read_text())
    assert metadata["state"] == "failed"
    assert "Image build failed" in (directory / "runner.log").read_text()
    assert list(directory.glob("build-*")) == []
