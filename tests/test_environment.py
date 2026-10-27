import io
import json
import subprocess
from pathlib import Path
from typing import BinaryIO

import pytest

from mlef import environment
from mlef.cli import init
from mlef.environment import (
    INSTALL_COMMAND,
    build_recipe,
    dependency_inputs,
    file_hash,
)
from mlef.runner import run, source_files

LOCKED_EXAMPLE = Path(__file__).resolve().parents[1] / "examples/locked"


def test_locked_inputs_and_digest_recipe(project: Path) -> None:
    for name in ["pyproject.toml", "uv.lock", "train.py"]:
        (project / name).write_bytes((LOCKED_EXAMPLE / name).read_bytes())
    inputs = dependency_inputs(project)
    assert inputs.hashes == {
        name: file_hash(project / name) for name in ["pyproject.toml", "uv.lock"]
    }
    assert inputs.install_command == INSTALL_COMMAND
    assert {Path("pyproject.toml"), Path("uv.lock")} <= set(
        source_files(project, Path("train.py"), [])
    )
    recipe = build_recipe("python@sha256:base", "uv@sha256:uv", "1000:1000")
    assert "FROM python@sha256:base" in recipe
    assert "COPY --from=uv@sha256:uv" in recipe
    assert "RUN " + " ".join(INSTALL_COMMAND) in recipe
    assert "--locked" in recipe
    assert "--no-build" in recipe
    assert 'ENV PATH="/opt/mlef-venv/bin:$PATH"' in recipe
    assert "RUN chown 1000:1000 /workspace" in recipe


def test_dependency_free_inputs(project: Path) -> None:
    assert dependency_inputs(project).install_command is None
    assert "uv" not in build_recipe("python@sha256:base", None, None)
    (project / "pyproject.toml").write_text("[tool.ruff]\nline-length = 100\n")
    assert dependency_inputs(project).hashes == {
        "pyproject.toml": file_hash(project / "pyproject.toml")
    }


@pytest.mark.parametrize(
    ("name", "content", "match"),
    [
        ("requirements.txt", "packaging==25.0", "Unsupported dependency"),
        ("uv.toml", "", "Unsupported dependency"),
        ("uv.lock", "version = 1", "requires pyproject"),
        ("pyproject.toml", "[project]\ndependencies = ['packaging']", "existing uv.lock"),
        ("pyproject.toml", "[tool.uv.workspace]\nmembers = []", "single-project"),
        ("pyproject.toml", "[tool.poetry]\nname = 'test'", "single-project"),
        ("pyproject.toml", "project = []", "validation error"),
        ("pyproject.toml", "invalid [", "Expected"),
    ],
)
def test_invalid_inputs_fail_before_docker(
    project: Path, monkeypatch: pytest.MonkeyPatch, name: str, content: str, match: str
) -> None:
    init(project)
    (project / "train.py").write_text("print('never executed')")
    (project / name).write_text(content)
    monkeypatch.setattr("mlef.runner.shutil.which", lambda name: None)
    assert run(project, Path("train.py"), [], [], []) == 1
    (directory,) = (project / ".mlef/runs").iterdir()
    metadata = json.loads((directory / "run.json").read_text())
    assert metadata["state"] == "failed"
    assert metadata["execution_exit_code"] is None
    assert metadata["error"]["category"] == "preparation"
    assert match in metadata["error"]["message"]
    assert (directory / "source.tar.gz").exists()
    captured = json.loads((directory / "environment.json").read_text())
    assert captured["container"] is None
    assert captured["warnings"]


@pytest.mark.parametrize("source", ["{path = '../other'}", "{git = 'https://example.org/repo'}"])
def test_non_registry_locks_are_rejected(project: Path, source: str) -> None:
    (project / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    (project / "uv.lock").write_text(f"version = 1\n[[package]]\nsource = {source}\n")
    with pytest.raises(ValueError, match="Only registry"):
        dependency_inputs(project)


def test_environment_failure_prevents_execution(
    project: Path, monkeypatch: pytest.MonkeyPatch, docker_images: None
) -> None:
    init(project)
    (project / "train.py").write_text("print('never executed')")
    monkeypatch.setattr("mlef.runner.shutil.which", lambda name: "docker")

    def docker(
        arguments: list[str], log: BinaryIO, *, check: bool = True
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(arguments, 0)

    def fail_probe(image: str, log: BinaryIO, user: str | None) -> None:
        raise RuntimeError("environment probe failed")

    monkeypatch.setattr("mlef.runner.docker", docker)
    monkeypatch.setattr("mlef.runner.container_info", fail_probe)
    assert run(project, Path("train.py"), [], [], []) == 1
    (directory,) = (project / ".mlef/runs").iterdir()
    metadata = json.loads((directory / "run.json").read_text())
    assert metadata["state"] == "failed"
    assert metadata["started_at"] is None
    assert metadata["inputs"]["build_recipe_sha256"] == file_hash(directory / "Dockerfile")
    captured = json.loads((directory / "environment.json").read_text())
    assert captured["execution_image"]["id"] == "sha256:test"
    assert captured["container"] is None
    assert captured["warnings"]


def test_image_and_container_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    def query(arguments: list[str], log: BinaryIO) -> str:
        if arguments[0] == "image":
            return json.dumps(
                [
                    {
                        "Id": "sha256:test",
                        "RepoDigests": ["python@sha256:test"],
                        "Os": "linux",
                        "Architecture": "amd64",
                    }
                ]
            )
        assert arguments[:4] == ["run", "--rm", "--network", "none"]
        assert "-I" in arguments
        return json.dumps(
            {
                "os": "Linux",
                "os_release": {"ID": "debian"},
                "architecture": "x86_64",
                "python": "3.12.0",
                "packages": {"packaging": "25.0"},
                "uv_version": "uv 0.13.6",
            }
        )

    monkeypatch.setattr(environment, "query", query)
    log = io.BytesIO()
    assert environment.image_identity("python", log).id == "sha256:test"
    assert environment.container_info("sha256:test", log, "1000:1000").packages == {
        "packaging": "25.0"
    }
