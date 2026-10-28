import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path
from typing import BinaryIO

import pytest

from mlef.cli import init
from mlef.provenance import capture, extract_source, git
from mlef.runner import run, source_files


def commit(project: Path) -> None:
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "base",
        ],
        cwd=project,
        check=True,
    )


def test_snapshot_unborn_and_explicit_untracked_inputs(project: Path, tmp_path: Path) -> None:
    (project / "train.py").write_bytes(b"print('original')\n")
    (project / "helper.py").write_bytes(b"value = 1\n")
    git(project, "add", "helper.py")
    saved = tmp_path / "saved"
    saved.mkdir()
    provenance, inputs = capture(project, saved, source_files(project, Path("train.py"), []))
    assert provenance.commit is None
    assert provenance.branch is not None
    assert provenance.dirty
    assert inputs.included_untracked == ["train.py"]
    assert (
        inputs.source_sha256 == hashlib.sha256((saved / "source.tar.gz").read_bytes()).hexdigest()
    )
    assert b"helper.py" in (saved / "changes.patch").read_bytes()
    (project / "train.py").write_text("changed")
    extract_source(saved / "source.tar.gz", saved / "extracted")
    assert (saved / "extracted/train.py").read_bytes() == b"print('original')\n"
    assert not (saved / "extracted/.git").exists()
    assert (saved / "source.tar.gz").read_bytes()[4:8] == bytes(4)
    _, repeated = capture(project, saved, source_files(project, Path("train.py"), []))
    assert repeated.source_sha256 != inputs.source_sha256  # The file was changed after capture.
    _, unchanged = capture(project, saved, source_files(project, Path("train.py"), []))
    assert unchanged.source_sha256 == repeated.source_sha256


def test_git_dirty_binary_deletion_and_detached_head(project: Path, tmp_path: Path) -> None:
    (project / "train.py").write_bytes(b"old\x00binary")
    (project / "removed.py").write_text("removed")
    git(project, "add", ".")
    commit(project)
    git(project, "checkout", "--detach", "--quiet")
    (project / "train.py").write_bytes(b"staged\x00binary")
    git(project, "add", "train.py")
    (project / "train.py").write_bytes(b"working\x00binary")
    (project / "removed.py").unlink()
    saved = tmp_path / "saved"
    saved.mkdir()
    provenance, _ = capture(project, saved, source_files(project, Path("train.py"), []))
    assert provenance.commit == git(project, "rev-parse", "HEAD").decode().strip()
    assert provenance.branch is None
    assert provenance.dirty
    patch = (saved / "changes.patch").read_bytes()
    assert b"GIT binary patch" in patch
    assert b"deleted file" in patch
    git(project, "reset", "--hard", "HEAD")
    git(project, "apply", str(saved / "changes.patch"))
    assert (project / "train.py").read_bytes() == b"working\x00binary"
    assert not (project / "removed.py").exists()


def test_clean_repository_and_explicit_inclusion(project: Path, tmp_path: Path) -> None:
    (project / "train.py").write_text("print('hello')")
    git(project, "add", "train.py")
    commit(project)
    saved = tmp_path / "saved"
    saved.mkdir()
    provenance, inputs = capture(project, saved, source_files(project, Path("train.py"), []))
    assert not provenance.dirty
    assert inputs.included_untracked == []
    assert (saved / "changes.patch").read_bytes() == b""
    (project / "extra.txt").write_text("included")
    provenance, inputs = capture(
        project, saved, source_files(project, Path("train.py"), [Path("extra.txt")])
    )
    assert provenance.dirty
    assert inputs.included_untracked == ["extra.txt"]
    extract_source(saved / "source.tar.gz", saved / "extracted")
    assert (saved / "extracted/extra.txt").read_text() == "included"


def test_capture_failure_preserves_failed_run(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init(project)
    (project / "train.py").write_text("original")

    def fail_capture(root: Path, run_dir: Path, files: list[Path]) -> tuple[object, object]:
        raise OSError("source capture failed")

    monkeypatch.setattr("mlef.runner.capture", fail_capture)
    monkeypatch.setattr("mlef.runner.shutil.which", lambda name: None)
    assert run(project, Path("train.py"), [], [], []) == 1
    (directory,) = (project / ".mlef/runs").iterdir()
    metadata = json.loads((directory / "run.json").read_text())
    assert metadata["state"] == "failed"
    assert metadata["git"] is None
    assert metadata["inputs"] is None
    assert "source capture failed" in metadata["error"]["message"]


@pytest.mark.parametrize("name", [".aws/credentials", ".env.local", ".pytest_cache/state"])
def test_excluded_source_is_never_archived(project: Path, name: str) -> None:
    (project / "train.py").write_text("print('hello')")
    excluded = project / name
    excluded.parent.mkdir(parents=True, exist_ok=True)
    excluded.write_text("excluded")
    with pytest.raises(ValueError, match="Excluded credential or generated path"):
        source_files(project, Path("train.py"), [Path(name)])


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/escape", "..\\escape"])
def test_unsafe_archive_paths_are_rejected(tmp_path: Path, name: str) -> None:
    archive_path = tmp_path / "source.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo(name)
        member.size = 4
        archive.addfile(member, io.BytesIO(b"data"))
    with pytest.raises(ValueError, match="Unsafe source archive"):
        extract_source(archive_path, tmp_path / "extracted")
    assert not (tmp_path / "extracted").exists()


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE])
def test_archive_links_and_special_files_are_rejected(tmp_path: Path, kind: bytes) -> None:
    archive_path = tmp_path / "source.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo("link")
        member.type = kind
        member.linkname = "../escape"
        archive.addfile(member)
    with pytest.raises(ValueError, match="Unsafe source archive"):
        extract_source(archive_path, tmp_path / "extracted")


def test_build_uses_saved_source_after_worktree_edit(
    project: Path, monkeypatch: pytest.MonkeyPatch, docker_images: None
) -> None:
    init(project)
    (project / "train.py").write_text("original")

    def docker(
        arguments: list[str], log: BinaryIO, *, check: bool = True
    ) -> subprocess.CompletedProcess[bytes]:
        if arguments[0] == "info":
            (project / "train.py").write_text("changed after capture")
        if arguments[0] == "build":
            assert (Path(arguments[-1]) / "workspace/train.py").read_text() == "original"
            raise RuntimeError("stop after checking build context")
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr("mlef.runner.shutil.which", lambda name: "docker")
    monkeypatch.setattr("mlef.runner.docker", docker)
    assert run(project, Path("train.py"), [], [], []) == 1
    (directory,) = (project / ".mlef/runs").iterdir()
    assert (directory / "source.tar.gz").exists()
    assert (directory / "Dockerfile").exists()
