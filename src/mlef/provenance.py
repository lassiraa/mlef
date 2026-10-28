"""Saved source inputs; the archive, rather than the Git patch, defines execution."""

import hashlib
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, Field


class GitProvenance(BaseModel):
    commit: str | None
    branch: str | None
    dirty: bool


class SourceInputs(BaseModel):
    source_sha256: str
    included_untracked: list[str]
    dependency_hashes: dict[str, str] = Field(default_factory=dict)
    build_recipe_sha256: str | None = None
    outputs: list[str] = Field(default_factory=list)


def git(root: Path, *arguments: str) -> bytes:
    return subprocess.run(["git", *arguments], cwd=root, capture_output=True, check=True).stdout


def capture(root: Path, run_dir: Path, files: list[Path]) -> tuple[GitProvenance, SourceInputs]:
    revision = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"], cwd=root, capture_output=True, check=False
    )
    commit = revision.stdout.decode().strip() if revision.returncode == 0 else None
    symbolic = subprocess.run(
        ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if symbolic.returncode not in (0, 1):
        symbolic.check_returncode()
    branch = symbolic.stdout.decode().strip() if symbolic.returncode == 0 else None
    baseline = commit
    if baseline is None:
        baseline = (
            subprocess.run(
                ["git", "hash-object", "-t", "tree", "--stdin"],
                cwd=root,
                input=b"",
                capture_output=True,
                check=True,
            )
            .stdout.decode()
            .strip()
        )
    patch = git(root, "diff", "--binary", "--no-ext-diff", "--no-textconv", baseline, "--")
    (run_dir / "changes.patch").write_bytes(patch)
    tracked = set(git(root, "ls-files", "--cached", "-z").decode("utf-8").split("\0"))
    archive_path = run_dir / "source.tar.gz"
    with tarfile.open(archive_path, "w:gz", dereference=False) as archive:
        for path in files:
            archive.add(root / path, arcname=path.as_posix(), recursive=False)
    with archive_path.open("rb") as source:
        source_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    return (
        GitProvenance(
            commit=commit,
            branch=branch,
            dirty=bool(git(root, "status", "--porcelain", "--untracked-files=normal")),
        ),
        SourceInputs(
            source_sha256=source_sha256,
            included_untracked=[
                path.as_posix() for path in files if path.as_posix() not in tracked
            ],
        ),
    )


def extract_source(archive_path: Path, destination: Path) -> None:
    """Validate the whole archive before extracting; links and special files are forbidden."""
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            target = destination / member.name
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in member.name
                or ":" in member.name
                or not (member.isfile() or member.isdir())
                or not target.resolve().is_relative_to(destination.resolve())
                or target.is_symlink()
            ):
                raise ValueError(f"Unsafe source archive member: {member.name}")
        for member in members:
            target = destination / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"Missing archive file content: {member.name}")
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(member.mode & 0o777)
