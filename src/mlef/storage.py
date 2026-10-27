"""Offline run queries and conservative artifact retrieval."""

import shutil
from datetime import datetime
from pathlib import Path

from .records import RunRecord, read_run, run_id


class QueryError(ValueError):
    def __init__(self, category: str, message: str) -> None:
        self.category = category
        super().__init__(message)


def directory(root: Path, identifier: str) -> Path:
    try:
        run_id(identifier)
    except ValueError as error:
        raise QueryError("invalid_run_id", str(error)) from error
    runs = root / ".mlef/runs"
    path = runs / identifier
    if path.is_symlink() or not path.resolve().is_relative_to(runs.resolve()):
        raise QueryError("unsafe_path", "Run directory escapes run storage.")
    if not path.is_dir():
        raise QueryError("unknown_run", f"Unknown run: {identifier}")
    return path


def inspect(root: Path, identifier: str) -> RunRecord:
    path = directory(root, identifier)
    try:
        if (path / "run.json").is_symlink():
            raise ValueError("Run metadata must not be a symbolic link.")
        record = read_run(path / "run.json")
        if record.id != identifier:
            raise ValueError("Run ID does not match its storage directory.")
        if record.state in {"preparing", "running"}:
            record.warnings.append("Last recorded state only; current liveness is unknown.")
        return record
    except (OSError, ValueError) as error:
        raise QueryError("invalid_record", f"Cannot read run {identifier}: {error}") from error


def list_runs(root: Path) -> tuple[list[RunRecord], list[str]]:
    runs: list[RunRecord] = []
    diagnostics: list[str] = []
    storage = root / ".mlef/runs"
    if not storage.is_dir():
        raise QueryError("not_initialized", "Project is not initialized; run mlef init first.")
    for path in storage.iterdir():
        try:
            runs.append(inspect(root, path.name))
        except QueryError as error:
            diagnostics.append(str(error))
    runs.sort(key=lambda record: (datetime.fromisoformat(record.created_at), record.id), reverse=True)
    return runs, diagnostics


def artifact_path(root: Path, identifier: str, relative: str) -> Path:
    base = directory(root, identifier) / "artifacts"
    path = Path(relative)
    if (
        not relative
        or path.is_absolute()
        or ".." in path.parts
        or ":" in relative
        or "\\" in relative
        or base.is_symlink()
    ):
        raise QueryError("unsafe_path", "Expected an artifact-relative path without links or '..'.")
    target = base / path
    if not target.resolve().is_relative_to(base.resolve()) or any(
        item.is_symlink() for item in [target, *target.parents] if item.is_relative_to(base)
    ):
        raise QueryError("unsafe_path", "Artifact path escapes storage or contains a link.")
    if not target.exists():
        raise QueryError("unknown_artifact", f"Unknown artifact: {relative}")
    return target


def artifacts(root: Path, identifier: str) -> list[str]:
    base = directory(root, identifier) / "artifacts"
    if not base.exists():
        return []
    if base.is_symlink():
        raise QueryError("unsafe_path", "Artifact storage must not be a symbolic link.")
    return sorted(
        item.relative_to(base).as_posix()
        for item in base.rglob("*")
        if item.is_file()
        and not item.is_symlink()
        and item.resolve().is_relative_to(base.resolve())
    )


def retrieve(root: Path, identifier: str, relative: str, destination: Path | None) -> Path:
    source = artifact_path(root, identifier, relative)
    destination = destination if destination is not None else Path(source.name)
    if destination.exists() or destination.is_symlink():
        raise QueryError("destination_exists", f"Destination already exists: {destination}")
    if not destination.parent.is_dir():
        raise QueryError("invalid_destination", "Destination parent directory must exist.")
    if source.is_dir() and destination.resolve().is_relative_to(source.resolve()):
        raise QueryError(
            "invalid_destination", "Destination cannot be inside the source directory."
        )
    if source.is_dir():
        for item in source.rglob("*"):
            if item.is_symlink() or not (item.is_file() or item.is_dir()):
                raise QueryError(
                    "unsafe_path", "Artifact directory contains a link or special file."
                )
        shutil.copytree(source, destination)
    elif source.is_file():
        with source.open("rb") as stream, destination.open("xb") as output:
            shutil.copyfileobj(stream, output)
    else:
        raise QueryError("unsafe_path", "Artifact is not a regular file or directory.")
    return destination
