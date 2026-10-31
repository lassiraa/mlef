"""Editable project-local run organization, separate from captured evidence."""

import hashlib
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Literal, Self
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from .locking import file_lock
from .records import Contract, read_json, run_id
from .sdk import _checked
from .storage import QueryError, inspect

Tag = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$", max_length=64)]
TagValue = Annotated[str, Field(max_length=1000)]


def _display_name(value: str | None) -> str | None:
    if value is not None and (
        not value.strip()
        or value != value.strip()
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
    ):
        raise ValueError("Names must be nonempty, trimmed and free of control characters.")
    return value


class RunLabels(Contract):
    name: str | None = Field(default=None, max_length=200)
    experiment: str | None = Field(default=None, max_length=200)
    tags: dict[Tag, TagValue] = Field(default_factory=dict, max_length=50)
    note: str | None = Field(default=None, max_length=20_000)

    _names = field_validator("name", "experiment")(_display_name)


class RunAnnotations(RunLabels):
    schema_version: Literal[1] = 1
    run_id: str
    updated_at: str | None = None
    experiment_id: str | None = None
    archived: bool = False

    _run_id = field_validator("run_id")(run_id)
    _experiment_id = field_validator("experiment_id")(
        lambda value: run_id(value) if value is not None else None
    )

    @model_validator(mode="after")
    def group_identity(self) -> Self:
        if (self.experiment is None) != (self.experiment_id is None):
            raise ValueError("Experiment annotations require both group name and identity.")
        return self


class ExperimentGroup(Contract):
    schema_version: Literal[1] = 1
    id: str
    name: str = Field(max_length=200)
    created_at: str
    description: str | None = Field(default=None, max_length=20_000)

    _id = field_validator("id")(run_id)
    _name = field_validator("name")(_display_name)


class ExperimentListing(Contract):
    schema_version: Literal[1] = 1
    groups: list[ExperimentGroup] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)


def _storage(root: Path) -> Path:
    if not _checked(root / ".mlef/runs").is_dir():
        raise QueryError("not_initialized", "Project is not initialized; run mlef init first.")
    return _checked(root / ".mlef")


@contextmanager
def _lock(root: Path) -> Generator[None, None, None]:
    # ponytail: one lock for human metadata edits; split per record if contention matters.
    try:
        with file_lock(_checked(_storage(root) / "metadata.lock")):
            yield
    except QueryError as error:
        if error.category == "run_active":
            raise QueryError(
                "metadata_busy", "Another project metadata update is active."
            ) from error
        raise


def _write(path: Path, record: Contract) -> None:
    from .runner import save

    _checked(path).parent.mkdir(parents=True, exist_ok=True)
    _checked(path.with_suffix(".tmp"))
    save(path, record)


def _group_path(root: Path, name: str) -> Path:
    _display_name(name)
    if len(name) > 200:
        raise ValueError("Experiment names cannot exceed 200 characters.")
    identity = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return _checked(_storage(root) / "experiments" / f"{identity}.json")


def group(root: Path, name: str) -> ExperimentGroup:
    path = _group_path(root, name)
    if not path.exists():
        raise QueryError("unknown_experiment", f"Unknown experiment group: {name}")
    try:
        record = ExperimentGroup.model_validate(read_json(path))
        if record.name != name:
            raise ValueError("Experiment name does not match its catalog location.")
        return record
    except (OSError, ValueError) as error:
        raise QueryError("invalid_record", f"Cannot read experiment group: {error}") from error


def _create_group(root: Path, name: str, description: str | None = None) -> ExperimentGroup:
    from .runner import now

    path = _group_path(root, name)
    if path.exists():
        return group(root, name)
    record = ExperimentGroup(id=str(uuid4()), name=name, created_at=now(), description=description)
    _write(path, record)
    return record


def create_group(root: Path, name: str, description: str | None = None) -> ExperimentGroup:
    # Validate before allocating storage, including an idempotent creation request.
    ExperimentGroup(id=str(uuid4()), name=name, created_at="", description=description)
    with _lock(root):
        existing = _create_group(root, name, description)
        if description is not None and existing.description != description:
            existing = existing.model_copy(update={"description": description})
            _write(_group_path(root, name), existing)
        return existing


def groups(root: Path) -> ExperimentListing:
    result = ExperimentListing()
    directory = _checked(_storage(root) / "experiments")
    if directory.exists():
        for path in sorted(directory.glob("*.json")):
            try:
                record = ExperimentGroup.model_validate(read_json(_checked(path)))
                if _group_path(root, record.name) != path:
                    raise ValueError("Experiment name does not match its catalog location.")
                result.groups.append(record)
            except (OSError, ValueError) as error:
                result.diagnostics.append(f"{path.name}: {error}")
    result.groups.sort(key=lambda item: (item.name, item.id))
    return result


def read(root: Path, identifier: str) -> RunAnnotations:
    run_id(identifier)
    path = _checked(_storage(root) / "annotations" / f"{identifier}.json")
    if not path.exists():
        return RunAnnotations(run_id=identifier)
    try:
        record = RunAnnotations.model_validate(read_json(path))
        if record.run_id != identifier:
            raise ValueError("Annotation run identity does not match its catalog location.")
        return record
    except (OSError, ValueError) as error:
        raise QueryError("invalid_record", f"Cannot read run annotations: {error}") from error


def update(
    root: Path,
    identifier: str,
    *,
    name: str | None = None,
    experiment: str | None = None,
    note: str | None = None,
    tags: dict[str, str] | None = None,
    remove_tags: list[str] | None = None,
    archived: bool | None = None,
    clear_name: bool = False,
    clear_note: bool = False,
    clear_experiment: bool = False,
) -> RunAnnotations:
    from .runner import now

    inspect(root, identifier)
    RunLabels(name=name, experiment=experiment, tags=tags if tags is not None else {}, note=note)
    if (
        (clear_name and name is not None)
        or (clear_note and note is not None)
        or (clear_experiment and experiment is not None)
    ):
        raise ValueError("A field cannot be set and cleared in one annotation update.")
    if tags is not None and set(tags) & set(remove_tags or []):
        raise ValueError("A tag cannot be set and removed in one annotation update.")
    with _lock(root):
        values = read(root, identifier).model_dump()
        for key, value, clear in [
            ("name", name, clear_name),
            ("note", note, clear_note),
            ("experiment", experiment, clear_experiment),
        ]:
            if value is not None or clear:
                values[key] = value
        merged = dict(values["tags"])
        merged.update(tags or {})
        for key in remove_tags or []:
            merged.pop(key, None)
        values["tags"] = merged
        labels = RunLabels.model_validate({key: values[key] for key in RunLabels.model_fields})
        if experiment is not None:
            values["experiment_id"] = _create_group(root, labels.experiment or "").id
        elif clear_experiment:
            values["experiment_id"] = None
        if archived is not None:
            values["archived"] = archived
        values["updated_at"] = now()
        record = RunAnnotations.model_validate(values)
        _write(_checked(_storage(root) / "annotations" / f"{identifier}.json"), record)
        return record
