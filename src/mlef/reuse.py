"""Explicit, verified copies of completed steps; never an implicit cache."""

import shutil
from pathlib import Path

from . import storage
from .environment import file_hash
from .execution import PipelineRecord, _save, attempt_directory, read_pipeline
from .locking import run_lock
from .sdk import _checked, read_step


def source_run(root: Path, identifier: str) -> Path:
    parent = storage.inspect(root, identifier)
    if parent.state not in {"succeeded", "failed", "cancelled"}:
        raise ValueError("Reuse requires a terminal source run; recover abandoned runs first.")
    return _checked(storage.directory(root, identifier))


def verify_source(root: Path, identifier: str) -> PipelineRecord:
    from .api import MlefError, pipeline_status

    source_run(root, identifier)
    try:
        return pipeline_status(identifier, root=root, verify=True)
    except MlefError as error:
        raise ValueError(str(error)) from error


def copy_snapshot(root: Path, identifier: str, destination: Path) -> PipelineRecord:
    source = source_run(root, identifier)
    with run_lock(source):
        record = verify_source(root, identifier)
        if not record.sdk_wheel.pipeline_reuse:
            raise ValueError("The captured SDK does not support pipeline reuse; fork instead.")
        for filename in ("source.tar.gz", "changes.patch"):
            shutil.copyfile(_checked(source / filename), destination / filename)
        target = destination / "sdk"
        target.mkdir()
        shutil.copyfile(
            _checked(source / "sdk" / record.sdk_wheel.filename),
            target / record.sdk_wheel.filename,
        )
        if (
            file_hash(destination / "source.tar.gz") != record.source_sha256
            or file_hash(target / record.sdk_wheel.filename) != record.sdk_wheel.sha256
        ):
            raise ValueError("Copied snapshot bytes changed.")
        return record


def reuse_steps(
    root: Path, identifier: str, directory: Path, from_step: str, *, resume: bool = False
) -> PipelineRecord:
    source = source_run(root, identifier)
    with run_lock(source):
        previous = verify_source(root, identifier)
        record = read_pipeline(directory)
        if record.state != "preparing" or record.reuse_from is not None:
            raise ValueError("Reuse must be selected before pipeline execution, exactly once.")
        if resume and (
            record.source_sha256 != previous.source_sha256 or record.sdk_wheel != previous.sdk_wheel
        ):
            raise ValueError("Resume must retain the original source and SDK snapshot.")
        entries = {entry.name: entry for entry in record.definition.steps}
        old_entries = {entry.name: entry for entry in previous.definition.steps}
        execute = record.execution_steps(from_step)
        selected = [name for name in record.order if name not in execute]
        for name in selected:
            if (
                name not in previous.attempts
                or previous.attempts[name].state != "complete"
                or record.contracts[name] != previous.contracts[name]
                or entries[name].bindings != old_entries[name].bindings
                or record.attempts[name].script_sha256 != previous.attempts[name].script_sha256
            ):
                raise ValueError(f"Cannot reuse incomplete or incompatible step: {name}")
        record = record.model_copy(
            update={"reuse_from": identifier, "from_step": from_step, "resume": resume}
        )
        for name in selected:
            origin = attempt_directory(source / "artifacts/pipeline", name)
            target = attempt_directory(directory, name)
            result = read_step(origin)
            target.mkdir(parents=True, exist_ok=False)
            shutil.copyfile(_checked(origin / "step.json"), target / "step.json")
            if (origin / "outputs").exists():
                shutil.copytree(origin / "outputs", target / "outputs", symlinks=True)
            copied = read_step(target)
            if copied != result:
                raise ValueError(f"Copied artifact evidence changed: {name}")
            attempt = record.attempts[name]
            attempt.state, attempt.exit_code = "complete", 0
            attempt.reused_from = copied.run_id
            attempt.started_at = previous.attempts[name].started_at
            attempt.finished_at = previous.attempts[name].finished_at
        _save(directory, record)
        return read_pipeline(directory, verify=True)
