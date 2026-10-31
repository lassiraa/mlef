"""Explicit, self-contained copies of exact cross-run artifact inputs."""

from pathlib import Path

from .execution import PipelineRecord, attempt_directory, imported_directory, read_pipeline
from .locking import run_lock
from .pipeline import ExternalInput, Input
from .reuse import copy_step, source_run, verify_source
from .sdk import ArtifactSource, read_step


def source(directory: Path, reference: ExternalInput, target: Input) -> ArtifactSource:
    origin = imported_directory(directory, reference)
    manifest = read_step(origin).artifacts.get(reference.artifact.output)
    if manifest is None or manifest.producer != reference.artifact:
        raise ValueError("Imported input does not match its exact producer reference.")
    if manifest.type != target.type or (
        target.format is not None and manifest.format != target.format
    ):
        raise ValueError("Imported input is incompatible with its declared type/format.")
    return ArtifactSource(directory=origin, name=reference.artifact.output)


def verify_imports(directory: Path, record: PipelineRecord) -> None:
    for entry in record.definition.steps:
        inputs = {item.name: item for item in record.contracts[entry.name].inputs}
        for binding, reference in entry.bindings.items():
            if isinstance(reference, ExternalInput):
                source(directory, reference, inputs[binding])


def stage(root: Path, directory: Path, *, resume_from: str | None = None) -> None:
    record = read_pipeline(directory)
    seen = set()
    for entry in record.definition.steps:
        for reference in entry.bindings.values():
            if not isinstance(reference, ExternalInput):
                continue
            target = imported_directory(directory, reference)
            if target in seen:
                continue
            seen.add(target)
            identifier = resume_from if resume_from is not None else reference.run_id
            run_directory = source_run(root, identifier)
            with run_lock(run_directory):
                previous = verify_source(root, identifier)
                origin = run_directory / "artifacts/pipeline"
                if resume_from is not None:
                    origin = imported_directory(origin, reference)
                else:
                    if (
                        reference.artifact.step not in previous.attempts
                        or previous.attempts[reference.artifact.step].state != "complete"
                    ):
                        raise ValueError("External inputs require a completed producer step.")
                    origin = attempt_directory(origin, reference.artifact.step)
                manifest = read_step(origin).artifacts.get(reference.artifact.output)
                if manifest is None or manifest.producer != reference.artifact:
                    raise ValueError("External input does not match its exact producer reference.")
                copy_step(origin, target)
    verify_imports(directory, record)
