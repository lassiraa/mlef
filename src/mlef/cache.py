"""Conservative, explicit candidate caching; no cache index or object deduplication."""

import hashlib
import json
import tarfile
from pathlib import Path

from .artifacts import safe_path
from .environment import Environment
from .execution import (
    CacheDecision,
    PipelineRecord,
    StepFingerprint,
    _save,
    attempt_directory,
    imported_directory,
)
from .locking import run_lock
from .pipeline import ExternalInput
from .records import read_json, read_run
from .reuse import copy_step, source_run, verify_source
from .sdk import _checked, read_step


def digest(value: object) -> str:
    content = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    return hashlib.sha256(content).hexdigest()


def source_identity(archive: Path) -> str:
    inventory = []
    with tarfile.open(_checked(archive.absolute()), "r:gz") as stream:
        for member in sorted(stream.getmembers(), key=lambda item: item.name):
            safe_path(member.name.rstrip("/"))
            if not (member.isfile() or member.isdir()):
                raise ValueError("Cache sources require ordinary captured files/directories.")
            content = None
            if member.isfile():
                payload = stream.extractfile(member)
                if payload is None:
                    raise ValueError("Captured source content is missing.")
                with payload:
                    checksum = hashlib.sha256()
                    while chunk := payload.read(1024 * 1024):
                        checksum.update(chunk)
                    content = checksum.hexdigest()
            inventory.append((member.name, member.mode, member.size, content))
    return digest(inventory)


def environment_identity(directory: Path) -> str:
    environment = Environment.model_validate(read_json(_checked(directory / "environment.json")))
    parent = read_run(_checked(directory / "run.json"))
    if (
        environment.container is None
        or environment.base_image is None
        or not environment.base_image.digests
        or environment.uv_image is None
        or not environment.uv_image.digests
        or environment.sdk_wheel is None
        or environment.dependencies is None
        or parent.inputs is None
        or parent.inputs.build_recipe_sha256 is None
    ):
        raise ValueError("Cache decisions require complete captured environment evidence.")
    return digest(
        {
            "container": environment.container.model_dump(mode="json"),
            "base": environment.base_image.model_dump(mode="json"),
            "uv": environment.uv_image.model_dump(mode="json"),
            "dependencies": environment.dependencies.model_dump(mode="json"),
            "recipe": parent.inputs.build_recipe_sha256,
            "controls": parent.controls.requested if parent.controls is not None else None,
        }
    )


def fingerprint(directory: Path, record: PipelineRecord, name: str) -> StepFingerprint:
    entry = next(entry for entry in record.definition.steps if entry.name == name)
    inputs = {}
    for binding, reference in entry.bindings.items():
        if isinstance(reference, ExternalInput):
            artifact = read_step(imported_directory(directory, reference)).artifacts[
                reference.artifact.output
            ]
            if artifact.producer != reference.artifact:
                raise ValueError(
                    "Imported cache input does not match its exact producer reference."
                )
            inputs[binding] = artifact.content_sha256()
            continue
        producer, output = reference.split(".")
        if record.attempts[producer].state != "complete":
            raise ValueError("Cache fingerprints require completed upstream inputs.")
        artifact = read_step(attempt_directory(directory, producer)).artifacts.get(output)
        inputs[binding] = artifact.content_sha256() if artifact is not None else None
    if record.cache_source is None or record.cache_environment is None:
        raise ValueError("Cache fingerprint source/environment evidence is missing.")
    return StepFingerprint(
        source=record.cache_source,
        script=record.attempts[name].script_sha256,
        inputs=digest(inputs),
        parameters=digest(entry.parameters),
        environment=record.cache_environment,
        sdk=record.sdk_wheel.sha256,
        contracts=digest(
            {
                "step": record.contracts[name].model_dump(mode="json"),
                "entry": entry.model_dump(mode="json"),
                "fingerprint_version": 1,
            }
        ),
    )


def _decide(
    directory: Path,
    record: PipelineRecord,
    previous: PipelineRecord | None,
    source: Path | None,
    force: list[str],
) -> None:
    for name in record.order:
        attempt = record.attempts[name]
        if attempt.state == "complete":
            attempt.cache = CacheDecision(
                action="explicit_reuse", reasons=["Explicit reuse frontier."]
            )
            continue
        entry = next(entry for entry in record.definition.steps if entry.name == name)
        reasons = []
        evidence = None
        if name in force:
            reasons.append("Forced execution requested.")
        if not entry.cache:
            reasons.append("Step caching is disabled.")
        if not entry.bindings and entry.cache_version is None:
            reasons.append("Mutable ingestion has no explicit cache_version.")
        if previous is None:
            reasons.append("No cache candidate selected.")
        elif name not in previous.attempts or previous.attempts[name].state != "complete":
            reasons.append("Candidate step is absent or incomplete.")
        elif any(
            record.attempts[reference.split(".")[0]].state != "complete"
            for reference in entry.bindings.values()
            if isinstance(reference, str)
        ):
            reasons.append("Upstream execution is required; inputs are not yet reusable.")
        else:
            evidence = fingerprint(directory, record, name)
            old = previous.attempts[name].cache
            if old is None or old.fingerprint is None:
                reasons.append("Candidate has no verified fingerprint.")
            else:
                reasons.extend(
                    f"Fingerprint changed: {component}."
                    for component in StepFingerprint.model_fields
                    if getattr(evidence, component) != getattr(old.fingerprint, component)
                )
        if reasons:
            attempt.cache = CacheDecision(action="run", reasons=reasons, fingerprint=evidence)
            continue
        assert previous is not None
        assert source is not None
        assert evidence is not None
        copied = copy_step(
            attempt_directory(source / "artifacts/pipeline", name),
            attempt_directory(directory, name),
        )
        attempt.state, attempt.exit_code, attempt.reused_from = "complete", 0, copied.run_id
        attempt.started_at = previous.attempts[name].started_at
        attempt.finished_at = previous.attempts[name].finished_at
        attempt.cache = CacheDecision(
            action="reused", reasons=["All fingerprint components match."], fingerprint=evidence
        )


def configure_cache(
    root: Path, directory: Path, *, candidate: str | None = None, force: list[str] | None = None
) -> PipelineRecord:
    from .execution import read_pipeline

    record = read_pipeline(directory)
    force = force or []
    if record.state != "preparing" or record.cache_environment is not None:
        raise ValueError("Cache decisions must be selected once, before pipeline execution.")
    if set(force) - set(record.order):
        raise ValueError("Forced steps must name known pipeline steps.")
    if (candidate is None and force) or (candidate is not None and record.reuse_from is not None):
        raise ValueError("Force requires a cache candidate; cache and explicit reuse cannot mix.")
    record.cache_environment = environment_identity(directory.parents[1])
    record.cache_from = candidate
    if candidate is not None:
        source = source_run(root, candidate)
        with run_lock(source):
            previous = verify_source(root, candidate)
            _decide(directory, record, previous, source, force)
    else:
        _decide(directory, record, None, None, force)
    _save(directory, record)
    return read_pipeline(directory, verify=True)
