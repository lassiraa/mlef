import csv
import io
import json
import os
import subprocess
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import pytest

from mlef import Artifact, Metrics, Step, api, recovery, sdk_distribution, supervision
from mlef.cache import configure_cache, fingerprint
from mlef.cli import main
from mlef.environment import (
    ContainerInfo,
    Environment,
    ImageIdentity,
    build_recipe,
    dependency_inputs,
    file_hash,
)
from mlef.events import Delivery
from mlef.execution import (
    _save,
    attempt_directory,
    execute_pipeline,
    imported_directory,
    prepare_pipeline,
    read_pipeline,
)
from mlef.external_inputs import stage as stage_external_inputs
from mlef.locking import file_lock, run_lock
from mlef.pipeline import ExternalInput, validate
from mlef.project import initialize
from mlef.provenance import capture, extract_source
from mlef.records import RunError, RunRecord, read_run
from mlef.reuse import copy_snapshot, reuse_steps, verify_source
from mlef.runner import now, run, save, source_files
from mlef.sdk import LocalContext, read_step
from mlef.sdk_distribution import capture_sdk
from mlef.snapshots import restore
from mlef.transfers import LocalStore, fetch, publish

LOCKED = Path(__file__).resolve().parents[1] / "examples/pipeline"


def cache_environment(project: Path, directory: Path, *, python: str = "3.12.0") -> None:
    record = read_pipeline(directory)
    run_dir = directory.parents[1]
    parent = read_run(run_dir / "run.json")
    assert parent.inputs is not None
    parent.inputs.build_recipe_sha256 = "a" * 64
    save(run_dir / "run.json", parent)
    image = ImageIdentity(
        id="sha256:test", digests=["test@sha256:immutable"], os="linux", architecture="amd64"
    )
    save(
        run_dir / "environment.json",
        Environment(
            container=ContainerInfo(
                os="linux",
                architecture="amd64",
                python=python,
                os_release={"ID": "debian"},
                packages={"mlef": record.sdk_wheel.version, "pydantic": "2"},
                uv_version="0.13.6",
            ),
            base_image=image,
            uv_image=image,
            execution_image=image,
            dependencies=dependency_inputs(project),
            sdk_wheel=record.sdk_wheel,
            warnings=[],
        ).model_dump(),
    )


@pytest.fixture
def cached_pipeline(project: Path) -> Path:
    setup(project)
    definition = project / "mlef.toml"
    definition.write_text(
        definition.read_text().replace(
            "script = ", 'cache = true\ncache_version = "public-v1"\nscript = '
        )
    )
    script = project / "steps/train.py"
    script.write_text(
        script.read_text().replace(
            '"prediction": mean(values)',
            '"prediction": mean(values) + context.parameters.get("offset", 0)',
        )
    )
    directory = prepare(project)
    cache_environment(project, directory)
    configure_cache(project, directory)
    assert execute_pipeline(project, directory) == 0
    finish_parent(directory)
    return directory


def test_cache_hits_are_self_contained_and_explained(
    project: Path,
    cached_pipeline: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    previous = read_pipeline(cached_pipeline)
    script = project / "steps/ingest.py"
    timestamp = script.stat().st_mtime + 10
    os.utime(script, (timestamp, timestamp))
    directory = prepare(project)
    cache_environment(project, directory)
    record = configure_cache(project, directory, candidate=previous.run_id)
    assert record.source_sha256 != previous.source_sha256
    assert record.cache_source == previous.cache_source
    assert all(attempt.reused_from == previous.run_id for attempt in record.attempts.values())
    assert all(
        attempt.cache is not None and attempt.cache.action == "reused"
        for attempt in record.attempts.values()
    )
    cached_pipeline.parents[1].rename(cached_pipeline.parents[1].with_name("removed-cache-source"))
    with monkeypatch.context() as context:
        context.setattr(
            "mlef.execution.subprocess.run", lambda *args, **kwargs: pytest.fail("reran")
        )
        assert execute_pipeline(project, directory) == 0
    finish_parent(directory)
    assert api.pipeline_status(record.run_id, root=project, verify=True).state == "complete"
    capsys.readouterr()
    assert main(["cache", record.run_id, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["cache_from"] == previous.run_id


@pytest.mark.parametrize("reuse", [False, True])
def test_module_entrypoint_uses_canonical_cache_models(
    project: Path, cached_pipeline: Path, reuse: bool
) -> None:
    previous = read_pipeline(cached_pipeline)
    directory = prepare(project)
    cache_environment(project, directory)
    record = configure_cache(
        project,
        directory,
        candidate=previous.run_id if reuse else None,
        force=["train"] if reuse else None,
    )
    process = subprocess.run(
        [sys.executable, "-I", "-m", "mlef.execution", str(directory)],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    finish_parent(directory)
    assert api.pipeline_status(record.run_id, root=project, verify=True).state == "complete"


def test_cached_external_metrics_remain_resolvable_without_original_run(project: Path) -> None:
    from mlef import ArtifactSource, Metrics, resolve_metric
    from mlef.metrics import ExternalMetric, Series

    setup(project)
    (project / "mlef.toml").write_text(
        '[[steps]]\nname="history"\nscript="steps/history.py"\ncache=true\ncache_version="v1"\n'
        '[[steps]]\nname="report"\nscript="steps/report.py"\ncache=true\n'
        'bindings={history="history.series"}\n'
    )
    (project / "steps/history.py").write_text(
        "from mlef import Artifact, ArtifactType, Output, Step\n"
        "from mlef.metrics import Series\n"
        'step = Step(name="history", outputs=[Output(name="series", '
        'type=ArtifactType.FILE, format="application/json")])\n'
        "with step.context() as context:\n"
        '    path = context.output_path("series", "series.json")\n'
        '    path.write_text(Series(definition="loss", index="epoch", '
        "x=[1, 2], y=[0.5, 0.2]).model_dump_json())\n"
        '    step.save([Artifact(name="series", type=ArtifactType.FILE, '
        'format="application/json", path=path)], context=context)\n'
    )
    (project / "steps/report.py").write_text(
        "from mlef import Artifact, ArtifactType, ExternalMetric, Input, Metrics, Output, Step\n"
        'step = Step(name="report", inputs=[Input(name="history", type=ArtifactType.FILE)], '
        'outputs=[Output(name="metrics", type=ArtifactType.METRICS, format="application/json")])\n'
        "with step.context() as context:\n"
        '    value = ExternalMetric(definition="loss", representation="series", '
        'artifact=context.input_reference("history"), path="series.json")\n'
        '    path = context.output_path("metrics", "metrics.json")\n'
        '    path.write_text(Metrics(scope="validation", '
        'values={"loss": value}).model_dump_json())\n'
        '    step.save([Artifact(name="metrics", type=ArtifactType.METRICS, '
        'format="application/json", path=path)], context=context)\n'
    )
    original = prepare(project)
    cache_environment(project, original)
    configure_cache(project, original)
    process = subprocess.run(
        [sys.executable, "-I", "-m", "mlef.execution", str(original)],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    finish_parent(original)
    directory = prepare(project)
    cache_environment(project, directory)
    record = configure_cache(project, directory, candidate=read_pipeline(original).run_id)
    assert all(attempt.reused_from is not None for attempt in record.attempts.values())
    original.parents[1].rename(original.parents[1].with_name("removed-metric-source"))
    assert execute_pipeline(project, directory) == 0
    finish_parent(directory)
    api.pipeline_status(record.run_id, root=project, verify=True)
    path = attempt_directory(directory, "report") / "outputs/metrics/metrics.json"
    external = Metrics.model_validate_json(path.read_text()).values["loss"]
    assert isinstance(external, ExternalMetric)
    resolved = resolve_metric(
        external, ArtifactSource(directory=attempt_directory(directory, "history"), name="series")
    )
    assert isinstance(resolved, Series)
    assert resolved.y == [0.5, 0.2]


def test_captured_custom_codec_and_explicit_reuse(project: Path) -> None:
    setup(project)
    (project / "steps/ingest.py").write_text(
        "import json\n"
        "from mlef import ArtifactCodec, ArtifactRegistry, CustomType, Output, Step\n"
        'step = Step(name="ingest", outputs=[Output(name="data", '
        'type=CustomType(name="example.observations", version=1), format="application/json")])\n'
        "def serialize(value, path):\n"
        "    path.write_text(json.dumps(value))\n"
        "def validate(path):\n"
        "    if not isinstance(json.loads(path.read_text()), list):\n"
        '        raise ValueError("Expected observations.")\n'
        "registry = ArtifactRegistry()\n"
        'registry.register(ArtifactCodec(type=CustomType(name="example.observations", version=1), '
        'format="application/json", serializer=serialize, validator=validate))\n'
        "with step.context() as context:\n"
        '    step.save([registry.artifact(name="data", '
        'artifact_type=CustomType(name="example.observations", version=1), '
        'media_format="application/json", value=[1, 2, 3, 4, 5, 6], '
        'path=context.output_path("data", "data.json"))], context=context)\n'
    )
    script = project / "steps/split.py"
    script.write_text(
        script.read_text()
        .replace("from mlef import ", "from mlef import CustomType, ")
        .replace(
            "type=ArtifactType.FILE", 'type=CustomType(name="example.observations", version=1)'
        )
    )
    original = prepare(project)
    cache_environment(project, original)
    configure_cache(project, original)
    process = subprocess.run(
        [sys.executable, "-I", "-m", "mlef.execution", str(original)],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    finish_parent(original)
    previous = read_pipeline(original, verify=True)
    fork = prepare(project)
    record = reuse_steps(project, previous.run_id, fork, "train")
    assert record.attempts["ingest"].reused_from == previous.run_id
    assert execute_pipeline(project, fork) == 0
    finish_parent(fork)
    assert api.pipeline_status(record.run_id, root=project, verify=True).state == "complete"


def test_forced_step_executes_and_invalidates_its_dependents(
    project: Path, cached_pipeline: Path
) -> None:
    previous = read_pipeline(cached_pipeline)
    directory = prepare(project)
    cache_environment(project, directory)
    record = configure_cache(project, directory, candidate=previous.run_id, force=["train"])
    assert record.attempts["split"].reused_from == previous.run_id
    assert record.attempts["train"].reused_from is None
    assert record.attempts["evaluate"].reused_from is None
    assert record.attempts["train"].cache is not None
    assert "Forced" in " ".join(record.attempts["train"].cache.reasons)
    assert execute_pipeline(project, directory) == 0
    finish_parent(directory)
    assert api.pipeline_status(record.run_id, root=project, verify=True).state == "complete"


@pytest.mark.parametrize("change", ["source", "dependencies", "parameters", "environment", "sdk"])
def test_cache_explains_fingerprint_invalidation(
    project: Path, cached_pipeline: Path, change: str
) -> None:
    previous = read_pipeline(cached_pipeline)
    if change == "source":
        path = project / "steps/ingest.py"
        path.write_text(path.read_text() + "\nprint('changed')\n")
    elif change == "dependencies":
        with (project / "uv.lock").open("a") as stream:
            stream.write("\n# changed dependency evidence\n")
    elif change == "parameters":
        path = project / "mlef.toml"
        path.write_text(
            path.read_text().replace('name = "ingest"', 'name = "ingest"\nparameters = {seed = 42}')
        )
    directory = prepare(project)
    if change == "sdk":
        record = read_pipeline(directory)
        path = directory.parents[1] / "sdk" / record.sdk_wheel.filename
        path.write_bytes(path.read_bytes() + b"\n")
        record.sdk_wheel.sha256 = file_hash(path)
        _save(directory, record)
    cache_environment(project, directory, python="3.13.0" if change == "environment" else "3.12.0")
    record = configure_cache(project, directory, candidate=previous.run_id)
    decision = record.attempts["ingest"].cache
    assert decision is not None
    assert decision.action == "run"
    component = "source" if change == "dependencies" else change
    assert f"Fingerprint changed: {component}." in decision.reasons
    assert all(attempt.reused_from is None for attempt in record.attempts.values())


def test_mutable_ingestion_does_not_hit_cache(project: Path, cached_pipeline: Path) -> None:
    path = project / "mlef.toml"
    path.write_text(path.read_text().replace('cache_version = "public-v1"\n', ""))
    baseline = prepare(project)
    cache_environment(project, baseline)
    configure_cache(project, baseline)
    assert execute_pipeline(project, baseline) == 0
    finish_parent(baseline)
    directory = prepare(project)
    cache_environment(project, directory)
    record = configure_cache(project, directory, candidate=read_pipeline(baseline).run_id)
    decision = record.attempts["ingest"].cache
    assert decision is not None
    assert "Mutable ingestion" in " ".join(decision.reasons)
    assert all(attempt.reused_from is None for attempt in record.attempts.values())


def test_pipeline_parameters_reach_scripts_and_explicit_reuse_checks_them(
    project: Path, cached_pipeline: Path
) -> None:
    previous = read_pipeline(cached_pipeline)
    path = project / "mlef.toml"
    path.write_text(
        path.read_text().replace('name = "train"', 'name = "train"\nparameters = {offset = 10}')
    )
    directory = prepare(project)
    cache_environment(project, directory)
    configure_cache(project, directory, candidate=previous.run_id)
    assert execute_pipeline(project, directory) == 0
    model = attempt_directory(directory, "train") / "outputs/model/model.json"
    original = attempt_directory(cached_pipeline, "train") / "outputs/model/model.json"
    assert (
        json.loads(model.read_text())["prediction"]
        == json.loads(original.read_text())["prediction"] + 10
    )
    with pytest.raises(ValueError, match="incompatible"):
        reuse_steps(project, previous.run_id, prepare(project), "evaluate")


def test_cache_refuses_corrupt_candidate_and_forged_environment(
    project: Path, cached_pipeline: Path
) -> None:
    previous = read_pipeline(cached_pipeline)
    path = cached_pipeline.parents[1] / "environment.json"
    environment = json.loads(path.read_text())
    environment["container"]["python"] = "changed"
    path.write_text(json.dumps(environment))
    directory = prepare(project)
    cache_environment(project, directory)
    with pytest.raises(ValueError, match="environment identity"):
        configure_cache(project, directory, candidate=previous.run_id)


def test_disabled_cache_and_invalid_force_options(project: Path, cached_pipeline: Path) -> None:
    previous = read_pipeline(cached_pipeline)
    definition = project / "mlef.toml"
    definition.write_text(definition.read_text().replace("cache = true", "cache = false"))
    directory = prepare(project)
    cache_environment(project, directory)
    with pytest.raises(ValueError, match="known pipeline"):
        configure_cache(project, directory, candidate=previous.run_id, force=["unknown"])
    with pytest.raises(ValueError, match="requires a cache candidate"):
        configure_cache(project, directory, force=["train"])
    record = configure_cache(project, directory, candidate=previous.run_id)
    assert record.attempts["ingest"].cache is not None
    assert "Step caching is disabled." in record.attempts["ingest"].cache.reasons
    with pytest.raises(ValueError, match="once"):
        configure_cache(project, directory, candidate=previous.run_id)


def test_input_bytes_change_fingerprints_independently_of_source(
    project: Path, cached_pipeline: Path
) -> None:
    previous = read_pipeline(cached_pipeline)
    directory = prepare(project)
    cache_environment(project, directory)
    record = configure_cache(project, directory)
    step = record.contracts["ingest"]
    with step.local(attempt_directory(directory, "ingest"), run_id=record.run_id) as context:
        path = context.output_path("data", "data.json")
        path.write_text("[100, 200, 300]")
        step.save(
            [
                Artifact(
                    name="data", type=step.outputs[0].type, format=step.outputs[0].format, path=path
                )
            ],
            context=context,
        )
    record.attempts["ingest"].state = "complete"
    record.attempts["ingest"].exit_code = 0
    _save(directory, record)
    current = fingerprint(directory, record, "split")
    decision = previous.attempts["split"].cache
    assert decision is not None
    assert decision.fingerprint is not None
    assert current.source == decision.fingerprint.source
    assert current.inputs != decision.fingerprint.inputs


def test_pipeline_serialization_preserves_cache_and_parameter_fields(project: Path) -> None:
    setup(project)
    directory = prepare(project)
    record = read_pipeline(directory)
    _save(directory, record)
    values = json.loads((directory / "pipeline.json").read_text())
    assert "cache_from" in values
    assert "cache" in values["attempts"]["ingest"]
    assert "parameters" in values["definition"]["steps"][0]
    assert execute_pipeline(project, directory) == 0


@pytest.mark.parametrize(
    ("candidate", "reuse", "resume", "force"),
    [
        (None, None, False, ["train"]),
        ("candidate", "other", False, []),
        ("candidate", None, True, []),
    ],
)
def test_runner_rejects_ambiguous_cache_options(
    project: Path, candidate: str | None, reuse: str | None, resume: bool, force: list[str]
) -> None:
    with pytest.raises(ValueError, match="Cache/force"):
        run(
            project,
            Path("mlef.toml"),
            [],
            [],
            [],
            cache_from=candidate,
            reuse_from=reuse,
            from_step="train" if reuse is not None else None,
            resume=resume,
            force=force,
        )
    assert not (project / ".mlef/runs").exists()


def finish_parent(directory: Path) -> None:
    run_dir = directory.parents[1]
    parent = read_run(run_dir / "run.json")
    parent.state = "succeeded" if read_pipeline(directory).state == "complete" else "failed"
    parent.finished_at = now()
    parent.execution_exit_code = 0 if parent.state == "succeeded" else 1
    save(run_dir / "run.json", parent)


def test_explicit_reuse_runs_edited_training_and_is_self_contained(project: Path) -> None:
    setup(project)
    original = prepare(project)
    assert execute_pipeline(project, original) == 0
    finish_parent(original)
    previous = read_pipeline(original)
    script = project / "steps/train.py"
    script.write_text(script.read_text().replace("mean(values)", "mean(values) + 10"))
    fork = prepare(project)
    record = reuse_steps(project, previous.run_id, fork, "train")
    assert record.attempts["ingest"].reused_from == previous.run_id
    assert record.attempts["split"].reused_from == previous.run_id
    assert record.attempts["train"].state == "pending"
    assert record.attempts["evaluate"].state == "pending"
    original.parents[1].rename(original.parents[1].with_name("hidden-source"))
    assert execute_pipeline(project, fork) == 0
    finish_parent(fork)
    record = api.pipeline_status(record.run_id, root=project, verify=True)
    result = read_step(attempt_directory(fork, "train"))
    assert result.inputs["partitions"].run_id == previous.run_id
    model = attempt_directory(fork, "train") / "outputs/model/model.json"
    assert json.loads(model.read_text())["prediction"] > 10
    assert read_step(attempt_directory(fork, "ingest")).run_id == previous.run_id
    with pytest.raises(ValueError, match="only start once"):
        execute_pipeline(project, fork)


def test_fetched_run_supports_verified_reuse_from_a_clean_workspace(
    project: Path, tmp_path: Path
) -> None:
    setup(project)
    original = prepare(project)
    assert execute_pipeline(project, original) == 0
    finish_parent(original)
    previous = read_pipeline(original)
    remote = str(tmp_path / "store")
    published = api.publish_run(previous.run_id, remote, root=project)
    original.parents[1].rename(original.parents[1].with_name("removed-source"))
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    subprocess.run(["git", "init", "--quiet", str(fresh)], check=True)
    setup(fresh)
    assert api.fetch_run(previous.run_id, remote, root=fresh) == published
    assert api.pipeline_status(previous.run_id, root=fresh, verify=True) == previous
    fork = prepare(fresh)
    reused = reuse_steps(fresh, previous.run_id, fork, "train")
    assert reused.attempts["ingest"].reused_from == previous.run_id
    assert execute_pipeline(fresh, fork) == 0
    finish_parent(fork)
    assert api.pipeline_status(reused.run_id, root=fresh, verify=True).state == "complete"


@pytest.mark.parametrize("change", ["bytes", "script", "contract", "sdk", "archive", "binding"])
def test_reuse_rejects_corruption_and_incompatibility(project: Path, change: str) -> None:
    setup(project)
    original = prepare(project)
    assert execute_pipeline(project, original) == 0
    finish_parent(original)
    previous = read_pipeline(original)
    if change == "bytes":
        (attempt_directory(original, "ingest") / "outputs/data/data.json").write_text("[]")
    elif change == "sdk":
        (original.parents[1] / "sdk" / previous.sdk_wheel.filename).write_bytes(b"changed")
    elif change == "archive":
        (original.parents[1] / "source.tar.gz").write_bytes(b"changed")
    elif change == "script":
        script = project / "steps/ingest.py"
        script.write_text(script.read_text() + "\nprint('changed')\n")
    elif change == "contract":
        script = project / "steps/ingest.py"
        script.write_text(
            script.read_text().replace(
                'format="application/json")]', 'format="application/json", required=False)]'
            )
        )
        split = project / "steps/split.py"
        split.write_text(
            split.read_text().replace(
                'Input(name="data", type=ArtifactType.FILE, format="application/json")',
                'Input(name="data", type=ArtifactType.FILE, '
                'format="application/json", required=False)',
            )
        )
    else:
        definition = project / "mlef.toml"
        definition.write_text(definition.read_text().replace('data = "ingest.data"', ""))
        script = project / "steps/split.py"
        script.write_text(
            script.read_text().replace(
                'Input(name="data", type=ArtifactType.FILE, format="application/json")',
                'Input(name="data", type=ArtifactType.FILE, '
                'format="application/json", required=False)',
            )
        )
    fork = prepare(project)
    with pytest.raises(ValueError, match=r"changed|incompatible"):
        reuse_steps(project, previous.run_id, fork, "train")
    assert read_pipeline(fork).state == "preparing"


def test_resume_retains_source_and_sdk_and_failed_steps_are_not_reused(
    project: Path, tmp_path: Path
) -> None:
    setup(project)
    original = prepare(project)
    record = read_pipeline(original)
    assert execute_pipeline(project, original) == 0
    finish_parent(original)
    destination = project / ".mlef/runs" / str(uuid4())
    destination.mkdir()
    copied = copy_snapshot(project, record.run_id, destination)
    assert copied == read_pipeline(original)
    workspace = tmp_path / "resume"
    workspace.mkdir()
    extract_source(destination / "source.tar.gz", workspace)
    (project / "steps/train.py").write_text("raise RuntimeError('live edit')")
    directory = destination / "artifacts/pipeline"
    prepare_pipeline(
        workspace,
        Path(record.definition_path),
        directory,
        destination.name,
        record.source_sha256,
        record.sdk_wheel,
    )
    reuse_steps(project, record.run_id, directory, "train", resume=True)
    assert execute_pipeline(workspace, directory) == 0
    assert read_pipeline(directory, verify=True).resume
    old = read_pipeline(original)
    old.attempts["split"].state = "failed"
    old.attempts["train"].state = "skipped"
    old.attempts["evaluate"].state = "skipped"
    _save(original, old.model_copy(update={"state": "failed", "error": "failure"}))
    (project / "steps/train.py").write_text((workspace / "steps/train.py").read_text())
    with pytest.raises(ValueError, match="incomplete or incompatible"):
        reuse_steps(project, record.run_id, prepare(project), "train")


def test_frontier_reuses_independent_branches_and_preserves_transitive_lineage(
    project: Path,
) -> None:
    setup(project)
    script = project / "steps/audit.py"
    script.write_text(
        (project / "steps/ingest.py").read_text().replace('name="ingest"', 'name="audit"')
    )
    definition = project / "mlef.toml"
    definition.write_text(
        definition.read_text() + '\n[[steps]]\nname = "audit"\nscript = "steps/audit.py"\n'
    )
    original = prepare(project)
    assert execute_pipeline(project, original) == 0
    finish_parent(original)
    previous = read_pipeline(original)
    fork = prepare(project)
    record = reuse_steps(project, previous.run_id, fork, "train")
    assert record.attempts["audit"].reused_from == previous.run_id
    assert execute_pipeline(project, fork) == 0
    finish_parent(fork)
    next_fork = prepare(project)
    chained = reuse_steps(project, record.run_id, next_fork, "evaluate")
    assert chained.attempts["ingest"].reused_from == previous.run_id
    assert chained.attempts["train"].reused_from == record.run_id
    assert execute_pipeline(project, next_fork) == 0
    read_pipeline(next_fork, verify=True)


@pytest.mark.parametrize(
    ("reuse_from", "from_step", "resume"),
    [
        ("x", None, False),
        (None, "train", False),
        (None, None, True),
        ("x", "train", True),
    ],
)
def test_reuse_options_reject_ambiguous_requests(
    project: Path, reuse_from: str | None, from_step: str | None, resume: bool
) -> None:
    with pytest.raises(ValueError, match=r"together|only supported"):
        run(
            project,
            Path("train.py"),
            [],
            [],
            [],
            reuse_from=reuse_from,
            from_step=from_step,
            resume=resume,
        )


def test_cli_forwards_explicit_reuse(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    initialize(project)
    calls = []
    monkeypatch.setattr("mlef.cli.run", lambda *args, **kwargs: calls.append(kwargs) or 0)
    assert main(["run", "--reuse-from", str(uuid4()), "--from-step", "train", "--resume"]) == 0
    assert calls[0]["from_step"] == "train"
    assert calls[0]["resume"] is True


def test_reuse_rejects_active_and_cleanup_pending_sources(project: Path) -> None:
    setup(project)
    original = prepare(project)
    record = read_pipeline(original)
    with pytest.raises(ValueError, match="terminal source"):
        verify_source(project, record.run_id)
    finish_parent(original)
    with run_lock(original.parents[1]), pytest.raises(ValueError, match="active runner"):
        reuse_steps(project, record.run_id, prepare(project), "train")
    parent = read_run(original.parents[1] / "run.json")
    assert parent.runtime is not None
    parent.runtime.cleanup_pending = True
    save(original.parents[1] / "run.json", parent)
    with pytest.raises(ValueError, match="resources need recovery"):
        verify_source(project, record.run_id)


def test_runner_resume_never_reads_live_source_or_recaptures_sdk(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup(project)
    original = prepare(project)
    assert execute_pipeline(project, original) == 0
    finish_parent(original)
    previous = read_pipeline(original)
    (project / "mlef.toml").unlink()
    (project / "steps").rename(project / "edited-steps")
    monkeypatch.setattr("mlef.runner.capture", lambda *args: pytest.fail("recaptured source"))
    monkeypatch.setattr("mlef.runner.capture_sdk", lambda *args: pytest.fail("recaptured SDK"))
    monkeypatch.setattr("mlef.runner.shutil.which", lambda name: None)
    identifier = str(uuid4())
    assert (
        run(
            project,
            Path("mlef.toml"),
            [],
            [],
            [],
            reuse_from=previous.run_id,
            from_step="train",
            resume=True,
            _identifier=identifier,
        )
        == 1
    )
    record = api.pipeline_status(identifier, root=project, verify=True)
    assert record.source_sha256 == previous.source_sha256
    assert record.sdk_wheel == previous.sdk_wheel
    assert record.attempts["ingest"].reused_from == previous.run_id
    assert record.attempts["train"].state == "skipped"
    assert record.resume


def setup(project: Path) -> None:
    initialize(project)
    for name in ("pyproject.toml", "uv.lock"):
        (project / name).write_bytes((LOCKED / name).read_bytes())


def prepare(project: Path) -> Path:
    identifier = str(uuid4())
    run_dir = project / ".mlef/runs" / identifier
    run_dir.mkdir()
    sdk = capture_sdk(run_dir / "sdk")
    definition = validate(project).definition
    assert definition is not None
    selected = source_files(
        project,
        Path("mlef.toml"),
        [Path(entry.script) for entry in definition.steps]
        + [Path(entry.contract) for entry in definition.steps if entry.contract is not None],
    )
    git, inputs = capture(project, run_dir, selected)
    save(
        run_dir / "run.json",
        RunRecord(
            id=identifier,
            state="preparing",
            created_at=now(),
            command=["python", "-I", "-m", "mlef.execution", "/outputs/pipeline"],
            git=git,
            inputs=inputs,
            runtime=supervision.identity(project, identifier, str(uuid4())),
        ),
    )
    directory = run_dir / "artifacts/pipeline"
    prepare_pipeline(project, Path("mlef.toml"), directory, identifier, inputs.source_sha256, sdk)
    return directory


def test_model_alias_changes_and_history(
    project: Path,
    evaluation_project: Callable[[str], tuple[Path, list[str]]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    original = prepare(project)
    assert execute_pipeline(project, original) == 0
    finish_parent(original)
    definition, selectors = evaluation_project(original.parents[1].name)
    first = api.model_version(selectors[0])
    second = api.model_version(selectors[1])
    first_path = project / ".mlef/models/constant" / f"{first.version}.json"
    immutable = first_path.read_bytes()
    pointer = api.set_model_alias(selectors[0], alias="champion", note="Initial baseline")
    assert api.model_version("constant:champion") == first
    assert api.set_model_alias(selectors[0], alias="champion") == pointer
    moved = api.set_model_alias(selectors[1], alias="champion", note="Selected candidate")
    assert len(moved.history) == 2
    assert moved.history[-1].previous_version == first.version
    assert moved.history[-1].version == second.version
    assert api.model_version("constant:champion") == second
    destination = api.get_model("constant:champion", output=project / "champion-model")
    assert json.loads((destination / "model.json").read_text())["prediction"] == 2
    with (
        file_lock(project / ".mlef/models/catalog.lock"),
        pytest.raises(api.MlefError, match="active"),
    ):
        api.set_model_alias(selectors[0], alias="champion")
    assert api.model_alias("constant:champion") == moved
    api.set_model_alias(selectors[0], alias="Champion")
    assert len(api.list_model_aliases("constant").aliases) == 2
    assert len(api.list_models().versions) == 2
    capsys.readouterr()
    assert main(["models", "alias-history", "constant:champion", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == moved.model_dump(mode="json")
    assert first_path.read_bytes() == immutable
    with pytest.raises(api.MlefError, match="exact"):
        api.set_model_alias("constant:champion", alias="chain")
    with pytest.raises(api.MlefError, match="distinct model producers"):
        api.evaluate(definition, [selectors[0], "constant:Champion"])
    assert not (project / ".mlef/evaluations").exists()
    from mlef.models import _alias_path

    path = _alias_path(project, "constant", "champion")
    broken = moved.model_dump(mode="json")
    broken["history"][0]["previous_version"] = "f" * 64
    path.write_text(json.dumps(broken))
    assert api.list_model_aliases("constant").diagnostics
    with pytest.raises(api.MlefError, match="history"):
        api.set_model_alias(selectors[0], alias="champion")


@pytest.mark.parametrize(
    ("behavior", "direction", "target"),
    [
        ("complete", "minimize", None),
        ("complete", "maximize", None),
        ("complete", "none", None),
        ("complete", "target", 0.5),
        ("complete", "target", 1.0),
        ("partial", "minimize", None),
        ("cancelled", "minimize", None),
    ],
)
def test_frozen_evaluation_batches(
    project: Path,
    evaluation_project: Callable[[str], tuple[Path, list[str]]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    behavior: str,
    direction: str,
    target: float | None,
) -> None:
    original = prepare(project)
    assert execute_pipeline(project, original) == 0
    finish_parent(original)
    definition, selectors = evaluation_project(original.parents[1].name)
    original_evaluator = (project / "evaluate.py").read_text()
    parameters = f'direction="{direction}"'
    if target is not None:
        parameters += f",target={target}"
    if behavior == "partial":
        parameters += ",fail_baseline=true"
    path = project / definition
    path.write_text("parameters={" + parameters + "}\n" + path.read_text())
    calls = []
    candidates = selectors
    if behavior == "complete" and direction == "minimize":
        api.set_model_alias(selectors[0], alias="baseline")
        api.set_model_alias(selectors[1], alias="candidate")
        candidates = ["constant:baseline", "constant:candidate"]

    def execute(request: supervision.Request, delivery: Delivery) -> int:
        calls.append(request.identifier)
        if len(calls) == 1 and candidates != selectors:
            api.set_model_alias(selectors[0], alias="candidate")
        if behavior == "cancelled":
            return 130
        target = project / ".mlef/runs" / request.identifier
        target.mkdir()
        assert request.source_snapshot is not None
        snapshot = restore(Path(request.source_snapshot), target)
        workspace = project / ".mlef/workspaces" / request.identifier
        workspace.mkdir(parents=True)
        extract_source(target / "source.tar.gz", workspace)
        parent = RunRecord(
            id=request.identifier,
            state="preparing",
            created_at=now(),
            command=["python", "evaluate"],
            git=snapshot.git,
            inputs=snapshot.inputs,
        )
        save(target / "run.json", parent)
        pipeline = target / "artifacts/pipeline"
        prepare_pipeline(
            workspace,
            Path(request.script),
            pipeline,
            request.identifier,
            snapshot.inputs.source_sha256,
            snapshot.sdk,
        )
        stage_external_inputs(project, pipeline)
        code = execute_pipeline(workspace, pipeline)
        parent.state = "succeeded" if code == 0 else "failed"
        parent.execution_exit_code = code
        parent.finished_at = now()
        if code != 0:
            parent.error = RunError(category="execution", message="candidate failed")
        save(target / "run.json", parent)
        cache_environment(project, pipeline)
        # Both children must use the frozen evaluator despite edits between runs.
        (project / "evaluate.py").write_text("raise RuntimeError('live edit')")
        return code

    monkeypatch.setattr("mlef.evaluations.supervise", execute)
    batch = api.evaluate(definition, candidates)
    assert batch.state == behavior
    assert api.evaluation_batch(batch.id) == batch
    assert api.list_evaluations(model=selectors[0]).batches == [batch]
    if candidates != selectors:
        assert batch.evaluations[1].model.version == api.model_version(selectors[1]).version
        assert (
            api.model_version("constant:candidate").version
            == api.model_version(selectors[0]).version
        )
        assert api.list_evaluations(model="constant:candidate").batches == [batch]
    assert api.list_evaluations(limit=1, offset=1).batches == []
    assert api.list_evaluations(limit=1, offset=1).total == 1
    if behavior == "cancelled":
        assert len(calls) == 1
        assert [item.state for item in batch.evaluations] == ["cancelled", "pending"]
    else:
        assert len(calls) == 2
        assert batch.evaluations[1].state == "succeeded"
        assert batch.evaluations[1].metric_artifacts
        assert "loss" not in batch.evaluations[1].metrics[0].metrics.values
        assert batch.evaluations[1].metrics[0].metrics.values["mae"].model_dump()["value"] == 0
        if behavior == "partial":
            assert batch.evaluations[0].state == "failed"
        else:
            assert batch.evaluations[0].metrics[0].metrics.values["mae"].model_dump()["value"] == 1
    capsys.readouterr()
    assert main(["evaluations", "show", batch.id, "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == batch.model_dump(mode="json")
    with monkeypatch.context() as patch:
        patch.setattr(api, "evaluate", lambda *args, **kwargs: batch)
        assert (
            main(["evaluate", str(definition), "--model", selectors[0], "--json"])
            == {
                "complete": 0,
                "partial": 1,
                "cancelled": 130,
            }[behavior]
        )
        assert json.loads(capsys.readouterr().out) == batch.model_dump(mode="json")
    if behavior == "complete":
        identifiers = [item.run_id for item in batch.evaluations]
        report = api.compare_evaluations(
            identifiers, baseline=identifiers[0], metric="mae", scope="test"
        )
        assert all(row.compatibility == "declared-compatible" for row in report.rows), (
            report.model_dump_json()
        )
        assert report.rows[1].delta == -1
        expected = (
            "unavailable"
            if direction == "none"
            else "improved"
            if direction == "minimize"
            else "unchanged"
            if target == 0.5
            else "regressed"
        )
        assert report.rows[1].observation == expected
        assert report.rows[1].rank == (
            None if direction == "none" else 2 if expected == "regressed" else 1
        )
        table = api.evaluation_leaderboard(batch.id, baseline=identifiers[0], metric="mae")
        assert table.rows[0].rank == (None if direction == "none" else 1)
        zero_baseline = api.compare_evaluations(identifiers, baseline=identifiers[1], metric="mae")
        assert zero_baseline.rows[0].relative_delta is None
        curves = api.compare_evaluations(identifiers, baseline=identifiers[0], metric="loss")
        assert all(not row.eligible for row in curves.rows)
        assert all(row.representation == "series" for row in curves.rows)
        assert (
            main(
                [
                    "evaluations",
                    "compare",
                    *identifiers,
                    "--baseline",
                    identifiers[0],
                    "--metric",
                    "mae",
                    "--scope",
                    "test",
                    "--json",
                ]
            )
            == 0
        )
        assert json.loads(capsys.readouterr().out) == report.model_dump(mode="json")
        if direction == "minimize":
            environment_path = project / ".mlef/runs" / identifiers[1] / "environment.json"
            saved_environment = environment_path.read_bytes()
            environment = json.loads(saved_environment)
            environment["container"]["python"] = "3.12.99"
            environment_path.write_text(json.dumps(environment))
            incompatible = api.compare_evaluations(
                identifiers, baseline=identifiers[0], metric="mae"
            )
            assert incompatible.rows[1].compatibility == "incompatible"
            assert incompatible.rows[1].rank is None
            environment_path.write_bytes(saved_environment)
            environment_path.unlink()
            missing = api.compare_evaluations(identifiers, baseline=identifiers[0], metric="mae")
            assert missing.rows[1].compatibility == "unknown"
            assert missing.rows[1].value == 0
            assert missing.rows[1].rank is None
            environment_path.write_bytes(saved_environment)
            context_path = environment_path.parent / "evaluation.json"
            saved_context = context_path.read_bytes()
            context = json.loads(saved_context)
            context["protocol_id"] = "f" * 64
            context_path.write_text(json.dumps(context))
            invalid = api.compare_evaluations(identifiers, baseline=identifiers[0], metric="mae")
            assert invalid.rows[1].compatibility == "unknown"
            assert invalid.rows[1].rank is None
            context_path.write_bytes(saved_context)
            (project / "evaluate.py").write_text(original_evaluator)
            repeated_batch = api.evaluate(definition, selectors)
            repeated = api.compare_evaluations(
                [identifiers[0], repeated_batch.evaluations[0].run_id],
                baseline=identifiers[0],
                metric="mae",
            )
            assert repeated.rows[1].duplicate_of == identifiers[0]
            assert repeated.rows[1].rank is None
            (project / "evaluate.py").write_text(original_evaluator)
            path.write_text(
                path.read_text().replace("parameters={", "parameters={seed=43, duplicate=true, ", 1)
            )
            second = api.evaluate(definition, selectors)
            assert second.state == "complete"
            second_ids = [item.run_id for item in second.evaluations]
            with pytest.raises(api.MlefError, match="ambiguous"):
                api.compare_evaluations(second_ids, baseline=second_ids[0], metric="mae")
            qualified = api.compare_evaluations(
                second_ids, baseline=second_ids[0], metric="mae", scope="test"
            )
            assert all(row.eligible for row in qualified.rows)
            mismatched = api.compare_evaluations(
                [identifiers[0], second.evaluations[1].run_id],
                baseline=identifiers[0],
                metric="mae",
                scope="test",
            )
            assert mismatched.rows[1].compatibility == "incompatible"
            assert any("recipe" in reason for reason in mismatched.rows[1].reasons)
            metric_path = (
                environment_path.parent
                / "artifacts/pipeline/steps/evaluate/1/outputs/metrics/metrics.json"
            )
            saved_metric = metric_path.read_bytes()
            metric_path.write_text("{}")
            corrupted = api.compare_evaluations(identifiers, baseline=identifiers[0], metric="mae")
            assert corrupted.rows[1].compatibility == "unknown"
            assert corrupted.rows[1].value is None
            assert corrupted.rows[1].rank is None
            metric_path.write_bytes(saved_metric)
            (project / "evaluate.py").write_text(original_evaluator)
            selected = api.model_version(selectors[0]).artifact
            assert selected.manifest.producer is not None
            path.write_text(
                path.read_text().replace("parameters={", "parameters={use_aux=true, ", 1)
                + "\n[bindings.aux]\nrun_id="
                + json.dumps(selected.location_run_id)
                + "\n[bindings.aux.artifact]\n"
                + "\n".join(
                    f"{key}={json.dumps(value)}"
                    for key, value in selected.manifest.producer.model_dump().items()
                )
            )
            wrong_model = api.evaluate(definition, selectors)
            assert wrong_model.state == "partial"
            assert wrong_model.evaluations[1].state == "failed"
            assert "different selected model" in (wrong_model.evaluations[1].error or "")
        snapshot = project / ".mlef/evaluations" / batch.id / batch.evaluations[0].run_id
        (snapshot / "source.tar.gz").write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="Frozen evaluation source"):
            restore(snapshot, project / ".mlef/changed-snapshot")
    else:
        identifiers = [item.run_id for item in batch.evaluations]
        failed_baseline = api.evaluation_leaderboard(
            batch.id, baseline=identifiers[0], metric="mae"
        )
        assert all(row.rank is None for row in failed_baseline.rows)
        if behavior == "partial":
            successful_baseline = api.evaluation_leaderboard(
                batch.id, baseline=identifiers[1], metric="mae"
            )
            assert successful_baseline.rows[0].run_id == identifiers[1]
            assert successful_baseline.rows[0].rank == 1


@pytest.mark.parametrize("failure", [None, "checksum", "type", "active"])
def test_exact_external_inputs_are_self_contained(project: Path, failure: str | None) -> None:
    setup(project)
    original = prepare(project)
    assert execute_pipeline(project, original) == 0
    identifier = original.parents[1].name
    parent = read_run(original.parents[1] / "run.json")
    if failure != "active":
        parent.state = "succeeded"
        parent.finished_at = now()
        parent.execution_exit_code = 0
        save(original.parents[1] / "run.json", parent)
    lines = ["schema_version = 2", "[[steps]]", 'name = "evaluate"', 'script = "steps/evaluate.py"']
    lines.append("cache = true")
    for binding, producer, output in [
        ("model", "train", "model"),
        ("partitions", "split", "partitions"),
    ]:
        reference = read_step(attempt_directory(original, producer)).artifacts[output].producer
        assert reference is not None
        if failure == "checksum":
            reference = reference.model_copy(update={"sha256": "f" * 64})
        lines.extend(
            [
                f"[steps.bindings.{binding}]",
                f'run_id = "{identifier}"',
                f"[steps.bindings.{binding}.artifact]",
            ]
        )
        lines.extend(
            f"{key} = {json.dumps(value)}" for key, value in reference.model_dump().items()
        )
    (project / "mlef.toml").write_text("\n".join(lines))
    if failure == "type":
        script = project / "steps/evaluate.py"
        script.write_text(script.read_text().replace("ArtifactType.FILE", "ArtifactType.DIRECTORY"))
    report = validate(project)
    assert report.status == "verified"
    directory = prepare(project)
    if failure is not None:
        with pytest.raises(
            ValueError,
            match={"checksum": "exact producer", "type": "incompatible", "active": "terminal"}[
                failure
            ],
        ):
            stage_external_inputs(project, directory)
        return
    stage_external_inputs(project, directory)
    record = read_pipeline(directory, verify=True)
    external = record.definition.steps[0].bindings["model"]
    assert isinstance(external, ExternalInput)
    assert record.order == ["evaluate"]
    assert record.execution_steps("evaluate") == {"evaluate"}
    # Hide the producer to demonstrate execution and verification use captured copies.
    source_dir = original.parents[1]
    hidden = project / ".mlef/hidden-producer"
    assert source_dir.resolve().is_relative_to((project / ".mlef/runs").resolve())
    assert hidden.resolve().is_relative_to((project / ".mlef").resolve())
    source_dir.rename(hidden)
    cache_environment(project, directory)
    configure_cache(project, directory)
    assert execute_pipeline(project, directory) == 0
    completed = read_pipeline(directory, verify=True)
    assert completed.state == "complete"
    assert read_step(attempt_directory(directory, "evaluate")).inputs["model"] == external.artifact
    parent = read_run(directory.parents[1] / "run.json")
    parent.state = "succeeded"
    parent.finished_at = now()
    parent.execution_exit_code = 0
    save(directory.parents[1] / "run.json", parent)
    store = LocalStore(project / ".mlef/transfer-store")
    publish(project, parent.id, store)
    fetched_root = project / ".mlef/fetched-project"
    initialize(fetched_root, scaffold=False)
    fetch(fetched_root, parent.id, store)
    fetched_pipeline = fetched_root / ".mlef/runs" / parent.id / "artifacts/pipeline"
    assert read_pipeline(fetched_pipeline, verify=True).state == "complete"
    resumed = prepare(project)
    stage_external_inputs(project, resumed, resume_from=parent.id)
    cache_environment(project, resumed)
    configure_cache(project, resumed, candidate=parent.id)
    assert read_pipeline(resumed).attempts["evaluate"].reused_from == parent.id
    assert execute_pipeline(project, resumed) == 0
    assert read_pipeline(resumed, verify=True).state == "complete"
    copied_model = imported_directory(resumed, external) / "outputs/model/model.json"
    copied_model.write_text("{}")
    with pytest.raises(ValueError, match="bytes or inventory changed"):
        read_pipeline(resumed, verify=True)


def test_model_catalog_registration_and_retrieval(
    project: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    setup(project)
    for script in ("train", "evaluate"):
        path = project / "steps" / f"{script}.py"
        path.write_text(path.read_text().replace("ArtifactType.FILE", "ArtifactType.MODEL"))
    path = project / "steps/train.py"
    path.write_text(
        "from mlef.artifacts import ModelMetadata\n"
        + path.read_text()
        .replace('name="train",', 'schema_version=2, kind="train", name="train",')
        .replace(
            "path=path)",
            'path=path, metadata=ModelMetadata(framework="constant", serialization="json"))',
        )
    )
    directory = prepare(project)
    assert execute_pipeline(project, directory) == 0
    identifier = directory.parents[1].name
    discovered = api.list_models(run_id=identifier)
    assert len(discovered.artifacts) == 1
    assert discovered.artifacts[0].output == "model"
    with pytest.raises(api.MlefError, match="terminal run"):
        api.register_model(identifier, "train", "model", name="baseline")
    parent = read_run(directory.parents[1] / "run.json")
    parent.state = "succeeded"
    parent.finished_at = now()
    parent.execution_exit_code = 0
    save(directory.parents[1] / "run.json", parent)
    version = api.register_model(identifier, "train", "model", name="baseline")
    assert api.register_model(identifier, "train", "model", name="baseline") == version
    selector = f"baseline@{version.version}"
    assert api.model_version(selector) == version
    assert api.list_models().versions == [version]
    capsys.readouterr()
    assert main(["models", "show", selector, "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == version.model_dump(mode="json")
    destination = api.get_model(selector, output=project / "retrieved-model")
    assert (destination / "model.json").read_bytes() == (
        attempt_directory(directory, "train") / "outputs/model/model.json"
    ).read_bytes()
    with pytest.raises(api.MlefError, match="already exists"):
        api.get_model(selector, output=destination)
    with run_lock(directory.parents[1]), pytest.raises(api.MlefError, match="active"):
        api.register_model(identifier, "train", "model", name="baseline")
    with (
        file_lock(project / ".mlef/models/catalog.lock"),
        pytest.raises(api.MlefError, match="active"),
    ):
        api.register_model(identifier, "train", "model", name="baseline")
    with pytest.raises(api.MlefError, match="No completed typed model"):
        api.register_model(identifier, "evaluate", "metrics", name="baseline")
    with pytest.raises(api.MlefError):
        api.register_model(identifier, "train", "model", name="../escape")
    catalog_path = project / ".mlef/models/baseline" / f"{version.version}.json"
    catalog_path.write_text("{}")
    assert api.list_models().diagnostics
    with pytest.raises(api.MlefError, match="Cannot read model version"):
        api.model_version(selector)
    model_path = attempt_directory(directory, "train") / "outputs/model/model.json"
    model_path.write_text("{}")
    with pytest.raises(api.MlefError, match="bytes or inventory changed"):
        api.register_model(identifier, "train", "model", name="other")


def test_offline_pipeline_metric_inspection(
    project: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from mlef.cli import main

    setup(project)
    directory = prepare(project)
    assert execute_pipeline(project, directory) == 0
    identifier = directory.parents[1].name
    evidence = api.run_metrics(identifier, step="evaluate", scope="test", metric="mae")
    # The scaffold declares test scope and retains the exact model reference.
    assert len(evidence.artifacts) == 1
    artifact = evidence.artifacts[0]
    assert artifact.metrics.values["mae"].model_dump()["value"] == 3
    assert artifact.metrics.model is not None
    assert artifact.producer.step == "evaluate"
    assert evidence.root is None
    assert api.run_metrics(identifier, scope="validation").artifacts == []
    assert api.run_metrics(identifier, metric="nonexistent").artifacts == []
    capsys.readouterr()
    assert main(["runs", "metrics", identifier, "--step", "evaluate", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == api.run_metrics(
        identifier,
        step="evaluate",
    ).model_dump(mode="json")
    with pytest.raises(api.MlefError, match="max_bytes"):
        api.run_metrics(identifier, max_bytes=1)
    with pytest.raises(api.MlefError, match="Unknown pipeline step"):
        api.run_metrics(identifier, step="missing")
    path = attempt_directory(directory, "evaluate") / "outputs/metrics/metrics.json"
    path.write_text("{}")
    with pytest.raises(api.MlefError, match="bytes or inventory changed"):
        api.run_metrics(identifier)


@pytest.mark.parametrize("imperative", [False, True])
def test_captured_scripts_ignore_live_edits_and_run_once(
    project: Path, tmp_path: Path, imperative: bool
) -> None:
    setup(project)
    if imperative:
        script = project / "steps/train.py"
        script.write_text(script.read_text().replace("    with context:", "    if True:"))
    directory = prepare(project)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    extract_source(directory.parents[1] / "source.tar.gz", snapshot)
    (project / "steps/ingest.py").write_text("raise RuntimeError('live edit')")
    assert execute_pipeline(snapshot, directory) == 0
    record = read_pipeline(directory, verify=True)
    assert record.state == "complete"
    assert all(attempt.state == "complete" for attempt in record.attempts.values())
    assert all(attempt.exit_code == 0 for attempt in record.attempts.values())
    for name in record.order:
        assert read_step(attempt_directory(directory, name)).run_id == record.run_id
    metrics = Metrics.model_validate_json(
        (attempt_directory(directory, "evaluate") / "outputs/metrics/metrics.json").read_text()
    )
    assert metrics.values["mae"].model_dump()["value"] == 3
    model = read_step(attempt_directory(directory, "train")).artifacts["model"]
    assert metrics.model == model.producer
    assert api.pipeline_status(record.run_id, root=project, verify=True) == record
    assert "pipeline/pipeline.json" in api.artifacts(record.run_id, root=project)
    assert api.list_runs(root=project).runs[0].id == record.run_id
    with pytest.raises(ValueError, match="only start once"):
        execute_pipeline(snapshot, directory)


@pytest.mark.parametrize(
    "failure", ["after_save", "hard_exit", "missing_save", "contract", "mutation"]
)
def test_failed_process_never_publishes_and_skips_downstream(project: Path, failure: str) -> None:
    setup(project)
    script = project / "steps/train.py"
    text = script.read_text()
    if failure == "after_save":
        text += "\nraise RuntimeError('after saving')\n"
    elif failure == "hard_exit":
        text += "\nimport os\nos._exit(17)\n"
    elif failure == "missing_save":
        text = text.split("def main()", 1)[0] + "\nwith step.context():\n    pass\n"
    elif failure == "mutation":
        text += (
            "\nimport os\n"
            "config = json.loads(Path(os.environ['MLEF_STEP_CONTEXT']).read_text())\n"
            "Path(config['directory'], 'outputs/model/model.json').write_text('changed')\n"
        )
    else:
        text = text.replace(
            'if __name__ == "__main__":',
            'step.outputs[0].format = "text/plain"\nif __name__ == "__main__":',
        )
    script.write_text(text, encoding="utf-8")
    directory = prepare(project)
    assert execute_pipeline(project, directory) == 1
    record = read_pipeline(directory, verify=True)
    assert record.state == "failed"
    assert record.attempts["ingest"].state == "complete"
    assert record.attempts["split"].state == "complete"
    assert record.attempts["train"].state == "failed"
    assert record.attempts["train"].error is not None
    assert record.attempts["evaluate"].state == "skipped"
    assert not attempt_directory(directory, "evaluate").exists()
    with pytest.raises((OSError, ValueError)):
        read_step(attempt_directory(directory, "train"))
    if failure in {"after_save", "hard_exit"}:
        assert (attempt_directory(directory, "train") / "outputs/model/model.json").exists()


def test_scripts_cannot_change_later_captured_script_bytes(project: Path) -> None:
    setup(project)
    script = project / "steps/ingest.py"
    script.write_text(script.read_text() + "\nPath('steps/train.py').write_text('print(123)')\n")
    directory = prepare(project)
    assert execute_pipeline(project, directory) == 1
    record = read_pipeline(directory)
    assert record.error is not None
    assert "Captured script changed" in record.error
    assert record.attempts["train"].exit_code is None
    assert record.attempts["evaluate"].state == "skipped"


def test_optional_output_becomes_an_unbound_optional_input(project: Path) -> None:
    setup(project)
    (project / "mlef.toml").write_text(
        '[[steps]]\nname="producer"\nscript="producer.py"\n'
        '[[steps]]\nname="consumer"\nscript="consumer.py"\n'
        'bindings={maybe="producer.maybe"}\n'
    )
    (project / "producer.py").write_text(
        "from mlef import Step, Output, ArtifactType\n"
        "step=Step(name='producer', outputs=[Output(name='maybe', type=ArtifactType.FILE, "
        "format='text/plain', required=False)])\n"
        "with step.context() as context:\n    step.save([], context=context)\n"
    )
    (project / "consumer.py").write_text(
        "from mlef import Step, Input, Output, Artifact, ArtifactType\n"
        "step=Step(name='consumer', inputs=[Input(name='maybe', type=ArtifactType.FILE, "
        "format='text/plain', required=False)], outputs=[Output(name='report', "
        "type=ArtifactType.FILE, format='text/plain')])\n"
        "with step.context() as context:\n"
        "    assert not context.inputs\n"
        "    path=context.output_path('report', 'value.txt')\n"
        "    path.write_text('no input')\n"
        "    step.save([Artifact(name='report', type=ArtifactType.FILE, "
        "format='text/plain', path=path)], context=context)\n"
    )
    directory = prepare(project)
    assert execute_pipeline(project, directory) == 0
    assert read_pipeline(directory, verify=True).state == "complete"
    assert not read_step(attempt_directory(directory, "consumer")).inputs


def test_explicit_contract_for_computed_declaration_is_captured(
    project: Path, tmp_path: Path
) -> None:
    setup(project)
    script = project / "steps/ingest.py"
    script.write_text(
        script.read_text()
        .replace("step = Step(", "declared = Step(")
        .replace("def main()", "step = declared\n\n\ndef main()")
    )
    (project / "ingest-contract.toml").write_text(
        'name="ingest"\n[[outputs]]\nname="data"\ntype="file"\nformat="application/json"\n'
    )
    definition = project / "mlef.toml"
    definition.write_text(
        definition.read_text().replace(
            'script = "steps/ingest.py"',
            'script = "steps/ingest.py"\ncontract = "ingest-contract.toml"',
        )
    )
    directory = prepare(project)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    extract_source(directory.parents[1] / "source.tar.gz", snapshot)
    assert (snapshot / "ingest-contract.toml").exists()
    assert execute_pipeline(snapshot, directory) == 0
    assert read_pipeline(directory, verify=True).state == "complete"


@pytest.mark.parametrize(
    "payload", ["source", "sdk", "output", "wiring", "script_hash", "definition"]
)
def test_offline_verification_rejects_changed_evidence(project: Path, payload: str) -> None:
    setup(project)
    directory = prepare(project)
    assert execute_pipeline(project, directory) == 0
    record = read_pipeline(directory)
    if payload == "source":
        path = directory.parents[1] / "source.tar.gz"
    elif payload == "sdk":
        path = directory.parents[1] / "sdk" / record.sdk_wheel.filename
    elif payload == "output":
        path = attempt_directory(directory, "train") / "outputs/model/model.json"
    else:
        path = directory / "pipeline.json"
    if payload in {"wiring", "script_hash", "definition"}:
        data = json.loads(path.read_text())
        if payload == "wiring":
            data["definition"]["steps"][-1]["bindings"]["model"] = "ingest.data"
        elif payload == "script_hash":
            data["attempts"]["train"]["script_sha256"] = "0" * 64
        else:
            data["definition_path"] = "another.toml"
        path.write_text(json.dumps(data))
    else:
        with path.open("ab") as stream:
            stream.write(b"changed")
    with pytest.raises(api.MlefError) as failure:
        api.pipeline_status(record.run_id, root=project, verify=True)
    assert failure.value.category == "invalid_record"


def test_sdk_wheel_is_deterministic_and_installable(tmp_path: Path) -> None:
    first = capture_sdk(tmp_path / "first")
    assert first == capture_sdk(tmp_path / "second")
    wheel = tmp_path / "first" / first.filename
    with zipfile.ZipFile(wheel) as archive:
        assert "mlef/execution.py" in archive.namelist()
        assert not any(
            "__pycache__" in name or name.endswith(".pyc") for name in archive.namelist()
        )
        inventory = archive.read(f"mlef-{first.version}.dist-info/RECORD").decode()
        assert len(list(csv.reader(io.StringIO(inventory)))) == len(archive.namelist())
    target = tmp_path / "installed"
    subprocess.run(
        ["uv", "pip", "install", "--no-deps", "--no-index", "--target", str(target), str(wheel)],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
            "import mlef; from mlef.execution import PipelineRecord; "
            "assert Path(mlef.__file__).is_relative_to(Path(sys.argv[1])); "
            "assert callable(mlef.Step.context)",
            str(target),
        ],
        check=True,
    )
    recipe = build_recipe("python@sha256:base", "uv@sha256:uv", None, first)
    assert first.sha256 in recipe
    assert first.filename in recipe
    assert "--no-deps --no-index" in recipe
    assert "uv pip check" in recipe
    with pytest.raises(ValueError, match="locked"):
        build_recipe("python@sha256:base", None, None, first)


def test_editable_sdk_capture_excludes_private_and_generated_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "package"
    (package / "templates").mkdir(parents=True)
    (package / "__pycache__").mkdir()
    for name in ("__init__.py", "templates/ingest.py.txt", "templates/mlef.toml"):
        (package / name).write_text("included", encoding="utf-8")
    for name in (".env", "secret.key", "private.txt", "__pycache__/cached.py", "module.pyc"):
        (package / name).write_text("must not be captured", encoding="utf-8")
    monkeypatch.setattr(sdk_distribution, "__file__", str(package / "sdk_distribution.py"))
    sdk = capture_sdk(tmp_path / "sdk")
    with zipfile.ZipFile(tmp_path / "sdk" / sdk.filename) as archive:
        files = {name for name in archive.namelist() if name.startswith("mlef/")}
    assert files == {"mlef/__init__.py", "mlef/templates/ingest.py.txt", "mlef/templates/mlef.toml"}


def test_pipeline_captures_untracked_steps_once_without_docker(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup(project)
    captures: list[list[Path]] = []

    def capture_once(root: Path, run_dir: Path, files: list[Path]) -> object:
        captures.append(files)
        return capture(root, run_dir, files)

    monkeypatch.setattr("mlef.runner.capture", capture_once)
    monkeypatch.setattr("mlef.runner.shutil.which", lambda name: None)
    assert run(project, Path("mlef.toml"), [], [], [], _identifier=str(uuid4())) == 1
    assert len(captures) == 1
    assert {Path(f"steps/{name}.py") for name in ("ingest", "split", "train", "evaluate")} <= set(
        captures[0]
    )
    (run_dir,) = (project / ".mlef/runs").iterdir()
    parent = read_run(run_dir / "run.json")
    record = read_pipeline(run_dir / "artifacts/pipeline")
    assert record.state == "failed"
    assert all(attempt.state == "skipped" for attempt in record.attempts.values())
    assert parent.inputs is not None
    assert parent.inputs.source_sha256 == record.source_sha256
    environment = Environment.model_validate_json((run_dir / "environment.json").read_text())
    assert environment.sdk_wheel == record.sdk_wheel
    assert file_hash(run_dir / "sdk" / record.sdk_wheel.filename) == record.sdk_wheel.sha256


def test_pipeline_requires_locked_environment_before_docker(project: Path) -> None:
    initialize(project)
    assert run(project, Path("mlef.toml"), [], [], [], _identifier=str(uuid4())) == 1
    (run_dir,) = (project / ".mlef/runs").iterdir()
    parent = read_run(run_dir / "run.json")
    assert parent.error is not None
    assert "Pipelines require pyproject.toml and uv.lock" in parent.error.message
    assert not (run_dir / "sdk").exists()


def test_cli_defaults_to_pipeline_and_rejects_script_only_options(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize(project)
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr("mlef.cli.run", lambda *args: calls.append(args) or 0)
    assert main(["run"]) == 0
    assert calls == [(project, Path("mlef.toml"), [], [], [])]
    with pytest.raises(ValueError, match="do not accept"):
        run(project, Path("mlef.toml"), ["--flag"], [], [])
    with pytest.raises(ValueError, match="do not accept"):
        run(project, Path("mlef.toml"), [], [], [Path("x")])
    with pytest.raises(ValueError, match="do not accept"):
        run(project, Path("mlef.toml"), [], [], [], Path("spec.json"))


def test_managed_context_requires_an_assignment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLEF_STEP_CONTEXT", raising=False)
    step = Step.model_validate_json(
        '{"name":"demo","outputs":[{"name":"value","type":"file","format":"text/plain"}]}'
    )
    with pytest.raises(ValueError, match="No pipeline context"):
        step.context()


@pytest.mark.parametrize("outcome", ["timeout", "cancelled", "recovery"])
def test_interrupted_pipeline_preserves_partial_attempt(
    project: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    setup(project)
    directory = prepare(project)
    run_dir = directory.parents[1]
    record = read_pipeline(directory)
    record.state = "running"
    record.attempts["ingest"].state = "running"
    _save(directory, record)
    context = LocalContext(
        directory=attempt_directory(directory, "ingest"),
        step=record.contracts["ingest"],
        run_id=record.run_id,
    ).prepare()
    context.output_path("data", "partial").write_text("keep")
    parent = read_run(run_dir / "run.json")
    parent.state = "running"
    save(run_dir / "run.json", parent)
    monkeypatch.setattr(supervision, "cleanup", lambda *args: True)
    if outcome == "recovery":
        monkeypatch.setattr(recovery, "cleanup", lambda *args: True)
        monkeypatch.setattr(recovery, "query", lambda *args: "server")
        recovery.recover(project, record.run_id)
    else:

        def worker(path: Path) -> subprocess.Popen[bytes]:
            process = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                start_new_session=os.name != "nt",
            )
            if outcome == "cancelled":
                wait = process.wait
                interrupted = False

                def interrupt(timeout: float | None = None) -> int:
                    nonlocal interrupted
                    if not interrupted:
                        interrupted = True
                        raise KeyboardInterrupt
                    return wait(timeout)

                monkeypatch.setattr(process, "wait", interrupt)
            return process

        monkeypatch.setattr(supervision, "start_worker", worker)
        assert parent.runtime is not None
        request = supervision.Request(
            root=str(project),
            script="mlef.toml",
            arguments=[],
            includes=[],
            outputs=[],
            experiment=None,
            identifier=record.run_id,
            token=parent.runtime.token,
            timeout=0.15 if outcome == "timeout" else 5,
        )
        assert supervision.supervise(request) == (130 if outcome == "cancelled" else 1)
    saved = read_pipeline(directory)
    assert saved.state == ("cancelled" if outcome == "cancelled" else "failed")
    assert saved.attempts["ingest"].state == saved.state
    assert all(saved.attempts[name].state == "skipped" for name in saved.order[1:])
    assert (context.directory / "outputs/data/partial").read_text() == "keep"
    with pytest.raises(ValueError, match="incomplete"):
        read_step(context.directory)
