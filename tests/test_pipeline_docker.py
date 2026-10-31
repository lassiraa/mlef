"""Captured pipelines through the real supervised Docker lifecycle."""

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import pytest

from mlef import api, supervision
from mlef.environment import Environment, file_hash
from mlef.execution import attempt_directory, read_pipeline
from mlef.project import initialize
from mlef.records import read_run
from mlef.sdk import read_step

LOCKED = Path(__file__).resolve().parents[1] / "examples/pipeline"
pytestmark = pytest.mark.skipif(
    os.environ.get("MLEF_TEST_DOCKER") != "1", reason="Set MLEF_TEST_DOCKER=1"
)


def setup(project: Path) -> None:
    initialize(project)
    for name in ("pyproject.toml", "uv.lock"):
        (project / name).write_bytes((LOCKED / name).read_bytes())


def test_real_frozen_multi_model_evaluation(
    project: Path,
    evaluation_project: Callable[[str], tuple[Path, list[str]]],
) -> None:
    trained = api.run()
    assert trained.state == "succeeded"
    definition, models = evaluation_project(trained.id)
    batch = api.evaluate(definition, models)
    assert batch.state == "complete", batch.model_dump_json()
    assert len(batch.evaluations) == 2
    assert [
        item.metrics[0].metrics.values["mae"].model_dump()["value"] for item in batch.evaluations
    ] == [1, 0]
    for item in batch.evaluations:
        assert api.pipeline_status(item.run_id, verify=True).order == ["evaluate"]
        context = json.loads((project / ".mlef/runs" / item.run_id / "evaluation.json").read_text())
        assert context["protocol_id"] == batch.protocol_id
        assert context["model"]["version"] == item.model.version
    table = api.evaluation_leaderboard(
        batch.id, baseline=batch.evaluations[0].run_id, metric="mae", scope="test"
    )
    assert table.rows[0].run_id == batch.evaluations[1].run_id, table.model_dump_json()
    assert table.rows[0].rank == 1
    assert table.rows[0].observation == "improved"


def test_real_cross_run_evaluation_without_training(project: Path) -> None:
    setup(project)
    trained = api.run()
    assert trained.state == "succeeded"
    directory = project / ".mlef/runs" / trained.id / "artifacts/pipeline"
    lines = ["schema_version = 2", "[[steps]]", 'name = "evaluate"', 'script = "steps/evaluate.py"']
    for binding, producer, output in [
        ("model", "train", "model"),
        ("partitions", "split", "partitions"),
    ]:
        reference = read_step(attempt_directory(directory, producer)).artifacts[output].producer
        assert reference is not None
        lines.extend(
            [
                f"[steps.bindings.{binding}]",
                f'run_id = "{trained.id}"',
                f"[steps.bindings.{binding}.artifact]",
            ]
        )
        lines.extend(
            f"{key} = {json.dumps(value)}" for key, value in reference.model_dump().items()
        )
    (project / "evaluation.toml").write_text("\n".join(lines))
    evaluated = api.run("evaluation.toml")
    assert evaluated.state == "succeeded"
    saved = api.pipeline_status(evaluated.id, verify=True)
    assert saved.order == ["evaluate"]
    evidence = api.run_metrics(evaluated.id, step="evaluate")
    assert evidence.artifacts[0].metrics.model is not None
    assert evidence.artifacts[0].metrics.model.run_id == trained.id
    assert evidence.artifacts[0].metrics.values["mae"].model_dump()["value"] == 3


def test_real_cache_hits_and_forced_dependent_execution(project: Path) -> None:
    setup(project)
    definition = project / "mlef.toml"
    definition.write_text(
        definition.read_text().replace(
            "script = ", 'cache = true\ncache_version = "public-v1"\nscript = '
        )
    )
    options = api.RunOptions(timeout_seconds=180)
    original = api.run(root=project, options=options)
    assert original.state == "succeeded"
    cached = api.run(root=project, options=options, cache_from=original.id)
    assert cached.state == "succeeded"
    record = api.pipeline_status(cached.id, root=project, verify=True)
    assert all(attempt.reused_from == original.id for attempt in record.attempts.values())
    forced = api.run(root=project, options=options, cache_from=cached.id, force=["train"])
    assert forced.state == "succeeded"
    record = api.pipeline_status(forced.id, root=project, verify=True)
    assert record.attempts["ingest"].reused_from == original.id
    assert record.attempts["train"].reused_from is None
    assert record.attempts["evaluate"].reused_from is None
    assert record.attempts["train"].cache is not None
    assert "Forced execution requested." in record.attempts["train"].cache.reasons


@pytest.mark.parametrize("resume", [False, True])
def test_real_explicit_reuse_and_captured_resume(project: Path, resume: bool) -> None:
    setup(project)
    ingest = project / "steps/ingest.py"
    ingest.write_text(ingest.read_text() + "\nprint('ingest executed')\n")
    options = api.RunOptions(timeout_seconds=180)
    original = api.run(root=project, options=options)
    assert original.state == "succeeded"
    previous = api.pipeline_status(original.id, root=project, verify=True)
    train = project / "steps/train.py"
    train.write_text(train.read_text().replace("mean(values)", "mean(values) + 10"))
    result = api.run(
        root=project, options=options, reuse_from=original.id, from_step="train", resume=resume
    )
    assert result.state == "succeeded"
    record = api.pipeline_status(result.id, root=project, verify=True)
    assert record.reuse_from == original.id
    assert record.resume is resume
    assert record.attempts["ingest"].reused_from == original.id
    assert record.attempts["split"].reused_from == original.id
    assert record.attempts["train"].reused_from is None
    assert (record.source_sha256 == previous.source_sha256) is resume
    directory = project / ".mlef/runs" / result.id
    assert "ingest executed" not in (directory / "stdout.log").read_text()
    model = (
        attempt_directory(directory / "artifacts/pipeline", "train") / "outputs/model/model.json"
    )
    prediction = json.loads(model.read_text())["prediction"]
    assert (prediction > 10) is (not resume)


@pytest.mark.parametrize("fail", [False, True])
def test_real_captured_pipeline_and_failed_step(project: Path, fail: bool) -> None:
    setup(project)
    if fail:
        script = project / "steps/train.py"
        script.write_text(script.read_text() + "\nraise RuntimeError('after saving')\n")
    result = api.run(root=project, options=api.RunOptions(timeout_seconds=180))
    assert result.state == ("failed" if fail else "succeeded")
    pipeline = api.pipeline_status(result.id, root=project, verify=True)
    run_dir = project / ".mlef/runs" / result.id
    directory = run_dir / "artifacts/pipeline"
    assert pipeline.state == ("failed" if fail else "complete")
    assert pipeline.attempts["ingest"].state == "complete"
    assert pipeline.attempts["split"].state == "complete"
    environment = Environment.model_validate_json((run_dir / "environment.json").read_text())
    assert environment.sdk_wheel == pipeline.sdk_wheel
    assert environment.container is not None
    assert environment.container.packages["mlef"] == pipeline.sdk_wheel.version
    assert file_hash(run_dir / "sdk" / pipeline.sdk_wheel.filename) == pipeline.sdk_wheel.sha256
    assert result.inputs is not None
    assert file_hash(run_dir / "source.tar.gz") == result.inputs.source_sha256
    assert api.status(result.id, root=project).state == result.state
    assert "pipeline/pipeline.json" in api.artifacts(result.id, root=project)
    if fail:
        assert pipeline.attempts["train"].state == "failed"
        assert pipeline.attempts["train"].exit_code == 1
        assert pipeline.attempts["evaluate"].state == "skipped"
        assert not attempt_directory(directory, "evaluate").exists()
        with pytest.raises(ValueError, match="incomplete"):
            read_step(attempt_directory(directory, "train"))
    else:
        assert read_step(attempt_directory(directory, "evaluate")).state == "complete"
    assert result.runtime is not None
    assert not result.runtime.cleanup_pending


@pytest.mark.parametrize("outcome", ["cancel", "recover"])
@pytest.mark.skipif(os.name == "nt", reason="This lifecycle test uses POSIX process signals.")
def test_real_pipeline_cancellation_and_abandoned_worker_recovery(
    project: Path, outcome: str
) -> None:
    setup(project)
    script = project / "steps/ingest.py"
    script.write_text(
        script.read_text() + "\nimport time\nprint('waiting', flush=True)\ntime.sleep(120)\n"
    )
    if outcome == "cancel":
        command = [sys.executable, "-m", "mlef", "run"]
    else:
        request = supervision.Request(
            root=str(project),
            script="mlef.toml",
            arguments=[],
            includes=[],
            outputs=[],
            experiment=None,
            identifier=str(uuid4()),
            token=str(uuid4()),
        )
        path = project / ".mlef/abandoned-request.json"
        path.write_text(request.model_dump_json())
        command = [sys.executable, "-m", "mlef.supervision", str(path)]
    process = subprocess.Popen(
        command,
        cwd=project,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 150
        while not list(
            (project / ".mlef/runs").glob(
                "*/artifacts/pipeline/steps/ingest/1/outputs/data/data.json"
            )
        ):
            if process.poll() is not None:
                pytest.fail(str(process.communicate()))
            if time.monotonic() > deadline:
                pytest.fail("Pipeline did not produce a pending artifact.")
            time.sleep(0.1)
        (run_dir,) = (project / ".mlef/runs").iterdir()
        directory = run_dir / "artifacts/pipeline"
        with pytest.raises(ValueError, match="incomplete"):
            read_step(attempt_directory(directory, "ingest"))
        if outcome == "cancel":
            os.kill(process.pid, signal.SIGINT)
            process.communicate(timeout=40)
            assert process.returncode == 130
        else:
            supervision.kill_tree(process)
            process.communicate(timeout=10)
            api.recover(run_dir.name, root=project)
        record = read_run(run_dir / "run.json")
        pipeline = read_pipeline(directory)
        assert record.state == ("cancelled" if outcome == "cancel" else "failed")
        assert pipeline.state == ("cancelled" if outcome == "cancel" else "failed")
        assert pipeline.attempts["ingest"].state == pipeline.state
        assert all(
            attempt.state == "skipped"
            for name, attempt in pipeline.attempts.items()
            if name != "ingest"
        )
        assert record.runtime is not None
        assert not record.runtime.cleanup_pending
        assert (attempt_directory(directory, "ingest") / "outputs/data/data.json").exists()
        assert (
            json.loads((attempt_directory(directory, "ingest") / "step.json").read_text())["state"]
            == "failed"
        )
    finally:
        if process.poll() is None:
            if outcome == "cancel":
                os.kill(process.pid, signal.SIGINT)
                try:
                    process.communicate(timeout=40)
                except subprocess.TimeoutExpired:
                    supervision.kill_tree(process)
                    process.communicate(timeout=10)
            else:
                supervision.kill_tree(process)
                process.communicate(timeout=10)
