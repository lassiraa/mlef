import csv
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from uuid import uuid4

import pytest

from mlef import Metrics, Step, api, recovery, sdk_distribution, supervision
from mlef.cli import main
from mlef.environment import Environment, build_recipe, file_hash
from mlef.execution import (
    _save,
    attempt_directory,
    execute_pipeline,
    prepare_pipeline,
    read_pipeline,
)
from mlef.pipeline import validate
from mlef.project import initialize
from mlef.provenance import capture, extract_source
from mlef.records import RunRecord, read_run
from mlef.runner import now, run, save, source_files
from mlef.sdk import LocalContext, read_step
from mlef.sdk_distribution import capture_sdk

LOCKED = Path(__file__).resolve().parents[1] / "examples/pipeline"


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
