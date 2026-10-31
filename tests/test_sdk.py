"""Local artifact saves preserve evidence and verify bytes before handoff."""

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from mlef import (
    Artifact,
    ArtifactSource,
    ArtifactType,
    Input,
    LocalContext,
    Output,
    Step,
    StepResult,
    load_artifacts,
    read_step,
)
from mlef.artifacts import Column, DatasetMetadata


def declaration(name: str = "ingest", *, inputs: list[Input] | None = None) -> Step:
    return Step(
        name=name,
        inputs=inputs or [],
        outputs=[Output(name="data", type=ArtifactType.FILE, format="text")],
    )


def payload(context: LocalContext, text: str = "hello") -> Artifact:
    path = context.output_path("data", "data.txt")
    path.write_text(text, encoding="utf-8")
    return Artifact(name="data", type=ArtifactType.FILE, format="text", path=path)


def saved(directory: Path) -> ArtifactSource:
    step = declaration()
    with step.local(directory) as context:
        step.save([payload(context)], context=context)
    return load_artifacts(directory)["data"]


def state(directory: Path) -> str:
    return str(json.loads((directory / "step.json").read_text(encoding="utf-8"))["state"])


def test_save_file_manifest_and_verified_handoff(tmp_path: Path) -> None:
    source = saved(tmp_path / "producer")
    record = read_step(source.directory)
    assert StepResult.model_validate_json(record.model_dump_json()) == record
    manifest = record.artifacts["data"]
    assert manifest.producer is not None
    assert manifest.producer.sha256 == manifest.content_sha256()
    step = declaration("train", inputs=[Input(name="raw", type=ArtifactType.FILE, format="text")])
    with step.local(tmp_path / "consumer", inputs={"raw": source}) as context:
        path = context.input_path("raw", "data.txt")
        assert path != source.directory / "outputs/data/data.txt"
        assert path.read_text(encoding="utf-8") == "hello"
        assert not path.stat().st_mode & stat.S_IWUSR
        assert context.input_reference("raw") == manifest.producer
        with pytest.raises(ValueError, match="Unknown input payload"):
            context.input_path("raw", "missing.txt")
        result = step.save([payload(context, path.read_text().upper())], context=context)
        assert result.state == "running"
        with pytest.raises(ValueError, match="incomplete"):
            read_step(context.directory)
    assert result.inputs["raw"] == manifest.producer
    assert (tmp_path / "consumer/outputs/data/data.txt").read_text() == "HELLO"
    assert (source.directory / "outputs/data/data.txt").read_text() == "hello"


def test_directory_artifact_and_canonical_inventory(tmp_path: Path) -> None:
    step = Step(
        name="bundle", outputs=[Output(name="data", type=ArtifactType.DIRECTORY, format="files")]
    )
    with step.local(tmp_path / "bundle") as context:
        root = context.output_path("data")
        context.output_path("data", "nested/B.txt").write_text("upper")
        context.output_path("data", "nested/a.txt").write_text("lower")
        result = step.save(
            [Artifact(name="data", type=ArtifactType.DIRECTORY, format="files", path=root)],
            context=context,
        )
    assert [item.path for item in result.artifacts["data"].files] == [
        "nested/B.txt",
        "nested/a.txt",
    ]
    consumer = declaration("consume", inputs=[Input(name="raw", type=ArtifactType.DIRECTORY)])
    with consumer.local(
        tmp_path / "consumer", inputs={"raw": load_artifacts(tmp_path / "bundle")["data"]}
    ) as context:
        assert (context.input_path("raw") / "nested/B.txt").read_text() == "upper"
        consumer.save([payload(context)], context=context)


def test_optional_serializer_and_optional_output(tmp_path: Path) -> None:
    step = declaration()

    def serialize(path: Path) -> None:
        path.write_text("serialized", encoding="utf-8")

    with step.local(tmp_path / "serializer") as context:
        artifact = Artifact(
            name="data",
            type=ArtifactType.FILE,
            format="text",
            path=context.output_path("data", "generated.txt"),
            serializer=serialize,
        )
        step.save([artifact], context=context)
    assert (tmp_path / "serializer/outputs/data/generated.txt").read_text() == "serialized"
    optional = Step(
        name="maybe",
        outputs=[Output(name="data", type=ArtifactType.FILE, format="text", required=False)],
    )
    with optional.local(tmp_path / "optional") as context:
        assert optional.save([], context=context).artifacts == {}


@pytest.mark.parametrize(
    "failure",
    [
        "duplicate",
        "undeclared",
        "missing",
        "type",
        "format",
        "absent",
        "empty",
        "outside",
        "extra",
        "sibling",
    ],
)
def test_invalid_outputs_leave_nonreusable_failure_evidence(tmp_path: Path, failure: str) -> None:
    step = declaration()
    context = step.local(tmp_path / "attempt")
    artifact = payload(context)
    artifacts = [artifact]
    if failure == "duplicate":
        artifacts.append(artifact)
    elif failure == "undeclared":
        artifact.name = "other"
    elif failure == "missing":
        artifacts = []
    elif failure == "type":
        artifact.type = ArtifactType.MODEL
    elif failure == "format":
        artifact.format = "csv"
    elif failure == "absent":
        artifact.path.unlink()
    elif failure == "empty":
        artifact.path.unlink()
        artifact.path = context.output_path("data")
    elif failure == "outside":
        artifact.path = tmp_path / "secret.txt"
        artifact.path.write_text("never publish")
    elif failure == "extra":
        context.output_path("data", "extra.txt").write_text("undeclared bytes")
    else:
        extra = context.directory / "outputs/other"
        extra.mkdir()
        (extra / "data.txt").write_text("undeclared output")
    messages = {
        "duplicate": "distinct",
        "undeclared": "distinct",
        "missing": "Missing required",
        "type": "type or format",
        "format": "type or format",
        "absent": "payload is missing",
        "empty": "require a file payload",
        "outside": "escapes",
        "extra": "additional payload",
        "sibling": "directories",
    }
    with pytest.raises(ValueError, match=messages[failure]):
        step.save(artifacts, context=context)
    assert state(context.directory) == "failed"
    with pytest.raises(ValueError, match="Failed or incomplete"):
        load_artifacts(context.directory)
    assert (context.directory / "outputs/data").is_dir()


def test_failed_serializer_retains_partial_payload(tmp_path: Path) -> None:
    step = declaration()
    context = step.local(tmp_path / "attempt")

    def fail(path: Path) -> None:
        path.write_text("partial evidence")
        raise RuntimeError("serializer failed")

    artifact = Artifact(
        name="data",
        type=ArtifactType.FILE,
        format="text",
        path=context.output_path("data", "partial.txt"),
        serializer=fail,
    )
    with pytest.raises(RuntimeError, match="serializer failed"):
        step.save([artifact], context=context)
    assert state(context.directory) == "failed"
    assert artifact.path.read_text() == "partial evidence"


def test_directory_shape_and_empty_payload_are_validated(tmp_path: Path) -> None:
    step = Step(
        name="directory",
        outputs=[Output(name="data", type=ArtifactType.DIRECTORY, format="files")],
    )
    context = step.local(tmp_path / "file")
    path = context.output_path("data", "file.txt")
    path.write_text("not a directory")
    artifact = Artifact(name="data", type=ArtifactType.DIRECTORY, format="files", path=path)
    with pytest.raises(ValueError, match="require a directory"):
        step.save([artifact], context=context)
    context = step.local(tmp_path / "empty")
    artifact.path = context.output_path("data")
    with pytest.raises(ValidationError, match="at least 1 item"):
        step.save([artifact], context=context)


def test_optional_outputs_cannot_hide_an_invalid_storage_root(tmp_path: Path) -> None:
    step = Step(
        name="optional",
        outputs=[Output(name="data", type=ArtifactType.FILE, format="text", required=False)],
    )
    context = step.local(tmp_path / "attempt")
    (context.directory / "outputs").write_text("not a directory")
    with pytest.raises(ValueError, match="storage must be a directory"):
        step.save([], context=context)
    assert state(context.directory) == "failed"


def test_serializer_cannot_overwrite_existing_payload(tmp_path: Path) -> None:
    context = declaration().local(tmp_path / "attempt")
    artifact = payload(context)
    artifact.serializer = lambda path: None
    with pytest.raises(ValueError, match="overwrite"):
        context.step.save([artifact], context=context)
    assert artifact.path.read_text() == "hello"


@pytest.mark.parametrize("target", ["staged", "source"])
def test_input_mutation_prevents_finalization(tmp_path: Path, target: str) -> None:
    source = saved(tmp_path / "producer")
    step = declaration("train", inputs=[Input(name="raw", type=ArtifactType.FILE)])
    context = step.local(tmp_path / "consumer", inputs={"raw": source})
    path = (
        context.input_path("raw", "data.txt")
        if target == "staged"
        else source.directory / "outputs/data/data.txt"
    )
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    path.write_text("tampered")
    with pytest.raises(ValueError, match="changed"):
        step.save([payload(context)], context=context)
    assert state(context.directory) == "failed"


def test_late_serializer_mutation_is_detected(tmp_path: Path) -> None:
    step = Step(
        name="two",
        outputs=[
            Output(name=name, type=ArtifactType.FILE, format="text") for name in ["first", "second"]
        ],
    )
    context = step.local(tmp_path / "attempt")
    first = context.output_path("first", "data.txt")
    first.write_text("original")

    def mutate(path: Path) -> None:
        first.write_text("mutated")
        path.write_text("second")

    artifacts = [
        Artifact(name="first", type=ArtifactType.FILE, format="text", path=first),
        Artifact(
            name="second",
            type=ArtifactType.FILE,
            format="text",
            path=context.output_path("second", "data.txt"),
            serializer=mutate,
        ),
    ]
    with pytest.raises(ValueError, match="changed"):
        step.save(artifacts, context=context)
    assert state(context.directory) == "failed"


@pytest.mark.parametrize("change", ["bytes", "missing", "added"])
def test_loading_rechecks_output_bytes(tmp_path: Path, change: str) -> None:
    source = saved(tmp_path / "producer")
    path = source.directory / "outputs/data/data.txt"
    if change == "bytes":
        path.write_text("changed")
    elif change == "missing":
        path.unlink()
    else:
        (path.parent / "extra.txt").write_text("extra")
    with pytest.raises(ValueError, match="changed"):
        load_artifacts(source.directory)


def test_incomplete_attempt_and_missing_context_save_are_not_reusable(tmp_path: Path) -> None:
    context = declaration().local(tmp_path / "incomplete")
    payload(context)
    assert state(context.directory) == "running"
    with pytest.raises(ValueError, match="incomplete"):
        read_step(context.directory)
    with (
        pytest.raises(ValueError, match="without saving"),
        declaration().local(tmp_path / "unsaved"),
    ):
        pass
    assert state(tmp_path / "unsaved") == "failed"


def test_failure_and_mutation_after_save_inside_context_invalidate_completion(
    tmp_path: Path,
) -> None:
    step = declaration()

    def execute(*, mutate: bool) -> None:
        with step.local(tmp_path / ("mutation" if mutate else "exception")) as context:
            artifact = payload(context)
            step.save([artifact], context=context)
            if mutate:
                artifact.path.write_text("late edit")
            else:
                raise RuntimeError("script failed")

    with pytest.raises(RuntimeError, match="script failed"):
        execute(mutate=False)
    assert state(tmp_path / "exception") == "failed"
    with pytest.raises(ValueError, match="changed"):
        execute(mutate=True)
    assert state(tmp_path / "mutation") == "failed"


def test_existing_attempts_and_finished_contexts_are_never_overwritten(tmp_path: Path) -> None:
    step = declaration()
    directory = tmp_path / "attempt"
    context = step.local(directory)
    artifact = payload(context)
    step.save([artifact], context=context)
    original = (directory / "step.json").read_bytes()
    with pytest.raises(FileExistsError):
        step.local(directory)
    with pytest.raises(ValueError, match="finalized"):
        context.output_path("data", "new.txt")
    with pytest.raises(ValueError, match="finalized"):
        step.save([artifact], context=context)
    assert (directory / "step.json").read_bytes() == original


def test_step_and_nested_metadata_mutation_are_revalidated(tmp_path: Path) -> None:
    step = declaration()
    context = step.local(tmp_path / "step")
    artifact = payload(context)
    step.outputs.append(Output(name="later", type=ArtifactType.FILE, format="text", required=False))
    with pytest.raises(ValueError, match="declaration changed"):
        step.save([artifact], context=context)
    dataset = Step(
        name="dataset", outputs=[Output(name="data", type=ArtifactType.DATASET, format="csv")]
    )
    context = dataset.local(tmp_path / "metadata")
    metadata = DatasetMetadata(rows=1, columns=[Column(name="id", dtype="int")], row_id=["id"])
    path = context.output_path("data", "data.csv")
    path.write_text("id\n1\n")
    artifact = Artifact(
        name="data", type=ArtifactType.DATASET, format="csv", path=path, metadata=metadata
    )
    metadata.columns.append(Column(name="id", dtype="int"))
    with pytest.raises(ValidationError, match="unique"):
        dataset.save([artifact], context=context)
    assert state(context.directory) == "failed"


@pytest.mark.parametrize("binding", ["missing", "unknown", "incompatible", "source_name"])
def test_invalid_input_bindings_fail_before_script_work(tmp_path: Path, binding: str) -> None:
    source = saved(tmp_path / "producer")
    inputs = {"raw": source}
    expected_type = ArtifactType.FILE
    if binding == "missing":
        inputs = {}
    elif binding == "unknown":
        inputs["other"] = source
    elif binding == "incompatible":
        expected_type = ArtifactType.DATASET
    else:
        inputs["raw"] = ArtifactSource(directory=source.directory, name="unknown")
    step = declaration("consumer", inputs=[Input(name="raw", type=expected_type)])
    with pytest.raises(ValueError, match=r"bindings|Incompatible|Unknown source"):
        step.local(tmp_path / "consumer", inputs=inputs)
    assert state(tmp_path / "consumer") == "failed"


def test_paths_and_optional_inputs_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    step = declaration(inputs=[Input(name="optional", type=ArtifactType.FILE, required=False)])
    context = step.local(Path("attempt"))
    for path in ["../escape", "/absolute", "a\\b", "a//b", "C:/escape"]:
        with pytest.raises(ValueError, match="normalized"):
            context.output_path("data", path)
    with pytest.raises(ValueError, match="Undeclared"):
        context.output_path("unknown")
    with pytest.raises(ValueError, match="Unknown bound"):
        context.input_path("optional")
    monkeypatch.chdir(tmp_path.parent)
    step.save([payload(context)], context=context)
    assert read_step(tmp_path / "attempt").state == "complete"


def test_symlinks_are_rejected(tmp_path: Path) -> None:
    context = declaration().local(tmp_path / "attempt")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    path = context.output_path("data", "data.txt")
    try:
        path.symlink_to(outside)
    except OSError:
        pytest.skip("Symlink creation requires platform privileges.")
    with pytest.raises(ValueError, match="symlink"):
        context.step.save(
            [Artifact(name="data", type=ArtifactType.FILE, format="text", path=path)],
            context=context,
        )


def test_hardlinks_are_rejected(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    other = declaration().local(tmp_path / "hardlink")
    target = other.output_path("data", "data.txt")
    os.link(outside, target)
    with pytest.raises(ValueError, match="hard links"):
        other.step.save(
            [Artifact(name="data", type=ArtifactType.FILE, format="text", path=target)],
            context=other,
        )


def test_hardlinked_lock_cannot_modify_external_file(tmp_path: Path) -> None:
    context = declaration().local(tmp_path / "attempt")
    artifact = payload(context)
    outside = tmp_path / "outside.lock"
    outside.touch()
    os.link(outside, context.directory / "active.lock")
    with pytest.raises(ValueError, match="hard links"):
        context.step.save([artifact], context=context)
    assert outside.read_bytes() == b""


def test_atomic_final_record_failure_keeps_attempt_nonreusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = declaration().local(tmp_path / "attempt")
    artifact = payload(context)
    replace = Path.replace

    def fail_complete(source: Path, target: Path) -> Path:
        if target.name == "step.json" and json.loads(source.read_text())["state"] == "complete":
            raise OSError("publication failed")
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_complete)
    with pytest.raises(OSError, match="publication failed"):
        context.step.save([artifact], context=context)
    assert state(context.directory) == "failed"
    assert artifact.path.exists()
    assert not list(context.directory.glob(".step-*.tmp"))
    with pytest.raises(ValueError, match="incomplete"):
        read_step(context.directory)


@pytest.mark.parametrize("crash", [False, True])
def test_root_import_and_script_handoff_need_no_ml_dependencies(
    tmp_path: Path, crash: bool
) -> None:
    path = tmp_path / "script.py"
    path.write_text(
        "import os\nimport sys\nfrom pathlib import Path\n"
        "from mlef import Artifact, ArtifactType, Output, Step\n"
        "assert not {'pandas', 'numpy', 'sklearn', 'xgboost'} & sys.modules.keys()\n"
        "step = Step(name='script', outputs=[\n"
        "    Output(name='data', type=ArtifactType.FILE, format='text')])\n"
        "with step.local(Path('attempt')) as context:\n"
        "    path = context.output_path('data', 'data.txt')\n"
        "    path.write_text('subprocess')\n"
        "    artifact = Artifact(name='data', type=ArtifactType.FILE, format='text', path=path)\n"
        "    step.save([artifact], context=context)\n"
        "    if '--crash' in sys.argv:\n"
        "        os._exit(7)\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(path), *(["--crash"] if crash else [])],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == (7 if crash else 0), result.stderr
    if crash:
        assert state(tmp_path / "attempt") == "running"
        with pytest.raises(ValueError, match="incomplete"):
            read_step(tmp_path / "attempt")
    else:
        assert read_step(tmp_path / "attempt").step.name == "script"


def test_local_handoff_example_is_runnable_and_non_destructive(tmp_path: Path) -> None:
    example = Path(__file__).resolve().parents[1] / "examples/artifacts/local_handoff.py"
    command = [sys.executable, str(example), str(tmp_path / "demo")]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    directory = tmp_path / "demo/transform"
    assert read_step(directory).state == "complete"
    path = directory / "outputs/report/report.txt"
    original = path.read_bytes()
    assert original == b"ORDINARY SCRIPTS, REPRODUCIBLE EVIDENCE"
    repeated = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert repeated.returncode != 0
    assert "FileExistsError" in repeated.stderr
    assert path.read_bytes() == original
