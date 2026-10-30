import json
import os
from pathlib import Path
from uuid import uuid4

import pytest

from mlef.cli import init, main
from mlef.records import RunRecord, read_run
from mlef.runner import save
from mlef.storage import QueryError, inspect, list_runs, retrieve


def saved_run(
    project: Path, *, state: str = "succeeded", created: str = "2026-01-01T00:00:00Z"
) -> Path:
    directory = project / ".mlef/runs" / str(uuid4())
    (directory / "artifacts").mkdir(parents=True)
    record = RunRecord.model_validate(
        {
            "id": directory.name,
            "state": state,
            "created_at": created,
            "command": ["python", "train.py"],
            "finished_at": None if state == "running" else created,
            "execution_exit_code": 0 if state == "succeeded" else None,
        }
    )
    save(directory / "run.json", record)
    return directory


def test_legacy_adapter_is_read_only(project: Path) -> None:
    path = saved_run(project) / "run.json"
    data = json.loads(path.read_text())
    data["schema_version"] = 1
    del data["experiment"]
    path.write_text(json.dumps(data))
    original = path.read_bytes()
    record = read_run(path)
    assert record.schema_version == 3
    assert record.git is record.inputs is record.experiment is None
    assert "Legacy" in record.warnings[-1]
    assert path.read_bytes() == original


def test_json_argument_errors_are_structured(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["status", "--json"])
    assert error.value.code == 2
    assert json.loads(capsys.readouterr().out)["error"]["category"] == "arguments"


def test_inconsistent_success_is_a_corrupt_record(project: Path) -> None:
    path = saved_run(project) / "run.json"
    record = json.loads(path.read_text())
    record["execution_exit_code"] = 7
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="exit code zero"):
        read_run(path)


@pytest.mark.parametrize(
    "content",
    [
        '{"schema_version": 99}',
        '{"schema_version": true}',
        '{"schema_version": 2, "schema_version": 1}',
        "[]",
        "broken",
    ],
)
def test_corrupt_records_are_reported_and_list_skips_them(
    project: Path, capsys: pytest.CaptureFixture[str], content: str
) -> None:
    directory = saved_run(project)
    (directory / "run.json").write_text(content)
    assert main(["status", directory.name, "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["category"] == "invalid_record"
    assert main(["list", "--json"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == []
    assert "Cannot read run" in captured.err


def test_sorted_offline_queries_and_liveness(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    init(project)
    capsys.readouterr()
    assert main(["list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []
    first = saved_run(project, state="failed")
    second = saved_run(project, state="running", created="2026-02-01T00:00:00Z")
    records, diagnostics = list_runs(project)
    assert [record.id for record in records] == [second.name, first.name]
    assert diagnostics == []
    assert "liveness is unknown" in inspect(project, second.name).warnings[-1]
    assert main(["status", first.name, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "failed"
    assert main(["status", str(uuid4()), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["category"] == "unknown_run"


def test_artifact_copy_never_overwrites_or_escapes(project: Path, tmp_path: Path) -> None:
    directory = saved_run(project)
    source = directory / "artifacts/nested/value.txt"
    source.parent.mkdir()
    source.write_text("saved")
    destination = tmp_path / "copy"
    retrieve(project, directory.name, "nested", destination)
    assert (destination / "value.txt").read_text() == "saved"
    with pytest.raises(QueryError, match="already exists"):
        retrieve(project, directory.name, "nested/value.txt", destination / "value.txt")
    for relative in ["../run.json", str(source), "C:/escape", "nested\\value.txt"]:
        with pytest.raises(QueryError, match="artifact-relative"):
            retrieve(project, directory.name, relative, tmp_path / "escape")
    with pytest.raises(QueryError, match="inside the source"):
        retrieve(project, directory.name, "nested", source.parent / "child")
    assert (destination / "value.txt").read_text() == "saved"


@pytest.mark.skipif(os.name == "nt", reason="Windows symlinks require developer mode or elevation")
def test_links_in_artifact_tree_are_rejected(project: Path, tmp_path: Path) -> None:
    directory = saved_run(project)
    (directory / "artifacts/link").symlink_to(project, target_is_directory=True)
    with pytest.raises(QueryError, match="link"):
        retrieve(project, directory.name, "link", tmp_path / "copy")
    with pytest.raises(QueryError, match="link"):
        retrieve(project, directory.name, ".", tmp_path / "copy")
