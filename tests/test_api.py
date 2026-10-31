import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import pytest

from mlef import api, supervision
from mlef.cli import main
from mlef.evidence import normalize
from mlef.records import Dataset, Evaluation, Experiment, RunRecord
from mlef.runner import save


def test_offline_api_matches_cli(
    project: Path,
    cli: Callable[..., subprocess.CompletedProcess[str]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    api.init()
    assert capsys.readouterr().out == ""
    identifiers = [str(uuid4()), str(uuid4())]
    for index, identifier in enumerate(identifiers):
        directory = project / ".mlef/runs" / identifier
        (directory / "artifacts").mkdir(parents=True)
        record = RunRecord(
            id=identifier,
            state="succeeded",
            command=["python", "train.py"],
            created_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
            execution_exit_code=0,
            experiment=Experiment(
                dataset=Dataset(fingerprint="dataset"),
                evaluation=Evaluation(
                    protocol_id="holdout",
                    split_fingerprint=f"split-{index}",
                    seeds={},
                    metric_definitions={"score": "accuracy"},
                ),
                metric_directions={"score": "maximize"},
            ),
        ).model_dump(mode="json")
        (directory / "run.json").write_text(json.dumps(record))
        (directory / "artifacts/metrics.json").write_text('{"score":0.5}')
        save(directory / "result.json", normalize(project, identifier))
    first, second = identifiers
    assert api.status(first).model_dump(mode="json") == json.loads(
        cli("status", first, "--json").stdout
    )
    listing = api.list_runs()
    assert [record.model_dump(mode="json") for record in listing.runs] == json.loads(
        cli("list", "--json").stdout
    )
    assert listing.diagnostics == []
    assert api.artifacts(first) == ["metrics.json"]
    retrieved = api.get(first, "metrics.json", output=project / "retrieved.json")
    assert json.loads(retrieved.read_text()) == {"score": 0.5}
    comparison = api.compare(first, second)
    assert comparison.compatibility == "incompatible"
    assert comparison.model_dump(mode="json") == json.loads(
        cli("compare", first, second, "--json").stdout
    )
    assert api.recover(first).state == "succeeded"


def test_api_errors_are_quiet_and_categorized(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(api.MlefError) as failure:
        api.list_runs()
    assert failure.value.category == "not_initialized"
    api.init()
    with pytest.raises(api.MlefError) as failure:
        api.status(str(uuid4()))
    assert failure.value.category == "unknown_run"
    with pytest.raises(api.MlefError) as failure:
        api.run("missing.py")
    assert failure.value.category == "configuration"
    assert list((project / ".mlef/runs").iterdir()) == []
    assert capsys.readouterr() == ("", "")


def test_api_rejects_nested_root(project: Path) -> None:
    nested = project / "nested"
    nested.mkdir()
    with pytest.raises(api.MlefError) as failure:
        api.init(root=nested)
    assert failure.value.category == "configuration"
    assert not (nested / ".mlef").exists()


def test_stalled_project_lookup_is_a_configuration_error(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def stalled(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(["git", "rev-parse"], 10)

    monkeypatch.setattr("mlef.project.subprocess.run", stalled)
    with pytest.raises(api.MlefError) as failure:
        api.init()
    assert failure.value.category == "configuration"
    assert main(["init"]) == 2


def test_failed_callback_stops_worker_and_preserves_record(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api.init()
    processes: list[subprocess.Popen[bytes]] = []

    def worker(path: Path) -> subprocess.Popen[bytes]:
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", "import time; print('progress'); time.sleep(60)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            start_new_session=os.name != "nt",
        )
        processes.append(process)
        return process

    def failed(event: api.LogEvent) -> None:
        raise ValueError("caller callback failed")

    monkeypatch.setattr(supervision, "start_worker", worker)
    monkeypatch.setattr(supervision, "cleanup", lambda *args: True)
    with pytest.raises(api.MlefError) as failure:
        api.run("train.py", on_output=failed, options=api.RunOptions(timeout_seconds=10))
    assert failure.value.category == "callback"
    assert processes[0].poll() is not None
    (record,) = api.list_runs().runs
    assert record.error is not None
    assert record.error.category == "callback"
    assert record.state == "failed"


def test_api_interrupt_returns_cancelled_evidence(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api.init()

    def worker(path: Path) -> subprocess.Popen[bytes]:
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            start_new_session=os.name != "nt",
        )
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
    monkeypatch.setattr(supervision, "cleanup", lambda *args: True)
    record = api.run("train.py")
    assert record.state == "cancelled"
    assert record.error is not None
    assert record.error.category == "cancelled"
    assert record.result.evidence_status == "missing"


@pytest.mark.skipif(os.environ.get("MLEF_TEST_DOCKER") != "1", reason="Set MLEF_TEST_DOCKER=1")
@pytest.mark.parametrize("outcome", ["success", "failure", "invalid", "timeout"])
def test_real_api_outcomes_and_bounded_delivery(
    project: Path,
    monkeypatch: pytest.MonkeyPatch,
    cli: Callable[..., subprocess.CompletedProcess[str]],
    capsys: pytest.CaptureFixture[str],
    outcome: str,
) -> None:
    api.init()
    (project / "train.py").write_text(
        "import os,sys,time\nfrom pathlib import Path\n"
        "out = Path(os.environ['MLEF_OUTPUT_DIR'])\n"
        "(out / 'partial').write_text('kept')\n"
        "metrics = 'invalid' if sys.argv[1]=='invalid' else '{\"score\":1}'\n"
        "(out / 'metrics.json').write_text(metrics)\n"
        "print('x'*10000, flush=True)\n"
        "if sys.argv[1]=='timeout': time.sleep(120)\n"
        "sys.exit(7 if sys.argv[1]=='failure' else 0)\n"
    )
    (project / "experiment.json").write_text(Experiment(hypothesis="API smoke").model_dump_json())
    events: list[api.LogEvent] = []
    monkeypatch.chdir(project.parent)
    record = api.run(
        "train.py",
        root=project,
        arguments=[outcome],
        experiment="experiment.json",
        options=api.RunOptions(timeout_seconds=15 if outcome == "timeout" else 90),
        on_output=events.append,
        callback_bytes=73,
    )
    assert capsys.readouterr() == ("", "")
    assert sum(len(event.data) for event in events) <= 73
    assert any("Callback byte budget" in warning for warning in record.warnings)
    assert record.experiment is not None
    assert record.experiment.hypothesis == "API smoke"
    assert "partial" in api.artifacts(record.id, root=project)
    assert record.state == ("succeeded" if outcome in {"success", "invalid"} else "failed")
    if outcome == "timeout":
        assert record.error is not None
        assert record.error.category == "timeout"
    else:
        assert record.execution_exit_code == (7 if outcome == "failure" else 0)
        assert record.result.evidence_status == ("invalid" if outcome == "invalid" else "available")
    monkeypatch.chdir(project)
    # Callback-delivery warnings describe this call; stored status matches CLI exactly.
    assert api.status(record.id).model_dump(mode="json") == json.loads(
        cli("status", record.id, "--json").stdout
    )
