import json
from pathlib import Path
from uuid import uuid4

import pytest

from mlef import api
from mlef.annotations import RunLabels
from mlef.cli import main
from mlef.locking import file_lock, run_lock
from mlef.records import Experiment, RunRecord
from mlef.runner import run, save


def saved(project: Path) -> str:
    identifier = str(uuid4())
    directory = project / ".mlef/runs" / identifier
    directory.mkdir()
    save(
        directory / "run.json",
        RunRecord(
            id=identifier,
            state="succeeded",
            command=["python", "train.py"],
            created_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:00:01Z",
            execution_exit_code=0,
            experiment=Experiment(hypothesis="Captured intent"),
        ),
    )
    return identifier


def test_run_organization_preserves_captured_evidence(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    api.init(scaffold=False)
    identifier = saved(project)
    path = project / ".mlef/runs" / identifier / "run.json"
    evidence = path.read_bytes()
    assert api.run_annotations(identifier).updated_at is None
    assert not (project / ".mlef/annotations").exists()
    with run_lock(path.parent):
        labels = api.annotate_run(
            identifier,
            name="Baseline",
            experiment="Churn models",
            tags={"family": "linear"},
            note="Review me",
        )
    assert path.read_bytes() == evidence
    intent = api.status(identifier).experiment
    assert intent is not None
    assert intent.hypothesis == "Captured intent"
    group = api.experiment_group("Churn models")
    assert labels.experiment_id == group.id
    assert api.create_experiment("Churn models", description="Validation study").id == group.id
    assert api.create_experiment("Churn models").description == "Validation study"
    page = api.run_history(experiment="Churn models", tags={"family": "linear"}, name="Baseline")
    assert [record.id for record in page.runs] == [identifier]
    assert page.annotations[identifier] == labels
    assert page.schema_version == 2
    api.annotate_run(identifier, tags={"reviewed": "yes"}, remove_tags=["family"], archived=True)
    assert api.run_history().total == 0
    assert api.run_history(include_archived=True).total == 1
    assert len(api.list_runs().runs) == 1
    cleared = api.annotate_run(
        identifier, clear_name=True, clear_note=True, clear_experiment=True, archived=False
    )
    assert cleared.name is None
    assert cleared.note is None
    assert cleared.experiment_id is None
    assert cleared.tags == {"reviewed": "yes"}
    assert main(["runs", "annotate", identifier, "--tag", "family=tree", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == api.run_annotations(identifier).model_dump(
        mode="json"
    )
    assert main(["experiments", "show", "Churn models", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == api.experiment_details("Churn models").model_dump(
        mode="json"
    )
    assert path.read_bytes() == evidence


def test_metadata_lock_and_corrupt_records(project: Path) -> None:
    api.init(scaffold=False)
    identifier = saved(project)
    initial = api.annotate_run(identifier, tags={"family": "linear"})
    with file_lock(project / ".mlef/metadata.lock"), pytest.raises(api.MlefError) as error:
        api.annotate_run(identifier, tags={"reviewed": "yes"})
    assert error.value.category == "metadata_busy"
    assert api.run_annotations(identifier) == initial
    path = project / ".mlef/annotations" / f"{identifier}.json"
    invalid = initial.model_dump(mode="json")
    invalid["run_id"] = str(uuid4())
    path.write_text(json.dumps(invalid))
    listing = api.run_history()
    assert listing.total == 0
    assert listing.diagnostics
    with pytest.raises(api.MlefError, match="identity"):
        api.annotate_run(identifier, note="Do not overwrite corruption")


def test_group_names_are_portable_and_updates_validate(project: Path) -> None:
    api.init(scaffold=False)
    first = api.create_experiment("Churn")
    second = api.create_experiment("churn")
    assert first.id != second.id
    assert len(api.list_experiments().groups) == 2
    identifier = saved(project)
    with pytest.raises(api.MlefError):
        api.annotate_run(identifier, name="   ")
    with pytest.raises(api.MlefError, match="set and cleared"):
        api.annotate_run(identifier, name="baseline", clear_name=True)
    with pytest.raises(api.MlefError, match="set and removed"):
        api.annotate_run(identifier, tags={"a": "1"}, remove_tags=["a"])
    assert not (project / ".mlef/annotations").exists()


def test_preparation_failure_keeps_launch_labels(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api.init(scaffold=False)
    (project / "train.py").write_text("print('example')")
    monkeypatch.setattr("mlef.runner.shutil.which", lambda executable: None)
    identifier = str(uuid4())
    assert (
        run(
            project,
            Path("train.py"),
            [],
            [],
            [],
            _identifier=identifier,
            labels=RunLabels(name="failed baseline", experiment="churn", tags={"seed": "7"}),
        )
        == 1
    )
    labels = api.run_annotations(identifier)
    assert labels.name == "failed baseline"
    assert labels.experiment == "churn"
    assert labels.tags == {"seed": "7"}
    assert api.status(identifier).state == "failed"


def test_cli_launch_labels_and_duplicate_tags(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api.init(scaffold=False)
    calls = []

    def execute(*args: object, **kwargs: object) -> int:
        labels = kwargs["labels"]
        assert isinstance(labels, RunLabels)
        assert labels.name == "baseline"
        assert labels.experiment == "Churn models"
        assert labels.tags == {"seed": "7"}
        calls.append(labels)
        return 0

    monkeypatch.setattr("mlef.cli.run", execute)
    assert (
        main(
            [
                "run",
                "train.py",
                "--name",
                "baseline",
                "--experiment-name",
                "Churn models",
                "--tag",
                "seed=7",
            ]
        )
        == 0
    )
    assert len(calls) == 1
    assert main(["run", "train.py", "--tag", "seed=7", "--tag", "seed=8"]) == 2
    assert len(calls) == 1
