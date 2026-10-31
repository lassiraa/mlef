import json
import subprocess
import sys
from pathlib import Path

import pytest

from mlef import Metrics, api, read_step
from mlef.cli import main
from mlef.project import initialize

STEPS = ("ingest", "split", "train", "evaluate")


def execute(project: Path, name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(project / "steps" / f"{name}.py"), ".mlef/example"],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_init_scaffold_validates_without_execution(project: Path) -> None:
    assert main(["init"]) == 0
    report = api.validate(root=project)
    assert report.status == "verified"
    assert report.order == list(STEPS)
    assert set(report.contracts) == set(STEPS)
    assert not (project / ".mlef/example").exists()
    assert {path.name for path in (project / "steps").iterdir()} == {f"{name}.py" for name in STEPS}


@pytest.mark.parametrize("conflict", ["mlef.toml", "steps", "steps/"])
def test_scaffold_conflict_skips_entire_example(project: Path, conflict: str) -> None:
    path = project / conflict
    if conflict.endswith("/"):
        path.mkdir()
        (path / "mine.py").write_text("existing script", encoding="utf-8")
    else:
        path.write_text("existing file", encoding="utf-8")
    before = {
        item.relative_to(project): item.read_bytes()
        for item in project.rglob("*")
        if item.is_file()
    }
    api.init(root=project)
    assert (project / ".mlef/runs").is_dir()
    for relative, content in before.items():
        assert (project / relative).read_bytes() == content
    assert not (project / "steps/ingest.py").exists()
    if conflict != "mlef.toml":
        assert not (project / "mlef.toml").exists()


def test_existing_scaffold_is_not_repaired_or_overwritten(project: Path) -> None:
    initialize(project)
    (project / "mlef.toml").write_text("user wiring", encoding="utf-8")
    (project / "steps/train.py").write_text("user training", encoding="utf-8")
    (project / "steps/evaluate.py").unlink()
    initialize(project)
    assert (project / "mlef.toml").read_text(encoding="utf-8") == "user wiring"
    assert (project / "steps/train.py").read_text(encoding="utf-8") == "user training"
    assert not (project / "steps/evaluate.py").exists()


@pytest.mark.parametrize("surface", ["cli", "api"])
def test_storage_only_can_be_followed_by_scaffolding(project: Path, surface: str) -> None:
    if surface == "cli":
        assert main(["init", "--storage-only"]) == 0
    else:
        api.init(root=project, scaffold=False)
    assert (project / ".mlef/runs").is_dir()
    assert not (project / "mlef.toml").exists()
    assert not (project / "steps").exists()
    api.init(root=project)
    assert api.validate(root=project).status == "verified"


@pytest.mark.parametrize("conflict", ["mlef.toml", "steps"])
def test_dangling_symlinks_are_not_followed_or_replaced(project: Path, conflict: str) -> None:
    path = project / conflict
    try:
        path.symlink_to(project / "missing", target_is_directory=conflict == "steps")
    except OSError:
        pytest.skip("Creating symlinks requires platform privileges.")
    initialize(project)
    assert path.is_symlink()
    assert not (project / "missing").exists()
    assert not (project / "steps/ingest.py").exists()


def test_generated_scripts_complete_verified_handoffs(project: Path) -> None:
    api.init(root=project)
    (project / "unrelated.txt").write_text("keep", encoding="utf-8")
    directory = project / ".mlef/example"
    for name in STEPS:
        result = execute(project, name)
        assert result.returncode == 0, result.stdout + result.stderr
        assert read_step(directory / name).state == "complete"
    ingest = read_step(directory / "ingest")
    split = read_step(directory / "split")
    train = read_step(directory / "train")
    evaluate = read_step(directory / "evaluate")
    assert split.inputs["data"] == ingest.artifacts["data"].producer
    assert train.inputs["partitions"] == split.artifacts["partitions"].producer
    assert evaluate.inputs["model"] == train.artifacts["model"].producer
    assert evaluate.inputs["partitions"] == split.artifacts["partitions"].producer
    metrics = Metrics.model_validate_json(
        (directory / "evaluate/outputs/metrics/metrics.json").read_text(encoding="utf-8")
    )
    assert metrics.scope == "test"
    assert metrics.values["mae"].model_dump()["value"] == 3
    assert metrics.values["mae"].sample_count == 2
    assert metrics.model == train.artifacts["model"].producer
    before = (directory / "ingest/step.json").read_bytes()
    assert execute(project, "ingest").returncode != 0
    assert (directory / "ingest/step.json").read_bytes() == before
    assert (project / "unrelated.txt").read_text(encoding="utf-8") == "keep"


def test_missing_or_corrupted_upstream_is_not_reingested(project: Path) -> None:
    initialize(project)
    assert execute(project, "split").returncode != 0
    assert not (project / ".mlef/example/ingest").exists()
    assert execute(project, "ingest").returncode == 0
    payload = project / ".mlef/example/ingest/outputs/data/data.json"
    payload.write_text(json.dumps([999]), encoding="utf-8")
    assert execute(project, "split").returncode != 0
    assert execute(project, "train").returncode != 0
    assert execute(project, "evaluate").returncode != 0
    assert payload.read_text(encoding="utf-8") == "[999]"
