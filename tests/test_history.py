import json
from pathlib import Path
from uuid import uuid4

import pytest

from mlef import api
from mlef.cli import main
from mlef.records import RunRecord
from mlef.runner import save


def test_history_filters_pages_and_diagnostics(
    project: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    api.init(scaffold=False)
    identifiers = [str(uuid4()) for _ in range(3)]
    for index, identifier in enumerate(identifiers):
        directory = project / ".mlef/runs" / identifier
        directory.mkdir()
        save(
            directory / "run.json",
            RunRecord(
                id=identifier,
                state="failed" if index == 0 else "succeeded",
                command=["python", "train.py"],
                created_at=f"2026-01-0{index + 1}T00:00:00Z",
                finished_at=f"2026-01-0{index + 1}T00:00:01Z",
                execution_exit_code=1 if index == 0 else 0,
            ),
        )
    (project / ".mlef/runs/broken").mkdir()
    page = api.run_history(state="succeeded", limit=1)
    assert [run.id for run in page.runs] == [identifiers[2]]
    assert page.total == 2
    assert page.next_offset == 1
    assert page.diagnostics
    assert api.run_history(state="succeeded", offset=1).runs[0].id == identifiers[1]
    assert api.run_history(since="2026-01-03T02:00:00+02:00").total == 1
    assert api.run_history(offset=20).next_offset is None
    assert main(["runs", "list", "--state", "succeeded", "--limit", "1", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == page.model_dump(mode="json")


@pytest.mark.parametrize(
    "options",
    [
        {"limit": 0},
        {"limit": True},
        {"offset": -1},
        {"since": "2026-01-01"},
        {"since": "bad"},
        {"state": "bad"},
    ],
)
def test_history_rejects_invalid_queries(project: Path, options: dict[str, object]) -> None:
    with pytest.raises(api.MlefError, match=r"."):
        api.run_history(**options)  # ty: ignore[invalid-argument-type]


def test_history_cli_invalid_date(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["runs", "list", "--since", "2026-01-01", "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["category"] == "configuration"
