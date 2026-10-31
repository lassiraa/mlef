from pathlib import Path
from uuid import uuid4

import pytest

from mlef import api
from mlef.cli import main


@pytest.mark.parametrize("models", [[], ["same", "same"], ["model"] * 101])
def test_invalid_candidate_selection(project: Path, models: list[str]) -> None:
    api.init(scaffold=False)
    with pytest.raises(api.MlefError, match="distinct model versions"):
        api.evaluate("missing.toml", models)
    assert not (project / ".mlef/evaluations").exists()


def test_empty_evaluation_history_is_read_only(project: Path) -> None:
    api.init(scaffold=False)
    assert api.list_evaluations().batches == []
    assert not (project / ".mlef/evaluations").exists()
    assert main(["evaluations", "show", "bad", "--json"]) == 2
    with pytest.raises(api.MlefError, match="limit"):
        api.list_evaluations(limit=0)


def test_invalid_comparison_selection(project: Path) -> None:
    identifier = str(uuid4())
    for identifiers, baseline, metric in [
        ([], identifier, "mae"),
        ([identifier, identifier], identifier, "mae"),
        ([identifier], str(uuid4()), "mae"),
        ([identifier], identifier, " "),
    ]:
        with pytest.raises(api.MlefError, match="distinct evaluation IDs"):
            api.compare_evaluations(identifiers, baseline=baseline, metric=metric)
