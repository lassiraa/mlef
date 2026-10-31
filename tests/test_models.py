from pathlib import Path

import pytest

from mlef import api
from mlef.cli import main


def test_empty_catalog_does_not_create_storage(project: Path) -> None:
    with pytest.raises(api.MlefError, match="not initialized"):
        api.list_models()
    api.init(scaffold=False)
    assert api.list_models().versions == []
    assert not (project / ".mlef/models").exists()


@pytest.mark.parametrize("selector", ["baseline", "../bad@" + "a" * 64, "baseline@../x"])
def test_invalid_model_selectors(project: Path, selector: str) -> None:
    api.init(scaffold=False)
    with pytest.raises(api.MlefError):
        api.model_version(selector)
    assert main(["models", "show", selector, "--json"]) == 2
