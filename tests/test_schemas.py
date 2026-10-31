import runpy
import shutil
import sys
from pathlib import Path

import pytest


def test_schema_export_rejects_obsolete_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "scripts/export_schemas.py"
    script.parent.mkdir()
    shutil.copyfile(Path(__file__).resolve().parents[1] / "scripts/export_schemas.py", script)
    monkeypatch.setattr(sys, "argv", [str(script)])
    runpy.run_path(str(script), run_name="__main__")
    monkeypatch.setattr(sys, "argv", [str(script), "--check"])
    runpy.run_path(str(script), run_name="__main__")
    (tmp_path / "schemas/obsolete-v1.schema.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Remove obsolete schemas: obsolete-v1\.schema\.json"):
        runpy.run_path(str(script), run_name="__main__")
