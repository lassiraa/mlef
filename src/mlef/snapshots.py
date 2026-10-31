"""Internal frozen source/SDK bundles for evaluation child runs."""

import shutil
from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue

from .environment import SdkWheel, file_hash
from .provenance import GitProvenance, SourceInputs
from .records import Contract, read_json
from .sdk import _checked


class SourceSnapshot(Contract):
    schema_version: Literal[1] = 1
    git: GitProvenance
    inputs: SourceInputs
    patch_sha256: str
    sdk: SdkWheel
    evaluation: dict[str, JsonValue] = Field(default_factory=dict)


def restore(origin: Path, destination: Path) -> SourceSnapshot:
    snapshot = SourceSnapshot.model_validate(read_json(_checked(origin / "snapshot.json")))
    files = {
        "source.tar.gz": snapshot.inputs.source_sha256,
        "changes.patch": snapshot.patch_sha256,
        f"sdk/{snapshot.sdk.filename}": snapshot.sdk.sha256,
    }
    for name, checksum in files.items():
        source = _checked(origin / name)
        if file_hash(source) != checksum:
            raise ValueError("Frozen evaluation source or SDK bytes changed.")
        target = _checked(destination / name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if file_hash(target) != checksum:
            raise ValueError("Copied evaluation source or SDK bytes changed.")
    if snapshot.evaluation:
        from .runner import save

        save(_checked(destination / "evaluation.json"), dict[str, object](snapshot.evaluation))
    return snapshot
