"""Tabular artifacts and portable helper export; importing this module needs no ML libraries."""

from pathlib import Path

from .tabular_runtime import Evaluation, Fold, Prediction, Protocol, Split

__all__ = ["Evaluation", "Fold", "Prediction", "Protocol", "Split", "export_helper"]


def export_helper(destination: Path | str) -> Path:
    """Copy the exact installed evaluator source into the experiment's captured inputs."""
    target = Path(destination)
    source = Path(__file__).with_name("tabular_runtime.py")
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(source.read_text(encoding="utf-8"))
    return target
