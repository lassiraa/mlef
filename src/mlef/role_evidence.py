"""Check typed step output evidence against exact bound artifact identities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from .artifacts import ArtifactManifest, ArtifactRef, ArtifactType
from .metrics import Metrics
from .records import json_object

if TYPE_CHECKING:
    from .pipeline import Step


def validate_role_evidence(
    directory: Path,
    step: Step,
    artifacts: dict[str, ArtifactManifest],
    inputs: dict[str, ArtifactManifest],
) -> None:
    """Validate lineage declarations, not execution semantics or leakage freedom."""
    if step.kind == "generic":
        return
    models = [item.producer for item in inputs.values() if item.type == ArtifactType.MODEL]
    datasets = [item.producer for item in inputs.values() if item.type == ArtifactType.DATASET]
    splits = [item for item in inputs.values() if item.type == ArtifactType.SPLIT]
    predictions = [item for item in inputs.values() if item.type == ArtifactType.PREDICTIONS]

    def check_split(reference: ArtifactRef | None, dataset: ArtifactRef | None) -> None:
        if reference is not None and not any(
            item.producer == reference
            and item.metadata.kind == "split"
            and item.metadata.dataset == dataset
            for item in splits
        ):
            raise ValueError("Output split must identify a bound split of the evaluated dataset.")

    for output, manifest in artifacts.items():
        metadata = manifest.metadata
        if (
            step.kind == "train"
            and metadata.kind == "model"
            and metadata.training_dataset is not None
            and metadata.training_dataset not in datasets
            and not any(
                item.metadata.kind == "split" and item.metadata.dataset == metadata.training_dataset
                for item in splits
            )
        ):
            raise ValueError(
                "Model training_dataset must identify a bound dataset or split dataset."
            )
        if step.kind == "predict" and metadata.kind == "predictions":
            if metadata.model not in models or metadata.dataset not in datasets:
                raise ValueError("Predictions must identify the exact bound model and dataset.")
            check_split(metadata.split, metadata.dataset)
        if step.kind != "evaluate" or manifest.type != ArtifactType.METRICS:
            continue
        if manifest.format != "application/json" or len(manifest.files) != 1:
            raise ValueError("Evaluation metrics require one application/json payload file.")
        entry = manifest.files[0]
        if entry.size_bytes > 32 * 1024 * 1024:
            raise ValueError("Evaluation metrics exceed the 32 MiB payload limit.")
        from .sdk import _checked

        path = _checked(directory.absolute() / "outputs" / output / entry.path)
        with path.open("rb") as stream:
            content = stream.read(entry.size_bytes + 1)
        if len(content) != entry.size_bytes or hashlib.sha256(content).hexdigest() != entry.sha256:
            raise ValueError("Evaluation metric bytes changed after verification.")
        try:
            metrics = Metrics.model_validate(json.loads(content, object_pairs_hook=json_object))
        except RecursionError as error:
            raise ValueError("Evaluation metric JSON is too deeply nested.") from error
        if metrics.dataset is None or metrics.dataset not in datasets or metrics.model is None:
            raise ValueError(
                "Evaluation metrics must identify the bound dataset and an exact model."
            )
        if metrics.predictions is None:
            if metrics.model not in models:
                raise ValueError(
                    "Evaluation metrics must identify a bound model or bound predictions."
                )
            check_split(metrics.split, metrics.dataset)
        else:
            matched = next(
                (item for item in predictions if item.producer == metrics.predictions), None
            )
            if (
                matched is None
                or matched.metadata.kind != "predictions"
                or matched.metadata.model != metrics.model
                or matched.metadata.dataset != metrics.dataset
                or matched.metadata.split != metrics.split
                or matched.metadata.scope != metrics.scope
            ):
                raise ValueError(
                    "Evaluation metrics disagree with their bound predictions evidence."
                )
