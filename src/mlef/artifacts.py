"""Versioned artifact contracts; identity is independent of storage and producer."""

import hashlib
import json
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from .records import Contract, run_id


class ArtifactType(StrEnum):
    FILE = "file"
    DIRECTORY = "directory"
    TABLE = "table"
    DATASET = "dataset"
    SPLIT = "split"
    PREDICTIONS = "predictions"
    MODEL = "model"
    CHECKPOINT = "checkpoint"
    METRICS = "metrics"
    PARAMETERS = "parameters"
    SCHEMA = "schema"
    ARRAY = "array"
    EMBEDDINGS = "embeddings"
    INDEX = "index"
    PLOT = "plot"
    REPORT = "report"
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    ANNOTATIONS = "annotations"
    EXPLANATIONS = "explanations"
    ARCHIVE = "archive"


class ArtifactRef(Contract):
    """An exact producer reference, never a mutable latest alias or storage URI."""

    run_id: str
    step: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    output: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    _run_id = field_validator("run_id")(run_id)


def safe_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "\\" in value
        or ":" in value
        or "\x00" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("Artifact files require normalized relative POSIX paths.")
    return value


class ArtifactFile(Contract):
    path: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    _path = field_validator("path")(safe_path)


class Column(Contract):
    name: str = Field(min_length=1)
    dtype: str = Field(min_length=1)
    role: Literal["feature", "target", "row_id", "group", "time", "auxiliary"] = "feature"
    nullable: bool = True


class DatasetMetadata(Contract):
    kind: Literal["dataset"] = "dataset"
    rows: int = Field(ge=0)
    columns: list[Column] = Field(min_length=1)
    row_id: list[str] = Field(min_length=1)
    statistics: Literal["omitted", "exact", "sampled", "unavailable"] = "omitted"

    @model_validator(mode="after")
    def column_names(self) -> Self:
        names = [column.name for column in self.columns]
        if len(set(names)) != len(names):
            raise ValueError("Dataset column names must be unique.")
        if len(set(self.row_id)) != len(self.row_id) or not set(self.row_id) <= set(names):
            raise ValueError("Row identity must name distinct existing columns.")
        return self


class SplitMetadata(Contract):
    kind: Literal["split"] = "split"
    dataset: ArtifactRef
    strategy: str = Field(min_length=1)
    seed: int | None = None
    assignments_file: str

    _assignment_path = field_validator("assignments_file")(safe_path)


class PredictionsMetadata(Contract):
    kind: Literal["predictions"] = "predictions"
    dataset: ArtifactRef
    model: ArtifactRef
    split: ArtifactRef | None = None
    task: str = Field(min_length=1)
    row_id: list[str] = Field(min_length=1)
    value_columns: list[str] = Field(min_length=1)
    class_order: list[str] | None = None
    scope: str = Field(min_length=1)

    @model_validator(mode="after")
    def distinct_columns(self) -> Self:
        columns = self.row_id + self.value_columns
        if any(not column for column in columns) or len(set(columns)) != len(columns):
            raise ValueError("Prediction identity and value columns must be nonempty and distinct.")
        if self.class_order is not None and (
            not self.class_order or len(set(self.class_order)) != len(self.class_order)
        ):
            raise ValueError("Class ordering must be nonempty and unique.")
        return self


class ModelMetadata(Contract):
    kind: Literal["model"] = "model"
    framework: str = Field(min_length=1)
    framework_version: str | None = None
    serialization: str = Field(min_length=1)
    training_dataset: ArtifactRef | None = None


class GenericMetadata(Contract):
    """Specialized metadata can be introduced without changing file storage."""

    kind: Literal["generic"] = "generic"


Metadata = DatasetMetadata | SplitMetadata | PredictionsMetadata | ModelMetadata | GenericMetadata


class ArtifactManifest(Contract):
    schema_version: Literal[1] = 1
    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    type: ArtifactType
    format: str = Field(min_length=1)
    files: list[ArtifactFile] = Field(min_length=1)
    metadata: Metadata = Field(default_factory=GenericMetadata, discriminator="kind")
    producer: ArtifactRef | None = None

    @model_validator(mode="after")
    def inventory(self) -> Self:
        paths = [file.path for file in self.files]
        if len(set(paths)) != len(paths):
            raise ValueError("Artifact inventory contains duplicate paths.")
        for path in paths:
            if any(parent.as_posix() in paths for parent in PurePosixPath(path).parents):
                raise ValueError("Artifact inventory contains conflicting file/directory paths.")
        if self.metadata.kind != "generic" and self.metadata.kind != self.type.value:
            raise ValueError("Artifact type and metadata kind disagree.")
        if (
            self.type
            in {
                ArtifactType.DATASET,
                ArtifactType.SPLIT,
                ArtifactType.PREDICTIONS,
                ArtifactType.MODEL,
            }
            and self.metadata.kind == "generic"
        ):
            raise ValueError("This artifact type requires specialized metadata.")
        if isinstance(self.metadata, SplitMetadata) and self.metadata.assignments_file not in paths:
            raise ValueError("Split assignments must be included in the file inventory.")
        if self.producer is not None and (
            self.producer.output != self.name or self.producer.sha256 != self.content_sha256()
        ):
            raise ValueError("Producer reference must identify this output and its content.")
        return self

    @property
    def size_bytes(self) -> int:
        return sum(file.size_bytes for file in self.files)

    def content_sha256(self) -> str:
        """Hash semantic metadata and canonical inventory, excluding name/provenance."""
        content = self.model_dump(mode="json", exclude={"name", "producer"})
        content["files"] = [
            file.model_dump(mode="json") for file in sorted(self.files, key=lambda file: file.path)
        ]
        encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode()).hexdigest()
