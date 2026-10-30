"""Versioned contracts and strict JSON shared by execution and inspection."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    field_validator,
)

from .provenance import GitProvenance, SourceInputs


def run_id(value: str) -> str:
    identifier = UUID(value)
    if identifier.version != 4 or str(identifier) != value:
        raise ValueError("Expected a full canonical UUID4 run ID.")
    return value


def json_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> JsonValue:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=json_object)
        json.dumps(value, allow_nan=False)  # Reject non-finite and overflowing JSON numbers.
        return TypeAdapter(JsonValue).validate_python(value)
    except RecursionError as error:
        raise ValueError("JSON nesting exceeds the supported depth.") from error


class Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, allow_inf_nan=False, validate_assignment=True
    )

    @field_validator("schema_version", mode="before", check_fields=False)
    @classmethod
    def integer_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Schema version must be an integer, not a boolean or string.")
        return value


class Dataset(Contract):
    reference: str | None = None
    fingerprint: str | None = None


class Evaluation(Contract):
    protocol_id: str | None = None
    split_fingerprint: str | None = None
    seeds: dict[str, int] | None = None
    metric_definitions: dict[str, str] = Field(default_factory=dict)


class Experiment(Contract):
    schema_version: Literal[1] = 1
    hypothesis: str | None = None
    parent_run_id: str | None = None
    changed_variables: dict[str, JsonValue] = Field(default_factory=dict)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    seeds: dict[str, int] = Field(default_factory=dict)
    dataset: Dataset | None = None
    evaluation: Evaluation | None = None
    metric_directions: dict[str, Literal["maximize", "minimize"]] = Field(default_factory=dict)

    @field_validator("parent_run_id")
    @classmethod
    def valid_parent(cls, value: str | None) -> str | None:
        return run_id(value) if value is not None else None


class RunError(Contract):
    category: str
    message: str


class Controls(Contract):
    requested: dict[str, JsonValue] = Field(default_factory=dict)
    applied: dict[str, JsonValue] = Field(default_factory=dict)


class Runtime(Contract):
    token: str
    project: str
    container_name: str
    probe_name: str
    container_id: str | None = None


class RunRecord(Contract):
    schema_version: Literal[3] = 3
    id: str
    state: Literal["preparing", "running", "succeeded", "failed", "cancelled"]
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    command: list[str] = Field(min_length=1)
    execution_exit_code: int | None = None
    execution_seconds: float | None = Field(default=None, ge=0)
    error: RunError | None = None
    warnings: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    git: GitProvenance | None = None
    inputs: SourceInputs | None = None
    experiment: Experiment | None = None
    controls: Controls | None = None
    runtime: Runtime | None = None
    stage: str | None = None

    @field_validator("id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        return run_id(value)

    @field_validator("created_at", "started_at", "finished_at")
    @classmethod
    def utc_timestamp(cls, value: str | None) -> str | None:
        if value is not None and datetime.fromisoformat(value).utcoffset() != timedelta(0):
            raise ValueError("Run timestamps must include a UTC offset.")
        return value


def read_run(path: Path) -> RunRecord:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError("Run metadata must be a JSON object.")
    version = data.get("schema_version")
    if type(version) is not int or version not in (1, 2, 3):
        raise ValueError(f"Unsupported run schema version: {version!r}")
    if version in (1, 2):
        data = dict(data)
        data["schema_version"] = 3
        warnings = data.get("warnings", [])
        if not isinstance(warnings, list):
            raise ValueError("Invalid legacy warnings.")
        data["warnings"] = [
            *warnings,
            f"Legacy version-{version} record; absent controls/evidence are unknown.",
        ]
    record = RunRecord.model_validate(data)
    if record.state in {"succeeded", "failed", "cancelled"} and record.finished_at is None:
        raise ValueError("A terminal run must have a finish timestamp.")
    if record.state == "succeeded" and record.execution_exit_code != 0:
        raise ValueError("A successful run must have execution exit code zero.")
    return record
