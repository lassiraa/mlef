"""Optional numeric evidence, independent of execution success."""

from pathlib import Path
from typing import Literal

from pydantic import Field, TypeAdapter

from .records import Contract, Experiment, read_json
from .storage import QueryError, artifact_path, artifacts, inspect


class Diagnostic(Contract):
    code: str
    severity: Literal["info", "warning", "error"]
    message: str


class Result(Contract):
    schema_version: Literal[1] = 1
    run_id: str
    evidence_status: Literal["available", "missing", "invalid"]
    metrics: dict[str, int | float] = Field(default_factory=dict)
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)


def experiment_spec(root: Path, path: Path) -> Experiment:
    try:
        experiment = Experiment.model_validate(read_json(path))
    except OSError as error:
        raise ValueError(f"Cannot read experiment specification: {error}") from error
    if experiment.parent_run_id is not None:
        try:
            inspect(root, experiment.parent_run_id)
        except QueryError as error:
            raise ValueError(f"Invalid experiment parent: {error}") from error
    return experiment


def normalize(root: Path, identifier: str) -> Result:
    references = [
        f".mlef/runs/{identifier}/artifacts/{name}" for name in artifacts(root, identifier)
    ]
    result = Result(run_id=identifier, evidence_status="missing", artifacts=references)
    try:
        metrics_path = artifact_path(root, identifier, "metrics.json")
    except QueryError as error:
        if error.category == "unknown_artifact":
            result.diagnostics.append(
                Diagnostic(
                    code="metrics_missing",
                    severity="info",
                    message="No optional metrics.json was written.",
                )
            )
            return result
        result.evidence_status = "invalid"
        result.diagnostics.append(
            Diagnostic(code="metrics_invalid", severity="warning", message=str(error))
        )
        return result
    try:
        if not metrics_path.is_file():
            raise ValueError("metrics.json must be a regular file.")
        values = read_json(metrics_path)
        result.metrics = TypeAdapter(dict[str, int | float]).validate_python(values, strict=True)
        # validate_assignment also rejects non-finite float values.
        result.evidence_status = "available"
    except (OSError, ValueError) as error:
        result.evidence_status = "invalid"
        result.diagnostics.append(
            Diagnostic(code="metrics_invalid", severity="warning", message=str(error))
        )
    return result


def read_result(root: Path, identifier: str) -> Result:
    path = root / ".mlef/runs" / identifier / "result.json"
    if not path.exists():
        return Result(
            run_id=identifier,
            evidence_status="missing",
            diagnostics=[
                Diagnostic(
                    code="result_unavailable",
                    severity="warning",
                    message="Normalized result evidence is unavailable.",
                )
            ],
        )
    try:
        if path.is_symlink():
            raise ValueError("Result metadata must not be a symbolic link.")
        result = Result.model_validate(read_json(path))
        if result.run_id != identifier:
            raise ValueError("Result run ID does not match its storage directory.")
        return result
    except (OSError, ValueError) as error:
        return Result(
            run_id=identifier,
            evidence_status="invalid",
            diagnostics=[Diagnostic(code="result_invalid", severity="warning", message=str(error))],
        )
