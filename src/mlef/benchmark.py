"""Framework-independent contracts for coordinator-recorded tabular trials."""

from typing import Literal

from pydantic import Field, JsonValue, model_validator

from .records import Contract


class Holdout(Contract):
    schema_version: Literal[1] = 1
    task: Literal["binary", "regression"]
    metric: Literal["roc_auc", "rmse"]
    direction: Literal["maximize", "minimize"]
    task_fingerprint: str = Field(pattern="^[a-f0-9]{64}$")
    protocol_id: str = Field(pattern="^[a-f0-9]{64}$")
    labels: dict[str, float] = Field(min_length=1)

    @model_validator(mode="after")
    def task_metric(self) -> "Holdout":
        expected = ("roc_auc", "maximize") if self.task == "binary" else ("rmse", "minimize")
        if (self.metric, self.direction) != expected or (
            self.task == "binary" and set(self.labels.values()) != {0, 1}
        ):
            raise ValueError("Invalid held-out task/scorer/label contract.")
        return self


class HoldoutScore(Contract):
    schema_version: Literal[1] = 1
    task_fingerprint: str = Field(pattern="^[a-f0-9]{64}$")
    protocol_id: str = Field(pattern="^[a-f0-9]{64}$")
    metric: Literal["roc_auc", "rmse"]
    direction: Literal["maximize", "minimize"]
    score: float = Field(ge=0)
    prediction_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    candidate_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    labels_sha256: str = Field(pattern="^[a-f0-9]{64}$")
    rows: int = Field(gt=0)


class TabularTrial(Contract):
    schema_version: Literal[2] = 2
    task_fingerprint: str = Field(pattern="^[a-f0-9]{64}$")
    protocol_id: str = Field(pattern="^[a-f0-9]{64}$")
    task: Literal["binary", "regression"]
    metric: Literal["roc_auc", "rmse"]
    direction: Literal["maximize", "minimize"]
    condition: Literal["control", "mlef", "reference"]
    agent: str
    model: str
    agent_settings: dict[str, JsonValue] = Field(default_factory=dict)
    budget_seconds: int = Field(gt=0)
    seed: int
    repetition: int = Field(ge=0)
    state: Literal["completed", "failed", "cancelled"]
    elapsed_seconds: float = Field(ge=0)
    tool_output_bytes: int = Field(ge=0)
    intervention_count: int = Field(ge=0)
    valid_experiments: int = Field(ge=0)
    valid_comparisons: int = Field(ge=0)
    time_to_first_valid_gain_seconds: float | None = Field(default=None, ge=0)
    validation_score: float | None = Field(default=None, ge=0)
    heldout_score: float | None = Field(default=None, ge=0)
    prediction_sha256: str | None = Field(default=None, pattern="^[a-f0-9]{64}$")
    candidate_sha256: str | None = Field(default=None, pattern="^[a-f0-9]{64}$")
    failure: str | None = None

    @model_validator(mode="after")
    def outcome(self) -> "TabularTrial":
        expected = ("roc_auc", "maximize") if self.task == "binary" else ("rmse", "minimize")
        if (self.metric, self.direction) != expected:
            raise ValueError("Task, scorer and direction disagree.")
        if (
            self.metric == "roc_auc"
            and self.validation_score is not None
            and self.validation_score > 1
        ):
            raise ValueError("Validation ROC-AUC must be between zero and one.")
        if self.state == "completed":
            if self.failure is not None:
                raise ValueError("Completed trials cannot contain a failure outcome.")
            if (
                self.heldout_score is None
                or self.prediction_sha256 is None
                or self.candidate_sha256 is None
            ):
                raise ValueError("Completed trials require a frozen scored submission.")
            if self.elapsed_seconds > self.budget_seconds:
                raise ValueError("Completed trial exceeded its matched budget.")
            if self.metric == "roc_auc" and self.heldout_score > 1:
                raise ValueError("ROC-AUC must be between zero and one.")
        elif self.heldout_score is not None or self.failure is None:
            raise ValueError("Failed/cancelled trials require a reason and no held-out score.")
        first = self.time_to_first_valid_gain_seconds
        if first is not None and (first > self.elapsed_seconds or self.valid_comparisons == 0):
            raise ValueError("First valid gain requires a comparison within the measured trial.")
        return self
