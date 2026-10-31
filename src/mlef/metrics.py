"""Structured evaluation evidence without a dependency on an ML framework."""

from typing import Annotated, Literal, Self

from pydantic import Field, JsonValue, model_validator

from .artifacts import ArtifactRef
from .records import Contract


class Interval(Contract):
    lower: float
    upper: float
    confidence: float = Field(gt=0, lt=1)
    method: str = Field(min_length=1)

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.lower > self.upper:
            raise ValueError("Interval lower bound exceeds upper bound.")
        return self


class Metric(Contract):
    definition: str = Field(min_length=1)
    version: str = Field(default="1", min_length=1)
    unit: str | None = None
    direction: Literal["maximize", "minimize", "target", "none"] = "none"
    target: float | None = None
    aggregation: str | None = None
    sample_count: int | None = Field(default=None, ge=0)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    dimensions: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def target_direction(self) -> Self:
        if (self.direction == "target") != (self.target is not None):
            raise ValueError("Target direction requires a target; other directions forbid it.")
        return self


class Scalar(Metric):
    kind: Literal["scalar"] = "scalar"
    value: float
    interval: Interval | None = None


class Series(Metric):
    kind: Literal["series"] = "series"
    index: str = Field(min_length=1)
    x: list[float] = Field(min_length=1)
    y: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def coordinates(self) -> Self:
        if len(self.x) != len(self.y):
            raise ValueError("Series coordinate lengths must match.")
        return self


class Curve(Metric):
    kind: Literal["curve"] = "curve"
    curve_type: str = Field(min_length=1)
    x_label: str = Field(min_length=1)
    y_label: str = Field(min_length=1)
    x: list[float] = Field(min_length=1)
    y: list[float] = Field(min_length=1)
    thresholds: list[float] | None = None

    @model_validator(mode="after")
    def coordinates(self) -> Self:
        if len(self.x) != len(self.y):
            raise ValueError("Curve coordinate lengths must match.")
        if self.thresholds is not None and len(self.thresholds) != len(self.x):
            raise ValueError("Thresholds must align with curve points; omit sentinel thresholds.")
        return self


class Matrix(Metric):
    kind: Literal["matrix"] = "matrix"
    values: list[list[float]] = Field(min_length=1)
    row_axis: str = Field(min_length=1)
    column_axis: str = Field(min_length=1)
    row_labels: list[str] = Field(min_length=1)
    column_labels: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def shape(self) -> Self:
        if len(self.values) != len(self.row_labels) or any(
            len(row) != len(self.column_labels) for row in self.values
        ):
            raise ValueError("Matrix shape must match its axis labels.")
        if len(set(self.row_labels)) != len(self.row_labels) or len(set(self.column_labels)) != len(
            self.column_labels
        ):
            raise ValueError("Matrix axis labels must be unique.")
        return self


class MetricTable(Metric):
    kind: Literal["table"] = "table"
    columns: list[str] = Field(min_length=1)
    rows: list[dict[str, str | float | bool | None]] = Field(min_length=1)

    @model_validator(mode="after")
    def shape(self) -> Self:
        if len(set(self.columns)) != len(self.columns) or any(not name for name in self.columns):
            raise ValueError("Table columns must be nonempty and unique.")
        if any(set(row) != set(self.columns) for row in self.rows):
            raise ValueError("Every table row must match the declared columns.")
        return self


class Distribution(Metric):
    kind: Literal["distribution"] = "distribution"
    samples: list[float] = Field(min_length=1)
    weights: list[float] | None = None

    @model_validator(mode="after")
    def weighting(self) -> Self:
        if self.weights is not None and (
            len(self.weights) != len(self.samples)
            or any(weight < 0 for weight in self.weights)
            or not any(weight > 0 for weight in self.weights)
        ):
            raise ValueError(
                "Weights must align with samples and have nonnegative, positive total mass."
            )
        return self


class Undefined(Metric):
    kind: Literal["undefined"] = "undefined"
    reason: str = Field(min_length=1)


MetricValue = Annotated[
    Scalar | Series | Curve | Matrix | MetricTable | Distribution | Undefined,
    Field(discriminator="kind"),
]


class Metrics(Contract):
    """A metric artifact payload; persistence is the responsibility of the artifact SDK."""

    schema_version: Literal[1] = 1
    model: ArtifactRef | None = None
    dataset: ArtifactRef | None = None
    split: ArtifactRef | None = None
    predictions: ArtifactRef | None = None
    scope: str = Field(min_length=1)
    values: dict[str, MetricValue] = Field(min_length=1)

    @model_validator(mode="after")
    def metric_names(self) -> Self:
        if any(not name.strip() for name in self.values):
            raise ValueError("Metric names must not be empty.")
        return self
