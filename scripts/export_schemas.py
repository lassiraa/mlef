"""Export the public Pydantic JSON schemas, or check the committed copies."""

import argparse
import json
from pathlib import Path

from mlef.api import RunHistory
from mlef.artifacts import ArtifactManifest, CustomType
from mlef.benchmark import HoldoutScore, TabularTrial
from mlef.comparison import Comparison
from mlef.environment import Environment
from mlef.evaluation_comparison import EvaluationComparison
from mlef.evaluations import EvaluationBatch, EvaluationContext, EvaluationListing, EvaluationSpec
from mlef.evidence import Result
from mlef.execution import CacheDecision, PipelineRecord, StepFingerprint
from mlef.inspection import RunMetrics
from mlef.metrics import ExternalMetric, Metrics
from mlef.models import ModelListing, ModelVersion
from mlef.pipeline import PipelineDefinition, Step, StepV1, ValidationReport
from mlef.records import Experiment, RunRecord
from mlef.sdk import StepResult
from mlef.tabular import Evaluation as TabularEvaluation
from mlef.tabular import Profile, Protocol
from mlef.tabular_artifacts import CvRecipe, TabularBundle
from mlef.tabular_comparison import TabularComparison
from mlef.tabular_diagnostics import Diagnostics
from mlef.transfers import RunTransfer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "schemas"
    root.mkdir(exist_ok=True)
    schemas = [
        ("run-history-v1", RunHistory),
        ("run-metrics-v1", RunMetrics),
        ("model-version-v1", ModelVersion),
        ("model-listing-v1", ModelListing),
        ("artifact-v1", ArtifactManifest),
        ("custom-type-v1", CustomType),
        ("tabular-bundle-v1", TabularBundle),
        ("tabular-cv-recipe-v1", CvRecipe),
        ("metrics-v1", Metrics),
        ("external-metric-v1", ExternalMetric),
        ("step-v1", StepV1),
        ("step-v2", Step),
        ("step-result-v1", StepResult),
        ("pipeline-v1", PipelineDefinition),
        ("pipeline-v2", PipelineDefinition),
        ("evaluation-spec-v1", EvaluationSpec),
        ("evaluation-batch-v1", EvaluationBatch),
        ("evaluation-listing-v1", EvaluationListing),
        ("evaluation-context-v1", EvaluationContext),
        ("evaluation-comparison-v1", EvaluationComparison),
        ("pipeline-validation-v1", ValidationReport),
        ("pipeline-run-v1", PipelineRecord),
        ("step-fingerprint-v1", StepFingerprint),
        ("cache-decision-v1", CacheDecision),
        ("run-v3", RunRecord),
        ("run-transfer-v1", RunTransfer),
        ("experiment-v1", Experiment),
        ("result-v1", Result),
        ("comparison-v1", Comparison),
        ("environment-v1", Environment),
        ("tabular-protocol-v2", Protocol),
        ("tabular-evaluation-v2", TabularEvaluation),
        ("tabular-comparison-v1", TabularComparison),
        ("tabular-profile-v1", Profile),
        ("tabular-diagnostics-v1", Diagnostics),
        ("tabular-trial-v2", TabularTrial),
        ("holdout-score-v1", HoldoutScore),
    ]
    expected = {f"{name}.schema.json" for name, _ in schemas}
    obsolete = {path.name for path in root.glob("*.schema.json")} - expected
    if obsolete:
        raise ValueError(f"Remove obsolete schemas: {', '.join(sorted(obsolete))}")
    for name, model in schemas:
        schema = model.model_json_schema()
        if name == "pipeline-v1":
            # Preserve the published schema for definitions with only local bindings.
            schema["properties"]["schema_version"] = {
                "const": 1,
                "default": 1,
                "title": "Schema Version",
                "type": "integer",
            }
            patterns = schema["$defs"]["PipelineStep"]["properties"]["bindings"][
                "patternProperties"
            ]
            for pattern in patterns:
                patterns[pattern] = {"type": "string"}
            schema["$defs"].pop("ExternalInput", None)
            schema["$defs"].pop("ArtifactRef", None)
        content = json.dumps(schema, indent=2, sort_keys=True) + "\n"
        path = root / f"{name}.schema.json"
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                raise ValueError(f"Outdated schema: {path.name}; run scripts/export_schemas.py.")
        else:
            path.write_text(content, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
