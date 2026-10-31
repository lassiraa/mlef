"""Export the public Pydantic JSON schemas, or check the committed copies."""

import argparse
import json
from pathlib import Path

from mlef.artifacts import ArtifactManifest
from mlef.benchmark import HoldoutScore, TabularTrial
from mlef.comparison import Comparison
from mlef.environment import Environment
from mlef.evidence import Result
from mlef.execution import PipelineRecord
from mlef.metrics import Metrics
from mlef.pipeline import PipelineDefinition, Step, ValidationReport
from mlef.records import Experiment, RunRecord
from mlef.sdk import StepResult
from mlef.tabular import Evaluation as TabularEvaluation
from mlef.tabular import Profile, Protocol
from mlef.tabular_artifacts import CvRecipe, TabularBundle
from mlef.tabular_comparison import TabularComparison
from mlef.tabular_diagnostics import Diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "schemas"
    root.mkdir(exist_ok=True)
    for name, model in [
        ("artifact-v1", ArtifactManifest),
        ("tabular-bundle-v1", TabularBundle),
        ("tabular-cv-recipe-v1", CvRecipe),
        ("metrics-v1", Metrics),
        ("step-v1", Step),
        ("step-result-v1", StepResult),
        ("pipeline-v1", PipelineDefinition),
        ("pipeline-validation-v1", ValidationReport),
        ("pipeline-run-v1", PipelineRecord),
        ("run-v3", RunRecord),
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
    ]:
        content = json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"
        path = root / f"{name}.schema.json"
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                raise ValueError(f"Outdated schema: {path.name}; run scripts/export_schemas.py.")
        else:
            path.write_text(content, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
