"""Export the public Pydantic JSON schemas, or check the committed copies."""

import argparse
import json
from pathlib import Path

from mlef.comparison import Comparison
from mlef.environment import Environment
from mlef.evidence import Result
from mlef.records import Experiment, RunRecord
from mlef.tabular import Evaluation as TabularEvaluation
from mlef.tabular import Profile, Protocol
from mlef.tabular_comparison import TabularComparison
from mlef.tabular_diagnostics import Diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "schemas"
    root.mkdir(exist_ok=True)
    for name, model in [
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
