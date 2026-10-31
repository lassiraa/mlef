"""From this copied example's Git root: python demo.py (installed MLEF and Docker)."""

import argparse
from pathlib import Path

from mlef import api, tabular
from mlef.records import Dataset, Evaluation, Experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regression", action="store_true")
    args = parser.parse_args()
    task = "regression" if args.regression else "binary"
    data = "regression.csv" if args.regression else "data.csv"
    numeric = (
        ["amount", "age", "tenure"] if args.regression else ["radius", "texture", "smoothness"]
    )
    categorical = ["region"] if args.regression else ["radius_band"]
    api.init()
    if not Path("evaluate.py").exists():
        tabular.export_helper("evaluate.py")
    options = api.RunOptions(timeout_seconds=180, max_log_bytes=65536)
    prepared = api.run(
        "evaluate.py",
        includes=[data],
        options=options,
        arguments=[
            "prepare",
            "--numeric",
            *numeric,
            "--categorical",
            *categorical,
            "--data",
            data,
            "--task",
            task,
        ],
    )
    if prepared.state != "succeeded":
        raise RuntimeError(prepared.model_dump_json())
    api.get(prepared.id, "protocol.json", output="protocol.json")
    protocol = tabular.Protocol.model_validate_json(Path("protocol.json").read_text())
    candidates: dict[str, api.RunStatus] = {}
    baseline: str | None = None
    spec: Experiment | None = None
    factories = (
        ["make_dummy_regressor", "make_regressor", "make_xgboost_regressor"]
        if args.regression
        else [
            "make_dummy",
            "make_estimator",
            "make_tree",
            "make_xgboost",
            "make_xgboost_candidate",
        ]
    )
    metric = protocol.metric + "_mean"
    for factory in factories:
        spec = Experiment(
            hypothesis=f"Evaluate {factory} on the saved folds",
            parent_run_id=baseline,
            changed_variables={"factory": factory},
            dataset=Dataset(fingerprint=protocol.dataset_sha256),
            evaluation=Evaluation(
                protocol_id=protocol.fingerprint(),
                split_fingerprint=protocol.fingerprint(),
                seeds={"evaluation": protocol.seed},
                metric_definitions={metric: f"mean fold {protocol.metric}"},
            ),
            metric_directions={metric: protocol.direction},
        )
        Path("experiment.json").write_text(spec.model_dump_json(), encoding="utf-8")
        result = api.run(
            "evaluate.py",
            includes=[data, "protocol.json", "model.py"],
            arguments=["evaluate", "--factory", f"model:{factory}", "--data", data],
            options=options,
            experiment="experiment.json",
        )
        print(factory, result.id, result.state, result.result.metrics)
        if result.state != "succeeded":
            raise RuntimeError(result.model_dump_json())
        api.export_tabular_artifacts(result.id, data, Path(".mlef/tabular-artifacts") / result.id)
        candidates[factory] = result
        if baseline is None:
            baseline = result.id
        else:
            print(api.compare_tabular(baseline, result.id).model_dump_json())
    choose = max if protocol.direction == "maximize" else min
    best = choose(candidates, key=lambda name: candidates[name].result.metrics[metric])
    assert spec is not None
    refit_spec = Experiment.model_validate(
        {
            **spec.model_dump(),
            "hypothesis": f"Refit selected {best}",
            "parent_run_id": candidates[best].id,
            "changed_variables": {"factory": best, "refit": True},
        }
    )
    Path("experiment.json").write_text(refit_spec.model_dump_json(), encoding="utf-8")
    selected = api.run(
        "evaluate.py",
        includes=[data, "protocol.json", "model.py"],
        options=options,
        arguments=["evaluate", "--factory", f"model:{best}", "--refit", "--data", data],
        experiment="experiment.json",
    )
    if selected.state != "succeeded":
        raise RuntimeError(selected.model_dump_json())
    for name in ["model.joblib", "model-recipe.json"]:
        api.get(selected.id, name, output=name)
    inference = api.run(
        "evaluate.py",
        includes=[data, "model.py", "model.joblib", "model-recipe.json"],
        arguments=["predict", "--data", data],
        options=options,
    )
    if inference.state != "succeeded":
        raise RuntimeError(inference.model_dump_json())
    api.get(inference.id, "inference.csv", output="inference.csv")
    print("Selected", best, selected.id)


if __name__ == "__main__":
    main()
