"""From this copied example's Git root: python demo.py (installed MLEF and Docker)."""

from pathlib import Path

from mlef import api, tabular
from mlef.records import Dataset, Evaluation, Experiment


def main() -> None:
    api.init()
    if not Path("evaluate.py").exists():
        tabular.export_helper("evaluate.py")
    options = api.RunOptions(timeout_seconds=180, max_log_bytes=65536)
    prepared = api.run(
        "evaluate.py",
        includes=["data.csv"],
        options=options,
        arguments=[
            "prepare",
            "--numeric",
            "radius",
            "texture",
            "smoothness",
            "--categorical",
            "radius_band",
        ],
    )
    if prepared.state != "succeeded":
        raise RuntimeError(prepared.model_dump_json())
    api.get(prepared.id, "protocol.json", output="protocol.json")
    protocol = tabular.Protocol.model_validate_json(Path("protocol.json").read_text())
    candidates: dict[str, api.RunStatus] = {}
    baseline: str | None = None
    for factory in [
        "make_dummy",
        "make_estimator",
        "make_tree",
        "make_xgboost",
        "make_xgboost_candidate",
    ]:
        spec = Experiment(
            hypothesis=f"Evaluate {factory} on the saved folds",
            parent_run_id=baseline,
            changed_variables={"factory": factory},
            dataset=Dataset(fingerprint=protocol.dataset_sha256),
            evaluation=Evaluation(
                protocol_id=protocol.fingerprint(),
                split_fingerprint=protocol.fingerprint(),
                seeds={"evaluation": protocol.seed},
                metric_definitions={"roc_auc_mean": "mean fold ROC-AUC"},
            ),
            metric_directions={"roc_auc_mean": "maximize"},
        )
        Path("experiment.json").write_text(spec.model_dump_json(), encoding="utf-8")
        result = api.run(
            "evaluate.py",
            includes=["data.csv", "protocol.json", "model.py"],
            arguments=["evaluate", "--factory", f"model:{factory}"],
            options=options,
            experiment="experiment.json",
        )
        print(factory, result.id, result.state, result.result.metrics)
        if result.state != "succeeded":
            raise RuntimeError(result.model_dump_json())
        candidates[factory] = result
        if baseline is None:
            baseline = result.id
        else:
            print(api.compare_tabular(baseline, result.id).model_dump_json())
    best = max(candidates, key=lambda name: candidates[name].result.metrics["roc_auc_mean"])
    selected = api.run(
        "evaluate.py",
        includes=["data.csv", "protocol.json", "model.py"],
        options=options,
        arguments=["evaluate", "--factory", f"model:{best}", "--refit"],
    )
    if selected.state != "succeeded":
        raise RuntimeError(selected.model_dump_json())
    for name in ["model.joblib", "model-recipe.json"]:
        api.get(selected.id, name, output=name)
    inference = api.run(
        "evaluate.py",
        includes=["data.csv", "model.joblib", "model-recipe.json"],
        arguments=["predict"],
        options=options,
    )
    if inference.state != "succeeded":
        raise RuntimeError(inference.model_dump_json())
    api.get(inference.id, "inference.csv", output="inference.csv")
    print("Selected", best, selected.id)


if __name__ == "__main__":
    main()
