"""From this copied example's Git root: python demo.py (installed MLEF and Docker)."""

from pathlib import Path

from mlef import api, tabular


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
    for factory in ["make_dummy", "make_estimator", "make_tree"]:
        result = api.run(
            "evaluate.py",
            includes=["data.csv", "protocol.json", "model.py"],
            arguments=["evaluate", "--factory", f"model:{factory}"],
            options=options,
        )
        print(factory, result.id, result.state, result.result.metrics)
        if result.state != "succeeded":
            raise RuntimeError(result.model_dump_json())


if __name__ == "__main__":
    main()
