"""Ordinary sklearn search reference, copied into both benchmark conditions."""

import hashlib
import importlib
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV


def main() -> None:
    began = time.monotonic()
    protocol = json.loads(Path("protocol.json").read_text())
    if (
        hashlib.sha256(Path("exploration.csv").read_bytes()).hexdigest()
        != protocol["dataset_sha256"]
    ):
        raise ValueError("Exploration data differs from the saved protocol.")
    binary = protocol["task"] == "binary"
    features = protocol["numeric"] + protocol["categorical"]

    def frame(path: str) -> pd.DataFrame:
        values = pd.read_csv(path, dtype=str, keep_default_na=False).set_index(
            protocol["id_column"]
        )
        for column in protocol["numeric"]:
            values[column] = pd.to_numeric(values[column].replace("", np.nan))
        for column in protocol["categorical"]:
            values[column] = values[column].replace("", np.nan)
        return values

    exploration, submission = frame("exploration.csv"), frame("submission.csv")
    target = (
        (exploration[protocol["target"]] == protocol["positive_label"]).astype(int)
        if binary
        else pd.to_numeric(exploration[protocol["target"]])
    )
    factory = "make_xgboost" if binary else "make_xgboost_regressor"
    estimator = getattr(importlib.import_module("model"), factory)(protocol["seed"])
    estimator.set_params(model__early_stopping_rounds=None)
    positions = {key: index for index, key in enumerate(exploration.index)}
    splits = [
        (
            np.array([positions[key] for key in split["train"]]),
            np.array([positions[key] for key in split["score"]]),
        )
        for split in protocol["splits"]
    ]
    search = RandomizedSearchCV(
        estimator,
        {
            "model__n_estimators": [50, 100],
            "model__max_depth": [2, 4],
            "model__learning_rate": [0.05, 0.1],
        },
        n_iter=4,
        cv=splits,
        scoring="roc_auc" if binary else "neg_root_mean_squared_error",
        n_jobs=1,
        random_state=protocol["seed"],
        error_score="raise",
    )
    search.fit(exploration[features], target)
    selected = search.best_estimator_
    predictions = (
        selected.predict_proba(submission[features])[:, 1]
        if binary
        else selected.predict(submission[features])
    )
    output = Path(os.environ.get("MLEF_OUTPUT_DIR", "reference-output"))
    output.mkdir(exist_ok=True)
    with (output / "predictions.csv").open("x", encoding="utf-8", newline="") as stream:
        stream.write("id,prediction\n")
        for identifier, value in zip(submission.index, predictions, strict=True):
            stream.write(f"{identifier},{float(value)}\n")
    joblib.dump(selected, output / "model.joblib")
    (output / "reference.json").write_text(
        json.dumps(
            {
                "search_candidates": 4,
                "folds": len(splits),
                "best_parameters": search.best_params_,
                "validation_score": float(search.best_score_) * (1 if binary else -1),
                "seconds": time.monotonic() - began,
                "limitation": "A conventional search reference, not a measured LLM trial.",
            },
            default=str,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
