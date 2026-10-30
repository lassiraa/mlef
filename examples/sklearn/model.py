"""Edit normal Python to test a new model, feature or preprocessing hypothesis."""

import importlib

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def _preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, numeric_columns),
            ("categorical", categorical, categorical_columns),
        ]
    )


def _pipeline(seed: int, name: str) -> BaseEstimator:
    preprocess = _preprocessor(["radius", "texture", "smoothness"], ["radius_band"])
    estimators = {
        "dummy": DummyClassifier(strategy="prior"),
        "linear": LogisticRegression(max_iter=500, random_state=seed),
        "tree": DecisionTreeClassifier(max_depth=4, random_state=seed),
    }
    return Pipeline([("preprocess", preprocess), ("model", estimators[name])])


def make_estimator(seed: int) -> BaseEstimator:
    return _pipeline(seed, "linear")


def make_dummy(seed: int) -> BaseEstimator:
    return _pipeline(seed, "dummy")


def make_tree(seed: int) -> BaseEstimator:
    return _pipeline(seed, "tree")


def make_xgboost(seed: int) -> BaseEstimator:
    pipeline = _pipeline(seed, "linear")
    xgb = importlib.import_module("xgboost")
    classifier = getattr(xgb, "XGBClassifier")  # noqa: B009 - optional runtime-loaded dependency
    pipeline.set_params(
        model=classifier(
            tree_method="hist",
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=seed,
            n_jobs=1,
            early_stopping_rounds=10,
            eval_metric="logloss",
        )
    )
    return pipeline


def make_xgboost_candidate(seed: int) -> BaseEstimator:
    pipeline = make_xgboost(seed)
    pipeline.set_params(
        model__max_depth=2, model__reg_lambda=5.0, preprocess__numeric__impute__add_indicator=True
    )
    return pipeline


def make_regressor(seed: int) -> BaseEstimator:
    return Pipeline(
        [("preprocess", _preprocessor(["amount", "age", "tenure"], ["region"])), ("model", Ridge())]
    )


def make_dummy_regressor(seed: int) -> BaseEstimator:
    pipeline = make_regressor(seed)
    pipeline.set_params(model=DummyRegressor())
    return pipeline


def make_xgboost_regressor(seed: int) -> BaseEstimator:
    xgb = importlib.import_module("xgboost")
    regressor = getattr(xgb, "XGBRegressor")  # noqa: B009 - optional runtime-loaded dependency
    pipeline = make_regressor(seed)
    pipeline.set_params(
        model=regressor(
            tree_method="hist",
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=seed,
            n_jobs=1,
            early_stopping_rounds=10,
            eval_metric="rmse",
        )
    )
    return pipeline
