"""Edit normal Python to test a new model, feature or preprocessing hypothesis."""

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def _pipeline(seed: int, name: str) -> BaseEstimator:
    numeric = Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric, ["radius", "texture", "smoothness"]),
            ("categorical", categorical, ["radius_band"]),
        ]
    )
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
