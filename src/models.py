"""
models.py
=========

Algorithm definitions and cross-validated training for the study.

Five algorithm families are evaluated on every dataset:

    Random Forest, Gradient Boosting, SVM, k-NN, Neural Network (MLP)

Each family has a classifier and a regressor variant so the same code path
handles binary, multiclass and regression tasks. Scale-sensitive models
(SVM, k-NN, MLP) are wrapped in a StandardScaler pipeline.

Evaluation uses 5-fold cross-validation:
  * StratifiedKFold for classification (preserves class balance per fold)
  * KFold for regression

Primary scores (higher is better):
  * classification -> accuracy
  * regression     -> R^2

Run directly to evaluate all algorithms on all datasets:

    python -m src.models
"""

from __future__ import annotations

import warnings
from typing import cast

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from .data_loader import REPO_ROOT, load_index, load_dataset

RESULTS_DIR = REPO_ROOT / "results"

N_FOLDS = 5
RANDOM_STATE = 42

# Canonical algorithm names (used as columns everywhere downstream).
ALGORITHMS = [
    "RandomForest",
    "GradientBoosting",
    "SVM",
    "kNN",
    "NeuralNetwork",
]


def _scaled(estimator) -> Pipeline:
    """Wrap a scale-sensitive estimator in a StandardScaler pipeline."""
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def build_estimators(task: str) -> dict:
    """
    Return a dict {algorithm_name: estimator} configured for the task type.

    Hyperparameters are modest, fixed defaults chosen for reasonable behaviour
    across small datasets without per-dataset tuning (tuning would confound the
    algorithm comparison).
    """
    is_clf = task in ("binary", "multiclass")

    if is_clf:
        return {
            "RandomForest": RandomForestClassifier(
                n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
            "GradientBoosting": GradientBoostingClassifier(
                random_state=RANDOM_STATE),
            "SVM": _scaled(SVC(kernel="rbf", random_state=RANDOM_STATE)),
            "kNN": _scaled(KNeighborsClassifier(n_neighbors=5)),
            "NeuralNetwork": _scaled(MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=1000,
                random_state=RANDOM_STATE)),
        }
    else:
        return {
            "RandomForest": RandomForestRegressor(
                n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
            "GradientBoosting": GradientBoostingRegressor(
                random_state=RANDOM_STATE),
            "SVM": _scaled(SVR(kernel="rbf")),
            "kNN": _scaled(KNeighborsRegressor(n_neighbors=5)),
            "NeuralNetwork": _scaled(MLPRegressor(
                hidden_layer_sizes=(64, 32), max_iter=1000,
                random_state=RANDOM_STATE)),
        }


def _cv_splitter(task: str):
    """Return the appropriate CV splitter for the task type."""
    if task in ("binary", "multiclass"):
        return StratifiedKFold(
            n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def evaluate_dataset(name: str, task: str) -> list[dict]:
    """
    Run 5-fold CV for all five algorithms on one dataset.

    Returns a list of long-format records: one row per (algorithm, fold),
    carrying the per-fold score plus the mean score for convenience.
    """
    X, y = load_dataset(name)
    Xv = X.to_numpy(dtype=float)
    yv = y.to_numpy()

    scoring = "accuracy" if task in ("binary", "multiclass") else "r2"
    splitter = _cv_splitter(task)
    estimators = build_estimators(task)

    records: list[dict] = []
    for algo_name, estimator in estimators.items():
        with warnings.catch_warnings():
            # MLP can warn about convergence on small data; keep output clean.
            warnings.simplefilter("ignore")
            fold_scores = cross_val_score(
                estimator, Xv, yv, cv=splitter, scoring=scoring, n_jobs=-1)
        mean_score = float(np.mean(fold_scores))
        for fold_idx, score in enumerate(fold_scores):
            records.append({
                "dataset": name,
                "task": task,
                "algorithm": algo_name,
                "fold": fold_idx,
                "metric": scoring,
                "score": float(score),
                "mean_score": mean_score,
            })
        print(f"    {algo_name:16s} {scoring}={mean_score:.4f}")
    return records


def run_all() -> pd.DataFrame:
    """
    Evaluate every algorithm on every dataset in the index.

    Writes two files to results/:
      * cv_results_long.csv  -- one row per (dataset, algorithm, fold)
      * performance_matrix.csv -- datasets x algorithms, mean CV score
    Returns the long-format DataFrame.
    """
    index = load_index()
    all_records: list[dict] = []

    for _, row in index.iterrows():
        name, task = str(row["dataset"]), str(row["task"])
        print(f"\n[{task}] {name}")
        all_records.extend(evaluate_dataset(name, task))

    long_df = pd.DataFrame(all_records)

    # Wide performance matrix: mean CV score per dataset x algorithm.
    matrix = cast(pd.DataFrame, (
        long_df.groupby(["dataset", "algorithm"])["score"].mean()
        .unstack("algorithm")
    ))[ALGORITHMS]
    # Keep dataset order aligned with the index.
    matrix = matrix.reindex(index["dataset"].tolist())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    long_path = RESULTS_DIR / "cv_results_long.csv"
    matrix_path = RESULTS_DIR / "performance_matrix.csv"
    long_df.to_csv(long_path, index=False)
    matrix.to_csv(matrix_path)

    print(f"\nLong results   -> {long_path}  ({len(long_df)} rows)")
    print(f"Perf. matrix   -> {matrix_path}  ({matrix.shape[0]} x "
          f"{matrix.shape[1]})")
    return long_df


def load_performance_matrix() -> pd.DataFrame:
    """Load the datasets x algorithms mean-score matrix."""
    path = RESULTS_DIR / "performance_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.models` first."
        )
    return pd.read_csv(path, index_col="dataset")


if __name__ == "__main__":
    run_all()
