"""
meta_learner.py
===============

The meta-learning model: predict the best algorithm for a dataset from its
meta-features alone.

This is the empirical centrepiece of the study. Inputs are the meta-feature
matrix (src.meta_features) and the per-dataset best-algorithm labels
(src.ranking). A Random Forest meta-classifier maps meta-features -> best
algorithm.

Because we only have 20 datasets, evaluation uses leave-one-out
cross-validation (LOO-CV): train on 19 datasets, predict the held-out one,
repeat. We report:

  * LOO accuracy of the meta-learner
  * a majority-class baseline (always predict the most common best algorithm)
    so the meta-learner's lift is interpretable
  * per-fold predictions (predicted vs actual best algorithm)
  * meta-feature importance (Table 2), from a model fit on all datasets

Run directly:

    python -m src.meta_learner
"""

from __future__ import annotations

import json
from typing import cast

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut

from .data_loader import REPO_ROOT
from .meta_features import load_meta_feature_matrix

RESULTS_DIR = REPO_ROOT / "results"
RANDOM_STATE = 42

# Meta-feature columns to feed the meta-learner (everything except labels).
NON_FEATURE_COLS = ["task"]


def _load_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Assemble the meta-learning table: meta-features (X) aligned with the
    best-algorithm label (y) for each dataset.
    """
    mf = load_meta_feature_matrix()
    best = pd.read_csv(RESULTS_DIR / "best_algorithm.csv", index_col="dataset")

    X = mf.drop(columns=[c for c in NON_FEATURE_COLS if c in mf.columns])
    # Encode task type numerically so it can serve as a meta-feature too.
    if "task" in mf.columns:
        task_map = {"binary": 0, "multiclass": 1, "regression": 2}
        X = X.assign(
            task_code=mf["task"].map(lambda t: task_map[t])
        )
    y = cast(pd.Series, best.loc[X.index, "best_algorithm"])
    return X, y


def _build_meta_classifier() -> RandomForestClassifier:
    """Random Forest meta-classifier with modest, fixed hyperparameters."""
    return RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def leave_one_out_evaluation(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Leave-one-out CV of the meta-learner. Returns accuracy, the baseline
    accuracy, and a per-fold prediction table.
    """
    loo = LeaveOneOut()
    Xv = X.to_numpy(dtype=float)
    yv = y.to_numpy()

    predictions: list[dict] = []
    correct = 0
    for train_idx, test_idx in loo.split(Xv):
        clf = _build_meta_classifier()
        clf.fit(Xv[train_idx], yv[train_idx])
        pred = clf.predict(Xv[test_idx])[0]
        actual = yv[test_idx][0]
        is_correct = bool(pred == actual)
        correct += int(is_correct)
        predictions.append({
            "dataset": X.index[int(test_idx[0])],
            "actual_best": str(actual),
            "predicted_best": str(pred),
            "correct": is_correct,
        })

    n = len(yv)
    accuracy = correct / n

    # Majority-class baseline: always predict the most frequent best algorithm.
    majority_label = y.value_counts().idxmax()
    baseline_accuracy = float((y == majority_label).mean())

    return {
        "loo_accuracy": accuracy,
        "baseline_accuracy": baseline_accuracy,
        "majority_label": str(majority_label),
        "n_datasets": n,
        "predictions": predictions,
    }


def feature_importance(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Meta-feature importance from a Random Forest fit on all datasets.
    Produces Table 2 (meta-feature importance).
    """
    clf = _build_meta_classifier()
    clf.fit(X.to_numpy(dtype=float), y.to_numpy())
    importance = pd.DataFrame({
        "meta_feature": X.columns,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return importance


def run_meta_learner() -> dict:
    """
    Train and evaluate the meta-learner, persist artefacts, print a report.
    """
    X, y = _load_training_data()

    evaluation = leave_one_out_evaluation(X, y)
    importance = feature_importance(X, y)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(evaluation["predictions"]).to_csv(
        RESULTS_DIR / "meta_learner_predictions.csv", index=False)
    importance.to_csv(
        RESULTS_DIR / "meta_feature_importance.csv", index=False)

    summary = {k: v for k, v in evaluation.items() if k != "predictions"}
    summary["top_meta_features"] = importance.head(5).to_dict(orient="records")
    (RESULTS_DIR / "meta_learner_summary.json").write_text(
        json.dumps(summary, indent=2))

    # --- Console report ---------------------------------------------------
    print("Meta-learner (predict best algorithm from meta-features)")
    print(f"  LOO-CV accuracy   : {evaluation['loo_accuracy']:.3f}"
          f"  ({int(evaluation['loo_accuracy'] * evaluation['n_datasets'])}"
          f"/{evaluation['n_datasets']} correct)")
    print(f"  Baseline (majority '{evaluation['majority_label']}'): "
          f"{evaluation['baseline_accuracy']:.3f}")
    lift = evaluation["loo_accuracy"] - evaluation["baseline_accuracy"]
    print(f"  Lift over baseline: {lift:+.3f}")

    print("\nTop meta-features (importance):")
    for _, r in importance.head(6).iterrows():
        print(f"    {r['meta_feature']:32s} {r['importance']:.4f}")

    print(f"\nMeta-learner artefacts -> {RESULTS_DIR}")
    return summary


if __name__ == "__main__":
    run_meta_learner()
