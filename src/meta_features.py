"""
meta_features.py
================

Extract dataset meta-features for the meta-learning study.

For every dataset in the processed index we compute a fixed vector of
meta-features describing its size, dimensionality, class structure and simple
statistical properties. These vectors are the inputs to the meta-learner,
which predicts the best algorithm for a dataset from its meta-features alone.

The features are deliberately cheap to compute and defined for both
classification and regression tasks (class-specific features are set to a
neutral value for regression).

Run directly to build the full meta-feature matrix:

    python -m src.meta_features
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loader import REPO_ROOT, load_index, load_dataset

RESULTS_DIR = REPO_ROOT / "results"


def extract_meta_features(X: pd.DataFrame, y: pd.Series, task: str) -> dict:
    """
    Compute the meta-feature vector for a single dataset.

    Parameters
    ----------
    X : feature matrix (numeric).
    y : target column.
    task : one of 'binary', 'multiclass', 'regression'.
    """
    n_instances, n_features = X.shape
    Xv = X.to_numpy(dtype=float)

    # --- Size / shape -----------------------------------------------------
    feats: dict[str, float] = {
        "n_instances": float(n_instances),
        "n_features": float(n_features),
        # log transforms tame the wide dynamic range across datasets
        "log_n_instances": float(np.log10(n_instances)),
        "log_n_features": float(np.log10(n_features)),
        # feature-to-instance ratio (dimensionality pressure)
        "feature_to_instance_ratio": float(n_features / n_instances),
    }

    # --- Class structure (classification only) ----------------------------
    if task in ("binary", "multiclass"):
        class_counts = y.value_counts()
        n_classes = int(class_counts.shape[0])
        proportions = (class_counts / n_instances).to_numpy()
        # imbalance ratio: largest class size / smallest class size (>= 1)
        imbalance_ratio = float(class_counts.max() / class_counts.min())
        # normalised class entropy (1.0 == perfectly balanced)
        entropy = float(-np.sum(proportions * np.log(proportions)))
        norm_entropy = entropy / np.log(n_classes) if n_classes > 1 else 0.0
    else:
        n_classes = 0
        imbalance_ratio = 1.0
        norm_entropy = 1.0

    feats["n_classes"] = float(n_classes)
    feats["class_imbalance_ratio"] = imbalance_ratio
    feats["normalised_class_entropy"] = float(norm_entropy)

    # --- Simple statistical descriptors -----------------------------------
    # Averaged over features so the vector length is fixed across datasets.
    with np.errstate(all="ignore"):
        means = Xv.mean(axis=0)
        stds = Xv.std(axis=0)
        # coefficient of variation, guarding divide-by-zero
        cv = np.where(np.abs(means) > 1e-12, stds / means, 0.0)
        # per-feature skewness and kurtosis
        centred = Xv - means
        with np.errstate(divide="ignore", invalid="ignore"):
            skew = np.where(stds > 1e-12,
                            (centred ** 3).mean(axis=0) / (stds ** 3), 0.0)
            kurt = np.where(stds > 1e-12,
                            (centred ** 4).mean(axis=0) / (stds ** 4) - 3.0, 0.0)

    feats["mean_feature_std"] = float(np.nanmean(stds))
    feats["mean_abs_coef_variation"] = float(np.nanmean(np.abs(cv)))
    feats["mean_skewness"] = float(np.nanmean(skew))
    feats["mean_kurtosis"] = float(np.nanmean(kurt))

    # --- Feature redundancy -----------------------------------------------
    # Mean absolute pairwise correlation between features (0 when <2 features).
    if n_features > 1:
        corr = np.corrcoef(Xv, rowvar=False)
        # off-diagonal absolute mean
        mask = ~np.eye(n_features, dtype=bool)
        mean_abs_corr = float(np.nanmean(np.abs(corr[mask])))
    else:
        mean_abs_corr = 0.0
    feats["mean_abs_feature_correlation"] = mean_abs_corr

    return feats


def build_meta_feature_matrix() -> pd.DataFrame:
    """
    Extract meta-features for every dataset in the index and return the
    matrix (one row per dataset). Also written to results/meta_features.csv.
    """
    index = load_index()
    rows: list[dict] = []

    for _, row in index.iterrows():
        name, task = str(row["dataset"]), str(row["task"])
        X, y = load_dataset(name)
        feats = extract_meta_features(X, y, task)
        feats = {"dataset": name, "task": task, **feats}
        rows.append(feats)
        print(f"  meta-features: {name} ({task})")

    matrix = pd.DataFrame(rows).set_index("dataset")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "meta_features.csv"
    matrix.to_csv(out_path)
    print(f"\nMeta-feature matrix: {matrix.shape[0]} datasets x "
          f"{matrix.shape[1] - 1} features -> {out_path}")
    return matrix


def load_meta_feature_matrix() -> pd.DataFrame:
    """Load the meta-feature matrix produced by ``build_meta_feature_matrix``."""
    path = RESULTS_DIR / "meta_features.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.meta_features` first."
        )
    return pd.read_csv(path, index_col="dataset")


if __name__ == "__main__":
    build_meta_feature_matrix()
