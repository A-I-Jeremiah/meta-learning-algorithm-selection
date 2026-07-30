"""
data_loader.py
==============

Dataset acquisition for the meta-learning study.

Selects a balanced set of ~20 datasets from the Penn Machine Learning
Benchmark (PMLB), downloads them, harmonises them into tidy CSVs with a
single ``target`` column, and writes an index describing each dataset and
its learning task (binary / multiclass / regression).

The index produced here (``data/processed/dataset_index.csv``) is the entry
point for the rest of the pipeline: meta-feature extraction, model training,
ranking and statistical tests all iterate over it.

Run directly to download everything:

    python -m src.data_loader

PMLB caches downloads locally, so re-running is fast and offline-friendly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pandas as pd
from pmlb import (
    classification_dataset_names,
    regression_dataset_names,
    fetch_data,
)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# Repository layout: this file lives in <repo>/src/, data lives in <repo>/data/
REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
METADATA_DIR = REPO_ROOT / "data" / "metadata"

# How many datasets of each task type we want (totals 20).
QUOTAS = {"binary": 8, "multiclass": 7, "regression": 5}

# Keep datasets small so five algorithms x 5-fold CV run quickly on a laptop.
MAX_INSTANCES = 3000
MAX_FEATURES = 60
MIN_INSTANCES = 100  # avoid tiny datasets where 5-fold CV is unstable

RANDOM_STATE = 42


# --------------------------------------------------------------------------
# Task-type detection
# --------------------------------------------------------------------------

def _classification_task_type(y: pd.Series) -> str:
    """Return 'binary' or 'multiclass' from the target column."""
    n_classes = y.nunique()
    return "binary" if n_classes == 2 else "multiclass"


def _passes_size_filter(df: pd.DataFrame) -> bool:
    """Keep only datasets small enough to train quickly."""
    n_instances, n_cols = df.shape
    n_features = n_cols - 1  # last column is the target
    return (
        MIN_INSTANCES <= n_instances <= MAX_INSTANCES
        and n_features <= MAX_FEATURES
    )


# --------------------------------------------------------------------------
# Selection + download
# --------------------------------------------------------------------------

def select_and_download(quotas: dict[str, int] | None = None) -> pd.DataFrame:
    """
    Select a balanced set of PMLB datasets, download them, save harmonised
    CSVs and return the dataset index as a DataFrame.

    Candidates are examined one at a time; a dataset is accepted only if it
    passes the size filter and its task type still has an unfilled quota.
    """
    quotas = dict(quotas or QUOTAS)
    remaining = dict(quotas)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict] = []

    # Candidate pools. Classification names cover both binary and multiclass;
    # we discover which by inspecting the target after download.
    candidate_pools = [
        ("classification", classification_dataset_names),
        ("regression", regression_dataset_names),
    ]

    for family, names in candidate_pools:
        for name in names:
            # Stop early once every quota is filled.
            if all(v <= 0 for v in remaining.values()):
                break

            # Skip whole family if its quota(s) are already met.
            if family == "regression" and remaining["regression"] <= 0:
                break
            if family == "classification" and (
                remaining["binary"] <= 0 and remaining["multiclass"] <= 0
            ):
                continue

            try:
                df = cast(pd.DataFrame, fetch_data(name))
            except Exception as exc:  # network hiccup, bad file, etc.
                print(f"  skip {name}: download failed ({exc})")
                continue

            if not _passes_size_filter(df):
                continue

            y = cast(pd.Series, df["target"])
            if family == "regression":
                task = "regression"
            else:
                task = _classification_task_type(y)

            if remaining.get(task, 0) <= 0:
                continue  # this task type is already full

            # Accept it.
            remaining[task] -= 1
            out_path = PROCESSED_DIR / f"{name}.csv"
            df.to_csv(out_path, index=False)

            n_instances, n_cols = df.shape
            index_rows.append(
                {
                    "dataset": name,
                    "task": task,
                    "n_instances": int(n_instances),
                    "n_features": int(n_cols - 1),
                    "n_classes": int(y.nunique()) if task != "regression" else 0,
                    "target_column": "target",
                    "source": "PMLB",
                    "file": f"data/processed/{name}.csv",
                }
            )
            filled = quotas[task] - remaining[task]
            print(f"  [{task:10s} {filled}/{quotas[task]}] {name} "
                  f"({n_instances} rows, {n_cols - 1} features)")

    index = pd.DataFrame(index_rows).sort_values(["task", "dataset"])
    _write_index(index, quotas, remaining)
    return index


def _write_index(index: pd.DataFrame, quotas: dict, remaining: dict) -> None:
    """Persist the dataset index as CSV + JSON and print a summary."""
    index_csv = PROCESSED_DIR / "dataset_index.csv"
    index_json = METADATA_DIR / "dataset_index.json"
    index.to_csv(index_csv, index=False)
    index_json.write_text(json.dumps(index.to_dict(orient="records"), indent=2))

    print("\n" + "=" * 60)
    print(f"Downloaded {len(index)} datasets -> {PROCESSED_DIR}")
    for task in quotas:
        got = quotas[task] - remaining[task]
        flag = "OK" if remaining[task] <= 0 else f"SHORT by {remaining[task]}"
        print(f"  {task:12s}: {got}/{quotas[task]}  [{flag}]")
    print(f"Index written to {index_csv}")
    print("=" * 60)


# --------------------------------------------------------------------------
# Loading helpers (used by the rest of the pipeline)
# --------------------------------------------------------------------------

def load_index() -> pd.DataFrame:
    """Load the dataset index produced by ``select_and_download``."""
    index_csv = PROCESSED_DIR / "dataset_index.csv"
    if not index_csv.exists():
        raise FileNotFoundError(
            f"{index_csv} not found. Run `python -m src.data_loader` first."
        )
    return pd.read_csv(index_csv)


def load_dataset(name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load one processed dataset, returning (X, y)."""
    path = PROCESSED_DIR / f"{name}.csv"
    df = pd.read_csv(path)
    X = df.drop(columns=["target"])
    y = cast(pd.Series, df["target"])
    return X, y


if __name__ == "__main__":
    select_and_download()
