"""
ranking.py
==========

Rank algorithms per dataset and build the meta-learning target.

Two views are produced from the performance matrix (datasets x algorithms):

  1. Per-dataset ranks (1 = best). Ranks are comparable across tasks even
     though the raw metrics differ (accuracy vs R^2), because ranking is
     scale-free. These feed the Friedman/Nemenyi tests.

  2. Normalised scores in [0, 1] within each dataset (min-max across the five
     algorithms), so a value of 1.0 marks the best algorithm on that dataset
     and 0.0 the worst. Useful for the complexity-performance scatter.

The best algorithm per dataset (rank 1) is the classification target the
meta-learner tries to predict from meta-features.

Run directly:

    python -m src.ranking
"""

from __future__ import annotations

from typing import cast

import pandas as pd

from .data_loader import REPO_ROOT
from .models import load_performance_matrix, ALGORITHMS

RESULTS_DIR = REPO_ROOT / "results"


def compute_ranks(perf: pd.DataFrame) -> pd.DataFrame:
    """
    Per-dataset ranks (1 = best). Higher score is better, so we rank the
    negated scores ascending. Ties get the average rank.
    """
    ranks = (-perf).rank(axis=1, method="average")
    return ranks


def compute_normalised_scores(perf: pd.DataFrame) -> pd.DataFrame:
    """
    Min-max normalise scores within each dataset (row). 1.0 = best algorithm
    on that dataset, 0.0 = worst. Rows where all scores are equal map to 1.0.
    """
    row_min = perf.min(axis=1)
    row_max = perf.max(axis=1)
    span = (row_max - row_min).replace(0, pd.NA)
    normalised = perf.sub(row_min, axis=0).div(span, axis=0)
    return normalised.fillna(1.0)


def best_algorithm_per_dataset(perf: pd.DataFrame) -> pd.Series:
    """Return the name of the highest-scoring algorithm for each dataset."""
    best = cast(pd.Series, perf[ALGORITHMS].idxmax(axis=1))
    best.name = "best_algorithm"
    return best


def average_ranks(ranks: pd.DataFrame) -> pd.Series:
    """Mean rank of each algorithm across all datasets (lower = better)."""
    avg = cast(pd.Series, ranks.mean(axis=0)).sort_values()
    avg.name = "average_rank"
    return avg


def build_rankings() -> dict:
    """
    Compute ranks, normalised scores, best-algorithm target and average
    ranks. Persist each to results/ and return them in a dict.
    """
    perf = cast(pd.DataFrame, load_performance_matrix()[ALGORITHMS])

    ranks = compute_ranks(perf)
    normalised = compute_normalised_scores(perf)
    best = best_algorithm_per_dataset(perf)
    avg_ranks = average_ranks(ranks)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ranks.to_csv(RESULTS_DIR / "ranks.csv")
    normalised.to_csv(RESULTS_DIR / "normalised_scores.csv")
    best.to_frame().to_csv(RESULTS_DIR / "best_algorithm.csv")
    avg_ranks.to_frame().to_csv(RESULTS_DIR / "average_ranks.csv")

    print("Average ranks (lower is better):")
    print(avg_ranks.to_string())
    print("\nBest-algorithm counts:")
    print(best.value_counts().to_string())
    print(f"\nRanking artefacts -> {RESULTS_DIR}")

    return {
        "ranks": ranks,
        "normalised": normalised,
        "best_algorithm": best,
        "average_ranks": avg_ranks,
    }


if __name__ == "__main__":
    build_rankings()
