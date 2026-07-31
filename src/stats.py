"""
stats.py
========

Statistical comparison of the five algorithms across datasets.

Tests implemented (the standard protocol for comparing classifiers over
multiple datasets, following Demsar 2006):

  * Friedman test  -- omnibus test for whether the algorithms differ in
    their per-dataset ranks. If significant, we proceed to a post-hoc test.

  * Nemenyi post-hoc -- all-pairs comparison. Computes the Critical Difference
    (CD): two algorithms differ significantly if their average ranks differ by
    more than CD. The CD value feeds the Critical Difference diagram (Figure 1).

  * Wilcoxon signed-rank -- pairwise comparison of the best algorithm against
    each competitor, using per-dataset performance scores.

Run directly:

    python -m src.stats
"""

from __future__ import annotations

import json
from itertools import combinations
from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, studentized_range

from .data_loader import REPO_ROOT
from .models import load_performance_matrix, ALGORITHMS

RESULTS_DIR = REPO_ROOT / "results"


def friedman_test(perf: pd.DataFrame) -> dict:
    """
    Friedman test across algorithms. Each column is an algorithm; each row a
    dataset. Returns the statistic and p-value.
    """
    samples = [perf[algo].to_numpy() for algo in ALGORITHMS]
    stat, p = friedmanchisquare(*samples)
    return {"statistic": float(stat), "p_value": float(p),
            "n_datasets": int(perf.shape[0]),
            "n_algorithms": len(ALGORITHMS)}


def nemenyi_critical_difference(n_algorithms: int, n_datasets: int,
                                alpha: float = 0.05) -> float:
    """
    Critical Difference for the Nemenyi test:

        CD = q_alpha * sqrt( k(k+1) / (6N) )

    where k = number of algorithms, N = number of datasets, and q_alpha is the
    critical value of the Studentized range statistic divided by sqrt(2).
    """
    k, N = n_algorithms, n_datasets
    # Studentized range critical value; divide by sqrt(2) per the Nemenyi test.
    q_alpha = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * N))
    return float(cd)


def nemenyi_posthoc(ranks: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    Nemenyi all-pairs comparison. Returns average ranks, the CD value, and a
    table of pairwise average-rank differences flagged as significant when the
    difference exceeds CD.
    """
    avg_ranks = ranks.mean(axis=0)
    k, N = len(ALGORITHMS), ranks.shape[0]
    cd = nemenyi_critical_difference(k, N, alpha)

    pairs = []
    for a, b in combinations(ALGORITHMS, 2):
        diff = abs(avg_ranks[a] - avg_ranks[b])
        pairs.append({
            "algorithm_a": a,
            "algorithm_b": b,
            "rank_diff": float(diff),
            "significant": bool(diff > cd),
        })

    return {
        "critical_difference": cd,
        "alpha": alpha,
        "average_ranks": {a: float(avg_ranks[a]) for a in ALGORITHMS},
        "pairwise": pairs,
    }


def wilcoxon_vs_best(perf: pd.DataFrame, avg_ranks: pd.Series) -> list[dict]:
    """
    Wilcoxon signed-rank test comparing the top-ranked algorithm against each
    other algorithm, using per-dataset scores.
    """
    best = str(avg_ranks.idxmin())  # lowest average rank = best
    results = []
    for algo in ALGORITHMS:
        if algo == best:
            continue
        x = perf[best].to_numpy()
        y = perf[algo].to_numpy()
        # Guard the all-equal case where Wilcoxon is undefined.
        try:
            stat, p = wilcoxon(x, y)
            stat, p = float(stat), float(p)
        except ValueError:
            stat, p = float("nan"), 1.0
        results.append({
            "best_algorithm": best,
            "compared_with": algo,
            "statistic": stat,
            "p_value": p,
            "best_mean_score": float(np.mean(x)),
            "other_mean_score": float(np.mean(y)),
        })
    return results


def run_all_tests(alpha: float = 0.05) -> dict:
    """
    Run Friedman + Nemenyi + Wilcoxon and persist a JSON summary plus
    pairwise CSVs. Returns the summary dict.
    """
    perf = cast(pd.DataFrame, load_performance_matrix()[ALGORITHMS])
    ranks = (-perf).rank(axis=1, method="average")
    avg_ranks = cast(pd.Series, ranks.mean(axis=0))

    friedman = friedman_test(perf)
    nemenyi = nemenyi_posthoc(ranks, alpha)
    wilcox = wilcoxon_vs_best(perf, avg_ranks)

    summary = {
        "friedman": friedman,
        "nemenyi": nemenyi,
        "wilcoxon_vs_best": wilcox,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "statistical_tests.json").write_text(
        json.dumps(summary, indent=2))
    pd.DataFrame(nemenyi["pairwise"]).to_csv(
        RESULTS_DIR / "nemenyi_pairwise.csv", index=False)
    pd.DataFrame(wilcox).to_csv(
        RESULTS_DIR / "wilcoxon_vs_best.csv", index=False)

    # --- Console report ---------------------------------------------------
    print("Friedman test:")
    print(f"  chi^2 = {friedman['statistic']:.4f}, "
          f"p = {friedman['p_value']:.4g} "
          f"(N={friedman['n_datasets']}, k={friedman['n_algorithms']})")
    verdict = ("significant -> algorithms differ"
               if friedman["p_value"] < alpha
               else "not significant")
    print(f"  {verdict} at alpha={alpha}")

    print(f"\nNemenyi critical difference (alpha={alpha}): "
          f"{nemenyi['critical_difference']:.4f}")
    sig_pairs = [p for p in nemenyi["pairwise"] if p["significant"]]
    print(f"  {len(sig_pairs)} of {len(nemenyi['pairwise'])} pairs "
          f"significantly different")
    for p in sig_pairs:
        print(f"    {p['algorithm_a']} vs {p['algorithm_b']}: "
              f"diff={p['rank_diff']:.3f}")

    best = str(avg_ranks.idxmin())
    print(f"\nWilcoxon signed-rank vs best ({best}):")
    for w in wilcox:
        mark = "*" if w["p_value"] < alpha else " "
        print(f"  {mark} vs {w['compared_with']:16s} p={w['p_value']:.4g}")

    print(f"\nStatistical artefacts -> {RESULTS_DIR}")
    return summary


if __name__ == "__main__":
    run_all_tests()
