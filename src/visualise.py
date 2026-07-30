"""
visualise.py
============

Publication-ready tables and figures for the study. Every artefact is built
from the committed files in results/ so it stays in sync with the analysis.

Outputs (written to figures/):
  * table1_dataset_characteristics.csv / .md  -- Table 1
  * table2_meta_feature_importance.csv / .md  -- Table 2
  * figure1_critical_difference.png           -- Figure 1 (Nemenyi CD)
  * figure2_complexity_performance.png        -- Figure 2 (scatter)
  * figure3_performance_boxplot.png           -- extra: score distributions
  * figure4_rank_heatmap.png                  -- extra: per-dataset ranks

Run directly:

    python -m src.visualise
"""

from __future__ import annotations

import json
from typing import cast

import matplotlib
matplotlib.use("Agg")  # headless backend; no display needed
import matplotlib.pyplot as plt
import pandas as pd

from .data_loader import REPO_ROOT
from .models import ALGORITHMS

RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
DPI = 300

plt.rcParams.update({
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_stats() -> dict:
    return json.loads((RESULTS_DIR / "statistical_tests.json").read_text())


def _load_perf() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "performance_matrix.csv",
                     index_col="dataset")
    return cast(pd.DataFrame, df[ALGORITHMS])


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def table1_dataset_characteristics() -> pd.DataFrame:
    """
    Table 1: dataset characteristics (name, task, instances, features,
    classes) joined with each dataset's best algorithm.
    """
    index = pd.read_csv(REPO_ROOT / "data" / "processed" / "dataset_index.csv")
    best = pd.read_csv(RESULTS_DIR / "best_algorithm.csv")

    table = index[["dataset", "task", "n_instances", "n_features",
                   "n_classes"]].merge(best, on="dataset", how="left")
    table = table.rename(columns={
        "dataset": "Dataset",
        "task": "Task",
        "n_instances": "Instances",
        "n_features": "Features",
        "n_classes": "Classes",
        "best_algorithm": "Best algorithm",
    }).sort_values(["Task", "Dataset"]).reset_index(drop=True)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(FIGURES_DIR / "table1_dataset_characteristics.csv",
                 index=False)
    (FIGURES_DIR / "table1_dataset_characteristics.md").write_text(
        table.to_markdown(index=False) or "")
    return table


def table2_meta_feature_importance() -> pd.DataFrame:
    """Table 2: meta-feature importance from the meta-learner."""
    imp = pd.read_csv(RESULTS_DIR / "meta_feature_importance.csv")
    imp = imp.rename(columns={
        "meta_feature": "Meta-feature",
        "importance": "Importance",
    })
    imp["Importance"] = imp["Importance"].round(4)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    imp.to_csv(FIGURES_DIR / "table2_meta_feature_importance.csv", index=False)
    (FIGURES_DIR / "table2_meta_feature_importance.md").write_text(
        imp.to_markdown(index=False) or "")
    return imp


# ---------------------------------------------------------------------------
# Figure 1: Critical Difference (Nemenyi) diagram
# ---------------------------------------------------------------------------

def figure1_critical_difference() -> None:
    """
    Critical Difference diagram (Demsar 2006 style): algorithms placed on an
    axis by average rank (best = lowest, on the left). Algorithms whose ranks
    differ by less than the CD are connected by a bar (not significantly
    different).
    """
    stats = _load_stats()
    avg_ranks = stats["nemenyi"]["average_ranks"]
    cd = stats["nemenyi"]["critical_difference"]

    algos = sorted(avg_ranks, key=lambda a: avg_ranks[a])  # best first
    ranks = [avg_ranks[a] for a in algos]
    k = len(algos)
    lo, hi = 1, k  # rank axis bounds

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Top axis line with rank ticks.
    axis_y = 0.8
    ax.plot([lo, hi], [axis_y, axis_y], "k-", lw=1.5)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [axis_y, axis_y + 0.03], "k-", lw=1.2)
        ax.text(r, axis_y + 0.06, str(r), ha="center", va="bottom",
                fontsize=9)

    # Place each algorithm; alternate label heights left/right for clarity.
    n_left = (k + 1) // 2
    for i, (algo, r) in enumerate(zip(algos, ranks)):
        if i < n_left:  # label to the left
            label_x, drop = lo - 0.4, axis_y - 0.12 - 0.12 * i
            ax.plot([r, r, label_x], [axis_y, drop, drop], "k-", lw=1.0)
            ax.text(label_x - 0.05, drop, f"{algo} ({r:.2f})",
                    ha="right", va="center", fontsize=9)
        else:  # label to the right
            j = i - n_left
            label_x, drop = hi + 0.4, axis_y - 0.12 - 0.12 * j
            ax.plot([r, r, label_x], [axis_y, drop, drop], "k-", lw=1.0)
            ax.text(label_x + 0.05, drop, f"{algo} ({r:.2f})",
                    ha="left", va="center", fontsize=9)

    # CD bar (scale reference) in the upper-left.
    bar_y = axis_y + 0.16
    ax.plot([lo, lo + cd], [bar_y, bar_y], "k-", lw=2.5)
    ax.plot([lo, lo], [bar_y - 0.02, bar_y + 0.02], "k-", lw=1.5)
    ax.plot([lo + cd, lo + cd], [bar_y - 0.02, bar_y + 0.02], "k-", lw=1.5)
    ax.text(lo + cd / 2, bar_y + 0.04, f"CD = {cd:.2f}",
            ha="center", va="bottom", fontsize=9)

    # Cliques: connect algorithms that are NOT significantly different.
    clique_y = axis_y - 0.05
    used_levels = 0
    i = 0
    while i < k:
        j = i
        while j + 1 < k and (ranks[j + 1] - ranks[i]) < cd:
            j += 1
        if j > i:  # a group of >=2 non-different algorithms
            y = clique_y - 0.03 * used_levels
            ax.plot([ranks[i] - 0.05, ranks[j] + 0.05], [y, y],
                    "-", lw=3, color="crimson", solid_capstyle="round")
            used_levels += 1
        i += 1

    ax.set_title("Figure 1. Critical Difference diagram (Nemenyi, "
                 f"alpha=0.05)\nlower average rank = better",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure1_critical_difference.png",
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Complexity vs performance scatter
# ---------------------------------------------------------------------------

def figure2_complexity_performance() -> None:
    """
    Complexity-performance scatter: dataset complexity (n_instances x
    n_features, log scale) vs the best mean score achieved on it, coloured by
    the winning algorithm.
    """
    meta = pd.read_csv(RESULTS_DIR / "meta_features.csv")
    perf = _load_perf()
    best_score = perf.max(axis=1)
    best_algo = perf.idxmax(axis=1)

    meta = meta.set_index("dataset")
    complexity = meta["n_instances"] * meta["n_features"]

    df = pd.DataFrame({
        "complexity": complexity,
        "best_score": best_score.reindex(meta.index),
        "best_algo": best_algo.reindex(meta.index),
        "task": meta["task"],
    }).dropna()

    fig, ax = plt.subplots(figsize=(7.5, 5))
    algos = sorted(df["best_algo"].unique())
    cmap = plt.get_cmap("tab10")
    for i, algo in enumerate(algos):
        sub = df[df["best_algo"] == algo]
        ax.scatter(sub["complexity"], sub["best_score"],
                   label=algo, s=70, color=cmap(i),
                   edgecolor="black", linewidth=0.5, alpha=0.85)

    ax.set_xscale("log")
    ax.set_xlabel("Dataset complexity  (instances x features, log scale)")
    ax.set_ylabel("Best mean score achieved  (accuracy or R2)")
    ax.set_title("Figure 2. Complexity vs best-achieved performance")
    ax.legend(title="Best algorithm", fontsize=8, title_fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure2_complexity_performance.png",
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Extra figures
# ---------------------------------------------------------------------------

def figure3_performance_boxplot() -> None:
    """Distribution of per-dataset scores for each algorithm."""
    perf = _load_perf()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    data = [perf[a].to_numpy() for a in ALGORITHMS]
    ax.boxplot(data, tick_labels=ALGORITHMS, showmeans=True)
    ax.set_ylabel("Mean CV score per dataset")
    ax.set_title("Figure 3. Score distribution by algorithm (20 datasets)")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure3_performance_boxplot.png",
                bbox_inches="tight")
    plt.close(fig)


def figure4_rank_heatmap() -> None:
    """Heatmap of per-dataset ranks (1 = best) across algorithms."""
    ranks = pd.read_csv(RESULTS_DIR / "ranks.csv", index_col="dataset")
    ranks = ranks[ALGORITHMS]

    fig, ax = plt.subplots(figsize=(8, 9))
    im = ax.imshow(ranks.to_numpy(), aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(ALGORITHMS)))
    ax.set_xticklabels(ALGORITHMS, rotation=30, ha="right")
    ax.set_yticks(range(len(ranks.index)))
    ax.set_yticklabels(ranks.index, fontsize=7)
    ax.set_title("Figure 4. Per-dataset algorithm ranks (1 = best)")
    # annotate cells
    for i in range(ranks.shape[0]):
        for j in range(ranks.shape[1]):
            ax.text(j, i, f"{ranks.iat[i, j]:.0f}", ha="center",
                    va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="Rank", shrink=0.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure4_rank_heatmap.png",
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def generate_all() -> None:
    """Build every table and figure, print a manifest."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    t1 = table1_dataset_characteristics()
    t2 = table2_meta_feature_importance()
    figure1_critical_difference()
    figure2_complexity_performance()
    figure3_performance_boxplot()
    figure4_rank_heatmap()

    print(f"Table 1: {len(t1)} datasets")
    print(f"Table 2: {len(t2)} meta-features")
    print(f"\nArtefacts written to {FIGURES_DIR}:")
    for path in sorted(FIGURES_DIR.glob("*")):
        if path.is_file():
            print(f"  {path.name:40s} {path.stat().st_size:>8,} bytes")


if __name__ == "__main__":
    generate_all()
