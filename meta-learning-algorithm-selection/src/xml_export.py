"""
xml_export.py
=============

Generate valid XML instances of the experiment from the real results.

Reads:
  * results/meta_features.csv     -> Dataset elements + MetaFeatures
  * results/cv_results_long.csv   -> Run elements (one per fold)

Writes (both validate against xml/schema/experiment.xsd):
  * xml/samples/sample_experiment.xml -- full experiment: all datasets,
    algorithms and every CV run.
  * xml/samples/sample_results.xml    -- a compact subset (first 3 datasets)
    handy for demonstrating XPath/XQuery quickly.

Algorithm hyperparameters are pulled from the same estimator definitions used
for training (src.models.build_estimators), so the XML documents the actual
configuration rather than a hand-written copy.

Run directly:

    python -m src.xml_export
"""

from __future__ import annotations

from typing import cast
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring

import pandas as pd

from .data_loader import REPO_ROOT
from .models import ALGORITHMS, build_estimators, N_FOLDS

RESULTS_DIR = REPO_ROOT / "results"
SAMPLES_DIR = REPO_ROOT / "xml" / "samples"

# Meta-feature columns, in schema order (must match MetaFeaturesType).
META_FEATURE_FIELDS = [
    "n_instances", "n_features", "log_n_instances", "log_n_features",
    "feature_to_instance_ratio", "n_classes", "class_imbalance_ratio",
    "normalised_class_entropy", "mean_feature_std", "mean_abs_coef_variation",
    "mean_skewness", "mean_kurtosis", "mean_abs_feature_correlation",
]

# Human-readable family label per algorithm (for Algorithm/@family).
ALGORITHM_FAMILY = {
    "RandomForest": "ensemble-bagging",
    "GradientBoosting": "ensemble-boosting",
    "SVM": "kernel-machine",
    "kNN": "instance-based",
    "NeuralNetwork": "neural-network",
}


def _dataset_id(name: str) -> str:
    """Make an XML-safe xs:ID from a dataset name (must not start w/ digit)."""
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)
    return f"ds_{safe}"


def _algorithm_id(name: str) -> str:
    return f"algo_{name}"


def _hyperparameters_for(algo: str) -> dict[str, str]:
    """
    Extract the actual hyperparameters used, from the estimator definitions.
    Classification variants are used (regression uses the same settings).
    """
    est = build_estimators("binary")[algo]
    # Unwrap pipelines (scaled models) to reach the real estimator.
    model = est.named_steps["model"] if hasattr(est, "named_steps") else est
    params = model.get_params(deep=False)
    # Keep only JSON-simple, informative params (drop None / callables).
    keep = {}
    for k, v in params.items():
        if v is None or callable(v):
            continue
        if k in ("random_state", "n_jobs", "verbose"):
            continue
        keep[k] = str(v)
    return keep


def _add_dataset_element(parent: Element, row: pd.Series) -> None:
    ds = SubElement(parent, "Dataset", {
        "id": _dataset_id(str(row["dataset"])),
        "name": str(row["dataset"]),
        "task": str(row["task"]),
        "source": "PMLB",
    })
    mf = SubElement(ds, "MetaFeatures")
    for field in META_FEATURE_FIELDS:
        el = SubElement(mf, field)
        el.text = repr(float(row[field]))


def _add_algorithm_element(parent: Element, algo: str) -> None:
    a = SubElement(parent, "Algorithm", {
        "id": _algorithm_id(algo),
        "name": algo,
        "family": ALGORITHM_FAMILY[algo],
    })
    hp = SubElement(a, "Hyperparameters")
    for name, value in _hyperparameters_for(algo).items():
        el = SubElement(hp, "Hyperparameter", {"name": name})
        el.text = value


def _add_run_element(parent: Element, r: pd.Series) -> None:
    run = SubElement(parent, "Run", {
        "datasetRef": _dataset_id(str(r["dataset"])),
        "algorithmRef": _algorithm_id(str(r["algorithm"])),
    })
    SubElement(run, "Dataset").text = str(r["dataset"])
    SubElement(run, "fold").text = str(int(r["fold"]))
    SubElement(run, "metric").text = str(r["metric"])
    SubElement(run, "value").text = repr(float(r["score"]))


def build_experiment_element(meta: pd.DataFrame,
                             runs: pd.DataFrame,
                             title: str) -> Element:
    """Assemble a complete <Experiment> element tree."""
    root = Element("Experiment", {
        "title": title,
        "cvFolds": str(N_FOLDS),
    })

    datasets_el = SubElement(root, "Datasets")
    for _, row in meta.iterrows():
        _add_dataset_element(datasets_el, row)

    algorithms_el = SubElement(root, "Algorithms")
    for algo in ALGORITHMS:
        _add_algorithm_element(algorithms_el, algo)

    runs_el = SubElement(root, "Runs")
    for _, r in runs.iterrows():
        _add_run_element(runs_el, r)

    return root


def _write_pretty(root: Element, path) -> None:
    rough = tostring(root, encoding="utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ",
                                                     encoding="UTF-8")
    path.write_bytes(pretty)


def generate() -> None:
    """Generate both sample XML files from results/."""
    meta = pd.read_csv(RESULTS_DIR / "meta_features.csv")
    runs = pd.read_csv(RESULTS_DIR / "cv_results_long.csv")

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    # Full experiment.
    full = build_experiment_element(
        meta, runs, title="Meta-Learning for Algorithm Selection (full)")
    full_path = SAMPLES_DIR / "sample_experiment.xml"
    _write_pretty(full, full_path)
    print(f"Full experiment  -> {full_path}  "
          f"({len(meta)} datasets, {len(runs)} runs)")

    # Compact subset: first 3 datasets and their runs.
    subset_names = meta["dataset"].head(3).tolist()
    meta_sub = cast(pd.DataFrame, meta[meta["dataset"].isin(subset_names)])
    runs_sub = cast(pd.DataFrame, runs[runs["dataset"].isin(subset_names)])
    subset = build_experiment_element(
        meta_sub, runs_sub,
        title="Meta-Learning for Algorithm Selection (sample subset)")
    subset_path = SAMPLES_DIR / "sample_results.xml"
    _write_pretty(subset, subset_path)
    print(f"Sample subset    -> {subset_path}  "
          f"({len(meta_sub)} datasets, {len(runs_sub)} runs)")


if __name__ == "__main__":
    generate()
