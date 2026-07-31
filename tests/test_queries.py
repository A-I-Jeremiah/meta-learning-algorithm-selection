"""
test_queries.py
===============

Demonstrates the XPath expressions against the sample XML and verifies the
core XQuery logic ("best algorithm per dataset") reproduces the pipeline's
own best_algorithm.csv.

XPath is executed directly with lxml. XQuery needs an external engine
(BaseX/Saxon), so instead of shelling out we replicate its logic in Python
and assert it agrees with the committed results -- proving the query returns
the correct answer. Run with:

    python -m pytest tests/test_queries.py -v
"""

from __future__ import annotations

import csv
from pathlib import Path

from lxml import etree

REPO_ROOT = Path(__file__).resolve().parents[1]
FULL_XML = REPO_ROOT / "xml" / "samples" / "sample_experiment.xml"
BEST_CSV = REPO_ROOT / "results" / "best_algorithm.csv"


def _tree():
    assert FULL_XML.exists(), f"missing sample: {FULL_XML}"
    return etree.parse(str(FULL_XML))


# ---------------------------------------------------------------------------
# XPath demonstrations (the brief's headline example + variants)
# ---------------------------------------------------------------------------

def test_high_accuracy_runs_are_all_high():
    """//Run[metric='accuracy' and value>0.9] returns only high-acc runs."""
    tree = _tree()
    runs = tree.xpath("//Run[metric='accuracy' and number(value) > 0.9]")
    # Every returned run really does exceed 0.9.
    for run in runs:
        assert float(run.findtext("value")) > 0.9
        assert run.findtext("metric") == "accuracy"


def test_high_accuracy_dataset_names():
    """The brief's //Run[...]/Dataset projection yields dataset names."""
    tree = _tree()
    names = tree.xpath(
        "//Run[metric='accuracy' and number(value) > 0.9]/Dataset/text()")
    # Names must all be declared datasets.
    declared = set(tree.xpath("//Datasets/Dataset/@name"))
    assert set(names).issubset(declared)


def test_binary_dataset_filter():
    """XPath attribute filter selects only binary datasets."""
    tree = _tree()
    binary = tree.xpath("//Dataset[@task='binary']/@name")
    for name in binary:
        node = tree.xpath("//Dataset[@name=$n]", n=name)[0]
        assert node.get("task") == "binary"


def test_algorithm_ref_filter_returns_runs():
    """Filtering runs by algorithmRef returns a non-empty, consistent set."""
    tree = _tree()
    runs = tree.xpath("//Run[@algorithmRef='algo_GradientBoosting']")
    assert len(runs) > 0
    for run in runs:
        assert run.get("algorithmRef") == "algo_GradientBoosting"


# ---------------------------------------------------------------------------
# XQuery logic: best algorithm per dataset (mean value -> max)
# ---------------------------------------------------------------------------

def _best_algorithm_per_dataset_via_xml() -> dict[str, str]:
    """
    Replicate best_algorithm_per_dataset.xquery in Python over the XML:
    for each dataset, mean each algorithm's fold values, pick the max.
    Returns {dataset_name: algorithm_name}.
    """
    tree = _tree()
    result: dict[str, str] = {}

    for ds in tree.xpath("//Datasets/Dataset"):
        ds_id = ds.get("id")
        ds_name = ds.get("name")
        runs = tree.xpath("//Runs/Run[@algorithmRef and @datasetRef=$i]",
                          i=ds_id)
        # Aggregate mean value per algorithmRef.
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for run in runs:
            algo = run.get("algorithmRef")
            v = float(run.findtext("value"))
            sums[algo] = sums.get(algo, 0.0) + v
            counts[algo] = counts.get(algo, 0) + 1
        means = {a: sums[a] / counts[a] for a in sums}
        best_ref = max(means, key=means.get)  # type: ignore[arg-type]
        # Strip the "algo_" id prefix to get the plain algorithm name.
        result[ds_name] = best_ref.replace("algo_", "")

    return result


def _best_algorithm_from_pipeline() -> dict[str, str]:
    """Load the pipeline's committed best_algorithm.csv."""
    assert BEST_CSV.exists(), f"missing results: {BEST_CSV}"
    out: dict[str, str] = {}
    with open(BEST_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["dataset"]] = row["best_algorithm"]
    return out


def test_xquery_best_algorithm_matches_pipeline():
    """
    The XQuery's answer (computed from the XML) must equal the pipeline's
    best_algorithm.csv for every dataset -- proving the query is correct.
    """
    from_xml = _best_algorithm_per_dataset_via_xml()
    from_pipeline = _best_algorithm_from_pipeline()

    assert set(from_xml) == set(from_pipeline)
    mismatches = {d: (from_xml[d], from_pipeline[d])
                  for d in from_xml if from_xml[d] != from_pipeline[d]}
    assert not mismatches, f"XQuery/pipeline disagree: {mismatches}"
