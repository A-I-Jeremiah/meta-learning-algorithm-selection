"""
test_schema_validation.py
=========================

Proof that the generated XML instances conform to experiment.xsd and are
internally consistent. Run with:

    python -m pytest tests/test_schema_validation.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
import xmlschema
from lxml import etree

REPO_ROOT = Path(__file__).resolve().parents[1]
XSD_PATH = REPO_ROOT / "xml" / "schema" / "experiment.xsd"
SAMPLES = [
    REPO_ROOT / "xml" / "samples" / "sample_experiment.xml",
    REPO_ROOT / "xml" / "samples" / "sample_results.xml",
]


@pytest.fixture(scope="module")
def schema() -> xmlschema.XMLSchema:
    assert XSD_PATH.exists(), f"missing schema: {XSD_PATH}"
    return xmlschema.XMLSchema(str(XSD_PATH))


def test_schema_is_wellformed(schema):
    """The XSD itself parses and builds without error."""
    assert schema.root_elements  # at least one global element (Experiment)
    assert "Experiment" in {e.local_name for e in schema.root_elements}


@pytest.mark.parametrize("xml_path", SAMPLES, ids=lambda p: p.name)
def test_sample_validates(schema, xml_path):
    """Each generated sample validates against the schema."""
    assert xml_path.exists(), f"missing sample: {xml_path}"
    schema.validate(str(xml_path))  # raises XMLSchemaValidationError on failure


@pytest.mark.parametrize("xml_path", SAMPLES, ids=lambda p: p.name)
def test_referential_integrity(xml_path):
    """Every Run references a declared Dataset id and Algorithm id."""
    tree = etree.parse(str(xml_path))
    dataset_ids = set(tree.xpath("//Datasets/Dataset/@id"))
    algorithm_ids = set(tree.xpath("//Algorithms/Algorithm/@id"))

    for run in tree.xpath("//Runs/Run"):
        assert run.get("datasetRef") in dataset_ids
        assert run.get("algorithmRef") in algorithm_ids


@pytest.mark.parametrize("xml_path", SAMPLES, ids=lambda p: p.name)
def test_run_count_matches_folds(xml_path):
    """Each dataset has exactly (n_algorithms * cvFolds) runs."""
    tree = etree.parse(str(xml_path))
    n_algorithms = len(tree.xpath("//Algorithms/Algorithm"))
    cv_folds = int(tree.xpath("string(//Experiment/@cvFolds)"))
    expected = n_algorithms * cv_folds

    for ds in tree.xpath("//Datasets/Dataset"):
        ds_id = ds.get("id")
        n_runs = len(tree.xpath(f"//Runs/Run[@datasetRef=$i]", i=ds_id))
        assert n_runs == expected, f"{ds.get('name')}: {n_runs} != {expected}"


@pytest.mark.parametrize("xml_path", SAMPLES, ids=lambda p: p.name)
def test_metric_values_in_range(xml_path):
    """Accuracy values lie in [0, 1]; R^2 values are <= 1."""
    tree = etree.parse(str(xml_path))
    for run in tree.xpath("//Runs/Run"):
        metric = run.findtext("metric")
        value = float(run.findtext("value"))
        if metric == "accuracy":
            assert 0.0 <= value <= 1.0
        elif metric == "r2":
            assert value <= 1.0
