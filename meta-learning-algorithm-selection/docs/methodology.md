# Methodology

This document describes the full empirical pipeline for the study
*Meta-Learning for Algorithm Selection from Dataset Meta-Features*. It is the
canonical reference for how every numeric result is produced, and it maps
one-to-one onto the modules in `src/`. All artefacts are regenerable by
running the modules in order (or `notebooks/main.ipynb` top to bottom).

## 1. Overview

The study asks: **which algorithms generalise best across diverse benchmark
datasets, and which dataset meta-features predict that success?** We answer it
empirically by (1) curating 20 benchmark datasets, (2) describing each with a
vector of meta-features, (3) evaluating five learning algorithms on every
dataset under identical cross-validation, (4) ranking the algorithms per
dataset, (5) training a meta-learner to predict the winning algorithm from
meta-features, and (6) testing all differences for statistical significance.

The pipeline is deterministic: every stochastic component is seeded
(`random_state = 42`), so results are reproducible across runs and machines.

| Stage | Module | Output |
|-------|--------|--------|
| Data acquisition | `src/data_loader.py` | `data/processed/*.csv`, `dataset_index.csv` |
| Meta-feature extraction | `src/meta_features.py` | `results/meta_features.csv` |
| Model training (5-fold CV) | `src/models.py` | `results/cv_results_long.csv`, `results/performance_matrix.csv` |
| Ranking & normalisation | `src/ranking.py` | `results/ranks.csv`, `results/best_algorithm.csv`, `results/average_ranks.csv` |
| Meta-learner | `src/meta_learner.py` | `results/meta_learner_*.csv/json`, `results/meta_feature_importance.csv` |
| Statistical tests | `src/stats.py` | `results/statistical_tests.json` and pairwise CSVs |
| XML modelling | `src/xml_export.py` | `xml/samples/*.xml` (valid against `xml/schema/experiment.xsd`) |
| Figures & tables | `src/visualise.py` | `figures/*` |

## 2. Data acquisition and harmonisation

**Source.** All datasets come from the Penn Machine Learning Benchmark (PMLB),
a curated collection of benchmark problems distributed as a Python package.
PMLB returns each dataset as a tidy pandas DataFrame in which the final column,
`target`, holds the label or response. Using PMLB removes ambiguity in dataset
provenance: every dataset is versioned and fetched programmatically, so the
selection is fully reproducible.

**Selection protocol (`src/data_loader.py`).** We select 20 datasets against a
fixed quota that balances the three task families the study compares:

- 8 binary classification
- 7 multiclass classification
- 5 regression

Candidate datasets are examined one at a time from PMLB's own name lists. A
candidate is accepted only if it passes a size filter and its task family still
has an unfilled quota. The size filter keeps the experiment tractable on a
single machine while retaining diversity:

- 100 to 3000 instances (lower bound keeps 5-fold CV stable),
- at most 60 features.

Task type is decided empirically from the data, not from the dataset name: a
classification dataset is *binary* if its target has exactly two distinct
values and *multiclass* otherwise. The resulting 20 datasets span 100 to 1728
instances and 3 to 32 features.

**Harmonisation.** PMLB data arrives numerically encoded and free of missing
values, so harmonisation is light: each dataset is written to
`data/processed/<name>.csv` with a single, consistently named `target` column.
A machine-readable index (`data/processed/dataset_index.csv`) records, for each
dataset, its task type, instance count, feature count, class count, target
column, and source. Every downstream stage iterates over this index, so the
dataset roster is defined in exactly one place.

## 3. Meta-feature extraction

For each dataset `src/meta_features.py` computes a fixed 13-dimensional
meta-feature vector. Meta-features are deliberately cheap to compute and are
defined for both classification and regression (class-specific features take a
neutral value for regression). They fall into four groups:

**Size and dimensionality**
- `n_instances`, `n_features` — raw counts.
- `log_n_instances`, `log_n_features` — log10 transforms that tame the wide
  dynamic range across datasets.
- `feature_to_instance_ratio` — dimensionality pressure (features / instances).

**Class structure** (classification only; neutral for regression)
- `n_classes` — number of distinct labels.
- `class_imbalance_ratio` — size of the largest class divided by the smallest
  (1.0 means perfectly balanced).
- `normalised_class_entropy` — label entropy divided by log(n_classes); 1.0
  means uniform class proportions.

**Statistical descriptors** (averaged over features so the vector length is
fixed regardless of feature count)
- `mean_feature_std` — mean per-feature standard deviation.
- `mean_abs_coef_variation` — mean absolute coefficient of variation.
- `mean_skewness`, `mean_kurtosis` — mean per-feature shape statistics.

**Feature redundancy**
- `mean_abs_feature_correlation` — mean absolute off-diagonal correlation
  between features (0 when there is a single feature).

The full matrix is written to `results/meta_features.csv`, one row per dataset.

## 4. Algorithms and cross-validation

**Algorithm families (`src/models.py`).** Five families are evaluated, each
with a classifier and a regressor variant so the same code path serves every
task:

| Name | Classifier / Regressor | Key hyperparameters |
|------|------------------------|---------------------|
| RandomForest | `RandomForest{Classifier,Regressor}` | 200 trees |
| GradientBoosting | `GradientBoosting{Classifier,Regressor}` | scikit-learn defaults |
| SVM | `SVC` / `SVR` | RBF kernel |
| kNN | `KNeighbors{Classifier,Regressor}` | k = 5 |
| NeuralNetwork | `MLP{Classifier,Regressor}` | hidden layers (64, 32), max_iter 1000 |

Hyperparameters are modest, fixed defaults chosen to behave reasonably across
small datasets. We deliberately do **not** tune per dataset: tuning would
confound the algorithm comparison with tuning effort and search budget, and the
research question concerns out-of-the-box generalisation.

**Preprocessing.** Scale-sensitive models (SVM, kNN, neural network) are wrapped
in a `StandardScaler` pipeline so standardisation is fit inside each CV fold,
never on held-out data. Tree ensembles are scale-invariant and are used
directly.

**Cross-validation.** Every algorithm is evaluated with 5-fold cross-validation
on every dataset. Classification uses `StratifiedKFold` (preserving class
proportions per fold); regression uses `KFold`. Both use `shuffle=True` and the
fixed seed. The primary score is accuracy for classification and the
coefficient of determination (R²) for regression — higher is better in both.

Results are stored in two forms: a long table with one row per
(dataset, algorithm, fold) in `results/cv_results_long.csv`, and a wide
performance matrix (datasets × algorithms, mean CV score) in
`results/performance_matrix.csv`.

## 5. Ranking and normalisation

Because accuracy and R² are not directly comparable, `src/ranking.py` converts
the performance matrix into scale-free views:

- **Per-dataset ranks** (1 = best). Higher scores rank better; ties receive the
  average rank. These ranks are comparable across classification and regression
  and are the basis for the Friedman and Nemenyi tests.
- **Normalised scores** in [0, 1] within each dataset (min-max across the five
  algorithms), where 1.0 marks the best algorithm on that dataset and 0.0 the
  worst.
- **Best algorithm per dataset** — the top-scoring algorithm, which becomes the
  target label for the meta-learner.
- **Average ranks** — each algorithm's mean rank across all datasets.

## 6. Meta-learner

The meta-learner (`src/meta_learner.py`) is the empirical centrepiece: a Random
Forest classifier that predicts the best algorithm for a dataset from its
meta-features alone. Inputs are the 13 meta-features plus a numeric task code;
the target is the best-algorithm label from stage 5.

**Evaluation.** With only 20 datasets, we use leave-one-out cross-validation
(LOO-CV): train on 19 datasets, predict the held-out one, repeat 20 times. We
report LOO accuracy against a **majority-class baseline** that always predicts
the most frequent winning algorithm, so the meta-learner's lift (or lack of it)
is interpretable rather than reported in a vacuum.

**Feature importance.** A Random Forest fit on all 20 datasets yields
meta-feature importances (Gini importance), written to
`results/meta_feature_importance.csv` and reported as Table 2.

## 7. Statistical tests

`src/stats.py` implements the standard protocol for comparing multiple
algorithms over multiple datasets (Demšar, 2006):

- **Friedman test** — an omnibus, non-parametric test of whether the five
  algorithms have equal mean ranks. If significant, we proceed to a post-hoc
  test.
- **Nemenyi post-hoc** — an all-pairs comparison. Two algorithms differ
  significantly if their average ranks differ by more than the Critical
  Difference (CD), where `CD = q_α · sqrt(k(k+1) / 6N)` for k algorithms and N
  datasets, and `q_α` is the Studentized-range critical value divided by √2.
  The CD drives the Critical Difference diagram (Figure 1).
- **Wilcoxon signed-rank** — a pairwise comparison of the top-ranked algorithm
  against each competitor, using per-dataset scores.

All test outputs are serialised to `results/statistical_tests.json` plus
pairwise CSVs.

## 8. XML modelling and queries

The experiment is modelled in XML so results can be validated and queried
declaratively, independent of the Python objects that produced them.

- **Schema (`xml/schema/experiment.xsd`).** Models the *experiment* structure —
  `Experiment → Datasets/Algorithms/Runs`. A `Dataset` carries its task and
  meta-features; an `Algorithm` carries its hyperparameters; a `Run` records one
  CV fold (dataset reference, algorithm reference, fold index, metric, value).
  Referential integrity between Runs and their Datasets/Algorithms is enforced
  with `xs:ID`/`xs:IDREF` and documented with `key`/`keyref`.
- **Instances (`src/xml_export.py`).** Valid XML is generated directly from the
  results files, so the XML always reflects the real experiment. A full
  instance (20 datasets, 500 runs) and a compact 3-dataset subset are produced.
- **Queries (`xml/queries/`).** XPath expressions extract runs by performance
  (e.g. all runs with accuracy > 0.9) and datasets by property; XQuery scripts
  return the best algorithm per dataset and run integrity checks. Their
  correctness is verified in `tests/` by replicating the query logic and
  asserting it matches the pipeline's committed results.

## 9. Reproducibility

- All randomness is seeded (`random_state = 42`).
- Dependencies are pinned in `requirements.txt`.
- The dataset roster lives only in `dataset_index.csv`; changing it changes the
  whole study consistently.
- `notebooks/main.ipynb` runs the entire pipeline end to end and was executed
  headlessly without error; `tests/` (14 tests) validate the schema, the
  generated XML, and the query logic.
- Heavy stages cache their outputs; deleting the relevant file in `results/`
  forces a clean recomputation.

## Reference

Demšar, J. (2006). Statistical comparisons of classifiers over multiple data
sets. *Journal of Machine Learning Research, 7*, 1–30.

