# Meta-Learning for Algorithm Selection from Dataset Meta-Features

**Group empirical study** · Fully data-driven · IMRAD manuscript · Reproducible pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Academic-green)](#)
[![Status](https://img.shields.io/badge/Status-Active%20Development-orange)](#)

---

## 1. Project Overview

**Research Question**  
Which algorithms generalise best across diverse benchmark datasets, and which meta-features best predict that success?

**Core Goal**  
Produce a fully data-driven empirical study. Every claim is backed by tables, figures, code outputs, or validated queries. Purely theoretical submissions are not accepted.

We:

1. Curate ~20 diverse datasets from the Penn Machine Learning Benchmark (PMLB).
2. Extract a standard set of dataset meta-features.
3. Train five algorithms (Random Forest, Gradient Boosting, SVM, k-NN, Neural Network) with 5-fold stratified cross-validation.
4. Rank algorithms per dataset using normalised performance scores.
5. Train a meta-learner that predicts the best algorithm from meta-features alone.
6. Perform rigorous statistical tests (Friedman + Nemenyi, Wilcoxon signed-rank).
7. Model the entire experiment in XSD and provide XPath/XQuery extraction scripts.
8. Deliver an IMRAD manuscript (4 000–6 000 words, APA 7) and a 10-minute presentation.

---

## 2. Team & Responsibilities

| Member | Role | Primary Ownership |
|--------|------|-------------------|
| **Akindipe Ireoluwawolemi Jeremiah** | Leader / ML Engineer | Full ML pipeline, meta-learner, statistical tests, core notebooks, final reproducibility |
| **AJIBOLA** | Data Curator & Metadata Specialist | Dataset acquisition, cleaning, harmonisation, data dictionary & ethics documentation |
| **AKINMOJU** | XML/XSD Architect | Complete XSD schema + sample valid XML instances + validation proof |
| **AKINWOLA** | XPath/XQuery Developer | XPath & XQuery scripts for extraction and validation |
| **AKINYELE** | Visualisation & Statistical Reporting | Tables, Critical Difference diagram, feature-importance table, scatter plots, high-res figures |
| **AKINYEMI** | Manuscript Lead | Full empirical manuscript (IMRAD, APA 7) |
| **ASAMU** | Presentation & Peer-Review Coordinator | 10-minute MP4 + peer-review process + final packaging |

Detailed task allocation and phased roadmap are maintained in the project documents.

---

## 3. Repository Structure

```
meta-learning-algorithm-selection/
├── data/
│   ├── raw/                  # Original downloaded datasets
│   ├── processed/            # Harmonised CSVs ready for the pipeline
│   └── metadata/             # Data dictionary, ethics notes, source citations
├── docs/                     # Methodology notes, contribution guide, meta-feature definitions
├── figures/                  # High-resolution figures (generated)
├── manuscript/               # IMRAD Word manuscript and supporting material
├── notebooks/                # Jupyter notebooks (main analysis lives here)
├── presentation/             # Scripts and final MP4
├── results/                  # Performance matrices, rankings, statistical outputs (CSV/JSON)
├── src/                      # Reusable Python modules
│   ├── __init__.py
│   ├── data_loader.py        # Load & prepare datasets
│   ├── meta_features.py      # Extract dataset meta-features
│   ├── models.py             # Algorithm definitions & training helpers
│   ├── ranking.py            # Normalised ranking logic
│   └── stats.py              # Friedman, Nemenyi, Wilcoxon, feature importance
├── tests/                    # Unit / integration tests
├── xml/                      # XSD schema, sample XML instances, XPath/XQuery scripts
├── requirements.txt
└── README.md                 # You are here
```

---

## 4. Quick Start (Onboarding)

### 4.1 Prerequisites

- Python **3.10 or higher**
- Git
- Recommended: a virtual environment tool (`venv`, `conda`, or `poetry`)

### 4.2 Clone the repository

```bash
git clone https://github.com/A-I-Jeremiah/meta-learning-algorithm-selection.git
cd meta-learning-algorithm-selection
```

### 4.3 Create and activate a virtual environment

```bash
# Using venv
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 4.4 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Current pinned packages include:

- `numpy`, `pandas`, `scikit-learn`
- `pmlb` (Penn Machine Learning Benchmark)
- `scipy`, `seaborn`, `matplotlib`
- `lxml`, `xmlschema` (XML validation)

### 4.5 Verify installation

```bash
python -c "import sklearn, pmlb, pandas, numpy; print('Environment OK')"
```

---

## 5. How to Run the Pipeline

> **Note:** The full pipeline requires the 20 processed datasets in `data/processed/`.  
> Until the Data Curator delivers them, you can still explore the modules and notebook skeletons.

### 5.1 Main analysis notebook

```bash
jupyter notebook notebooks/01_meta_learning_pipeline.ipynb
```

(or open the notebook in VS Code / JupyterLab)

The notebook is structured as:

1. Setup & imports  
2. Load dataset list  
3. Extract meta-features  
4. Train five algorithms (5-fold stratified CV)  
5. Build performance matrix  
6. Rank algorithms per dataset  
7. Train meta-learner  
8. Statistical tests  
9. Export results for visualisation and XML teams  

### 5.2 Reproducibility check (Leader only)

From a clean environment:

```bash
pip install -r requirements.txt
python -m pytest tests/          # if tests are present
jupyter nbconvert --to notebook --execute notebooks/01_meta_learning_pipeline.ipynb
```

All numeric results must be regenerable from the committed code and data.

---

## 6. Data Guidelines

- **Primary source:** [Penn Machine Learning Benchmark (PMLB)](https://github.com/EpistasisLab/pmlb)
- **Target:** approximately 20 datasets covering binary classification, multiclass classification, and regression.
- Processed files live in `data/processed/` with consistent naming and a clear target column.
- A full data dictionary and ethics notes live in `data/metadata/`.

Do **not** commit large raw binary files. Use Git LFS or keep large artefacts outside the repository if necessary.

---

## 7. Contribution Workflow

We work with **feature branches + Pull Requests**. Direct pushes to `main` are discouraged.

1. Create a branch named after your task, e.g.  
   `git checkout -b feature/meta-features`  
   `git checkout -b data/20-datasets`
2. Make your changes and commit with clear messages.
3. Push the branch and open a Pull Request against `main`.
4. The Leader (Akindipe) reviews for correctness, style, and reproducibility.
5. Once approved, the PR is merged.

### Coding standards

- Follow PEP 8.
- Keep notebooks clean and well-commented — the meta-learning notebook is the heart of the empirical contribution.
- Prefer functions in `src/` over long notebook cells.
- Document every transformation so the Methods section remains reproducible.

---

## 8. Deliverables Checklist

| Deliverable | Location / Owner | Status |
|-------------|------------------|--------|
| Documented raw + processed datasets + data dictionary | `data/` · AJIBOLA | Pending |
| Reproducible analysis code + notebooks | `src/` + `notebooks/` · Akindipe | In progress |
| XML Schema (XSD) + sample instances | `xml/` · AKINMOJU | Pending |
| XPath / XQuery scripts | `xml/` · AKINWOLA | Pending |
| Tables & high-resolution figures | `figures/` + `results/` · AKINYELE | Pending |
| Empirical manuscript (4–6 k words, IMRAD, APA 7) | `manuscript/` · AKINYEMI | Pending |
| 10-minute presentation (MP4) | `presentation/` · ASAMU | Pending |
| Clean, runnable GitHub repository | This repo · Akindipe | Active |

---

## 9. Assessment Mapping

| Component | Weight | Evidence required |
|-----------|--------|-------------------|
| Dataset Acquisition & Documentation | 15 % | Raw + processed data, metadata, ethics notes |
| Data Analysis & Visualisation | 20 % | Notebooks, statistical tests, figures |
| XML/XSD Modelling | 15 % | Valid XSD, sample XML, working queries |
| Empirical Manuscript (IMRAD) | 30 % | 4 000–6 000 word paper, APA 7 |
| Reproducibility & Code Quality | 10 % | Runnable repo, clear README, requirements.txt |
| Presentation & Peer Review | 10 % | 10-min MP4 + peer-review forms |

All grading is **data-driven**. Every claim needs a table, figure, code output, or validated query behind it.

---

## 10. Communication & Tracking

- Use **GitHub Issues** for task tracking and blockers.
- Use **Pull Requests** for code and document review.
- Major decisions and phase gates are recorded in the project board.
- Raise blockers immediately so the critical path (Datasets → ML Pipeline → Parallel Tracks → Manuscript → Presentation) is protected.

---

## 11. Licence & Academic Use

This repository is created for academic assessment purposes.  
All third-party datasets remain under their original licences (primarily PMLB / UCI).  
Cite sources appropriately in the manuscript and data dictionary.

---

## 12. Contact

**Group Leader / ML Engineer**  
Akindipe Ireoluwawolemi Jeremiah  
Repository: https://github.com/A-I-Jeremiah/meta-learning-algorithm-selection

For onboarding questions, open an Issue or contact the Leader directly.

---

*Last updated: July 2026 · Keep this README the single source of truth for setup and structure.*
