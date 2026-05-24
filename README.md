# AML/MDS Survival Risk Modeling

This repository contains survival-analysis experiments for an AML/MDS leukemia risk prediction challenge. The goal is to predict patient-level survival risk from clinical, cytogenetic, and molecular data while avoiding leakage across treatment centers and hidden test cohorts.

## Highlights

- Built leakage-safe survival pipelines with `scikit-survival`, XGBoost, LightGBM, and CatBoost.
- Used GroupKFold by clinical center for model selection and kept `CENTER` out of model features.
- Engineered 400+ V6 dense features from labs, cytogenetics, variant annotations, VAF summaries, mutation burden, and gene/pathway indicators.
- Tested OOF rank blending, cytogenetics text rankers, XGBoost survival-Cox models, and CatBoost pairwise survival ranking.
- Best exploratory public submission reached a 0.757 IPCW C-index using an ExtraSurvivalTrees + XGBoost survival-Cox rank blend.

## Main Files

- `leukemia_survival_colab.py` / `.ipynb`: original benchmark-style workflow.
- `leukemia_survival_v7.py` / `.ipynb`: domain-informed AML/MDS feature ablations.
- `leukemia_survival_v8a.py` / `.ipynb`: full-target versus tau=7 target comparison.
- `leukemia_survival_v8cd.py` / `.ipynb`: XGBoost and LightGBM diagnostic experiments.
- `leukemia_survival_v8e.py` / `.ipynb`: OOF rank-blending diagnostics.
- `leukemia_survival_v9.py` / `.ipynb`: focused ExtraSurvivalTrees and XGBoost survival-Cox search.
- `leukemia_survival_exp01_catboost_ipcw_pairrank.py` / `.ipynb`: CatBoost pairwise ranking diagnostics.
- `leukemia_survival_exp02_cyto_text_rankers.py` / `.ipynb`: cytogenetics text-ranker diagnostics.

## Reproducibility Notes

Raw challenge data and generated outputs are not included in this repository. Place the challenge files in a local `QRT_blood` directory or update the paths in the scripts:

- `X_train_9po2I7U.zip`
- `X_test_xzVefmA.zip`
- `target_train.csv`
- `random_submission_FRacdcw_v9kP4pP.csv`

Install dependencies in a local virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Most scripts include quick-check flags so feature shapes, dependencies, and leakage guards can be validated before running full experiments.
