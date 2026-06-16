# TriageGeist Holistic Suite: Leak-Free Physiological Engine & NLP Override

**Subtitle:** Dual-track triage AI guaranteeing clinical safety via conformal prediction while maximizing leaderboard QWK through NLP reverse-engineering.

## Project Overview
This repository contains the complete, reproducible source code for the TriageGeist competition submission. Emergency triage is a high-stakes environment where the primary danger is **undertriage**. Current AI models proposed for triage often achieve perfect statistical scores by memorizing dataset artifacts. 

Our dual-track decision support system provides an interpretable, mathematically safe secondary triage verification that penalizes undertriage and abstains when uncertain, ensuring physicians aren't misled by overconfident algorithms.

## Setup Instructions

### 1. Prerequisites
* Python 3.9+
* Required packages (install via `pip`):
  ```bash
  pip install -r requirements.txt
  ```

### 2. Dataset Setup
Ensure you have downloaded the Triagegeist dataset from Kaggle.
* Place `train.csv`, `test.csv`, `patient_history.csv`, and `chief_complaints.csv` inside a local `data/` directory.

### 3. Reproducing the Submission
The entire pipeline is executed statelessly via a single python script. To generate the final Kaggle notebook and the submission predictions, run:

```bash
python create_notebook.py
```

This will:
1. Load all data strictly observing our "Clean-Room" protocol (dropping target-leaking variables).
2. Extract *Missing Not At Random* (MNAR) boolean indicators for missing vitals.
3. Train the **Physiological Mega-Ensemble** (LightGBM, XGBoost, CatBoost) via strict 5-fold CV.
4. Train the **NLP Leaderboard Maximizer** (TF-IDF + Ridge) on chief complaints.
5. Dynamically calculate the algorithmic fairness probability shift to cap elderly undertriage `<5%`.
6. Output `submissions/final_submission.ipynb`.

### 4. Evaluating Safety & Cost
To evaluate the Asymmetric Clinical Cost Matrix and the Conformal Selective Risk Control (CARES) bounds, you can run the individual experiment scripts located in the `experiments/` directory.

```bash
PYTHONPATH=. python experiments/exp0036_clinical_cost_evaluation.py
PYTHONPATH=. python experiments/exp0010_cares_selective_risk.py
```

## Data Citation
Olaf Yunus Laitinen Imanov (2026). Triagegeist. https://kaggle.com/competitions/triagegeist, 2026. Kaggle. We confirm that our use of the Triagegeist dataset complies with its terms of access. No external datasets were utilized.
