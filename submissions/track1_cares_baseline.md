# Triagegeist: The Clean-Room Baseline & CARES Selective Risk Control

## 1. Clinical Problem Statement

In the high-stakes environment of emergency department triage, machine learning models are often evaluated on their ability to perfectly predict Emergency Severity Index (ESI) labels. However, perfectly predicting a dataset often involves exploiting non-causal artifacts, such as outcome leakage and deterministic synthetic text generation. 

A model achieving a 0.99 Quadratic Weighted Kappa (QWK) on a synthetic dataset by memorizing "hacked" text strings is clinically dangerous. If deployed, it would fail catastrophically on real-world patients whose presentations do not adhere to the synthetic generator's exact templates. 

**Our Objective:** Establish the true clinical ceiling of physiological triage prediction (the "Clean-Room Baseline") and implement **Conformal Selective Risk Control** to guarantee patient safety. By forcing the model to abstain when calibrated reliability is low, we ensure that the AI acts as a safe decision-support tool, handing off uncertain cases back to human emergency physicians.

---

## 2. Methodology: The Clean-Room Protocol

To build a model that can genuinely generalize to real-world ED data (e.g., NHAMCS or MIMIC-IV-ED), we established a strict "Clean-Room" protocol:

1. **Exclusion of Outcome Leakage**: We permanently dropped `disposition` and `ed_los_hours`. Triage occurs at the front door; variables that are only known hours after the patient's disposition are explicitly illegal for an acuity prediction model.
2. **Exclusion of Deterministic Text**: Through forensic auditing, we discovered that `chief_complaint_raw` acts as a deterministic lookup table for the synthetic target in 99.4% of cases. We dropped all raw text features to prevent the model from learning synthetic generator rules.
3. **Pure Physiological Modeling**: Our feature space is restricted purely to demographics, vital signs, structured clinical indices (e.g., NEWS2, Shock Index, GCS), and prior medical history. We trained a LightGBM architecture to predict acuity based strictly on these clinically validated inputs.

---

## 3. Integrating CARES (Calibrated Aspect Reliable Explanations and Safety)

Intrinsic model confidence (Max Softmax Probability) is notoriously overconfident and poorly calibrated, especially in complex clinical settings. A model might be 90% "confident" but entirely incorrect because it has never seen that specific presentation before.

Adapting the **CARES framework** (first developed for GI Endoscopy VQA), we implemented an output-conditioned reliability layer:

### A. Aspect-Answer Reliability Tables
Instead of trusting the softmax output, we compute historical correctness based on the combination of the patient's `chief_complaint_system` (the clinical aspect) and the `predicted_acuity` (the answer). 

### B. Empirical Bayes Shrinkage
To prevent the model from becoming overly confident on rare clinical presentations (e.g., a rare pediatric cardiac presentation), we apply Empirical Bayes shrinkage ($\alpha=5$). This pulls the reliability estimate of rare events toward the global mean, enforcing cautiousness.

### C. Conformal Selective Risk Control
Using exact Clopper-Pearson bounds ($\delta = 0.1$), we calculate the statistical upper bound of the model's error rate. We then set a strict threshold to guarantee that our selected subset of predictions maintains an error rate of $\le 5\%$. If a patient's calibrated reliability falls below this threshold, the model **abstains** and requests a human physician.

---

## 4. Empirical Results and Findings

### Baseline Performance
When adhering to the Clean-Room protocol, the true physiological ceiling of the dataset emerges. 
- **Clean Baseline QWK**: `0.930` 
- While significantly lower than the 0.99 leaderboard scores, this represents the *true* clinical signal of the patient's vitals and history, stripped of artificial leakage.

### Correctness Prediction (CARES vs. Intrinsic)
Our aspect-answer reliability score underperforms compared to the model's own softmax confidence when predicting whether a triage decision is actually correct on this synthetic dataset:
- **AUROC (Intrinsic Softmax Confidence)**: `0.8533`
- **AUROC (CARES Calibrated Reliability)**: `0.6758`

*(Insight: Unlike in VQA tasks where aspect-answer pairs carry strong prior correctness signals, our analysis shows that in this synthetic triage dataset, grouping by `chief_complaint_system` degrades correctness prediction compared to intrinsic confidence. This highlights a failure mode of aspect-based reliability when the underlying data generator lacks causal links between systemic categories and prediction difficulty.)*

### Conformal Safety Bounds
By implementing selective risk control targeted at a 5% maximum error rate:
- **Verified Error Rate**: `4.76`%
- **Coverage (Patients Handled Autonomously)**: `27.62`%
- **Abstention Rate (Flagged for Physician)**: `72.38`%

---

## 5. Novelty and Clinical Impact

By prioritizing **statistical safety over leaderboard ranking**, this notebook presents a deployable blueprint for emergency triage AI. 

We demonstrate that a 0.88 QWK model equipped with strict abstention bounds is infinitely more valuable to an Emergency Department than an unchecked 0.99 QWK model that hallucinates on real-world data. The CARES conformal framework acts as a critical safety net, ensuring the AI only operates within its verified competency zone.
