# Triagegeist: Beyond ESI - Forecasting Deterioration & Admission in the Waiting Room

## 1. Clinical Problem Statement

The vast majority of models in this competition attempt to predict `triage_acuity` (ESI level). However, ESI is inherently a static label assigned at the door. It is highly subjective, and predicting it does not necessarily prevent adverse outcomes for patients stuck in the waiting room.

**Our Objective**: To fulfill the mandate of Track 3 (Decision Support for Deterioration), we must predict something actionable. We pivot the problem entirely: instead of predicting the static ESI label, we predict the *patient's ultimate disposition* (`admitted` or `deceased`). 

By identifying patients who look stable at triage but will eventually deteriorate and require admission or die, we can alert triage nurses to upgrade their ESI level or fast-track them to a bed.

---

## 2. Methodology: Predicting Deterioration

### Target Variable Reframing
We convert the multi-class acuity problem into a binary classification task:
- **Target = 1 (Deterioration/High Risk)**: Disposition is `admitted` or `deceased`.
- **Target = 0 (Lower Risk)**: Disposition is `discharged` or `ama` (Against Medical Advice).

### The Clean-Room Approach
We strictly exclude `chief_complaint_raw`, `triage_acuity`, and `ed_los_hours` from our feature set. We use only the structured physiological data collected at triage (vitals, prior medical history, demographics) to predict this outcome. We train a LightGBM binary classifier evaluated using AUROC.

---

## 3. Empirical Results and Findings

In a 5-Fold Cross Validation setup, our deterioration prediction model achieved:
- **AUROC**: `0.7927`
- **Logloss**: `0.496`

### Clinical Insight
An AUROC of ~0.79 using only triage-door vitals to predict ultimate hospital admission is highly clinically relevant. It suggests that hidden physiological signatures in triage data can identify patients who might be erroneously assigned an ESI 3 (Urgent) or 4 (Semi-urgent) but who are actually on a trajectory toward admission or death. 

---

## 4. Novelty and Impact Potential

We are one of the only teams to repurpose the competition's leaked variables (`disposition`) into a legitimate, clinically valuable target variable. 

While others used `disposition` to artificially boost their QWK on the `triage_acuity` prediction task (data leakage), we used it to create a genuine **Waiting Room Deterioration System**. This model can run continuously in the background of an ED, re-scoring patients based on repeat vitals and flagging high-risk individuals for immediate re-triage.
