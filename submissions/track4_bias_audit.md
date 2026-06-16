# Triagegeist: Bias Audit - Demographic Undertriage in Physiological Models

## 1. Clinical Problem Statement

Machine learning models deployed in emergency triage must be equitable. If a model systematically undertriages (assigns a lower acuity than necessary) specific demographics, it can lead to preventable morbidity and mortality for marginalized groups.

**Our Objective**: To conduct a rigorous demographic audit of our "Clean-Room" physiological baseline model (Track 1) to determine if reliance solely on vitals and history introduces systemic bias across age, sex, language, or insurance type.

---

## 2. Methodology: Undertriage Auditing

We define **Undertriage** as the model predicting a numerically higher ESI level (lower acuity) than the true label (e.g., True = ESI 1 Critical, Pred = ESI 3 Urgent).

We evaluate the out-of-fold predictions from our LightGBM physiological baseline across four key demographic axes:
1. `age_group`
2. `sex`
3. `language`
4. `insurance_type`

We specifically calculate the `undertriage_rate` (percentage of patients in the subgroup who were undertriaged) and the absolute `accuracy`.

---

## 3. Empirical Results and Findings

### Overall System Metrics
- **Overall Undertriage Rate**: `8.38%`
- **Overall Accuracy**: `84.8%`

### Subgroup Highlights: Age Disparities
Our most significant finding relates to age. 
- Patients aged **18-29** had an undertriage rate of `4.8%`.
- Patients aged **80+** had an undertriage rate of `10.8%`.

**Insight**: The model is more than twice as likely to undertriage elderly patients compared to young adults. This likely occurs because elderly patients often present with blunted physiological responses (e.g., they may not mount a tachycardia response to sepsis due to beta-blockers). Relying purely on vitals systematically disadvantages this group.

### Subgroup Highlights: Sex and Language
- **Female** undertriage rate: `8.5%` vs **Male**: `8.2%` (No massive disparity).
- **Arabic** speakers: `9.2%` vs **English** speakers: `8.6%` vs **Estonian**: `7.8%`.

### Subgroup Highlights: Insurance Type
- **Military**: `9.1%` undertriage.
- **None (Uninsured)**: `8.0%` undertriage.
- **Public**: `8.5%` undertriage.

---

## 4. Bias Mitigation: The "Elderly Safety Override"

Recognizing this critical failure mode, we developed a post-processing mitigation layer. 

**The Heuristic**: We applied a rule-based override where any patient in the `elderly` age group predicted to have an ESI $\ge 2$ (Urgent or lower) was automatically upgraded by 1 acuity level. 

**The Trade-off**:
- **Original Elderly Undertriage**: 7.55%
- **Mitigated Elderly Undertriage**: **0.51%**
- **Global Accuracy**: Dropped from 85.5% to 69.8%.

This represents a conscious, ethical clinical trade-off. We purposefully sacrificed global accuracy (and Kaggle Leaderboard points) to drastically overtriage the elderly population, ensuring a near-zero undertriage rate. In a real-world ED, overtriage requires extra resources, but undertriage costs lives.

---

## 5. Novelty and Impact Potential

While the dataset itself is synthetic, this audit mirrors real-world clinical challenges. By proving that a pure physiological model systematically undertriages the elderly, and by demonstrating how to mathematically mitigate it, we provide actionable guidance for EDs: AI triage tools must include age-adjusted interaction terms or rely on strict rule-based overrides to prevent equitable harm.
