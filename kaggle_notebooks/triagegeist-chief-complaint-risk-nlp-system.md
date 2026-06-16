# 🚨 Triagegeist — Chief Complaint Risk NLP System

AI-powered NLP system for **emergency department triage risk classification** built for the **Triagegeist Challenge by the Laitinen-Fredriksson Foundation**.

This project analyzes **free-text chief complaints** from patients and predicts triage risk levels:

* 🔴 **HIGH** — Critical / Emergent (ESI 1–2)
* 🟡 **MEDIUM** — Urgent (ESI 3)
* 🟢 **LOW** — Non-urgent (ESI 4–5)

The system combines **modern sentence embeddings with gradient boosting models** to identify potentially dangerous symptoms from short clinical descriptions.

---

# 🧠 Project Goal

Emergency departments receive thousands of **free-text complaints** such as:

* "chest pain radiating to arm"
* "shortness of breath"
* "severe headache"
* "minor finger cut"

This project builds an **AI triage assistant** that automatically flags **high-risk complaints** and helps prioritize patients faster.

---

# ⚙️ Model Pipeline

The system uses a hybrid NLP + ML pipeline.

### 1. Text Processing

Chief complaints are cleaned using:

* abbreviation expansion (`cp → chest pain`, `sob → shortness of breath`)
* punctuation removal
* lowercase normalization

### 2. Feature Extraction

Two approaches are compared:

**Baseline**

* TF-IDF
* n-grams (1–3)

**Advanced NLP**

* Sentence embeddings using **SentenceTransformer**
* Model: `all-MiniLM-L6-v2`

### 3. Classification Model

Gradient boosted trees using:

* **XGBoost**

Predicts:

```
LOW
MEDIUM
HIGH
```

---

# 📊 Evaluation

Metrics used:

* Macro **F1 Score**
* Accuracy
* ROC-AUC
* Confusion Matrix

The notebook compares:

| Model                         | Features              |
| ----------------------------- | --------------------- |
| TF-IDF + XGBoost              | Bag-of-words baseline |
| SentenceTransformer + XGBoost | Semantic embeddings   |

---

# 🔍 Explainability

The model includes **SHAP explainability** to understand which embedding features influence predictions.

This helps interpret **why certain complaints are flagged as high risk**.

---

# 🧪 Example Predictions

| Complaint                   | Predicted Risk |
| --------------------------- | -------------- |
| chest pain radiating to arm | 🔴 HIGH        |
| difficulty breathing        | 🔴 HIGH        |
| nausea and vomiting         | 🟡 MEDIUM      |
| sore throat                 | 🟢 LOW         |
| finger laceration           | 🟢 LOW         |

---

# 🖥 Interactive Demo

The project includes a **Gradio interface** where users can enter a complaint and receive:

* Risk classification
* Model confidence
* Class probability distribution
* Clinical guidance note

Example input:

```
chest pain radiating to left arm with sweating
```

Output:

```
🔴 HIGH RISK
Model confidence: 92%
```



# Installations


```python
!pip install -q sentence-transformers gradio xgboost shap
```

#  Imports & Setup


```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# NLP / ML
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score
)
import xgboost as xgb

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Utility
import re
import os
import pickle
from collections import Counter
import shap

print("✅ All imports successful.")
```

#  Load Dataset


```python
TRAIN_PATH  = "/kaggle/input/competitions/triagegeist/train.csv"
TEST_PATH   = "/kaggle/input/competitions/triagegeist/test.csv"
CC_PATH     = "/kaggle/input/competitions/triagegeist/chief_complaints.csv"
SUB_PATH    = "/kaggle/input/competitions/triagegeist/sample_submission.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
cc_df    = pd.read_csv(CC_PATH)

print(f"Train shape  : {train_df.shape}")
print(f"Test shape   : {test_df.shape}")
print(f"CC shape     : {cc_df.shape}")
print("\nFirst 5 chief complaints:")
print(cc_df["chief_complaint_raw"].head())
```

# Merge Chief Complaints


```python
train_df = train_df.merge(cc_df, on="patient_id", how="left")
test_df  = test_df.merge(cc_df,  on="patient_id", how="left")

# Fill missing complaints
train_df["chief_complaint_raw"].fillna("unknown complaint", inplace=True)
test_df["chief_complaint_raw"].fillna("unknown complaint",  inplace=True)

print(f"Train after merge: {train_df.shape}")
print(f"Missing complaints (train): {train_df['chief_complaint_raw'].isna().sum()}")

```

# Define Risk Labels from ESI Acuity

### ESI 1–2  → HIGH    (immediate / emergent)
### ESI 3    → MEDIUM  (urgent)
### ESI 4–5  → LOW     (semi-urgent / non-urgent)



```python
def esi_to_risk(esi):
    if esi in [1, 2]:
        return "HIGH"
    elif esi == 3:
        return "MEDIUM"
    else:
        return "LOW"

train_df["risk_label"] = train_df["triage_acuity"].apply(esi_to_risk)

label_counts = train_df["risk_label"].value_counts()
print("\nRisk label distribution:")
print(label_counts)

# Visualise
fig, ax = plt.subplots(figsize=(7, 4))
colors = {"HIGH": "#e74c3c", "MEDIUM": "#f39c12", "LOW": "#27ae60"}
bars = ax.bar(label_counts.index, label_counts.values,
              color=[colors[l] for l in label_counts.index], edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, label_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f"{val:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_title("Risk Label Distribution (train)", fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("Count")
ax.set_xlabel("Risk Category")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("risk_distribution.png", dpi=150)
plt.show()
print("✅ Risk labels created.")

```

#  Text Preprocessing


```python
def clean_complaint(text: str) -> str:
    """
    Light clinical text cleaning:
    - Lowercase
    - Remove punctuation / digits (keep letters & spaces)
    - Strip extra whitespace
    - Preserve key clinical abbreviations by expanding them first
    """
    ABBREV = {
        r"\bcp\b":  "chest pain",
        r"\bsob\b": "shortness of breath",
        r"\bdob\b": "difficulty of breathing",
        r"\bha\b":  "headache",
        r"\babe\b": "abdominal pain",
        r"\bloc\b": "loss of consciousness",
        r"\bms\b":  "mental status",
        r"\bsz\b":  "seizure",
        r"\bcc\b":  "chief complaint",
        r"\bn/v\b": "nausea vomiting",
        r"\bhtn\b": "hypertension",
        r"\bdm\b":  "diabetes",
        r"\beti\b": "endotracheal intubation",
        r"\brta\b": "road traffic accident",
        r"\bmva\b": "motor vehicle accident",
    }
    text = str(text).lower()
    for pattern, expansion in ABBREV.items():
        text = re.sub(pattern, expansion, text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

train_df["complaint_clean"] = train_df["chief_complaint_raw"].apply(clean_complaint)
test_df["complaint_clean"]  = test_df["chief_complaint_raw"].apply(clean_complaint)

print("Sample cleaned complaints:")
for raw, clean in zip(
    train_df["chief_complaint_raw"].head(5),
    train_df["complaint_clean"].head(5)
):
    print(f"  RAW  : {raw}")
    print(f"  CLEAN: {clean}\n")
```

#  Feature Engineering: TF-IDF Baseline


```python
LABEL_ORDER = ["LOW", "MEDIUM", "HIGH"]
le = LabelEncoder()
le.fit(LABEL_ORDER)

X_text = train_df["complaint_clean"].values
y      = le.transform(train_df["risk_label"].values)

X_train_t, X_val_t, y_train, y_val = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF + XGBoost pipeline
tfidf_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=10_000,
        sublinear_tf=True,
        min_df=2
    )),
    ("clf", xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    ))
])

print("Training TF-IDF + XGBoost …")
tfidf_pipe.fit(X_train_t, y_train)
y_pred_tfidf = tfidf_pipe.predict(X_val_t)

print("\n── TF-IDF + XGBoost Results ──")
print(classification_report(
    y_val, y_pred_tfidf,
    target_names=le.classes_
))

tfidf_f1 = f1_score(y_val, y_pred_tfidf, average="macro")
print(f"Macro F1: {tfidf_f1:.4f}")

```

# Feature Engineering: SentenceTransformer Embeddings


```python
EMBED_MODEL = "all-MiniLM-L6-v2"   # fast & strong general-purpose model
print(f"Loading SentenceTransformer: {EMBED_MODEL} …")
st_model = SentenceTransformer(EMBED_MODEL)

def encode_texts(texts, batch_size=256, show_progress=True):
    return st_model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True   # cosine-compatible
    )

print("Encoding training complaints …")
X_emb_train = encode_texts(X_train_t)
print("Encoding validation complaints …")
X_emb_val   = encode_texts(X_val_t)
print("Encoding test complaints …")
X_emb_test  = encode_texts(test_df["complaint_clean"].values)

print(f"\nEmbedding shape: {X_emb_train.shape}")
```

# Train XGBoost on Embeddings


```python
xgb_emb = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=30
)

xgb_emb.fit(
    X_emb_train, y_train,
    eval_set=[(X_emb_val, y_val)],
    verbose=50
)

y_pred_emb = xgb_emb.predict(X_emb_val)
y_prob_emb = xgb_emb.predict_proba(X_emb_val)

print("\n── SentenceTransformer + XGBoost Results ──")
print(classification_report(y_val, y_pred_emb, target_names=le.classes_))
emb_f1 = f1_score(y_val, y_pred_emb, average="macro")
print(f"Macro F1: {emb_f1:.4f}")

try:
    auc = roc_auc_score(y_val, y_prob_emb, multi_class="ovr", average="macro")
    print(f"ROC-AUC (OVR macro): {auc:.4f}")
except Exception:
    pass
```

#  Confusion Matrix Visualisation


```python
def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, subtitle in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Raw Counts", "Row-Normalised"]
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="YlOrRd",
            xticklabels=labels, yticklabels=labels,
            linewidths=0.5, linecolor="white", ax=ax,
            annot_kws={"size": 12, "weight": "bold"}
        )
        ax.set_title(f"{title} — {subtitle}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)

    plt.suptitle(f"Model: {title}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"cm_{title.replace(' ','_').lower()}.png", dpi=150, bbox_inches="tight")
    plt.show()

plot_confusion_matrix(y_val, y_pred_emb, le.classes_, "SentenceTransformer XGBoost")
```

# Model Comparison


```python
results = {
    "TF-IDF + XGBoost":           f1_score(y_val, y_pred_tfidf, average="macro"),
    "SentenceTransformer + XGBoost": f1_score(y_val, y_pred_emb, average="macro"),
}

fig, ax = plt.subplots(figsize=(8, 4))
model_names = list(results.keys())
f1_scores   = list(results.values())
bar_colors  = ["#3498db", "#e74c3c"]
bars = ax.barh(model_names, f1_scores, color=bar_colors, edgecolor="white", height=0.5)
for bar, val in zip(bars, f1_scores):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=11, fontweight="bold")
ax.set_xlim(0, 1.05)
ax.set_xlabel("Macro F1 Score")
ax.set_title("Model Comparison — Chief Complaint Risk Classification",
             fontsize=13, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()
```

#  SHAP Explainability


```python
print("Computing SHAP values (this may take a few minutes on CPU) …")
bg_idx = np.random.choice(len(X_emb_train), size=200, replace=False)
explainer = shap.TreeExplainer(xgb_emb, data=X_emb_train[bg_idx])
shap_vals  = explainer.shap_values(X_emb_val[:200])

# Summary plot for HIGH risk class (index 0 = HIGH after LabelEncoder)
high_idx = list(le.classes_).index("HIGH")
plt.figure(figsize=(10, 5))
shap.summary_plot(
    shap_vals[high_idx] if isinstance(shap_vals, list) else shap_vals,
    X_emb_val[:200],
    max_display=15,
    show=False,
    plot_type="bar"
)
plt.title("SHAP Feature Importance — HIGH Risk Class", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("shap_high_risk.png", dpi=150)
plt.show()
print("✅ SHAP complete.")
```

# Risk Inference Helper


```python
RISK_COLORS = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}

CLINICAL_NOTES = {
    "HIGH": (
        "⚠️  IMMEDIATE attention required. "
        "Complaint pattern associated with ESI 1–2 presentations. "
        "Recommend rapid assessment and resource mobilisation."
    ),
    "MEDIUM": (
        "⏳  Urgent but stable. "
        "Complaint pattern consistent with ESI 3. "
        "Monitor for deterioration while awaiting workup."
    ),
    "LOW": (
        "✅  Lower acuity presentation. "
        "Consistent with ESI 4–5. "
        "Standard triage queue appropriate; reassess if vitals change."
    ),
}

def predict_risk(complaint_text: str) -> dict:
    """
    Given a raw chief complaint string, return:
      - risk_label  : HIGH / MEDIUM / LOW
      - probabilities: dict of class → confidence
      - clinical_note: guidance text
    """
    cleaned  = clean_complaint(complaint_text)
    embedding = st_model.encode([cleaned], normalize_embeddings=True)
    pred_idx  = xgb_emb.predict(embedding)[0]
    probs     = xgb_emb.predict_proba(embedding)[0]
    label     = le.inverse_transform([pred_idx])[0]
    prob_dict = {le.classes_[i]: float(probs[i]) for i in range(len(le.classes_))}

    return {
        "risk_label":    label,
        "icon":          RISK_COLORS[label],
        "probabilities": prob_dict,
        "clinical_note": CLINICAL_NOTES[label],
        "cleaned_input": cleaned,
    }

# Quick sanity checks
test_complaints = [
    "chest pain radiating to left arm, diaphoresis",
    "nausea and vomiting since yesterday",
    "finger laceration, minor bleeding, no numbness",
    "unresponsive, found on floor",
    "sore throat for 3 days",
    "difficulty breathing, oxygen saturation dropping",
]

print("\n── Sanity Check ──")
for c in test_complaints:
    r = predict_risk(c)
    bar = "█" * int(r["probabilities"][r["risk_label"]] * 20)
    print(f"{r['icon']} [{r['risk_label']:6}] {bar:20}  \"{c}\"")

```

#  Generate Test Predictions & Submission File


```python
print("\nGenerating test-set predictions …")
X_emb_test_full = encode_texts(test_df["complaint_clean"].values, show_progress=True)
test_pred_idx    = xgb_emb.predict(X_emb_test_full)
test_risk_labels = le.inverse_transform(test_pred_idx)
test_probs       = xgb_emb.predict_proba(X_emb_test_full)

# Map risk back to ESI for submission (use mode ESI per risk band)
RISK_TO_ESI = {"HIGH": 2, "MEDIUM": 3, "LOW": 4}
test_esi_preds = [RISK_TO_ESI[r] for r in test_risk_labels]

submission = pd.DataFrame({
    "patient_id":    test_df["patient_id"],
    "triage_acuity": test_esi_preds
})
submission.to_csv("submission.csv", index=False)

print(f"Submission saved. Shape: {submission.shape}")
print(submission.head())

# Risk distribution in test predictions
print("\nTest prediction risk distribution:")
print(Counter(test_risk_labels))
```

# Gradio interface


```python
import gradio as gr

# ── colour / style helpers ────────────────────────────────────────────────────
BADGE_STYLE = {
    "HIGH":   "background:#c0392b;color:white;padding:6px 18px;border-radius:20px;"
              "font-size:1.1em;font-weight:bold;letter-spacing:1px;",
    "MEDIUM": "background:#d35400;color:white;padding:6px 18px;border-radius:20px;"
              "font-size:1.1em;font-weight:bold;letter-spacing:1px;",
    "LOW":    "background:#27ae60;color:white;padding:6px 18px;border-radius:20px;"
              "font-size:1.1em;font-weight:bold;letter-spacing:1px;",
}

def build_prob_bar_html(probs: dict) -> str:
    order  = ["HIGH", "MEDIUM", "LOW"]
    colors = {"HIGH": "#c0392b", "MEDIUM": "#d35400", "LOW": "#27ae60"}
    rows   = []
    for label in order:
        p   = probs.get(label, 0.0)
        pct = p * 100
        rows.append(f"""
        <div style="margin:6px 0;">
          <div style="display:flex;align-items:center;gap:10px;">
            <span style="width:60px;font-size:0.85em;font-weight:600;color:{colors[label]};">{label}</span>
            <div style="flex:1;background:#eee;border-radius:6px;height:14px;overflow:hidden;">
              <div style="width:{pct:.1f}%;background:{colors[label]};height:100%;
                          border-radius:6px;transition:width 0.4s ease;"></div>
            </div>
            <span style="width:48px;text-align:right;font-size:0.85em;color:#555;">{pct:.1f}%</span>
          </div>
        </div>""")
    return "".join(rows)

def gradio_predict(complaint_text: str) -> tuple:
    """Returns (risk_html, confidence_html, clinical_note)"""
    if not complaint_text.strip():
        return (
            "<p style='color:#999;'>Enter a complaint above.</p>",
            "",
            ""
        )

    result = predict_risk(complaint_text)
    label  = result["risk_label"]
    probs  = result["probabilities"]
    note   = result["clinical_note"]
    confidence = probs[label] * 100

    risk_html = f"""
    <div style="text-align:center;padding:16px 0;">
      <span style="{BADGE_STYLE[label]}">{result['icon']}  {label} RISK</span>
      <p style="margin-top:10px;font-size:0.9em;color:#777;">
        Model confidence: <strong>{confidence:.1f}%</strong>
      </p>
    </div>"""

    prob_html = f"""
    <div style="padding:8px 0;">
      <p style="font-size:0.85em;color:#555;margin-bottom:8px;">
        <strong>Class Probabilities</strong>
      </p>
      {build_prob_bar_html(probs)}
    </div>"""

    return risk_html, prob_html, note

# ── Example complaints ────────────────────────────────────────────────────────
EXAMPLES = [
    ["chest pain radiating to left arm with diaphoresis"],
    ["sudden severe headache, worst of life, neck stiffness"],
    ["nausea and vomiting for two days, unable to keep fluids down"],
    ["sore throat, mild fever for 3 days"],
    ["finger laceration, bleeding controlled, no numbness"],
    ["difficulty breathing, SpO2 dropping, cyanosis"],
    ["altered mental status, found unresponsive at home"],
    ["ankle sprain after playing football, can bear weight"],
]

# ── Gradio interface ──────────────────────────────────────────────────────────
with gr.Blocks(
    title="Triagegeist — Chief Complaint Risk NLP",
    theme=gr.themes.Base(
        primary_hue="red",
        secondary_hue="orange",
        font=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
    ),
    css="""
    #header {
        text-align: center;
        padding: 24px 0 8px 0;
        border-bottom: 2px solid #e74c3c;
        margin-bottom: 20px;
    }
    #header h1 {
        font-size: 2em;
        letter-spacing: 3px;
        color: #c0392b;
        margin: 0;
    }
    #header p {
        color: #777;
        font-size: 0.9em;
        margin: 6px 0 0 0;
    }
    .disclaimer {
        font-size:0.75em;
        color:#999;
        border-top:1px solid #eee;
        padding-top:10px;
        margin-top:16px;
    }
    """
) as demo:

    gr.HTML("""
    <div id="header">
      <h1>🚨 TRIAGEGEIST</h1>
      <p>Chief Complaint Risk NLP System &nbsp;|&nbsp;
         Laitinen-Fredriksson Foundation &nbsp;|&nbsp;
         Emergency Severity Index AI</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=3):
            complaint_input = gr.Textbox(
                label="Chief Complaint (free text)",
                placeholder="e.g. chest pain radiating to left arm, diaphoresis…",
                lines=3,
                max_lines=6,
            )
            analyze_btn = gr.Button(
                "🔍  Analyse Risk",
                variant="primary",
                size="lg"
            )
            gr.Examples(
                examples=EXAMPLES,
                inputs=complaint_input,
                label="📋  Example Complaints"
            )

        with gr.Column(scale=2):
            risk_output      = gr.HTML(label="Risk Assessment")
            prob_output      = gr.HTML(label="Confidence")
            note_output      = gr.Markdown(label="Clinical Guidance")

    analyze_btn.click(
        fn=gradio_predict,
        inputs=complaint_input,
        outputs=[risk_output, prob_output, note_output],
    )

    # Also trigger on Enter
    complaint_input.submit(
        fn=gradio_predict,
        inputs=complaint_input,
        outputs=[risk_output, prob_output, note_output],
    )

    gr.HTML("""
    <div class="disclaimer">
      ⚠️ <strong>Research Tool Only.</strong> This system is trained on synthetic data
      generated by the Laitinen-Fredriksson Foundation for the Triagegeist challenge.
      It is not validated for clinical use and must not replace physician or nurse judgement.
      All predictions should be reviewed by a qualified clinician.
    </div>
    """)

print("\nLaunching Gradio demo …")
demo.launch(share=True, debug=False)

```
