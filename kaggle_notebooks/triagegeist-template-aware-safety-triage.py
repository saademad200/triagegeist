import json
import re
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DATA = Path("/kaggle/input/competitions/triagegeist")
DATA = KAGGLE_DATA if KAGGLE_DATA.exists() else ROOT / "data"
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else ROOT / "deliverable"
TARGET = "triage_acuity"
ID = "patient_id"
SEED = 2026
N_FOLDS = 3


def add_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text = df["chief_complaint_raw"].fillna("").astype(str).str.lower()

    keyword_patterns = {
        "kw_chest": r"chest|pressure|palpitation|cardiac|heart",
        "kw_breath": r"breath|dyspnea|shortness|wheeze|respirat|asthma|copd",
        "kw_neuro": r"stroke|seizure|confusion|syncope|weakness|headache|thunderclap|vision|slurred",
        "kw_trauma": r"trauma|fall|fracture|crush|laceration|bleed|burn|gunshot|stab",
        "kw_infect": r"fever|sepsis|infection|cellulitis|cough|chills",
        "kw_abdo": r"abdom|vomit|diarrhea|nausea|pelvic",
        "kw_psych": r"suicid|self harm|psych|panic|anxiety|depress",
        "kw_preg": r"pregnan|vaginal bleeding|labor",
        "kw_life": r"anaphyl|unresponsive|arrest|overdose|poison|cyanosis",
    }
    for col, pattern in keyword_patterns.items():
        df[col] = text.str.contains(pattern, regex=True).astype(np.int8)

    df["complaint_len"] = text.str.len()
    df["complaint_words"] = text.str.split().map(len).fillna(0)
    df["map_recalc"] = df["diastolic_bp"] + (df["systolic_bp"] - df["diastolic_bp"]) / 3
    df["map_delta"] = (df["mean_arterial_pressure"] - df["map_recalc"]).abs()
    df["pulse_pressure_recalc"] = df["systolic_bp"] - df["diastolic_bp"]
    df["pulse_pressure_delta"] = (df["pulse_pressure"] - df["pulse_pressure_recalc"]).abs()
    df["shock_index_recalc"] = df["heart_rate"] / df["systolic_bp"].replace(0, np.nan)
    df["shock_index_delta"] = (df["shock_index"] - df["shock_index_recalc"]).abs()
    df["age_x_news2"] = df["age"] * df["news2_score"]
    df["hr_x_rr"] = df["heart_rate"] * df["respiratory_rate"]
    df["spo2_x_rr"] = df["spo2"] * df["respiratory_rate"]
    df["gcs_low"] = (df["gcs_total"] <= 13).astype(np.int8)
    df["spo2_low"] = (df["spo2"] < 92).astype(np.int8)
    df["sbp_low"] = (df["systolic_bp"] < 90).astype(np.int8)
    df["rr_high"] = (df["respiratory_rate"] >= 24).astype(np.int8)
    df["hr_high"] = (df["heart_rate"] >= 120).astype(np.int8)
    df["temp_fever"] = (df["temperature_c"] >= 38).astype(np.int8)
    df["pain_high"] = (df["pain_score"] >= 8).astype(np.int8)
    hx_cols = [c for c in df.columns if c.startswith("hx_")]
    df["history_burden"] = df[hx_cols].sum(axis=1)
    return df


def load_data():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    history = pd.read_csv(DATA / "patient_history.csv")
    complaints = pd.read_csv(DATA / "chief_complaints.csv")

    train_features = train.drop(columns=["disposition", "ed_los_hours"])
    full = pd.concat([train_features, test], ignore_index=True)
    full = full.merge(history, on=ID, how="left")
    full = full.merge(complaints, on=ID, how="left", suffixes=("", "_from_text_file"))
    duplicate = "chief_complaint_system_from_text_file"
    if duplicate in full.columns:
        full = full.drop(columns=[duplicate])

    full = add_clinical_features(full)
    x_train = full.iloc[: len(train)].copy()
    x_test = full.iloc[len(train) :].copy()
    y = train[TARGET].astype(int).to_numpy()
    return train, test, x_train, x_test, y


def model_columns(x_train: pd.DataFrame):
    categorical = [
        c
        for c in x_train.columns
        if x_train[c].dtype == "object" and c not in [ID, "chief_complaint_raw"]
    ]
    numeric = [
        c
        for c in x_train.columns
        if c not in [ID, TARGET, "chief_complaint_raw", *categorical]
        and pd.api.types.is_numeric_dtype(x_train[c])
    ]
    return numeric, categorical


def lgbm_probabilities(x_train, x_test, y, numeric, categorical, folds):
    oof = np.zeros((len(x_train), 5))
    test_probs = np.zeros((len(x_test), 5))
    columns = numeric + categorical

    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        x_tr = x_train.iloc[tr_idx][columns].copy()
        x_va = x_train.iloc[va_idx][columns].copy()
        x_te = x_test[columns].copy()

        for col in numeric:
            med = x_tr[col].median()
            x_tr[col] = x_tr[col].fillna(med)
            x_va[col] = x_va[col].fillna(med)
            x_te[col] = x_te[col].fillna(med)

        for col in categorical:
            x_tr[col] = x_tr[col].fillna("Unknown").astype(str)
            x_va[col] = x_va[col].fillna("Unknown").astype(str)
            x_te[col] = x_te[col].fillna("Unknown").astype(str)
            freq = x_tr[col].value_counts(normalize=True)
            x_tr[col] = x_tr[col].map(freq).fillna(0).astype(float)
            x_va[col] = x_va[col].map(freq).fillna(0).astype(float)
            x_te[col] = x_te[col].map(freq).fillna(0).astype(float)

        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=5,
            n_estimators=420,
            learning_rate=0.055,
            num_leaves=48,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=0.7,
            class_weight="balanced",
            random_state=SEED + fold,
            verbosity=-1,
        )
        model.fit(
            x_tr,
            y[tr_idx] - 1,
            eval_set=[(x_va, y[va_idx] - 1)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        oof[va_idx] = model.predict_proba(x_va)
        test_probs += model.predict_proba(x_te) / len(folds)
    return oof, test_probs


def text_probabilities(x_train, x_test, y, folds):
    oof = np.zeros((len(x_train), 5))
    test_probs = np.zeros((len(x_test), 5))

    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        train_text = x_train.iloc[tr_idx]["chief_complaint_raw"].fillna("")
        valid_text = x_train.iloc[va_idx]["chief_complaint_raw"].fillna("")
        test_text = x_test["chief_complaint_raw"].fillna("")

        word = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            min_df=2,
            max_features=40000,
            sublinear_tf=True,
        )
        char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=20000,
            sublinear_tf=True,
        )
        xw = word.fit_transform(train_text)
        xc = char.fit_transform(train_text)
        vw = word.transform(valid_text)
        vc = char.transform(valid_text)
        tw = word.transform(test_text)
        tc = char.transform(test_text)

        model = LogisticRegression(
            C=3.0,
            max_iter=400,
            solver="lbfgs",
            multi_class="multinomial",
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED + 100 + fold,
        )
        model.fit(sparse.hstack([xw, xc]), y[tr_idx])
        oof[va_idx] = model.predict_proba(sparse.hstack([vw, vc]))
        test_probs += model.predict_proba(sparse.hstack([tw, tc])) / len(folds)
    return oof, test_probs


def template_prior_probabilities(x_train, x_test, y, folds):
    """Complaint-template prior, fit only on each training fold for OOF hygiene."""
    oof = np.zeros((len(x_train), 5))
    test_probs = np.zeros((len(x_test), 5))
    base = np.bincount(y, minlength=6)[1:].astype(float)
    base = base / base.sum()

    train_key = x_train["chief_complaint_raw"].fillna("").astype(str)
    test_key = x_test["chief_complaint_raw"].fillna("").astype(str)

    for tr_idx, va_idx in folds:
        fold_base = np.bincount(y[tr_idx], minlength=6)[1:].astype(float)
        fold_base = fold_base / fold_base.sum()
        lookup = {}
        frame = pd.DataFrame({"key": train_key.iloc[tr_idx].to_numpy(), "target": y[tr_idx]})
        counts = frame.groupby(["key", "target"]).size().rename("n").reset_index()
        for key, group in counts.groupby("key"):
            probs = np.full(5, 0.001)
            for _, row in group.iterrows():
                probs[int(row["target"]) - 1] += row["n"]
            lookup[key] = probs / probs.sum()

        for idx in va_idx:
            oof[idx] = lookup.get(train_key.iloc[idx], fold_base)
        for i, key in enumerate(test_key):
            test_probs[i] += lookup.get(key, fold_base) / len(folds)

    return oof, test_probs


def extra_trees_probabilities(x_train, x_test, y, numeric, categorical, folds):
    oof = np.zeros((len(x_train), 5))
    test_probs = np.zeros((len(x_test), 5))
    columns = numeric + categorical

    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        x_tr = x_train.iloc[tr_idx][columns].copy()
        x_va = x_train.iloc[va_idx][columns].copy()
        x_te = x_test[columns].copy()
        for col in numeric:
            med = x_tr[col].median()
            x_tr[col] = x_tr[col].fillna(med)
            x_va[col] = x_va[col].fillna(med)
            x_te[col] = x_te[col].fillna(med)
        for col in categorical:
            x_tr[col] = x_tr[col].fillna("Unknown").astype(str)
            x_va[col] = x_va[col].fillna("Unknown").astype(str)
            x_te[col] = x_te[col].fillna("Unknown").astype(str)

        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )
        x_tr[categorical] = enc.fit_transform(x_tr[categorical])
        x_va[categorical] = enc.transform(x_va[categorical])
        x_te[categorical] = enc.transform(x_te[categorical])

        model = ExtraTreesClassifier(
            n_estimators=180,
            max_features=0.55,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEED + 200 + fold,
            n_jobs=-1,
        )
        model.fit(x_tr, y[tr_idx])
        oof[va_idx] = model.predict_proba(x_va)
        test_probs += model.predict_proba(x_te) / len(folds)
    return oof, test_probs


def evaluate(y, probabilities):
    pred = probabilities.argmax(axis=1) + 1
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y, pred, weights="quadratic")),
        "mean_absolute_error": float(mean_absolute_error(y, pred)),
        "high_acuity_recall_esi_1_2": float(recall_score(y <= 2, pred <= 2)),
        "distribution": {str(k): int(v) for k, v in pd.Series(pred).value_counts().sort_index().items()},
    }


def prediction_audit(y, probabilities):
    pred = probabilities.argmax(axis=1) + 1
    under = pred > y
    over = pred < y
    severe_under = (y <= 2) & (pred >= 3)
    cm = confusion_matrix(y, pred, labels=[1, 2, 3, 4, 5])
    return {
        "undertriage_rate": float(under.mean()),
        "overtriage_rate": float(over.mean()),
        "severe_undertriage_count": int(severe_under.sum()),
        "severe_undertriage_rate": float(severe_under.mean()),
        "error_count": int((pred != y).sum()),
        "confusion_matrix": cm.tolist(),
    }


def calibration_audit(y, probabilities, out_dir):
    pred = probabilities.argmax(axis=1) + 1
    confidence = probabilities.max(axis=1)
    correct = pred == y
    bins = np.linspace(0.0, 1.0, 11)
    rows = []

    for i in range(len(bins) - 1):
        left, right = bins[i], bins[i + 1]
        if i == len(bins) - 2:
            mask = (confidence >= left) & (confidence <= right)
        else:
            mask = (confidence >= left) & (confidence < right)
        if not mask.any():
            continue
        yt = y[mask]
        yp = pred[mask]
        rows.append(
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "n": int(mask.sum()),
                "mean_confidence": float(confidence[mask].mean()),
                "accuracy": float(correct[mask].mean()),
                "error_rate": float((yp != yt).mean()),
                "undertriage_rate": float((yp > yt).mean()),
                "severe_undertriage_rate": float(((yt <= 2) & (yp >= 3)).mean()),
                "overtriage_rate": float((yp < yt).mean()),
            }
        )

    confidence_audit = pd.DataFrame(rows)
    confidence_audit.to_csv(out_dir / "confidence_audit.csv", index=False)
    if confidence_audit.empty:
        ece = 0.0
    else:
        ece = float(
            (
                confidence_audit["n"]
                / confidence_audit["n"].sum()
                * (confidence_audit["accuracy"] - confidence_audit["mean_confidence"]).abs()
            ).sum()
        )

    class_rows = []
    y_matrix = np.zeros_like(probabilities)
    y_matrix[np.arange(len(y)), y - 1] = 1.0
    for cls in range(1, 6):
        class_rows.append(
            {
                "class": int(cls),
                "one_vs_rest_brier": float(np.mean((probabilities[:, cls - 1] - y_matrix[:, cls - 1]) ** 2)),
                "mean_probability": float(probabilities[:, cls - 1].mean()),
                "empirical_prevalence": float((y == cls).mean()),
            }
        )
    class_audit = pd.DataFrame(class_rows)
    class_audit.to_csv(out_dir / "calibration_audit.csv", index=False)

    return {
        "expected_calibration_error": ece,
        "mean_one_vs_rest_brier": float(class_audit["one_vs_rest_brier"].mean()),
        "max_one_vs_rest_brier": float(class_audit["one_vs_rest_brier"].max()),
    }


def fairness_audit(x_train, y, probabilities, out_dir):
    pred = probabilities.argmax(axis=1) + 1
    rows = []
    for column in ["sex", "age_group", "language", "insurance_type", "arrival_mode"]:
        if column not in x_train:
            continue
        for value, idx in x_train.groupby(column, dropna=False).groups.items():
            mask = np.asarray(list(idx), dtype=int)
            if len(mask) < 50:
                continue
            yt = y[mask]
            yp = pred[mask]
            rows.append(
                {
                    "group_column": column,
                    "group_value": str(value),
                    "n": int(len(mask)),
                    "accuracy": float(accuracy_score(yt, yp)),
                    "macro_f1": float(f1_score(yt, yp, average="macro")),
                    "qwk": float(cohen_kappa_score(yt, yp, weights="quadratic")),
                    "undertriage_rate": float((yp > yt).mean()),
                    "severe_undertriage_rate": float(((yt <= 2) & (yp >= 3)).mean()),
                    "overtriage_rate": float((yp < yt).mean()),
                }
            )
    audit = pd.DataFrame(rows)
    audit.to_csv(out_dir / "fairness_audit.csv", index=False)
    return audit


def template_audit(x_train, x_test, y, out_dir):
    train_key = x_train["chief_complaint_raw"].fillna("").astype(str)
    test_key = x_test["chief_complaint_raw"].fillna("").astype(str)
    grouped = pd.DataFrame({"key": train_key, "target": y}).groupby("key")["target"]
    n_classes = grouped.nunique()
    purity = grouped.value_counts(normalize=True).groupby(level=0).max()
    audit = {
        "unique_train_complaints": int(train_key.nunique()),
        "test_complaint_coverage_in_train": float(test_key.isin(set(train_key)).mean()),
        "pure_template_fraction": float((n_classes == 1).mean()),
        "train_row_fraction_in_pure_templates": float(train_key.isin(n_classes[n_classes == 1].index).mean()),
        "median_template_majority_fraction": float(purity.median()),
        "conflicting_template_count": int((n_classes > 1).sum()),
    }
    conflicts = (
        pd.DataFrame({"chief_complaint_raw": train_key, "triage_acuity": y})
        .groupby(["chief_complaint_raw", "triage_acuity"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["chief_complaint_raw", "count"], ascending=[True, False])
    )
    conflict_keys = n_classes[n_classes > 1].index
    conflicts[conflicts["chief_complaint_raw"].isin(conflict_keys)].to_csv(
        out_dir / "template_conflicts.csv", index=False
    )
    return audit


def make_high_risk_review(x_test, test_probabilities, pred_test, out_dir):
    review = x_test[[ID, "chief_complaint_raw", "age", "sex", "arrival_mode", "mental_status_triage"]].copy()
    for cls in range(1, 6):
        review[f"prob_esi_{cls}"] = test_probabilities[:, cls - 1]
    review["predicted_triage_acuity"] = pred_test
    review["high_acuity_probability"] = test_probabilities[:, :2].sum(axis=1)
    reason_cols = [
        "kw_life",
        "kw_chest",
        "kw_breath",
        "kw_neuro",
        "gcs_low",
        "spo2_low",
        "sbp_low",
        "rr_high",
        "hr_high",
        "pain_high",
    ]
    present = [c for c in reason_cols if c in x_test]
    review["safety_flags"] = x_test[present].apply(
        lambda row: ", ".join([col for col, value in row.items() if value == 1]), axis=1
    )
    review = review.sort_values(
        ["high_acuity_probability", "predicted_triage_acuity"], ascending=[False, True]
    )
    review.head(500).to_csv(out_dir / "high_risk_review.csv", index=False)


def write_audit_report(metrics, template_metrics, fairness, out_dir):
    fairness_summary = []
    if not fairness.empty:
        for col, group in fairness.groupby("group_column"):
            fairness_summary.append(
                f"- `{col}`: accuracy range {group['accuracy'].min():.6f} to "
                f"{group['accuracy'].max():.6f}; severe-undertriage max "
                f"{group['severe_undertriage_rate'].max():.6f}"
            )
    report = [
        "# Triagegeist Safety and Shortcut Audit",
        "",
        "## Model Result",
        "",
        f"- Final blend accuracy: `{metrics['blend']['accuracy']:.6f}`",
        f"- Final blend macro F1: `{metrics['blend']['macro_f1']:.6f}`",
        f"- Final blend QWK: `{metrics['blend']['quadratic_weighted_kappa']:.6f}`",
        f"- High-acuity ESI 1-2 recall: `{metrics['blend']['high_acuity_recall_esi_1_2']:.6f}`",
        f"- Severe undertriage count: `{metrics['safety_audit']['severe_undertriage_count']}`",
        f"- Expected calibration error: `{metrics['calibration_audit']['expected_calibration_error']:.6f}`",
        f"- Mean one-vs-rest Brier score: `{metrics['calibration_audit']['mean_one_vs_rest_brier']:.6f}`",
        "",
        "## Shortcut Risk",
        "",
        f"- Unique chief-complaint templates in train: `{template_metrics['unique_train_complaints']}`",
        f"- Test complaint coverage in train templates: `{template_metrics['test_complaint_coverage_in_train']:.6f}`",
        f"- Pure template fraction: `{template_metrics['pure_template_fraction']:.6f}`",
        f"- Conflicting template count: `{template_metrics['conflicting_template_count']}`",
        "",
        "The result is competitive because the synthetic text templates are highly predictive. "
        "The notebook keeps this signal explicit by separating a template-prior component from "
        "clinical feature models and by writing `template_conflicts.csv` for review.",
        "",
        "## Confidence and Calibration",
        "",
        "The notebook writes confidence-binned performance to `confidence_audit.csv` and one-vs-rest "
        "class calibration to `calibration_audit.csv`. These files are meant to separate score "
        "optimization from operational reliability: a reviewer can inspect whether high-confidence "
        "predictions are accurate and whether high-acuity cases are being hidden in lower-acuity bins.",
        "",
        "## Fairness and Undertriage Audit",
        "",
        *(fairness_summary or ["- No eligible fairness groups were available."]),
        "",
        "Generated artifacts: `submission.csv`, `metrics.json`, `fairness_audit.csv`, "
        "`confidence_audit.csv`, `calibration_audit.csv`, `template_conflicts.csv`, "
        "`high_risk_review.csv`, `model_card.md`, and `probabilities.npz`.",
        "",
    ]
    (out_dir / "audit_report.md").write_text("\n".join(report))


def write_model_card(metrics, out_dir):
    card = [
        "# Triagegeist Model Card",
        "",
        "## Intended Use",
        "",
        "This is a Kaggle competition model for assigning Emergency Severity Index style acuity "
        "labels from synthetic emergency department triage data. It is designed for retrospective "
        "benchmarking, model comparison, and safety-audit demonstration.",
        "",
        "## Not Intended Use",
        "",
        "This artifact is not a deployable medical device, not a replacement for triage nurses or "
        "physicians, and not validated for real patients. Any clinical translation would require "
        "prospective validation, local workflow study, monitoring, governance, and regulatory review.",
        "",
        "## Model",
        "",
        "The final prediction is a tuned blend of four fold-validated components: LightGBM on tabular "
        "vitals and history, TF-IDF logistic regression on chief complaints, ExtraTrees on structured "
        "features, and a fold-safe chief-complaint template prior. The template prior is kept explicit "
        "because the public competition data contain repeated synthetic complaint templates that are "
        "strongly predictive of the label.",
        "",
        "## Validation",
        "",
        f"- Accuracy: `{metrics['blend']['accuracy']:.6f}`",
        f"- Macro F1: `{metrics['blend']['macro_f1']:.6f}`",
        f"- Quadratic weighted kappa: `{metrics['blend']['quadratic_weighted_kappa']:.6f}`",
        f"- High-acuity ESI 1-2 recall: `{metrics['blend']['high_acuity_recall_esi_1_2']:.6f}`",
        f"- Severe undertriage count: `{metrics['safety_audit']['severe_undertriage_count']}`",
        f"- Expected calibration error: `{metrics['calibration_audit']['expected_calibration_error']:.6f}`",
        "",
        "## Safety Controls",
        "",
        "The deliverable includes subgroup auditing, confidence-binned calibration, severe-undertriage "
        "counts, explicit shortcut analysis, and a high-risk review queue. In a real workflow these "
        "outputs would support human review and monitoring rather than autonomous triage.",
        "",
        "## Research and Governance Basis",
        "",
        "- AHRQ describes ESI as a five-level emergency department triage algorithm based on acuity "
        "and expected resources.",
        "- FDA/Health Canada/MHRA transparency guidance for machine-learning enabled medical devices "
        "emphasizes clear communication of intended users, performance, limitations, logic, and "
        "human-AI team performance.",
        "- CONSORT-AI reporting guidance emphasizes transparent reporting for AI interventions.",
        "",
        "Reference URLs:",
        "",
        "- https://www.ahrq.gov/topics/emergency-severity-index.html",
        "- https://www.ahrq.gov/patient-safety/settings/hospital/resource/about.html",
        "- https://www.fda.gov/medical-devices/software-medical-device-samd/transparency-machine-learning-enabled-medical-devices-guiding-principles",
        "- https://www.nature.com/articles/s41591-020-1034-x",
        "",
        "## Reproducibility",
        "",
        "The notebook writes `submission.csv`, `metrics.json`, `probabilities.npz`, and all audit CSVs "
        "from one deterministic script entrypoint.",
        "",
    ]
    (out_dir / "model_card.md").write_text("\n".join(card))


def choose_blend(y, parts):
    best = None
    for w_lgb in np.linspace(0.0, 0.5, 11):
        for w_text in np.linspace(0.25, 0.9, 14):
            for w_template in np.linspace(0.0, 0.35, 8):
                w_et = 1.0 - w_lgb - w_text - w_template
                if w_et < 0:
                    continue
                blend = (
                    w_lgb * parts["lgbm"]
                    + w_text * parts["text"]
                    + w_et * parts["extra_trees"]
                    + w_template * parts["template"]
                )
                pred = blend.argmax(axis=1) + 1
                acc = accuracy_score(y, pred)
                qwk = cohen_kappa_score(y, pred, weights="quadratic")
                hi = recall_score(y <= 2, pred <= 2)
                severe_under = ((y <= 2) & (pred >= 3)).mean()
                score = acc + 0.03 * qwk + 0.012 * hi - 0.02 * severe_under
                if best is None or score > best[0]:
                    best = (score, w_lgb, w_text, w_et, w_template)
    return {"lgbm": best[1], "text": best[2], "extra_trees": best[3], "template": best[4]}


def main():
    train, test, x_train, x_test, y = load_data()
    numeric, categorical = model_columns(x_train)
    split = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(split.split(x_train, y))

    oof_lgb, test_lgb = lgbm_probabilities(x_train, x_test, y, numeric, categorical, folds)
    oof_text, test_text = text_probabilities(x_train, x_test, y, folds)
    oof_et, test_et = extra_trees_probabilities(x_train, x_test, y, numeric, categorical, folds)
    oof_template, test_template = template_prior_probabilities(x_train, x_test, y, folds)

    oof_parts = {"lgbm": oof_lgb, "text": oof_text, "extra_trees": oof_et, "template": oof_template}
    test_parts = {"lgbm": test_lgb, "text": test_text, "extra_trees": test_et, "template": test_template}
    weights = choose_blend(y, oof_parts)
    oof_blend = sum(weights[name] * probs for name, probs in oof_parts.items())
    test_blend = sum(weights[name] * probs for name, probs in test_parts.items())
    pred_test = test_blend.argmax(axis=1) + 1

    submission = pd.read_csv(DATA / "sample_submission.csv")
    submission[TARGET] = pred_test.astype(int)
    assert list(submission.columns) == [ID, TARGET]
    assert submission[TARGET].between(1, 5).all()
    submission.to_csv(OUT / "submission.csv", index=False)

    metrics = {
        "weights": weights,
        "lgbm": evaluate(y, oof_lgb),
        "text": evaluate(y, oof_text),
        "extra_trees": evaluate(y, oof_et),
        "template": evaluate(y, oof_template),
        "blend": evaluate(y, oof_blend),
        "safety_audit": prediction_audit(y, oof_blend),
        "calibration_audit": calibration_audit(y, oof_blend, OUT),
        "template_audit": template_audit(x_train, x_test, y, OUT),
        "test_distribution": {
            str(k): int(v)
            for k, v in submission[TARGET].value_counts().sort_index().items()
        },
        "feature_counts": {"numeric": len(numeric), "categorical": len(categorical)},
    }
    fairness = fairness_audit(x_train, y, oof_blend, OUT)
    make_high_risk_review(x_test, test_blend, pred_test, OUT)
    write_audit_report(metrics, metrics["template_audit"], fairness, OUT)
    write_model_card(metrics, OUT)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(
        OUT / "probabilities.npz",
        oof_blend=oof_blend,
        test_blend=test_blend,
        oof_lgb=oof_lgb,
        oof_text=oof_text,
        oof_extra_trees=oof_et,
        oof_template=oof_template,
        test_lgb=test_lgb,
        test_text=test_text,
        test_extra_trees=test_et,
        test_template=test_template,
        y=y,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
