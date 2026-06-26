# make_all_plots.py
# Generates confusion matrices + ROC/PR + probability histograms for:
# - TF-IDF style model vs CSV label (real/fake)
# - Support gate vs derived "supportable" target
# - Contradiction gate vs derived "contradictable" target
#
# Usage:
#   py make_all_plots.py --csv train_features_rss_v3.csv --out_dir plots

import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    precision_recall_curve,
)

# --------- helpers ---------
FEATURE_COLUMNS = [
    "num_evidence",
    "max_rel", "mean_rel", "p90_rel",
    "max_ent", "mean_ent",
    "max_con_used", "mean_con",
    "body_success_rate",
    "context_mismatch_rate",
    "update_style_rate",
]

def ensure_out(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

def save_confusion(y_true, y_pred, title, out_path, labels=("0","1")):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(cm, display_labels=list(labels))
    fig, ax = plt.subplots(figsize=(5,4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def save_roc(y_true, y_prob, title, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    ax.plot([0,1],[0,1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def save_pr(y_true, y_prob, title, out_path):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(rec, prec)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def save_prob_hist(y_true, y_prob, title, out_path):
    fig, ax = plt.subplots(figsize=(6,4))
    y_true = np.asarray(y_true).astype(int)
    ax.hist(y_prob[y_true==0], bins=25, alpha=0.7, label="True=0")
    ax.hist(y_prob[y_true==1], bins=25, alpha=0.7, label="True=1")
    ax.set_xlabel("Predicted probability (positive class)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def report_metrics(y_true, y_pred, y_prob=None):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    out = {"accuracy": float(acc), "precision": float(p), "recall": float(r), "f1": float(f1)}
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        out["auc"] = float(auc(fpr, tpr))
    return out

def load_meta(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def load_gate(joblib_path: str):
    obj = joblib.load(joblib_path)
    if hasattr(obj, "predict_proba"):
        return obj
    if isinstance(obj, dict):
        for k in ["model", "pipeline", "clf", "estimator"]:
            if k in obj and hasattr(obj[k], "predict_proba"):
                return obj[k]
    raise TypeError(f"{joblib_path} is not a sklearn estimator with predict_proba")

def gate_predict_proba(model, df_features: pd.DataFrame, feat_list):
    X = df_features.reindex(columns=feat_list).fillna(0.0)
    p = model.predict_proba(X)[:, 1]
    return p

# --------- Derived targets (same logic as your trainers) ---------
def derived_support_target(df: pd.DataFrame):
    # matches train_support_gate_v2.py behavior:
    # true_mask = (label == 1) then thresholds from true_df quantiles
    d = df.copy()
    true_mask = (d["label"].astype(int) == 1)

    true_df = d[true_mask]
    if len(true_df) < 10:
        # edge fallback: if very few positives, return zeros
        return np.zeros(len(d), dtype=int), {"note": "too_few_true_rows"}

    ent_th = float(np.quantile(true_df["max_ent"].values, 0.75))
    rel_th = float(np.quantile(true_df["max_rel"].values, 0.60))
    margin_th = 0.0

    ent_th = max(0.08, min(ent_th, 0.80))
    rel_th = max(0.35, min(rel_th, 0.90))

    y = (
        true_mask
        & (d["max_ent"] >= ent_th)
        & (d["max_rel"] >= rel_th)
        & ((d["max_ent"] - d["max_con_used"]) >= margin_th)
    ).astype(int).values

    return y, {"ent_th": ent_th, "rel_th": rel_th, "margin_th": margin_th}

def robust_quantile(x, q):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0
    return float(np.quantile(x, q))

def derived_contra_target(df: pd.DataFrame):
    # matches train_contradiction_gate_v1.py behavior (approx exact)
    d = df.copy()
    max_rel = d["max_rel"].astype(float).values
    max_con_used = d["max_con_used"].astype(float).values
    max_ent = d["max_ent"].astype(float).values
    ctx_mismatch = d.get("context_mismatch_rate", pd.Series(np.zeros(len(d)))).astype(float).values
    upd_style = d.get("update_style_rate", pd.Series(np.zeros(len(d)))).astype(float).values

    rel_th = max(0.30, robust_quantile(max_rel, 0.60))
    con_th = max(0.10, robust_quantile(max_con_used, 0.90))
    ent_max = max(0.10, robust_quantile(max_ent, 0.35))
    ctx_th = robust_quantile(ctx_mismatch, 0.80)
    upd_th = robust_quantile(upd_style, 0.80)

    y = (
        (max_con_used >= con_th) &
        (max_rel >= rel_th) &
        (max_ent <= ent_max)
    )

    if np.nanmax(ctx_mismatch) > 0:
        y = y | ((max_con_used >= con_th*0.85) & (max_rel >= rel_th) & (ctx_mismatch >= ctx_th) & (max_ent <= ent_max))
    if np.nanmax(upd_style) > 0:
        y = y | ((max_con_used >= con_th*0.85) & (max_rel >= rel_th) & (upd_style >= upd_th) & (max_ent <= ent_max))

    return y.astype(int), {
        "rel_th": float(rel_th),
        "con_th": float(con_th),
        "ent_max": float(ent_max),
        "ctx_th": float(ctx_th),
        "upd_th": float(upd_th),
    }

# --------- main ---------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="features csv (e.g., train_features_rss_v3.csv)")
    ap.add_argument("--out_dir", default="plots", help="folder to save png outputs")
    args = ap.parse_args()

    ensure_out(args.out_dir)

    df = pd.read_csv(args.csv)
    # make sure numeric cols are numeric
    for c in FEATURE_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # ---------- 1) TF-IDF language/style model (real/fake) ----------
    metrics_report = {}

    if os.path.exists("tfidf_style_model.joblib"):
        model = joblib.load("tfidf_style_model.joblib")
        if "headline" in df.columns and "label" in df.columns:
            y_true = df["label"].astype(int).values
            # model expects raw text; use headline
            probs = model.predict_proba(df["headline"].astype(str).tolist())[:, 1]
            y_pred = (probs >= 0.5).astype(int)

            save_confusion(
                y_true, y_pred,
                "TF-IDF Style Model: Confusion Matrix (vs label)",
                os.path.join(args.out_dir, "tfidf_confusion.png"),
                labels=("Real(0)", "Fake(1)")
            )
            save_roc(y_true, probs, "TF-IDF Style Model: ROC", os.path.join(args.out_dir, "tfidf_roc.png"))
            save_pr(y_true, probs, "TF-IDF Style Model: Precision-Recall", os.path.join(args.out_dir, "tfidf_pr.png"))
            save_prob_hist(y_true, probs, "TF-IDF Style Model: Probability Histogram", os.path.join(args.out_dir, "tfidf_prob_hist.png"))
            metrics_report["tfidf_style"] = report_metrics(y_true, y_pred, probs)
        else:
            metrics_report["tfidf_style"] = {"error": "CSV missing headline/label columns"}
    else:
        metrics_report["tfidf_style"] = {"error": "tfidf_style_model.joblib not found in current folder"}

    # ---------- 2) Support Gate (derived target) ----------
    if os.path.exists("support_gate.joblib"):
        gate = load_gate("support_gate.joblib")
        meta = load_meta("support_gate_meta.json")
        feat_list = meta.get("required_features") or meta.get("feature_names") or FEATURE_COLUMNS

        y_true, th_info = derived_support_target(df)
        probs = gate_predict_proba(gate, df, feat_list)
        thr = float(meta.get("best_threshold", 0.35))
        y_pred = (probs >= thr).astype(int)

        save_confusion(
            y_true, y_pred,
            f"Support Gate: Confusion Matrix (thr={thr:.2f})",
            os.path.join(args.out_dir, "support_gate_confusion.png"),
            labels=("NotSupportable(0)", "Supportable(1)")
        )
        save_roc(y_true, probs, "Support Gate: ROC", os.path.join(args.out_dir, "support_gate_roc.png"))
        save_pr(y_true, probs, "Support Gate: Precision-Recall", os.path.join(args.out_dir, "support_gate_pr.png"))
        save_prob_hist(y_true, probs, "Support Gate: Probability Histogram", os.path.join(args.out_dir, "support_gate_prob_hist.png"))

        metrics_report["support_gate"] = {
            **report_metrics(y_true, y_pred, probs),
            "threshold_used": thr,
            "derived_target_thresholds": th_info,
            "positives_in_derived_target": int(y_true.sum()),
        }
    else:
        metrics_report["support_gate"] = {"error": "support_gate.joblib not found in current folder"}

    # ---------- 3) Contradiction Gate (derived target) ----------
    if os.path.exists("contradiction_gate.joblib"):
        gate = load_gate("contradiction_gate.joblib")
        meta = load_meta("contradiction_gate_meta.json")
        feat_list = meta.get("required_features") or meta.get("feature_names") or FEATURE_COLUMNS

        y_true, th_info = derived_contra_target(df)
        probs = gate_predict_proba(gate, df, feat_list)
        thr = float(meta.get("best_threshold", 0.70))
        y_pred = (probs >= thr).astype(int)

        save_confusion(
            y_true, y_pred,
            f"Contradiction Gate: Confusion Matrix (thr={thr:.2f})",
            os.path.join(args.out_dir, "contradiction_gate_confusion.png"),
            labels=("NotContradictable(0)", "Contradictable(1)")
        )
        save_roc(y_true, probs, "Contradiction Gate: ROC", os.path.join(args.out_dir, "contradiction_gate_roc.png"))
        save_pr(y_true, probs, "Contradiction Gate: Precision-Recall", os.path.join(args.out_dir, "contradiction_gate_pr.png"))
        save_prob_hist(y_true, probs, "Contradiction Gate: Probability Histogram", os.path.join(args.out_dir, "contradiction_gate_prob_hist.png"))

        metrics_report["contradiction_gate"] = {
            **report_metrics(y_true, y_pred, probs),
            "threshold_used": thr,
            "derived_target_thresholds": th_info,
            "positives_in_derived_target": int(y_true.sum()),
        }
    else:
        metrics_report["contradiction_gate"] = {"error": "contradiction_gate.joblib not found in current folder"}

    # Save metrics JSON for paper/poster
    out_json = os.path.join(args.out_dir, "metrics_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)

    print("\nDONE. Saved plots to:", args.out_dir)
    print("Saved metrics JSON:", out_json)
    print("\nKey files created:")
    for fn in sorted(os.listdir(args.out_dir)):
        if fn.endswith(".png") or fn.endswith(".json"):
            print(" -", fn)

if __name__ == "__main__":
    main()
