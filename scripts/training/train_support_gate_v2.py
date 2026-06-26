# train_support_gate_v2.py
# Train support gate + write support_gate_meta.json
# Reads: train_features_rss_v3.csv (or any features CSV you pass)
# Writes:
#   - support_gate.joblib   (sklearn pipeline + required feature list + threshold)
#   - support_gate_meta.json (metadata Streamlit can read)

import argparse
import json
import numpy as np
import pandas as pd
import joblib
from typing import Tuple, List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score


BASE_FEATURES = [
    "num_evidence",
    "max_rel", "mean_rel", "p90_rel",
    "max_ent", "mean_ent",
    "max_con_used", "mean_con",
    "context_mismatch_rate",
    "update_style_rate",
]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["margin_ent_minus_con"] = df["max_ent"] - df["max_con_used"]
    df["rel_support"] = df["max_ent"] * df["max_rel"]
    df["rel_contra"] = df["max_con_used"] * df["max_rel"]
    df["support_to_con_ratio"] = (df["max_ent"] + 1e-6) / (df["max_con_used"] + 1e-6)
    df["coverage_strength"] = (
        ((df["num_evidence"] >= 5).astype(float) + (df["max_rel"] >= 0.45).astype(float)) / 2.0
    )
    return df


def pick_best_threshold(y_true: np.ndarray, p1: np.ndarray) -> Tuple[float, float]:
    best_t, best_f1 = 0.50, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        pred = (p1 >= t).astype(int)
        f1 = f1_score(y_true, pred, pos_label=1)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)
    return best_t, best_f1


def make_gate_label(df: pd.DataFrame) -> Tuple[np.ndarray, dict]:
    """
    Positives are: label==1 AND strong evidence for entailment (relative to dataset)
    We auto-pick thresholds from TRUE rows.
    Returns:
      y (0/1) and a dict of the thresholds used (for meta.json)
    """
    d = df.copy()
    true_mask = (d["label"].astype(int) == 1)
    true_df = d[true_mask]
    if len(true_df) < 50:
        true_df = d

    ent_th = float(np.quantile(true_df["max_ent"].values, 0.75))
    rel_th = float(np.quantile(true_df["max_rel"].values, 0.60))
    margin_th = 0.0

    # clamp for safety
    ent_th = max(0.08, min(ent_th, 0.80))
    rel_th = max(0.35, min(rel_th, 0.90))

    y = (
        true_mask
        & (d["max_ent"] >= ent_th)
        & (d["max_rel"] >= rel_th)
        & ((d["max_ent"] - d["max_con_used"]) >= margin_th)
    ).astype(int).values

    th = {"ent_th": ent_th, "rel_th": rel_th, "margin_th": margin_th}
    return y, th


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train_features_rss_v3.csv")
    ap.add_argument("--out", default="support_gate.joblib")
    ap.add_argument("--meta_out", default="support_gate_meta.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.data).fillna(0.0)

    # Validate required base columns exist
    for f in BASE_FEATURES:
        if f not in df.columns:
            raise ValueError(f"Missing feature column: {f}")

    df = add_derived_features(df)

    FEATURES: List[str] = BASE_FEATURES + [
        "margin_ent_minus_con",
        "rel_support",
        "rel_contra",
        "support_to_con_ratio",
        "coverage_strength",
    ]

    X = df[FEATURES].astype(float).values
    y, label_thresholds = make_gate_label(df)

    pos = int(y.sum())
    neg = int((1 - y).sum())
    print(f"[INFO] Gate positives={pos} | negatives={neg}")
    if pos < 20:
        raise ValueError(
            "Too few positives to train support gate. "
            "Increase max_rows or ensure evidence/NLI is producing stronger max_ent."
        )

    strat = y if len(set(y)) > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=strat
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(class_weight="balanced", max_iter=8000, solver="lbfgs"))
    ])
    model.fit(X_train, y_train)

    p1 = model.predict_proba(X_val)[:, 1]
    best_t, best_f1 = pick_best_threshold(y_val, p1)

    print("\n=== Default threshold 0.50 ===")
    print("Confusion matrix:\n", confusion_matrix(y_val, (p1 >= 0.50).astype(int)))
    print("\nReport:\n", classification_report(y_val, (p1 >= 0.50).astype(int), digits=3))

    print(f"\n=== Best threshold {best_t:.2f} (F1 pos = {best_f1:.3f}) ===")
    print("Confusion matrix:\n", confusion_matrix(y_val, (p1 >= best_t).astype(int)))
    print("\nReport:\n", classification_report(y_val, (p1 >= best_t).astype(int), digits=3))

    # Save the joblib in a way Streamlit can load reliably
    payload = {
        "model": model,
        "required_features": FEATURES,
        "best_threshold": best_t,
        "label_thresholds": label_thresholds,
        "version": "support_gate_v2_meta",
    }
    joblib.dump(payload, args.out)
    print(f"\n[DONE] saved: {args.out}")

    # Save meta json (so you can share it + app can read it)
    meta = {
        "model_file": args.out,
        "best_threshold": best_t,
        "required_features": FEATURES,
        "label_thresholds": label_thresholds,
        "train_rows": int(len(df)),
        "positives": pos,
        "negatives": neg,
        "note": "Support gate predicts when evidence is strong enough to assert SUPPORTED safely.",
    }
    with open(args.meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[DONE] saved: {args.meta_out}")


if __name__ == "__main__":
    main()

