# train_support_gate.py (v3)
# Train SUPPORT ASSERTION GATE with a robust auto-labeling scheme.
# Fix: do NOT require context_mismatch_rate==0 (too strict)
# Uses percentiles on TRUE rows to form positives.

import argparse
import numpy as np
import pandas as pd
import joblib
from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score


BASE_FEATURES = [
    "num_evidence",
    "max_rel","mean_rel","p90_rel",
    "max_ent","mean_ent",
    "max_con_used","mean_con",
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

def pick_best_threshold(y_true, p1):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        pred = (p1 >= t).astype(int)
        f1 = f1_score(y_true, pred, pos_label=1)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1

def make_gate_label_auto(df: pd.DataFrame) -> Tuple[np.ndarray, dict]:
    """
    Positives: TRUE rows that look "high evidence support" relative to other TRUE rows.
    Robust version: do NOT hard-require mismatch==0.
    """
    d = df.copy()
    true_mask = (d["label"].astype(int) == 1)
    true_df = d[true_mask]
    if len(true_df) < 50:
        true_df = d

    # Use stricter cutoffs, but only on stable signals.
    # These are percentiles within TRUE rows, so they adapt automatically.
    rel_th = float(np.quantile(true_df["max_rel"].values, 0.75))        # top 25%
    ent_th = float(np.quantile(true_df["max_ent"].values, 0.75))        # top 25%
    margin_th = float(np.quantile(true_df["margin_ent_minus_con"].values, 0.60))  # above median-ish

    # Clamp to sensible ranges
    rel_th = max(0.30, min(rel_th, 0.90))
    ent_th = max(0.03, min(ent_th, 0.70))
    margin_th = max(-0.10, min(margin_th, 0.40))

    evidence_strong = (
        (d["max_rel"] >= rel_th) &
        (d["max_ent"] >= ent_th) &
        (d["margin_ent_minus_con"] >= margin_th)
    )

    y_gate = (true_mask & evidence_strong).astype(int).values

    info = {
        "rel_th": rel_th,
        "ent_th": ent_th,
        "margin_th": margin_th,
        "positives": int(y_gate.sum()),
        "negatives": int((1 - y_gate).sum()),
    }
    return y_gate, info

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train_features.csv")
    ap.add_argument("--out", default="support_gate.joblib")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.data).fillna(0.0)

    for f in BASE_FEATURES:
        if f not in df.columns:
            raise ValueError(f"Missing feature column: {f}")

    df = add_derived_features(df)

    FEATURES = BASE_FEATURES + [
        "margin_ent_minus_con",
        "rel_support",
        "rel_contra",
        "support_to_con_ratio",
        "coverage_strength",
    ]

    X = df[FEATURES].astype(float).values
    y_gate, info = make_gate_label_auto(df)

    print(
        f"[INFO] Auto thresholds: rel>={info['rel_th']:.3f}, ent>={info['ent_th']:.3f}, margin>={info['margin_th']:.3f}"
    )
    print(f"[INFO] Gate positives (assert SUPPORTED)= {info['positives']} | negatives= {info['negatives']}")

    if info["positives"] < 20:
        raise ValueError(
            "Too few positives to train a gate even after robust auto-labeling.\n"
            "This means your max_ent signal is extremely weak in train_features.csv.\n"
            "Next step: rebuild training features using stronger evidence text (article body extraction)."
        )

    strat = y_gate if len(set(y_gate)) > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_gate, test_size=0.2, random_state=args.seed, stratify=strat
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            class_weight="balanced",
            max_iter=8000,
            solver="lbfgs"
        ))
    ])

    model.fit(X_train, y_train)

    p1 = model.predict_proba(X_val)[:, 1]
    best_t, best_f1 = pick_best_threshold(y_val, p1)

    pred_default = (p1 >= 0.50).astype(int)
    pred_best = (p1 >= best_t).astype(int)

    print("\n=== Default threshold 0.50 ===")
    print("Confusion matrix:\n", confusion_matrix(y_val, pred_default))
    print("\nReport:\n", classification_report(y_val, pred_default, digits=3))

    print(f"\n=== Best threshold {best_t:.2f} (F1 pos = {best_f1:.3f}) ===")
    print("Confusion matrix:\n", confusion_matrix(y_val, pred_best))
    print("\nReport:\n", classification_report(y_val, pred_best, digits=3))

    joblib.dump(
        {"model": model, "features": FEATURES, "threshold": best_t, "auto_thresholds": info},
        args.out
    )
    print(f"\n[DONE] saved: {args.out}")

if __name__ == "__main__":
    main()




