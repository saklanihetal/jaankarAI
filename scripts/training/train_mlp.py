# train_support_gate.py
# Train a "SUPPORT ASSERTION GATE" model.
# It does NOT learn real vs fake.
# It learns: "Given evidence features, should we assert SUPPORTED?"
#
# Output: support_gate.joblib (contains model + feature list + learned threshold)

import argparse
import numpy as np
import pandas as pd
import joblib

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
        (df["num_evidence"] >= 5).astype(float) + (df["max_rel"] >= 0.45).astype(float)
    ) / 2.0
    return df

def make_gate_label(df: pd.DataFrame) -> np.ndarray:
    """
    y_gate = 1 means: we SHOULD assert SUPPORTED (conservative)
    We create positives only when:
      - underlying dataset label is TRUE (label==1)
      - evidence is clearly strong and not mismatchy
    Everything else is a negative (do NOT assert support).
    """
    label_true = (df["label"].astype(int) == 1)

    evidence_strong = (
        (df["max_rel"] >= 0.45) &
        (df["max_ent"] >= 0.35) &
        ((df["max_ent"] - df["max_con_used"]) >= 0.10) &
        (df["context_mismatch_rate"] <= 0.34)
    )

    return (label_true & evidence_strong).astype(int).values

def pick_best_threshold(y_true, p1):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        pred = (p1 >= t).astype(int)
        f1 = f1_score(y_true, pred, pos_label=1)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1

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
    y_gate = make_gate_label(df)

    pos, neg = int(y_gate.sum()), int((1 - y_gate).sum())
    print(f"[INFO] Gate positives (assert support)= {pos} | negatives= {neg}")
    if pos < 30:
        print("[WARN] Too few positive examples. Consider increasing data or relaxing heuristic inside make_gate_label().")

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

    joblib.dump({"model": model, "features": FEATURES, "threshold": best_t}, args.out)
    print(f"\n[DONE] saved: {args.out}")

if __name__ == "__main__":
    main()



