# train_fake_real_mlp_v3.py
# FINAL FAKE/REAL MLP trainer that is compatible with gates trained on different feature sets.
# It:
#  - loads support_gate.joblib + contradiction_gate.joblib (model or dict wrapper)
#  - reads each gate's expected feature names (if available)
#  - builds X for each gate in the correct column order, filling missing cols with 0.0
#  - computes P_support and P_contradict
#  - trains an MLP on [base features + gate probs + margin]

import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix


def unwrap_model(obj):
    if hasattr(obj, "predict_proba"):
        return obj, None  # (model, meta)
    if isinstance(obj, dict):
        # common wrappers
        for k in ["model", "pipeline", "clf", "estimator"]:
            if k in obj and hasattr(obj[k], "predict_proba"):
                return obj[k], obj
    raise TypeError("Gate joblib is not a sklearn estimator and cannot be unwrapped.")


def get_gate_feature_names(gate_meta, fallback):
    """
    Try to read expected feature names from a dict wrapper (meta).
    Otherwise return fallback.
    """
    if isinstance(gate_meta, dict):
        for k in ["feature_names", "required_features", "features", "cols"]:
            if k in gate_meta and isinstance(gate_meta[k], list) and len(gate_meta[k]) > 0:
                return list(gate_meta[k])
    return list(fallback)


def build_matrix(df: pd.DataFrame, feature_names: list) -> np.ndarray:
    """
    Returns numpy matrix with columns in the exact given order.
    Missing columns are filled with 0.0.
    """
    for c in feature_names:
        if c not in df.columns:
            df[c] = 0.0
    return df[feature_names].astype(float).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--support_gate", default="support_gate.joblib")
    ap.add_argument("--contradiction_gate", default="contradiction_gate.joblib")
    ap.add_argument("--out", default="fake_real_mlp.joblib")
    args = ap.parse_args()

    df = pd.read_csv(args.data).fillna(0.0)

    # Your RSS v3 features (baseline)
    BASE = [
        "num_evidence",
        "max_rel", "mean_rel", "p90_rel",
        "max_ent", "mean_ent",
        "max_con_used", "mean_con",
        "context_mismatch_rate",
        "update_style_rate",
        "body_success_rate",
    ]
    for c in BASE:
        if c not in df.columns:
            df[c] = 0.0

    y = df["label"].astype(int).values

    # Load and unwrap gates
    support_model, support_meta = unwrap_model(joblib.load(args.support_gate))
    contra_model, contra_meta = unwrap_model(joblib.load(args.contradiction_gate))

    # Determine what features each gate expects
    # If not stored, we fall back to BASE (but your support gate likely has extra cols)
    support_feats = get_gate_feature_names(support_meta, fallback=BASE)
    contra_feats = get_gate_feature_names(contra_meta, fallback=BASE)

    # Build input matrices for each gate in correct order
    X_support = build_matrix(df.copy(), support_feats)
    X_contra = build_matrix(df.copy(), contra_feats)

    # Gate probabilities
    P_support = support_model.predict_proba(X_support)[:, 1]
    P_contra = contra_model.predict_proba(X_contra)[:, 1]

    # Final model features = BASE + gate probs + margin
    X_base = build_matrix(df.copy(), BASE)
    X = np.column_stack([X_base, P_support, P_contra, (P_support - P_contra)])

    feature_names = BASE + ["P_support", "P_contradict", "support_minus_contradict"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # MLP
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            max_iter=600,
            random_state=42
        ))
    ])
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nReport:")
    print(classification_report(y_test, y_pred, digits=3))

    # Save
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "base_features": BASE,
            "support_gate_features_used": support_feats,
            "contradiction_gate_features_used": contra_feats,
        },
        args.out
    )
    print(f"\n[DONE] saved: {args.out}")
    print(f"[INFO] support_gate expects {len(support_feats)} features")
    print(f"[INFO] contradiction_gate expects {len(contra_feats)} features")


if __name__ == "__main__":
    main()

