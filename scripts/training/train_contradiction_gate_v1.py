# train_contradiction_gate_v1.py
# Trains a CONTRADICTION gate model using your feature CSV (RSS v3).
# Output: contradiction_gate.joblib
#
# Gate target = "safe to assert CONTRADICTED" (1) vs "do NOT assert" (0).
# Auto-labeling is based on strong contradiction signal with weak support and decent relevance.

import argparse
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score


# ---- utilities ----
def ensure_cols(df: pd.DataFrame, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    return df

def robust_quantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0
    return float(np.quantile(x, q))

def auto_label_contradiction(df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """
    Auto-label rule (CONTRADICTION gate):
    Positive (1) if:
      - max_con_used is high enough
      - max_rel is not tiny
      - max_ent is low (not supported)
      - optionally context mismatch high (if present) -> helps replica/location style mismatch cases
    """
    max_rel = df["max_rel"].astype(float).values
    max_ent = df["max_ent"].astype(float).values
    max_con_used = df["max_con_used"].astype(float).values
    ctx_mismatch = df.get("context_mismatch_rate", pd.Series(np.zeros(len(df)))).astype(float).values
    upd_style = df.get("update_style_rate", pd.Series(np.zeros(len(df)))).astype(float).values

    # Data-driven thresholds (stable across datasets)
    # We want contradiction gate to be conservative, so use higher quantiles.
    rel_th = max(0.25, robust_quantile(max_rel, 0.60))
    con_th = max(0.10, robust_quantile(max_con_used, 0.90))
    ent_max = max(0.10, robust_quantile(max_ent, 0.35))  # "support should be low"
    ctx_th = robust_quantile(ctx_mismatch, 0.80)         # optional boost signal
    upd_th = robust_quantile(upd_style, 0.80)            # optional boost signal

    # Base rule
    y = (
        (max_con_used >= con_th) &
        (max_rel >= rel_th) &
        (max_ent <= ent_max)
    )

    # If mismatch-style features exist and are meaningful, allow them to strengthen positives slightly
    # (but do not make them mandatory)
    if np.nanmax(ctx_mismatch) > 0:
        y = y | ( (max_con_used >= con_th*0.85) & (max_rel >= rel_th) & (ctx_mismatch >= ctx_th) & (max_ent <= ent_max) )
    if np.nanmax(upd_style) > 0:
        y = y | ( (max_con_used >= con_th*0.85) & (max_rel >= rel_th) & (upd_style >= upd_th) & (max_ent <= ent_max) )

    y = y.astype(int)

    info = {
        "rel_th": float(rel_th),
        "con_th": float(con_th),
        "ent_max": float(ent_max),
        "ctx_th": float(ctx_th),
        "upd_th": float(upd_th),
        "positives": int(y.sum()),
        "negatives": int((1 - y).sum()),
    }
    return y, info

def best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """
    Pick threshold maximizing F1 for the positive class.
    """
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 19):
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, float(best_f1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV of features (e.g., train_features_rss_v3.csv)")
    ap.add_argument("--out", default="contradiction_gate.joblib")
    ap.add_argument("--meta", default="contradiction_gate_meta.json")
    args = ap.parse_args()

    df = pd.read_csv(args.data).fillna(0.0)

    REQUIRED = [
        "num_evidence",
        "max_rel", "mean_rel", "p90_rel",
        "max_ent", "mean_ent",
        "max_con_used", "mean_con",
        "body_success_rate",
        "context_mismatch_rate",
        "update_style_rate",
    ]
    df = ensure_cols(df, REQUIRED)

    # ---- auto label ----
    y, info = auto_label_contradiction(df)
    print(f"[INFO] Auto thresholds: rel>={info['rel_th']:.3f}, con>={info['con_th']:.3f}, ent<={info['ent_max']:.3f}")
    print(f"[INFO] Gate positives (assert CONTRADICTED)={info['positives']} | negatives={info['negatives']}")

    if info["positives"] < 10:
        raise ValueError(
            "Too few positives to train contradiction gate. "
            "This usually means max_con_used is too weak. "
            "Next step: increase max_rows or strengthen contradiction features."
        )

    # ---- features ----
    X = df[REQUIRED].astype(float).values

    # Stratify so we actually get positives in the test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ---- model ----
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=300, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)

    # ---- evaluate default threshold ----
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)

    print("\n=== Default threshold 0.50 ===")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nReport:\n", classification_report(y_test, y_pred, digits=3, zero_division=0))

    # ---- best threshold ----
    t_best, f1_best = best_threshold(y_test, y_prob)
    y_pred2 = (y_prob >= t_best).astype(int)

    print(f"\n=== Best threshold {t_best:.2f} (F1 pos = {f1_best:.3f}) ===")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred2))
    print("\nReport:\n", classification_report(y_test, y_pred2, digits=3, zero_division=0))

    # ---- save ----
    import joblib
    joblib.dump(model, args.out)

    meta = {
        "required_features": REQUIRED,
        "auto_label_info": info,
        "default_threshold": 0.50,
        "best_threshold": t_best,
        "best_f1_pos": f1_best,
    }
    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[DONE] saved: {args.out}")
    print(f"[DONE] saved: {args.meta}")


if __name__ == "__main__":
    main()
