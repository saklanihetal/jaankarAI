# train_tfidf_style_model.py
# Multilingual (Hindi/Kannada/Telugu/English) "language risk / fake-style" classifier
# CPU-friendly. Uses character n-grams TF-IDF + Logistic Regression.
#
# Input file: fake_news_simplified_LANGFILTER_labels01.csv
# Required columns: headline, language, label  (label: 1=true/real, 0=false/fake OR your scheme)
#
# Output:
#   - tfidf_style_model.joblib
#   - tfidf_style_meta.json
#
# NOTE:
# - This predicts "fake vs real" from LANGUAGE ONLY (tone/style), NOT evidence.
# - Use it ONLY when evidence verdict is UNVERIFIED.

import argparse
import json
import re
import unicodedata

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="fake_news_simplified_LANGFILTER_labels01.csv")
    ap.add_argument("--out", default="tfidf_style_model.joblib")
    ap.add_argument("--meta", default="tfidf_style_meta.json")
    ap.add_argument("--max_rows", type=int, default=60000)  # increase if you want
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--invert_label", action="store_true",
                    help="Use if your label is 1=REAL and you want model to predict 1=FAKE-style.")
    args = ap.parse_args()

    df = pd.read_csv(args.data).dropna(subset=["headline", "label"]).copy()
    df["headline"] = df["headline"].astype(str).map(normalize_text)
    df["label"] = df["label"].astype(int)

    # Optional: limit rows for speed
    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)

    # Deduplicate exact headline duplicates (prevents leakage)
    df = df.drop_duplicates(subset=["headline"]).reset_index(drop=True)

    X = df["headline"].values
    y = df["label"].values

    # If your dataset label is 1=REAL and 0=FAKE, and you want "fake-style risk" as 1,
    # set --invert_label so 1 becomes fake-style and 0 becomes real-style.
    if args.invert_label:
        y = 1 - y

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y
    )

    # Multilingual-safe TF-IDF: character n-grams
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=3,
            max_df=0.98,
            max_features=250000,
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    print("[INFO] Training TF-IDF style model...")
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nReport:")
    print(classification_report(y_test, y_pred, digits=3))

    joblib.dump(pipe, args.out)

    meta = {
        "model_type": "tfidf_char_ngrams_logreg",
        "analyzer": "char",
        "ngram_range": [3, 5],
        "min_df": 3,
        "max_df": 0.98,
        "max_features": 250000,
        "invert_label": bool(args.invert_label),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "label_mean_test": float(np.mean(y_test)),
    }

    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[DONE] saved model: {args.out}")
    print(f"[DONE] saved meta : {args.meta}")


if __name__ == "__main__":
    main()
