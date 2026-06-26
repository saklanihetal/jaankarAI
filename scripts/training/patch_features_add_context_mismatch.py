# patch_features_add_context_mismatch.py
import pandas as pd

path = "train_features_rss_v3.csv"
df = pd.read_csv(path)

if "context_mismatch_rate" not in df.columns:
    df["context_mismatch_rate"] = 0.0

df.to_csv(path, index=False, encoding="utf-8")
print("[DONE] Patched:", path)
print("Columns now include context_mismatch_rate =", "context_mismatch_rate" in df.columns)
