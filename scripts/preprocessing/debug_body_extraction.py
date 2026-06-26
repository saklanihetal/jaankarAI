# debug_body_extraction.py
import json, random, os
import pandas as pd

CACHE = "retrieval_cache.json"
FEAT = "train_features_v2.csv"

def main():
    if os.path.exists(CACHE):
        with open(CACHE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print("[INFO] cache keys:", len(cache))
        # pick one key with evidence
        keys = [k for k,v in cache.items() if isinstance(v, list) and len(v) > 0]
        print("[INFO] keys with evidence:", len(keys))
        if keys:
            k = random.choice(keys)
            items = cache[k][:5]
            print("\n[DEBUG] sample cache key:", k[:120])
            for i,a in enumerate(items,1):
                print(f"\n--- item {i} ---")
                print("title:", (a.get("title","") or "")[:160])
                print("url:", a.get("url",""))
                d = (a.get("description","") or "")
                print("desc:", d[:200])
    else:
        print("[WARN] retrieval_cache.json not found")

    if os.path.exists(FEAT):
        df = pd.read_csv(FEAT).fillna(0.0)
        print("\n[INFO] train_features_v2 rows:", len(df))
        for col in ["max_ent","max_rel","max_con_used","context_mismatch_rate"]:
            if col in df.columns:
                print(f"[STATS] {col}: min={df[col].min():.4f}  mean={df[col].mean():.4f}  p90={df[col].quantile(0.90):.4f}  max={df[col].max():.4f}")
        if "max_ent" in df.columns:
            print("\n[HIST] max_ent counts > thresholds:")
            for t in [0.01,0.03,0.05,0.08,0.10,0.15,0.20]:
                print(f"  >{t:.2f}: {(df['max_ent']>t).sum()}")
    else:
        print("[WARN] train_features_v2.csv not found")

if __name__ == "__main__":
    main()
