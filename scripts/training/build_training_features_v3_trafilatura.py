# build_training_features_v3_trafilatura.py
# Build training features using:
# - retrieval_cache.json (keyed by MD5 hash of normalized claim)
# - LaBSE relevance ranking
# - article body extraction via trafilatura
# - NLI (bart-large-mnli) on body snippet vs claim
#
# Output: train_features_v3.csv

import argparse
import json
import os
import re
import time
import unicodedata
import hashlib
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import trafilatura

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util


HEADERS = {"User-Agent": "FakeNewsFeatureBuilder/3.0"}
SUPPORTED_LANGS = {"english", "hindi", "kannada", "telugu", "english-hindi-mixed"}

# --------------------- text + cache key ---------------------
def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_for_key(s: str) -> str:
    return normalize_text(s).lower()

def make_cache_key(text: str) -> str:
    norm = normalize_for_key(text)
    return hashlib.md5(norm.encode("utf-8")).hexdigest()

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def percentile(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    v = sorted(vals)
    if len(v) == 1:
        return v[0]
    k = (len(v) - 1) * p
    f = int(k)
    c = min(f + 1, len(v) - 1)
    if f == c:
        return v[f]
    return v[f] + (k - f) * (v[c] - v[f])

# --------------------- fetching + extraction ---------------------
def safe_get(url: str, timeout: int = 20) -> str:
    try:
        r = requests.get(url, timeout=timeout, headers=HEADERS)
        if r.status_code != 200:
            return ""
        return r.text
    except Exception:
        return ""

def extract_article_text_from_html(html: str) -> str:
    """
    Robust extraction using trafilatura.
    """
    if not html:
        return ""
    try:
        txt = trafilatura.extract(html, include_comments=False, include_tables=False)
        return normalize_text(txt or "")
    except Exception:
        return ""

def chunk_words(text: str, max_words: int = 240) -> str:
    w = text.split()
    if len(w) <= max_words:
        return text
    return " ".join(w[:max_words])

# --------------------- models ---------------------
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb = SentenceTransformer("sentence-transformers/LaBSE")
    nli_name = "facebook/bart-large-mnli"
    tok = AutoTokenizer.from_pretrained(nli_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(nli_name).to(device)
    return device, emb, tok, mdl

def nli_probs(device, tok, mdl, premise: str, hypothesis: str) -> Tuple[float, float, float]:
    inputs = tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        logits = mdl(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
    p_con, p_neu, p_ent = probs[0], probs[1], probs[2]
    return float(p_ent), float(p_con), float(p_neu)

# --------------------- cache ---------------------
def load_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# --------------------- feature computation ---------------------
def compute_features(
    device, emb, tok, mdl,
    claim_en: str,
    evidence: List[Dict],
    top_k: int = 5,
    sleep_s: float = 0.2,
    fetch_body: bool = True,
) -> Dict[str, float]:

    if not evidence:
        return {
            "num_evidence": 0.0,
            "max_rel": 0.0, "mean_rel": 0.0, "p90_rel": 0.0,
            "max_ent": 0.0, "mean_ent": 0.0,
            "max_con_used": 0.0, "mean_con": 0.0,
            "body_success_rate": 0.0
        }

    # Build blobs for relevance ranking
    blobs = []
    for a in evidence:
        t = normalize_text(a.get("title", ""))
        d = normalize_text(a.get("description", ""))
        blob = (t + ". " + d).strip() if d else t
        blobs.append(blob)

    # Rank via LaBSE
    claim_emb = emb.encode([claim_en], convert_to_tensor=True, normalize_embeddings=True)
    ev_emb = emb.encode(blobs, convert_to_tensor=True, normalize_embeddings=True)
    rels_all = util.cos_sim(claim_emb, ev_emb).squeeze(0).tolist()

    ranked = []
    for a, rel, blob in zip(evidence, rels_all, blobs):
        ranked.append({**a, "relevance": float(rel), "blob": blob})
    ranked.sort(key=lambda x: x["relevance"], reverse=True)

    top = ranked[:top_k]

    # Universal contradiction gating constants
    MIN_REL_FOR_CON = 0.35
    R0, R1 = 0.50, 0.70

    rels, ents, cons = [], [], []
    max_ent, max_con_used = 0.0, 0.0
    body_ok = 0

    for art in top:
        rel = float(art["relevance"])
        rels.append(rel)

        # Premise starts as title+desc blob
        premise = art["blob"]

        # Try to replace premise with extracted body snippet
        if fetch_body:
            url = (art.get("url") or "").strip()
            if url:
                html = safe_get(url, timeout=20)
                body = extract_article_text_from_html(html)
                body = chunk_words(body, 240)
                if len(body.split()) >= 60:
                    premise = body
                    body_ok += 1
                time.sleep(sleep_s)

        ent, con, neu = nli_probs(device, tok, mdl, premise, claim_en)

        support = ent
        weighted_con = con * (1.0 - ent)

        if rel >= MIN_REL_FOR_CON:
            g_rel = clamp01((rel - R0) / (R1 - R0)) if (R1 > R0) else 1.0
            con_used = weighted_con * g_rel
        else:
            con_used = 0.0

        max_ent = max(max_ent, support)
        max_con_used = max(max_con_used, con_used)

        ents.append(support)
        cons.append(con_used)

    return {
        "num_evidence": float(len(evidence)),
        "max_rel": float(max(rels)) if rels else 0.0,
        "mean_rel": float(np.mean(rels)) if rels else 0.0,
        "p90_rel": float(percentile(rels, 0.90)) if rels else 0.0,
        "max_ent": float(max_ent),
        "mean_ent": float(np.mean(ents)) if ents else 0.0,
        "max_con_used": float(max_con_used),
        "mean_con": float(np.mean(cons)) if cons else 0.0,
        "body_success_rate": float(body_ok / max(1, len(top)))
    }

# --------------------- main ---------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset csv with columns headline, language, label")
    ap.add_argument("--cache", default="retrieval_cache.json")
    ap.add_argument("--out", default="train_features_v3.csv")
    ap.add_argument("--max_rows", type=int, default=1500)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--no_body", action="store_true", help="Disable article body extraction (debug)")
    args = ap.parse_args()

    df = pd.read_csv(args.data).dropna(subset=["headline", "label"]).copy()

    # Normalize language buckets
    df["language_norm"] = df["language"].astype(str).str.strip().str.lower()

    langs = sorted(df["language_norm"].unique().tolist())
    per_lang = max(1, args.max_rows // max(1, len(langs)))

    sampled = []
    for L in langs:
        sub = df[df["language_norm"] == L]
        sampled.append(sub.sample(n=min(per_lang, len(sub)), random_state=42))
    df_s = pd.concat(sampled, axis=0).reset_index(drop=True)

    if len(df_s) > args.max_rows:
        df_s = df_s.sample(n=args.max_rows, random_state=42).reset_index(drop=True)

    print(f"[INFO] Languages found: {langs}")
    print(f"[INFO] Rows sampled (per language cap): {per_lang}")
    print(f"[INFO] Total rows to process: {len(df_s)}")

    cache = load_cache(args.cache)
    print(f"[INFO] Cache keys loaded: {len(cache)}")

    device, emb, tok, mdl = load_models()

    rows = []
    hits = 0

    for i, r in df_s.iterrows():
        claim = normalize_text(str(r["headline"]))
        label = int(r["label"])
        lang = str(r["language"]).strip()

        k = make_cache_key(claim)
        evidence = cache.get(k, [])
        if evidence:
            hits += 1

        feats = compute_features(
            device, emb, tok, mdl,
            claim_en=claim,
            evidence=evidence,
            top_k=args.top_k,
            sleep_s=args.sleep,
            fetch_body=(not args.no_body),
        )

        rows.append({
            "headline": claim,
            "language": lang,
            "label": label,
            **feats
        })

        if (i + 1) % 25 == 0 or (i + 1) == len(df_s):
            print(f"Processed {i+1}/{len(df_s)}  | cache_hits={hits}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[DONE] wrote: {args.out}")
    print(f"[DONE] cache_hit_rows: {hits}/{len(df_s)} (should be high; if near 0 cache is mismatched)")

if __name__ == "__main__":
    main()
