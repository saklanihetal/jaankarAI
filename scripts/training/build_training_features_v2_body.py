# build_training_features_v2_body.py
# Rebuild training features using stronger evidence text: ARTICLE BODY extraction.
# Output: train_features_v2.csv + retrieval_cache.json (updated)

import argparse
import json
import os
import re
import time
import unicodedata
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from langdetect import detect

# --------------------- Utilities ---------------------
HEADERS = {"User-Agent": "FakeNewsFeatureBuilder/2.0"}
SUPPORTED_LANGS = {"en", "hi", "kn", "te"}

def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s)
    return s

def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return lang if lang in SUPPORTED_LANGS else "en"
    except Exception:
        return "en"

def keyword_set_en(text: str) -> set:
    text = normalize_text(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return {t for t in text.split() if len(t) > 2}

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def safe_get(url: str, timeout: int = 20) -> str:
    try:
        r = requests.get(url, timeout=timeout, headers=HEADERS)
        if r.status_code != 200:
            return ""
        return r.text
    except Exception:
        return ""

def extract_article_text(html: str) -> str:
    """
    Fast heuristic extraction:
    - remove script/style/nav/footer
    - take paragraphs with enough words
    """
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form", "aside"]):
        tag.decompose()

    paras = []
    for p in soup.find_all("p"):
        txt = normalize_text(p.get_text(" "))
        if len(txt.split()) >= 8:
            paras.append(txt)

    text = " ".join(paras)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def chunk_text(text: str, max_words: int = 220) -> str:
    """
    Keep a reasonably sized snippet for NLI premise.
    """
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])

# --------------------- NLI + Embeddings ---------------------
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

# --------------------- Evidence retrieval cache ---------------------
def load_cache(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(path: str, cache: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

# --------------------- Minimal retrieval (uses EventRegistry only if present in cache) ---------------------
# NOTE: This offline builder assumes you already have retrieval_cache.json from your earlier run.
# If not, it will still work but only with cached items.

def compute_features_for_row(
    device, emb, tok, mdl,
    claim_en: str,
    evidence_items: List[Dict],
    top_k: int = 5,
    fetch_body: bool = True,
    sleep_s: float = 0.2
) -> Dict[str, float]:

    # If no evidence: empty features
    if not evidence_items:
        return {
            "num_evidence": 0.0,
            "max_rel": 0.0, "mean_rel": 0.0, "p90_rel": 0.0,
            "max_ent": 0.0, "mean_ent": 0.0,
            "max_con_used": 0.0, "mean_con": 0.0,
            "context_mismatch_rate": 0.0,
            "update_style_rate": 0.0
        }

    # Build blobs for ranking
    blobs = []
    for a in evidence_items:
        t = normalize_text(a.get("title",""))
        d = normalize_text(a.get("description",""))
        blob = (t + ". " + d).strip() if d else t
        blobs.append(blob)

    claim_emb = emb.encode([claim_en], convert_to_tensor=True, normalize_embeddings=True)
    ev_emb = emb.encode(blobs, convert_to_tensor=True, normalize_embeddings=True)
    rels_all = util.cos_sim(claim_emb, ev_emb).squeeze(0).tolist()

    ranked = []
    for a, rel, blob in zip(evidence_items, rels_all, blobs):
        ranked.append({**a, "relevance": float(rel), "blob": blob})
    ranked.sort(key=lambda x: x["relevance"], reverse=True)
    top = ranked[:top_k]

    # Universal contradiction gating
    MIN_REL_FOR_CON = 0.35
    R0, R1 = 0.50, 0.70

    rels, ents, cons = [], [], []
    max_ent, max_con_used = 0.0, 0.0

    ctx_mismatch = 0
    upd_style = 0

    for art in top:
        rel = float(art["relevance"])
        rels.append(rel)

        premise = art["blob"]

        # Fetch and use article body snippet if possible
        if fetch_body:
            url = (art.get("url") or "").strip()
            if url:
                html = safe_get(url, timeout=18)
                body = extract_article_text(html)
                body = chunk_text(body, 220)
                if len(body.split()) >= 40:
                    premise = body
                time.sleep(sleep_s)

        # NLI
        ent, con, neu = nli_probs(device, tok, mdl, premise, claim_en)

        # context mismatch (simple qualifier check)
        qualifiers = {"replica","model","toy","miniature","lookalike","imitation","reproduction"}
        q_in_ev = qualifiers & keyword_set_en(premise)
        q_in_claim = qualifiers & keyword_set_en(claim_en)
        mult = 1.0
        if q_in_ev and not q_in_claim:
            mult = 0.0
            ctx_mismatch += 1

        support = ent * mult

        # update-style (weak contradiction)
        if re.search(r"\b(toll|death toll|count|number)\b", premise.lower()):
            upd_style += 1
            con = con * 0.15

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

    rels_sorted = sorted(rels)
    p90 = rels_sorted[int(0.9 * (len(rels_sorted)-1))] if len(rels_sorted) >= 2 else (rels_sorted[0] if rels_sorted else 0.0)

    out = {
        "num_evidence": float(len(evidence_items)),
        "max_rel": float(max(rels)) if rels else 0.0,
        "mean_rel": float(np.mean(rels)) if rels else 0.0,
        "p90_rel": float(p90),
        "max_ent": float(max_ent),
        "mean_ent": float(np.mean(ents)) if ents else 0.0,
        "max_con_used": float(max_con_used),
        "mean_con": float(np.mean(cons)) if cons else 0.0,
        "context_mismatch_rate": float(ctx_mismatch / max(1, len(top))),
        "update_style_rate": float(upd_style / max(1, len(top))),
    }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Your labeled dataset CSV")
    ap.add_argument("--cache", default="retrieval_cache.json", help="Existing retrieval cache path")
    ap.add_argument("--out", default="train_features_v2.csv")
    ap.add_argument("--max_rows", type=int, default=1500)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--no_body", action="store_true", help="Disable article body fetch (debug)")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df = df.dropna(subset=["headline","label"]).copy()

    # Sample evenly by language (cap)
    # language column examples: English, Hindi, Kannada, Telugu, English-Hindi-mixed
    langs = sorted(df["language"].astype(str).unique().tolist())
    per_lang = max(1, args.max_rows // max(1, len(langs)))

    sampled = []
    for L in langs:
        sub = df[df["language"].astype(str) == L]
        sampled.append(sub.sample(n=min(per_lang, len(sub)), random_state=42))
    df_s = pd.concat(sampled, axis=0).reset_index(drop=True)
    if len(df_s) > args.max_rows:
        df_s = df_s.sample(n=args.max_rows, random_state=42).reset_index(drop=True)

    print(f"[INFO] Languages found: {langs}")
    print(f"[INFO] Rows sampled (per language cap): {per_lang}")
    print(f"[INFO] Total rows to process: {len(df_s)}")

    cache = load_cache(args.cache)

    device, emb, tok, mdl = load_models()

    rows = []
    for i, r in df_s.iterrows():
        headline = normalize_text(str(r["headline"]))
        label = int(r["label"])
        lang = str(r.get("language","")).lower()
        lang_detected = detect_language(headline)

        # Use cached retrieval by headline if present
        key = headline.lower()
        evidence = cache.get(key, [])

        feats = compute_features_for_row(
            device, emb, tok, mdl,
            claim_en=headline if lang_detected == "en" else headline,  # (we assume your cache was built already in English-claim mode)
            evidence_items=evidence,
            top_k=args.top_k,
            fetch_body=(not args.no_body),
            sleep_s=args.sleep
        )

        out_row = {
            "headline": headline,
            "label": label,
            "language": r.get("language",""),
            **feats
        }
        rows.append(out_row)

        if (i+1) % 25 == 0 or (i+1) == len(df_s):
            print(f"Processed {i+1}/{len(df_s)}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[DONE] wrote: {args.out}")

if __name__ == "__main__":
    main()

