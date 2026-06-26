# build_training_features_rss_v3.py
# Uses retrieval_cache_rss.json (built from Google News RSS)
# Builds train_features_rss_v3.csv using:
# - LaBSE relevance ranking on (title+description)
# - TOP-K NLI using (optional) extracted article body via trafilatura when possible
# Output features are non-zero even without APIs.

import argparse, json, os, re, time, unicodedata, hashlib
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import trafilatura
import torch

from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect
from transformers import AutoTokenizer as T2, AutoModelForSeq2SeqLM

HEADERS = {"User-Agent": "Mozilla/5.0"}
SUPPORTED_DETECT = {"en", "hi", "kn", "te"}
LANG_MAP = {"hi": "hin_Deva", "kn": "kan_Knda", "te": "tel_Telu"}

# ---------- text ----------
def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s)
    return s

def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return lang if lang in SUPPORTED_DETECT else "en"
    except Exception:
        return "en"

def make_cache_key(text_en: str) -> str:
    norm = normalize_text(text_en).lower()
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

# ---------- translate ----------
@torch.inference_mode()
def translate_to_english(nllb_tok, nllb_model, device, text: str, lang: str) -> str:
    text = normalize_text(text)
    if lang == "en":
        return text
    if lang not in LANG_MAP:
        return text
    inputs = nllb_tok(text, return_tensors="pt").to(device)
    eng_id = nllb_tok.convert_tokens_to_ids("eng_Latn")
    out = nllb_model.generate(**inputs, forced_bos_token_id=eng_id, max_length=96)
    return nllb_tok.decode(out[0], skip_special_tokens=True)

# ---------- NLI ----------
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb = SentenceTransformer("sentence-transformers/LaBSE")
    nli_name = "facebook/bart-large-mnli"
    tok = AutoTokenizer.from_pretrained(nli_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(nli_name).to(device)

    nllb_name = "facebook/nllb-200-distilled-600M"
    nllb_tok = T2.from_pretrained(nllb_name)
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_name).to(device)
    return device, emb, tok, mdl, nllb_tok, nllb_model

@torch.inference_mode()
def nli_probs(device, tok, mdl, premise: str, hypothesis: str) -> Tuple[float, float, float]:
    inputs = tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256).to(device)
    logits = mdl(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
    p_con, p_neu, p_ent = probs[0], probs[1], probs[2]
    return float(p_ent), float(p_con), float(p_neu)

# ---------- url + body extraction ----------
def fetch_html(url: str, timeout: int = 18) -> str:
    try:
        r = requests.get(url, timeout=timeout, headers=HEADERS, allow_redirects=True)
        if r.status_code != 200:
            return ""
        return r.text
    except Exception:
        return ""

def extract_body(html: str) -> str:
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

# ---------- feature compute ----------
def compute_features(device, emb, tok, mdl, claim_en: str, evidence: List[Dict], top_k: int, sleep: float):
    if not evidence:
        return {
            "num_evidence": 0.0,
            "max_rel": 0.0, "mean_rel": 0.0, "p90_rel": 0.0,
            "max_ent": 0.0, "mean_ent": 0.0,
            "max_con_used": 0.0, "mean_con": 0.0,
            "body_success_rate": 0.0
        }

    blobs = []
    for a in evidence:
        t = normalize_text(a.get("title",""))
        d = normalize_text(a.get("description",""))
        blob = (t + ". " + d).strip() if d else t
        blobs.append(blob)

    claim_emb = emb.encode([claim_en], convert_to_tensor=True, normalize_embeddings=True)
    ev_emb = emb.encode(blobs, convert_to_tensor=True, normalize_embeddings=True)
    rels_all = util.cos_sim(claim_emb, ev_emb).squeeze(0).tolist()

    ranked = []
    for a, rel, blob in zip(evidence, rels_all, blobs):
        ranked.append({**a, "relevance": float(rel), "blob": blob})
    ranked.sort(key=lambda x: x["relevance"], reverse=True)
    top = ranked[:top_k]

    # fixed gating
    MIN_REL_FOR_CON = 0.35
    R0, R1 = 0.50, 0.70

    rels, ents, cons = [], [], []
    max_ent, max_con_used = 0.0, 0.0
    body_ok = 0

    for it in top:
        rel = float(it["relevance"])
        rels.append(rel)

        # base premise = title+desc
        premise = it["blob"]

        # try body (best effort)
        url = (it.get("url") or "").strip()
        if url:
            html = fetch_html(url, timeout=18)
            body = extract_body(html)
            body = chunk_words(body, 240)
            if len(body.split()) >= 60:
                premise = body
                body_ok += 1
            time.sleep(sleep)

        ent, con, neu = nli_probs(device, tok, mdl, premise, claim_en)

        weighted_con = con * (1.0 - ent)
        if rel >= MIN_REL_FOR_CON and R1 > R0:
            g_rel = clamp01((rel - R0) / (R1 - R0))
            con_used = weighted_con * g_rel
        else:
            con_used = 0.0

        max_ent = max(max_ent, ent)
        max_con_used = max(max_con_used, con_used)
        ents.append(ent)
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
        "body_success_rate": float(body_ok / max(1, len(top))),
    }

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cache", default="retrieval_cache_rss.json")
    ap.add_argument("--out", default="train_features_rss_v3.csv")
    ap.add_argument("--max_rows", type=int, default=1500)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    df = pd.read_csv(args.data).dropna(subset=["headline","label"]).copy()
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

    print(f"[INFO] Languages: {langs}")
    print(f"[INFO] Total rows: {len(df_s)}")

    if not os.path.exists(args.cache):
        raise FileNotFoundError(f"Cache not found: {args.cache}")

    with open(args.cache, "r", encoding="utf-8") as f:
        cache = json.load(f)
    print(f"[INFO] Cache keys: {len(cache)}")

    device, emb, tok, mdl, nllb_tok, nllb_model = load_models()

    rows = []
    hits = 0

    for i, r in df_s.iterrows():
        raw = normalize_text(str(r["headline"]))
        label = int(r["label"])
        lang = detect_language(raw)

        claim_en = translate_to_english(nllb_tok, nllb_model, device, raw, lang)
        claim_en = normalize_text(claim_en)

        key = make_cache_key(claim_en)
        evidence = cache.get(key, [])
        if evidence:
            hits += 1

        feats = compute_features(device, emb, tok, mdl, claim_en, evidence, top_k=args.top_k, sleep=args.sleep)

        rows.append({
            "headline": claim_en,
            "language": str(r["language"]),
            "label": label,
            **feats
        })

        if (i + 1) % 25 == 0 or (i + 1) == len(df_s):
            print(f"Processed {i+1}/{len(df_s)} | cache_hit_rows={hits}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[DONE] wrote: {args.out}")
    print(f"[DONE] cache_hit_rows: {hits}/{len(df_s)}")

if __name__ == "__main__":
    main()
