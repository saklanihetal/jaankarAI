# build_retrieval_cache_rss.py
# Build retrieval_cache_rss.json from your dataset using Google News RSS (no API keys).
# Key: md5(normalized English claim)
# Value: list of items {title, description, url, source, lang}

import argparse, json, re, time, unicodedata, hashlib
from typing import Dict, List, Tuple

import pandas as pd
import requests
import feedparser
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect

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

def keyword_set_en(text: str) -> List[str]:
    t = normalize_text(text).lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    toks = [x for x in t.split() if len(x) > 2]
    stop = {"the","and","for","with","from","into","over","near","after","before","will","has","have","had",
            "was","were","are","is","be","this","that","these","those","today","latest","news"}
    toks = [x for x in toks if x not in stop]
    seen, out = set(), []
    for x in toks:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def build_queries(claim_en: str) -> List[str]:
    claim_en = normalize_text(claim_en)
    toks = keyword_set_en(claim_en)[:7]
    qs = []
    if len(toks) >= 5:
        qs.append(" ".join(toks[:5]) + " India")
    if len(toks) >= 3:
        qs.append(" ".join(toks[:3]) + " India")
    qs.append(claim_en + " India")
    # de-dup
    seen, out = set(), []
    for q in qs:
        qn = q.lower().strip()
        if qn and qn not in seen:
            out.append(q); seen.add(qn)
    return out[:3]

# ---------- NLLB translate ----------
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

# ---------- RSS ----------
def fetch_google_news_rss(query: str) -> Tuple[List[dict], str]:
    url = "https://news.google.com/rss/search"
    params = {"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"}
    try:
        r = requests.get(url, params=params, timeout=15, headers=HEADERS)
        if r.status_code != 200:
            return [], f"HTTP {r.status_code}"
        feed = feedparser.parse(r.text)
        items = []
        for e in feed.entries[:30]:
            title = getattr(e, "title", "") or ""
            link = getattr(e, "link", "") or ""
            summary = getattr(e, "summary", "") or ""
            if title and link:
                # strip html tags
                summary = re.sub("<.*?>", " ", summary)
                items.append({
                    "title": title,
                    "description": normalize_text(summary)[:500],
                    "url": link,
                    "source": "GoogleNewsRSS",
                    "lang": "en"
                })
        return items, ""
    except Exception as e:
        return [], str(e)

def dedup_items(items: List[dict]) -> List[dict]:
    seen, out = set(), []
    for a in items:
        t = normalize_text(a.get("title", "")).lower()
        u = (a.get("url","") or "").strip()
        if not t or not u:
            continue
        k = (t, u)
        if k in seen:
            continue
        seen.add(k)
        out.append(a)
    return out

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="retrieval_cache_rss.json")
    ap.add_argument("--max_rows", type=int, default=1500)
    ap.add_argument("--sleep", type=float, default=0.25)
    args = ap.parse_args()

    df = pd.read_csv(args.data).dropna(subset=["headline","language"]).copy()
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
    print(f"[INFO] Total rows sampled: {len(df_s)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Loading NLLB translator...")
    nllb_name = "facebook/nllb-200-distilled-600M"
    nllb_tok = AutoTokenizer.from_pretrained(nllb_name)
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_name).to(device)

    cache: Dict[str, List[dict]] = {}
    rows_with_evidence = 0

    for i, r in df_s.iterrows():
        raw = normalize_text(str(r["headline"]))
        lang = detect_language(raw)
        claim_en = translate_to_english(nllb_tok, nllb_model, device, raw, lang)
        claim_en = normalize_text(claim_en)

        key = make_cache_key(claim_en)
        queries = build_queries(claim_en)

        items: List[dict] = []
        for q in queries:
            a, _ = fetch_google_news_rss(q)
            items += a
            time.sleep(args.sleep)

        items = dedup_items(items)
        cache[key] = items
        if items:
            rows_with_evidence += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(df_s):
            print(f"Processed {i+1}/{len(df_s)} | rows_with_evidence={rows_with_evidence}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    print(f"[DONE] wrote: {args.out}")
    print(f"[DONE] rows_with_evidence: {rows_with_evidence}/{len(df_s)}")

if __name__ == "__main__":
    main()
