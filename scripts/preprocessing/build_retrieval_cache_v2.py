# build_retrieval_cache_v2.py
# Build retrieval_cache.json from your labeled dataset.
# Keys: md5(normalized English claim)
# Values: list of evidence dicts: {title, description, url, source, lang}

import argparse, json, os, re, time, unicodedata, hashlib
from typing import Dict, List, Tuple

import pandas as pd
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect

# ----------- text + key -----------
def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_for_key(s: str) -> str:
    return normalize_text(s).lower()

def make_cache_key(text_en: str) -> str:
    norm = normalize_for_key(text_en)
    return hashlib.md5(norm.encode("utf-8")).hexdigest()

SUPPORTED_DETECT = {"en", "hi", "kn", "te"}

def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return lang if lang in SUPPORTED_DETECT else "en"
    except Exception:
        return "en"

# ----------- translation (NLLB) -----------
LANG_MAP = {"hi": "hin_Deva", "kn": "kan_Knda", "te": "tel_Telu"}
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

# ----------- query builder -----------
def keyword_set(text: str) -> List[str]:
    t = normalize_text(text).lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    toks = [x for x in t.split() if len(x) > 2]
    # lightweight stopwords
    stop = {"the","and","for","with","from","into","over","near","after","before","will","has","have","had","was","were","are","is","be"}
    toks = [x for x in toks if x not in stop]
    # dedup keep order
    seen, out = set(), []
    for x in toks:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def build_queries(claim_en: str) -> List[str]:
    claim_en = normalize_text(claim_en)
    toks = keyword_set(claim_en)
    qs = []
    if len(toks) >= 6:
        qs.append(" ".join(toks[:6]))
    if len(toks) >= 3:
        qs.append(" ".join(toks[:3]))
    qs.append(claim_en)
    # dedup
    seen, out = set(), []
    for q in qs:
        qn = q.lower().strip()
        if qn and qn not in seen:
            out.append(q); seen.add(qn)
    return out[:3]

# ----------- API helpers -----------
def safe_get_json(url: str, timeout: int = 15) -> dict:
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return {}
        return r.json()
    except Exception:
        return {}

def safe_post_json(url: str, payload: dict, timeout: int = 20) -> dict:
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        if r.status_code != 200:
            return {}
        return r.json()
    except Exception:
        return {}

def dedup_items(items: List[dict]) -> List[dict]:
    seen, out = set(), []
    for a in items:
        t = normalize_text(a.get("title","")).lower()
        u = (a.get("url","") or "").strip()
        if not t or not u:
            continue
        key = (t, u)
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out

# ----------- fetchers -----------
def fetch_newsapi_org(q: str, api_key: str) -> List[dict]:
    if not api_key:
        return []
    url = "https://newsapi.org/v2/everything"
    params = f"?q={requests.utils.quote(q)}&language=en&pageSize=15&sortBy=publishedAt&apiKey={api_key}"
    res = safe_get_json(url + params)
    out = []
    for a in res.get("articles", []) or []:
        out.append({
            "title": a.get("title",""),
            "description": a.get("description",""),
            "url": a.get("url",""),
            "source": "NewsAPI.org",
            "lang": "en"
        })
    return out

def fetch_gnews(q: str, api_key: str) -> List[dict]:
    if not api_key:
        return []
    url = f"https://gnews.io/api/v4/search?q={requests.utils.quote(q)}&lang=en&max=15&token={api_key}"
    res = safe_get_json(url)
    out = []
    for a in res.get("articles", []) or []:
        out.append({
            "title": a.get("title",""),
            "description": a.get("description",""),
            "url": a.get("url",""),
            "source": "GNews",
            "lang": "en"
        })
    return out

def fetch_newsdata(q: str, api_key: str) -> List[dict]:
    if not api_key:
        return []
    url = f"https://newsdata.io/api/1/news?q={requests.utils.quote(q)}&language=en&apikey={api_key}"
    res = safe_get_json(url)
    out = []
    for a in res.get("results", []) or []:
        out.append({
            "title": a.get("title",""),
            "description": a.get("description",""),
            "url": a.get("link",""),
            "source": "NewsData.io",
            "lang": "en"
        })
    return out

def fetch_eventregistry(q: str, api_key: str) -> List[dict]:
    if not api_key:
        return []
    url = "https://eventregistry.org/api/v1/article/getArticles"
    payload = {
        "action": "getArticles",
        "keyword": q,
        "lang": "eng",
        "articlesCount": 15,
        "apiKey": api_key
    }
    res = safe_post_json(url, payload)
    out = []
    for a in (((res.get("articles") or {}).get("results")) or []):
        out.append({
            "title": a.get("title",""),
            "description": a.get("body","") or a.get("summary","") or "",
            "url": a.get("url",""),
            "source": "EventRegistry",
            "lang": "en"
        })
    return out

# ----------- main -----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="retrieval_cache.json")
    ap.add_argument("--max_rows", type=int, default=500)
    ap.add_argument("--sleep", type=float, default=0.35)

    # keys (prefer env vars; you can hardcode temporarily)
    ap.add_argument("--newsapi_key", default=os.getenv("NEWSAPI_KEY",""))
    ap.add_argument("--gnews_key", default=os.getenv("GNEWS_KEY",""))
    ap.add_argument("--newsdata_key", default=os.getenv("NEWSDATA_KEY",""))
    ap.add_argument("--eventregistry_key", default=os.getenv("EVENTREGISTRY_KEY",""))

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

    # Load translator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Loading NLLB translator...")
    nllb_name = "facebook/nllb-200-distilled-600M"
    nllb_tok = AutoTokenizer.from_pretrained(nllb_name)
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_name).to(device)

    cache: Dict[str, List[dict]] = {}
    hits = 0

    for i, r in df_s.iterrows():
        raw = normalize_text(str(r["headline"]))
        lang_detected = detect_language(raw)

        claim_en = translate_to_english(nllb_tok, nllb_model, device, raw, lang_detected)
        claim_en = normalize_text(claim_en)
        key = make_cache_key(claim_en)

        queries = build_queries(claim_en)

        items: List[dict] = []
        for q in queries:
            items += fetch_eventregistry(q, args.eventregistry_key)
            items += fetch_newsapi_org(q, args.newsapi_key)
            items += fetch_gnews(q, args.gnews_key)
            items += fetch_newsdata(q, args.newsdata_key)
            time.sleep(args.sleep)

        items = dedup_items(items)
        cache[key] = items

        if items:
            hits += 1

        if (i + 1) % 25 == 0 or (i + 1) == len(df_s):
            print(f"Processed {i+1}/{len(df_s)} | rows_with_evidence={hits}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    print(f"[DONE] wrote: {args.out}")
    print(f"[DONE] rows_with_evidence: {hits}/{len(df_s)}")

if __name__ == "__main__":
    main()
