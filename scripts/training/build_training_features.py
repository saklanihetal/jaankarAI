# build_training_features.py
# Build training features by running your retrieval+NLI pipeline offline.
# Output: train_features.csv
#
# IMPORTANT CHANGE:
#   --max_rows now means "max rows PER LANGUAGE", not total.
#   Example: --max_rows 300 => up to 300 rows each for en/hi/kn/te (total up to 1200)
#
# Usage (recommended small first):
#   python build_training_features.py --data fake_news_simplified_LANGFILTER_labels01.csv --cache lexicon_out/translated_cache.csv --out train_features.csv --max_rows 100
#
# Then scale:
#   python build_training_features.py --data fake_news_simplified_LANGFILTER_labels01.csv --cache lexicon_out/translated_cache.csv --out train_features.csv --max_rows 300
#
# Notes:
# - This DOES call News APIs. Start small to avoid rate limits.
# - Uses translated_cache.csv to avoid translating again.
# - Writes retrieval cache so re-runs get faster.

import os, re, json, time, hashlib, unicodedata
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import toml
from langdetect import detect

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util

# ----------------- Helpers -----------------
SUPPORTED_LANGS = {"en", "hi", "kn", "te"}

def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s)
    return s

def keyword_set_en(text: str) -> set:
    """
    Simple English keyword set used for context mismatch checks.
    """
    text = normalize_text(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return {t for t in text.split() if len(t) > 2}

def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return lang if lang in SUPPORTED_LANGS else "en"
    except Exception:
        return "en"

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

def read_streamlit_secrets() -> Dict[str, str]:
    path = os.path.join(".streamlit", "secrets.toml")
    if not os.path.exists(path):
        return {}
    try:
        data = toml.load(path)
        out = {}
        for k, v in data.items():
            if isinstance(v, str):
                out[k] = v
        return out
    except Exception:
        return {}

def get_key(name: str, secrets: Dict[str, str]) -> str:
    if name in secrets and secrets[name]:
        return str(secrets[name])
    return os.getenv(name, "")

def clean_desc(s: str) -> str:
    s = normalize_text(s or "")
    s = re.sub(r"\[\+\d+\s+chars\]$", "", s).strip()
    return s

def dedup_articles(articles: List[Dict]) -> List[Dict]:
    seen, out = set(), []
    for a in articles:
        t = normalize_text(a.get("title","")).lower()
        u = (a.get("url") or "").strip().lower()
        if not t:
            continue
        key = (t, u)
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out

# ----------------- APIs -----------------
HEADERS = {"User-Agent": "FakeNewsTrainFeatures/1.0"}

def safe_get_json(url: str, timeout: int = 18) -> dict:
    try:
        r = requests.get(url, timeout=timeout, headers=HEADERS)
        if r.status_code != 200:
            return {}
        return r.json()
    except Exception:
        return {}

def safe_post_json(url: str, payload: dict, timeout: int = 22) -> dict:
    try:
        r = requests.post(url, json=payload, timeout=timeout, headers=HEADERS)
        if r.status_code != 200:
            return {}
        return r.json()
    except Exception:
        return {}

GNEWS_LANG = {"en":"en","hi":"hi","kn":"kn","te":"te"}
NEWSDATA_LANG = {"en":"en","hi":"hi","kn":"kn","te":"te"}
EVENT_LANG = {"en":"eng","hi":"hin","kn":"kan","te":"tel"}

def fetch_newsapi_org(query: str, NEWSAPI_KEY: str) -> List[Dict]:
    if not NEWSAPI_KEY:
        return []
    url = (
        "https://newsapi.org/v2/everything?"
        f"q={requests.utils.quote(query)}&language=en&pageSize=20&sortBy=publishedAt&apiKey={NEWSAPI_KEY}"
    )
    res = safe_get_json(url)
    out = []
    for a in (res.get("articles", []) or []):
        t = a.get("title") or ""
        u = a.get("url") or ""
        d = clean_desc(a.get("description") or a.get("content") or "")
        if t and u:
            out.append({"title": t, "description": d, "url": u, "api":"NewsAPI.org"})
    return out

def fetch_gnews(query: str, lang: str, GNEWS_KEY: str) -> List[Dict]:
    if not GNEWS_KEY:
        return []
    gl = GNEWS_LANG.get(lang, "en")
    url = f"https://gnews.io/api/v4/search?q={requests.utils.quote(query)}&lang={gl}&max=20&token={GNEWS_KEY}"
    res = safe_get_json(url)
    out = []
    for a in (res.get("articles", []) or []):
        t = a.get("title") or ""
        u = a.get("url") or ""
        d = clean_desc(a.get("description") or a.get("content") or "")
        if t and u:
            out.append({"title": t, "description": d, "url": u, "api": f"GNews({gl})"})
    return out

def fetch_newsdata(query: str, lang: str, NEWSDATA_KEY: str) -> List[Dict]:
    if not NEWSDATA_KEY:
        return []
    nl = NEWSDATA_LANG.get(lang, "en")
    url = f"https://newsdata.io/api/1/news?q={requests.utils.quote(query)}&language={nl}&apikey={NEWSDATA_KEY}"
    res = safe_get_json(url)
    out = []
    for a in (res.get("results", []) or []):
        t = a.get("title") or ""
        u = a.get("link") or ""
        d = clean_desc(a.get("description") or a.get("content") or "")
        if t and u:
            out.append({"title": t, "description": d, "url": u, "api": f"NewsData.io({nl})"})
    return out

def fetch_eventregistry(query: str, lang: str, EVENTREGISTRY_KEY: str) -> List[Dict]:
    if not EVENTREGISTRY_KEY:
        return []
    el = EVENT_LANG.get(lang, "eng")
    url = "https://eventregistry.org/api/v1/article/getArticles"
    payload = {
        "action": "getArticles",
        "keyword": query,
        "lang": el,
        "articlesPage": 1,
        "articlesCount": 20,
        "articlesSortBy": "date",
        "articlesSortByAsc": False,
        "resultType": "articles",
        "apiKey": EVENTREGISTRY_KEY,
    }
    res = safe_post_json(url, payload)
    results = (((res.get("articles") or {}).get("results")) or [])
    out = []
    for a in results:
        t = a.get("title") or ""
        u = a.get("url") or ""
        d = clean_desc(a.get("summary") or a.get("body") or a.get("snippet") or "")
        if t and u:
            out.append({"title": t, "description": d, "url": u, "api": f"EventRegistry({el})"})
    return out

# ----------------- Query builder -----------------
ALIASES = [
    (r"\bmysuru\b", "mysore"),
    (r"\bmysore\b", "mysuru"),
    (r"\bbengaluru\b", "bangalore"),
    (r"\bbangalore\b", "bengaluru"),
]

def apply_aliases(q: str) -> List[str]:
    outs = [q]
    for pat, rep in ALIASES:
        outs = [re.sub(pat, rep, x, flags=re.I) for x in outs]
    seen, out = set(), []
    for x in outs:
        k = normalize_text(x).lower()
        if k and k not in seen:
            out.append(normalize_text(x))
            seen.add(k)
    return out

def build_queries(claim_en: str, claim_native: str, lang: str) -> Tuple[List[str], List[str]]:
    claim_en = normalize_text(claim_en)
    q_en = [claim_en, " ".join(claim_en.split()[:12])]
    q_en2 = []
    for q in q_en:
        q_en2.extend(apply_aliases(q))
    q_en2 = list(dict.fromkeys([x for x in q_en2 if len(x) >= 6]))[:6]

    q_native = []
    if lang != "en":
        cn = normalize_text(claim_native)
        if cn:
            q_native = [cn, " ".join(cn.split()[:8])]
            q_native = list(dict.fromkeys([x for x in q_native if len(x) >= 6]))[:2]

    return q_en2, q_native

# ----------------- Models -----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
labse = SentenceTransformer("sentence-transformers/LaBSE")

nli_name = "facebook/bart-large-mnli"
nli_tok = AutoTokenizer.from_pretrained(nli_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_name).to(DEVICE)

def nli_probs(premise: str, hypothesis: str) -> Tuple[float,float,float]:
    inputs = nli_tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
    p_con, p_neu, p_ent = probs[0], probs[1], probs[2]
    return float(p_ent), float(p_con), float(p_neu)

PROMO_PATTERNS_EN = [r"\bwhere\b.*$", r"\bhow\b.*$", r"\bfree\b.*$", r"\bfor free\b.*$"]

def core_claim(text_en: str) -> str:
    t = normalize_text(text_en).split("?")[0].split(";")[0].strip()
    low = t.lower()
    for pat in PROMO_PATTERNS_EN:
        low2 = re.sub(pat, "", low).strip()
        if len(low2) >= 6:
            low = low2
    low = re.sub(r"[,\-:;]+", " ", low)
    low = re.sub(r"\s+", " ", low).strip()
    return low if len(low.split()) >= 3 else normalize_text(text_en)

def event_core_claim(text_en: str) -> str:
    t = normalize_text(text_en).lower().split("?")[0].split(";")[0].strip()
    t = re.sub(r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\b", " ", t)
    t = re.sub(r"\b\d+\b", " ", t)
    t = re.sub(r"\b(many|several|multiple|numerous|serious|seriously|severe|severely)\b", " ", t)
    t = re.sub(r"\b(killed|dead|deaths|death|injured|injuries)\b", " ", t)
    t = re.sub(r"[,\-:;]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t if len(t.split()) >= 4 else normalize_text(text_en)

def claim_hypotheses(claim_en: str) -> List[str]:
    hs = [normalize_text(claim_en), core_claim(claim_en), event_core_claim(claim_en)]
    out, seen = [], set()
    for h in hs:
        k = h.lower().strip()
        if k and k not in seen and len(k.split()) >= 3:
            out.append(h)
            seen.add(k)
    return out[:3]

CONTEXT_QUALIFIERS = {
    "replica","model","toy","miniature","lookalike","imitation","reproduction",
    "parody","satire","prank","rumor","alleged","reportedly","unconfirmed"
}

def context_mismatch_multiplier(claim_en: str, blob_en: str) -> float:
    c = normalize_text(claim_en).lower()
    b = normalize_text(blob_en).lower()
    c_tokens = keyword_set_en(c)
    b_tokens = keyword_set_en(b)
    q = (b_tokens & CONTEXT_QUALIFIERS)
    if q and not (c_tokens & q):
        return 0.0
    return 1.0

def looks_like_update_style(claim_en: str, blob_any: str) -> bool:
    c = normalize_text(claim_en).lower()
    b = normalize_text(blob_any).lower()
    if not re.search(r"\b(one|two|three|four|five|\d+)\b", c):
        return False
    en = bool(re.search(r"\b(toll|count|number)\b.*\b(rises|rise|reaches|hits|climbs|mounts|increases|increased)\b", b))
    kn = (("ಸಂಖ್ಯೆ" in b or "ಸಾವಿನ ಸಂಖ್ಯೆ" in b) and ("ಏರಿಕೆ" in b or "ಏರಿಕೆಯಾಗಿದೆ" in b or "ಏರಿತು" in b))
    te = (("సంఖ్య" in b or "మృతుల సంఖ్య" in b) and ("పెరిగ" in b or "ఎక్కువ" in b or "పెరిగింది" in b))
    hi = (("संख्या" in b) and ("बढ़" in b or "बढ़" in b or "पहुंच" in b))
    return en or kn or te or hi

def rank_by_relevance(evidence: List[Dict], claim_en: str) -> List[Dict]:
    blobs = []
    for a in evidence:
        t = normalize_text(a.get("title",""))
        d = normalize_text(a.get("description",""))
        blob = (t + ". " + d).strip() if d else t
        a["blob"] = blob
        blobs.append(blob)

    claim_emb = labse.encode([claim_en], convert_to_tensor=True, normalize_embeddings=True)
    ev_emb = labse.encode(blobs, convert_to_tensor=True, normalize_embeddings=True)
    rels = util.cos_sim(claim_emb, ev_emb).squeeze(0).tolist()

    ranked = []
    for art, rel in zip(evidence, rels):
        ranked.append({**art, "relevance": float(rel)})
    ranked.sort(key=lambda x: x["relevance"], reverse=True)
    return ranked

def compute_features_for_claim(
    claim_native: str,
    claim_en: str,
    lang: str,
    keys: Dict[str,str],
    top_k: int = 5,
    max_total: int = 120,
    request_sleep: float = 0.2,
    retrieval_cache: Optional[dict] = None,
) -> Tuple[Dict, dict]:

    NEWSAPI_KEY = keys.get("NEWSAPI_KEY","")
    GNEWS_KEY = keys.get("GNEWS_KEY","")
    NEWSDATA_KEY = keys.get("NEWSDATA_KEY","")
    EVENTREGISTRY_KEY = keys.get("EVENTREGISTRY_KEY","")

    q_en, q_nat = build_queries(claim_en, claim_native, lang)

    cache_key = hashlib.md5(("||".join(q_en) + "||" + "||".join(q_nat) + f"||{lang}").encode("utf-8")).hexdigest()
    if retrieval_cache is not None and cache_key in retrieval_cache:
        evidence = retrieval_cache[cache_key]
    else:
        evidence = []
        for q in q_en:
            evidence += fetch_newsapi_org(q, NEWSAPI_KEY)
            evidence += fetch_gnews(q, "en", GNEWS_KEY)
            evidence += fetch_newsdata(q, "en", NEWSDATA_KEY)
            evidence += fetch_eventregistry(q, "en", EVENTREGISTRY_KEY)
            time.sleep(request_sleep)

        if lang != "en":
            for q in q_nat:
                evidence += fetch_gnews(q, lang, GNEWS_KEY)
                evidence += fetch_newsdata(q, lang, NEWSDATA_KEY)
                evidence += fetch_eventregistry(q, lang, EVENTREGISTRY_KEY)
                time.sleep(request_sleep)

        evidence = dedup_articles(evidence)[:max_total]
        if retrieval_cache is not None:
            retrieval_cache[cache_key] = evidence

    if not evidence:
        feats = {
            "num_evidence": 0,
            "max_rel": 0.0, "mean_rel": 0.0, "p90_rel": 0.0,
            "max_ent": 0.0, "max_con_used": 0.0,
            "mean_ent": 0.0, "mean_con": 0.0,
            "context_mismatch_rate": 0.0,
            "update_style_rate": 0.0,
        }
        return feats, retrieval_cache

    ranked = rank_by_relevance(evidence, claim_en)
    top = ranked[:top_k]

    MIN_REL_FOR_CON = 0.35
    R0, R1 = 0.50, 0.70
    UPDATE_CON_DOWNWEIGHT = 0.15

    hyps = claim_hypotheses(claim_en)

    ents, cons, rels = [], [], []
    ctx_mismatch_flags = []
    update_flags = []
    max_ent, max_con_used = 0.0, 0.0

    for art in top:
        blob = normalize_text(art.get("blob",""))
        rel = float(art.get("relevance", 0.0))
        rels.append(rel)

        best_ent, best_con, best_neu = 0.0, 0.0, 1.0
        for h in hyps:
            e, c, n = nli_probs(blob, h)
            if e > best_ent:
                best_ent, best_con, best_neu = e, c, n

        mult = context_mismatch_multiplier(claim_en, blob)
        ctx_mismatch_flags.append(1.0 if mult < 1.0 else 0.0)
        support = best_ent * mult

        weighted_con = best_con * (1.0 - best_ent)
        upd = looks_like_update_style(claim_en, blob)
        update_flags.append(1.0 if upd else 0.0)
        if upd:
            weighted_con *= UPDATE_CON_DOWNWEIGHT

        if rel >= MIN_REL_FOR_CON:
            g_rel = clamp01((rel - R0) / (R1 - R0)) if (R1 > R0) else 1.0
            con_used = weighted_con * g_rel
        else:
            con_used = 0.0

        ents.append(support)
        cons.append(con_used)

        max_ent = max(max_ent, support)
        max_con_used = max(max_con_used, con_used)

    feats = {
        "num_evidence": int(len(evidence)),
        "max_rel": float(max(rels)) if rels else 0.0,
        "mean_rel": float(np.mean(rels)) if rels else 0.0,
        "p90_rel": float(percentile(rels, 0.90)) if rels else 0.0,
        "max_ent": float(max_ent),
        "max_con_used": float(max_con_used),
        "mean_ent": float(np.mean(ents)) if ents else 0.0,
        "mean_con": float(np.mean(cons)) if cons else 0.0,
        "context_mismatch_rate": float(np.mean(ctx_mismatch_flags)) if ctx_mismatch_flags else 0.0,
        "update_style_rate": float(np.mean(update_flags)) if update_flags else 0.0,
    }
    return feats, retrieval_cache

# ----------------- Merge translations -----------------
def find_translation_columns(df: pd.DataFrame) -> Tuple[str,str]:
    cols = [c.lower() for c in df.columns]
    orig_candidates = ["headline", "text", "claim", "original", "headline_original"]
    en_candidates = ["headline_en", "translated_en", "translation_en", "english", "en", "claim_en"]

    orig_col = None
    en_col = None
    for c in df.columns:
        if c.lower() in orig_candidates:
            orig_col = c
            break
    for c in df.columns:
        if c.lower() in en_candidates:
            en_col = c
            break

    if orig_col is None:
        orig_col = df.columns[0]
    if en_col is None:
        for c in df.columns:
            if "en" in c.lower():
                en_col = c
                break
        if en_col is None and len(df.columns) >= 2:
            en_col = df.columns[1]
        if en_col is None:
            raise ValueError("Could not find an English translation column in translated_cache.csv")

    return orig_col, en_col

# ----------------- Main -----------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Labeled dataset CSV with columns headline, language, label")
    ap.add_argument("--cache", required=True, help="translated_cache.csv path")
    ap.add_argument("--out", default="train_features.csv")
    ap.add_argument("--max_rows", type=int, default=100, help="Max rows PER language")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--retrieval_cache_out", default="retrieval_cache.json")
    args = ap.parse_args()

    secrets = read_streamlit_secrets()
    keys = {
        "NEWSAPI_KEY": get_key("NEWSAPI_KEY", secrets),
        "GNEWS_KEY": get_key("GNEWS_KEY", secrets),
        "NEWSDATA_KEY": get_key("NEWSDATA_KEY", secrets),
        "EVENTREGISTRY_KEY": get_key("EVENTREGISTRY_KEY", secrets),
    }

    df = pd.read_csv(args.data)
    need_cols = {"headline","language","label"}
    if not need_cols.issubset({c.lower() for c in df.columns}):
        print("Your labeled CSV must contain columns: headline, language, label")
        print("Found:", df.columns.tolist())
        return

    def col_by_lower(name: str) -> str:
        for c in df.columns:
            if c.lower() == name:
                return c
        raise KeyError(name)

    hcol = col_by_lower("headline")
    lcol = col_by_lower("language")
    ycol = col_by_lower("label")

    cache_df = pd.read_csv(args.cache)
    orig_col, en_col = find_translation_columns(cache_df)

    cache_df["_k"] = cache_df[orig_col].astype(str).map(lambda x: normalize_text(x))
    df["_k"] = df[hcol].astype(str).map(lambda x: normalize_text(x))

    merged = df.merge(cache_df[["_k", en_col]], on="_k", how="left")
    merged.rename(columns={en_col: "headline_en"}, inplace=True)

    # ✅ BALANCED SAMPLING: up to max_rows PER language
    rows_by_lang = []
    langs = list(merged[lcol].astype(str).str.lower().unique())
    langs = [lg for lg in langs if lg]  # drop empty
    for lg in langs:
        subset = merged[merged[lcol].astype(str).str.lower() == lg]
        rows_by_lang.append(subset.head(args.max_rows))
    use = pd.concat(rows_by_lang, ignore_index=True)

    print("[INFO] Languages found:", langs)
    print("[INFO] Rows sampled (per language cap):", args.max_rows)
    print("[INFO] Total rows to process:", len(use))

    retrieval_cache = {}
    if os.path.exists(args.retrieval_cache_out):
        try:
            with open(args.retrieval_cache_out, "r", encoding="utf-8") as f:
                retrieval_cache = json.load(f)
        except Exception:
            retrieval_cache = {}

    rows = []
    for idx, r in use.iterrows():
        headline = str(r[hcol])
        lang = str(r[lcol]).strip().lower()
        label = int(r[ycol])

        en = r.get("headline_en", "")
        en = normalize_text(str(en)) if pd.notna(en) else ""
        if not en:
            en = normalize_text(headline)

        lang_used = lang if lang in SUPPORTED_LANGS else detect_language(headline)

        feats, retrieval_cache = compute_features_for_claim(
            claim_native=headline,
            claim_en=en,
            lang=lang_used,
            keys=keys,
            top_k=args.top_k,
            request_sleep=args.sleep,
            retrieval_cache=retrieval_cache,
        )

        out_row = {
            "headline": headline,
            "language": lang_used,
            "label": label,
            "headline_en": en,
            **feats,
        }
        rows.append(out_row)

        if (len(rows) % 25) == 0:
            print(f"Processed {len(rows)}/{len(use)}")
            with open(args.retrieval_cache_out, "w", encoding="utf-8") as f:
                json.dump(retrieval_cache, f)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False, encoding="utf-8")
    with open(args.retrieval_cache_out, "w", encoding="utf-8") as f:
        json.dump(retrieval_cache, f)

    print("[DONE] wrote:", args.out)
    print("[DONE] retrieval cache:", args.retrieval_cache_out)

if __name__ == "__main__":
    main()

