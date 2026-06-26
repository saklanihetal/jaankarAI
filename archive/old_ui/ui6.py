# app_streamlit_final_v5_pro.py
# Product-grade Investigation Console (Evidence-first UI + Debug Console + Faster pipeline)
# - Evidence cards show provider + source + retrieved-by query + scores
# - Debug tab shows source health, queries, provider breakdown, features, gates
# - Faster: parallel fetch, batch LaBSE encode, translate only top displayed
# - Audio: browser speech synthesis reads verdict + explanation

import os, re, io, json, time, unicodedata
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import streamlit as st
import streamlit.components.v1 as components
import requests
import numpy as np
import pandas as pd
from PIL import Image

try:
    import feedparser
except Exception:
    feedparser = None

try:
    import spacy
except Exception:
    spacy = None

import torch
from langdetect import detect
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification,
    CLIPProcessor, CLIPModel,
)
from sentence_transformers import SentenceTransformer, util

from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="FactCheck Console", layout="wide")

TOP_K_NLI = 5
RSS_PER_QUERY = 12
MAX_ITEMS_FETCH = 140
HTTP_TIMEOUT = 12

MIN_REL_FOR_NLI = 0.18
REL_GATE_R0 = 0.45
REL_GATE_R1 = 0.70

SUPPORTED_LANGS = {"en", "hi", "kn", "te"}
LANG_MAP = {"hi": "hin_Deva", "kn": "kan_Knda", "te": "tel_Telu"}
CEID_MAP = {"en": "IN:en", "hi": "IN:hi", "kn": "IN:kn", "te": "IN:te"}

EVIDENCE_TRANSLATE_TOP_N = 12  # translate only what you show

_EN_STOP = {
    "the","a","an","and","or","to","of","in","on","for","with","from","by","at","as","is","are","was","were","be","been",
    "it","this","that","these","those","after","before","new","latest","today","yesterday","tomorrow","over","into",
    "near","around","amid","says","say","said","will","would","can","could","may","might","has","have","had",
    "about","more","most","very",
}

# =========================
# STYLES (minimal but “senior”)
# =========================
st.markdown("""
<style>
.block-container { padding-top: 1.6rem; padding-bottom: 2.0rem; }
h1 { font-size: 1.8rem !important; margin-bottom: 0.1rem !important; }
.sub { opacity: 0.75; margin-bottom: 1.0rem; }

.card {
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  border-radius: 16px;
  padding: 14px 16px;
}
.rowgap { margin-top: 12px; }
.kpi { font-size: 0.8rem; opacity: 0.75; }
.big { font-size: 1.25rem; font-weight: 750; letter-spacing: 0.2px; }
.badge {
  display:inline-block; padding: 4px 10px; border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.04);
  font-size: 0.82rem; margin-right: 8px; margin-top: 6px;
}
.evi {
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.02);
  border-radius: 14px;
  padding: 12px 12px;
  margin-bottom: 10px;
}
.evi .t { font-weight: 700; margin-bottom: 0.3rem; }
.evi .m { opacity: 0.78; font-size: 0.85rem; }
.evi .q { opacity: 0.70; font-size: 0.83rem; margin-top: 6px; }
hr { opacity: 0.25; }
</style>
""", unsafe_allow_html=True)

st.markdown("# FactCheck Console")
st.markdown("<div class='sub'>Evidence-first verification console · sources, queries, consensus, and explainability</div>", unsafe_allow_html=True)

# =========================
# UTILS
# =========================
def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s)
    return s

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return lang if lang in SUPPORTED_LANGS else "en"
    except Exception:
        return "en"

def sanitize_api_query(q: str, max_len: int = 140) -> str:
    q = normalize_text(q)
    q = re.sub(r"[^\w\s]", " ", q, flags=re.UNICODE)
    q = re.sub(r"\s+", " ", q).strip()
    return q[:max_len].strip()

def rel_gate(relevance: float) -> float:
    if REL_GATE_R1 <= REL_GATE_R0:
        return 1.0
    return clamp01((relevance - REL_GATE_R0) / (REL_GATE_R1 - REL_GATE_R0))

def strip_html_garbage(s: str) -> str:
    s = s or ""
    s = re.sub(r"<a\b[^>]*>(.*?)</a>", r"\1", s, flags=re.I | re.S)
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("&nbsp;", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def keyword_set_en(text: str) -> set:
    text = normalize_text(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    toks = [t for t in text.split() if len(t) > 2 and t not in _EN_STOP]
    return set(toks)

def safe_get(url: str, timeout: int = HTTP_TIMEOUT) -> Tuple[str, str]:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code != 200:
            return "", f"HTTP {r.status_code}"
        return r.text, ""
    except Exception as e:
        return "", str(e)

def safe_get_json(url: str, timeout: int = HTTP_TIMEOUT) -> Tuple[dict, str]:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code != 200:
            return {}, f"HTTP {r.status_code}: {r.text[:200]}"
        return r.json(), ""
    except Exception as e:
        return {}, str(e)

def safe_post_json(url: str, payload: dict, timeout: int = HTTP_TIMEOUT) -> Tuple[dict, str]:
    try:
        r = requests.post(url, json=payload, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code != 200:
            return {}, f"HTTP {r.status_code}: {r.text[:200]}"
        return r.json(), ""
    except Exception as e:
        return {}, str(e)

# =========================
# MODELS (reuse your v4 ones)
# =========================
@st.cache_resource
def load_spacy():
    if spacy is None:
        return None
    for name in ["en_core_web_trf", "en_core_web_sm"]:
        try:
            return spacy.load(name)
        except Exception:
            continue
    return None

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    labse = SentenceTransformer("sentence-transformers/LaBSE")

    translator_name = "facebook/nllb-200-distilled-600M"
    nllb_tok = AutoTokenizer.from_pretrained(translator_name)
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained(translator_name).to(device)

    nli_name = "facebook/bart-large-mnli"
    nli_tok = AutoTokenizer.from_pretrained(nli_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_name).to(device)

    return device, labse, nllb_tok, nllb_model, nli_tok, nli_model

def translate_to_english(text: str, lang: str, device, tok, model) -> str:
    text = normalize_text(text)
    if not text or lang == "en" or lang not in LANG_MAP:
        return text
    try:
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=256).to(device)
        eng_token_id = tok.convert_tokens_to_ids("eng_Latn")
        with torch.no_grad():
            out = model.generate(**inputs, forced_bos_token_id=eng_token_id, max_length=128)
        return tok.decode(out[0], skip_special_tokens=True)
    except Exception:
        return text

def nli_probs(premise: str, hypothesis: str, device, tok, model) -> Tuple[float, float, float]:
    inputs = tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
    # contradiction, neutral, entailment
    return float(probs[2]), float(probs[0]), float(probs[1])

# =========================
# DATA
# =========================
@dataclass
class EvidenceItem:
    title: str
    url: str
    provider: str      # NewsAPI.org / GNews / NewsData / EventRegistry / GoogleNewsRSS
    source: str        # publisher/domain if available; else provider
    desc: str = ""
    published: str = ""
    retrieved_by_query: str = ""
    fetch_ms: int = 0

    relevance: float = 0.0
    nli_label: str = "NEUTRAL"
    f_ent: float = 0.0
    f_con: float = 0.0
    f_neu: float = 0.0
    weighted_con_used: float = 0.0
    rel_gate: float = 0.0

    title_lang: str = "en"
    title_en: str = ""

# =========================
# FETCHERS (attach provider + retrieved_by_query)
# =========================
def google_news_rss(query: str, hl: str, gl: str, ceid: str) -> Tuple[List[EvidenceItem], str, int]:
    t0 = time.time()
    if feedparser is None:
        return [], "feedparser not installed", int((time.time()-t0)*1000)
    q = requests.utils.quote(query)
    url = f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"
    txt, err = safe_get(url)
    ms = int((time.time()-t0)*1000)
    if err:
        return [], err, ms

    feed = feedparser.parse(txt)
    out: List[EvidenceItem] = []
    for e in feed.entries[:RSS_PER_QUERY]:
        title = normalize_text(getattr(e, "title", "") or "")
        link  = getattr(e, "link", "") or ""
        desc  = strip_html_garbage(getattr(e, "summary", "") or "")
        published = normalize_text(getattr(e, "published", "") or "")
        if title and link:
            out.append(EvidenceItem(
                title=title, url=link,
                provider="GoogleNewsRSS",
                source="GoogleNewsRSS",
                desc=desc, published=published,
                retrieved_by_query=query,
                fetch_ms=ms,
            ))
    return out, "", ms

def fetch_newsapi_org(query: str, api_key: str) -> Tuple[List[EvidenceItem], str, int]:
    t0 = time.time()
    q = sanitize_api_query(query)
    url = ("https://newsapi.org/v2/everything?"
           f"q={requests.utils.quote(q)}&language=en&pageSize=20&sortBy=publishedAt&apiKey={api_key}")
    res, err = safe_get_json(url)
    ms = int((time.time()-t0)*1000)
    if err:
        return [], err, ms

    out: List[EvidenceItem] = []
    for a in (res.get("articles", []) or [])[:20]:
        title = normalize_text(a.get("title","") or "")
        link  = a.get("url","") or ""
        desc  = strip_html_garbage(a.get("description","") or "")
        src   = (a.get("source", {}) or {}).get("name") or "NewsAPI.org"
        if title and link:
            out.append(EvidenceItem(title=title, url=link, provider="NewsAPI.org", source=src,
                                    desc=desc, retrieved_by_query=query, fetch_ms=ms))
    return out, "", ms

def fetch_gnews(query: str, api_key: str) -> Tuple[List[EvidenceItem], str, int]:
    t0 = time.time()
    q = sanitize_api_query(query)
    url = f"https://gnews.io/api/v4/search?q={requests.utils.quote(q)}&lang=en&max=20&token={api_key}"
    res, err = safe_get_json(url)
    ms = int((time.time()-t0)*1000)
    if err:
        return [], err, ms

    out: List[EvidenceItem] = []
    for a in (res.get("articles", []) or [])[:20]:
        title = normalize_text(a.get("title","") or "")
        link  = a.get("url","") or ""
        desc  = strip_html_garbage(a.get("description","") or "")
        src   = a.get("source", {}).get("name") if isinstance(a.get("source",{}), dict) else None
        out.append(EvidenceItem(title=title, url=link, provider="GNews", source=src or "GNews",
                                desc=desc, retrieved_by_query=query, fetch_ms=ms))
    return out, "", ms

def fetch_newsdata(query: str, api_key: str) -> Tuple[List[EvidenceItem], str, int]:
    t0 = time.time()
    q = sanitize_api_query(query)
    url = f"https://newsdata.io/api/1/news?q={requests.utils.quote(q)}&language=en&apikey={api_key}"
    res, err = safe_get_json(url)
    ms = int((time.time()-t0)*1000)
    if err:
        return [], err, ms

    out: List[EvidenceItem] = []
    for a in (res.get("results", []) or [])[:20]:
        title = normalize_text(a.get("title","") or "")
        link  = a.get("link","") or ""
        desc  = strip_html_garbage(a.get("description","") or "")
        src   = a.get("source_id") or "NewsData.io"
        if title and link:
            out.append(EvidenceItem(title=title, url=link, provider="NewsData.io", source=src,
                                    desc=desc, retrieved_by_query=query, fetch_ms=ms))
    return out, "", ms

def fetch_eventregistry(query: str, api_key: str) -> Tuple[List[EvidenceItem], str, int]:
    t0 = time.time()
    q = sanitize_api_query(query)
    url = "https://eventregistry.org/api/v1/article/getArticles"
    payload = {"action":"getArticles","keyword":q,"lang":"eng","articlesCount":20,"apiKey":api_key}
    res, err = safe_post_json(url, payload)
    ms = int((time.time()-t0)*1000)
    if err:
        return [], err, ms

    out: List[EvidenceItem] = []
    for a in (res.get("articles", {}) or {}).get("results", [])[:20]:
        title = normalize_text(a.get("title","") or "")
        link  = a.get("url","") or ""
        body  = strip_html_garbage(a.get("body","") or "")[:240]
        src   = (a.get("source", {}) or {}).get("title") if isinstance(a.get("source",{}), dict) else None
        if title and link:
            out.append(EvidenceItem(title=title, url=link, provider="EventRegistry", source=src or "EventRegistry",
                                    desc=body, retrieved_by_query=query, fetch_ms=ms))
    return out, "", ms

def dedup_items(items: List[EvidenceItem]) -> List[EvidenceItem]:
    seen = set()
    out = []
    for it in items:
        t = normalize_text(it.title).lower()
        u = (it.url or "").strip()
        key = (t, u[:100])
        if not t or key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out

# =========================
# QUERY GENERATION (entity-driven)
# =========================
def generate_queries(claim_raw: str, claim_en: str, nlp) -> Dict[str, List[str]]:
    # RSS: allow raw + en + short
    rss = []
    if len(claim_raw) >= 5: rss.append(claim_raw)
    if len(claim_en) >= 5 and claim_en.lower() != claim_raw.lower(): rss.append(claim_en)
    rss.append(" ".join(claim_en.split()[:10]))

    # API: strict, sanitized, entity-driven
    api = []
    api.append(sanitize_api_query(claim_en))
    api.append(sanitize_api_query(" ".join(claim_en.split()[:10])))

    ents = []
    kws  = []
    if nlp is not None and claim_en.strip():
        doc = nlp(claim_en)
        for ent in doc.ents:
            if ent.label_ in {"PERSON","ORG","GPE","LOC","EVENT","PRODUCT"}:
                v = sanitize_api_query(ent.text, max_len=60)
                if v and v not in ents:
                    ents.append(v)
        # noun chunks / keywords
        for t in doc:
            if t.is_stop or t.is_punct: 
                continue
            if t.pos_ in {"NOUN","PROPN"} and len(t.text) > 2:
                v = sanitize_api_query(t.text, max_len=30)
                if v and v not in kws:
                    kws.append(v)

    if ents:
        api.append(sanitize_api_query(" ".join(ents[:4]), max_len=120))
        # entity + top keywords
        api.append(sanitize_api_query(" ".join((ents[:2] + kws[:4]))[:120]))

    # uniq + keep order
    def uniq(xs):
        out, s = [], set()
        for x in xs:
            x = normalize_text(x)
            if len(x) < 4: 
                continue
            k = x.lower()
            if k not in s:
                s.add(k); out.append(x)
        return out

    return {"rss": uniq(rss)[:5], "api": uniq(api)[:6], "entities": ents[:8], "keywords": kws[:12]}

# =========================
# RANK + CONSENSUS + NLI (batch relevance)
# =========================
def rank_by_relevance_batch(items: List[EvidenceItem], claim_en: str, labse: SentenceTransformer) -> List[EvidenceItem]:
    texts = [normalize_text((it.title + " " + (it.desc or "")).strip()) for it in items]
    claim_emb = labse.encode([claim_en], convert_to_tensor=True)
    embs = labse.encode(texts, convert_to_tensor=True, batch_size=32, show_progress_bar=False)
    sims = util.cos_sim(claim_emb, embs).squeeze(0).cpu().numpy().tolist()
    for it, s in zip(items, sims):
        it.relevance = float(s)
    items.sort(key=lambda x: x.relevance, reverse=True)
    return items

def consensus_context(items: List[EvidenceItem], claim_en: str, nlp) -> Tuple[List[str], List[str], float]:
    top = items[: min(14, len(items))]
    if not top:
        return [], [], 0.0

    loc_counts: Dict[str,int] = {}
    if nlp is not None:
        for it in top:
            doc = nlp(normalize_text(it.title))
            for ent in doc.ents:
                if ent.label_ in {"GPE","LOC"}:
                    k = ent.text.strip().lower()
                    if len(k) >= 3:
                        loc_counts[k] = loc_counts.get(k,0) + 1
    rep_locs = [k for k,c in sorted(loc_counts.items(), key=lambda x:-x[1]) if c >= 2][:6]

    word_counts: Dict[str,int] = {}
    for it in top:
        toks = keyword_set_en(it.title + " " + (it.desc or ""))
        for t in toks:
            word_counts[t] = word_counts.get(t,0)+1
    rep_words = [w for w,c in sorted(word_counts.items(), key=lambda x:-x[1]) if c >= 3][:12]

    claim_tokens = keyword_set_en(claim_en)
    missing = [w for w in rep_words if w not in claim_tokens]
    mismatch_rate = (len(missing) / max(1, len(rep_words))) if rep_words else 0.0
    return rep_locs, rep_words, float(mismatch_rate)

def aggregate_topk_nli(items: List[EvidenceItem], claim_en: str, device, nli_tok, nli_model) -> Tuple[float, float, List[EvidenceItem]]:
    top = items[: min(TOP_K_NLI, len(items))]
    scored: List[EvidenceItem] = []
    max_ent = 0.0
    max_con_used = 0.0

    for it in top:
        if it.relevance < MIN_REL_FOR_NLI:
            continue
        premise = strip_html_garbage(normalize_text((it.title + ". " + (it.desc or "")).strip()))
        premise = premise[:280]
        if not premise:
            continue

        f_ent, f_con, f_neu = nli_probs(premise, claim_en, device, nli_tok, nli_model)
        g = rel_gate(it.relevance)
        w_con = f_con * (1.0 - f_ent)
        w_used = w_con * g

        it.f_ent, it.f_con, it.f_neu = f_ent, f_con, f_neu
        it.rel_gate = float(g)
        it.weighted_con_used = float(w_used)

        if f_ent >= f_con and f_ent >= f_neu:
            it.nli_label = "ENTAILS"
        elif f_con >= f_neu:
            it.nli_label = "CONTRADICTS"
        else:
            it.nli_label = "NEUTRAL"

        max_ent = max(max_ent, f_ent)
        max_con_used = max(max_con_used, w_used)
        scored.append(it)

    return float(max_ent), float(max_con_used), scored

# =========================
# TTS (browser SpeechSynthesis)
# =========================
def speak_block(text: str, key: str):
    safe = (text or "").replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    html = f"""
    <script>
    const speak = () => {{
      const msg = new SpeechSynthesisUtterance(`{safe}`);
      msg.rate = 1.0; msg.pitch = 1.0; msg.lang = 'en-US';
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(msg);
    }};
    speak();
    </script>
    """
    components.html(html, height=0)

# =========================
# SIDEBAR INPUT (form to prevent reruns)
# =========================
nlp = load_spacy()
device, labse, nllb_tok, nllb_model, nli_tok, nli_model = load_models()

st.sidebar.markdown("### Input")
with st.sidebar.form("run_form", clear_on_submit=False):
    news_text = st.text_area("Headline / claim (any language)", height=120)
    show_debug = st.checkbox("Show Debug Console", value=True)
    run = st.form_submit_button("Run verification")

# =========================
# MAIN
# =========================
if run and news_text.strip():
    claim_raw = normalize_text(news_text)
    lang = detect_language(claim_raw)
    claim_en = translate_to_english(claim_raw, lang, device, nllb_tok, nllb_model)

    # Generate queries
    qpack = generate_queries(claim_raw, claim_en, nlp)
    rss_queries = qpack["rss"]
    api_queries = qpack["api"]

    # Load keys
    keys = {}
    try:
        keys = {
            "NEWSAPI_KEY": st.secrets.get("NEWSAPI_KEY",""),
            "GNEWS_KEY": st.secrets.get("GNEWS_KEY",""),
            "NEWSDATA_KEY": st.secrets.get("NEWSDATA_KEY",""),
            "EVENTREGISTRY_KEY": st.secrets.get("EVENTREGISTRY_KEY",""),
        }
        keys = {k:v for k,v in keys.items() if v}
    except Exception:
        keys = {}

    # -------- Fetch evidence in parallel --------
    source_status: Dict[str, str] = {}
    provider_rows = []  # for debug breakdown
    all_items: List[EvidenceItem] = []

    jobs = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        # APIs (only if keys exist)
        if keys.get("NEWSAPI_KEY"):
            for q in api_queries:
                jobs.append(("NewsAPI.org", q, ex.submit(fetch_newsapi_org, q, keys["NEWSAPI_KEY"])))
        if keys.get("GNEWS_KEY"):
            for q in api_queries:
                jobs.append(("GNews", q, ex.submit(fetch_gnews, q, keys["GNEWS_KEY"])))
        if keys.get("NEWSDATA_KEY"):
            for q in api_queries:
                jobs.append(("NewsData.io", q, ex.submit(fetch_newsdata, q, keys["NEWSDATA_KEY"])))
        if keys.get("EVENTREGISTRY_KEY"):
            for q in api_queries:
                jobs.append(("EventRegistry", q, ex.submit(fetch_eventregistry, q, keys["EVENTREGISTRY_KEY"])))

        # RSS always
        for q in rss_queries:
            jobs.append(("GoogleNewsRSS(en-IN)", q, ex.submit(google_news_rss, q, "en-IN", "IN", CEID_MAP["en"])))

        # Local RSS bias
        if lang in {"hi","kn","te"}:
            hl_map = {"hi":"hi-IN","kn":"kn-IN","te":"te-IN"}
            hl = hl_map[lang]
            for q in rss_queries:
                jobs.append((f"GoogleNewsRSS({hl})", q, ex.submit(google_news_rss, q, hl, "IN", CEID_MAP[lang])))

        # Collect
        per_provider_ok = {}
        per_provider_err = {}
        per_provider_cnt = {}
        per_provider_ms = {}

        for prov, q, fut in jobs:
            its, err, ms = fut.result()
            per_provider_ms.setdefault(prov, []).append(ms)
            if err:
                per_provider_err[prov] = err
            else:
                per_provider_ok[prov] = True
                all_items.extend(its)
                per_provider_cnt[prov] = per_provider_cnt.get(prov, 0) + len(its)

        # Source status summary
        for prov in sorted(set([p for p,_,_ in jobs])):
            if per_provider_ok.get(prov):
                source_status[prov] = "OK"
            else:
                source_status[prov] = per_provider_err.get(prov, "No results")

        # Provider breakdown rows
        for prov in sorted(set([p for p,_,_ in jobs])):
            provider_rows.append({
                "provider": prov,
                "status": source_status.get(prov),
                "items": int(per_provider_cnt.get(prov, 0)),
                "avg_fetch_ms": int(np.mean(per_provider_ms.get(prov, [0]))),
                "error": per_provider_err.get(prov, ""),
            })

    # Dedup + cap
    before = len(all_items)
    items = dedup_items(all_items)[:MAX_ITEMS_FETCH]
    after = len(items)

    if not items:
        st.markdown("<div class='card'><b>No evidence found.</b><br/>Try adding a key entity (person/place) or shorten the claim.</div>", unsafe_allow_html=True)
        if show_debug:
            st.markdown("## Debug Console")
            st.json(source_status)
            st.write("API queries:", api_queries)
            st.write("RSS queries:", rss_queries)
        st.stop()

    # Rank + consensus + NLI
    ranked = rank_by_relevance_batch(items, claim_en, labse)
    rep_locs, rep_words, ctx_mismatch_rate = consensus_context(ranked, claim_en, nlp)
    max_ent, max_con_used, scored = aggregate_topk_nli(ranked, claim_en, device, nli_tok, nli_model)

    # Verdict rule (keep simple here; plug your gates back in)
    if max_con_used >= 0.35 and ranked[0].relevance >= 0.45:
        decision = "CONTRADICTED"
    elif max_ent >= 0.80 and ranked[0].relevance >= 0.45 and max_con_used < 0.15:
        decision = "SUPPORTED"
    else:
        decision = "UNVERIFIED"

    # Explanation (human-readable)
    expl = []
    if decision == "SUPPORTED":
        expl.append("Multiple relevant sources align with the claim.")
    elif decision == "CONTRADICTED":
        expl.append("Relevant sources contain conflicting information versus the claim.")
    else:
        expl.append("Evidence is insufficient to strongly support or contradict the claim.")

    if ctx_mismatch_rate >= 0.50 and rep_words:
        expl.append("Evidence contains repeated context details that are missing from the claim (lowers confidence).")

    if rep_locs:
        expl.append("Consensus locations in evidence: " + ", ".join(rep_locs[:5]))
    if rep_words:
        expl.append("Repeated context keywords in evidence: " + ", ".join(rep_words[:10]))

    # =========================
    # UI LAYOUT
    # =========================
    tab_results, tab_evidence, tab_debug = st.tabs(["Results", "Evidence", "Debug Console" if show_debug else "Debug (hidden)"])

    with tab_results:
        # Claim + Verdict row
        l, r = st.columns([1.25, 0.75], gap="large")
        with l:
            st.markdown("### Claim")
            st.markdown(f"<div class='card'><div class='kpi'>Original</div><div><b>{claim_raw}</b></div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='card rowgap'><div class='kpi'>English (for retrieval/NLI)</div><div><b>{claim_en}</b></div></div>", unsafe_allow_html=True)
            st.markdown(
                f"<span class='badge'><b>Language</b>: {lang}</span>"
                f"<span class='badge'><b>Evidence</b>: {after} (dedup from {before})</span>",
                unsafe_allow_html=True
            )

        with r:
            st.markdown("### Verdict")
            st.markdown(
                f"<div class='card'><div class='kpi'>Final</div><div class='big'>{decision}</div>"
                f"<div class='rowgap'><span class='badge'><b>Max entail</b>: {max_ent:.2f}</span>"
                f"<span class='badge'><b>Max contra-used</b>: {max_con_used:.2f}</span>"
                f"<span class='badge'><b>Top relevance</b>: {ranked[0].relevance:.2f}</span></div>"
                f"</div>",
                unsafe_allow_html=True
            )

            if st.button("🔊 Read verdict & explanation", key="speak"):
                speak_text = f"Verdict: {decision}. " + " ".join(expl[:4])
                speak_block(speak_text, key="tts")

        # Consensus (always visible)
        st.markdown("### Consensus context found in evidence")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("<div class='card'><div class='kpi'>Locations repeated by multiple sources</div>"
                        + ("<br/>".join([f"<span class='badge'>{x}</span>" for x in rep_locs]) if rep_locs else "<i>(none)</i>")
                        + "</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='card'><div class='kpi'>Repeated context words</div>"
                        + ("<br/>".join([f"<span class='badge'>{x}</span>" for x in rep_words[:12]]) if rep_words else "<i>(none)</i>")
                        + "</div>", unsafe_allow_html=True)

        # Explanation
        st.markdown("### Human-readable explanation")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        for line in expl:
            st.markdown(f"- {line}")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_evidence:
        st.markdown("### Evidence feed")

        # Filter controls
        f1, f2, f3 = st.columns([0.35, 0.35, 0.30])
        with f1:
            filt = st.selectbox("Filter", ["All", "Supports (ENTAILS)", "Contradicts", "Neutral"], index=0)
        with f2:
            sort = st.selectbox("Sort", ["Relevance", "Contradiction-used", "Entailment"], index=0)
        with f3:
            show_n = st.slider("Items", 5, 30, 12)

        # Choose set
        base = scored if scored else ranked
        def passes(it: EvidenceItem) -> bool:
            if filt == "All": return True
            if filt.startswith("Supports"): return it.nli_label == "ENTAILS"
            if filt == "Contradicts": return it.nli_label == "CONTRADICTS"
            if filt == "Neutral": return it.nli_label == "NEUTRAL"
            return True

        view = [it for it in base if passes(it)]
        if sort == "Relevance":
            view.sort(key=lambda x: x.relevance, reverse=True)
        elif sort == "Contradiction-used":
            view.sort(key=lambda x: x.weighted_con_used, reverse=True)
        else:
            view.sort(key=lambda x: x.f_ent, reverse=True)

        # Translate only top displayed non-English
        for it in view[:min(show_n, EVIDENCE_TRANSLATE_TOP_N)]:
            it.title_lang = detect_language(it.title)
            if it.title_lang != "en":
                # lightweight: use your NLLB only when needed
                it.title_en = translate_to_english(it.title, it.title_lang, device, nllb_tok, nllb_model)
            else:
                it.title_en = it.title

        for it in view[:show_n]:
            title_block = it.title
            if it.title_lang != "en" and it.title_en and it.title_en != it.title:
                title_block += f"<div class='q'><i>English:</i> {it.title_en}</div>"

            st.markdown(
                f"""
<div class="evi">
  <div class="t">{title_block}</div>
  <div class="m">
    <b>Provider</b>: {it.provider} · <b>Source</b>: {it.source} · <b>Fetch</b>: {it.fetch_ms}ms
    <br/>
    <b>Relevance</b>: {it.relevance:.3f} · <b>NLI</b>: {it.nli_label} (ent={it.f_ent:.2f}, con={it.f_con:.2f}, neu={it.f_neu:.2f})
    · <b>Contradiction used</b>: {it.weighted_con_used:.2f} (rel_gate={it.rel_gate:.2f})
  </div>
  <div class="q"><b>Retrieved by query:</b> {it.retrieved_by_query}</div>
  <div style="margin-top:8px;"><a href="{it.url}" target="_blank">Open article</a></div>
</div>
""",
                unsafe_allow_html=True
            )

    with tab_debug:
        if not show_debug:
            st.info("Debug Console is hidden. Enable it from sidebar.")
        else:
            st.markdown("### Source status")
            st.json(source_status)

            st.markdown("### Queries generated")
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.write("API queries:", api_queries)
            st.write("RSS queries:", rss_queries)
            st.write("Entities:", qpack.get("entities", []))
            st.write("Keywords:", qpack.get("keywords", []))
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("### Provider breakdown")
            st.dataframe(pd.DataFrame(provider_rows))

            st.markdown("### Dedup stats")
            st.write({"fetched_total": before, "deduped": after, "dropped": before - after})

            st.markdown("### Scoring snapshot (top 5)")
            snap = [{
                "title": it.title[:90],
                "provider": it.provider,
                "source": it.source,
                "query": it.retrieved_by_query[:60],
                "rel": round(it.relevance, 3),
                "nli": it.nli_label,
                "ent": round(it.f_ent, 2),
                "con": round(it.f_con, 2),
                "con_used": round(it.weighted_con_used, 2),
            } for it in (scored[:5] if scored else ranked[:5])]
            st.dataframe(pd.DataFrame(snap))

st.caption("Decision-support only. Open articles for full verification.")

