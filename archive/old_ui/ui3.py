# app_streamlit_final_v4_ui_polished.py
# V4 (UI polished): same pipeline + fixed RSS CEID + safer API queries + Rekognition face fallback
# UI changes:
# - Professional title + subtitle
# - Consistent typography via CSS (no random header/body size differences)
# - Verdict + key visual signals placed directly under Claim (in the blank space)
# - No yellow warning blocks (replaced with neutral styled callouts)
# - Removed "items per source" line
# - Removed deprecated use_column_width (uses use_container_width / width)

import os
import re
import io
import json
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import streamlit as st
import requests
import torch
from PIL import Image

try:
    import feedparser
except Exception:
    feedparser = None

try:
    import spacy
except Exception:
    spacy = None

try:
    import boto3
except Exception:
    boto3 = None

from langdetect import detect
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    CLIPProcessor,
    CLIPModel,
)
from sentence_transformers import SentenceTransformer, util
import joblib
import numpy as np
import pandas as pd

# TensorFlow for deepfake check
try:
    import tensorflow as tf
except Exception:
    tf = None


# ===========================
# PAGE SETUP
# ===========================
st.set_page_config(page_title="AI-based Regional Fake News Detection", layout="wide")

# ===========================
# GLOBAL STYLES
# ===========================
st.markdown(
    """
<style>
/* Layout breathing room */
.block-container { padding-top: 2.0rem; padding-bottom: 2.0rem; }

/* Typography consistency */
h1, h2, h3 { letter-spacing: 0.2px; }
h1 { font-size: 2.0rem !important; margin-bottom: 0.25rem !important; }
h2 { font-size: 1.25rem !important; margin-top: 1.25rem !important; }
h3 { font-size: 1.05rem !important; margin-top: 1.0rem !important; }

p, li, div, span { font-size: 0.98rem; line-height: 1.55; }

/* Section label (small caps style) */
.smalllabel {
  font-size: 0.82rem;
  opacity: 0.72;
  margin-bottom: 0.2rem;
}

/* Cards */
.card {
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.03);
  border-radius: 16px;
  padding: 14px 16px;
}

.card + .card { margin-top: 12px; }

/* Verdict */
.verdict {
  border-radius: 16px;
  padding: 14px 16px;
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.03);
}
.verdict .k { font-size: 0.82rem; opacity: 0.75; margin-bottom: 0.25rem; }
.verdict .v { font-size: 1.35rem; font-weight: 700; letter-spacing: 0.3px; margin: 0; }

/* Badges */
.badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 0.82rem;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.04);
  margin-right: 8px;
}
.badge strong { font-weight: 700; }

/* Neutral callouts (no yellow) */
.callout {
  border-radius: 14px;
  padding: 10px 12px;
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.03);
}
.callout p { margin: 0; }

/* Evidence items */
.evidence {
  border: 1px solid rgba(255,255,255,0.08);
  background: rgba(255,255,255,0.02);
  border-radius: 14px;
  padding: 12px 12px;
  margin-bottom: 10px;
}
.evidence .title { font-weight: 650; margin-bottom: 0.3rem; }
.evidence .meta { opacity: 0.75; font-size: 0.86rem; }
.evidence .desc { margin-top: 0.35rem; }

/* Sidebar subtle */
[data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,0.06); }

/* Hide Streamlit default status color blocks a bit */
div[data-testid="stAlert"] { border-radius: 14px; }
</style>
""",
    unsafe_allow_html=True,
)

# ===========================
# HEADER
# ===========================
st.markdown("# AI-based Regional Fake News Detection")
st.markdown(
    "Decision-support system that cross-checks a claim against retrieved evidence and visual signals."
)
st.markdown("---")


# ===========================
# CONSTANTS
# ===========================
TOP_K_NLI = 5
MAX_ITEMS_FETCH = 120
RSS_PER_QUERY = 12
HTTP_TIMEOUT = 12

DEFAULT_SUPPORT_THR = 0.35
DEFAULT_CONTRA_THR = 0.70

REL_GATE_R0 = 0.45
REL_GATE_R1 = 0.70
MIN_REL_FOR_NLI = 0.18

CLIP_A = 0.20
CLIP_B = 0.35

LIKELY_THR = 0.80
NEUTRAL_BAND = 0.15

SUPPORTED_LANGS = {"en", "hi", "kn", "te"}
LANG_MAP = {"hi": "hin_Deva", "kn": "kan_Knda", "te": "tel_Telu"}

# Correct Google News edition ids
CEID_MAP = {"en": "IN:en", "hi": "IN:hi", "kn": "IN:kn", "te": "IN:te"}

FEATURE_COLUMNS = [
    "num_evidence",
    "max_rel", "mean_rel", "p90_rel",
    "max_ent", "mean_ent",
    "max_con_used", "mean_con",
    "body_success_rate",
    "context_mismatch_rate",
    "update_style_rate",
]

_EN_STOP = {
    "the","a","an","and","or","to","of","in","on","for","with","from","by","at","as","is","are","was","were","be","been",
    "it","this","that","these","those","after","before","new","latest","today","yesterday","tomorrow","over","into",
    "near","around","amid","says","say","said","will","would","can","could","may","might","has","have","had","up","down","out",
    "about","more","most","very",
}


# ===========================
# UTILS
# ===========================
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

def keyword_set_en(text: str) -> set:
    text = normalize_text(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    toks = [t for t in text.split() if len(t) > 2 and t not in _EN_STOP]
    return set(toks)

def rel_gate(relevance: float, r0: float = REL_GATE_R0, r1: float = REL_GATE_R1) -> float:
    if r1 <= r0:
        return 1.0
    return clamp01((relevance - r0) / (r1 - r0))

def sanitize_api_query(q: str, max_len: int = 140) -> str:
    """
    Make queries safe for stricter APIs (esp. GNews).
    - Remove punctuation
    - Collapse whitespace
    - Truncate
    """
    q = normalize_text(q)
    q = re.sub(r"[^\w\s]", " ", q, flags=re.UNICODE)
    q = re.sub(r"\s+", " ", q).strip()
    if len(q) > max_len:
        q = q[:max_len].strip()
    return q

def safe_get(url: str, timeout: int = HTTP_TIMEOUT) -> Tuple[str, str]:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return "", f"HTTP {r.status_code}"
        return r.text, ""
    except Exception as e:
        return "", str(e)

def safe_get_json(url: str, timeout: int = HTTP_TIMEOUT) -> Tuple[dict, str]:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return {}, f"HTTP {r.status_code}: {r.text[:200]}"
        return r.json(), ""
    except Exception as e:
        return {}, str(e)

def safe_post_json(url: str, payload: dict, timeout: int = HTTP_TIMEOUT) -> Tuple[dict, str]:
    try:
        r = requests.post(url, json=payload, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return {}, f"HTTP {r.status_code}: {r.text[:200]}"
        return r.json(), ""
    except Exception as e:
        return {}, str(e)

def strip_html_garbage(s: str) -> str:
    s = s or ""
    s = re.sub(r"<a\b[^>]*>(.*?)</a>", r"\1", s, flags=re.I | re.S)
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("&nbsp;", " ").replace("nbsp", " ")
    s = s.replace("font", " ")
    s = re.sub(r"\b(com|www|http|https|google|blank|articles)\b", " ", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ===========================
# MODELS
# ===========================
@st.cache_resource
def load_spacy():
    if spacy is None:
        return None, "spaCy not installed"
    for name in ["en_core_web_trf", "en_core_web_sm"]:
        try:
            return spacy.load(name), name
        except Exception:
            continue
    return None, "spaCy model not installed"

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_name)

    labse = SentenceTransformer("sentence-transformers/LaBSE")

    translator_name = "facebook/nllb-200-distilled-600M"
    nllb_tokenizer = AutoTokenizer.from_pretrained(translator_name)
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained(translator_name).to(device)

    nli_name = "facebook/bart-large-mnli"
    nli_tokenizer = AutoTokenizer.from_pretrained(nli_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_name).to(device)

    return device, clip_model, clip_processor, labse, nllb_tokenizer, nllb_model, nli_tokenizer, nli_model

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

def clip_gate_score(image: Image.Image, text_en: str, device, clip_model, clip_processor) -> Tuple[float, float]:
    inputs = clip_processor(text=[text_en], images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        img = outputs.image_embeds
        txt = outputs.text_embeds
        img = img / img.norm(dim=-1, keepdim=True)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        s = (img * txt).sum(dim=-1).item()
    g = (s - CLIP_A) / (CLIP_B - CLIP_A) if CLIP_B != CLIP_A else 0.0
    return s, clamp01(g)

def nli_probs(premise: str, hypothesis: str, device, tok, model) -> Tuple[float, float, float]:
    inputs = tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
    # BART MNLI order: contradiction, neutral, entailment
    return float(probs[2]), float(probs[0]), float(probs[1])


# ===========================
# Deepfake (CNN .h5)
# ===========================
@st.cache_resource
def load_deepfake_model():
    if tf is None:
        return None, "TensorFlow not installed (pip install tensorflow tf-keras)"
    path = os.path.join("deepfake", "deepfake_cnn.h5")
    if not os.path.exists(path):
        return None, f"Deepfake model not found: {path}"
    try:
        try:
            tf.keras.config.enable_legacy_deserialization()
        except Exception:
            pass
        model = tf.keras.models.load_model(path, compile=False)
        return model, "OK"
    except Exception as e1:
        try:
            import keras
            model = keras.models.load_model(path, compile=False)
            return model, "OK (keras fallback)"
        except Exception as e2:
            return None, f"Load failed: {e1} | fallback failed: {e2}"

def deepfake_predict_pil(pil_img: Image.Image, model):
    img = pil_img.convert("RGB").resize((128, 128))
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    pred = model.predict(arr, verbose=0)
    p_real = float(pred[0][0]) if getattr(pred, "ndim", 0) == 2 else float(pred[0])
    if p_real >= 0.5:
        return p_real, "LIKELY AUTHENTIC", p_real
    else:
        return p_real, "LIKELY MANIPULATED", 1.0 - p_real


# ===========================
# AWS Rekognition (Celebrity)
# ===========================
@st.cache_resource
def load_rekognition_client():
    debug = {"boto3_installed": bool(boto3), "keys_present": False, "region": None}
    if boto3 is None:
        return None, "boto3 not installed (pip install boto3)", debug

    try:
        ak = st.secrets.get("AWS_ACCESS_KEY_ID", "")
        sk = st.secrets.get("AWS_SECRET_ACCESS_KEY", "")
        region = st.secrets.get("AWS_REGION", "ap-south-1")
    except Exception as e:
        debug["region"] = None
        return None, f"st.secrets read failed: {e}", debug

    debug["region"] = region
    debug["keys_present"] = bool(ak and sk)

    if not ak or not sk:
        return None, "AWS creds missing in secrets", debug

    try:
        client = boto3.client(
            "rekognition",
            aws_access_key_id=ak,
            aws_secret_access_key=sk,
            region_name=region
        )
        return client, "OK", debug
    except Exception as e:
        return None, f"Init failed: {e}", debug

def rekognition_detect_celebrity_from_image(pil_img: Image.Image, client) -> Tuple[Optional[str], float, str]:
    try:
        buf = io.BytesIO()
        pil_img.convert("RGB").save(buf, format="JPEG", quality=92)
        img_bytes = buf.getvalue()
        resp = client.recognize_celebrities(Image={"Bytes": img_bytes})
        faces = resp.get("CelebrityFaces", []) or []
        if not faces:
            return None, 0.0, "No celebrity detected (AWS returned empty CelebrityFaces)"
        best = max(faces, key=lambda x: float(x.get("MatchConfidence", 0.0)))
        name = best.get("Name", None)
        conf = float(best.get("MatchConfidence", 0.0))
        return name, conf, "OK"
    except Exception as e:
        return None, 0.0, f"AWS call failed: {e}"

def rekognition_detect_faces_count(pil_img: Image.Image, client) -> Tuple[int, str]:
    try:
        buf = io.BytesIO()
        pil_img.convert("RGB").save(buf, format="JPEG", quality=92)
        resp = client.detect_faces(Image={"Bytes": buf.getvalue()}, Attributes=["DEFAULT"])
        faces = resp.get("FaceDetails", []) or []
        return int(len(faces)), "OK"
    except Exception as e:
        return 0, f"detect_faces failed: {e}"

def extract_person_from_claim(claim_en: str, nlp) -> Optional[str]:
    if nlp is None:
        return None
    try:
        doc = nlp(normalize_text(claim_en))
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text.strip()
    except Exception:
        return None
    return None

def extract_keywords_for_api(claim_en: str, nlp) -> List[str]:
    """
    Help APIs return results more reliably:
    - include PERSON/ORG/GPE entities if available
    - include a short keyword query fallback
    """
    out: List[str] = []
    ce = sanitize_api_query(claim_en)
    if ce:
        out.append(ce)

    short = sanitize_api_query(" ".join(claim_en.split()[:9]))
    if short and short not in out:
        out.append(short)

    if nlp is not None and claim_en.strip():
        try:
            doc = nlp(claim_en)
            ents = []
            for ent in doc.ents:
                if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "EVENT"}:
                    t = sanitize_api_query(ent.text, max_len=60)
                    if t and t not in ents:
                        ents.append(t)
            if ents:
                # a compact entity query tends to hit APIs well
                ent_q = sanitize_api_query(" ".join(ents[:4]), max_len=120)
                if ent_q and ent_q not in out:
                    out.append(ent_q)
        except Exception:
            pass

    return out[:4]


# ===========================
# EVIDENCE DATA STRUCT
# ===========================
@dataclass
class EvidenceItem:
    title: str
    url: str
    source: str
    desc: str = ""
    published: str = ""
    relevance: float = 0.0
    f_ent: float = 0.0
    f_con: float = 0.0
    f_neu: float = 0.0
    weighted_con_used: float = 0.0
    rel_gate: float = 0.0
    nli_label: str = "NEUTRAL"


# ===========================
# EVIDENCE FETCHERS
# ===========================
def google_news_rss(query: str, hl: str = "en-IN", gl: str = "IN", ceid: str = "IN:en") -> Tuple[List[EvidenceItem], str]:
    if feedparser is None:
        return [], "feedparser not installed (pip install feedparser)"
    q = requests.utils.quote(query)
    url = f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"
    txt, err = safe_get(url)
    if err:
        return [], err
    feed = feedparser.parse(txt)
    items: List[EvidenceItem] = []
    for e in feed.entries[:RSS_PER_QUERY]:
        title = normalize_text(getattr(e, "title", "") or "")
        link = getattr(e, "link", "") or ""
        desc = strip_html_garbage(getattr(e, "summary", "") or "")
        published = normalize_text(getattr(e, "published", "") or "")
        if title and link:
            items.append(EvidenceItem(title=title, url=link, source="GoogleNewsRSS", desc=desc, published=published))
    return items, ""

def fetch_newsapi_org(query: str, api_key: str) -> Tuple[List[EvidenceItem], str]:
    q = sanitize_api_query(query)
    url = (
        "https://newsapi.org/v2/everything?"
        f"q={requests.utils.quote(q)}&language=en&pageSize=20&sortBy=publishedAt&apiKey={api_key}"
    )
    res, err = safe_get_json(url)
    if err:
        return [], err
    items: List[EvidenceItem] = []
    for a in res.get("articles", [])[:20]:
        title = normalize_text(a.get("title", "") or "")
        link = a.get("url", "") or ""
        desc = strip_html_garbage(a.get("description", "") or "")
        if title and link:
            items.append(EvidenceItem(title=title, url=link, source="NewsAPI.org", desc=desc))
    return items, ""

def fetch_gnews(query: str, api_key: str) -> Tuple[List[EvidenceItem], str]:
    q = sanitize_api_query(query)
    url = f"https://gnews.io/api/v4/search?q={requests.utils.quote(q)}&lang=en&max=20&token={api_key}"
    res, err = safe_get_json(url)
    if err:
        return [], err
    items: List[EvidenceItem] = []
    for a in res.get("articles", [])[:20]:
        title = normalize_text(a.get("title", "") or "")
        link = a.get("url", "") or ""
        desc = strip_html_garbage(a.get("description", "") or "")
        if title and link:
            items.append(EvidenceItem(title=title, url=link, source="GNews(en)", desc=desc))
    return items, ""

def fetch_newsdata(query: str, api_key: str) -> Tuple[List[EvidenceItem], str]:
    q = sanitize_api_query(query)
    url = f"https://newsdata.io/api/1/news?q={requests.utils.quote(q)}&language=en&apikey={api_key}"
    res, err = safe_get_json(url)
    if err:
        return [], err
    items: List[EvidenceItem] = []
    for a in res.get("results", [])[:20]:
        title = normalize_text(a.get("title", "") or "")
        link = a.get("link", "") or ""
        desc = strip_html_garbage(a.get("description", "") or "")
        if title and link:
            items.append(EvidenceItem(title=title, url=link, source="NewsData.io", desc=desc))
    return items, ""

def fetch_eventregistry(query: str, api_key: str) -> Tuple[List[EvidenceItem], str]:
    q = sanitize_api_query(query)
    url = "https://eventregistry.org/api/v1/article/getArticles"
    payload = {"action": "getArticles", "keyword": q, "lang": "eng", "articlesCount": 20, "apiKey": api_key}
    res, err = safe_post_json(url, payload)
    if err:
        return [], err
    items: List[EvidenceItem] = []
    for a in res.get("articles", {}).get("results", [])[:20]:
        title = normalize_text(a.get("title", "") or "")
        link = a.get("url", "") or ""
        desc = strip_html_garbage(a.get("body", "") or "")[:240]
        if title and link:
            items.append(EvidenceItem(title=title, url=link, source="EventRegistry(en)", desc=desc))
    return items, ""

def dedup_items(items: List[EvidenceItem]) -> List[EvidenceItem]:
    seen = set()
    out = []
    for it in items:
        t = normalize_text(it.title).lower()
        u = (it.url or "").strip()
        if not t:
            continue
        key = (t, u[:80])
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


# ===========================
# QUERY BUILDER
# ===========================
def build_queries(original_text: str, claim_en: str) -> List[str]:
    original_text = normalize_text(original_text)
    claim_en = normalize_text(claim_en)
    queries: List[str] = []
    if len(original_text) >= 5:
        queries.append(original_text)
    if len(claim_en) >= 5 and claim_en.lower() != original_text.lower():
        queries.append(claim_en)
    queries.append(" ".join(claim_en.split()[:9]))  # short fallback
    out, seen = [], set()
    for q in queries:
        qn = q.lower().strip()
        if len(qn) < 4:
            continue
        if qn not in seen:
            out.append(q.strip())
            seen.add(qn)
    return out[:5]


# ===========================
# RANKING + CONSENSUS + NLI
# ===========================
def rank_by_relevance(items: List[EvidenceItem], claim_en: str, labse: SentenceTransformer) -> List[EvidenceItem]:
    claim_emb = labse.encode(claim_en, convert_to_tensor=True)
    for it in items:
        text = normalize_text((it.title + " " + (it.desc or "")).strip())
        emb = labse.encode(text, convert_to_tensor=True)
        it.relevance = float(util.cos_sim(claim_emb, emb).item())
    items.sort(key=lambda x: x.relevance, reverse=True)
    return items

def update_style_rate(items: List[EvidenceItem]) -> float:
    pat = re.compile(r"\b(death toll|toll|rises|rising|latest|update|updates|after|amid|live)\b", re.I)
    top = items[: min(TOP_K_NLI, len(items))]
    if not top:
        return 0.0
    return sum(1 for it in top if pat.search(it.title)) / len(top)

def consensus_context(items: List[EvidenceItem], claim_en: str, nlp) -> Tuple[List[str], List[str], float]:
    top = items[: min(12, len(items))]
    if not top:
        return [], [], 0.0

    loc_counts: Dict[str, int] = {}
    if nlp is not None:
        for it in top:
            doc = nlp(normalize_text(it.title))
            for ent in doc.ents:
                if ent.label_ in {"GPE", "LOC"}:
                    k = ent.text.strip().lower()
                    if len(k) >= 3:
                        loc_counts[k] = loc_counts.get(k, 0) + 1
    repeated_locations = [k for k, c in sorted(loc_counts.items(), key=lambda x: -x[1]) if c >= 2][:5]

    word_counts: Dict[str, int] = {}
    for it in top:
        toks = keyword_set_en(it.title + " " + (it.desc or ""))
        for t in toks:
            word_counts[t] = word_counts.get(t, 0) + 1
    repeated_words = [w for w, c in sorted(word_counts.items(), key=lambda x: -x[1]) if c >= 3][:10]

    claim_tokens = keyword_set_en(claim_en)
    missing = [w for w in repeated_words if w not in claim_tokens]
    mismatch_rate = (len(missing) / max(1, len(repeated_words))) if repeated_words else 0.0
    return repeated_locations, repeated_words, float(mismatch_rate)

def aggregate_topk_nli(
    ranked: List[EvidenceItem],
    claim_en: str,
    device,
    nli_tok,
    nli_model,
) -> Tuple[float, float, List[EvidenceItem], float]:
    scored: List[EvidenceItem] = []
    max_ent = 0.0
    max_con_used = 0.0

    top_for_body = ranked[: min(12, len(ranked))]
    body_success_rate = 0.0
    if top_for_body:
        body_success_rate = sum(1 for it in top_for_body if (it.desc or "").strip()) / len(top_for_body)

    top = ranked[: min(TOP_K_NLI, len(ranked))]
    for it in top:
        if it.relevance < MIN_REL_FOR_NLI:
            continue

        premise = normalize_text((it.title + ". " + (it.desc or "")).strip())
        premise = strip_html_garbage(premise)
        if len(premise) > 280:
            premise = premise[:280]
        if not premise:
            continue

        f_ent, f_con, f_neu = nli_probs(premise, claim_en, device, nli_tok, nli_model)

        w_con = f_con * (1.0 - f_ent)
        g_rel = rel_gate(it.relevance)
        w_con_used = w_con * g_rel

        it.f_ent, it.f_con, it.f_neu = f_ent, f_con, f_neu
        it.rel_gate = float(g_rel)
        it.weighted_con_used = float(w_con_used)

        if f_ent >= f_con and f_ent >= f_neu:
            it.nli_label = "ENTAILS"
        elif f_con >= f_neu:
            it.nli_label = "CONTRADICTS"
        else:
            it.nli_label = "NEUTRAL"

        max_ent = max(max_ent, f_ent)
        max_con_used = max(max_con_used, float(w_con_used))
        scored.append(it)

    scored.sort(key=lambda x: (x.f_ent, x.relevance), reverse=True)
    return float(max_ent), float(max_con_used), scored, float(body_success_rate)


# ===========================
# GATES (optional)
# ===========================
def load_json_if_exists(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def load_gate(path: str) -> object:
    obj = joblib.load(path)
    if hasattr(obj, "predict_proba"):
        return obj
    if isinstance(obj, dict):
        for k in ["model", "pipeline", "clf", "estimator"]:
            if k in obj and hasattr(obj[k], "predict_proba"):
                return obj[k]
    raise TypeError(f"Gate at {path} is not a sklearn estimator with predict_proba")

def get_gate_feature_list(meta: dict, fallback: List[str]) -> List[str]:
    for k in ["required_features", "feature_names", "features", "cols"]:
        if k in meta and isinstance(meta[k], list) and len(meta[k]) > 0:
            return list(meta[k])
    return list(fallback)

def gate_predict_proba_safe(gate_model, feat_dict: Dict[str, float], feat_list: List[str]) -> Tuple[float, str]:
    try:
        X = pd.DataFrame([{k: float(feat_dict.get(k, 0.0)) for k in feat_list}], columns=feat_list)
        p = float(gate_model.predict_proba(X)[0][1])
        return p, "ok"
    except Exception as e:
        return 0.0, f"gate_error: {e}"


# ===========================
# TF-IDF Style model (optional)
# ===========================
def load_tfidf(path: str):
    return joblib.load(path)


# ===========================
# SIDEBAR
# ===========================
st.sidebar.markdown("### Input")
uploaded_image = st.sidebar.file_uploader("Upload image (optional)", type=["jpg", "jpeg", "png"])
news_text = st.sidebar.text_area("Enter headline / claim (any language)", height=120)
show_debug = st.sidebar.checkbox("Show debug details", value=False)

with st.sidebar.expander("Run command", expanded=False):
    st.code("streamlit run app_streamlit_final_v4_ui_polished.py", language="bash")


# ===========================
# LOAD RESOURCES
# ===========================
device, clip_model, clip_processor, labse, nllb_tok, nllb_model, nli_tok, nli_model = load_models()
nlp, spacy_name = load_spacy()

support_gate = None
contra_gate = None
support_meta = {}
contra_meta = {}
tfidf_model = None

if os.path.exists("support_gate.joblib"):
    try:
        support_gate = load_gate("support_gate.joblib")
    except Exception as e:
        st.error(f"Failed to load support_gate.joblib: {e}")

if os.path.exists("contradiction_gate.joblib"):
    try:
        contra_gate = load_gate("contradiction_gate.joblib")
    except Exception as e:
        st.error(f"Failed to load contradiction_gate.joblib: {e}")

support_meta = load_json_if_exists("support_gate_meta.json")
contra_meta = load_json_if_exists("contradiction_gate_meta.json")

support_thr = float(support_meta.get("best_threshold", DEFAULT_SUPPORT_THR))
contra_thr = float(contra_meta.get("best_threshold", DEFAULT_CONTRA_THR))

try:
    if os.path.exists("tfidf_style_model.joblib"):
        tfidf_model = load_tfidf("tfidf_style_model.joblib")
except Exception as e:
    st.error(f"Failed to load tfidf_style_model.joblib: {e}")

deepfake_model, deepfake_status = load_deepfake_model()
rek_client, rek_status, rek_debug = load_rekognition_client()


# ===========================
# DISPLAY IMAGE (right column)
# ===========================
image = None
if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")


# ===========================
# MAIN
# ===========================
if news_text and len(news_text.strip()) > 0:
    T_raw = normalize_text(news_text)
    lang = detect_language(T_raw)
    claim_en = translate_to_english(T_raw, lang, device, nllb_tok, nllb_model)

    # ---------- Top layout: Claim (left) + Image (right)
    left, right = st.columns([1.25, 0.9], gap="large")

    with left:
        st.markdown("## Claim")
        st.markdown(f"<div class='smalllabel'>Original</div><div class='card'><b>{T_raw}</b></div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='margin-top:10px;'>"
            f"<span class='badge'><strong>Detected language:</strong> {lang}</span>"
            f"<span class='badge'><strong>spaCy:</strong> {spacy_name}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='margin-top:10px;' class='card'>"
            f"<div class='smalllabel'>English (for retrieval/NLI)</div>"
            f"<div><b>{claim_en}</b></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("## Image (optional)")
        if image is not None:
            st.image(image, caption="Uploaded image", use_container_width=True)
        else:
            st.markdown("<div class='card'>No image uploaded.</div>", unsafe_allow_html=True)

    # ---------- Debug: secrets presence
    if show_debug:
        st.sidebar.markdown("### Secrets debug (safe)")
        try:
            st.sidebar.write("NEWSAPI_KEY loaded:", bool(st.secrets.get("NEWSAPI_KEY", "")))
            st.sidebar.write("GNEWS_KEY loaded:", bool(st.secrets.get("GNEWS_KEY", "")))
            st.sidebar.write("NEWSDATA_KEY loaded:", bool(st.secrets.get("NEWSDATA_KEY", "")))
            st.sidebar.write("EVENTREGISTRY_KEY loaded:", bool(st.secrets.get("EVENTREGISTRY_KEY", "")))
            st.sidebar.write("AWS_ACCESS_KEY_ID loaded:", bool(st.secrets.get("AWS_ACCESS_KEY_ID", "")))
            st.sidebar.write("AWS_SECRET_ACCESS_KEY loaded:", bool(st.secrets.get("AWS_SECRET_ACCESS_KEY", "")))
            st.sidebar.write("AWS_REGION:", st.secrets.get("AWS_REGION", "MISSING"))
        except Exception as e:
            st.sidebar.error(f"st.secrets read error: {e}")

    # ===========================
    # VISUAL SIGNALS (compute first)
    # ===========================
    g_clip = 0.0
    clip_note = "No image provided."
    df_label = "n/a"
    df_conf = None
    df_p_real = None
    df_note = "No image provided."
    celeb_name = None
    celeb_conf = 0.0
    rek_note = "No image provided."
    faces_n = 0
    mismatch_msg = ""

    if image is not None:
        # CLIP
        _, g_clip = clip_gate_score(image, claim_en, device, clip_model, clip_processor)
        if g_clip >= 0.65:
            clip_note = "High alignment"
        elif g_clip <= 0.35:
            clip_note = "Low alignment"
        else:
            clip_note = "Moderate alignment"

        # Deepfake CNN
        if deepfake_model is None:
            df_note = f"Unavailable: {deepfake_status}"
        else:
            df_p_real, df_label, df_conf = deepfake_predict_pil(image, deepfake_model)
            df_note = "OK"

        # Rekognition
        if rek_client is None:
            rek_note = f"Unavailable: {rek_status}"
        else:
            celeb_name, celeb_conf, rek_note = rekognition_detect_celebrity_from_image(image, rek_client)
            if celeb_name is None:
                faces_n, faces_note = rekognition_detect_faces_count(image, rek_client)
                rek_note = f"{rek_note} | Faces detected: {faces_n} ({faces_note})"
            else:
                claim_person = extract_person_from_claim(claim_en, nlp)
                if claim_person:
                    same = celeb_name.lower() in claim_person.lower() or claim_person.lower() in celeb_name.lower()
                    if (not same) and celeb_conf >= 90.0:
                        mismatch_msg = (
                            f"Possible mismatch: claim mentions “{claim_person}”, "
                            f"but image matches “{celeb_name}” with high confidence."
                        )

    # ===========================
    # LANGUAGE STYLE HINT (TF-IDF)
    # ===========================
    p_real = None
    p_fake = None
    language_hint = "n/a"
    if tfidf_model is not None:
        try:
            pr = tfidf_model.predict_proba([T_raw])[0]
            p_real = float(pr[0])
            p_fake = float(pr[1])

            if (p_fake >= LIKELY_THR) and (p_fake - p_real >= NEUTRAL_BAND):
                language_hint = "Likely False (language patterns)"
            elif (p_real >= LIKELY_THR) and (p_real - p_fake >= NEUTRAL_BAND):
                language_hint = "Likely True (language patterns)"
            else:
                language_hint = "Neutral / inconclusive (language patterns)"
        except Exception:
            p_real = p_fake = None
            language_hint = "n/a"

    # ===========================
    # FETCH EVIDENCE (APIs + RSS)
    # ===========================
    keys = {}
    try:
        keys = {
            "NEWSAPI_KEY": st.secrets.get("NEWSAPI_KEY", ""),
            "GNEWS_KEY": st.secrets.get("GNEWS_KEY", ""),
            "NEWSDATA_KEY": st.secrets.get("NEWSDATA_KEY", ""),
            "EVENTREGISTRY_KEY": st.secrets.get("EVENTREGISTRY_KEY", ""),
        }
        keys = {k: v for k, v in keys.items() if v}
    except Exception:
        keys = {}

    # RSS queries (can include original language)
    rss_queries = build_queries(T_raw, claim_en)

    # API queries (more robust): claim_en + short + entities
    api_queries = extract_keywords_for_api(claim_en, nlp)

    source_status: Dict[str, str] = {}
    items: List[EvidenceItem] = []

    with st.spinner("Fetching evidence (APIs + RSS)..."):
        # APIs
        if keys.get("NEWSAPI_KEY"):
            any_ok = False
            last_err = ""
            for q in api_queries:
                its, err = fetch_newsapi_org(q, keys["NEWSAPI_KEY"])
                if err:
                    last_err = err
                else:
                    any_ok = True
                    items.extend(its)
            source_status["NewsAPI.org"] = "OK" if any_ok else (last_err or "No results")

        if keys.get("GNEWS_KEY"):
            any_ok = False
            last_err = ""
            for q in api_queries:
                its, err = fetch_gnews(q, keys["GNEWS_KEY"])
                if err:
                    last_err = err
                else:
                    any_ok = True
                    items.extend(its)
            source_status["GNews"] = "OK" if any_ok else (last_err or "No results")

        if keys.get("NEWSDATA_KEY"):
            any_ok = False
            last_err = ""
            for q in api_queries:
                its, err = fetch_newsdata(q, keys["NEWSDATA_KEY"])
                if err:
                    last_err = err
                else:
                    any_ok = True
                    items.extend(its)
            source_status["NewsData.io"] = "OK" if any_ok else (last_err or "No results")

        if keys.get("EVENTREGISTRY_KEY"):
            any_ok = False
            last_err = ""
            for q in api_queries:
                its, err = fetch_eventregistry(q, keys["EVENTREGISTRY_KEY"])
                if err:
                    last_err = err
                else:
                    any_ok = True
                    items.extend(its)
            source_status["EventRegistry"] = "OK" if any_ok else (last_err or "No results")

        # RSS (always)
        rss_any_ok = False
        last_err = ""
        for q in rss_queries:
            its, err = google_news_rss(q, hl="en-IN", gl="IN", ceid=CEID_MAP["en"])
            if err:
                last_err = err
            else:
                rss_any_ok = True
                items.extend(its)
        source_status["GoogleNewsRSS(en-IN)"] = "OK" if rss_any_ok else (last_err or "No results")

        # Language-biased RSS (correct ceid)
        if lang in {"hi", "kn", "te"}:
            hl_map = {"hi": "hi-IN", "kn": "kn-IN", "te": "te-IN"}
            hl = hl_map.get(lang, "en-IN")
            ceid_local = CEID_MAP.get(lang, "IN:en")

            rss2_any_ok = False
            last_err2 = ""
            for q in rss_queries:
                its, err = google_news_rss(q, hl=hl, gl="IN", ceid=ceid_local)
                if err:
                    last_err2 = err
                else:
                    rss2_any_ok = True
                    items.extend(its)
            source_status[f"GoogleNewsRSS({hl})"] = "OK" if rss2_any_ok else (last_err2 or "No results")

    # Dedup + cap
    items = dedup_items(items)
    items = items[:MAX_ITEMS_FETCH]

    # If nothing was fetched, show a neutral callout and stop
    if not items:
        st.markdown(
            "<div class='callout'><p><b>No evidence items were found.</b> "
            "This usually happens if APIs returned no matches for the translated query or the topic is very local/recent. "
            "Try a shorter claim (fewer words), or include a key entity name (place/person).</p></div>",
            unsafe_allow_html=True,
        )
        if show_debug:
            st.markdown("### Source status (debug)")
            st.json(source_status)
            st.write("API queries:", api_queries)
            st.write("RSS queries:", rss_queries)
        st.stop()

    # ===========================
    # RANK + CONSENSUS + NLI
    # ===========================
    ranked = rank_by_relevance(items, claim_en, labse)
    rep_locs, rep_words, ctx_mismatch_rate = consensus_context(ranked, claim_en, nlp)
    max_ent, max_con_used, scored, body_success_rate = aggregate_topk_nli(
        ranked, claim_en, device, nli_tok, nli_model
    )

    # Relevance stats
    rels = [it.relevance for it in ranked[: min(30, len(ranked))]]
    max_rel = float(max(rels)) if rels else 0.0
    mean_rel = float(np.mean(rels)) if rels else 0.0
    p90_rel = float(np.percentile(rels, 90)) if rels else 0.0

    ents = [it.f_ent for it in scored] if scored else [0.0]
    cons = [it.weighted_con_used for it in scored] if scored else [0.0]
    mean_ent = float(np.mean(ents)) if ents else 0.0
    mean_con = float(np.mean(cons)) if cons else 0.0
    upd_rate = float(update_style_rate(ranked))

    feats = {
        "num_evidence": float(len(ranked)),
        "max_rel": max_rel,
        "mean_rel": mean_rel,
        "p90_rel": p90_rel,
        "max_ent": float(max_ent),
        "mean_ent": mean_ent,
        "max_con_used": float(max_con_used),
        "mean_con": mean_con,
        "body_success_rate": float(body_success_rate),
        "context_mismatch_rate": float(ctx_mismatch_rate),
        "update_style_rate": float(upd_rate),
    }

    # Gate predictions
    support_p = 0.0
    contra_p = 0.0
    support_note = ""
    contra_note = ""

    if support_gate is not None:
        support_feats = get_gate_feature_list(support_meta, FEATURE_COLUMNS)
        support_p, support_note = gate_predict_proba_safe(support_gate, feats, support_feats)

    if contra_gate is not None:
        contra_feats = get_gate_feature_list(contra_meta, FEATURE_COLUMNS)
        contra_p, contra_note = gate_predict_proba_safe(contra_gate, feats, contra_feats)

    gate_failed = ("gate_error" in support_note) or ("gate_error" in contra_note)
    fallback_supported = (max_ent >= 0.80 and max_rel >= 0.45 and max_con_used < 0.15)
    fallback_contradicted = (max_con_used >= 0.35 and max_rel >= 0.45)

    if gate_failed:
        if fallback_contradicted:
            decision = "CONTRADICTED"
        elif fallback_supported:
            decision = "SUPPORTED"
        else:
            decision = "UNVERIFIED"
    else:
        if contra_p >= contra_thr:
            decision = "CONTRADICTED"
        elif support_p >= support_thr:
            decision = "SUPPORTED"
        else:
            decision = "UNVERIFIED"

    # ===========================
    # PLACE VERDICT + VISUAL SIGNALS UNDER CLAIM (the "blank space")
    # ===========================
    st.markdown("## Verdict & Signals")

    vcol, scol = st.columns([0.6, 1.4], gap="large")

    with vcol:
        st.markdown(
            f"""
<div class="verdict">
  <div class="k">Final verdict</div>
  <p class="v">{decision}</p>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
<div class="card" style="margin-top:12px;">
  <div class="smalllabel">Evidence strength (max entail)</div>
  <div style="font-size:1.25rem; font-weight:650;">{max_ent:.2f}</div>
</div>
""",
            unsafe_allow_html=True,
        )

        if p_fake is not None:
            st.markdown(
                f"""
<div class="card" style="margin-top:12px;">
  <div class="smalllabel">Language hint</div>
  <div style="font-weight:650;">{language_hint}</div>
  <div class="smalllabel" style="margin-top:6px;">Language probs</div>
  <div style="opacity:0.85;">P(real-style)={p_real:.3f} · P(fake-style)={p_fake:.3f}</div>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
<div class="card" style="margin-top:12px;">
  <div class="smalllabel">Language hint</div>
  <div style="opacity:0.85;">n/a</div>
</div>
""",
                unsafe_allow_html=True,
            )

    with scol:
        # Visual signals (all in one place, same size)
        # Rekognition display text
        rek_display = "Not run"
        if image is None:
            rek_display = "No image"
        elif rek_client is None:
            rek_display = rek_note
        else:
            if celeb_name is None:
                rek_display = rek_note
            else:
                rek_display = f"{celeb_name} ({celeb_conf:.1f}%)"

        # Deepfake display text
        df_display = "Not run"
        if image is None:
            df_display = "No image"
        elif deepfake_model is None:
            df_display = df_note
        else:
            df_display = f"{df_label} (conf {df_conf:.2f}, p_real {df_p_real:.2f})"

        st.markdown(
            f"""
<div class="card">
  <div style="font-weight:700; margin-bottom:8px;">Visual signals</div>
  <div>
    <span class="badge"><strong>CLIP alignment:</strong> {g_clip:.3f} · {clip_note}</span>
  </div>
  <div style="margin-top:8px;">
    <span class="badge"><strong>Deepfake CNN:</strong> {df_display}</span>
  </div>
  <div style="margin-top:8px;">
    <span class="badge"><strong>AWS Rekognition:</strong> {rek_display}</span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        if mismatch_msg:
            st.markdown(
                f"<div class='callout' style='margin-top:12px;'><p><b>{mismatch_msg}</b></p></div>",
                unsafe_allow_html=True,
            )

    # ===========================
    # EXPLANATION
    # ===========================
    st.markdown("## Explanation")

    expl: List[str] = []
    if decision == "SUPPORTED":
        expl.append("Multiple relevant sources align with the claim, so it is marked supported.")
    elif decision == "CONTRADICTED":
        expl.append("Relevant sources contain conflicting information, so it is marked contradicted.")
    else:
        expl.append("Evidence is insufficient to strongly support or contradict the claim, so it is marked unverified.")

    if decision == "UNVERIFIED" and p_fake is not None:
        if "Likely True" in language_hint:
            expl.append("Although evidence is insufficient, the headline language looks closer to normal reporting (likely true by style).")
        elif "Likely False" in language_hint:
            expl.append("Although evidence is insufficient, the headline language shows misinformation-like patterns (likely false by style).")
        else:
            expl.append("Language patterns are not decisive for this headline.")

    if ctx_mismatch_rate >= 0.50 and rep_words:
        expl.append("Evidence contains repeated context details not present in the claim, which reduces verification confidence.")

    if rep_locs:
        expl.append("Consensus locations from evidence: " + ", ".join(rep_locs))
    if rep_words:
        expl.append("Repeated context keywords in evidence: " + ", ".join(rep_words))

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    for line in expl:
        st.markdown(f"- {line}")
    st.markdown("</div>", unsafe_allow_html=True)

    # ===========================
    # EVIDENCE
    # ===========================
    st.markdown("## Evidence")
    shown = scored[: min(10, len(scored))] if scored else ranked[: min(10, len(ranked))]

    for it in shown:
        desc_line = (it.desc or "").strip()
        if len(desc_line) > 240:
            desc_line = desc_line[:240] + "..."

        st.markdown(
            f"""
<div class="evidence">
  <div class="title">{it.title}</div>
  <div class="meta">Source: {it.source} · Relevance: {it.relevance:.3f} · NLI: {it.nli_label} (ent={it.f_ent:.2f}, con={it.f_con:.2f}, neu={it.f_neu:.2f}) · Contradiction used: {it.weighted_con_used:.2f}</div>
  <div class="desc">{desc_line}</div>
  <div style="margin-top:8px;"><a href="{it.url}" target="_blank">Read article</a></div>
</div>
""",
            unsafe_allow_html=True,
        )

    # ===========================
    # SOURCE STATUS (kept, but not noisy)
    # ===========================
    if show_debug:
        st.markdown("## Debug")
        st.markdown("### Source status")
        st.json(source_status)
        st.write("API queries:", api_queries)
        st.write("RSS queries:", rss_queries)
        st.write("Features:", feats)
        st.write("Gates:", {"support_p": support_p, "contra_p": contra_p, "support_note": support_note, "contra_note": contra_note})
        st.write("Rekognition debug:", rek_debug)

st.markdown("---")
st.caption("Decision-support system only. Outputs are based on retrieved headlines/snippets, NLI, and learned gating; verify using full articles when needed.")

