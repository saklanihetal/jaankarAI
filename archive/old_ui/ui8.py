# app_streamlit_final_v7_cpu_optimized.py
# V7 (Best for Laptop): CPU-Optimized + Interactive UI
# - Replaced NLLB (GPU heavy) with Google Translate (Cloud API via deep_translator)
# - Solves "One vs Many" translation errors
# - Adds "Progressive Disclosure" UI (live logs, instant query display)
# - Removes heavy model loading for translation to save RAM

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

# --- NEW: Google Translate Library ---
try:
    from deep_translator import GoogleTranslator
except ImportError:
    GoogleTranslator = None

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
h1 { font-size: 2.45rem !important; margin-bottom: 0.25rem !important; }
h2 { font-size: 1.30rem !important; margin-top: 1.25rem !important; }
h3 { font-size: 1.08rem !important; margin-top: 1.0rem !important; }

/* Body sizes */
p, li, div, span { font-size: 0.99rem; line-height: 1.55; }

/* Section label */
.smalllabel {
  font-size: 0.84rem;
  opacity: 0.72;
  margin-bottom: 0.2rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;
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
  padding: 20px 24px;
  border: 1px solid rgba(255,255,255,0.08);
  background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
}
.verdict .k { font-size: 0.84rem; opacity: 0.75; margin-bottom: 0.25rem; text-transform: uppercase; }
.verdict .v { font-size: 2.0rem; font-weight: 800; letter-spacing: 0.3px; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2); }

/* Badges */
.badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 0.86rem;
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.04);
  margin-right: 8px;
  margin-bottom: 8px;
}
.badge strong { font-weight: 750; }

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
  padding: 13px 13px;
  margin-bottom: 10px;
  transition: transform 0.2s;
}
.evidence:hover {
    background: rgba(255,255,255,0.04);
}
.evidence .title { font-weight: 700; font-size: 1.05rem; margin-bottom: 0.35rem; color: #fff; }
.evidence .meta { opacity: 0.78; font-size: 0.92rem; }
.evidence .desc { margin-top: 0.40rem; font-size: 1.00rem; opacity: 0.9; }

/* Sidebar subtle */
[data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,0.06); }

/* Streamlit alerts less shouty */
div[data-testid="stAlert"] { border-radius: 14px; }
</style>
""",
    unsafe_allow_html=True,
)

# ===========================
# CONSTANTS
# ===========================
TOP_K_NLI = 5
MAX_ITEMS_FETCH = 140
RSS_PER_QUERY = 14
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

def keyword_set_en(text: str) -> List[str]:
    text = normalize_text(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    toks = [t for t in text.split() if len(t) > 2 and t not in _EN_STOP]
    return toks

def rel_gate(relevance: float, r0: float = REL_GATE_R0, r1: float = REL_GATE_R1) -> float:
    if r1 <= r0:
        return 1.0
    return clamp01((relevance - r0) / (r1 - r0))

def sanitize_api_query(q: str, max_len: int = 120) -> str:
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

def ui_callout(msg: str):
    st.markdown(f"<div class='callout'><p>{msg}</p></div>", unsafe_allow_html=True)


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
    """
    Loads core models. NLLB (Translation) has been removed to use Google API instead.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_name)

    labse = SentenceTransformer("sentence-transformers/LaBSE")

    nli_name = "facebook/bart-large-mnli"
    nli_tokenizer = AutoTokenizer.from_pretrained(nli_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_name).to(device)

    return device, clip_model, clip_processor, labse, nli_tokenizer, nli_model

def translate_to_english(text: str) -> str:
    """
    Uses Google Translate (via deep_translator) for CPU-friendly translation.
    """
    text = normalize_text(text)
    if not text:
        return ""
    
    if GoogleTranslator is None:
        return text  # Fallback if library missing
        
    try:
        # 'auto' lets Google detect source lang
        translator = GoogleTranslator(source='auto', target='en')
        return translator.translate(text)
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
        return None, "TensorFlow not installed"
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
            return None, f"Load failed: {e1}"

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
# AWS Rekognition
# ===========================
@st.cache_resource
def load_rekognition_client():
    debug = {"boto3_installed": bool(boto3), "keys_present": False, "region": None}
    if boto3 is None:
        return None, "boto3 not installed", debug

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
        resp = client.recognize_celebrities(Image={"Bytes": buf.getvalue()})
        faces = resp.get("CelebrityFaces", []) or []
        if not faces:
            return None, 0.0, "No celebrity detected"
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
        return [], "feedparser not installed"
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
        key = (t, u[:90])
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


# ===========================
# QUERY BUILDERS (RSS + APIs)
# ===========================
def build_rss_queries(original_text: str, claim_en: str) -> List[str]:
    original_text = normalize_text(original_text)
    claim_en = normalize_text(claim_en)

    queries = []
    if len(original_text) >= 5:
        queries.append(original_text)
    if len(claim_en) >= 5 and claim_en.lower() != original_text.lower():
        queries.append(claim_en)
    queries.append(" ".join(claim_en.split()[:9]))
    queries.append(" ".join(original_text.split()[:9]))

    out, seen = [], set()
    for q in queries:
        qn = q.lower().strip()
        if len(qn) < 4:
            continue
        if qn not in seen:
            out.append(q.strip())
            seen.add(qn)
    return out[:6]

def _ngrams(tokens: List[str], n: int) -> List[str]:
    if len(tokens) < n:
        return []
    out = []
    for i in range(len(tokens) - n + 1):
        out.append(" ".join(tokens[i:i+n]))
    return out

def build_api_queries(claim_en: str, nlp) -> List[str]:
    claim_en = normalize_text(claim_en)
    base_tokens = keyword_set_en(claim_en)
    base_tokens = [t for t in base_tokens if t not in _EN_STOP]
    base_tokens = base_tokens[:18]

    queries: List[str] = []

    # 1) Full
    q_full = sanitize_api_query(claim_en, max_len=120)
    if q_full:
        queries.append(q_full)

    # 2) Short
    q_short = sanitize_api_query(" ".join(claim_en.split()[:9]), max_len=90)
    if q_short and q_short not in queries:
        queries.append(q_short)

    # 3) Entities
    ents: List[str] = []
    if nlp is not None and claim_en.strip():
        try:
            doc = nlp(claim_en)
            for ent in doc.ents:
                if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "EVENT", "NORP"}:
                    t = sanitize_api_query(ent.text, max_len=55)
                    if t and t not in ents:
                        ents.append(t)
        except Exception:
            ents = []

    if ents:
        ent_q = sanitize_api_query(" ".join(ents[:4]), max_len=90)
        if ent_q and ent_q not in queries:
            queries.append(ent_q)

        signal = []
        for t in base_tokens:
            if t in {"retire", "retired", "retirement", "odi", "test", "t20", "election", "blast", "explosion",
                     "killed", "death", "dead", "attack", "arrested", "budget", "ban", "released"}:
                signal.append(t)
        if not signal:
            signal = base_tokens[:4]

        for ent in ents[:2]:
            combo = sanitize_api_query(" ".join([ent] + signal[:4]), max_len=100)
            if combo and combo not in queries:
                queries.append(combo)

    # 4) N-grams
    bigrams = _ngrams(base_tokens, 2)[:6]
    trigrams = _ngrams(base_tokens, 3)[:4]
    for ng in (trigrams + bigrams):
        q = sanitize_api_query(ng, max_len=90)
        if q and q not in queries:
            queries.append(q)

    # 5) Compact keywords
    kw_q = sanitize_api_query(" ".join(base_tokens[:7]), max_len=90)
    if kw_q and kw_q not in queries:
        queries.append(kw_q)

    return queries[:7]


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
            try:
                doc = nlp(normalize_text(it.title))
                for ent in doc.ents:
                    if ent.label_ in {"GPE", "LOC"}:
                        k = ent.text.strip().lower()
                        if len(k) >= 3:
                            loc_counts[k] = loc_counts.get(k, 0) + 1
            except Exception:
                pass
    repeated_locations = [k for k, c in sorted(loc_counts.items(), key=lambda x: -x[1]) if c >= 2][:5]

    word_counts: Dict[str, int] = {}
    for it in top:
        toks = set(keyword_set_en(it.title + " " + (it.desc or "")))
        for t in toks:
            word_counts[t] = word_counts.get(t, 0) + 1
    repeated_words = [w for w, c in sorted(word_counts.items(), key=lambda x: -x[1]) if c >= 3][:10]

    claim_tokens = set(keyword_set_en(claim_en))
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
# GATES
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
    raise TypeError(f"Gate at {path} is not a sklearn estimator")

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

def load_tfidf(path: str):
    return joblib.load(path)


# ===========================
# NAVIGATION
# ===========================
if "page" not in st.session_state:
    st.session_state.page = "Landing"

st.sidebar.markdown("### Navigation")
page_choice = st.sidebar.radio(
    "Go to",
    ["Landing", "Fake News Detection System"],
    index=0 if st.session_state.page == "Landing" else 1,
)
st.session_state.page = page_choice


# ===========================
# LANDING PAGE
# ===========================
def render_landing():
    st.markdown("# AI-based Regional Fake News Detection")
    st.markdown("A decision-support system for multilingual misinformation verification.")
    st.markdown("---")

    c1, c2 = st.columns([1.1, 0.9], gap="large")

    with c1:
        st.markdown(
            """
<div class="card">
  <div class="smalllabel">Project overview</div>
  <div style="font-size:1.06rem; font-weight:700; margin-bottom:6px;">
    Regional Fake News Detection using Evidence Retrieval & Multimodal Signals
  </div>
  <div style="opacity:0.88;">
    • Retrieves supporting/contradicting evidence via RSS + News APIs<br/>
    • Runs NLI to judge claim alignment<br/>
    • Uses visual signals (CLIP, Deepfake CNN, AWS Rekognition)<br/>
    • <b>CPU Optimized Translation (Google API)</b>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
<div class="card">
  <div class="smalllabel">Team</div>
  <div style="font-size:1.06rem; font-weight:750; margin-bottom:8px;">[Your Team Name]</div>
  <div style="opacity:0.88; line-height:1.6;">
    Member 1: [Name]<br/>
    Member 2: [Name]<br/>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    if st.button("Enter System", use_container_width=True):
        st.session_state.page = "Fake News Detection System"
        st.rerun()


# ===========================
# SYSTEM PAGE
# ===========================
@st.cache_resource
def load_all_resources():
    device, clip_model, clip_processor, labse, nli_tok, nli_model = load_models()
    nlp, spacy_name = load_spacy()

    support_gate = None
    contra_gate = None

    if os.path.exists("support_gate.joblib"):
        try:
            support_gate = load_gate("support_gate.joblib")
        except Exception:
            support_gate = None

    if os.path.exists("contradiction_gate.joblib"):
        try:
            contra_gate = load_gate("contradiction_gate.joblib")
        except Exception:
            contra_gate = None

    support_meta = load_json_if_exists("support_gate_meta.json")
    contra_meta = load_json_if_exists("contradiction_gate_meta.json")

    support_thr = float(support_meta.get("best_threshold", DEFAULT_SUPPORT_THR))
    contra_thr = float(contra_meta.get("best_threshold", DEFAULT_CONTRA_THR))

    tfidf_model = None
    try:
        if os.path.exists("tfidf_style_model.joblib"):
            tfidf_model = load_tfidf("tfidf_style_model.joblib")
    except Exception:
        tfidf_model = None

    deepfake_model, deepfake_status = load_deepfake_model()
    rek_client, rek_status, rek_debug = load_rekognition_client()

    return {
        "device": device,
        "clip_model": clip_model,
        "clip_processor": clip_processor,
        "labse": labse,
        "nli_tok": nli_tok,
        "nli_model": nli_model,
        "nlp": nlp,
        "spacy_name": spacy_name,
        "support_gate": support_gate,
        "contra_gate": contra_gate,
        "support_meta": support_meta,
        "contra_meta": contra_meta,
        "support_thr": support_thr,
        "contra_thr": contra_thr,
        "tfidf_model": tfidf_model,
        "deepfake_model": deepfake_model,
        "deepfake_status": deepfake_status,
        "rek_client": rek_client,
        "rek_status": rek_status,
        "rek_debug": rek_debug,
    }

def render_system():
    # Load resources immediately (cached)
    R = load_all_resources()
    
    device = R["device"]
    clip_model = R["clip_model"]
    clip_processor = R["clip_processor"]
    labse = R["labse"]
    nli_tok = R["nli_tok"]
    nli_model = R["nli_model"]
    nlp = R["nlp"]
    spacy_name = R["spacy_name"]
    support_gate = R["support_gate"]
    contra_gate = R["contra_gate"]
    support_meta = R["support_meta"]
    contra_meta = R["contra_meta"]
    support_thr = R["support_thr"]
    contra_thr = R["contra_thr"]
    tfidf_model = R["tfidf_model"]
    deepfake_model = R["deepfake_model"]
    deepfake_status = R["deepfake_status"]
    rek_client = R["rek_client"]
    rek_status = R["rek_status"]
    rek_debug = R["rek_debug"]

    st.markdown("# AI-based Regional Fake News Detection")
    st.markdown("Decision-support system that cross-checks a claim against retrieved evidence and visual signals.")
    st.markdown("---")

    # Sidebar inputs
    st.sidebar.markdown("### Input")
    uploaded_image = st.sidebar.file_uploader("Upload image (optional)", type=["jpg", "jpeg", "png"])
    news_text = st.sidebar.text_area("Enter headline / claim (any language)", height=120)
    
    # "Run Verification" Button
    run_btn = st.sidebar.button("Verify Claim", type="primary", use_container_width=True)
    show_debug = st.sidebar.checkbox("Show debug details", value=False)
    
    # Check for Deep Translator
    if GoogleTranslator is None:
        st.sidebar.error("Missing library: `pip install deep-translator`")

    if not news_text or not news_text.strip():
        ui_callout("Enter a headline/claim in the sidebar and click <b>Verify Claim</b> to start.")
        return

    # If button not pressed and no state, stop here
    if not run_btn and "last_result" not in st.session_state:
        return

    # ==========================================
    # INTERACTIVE LOADING / PROGRESSIVE UI
    # ==========================================
    
    status_container = st.container()
    results_container = st.container()

    if run_btn:
        with status_container:
            # Expandable status log
            with st.status("Initiating verification pipeline...", expanded=True) as status:
                
                # 1. PREP & TRANSLATE
                st.write("Processing text & detecting language...")
                T_raw = normalize_text(news_text)
                lang = detect_language(T_raw)
                
                # Use Google Translate API here
                claim_en = translate_to_english(T_raw)
                
                # Image processing
                image = None
                if uploaded_image:
                    try:
                        image = Image.open(uploaded_image).convert("RGB")
                    except Exception:
                        image = None
                
                # 2. GENERATE QUERIES (Show immediately)
                st.write(" Generating search queries...")
                rss_queries = build_rss_queries(T_raw, claim_en)
                api_queries = build_api_queries(claim_en, nlp)
                
                st.markdown(f"**Generated Queries:**")
                st.code("\n".join(api_queries[:3]), language="text")

                # 3. FETCH EVIDENCE
                status.write(" Contacting News APIs & RSS Feeds...")
                
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

                items: List[EvidenceItem] = []

                if keys.get("NEWSAPI_KEY"):
                    for q in api_queries:
                        its, err = fetch_newsapi_org(q, keys["NEWSAPI_KEY"])
                        if not err: items.extend(its)
                
                if keys.get("GNEWS_KEY"):
                    for q in api_queries:
                        its, err = fetch_gnews(q, keys["GNEWS_KEY"])
                        if not err: items.extend(its)

                if keys.get("NEWSDATA_KEY"):
                    for q in api_queries:
                        its, err = fetch_newsdata(q, keys["NEWSDATA_KEY"])
                        if not err: items.extend(its)

                if keys.get("EVENTREGISTRY_KEY"):
                    for q in api_queries:
                        its, err = fetch_eventregistry(q, keys["EVENTREGISTRY_KEY"])
                        if not err: items.extend(its)

                # RSS
                for q in rss_queries:
                    its, err = google_news_rss(q, hl="en-IN", gl="IN", ceid=CEID_MAP["en"])
                    if not err: items.extend(its)

                if lang in {"hi", "kn", "te"}:
                    hl_map = {"hi": "hi-IN", "kn": "kn-IN", "te": "te-IN"}
                    hl = hl_map.get(lang, "en-IN")
                    ceid_local = CEID_MAP.get(lang, "IN:en")
                    for q in rss_queries:
                        its, err = google_news_rss(q, hl=hl, gl="IN", ceid=ceid_local)
                        if not err: items.extend(its)

                items = dedup_items(items)
                items = items[:MAX_ITEMS_FETCH]

                st.write(f" Found **{len(items)}** potential evidence articles.")
                
                if items:
                    st.markdown("**Scanning headlines:**")
                    preview_titles = [f"- {it.title}" for it in items[:3]]
                    st.text("\n".join(preview_titles))
                
                # 4. RUN MODELS
                status.write(" Running Natural Language Inference (NLI) & Visual Checks...")
                
                ranked = rank_by_relevance(items, claim_en, labse)
                rep_locs, rep_words, ctx_mismatch_rate = consensus_context(ranked, claim_en, nlp)
                max_ent, max_con_used, scored, body_success_rate = aggregate_topk_nli(
                    ranked, claim_en, device, nli_tok, nli_model
                )
                
                g_clip = 0.0
                clip_note = "No image"
                df_display = "No image"
                rek_display = "No image"
                mismatch_msg = ""
                p_real_style, p_fake_style, language_hint = None, None, "n/a"

                if image is not None:
                    _, g_clip = clip_gate_score(image, claim_en, device, clip_model, clip_processor)
                    if g_clip >= 0.65: clip_note = "High alignment"
                    elif g_clip <= 0.35: clip_note = "Low alignment"
                    else: clip_note = "Moderate alignment"

                    if deepfake_model:
                        df_p, df_l, df_c = deepfake_predict_pil(image, deepfake_model)
                        df_display = f"{df_l} (conf {df_c:.2f})"
                    else:
                        df_display = "Unavailable"

                    if rek_client:
                        c_name, c_conf, _ = rekognition_detect_celebrity_from_image(image, rek_client)
                        if c_name:
                            rek_display = f"{c_name} ({c_conf:.1f}%)"
                            c_pers = extract_person_from_claim(claim_en, nlp)
                            if c_pers and (c_pers.lower() not in c_name.lower()) and c_conf > 90:
                                mismatch_msg = f"Possible mismatch: Claim mentions {c_pers}, image shows {c_name}."
                        else:
                            f_n, _ = rekognition_detect_faces_count(image, rek_client)
                            rek_display = f"Faces: {f_n}"

                if tfidf_model:
                    pr = tfidf_model.predict_proba([T_raw])[0]
                    p_real_style, p_fake_style = float(pr[0]), float(pr[1])
                    if (p_fake_style >= LIKELY_THR) and (p_fake_style - p_real_style >= NEUTRAL_BAND):
                        language_hint = "Likely False (style)"
                    elif (p_real_style >= LIKELY_THR) and (p_real_style - p_fake_style >= NEUTRAL_BAND):
                        language_hint = "Likely True (style)"
                    else:
                        language_hint = "Neutral (style)"

                rels = [it.relevance for it in ranked[:30]]
                max_rel = max(rels) if rels else 0.0
                mean_rel = float(np.mean(rels)) if rels else 0.0
                p90_rel = float(np.percentile(rels, 90)) if rels else 0.0
                ents = [it.f_ent for it in scored] if scored else [0.0]
                cons = [it.weighted_con_used for it in scored] if scored else [0.0]
                
                feats = {
                    "num_evidence": float(len(ranked)),
                    "max_rel": max_rel, "mean_rel": mean_rel, "p90_rel": p90_rel,
                    "max_ent": max_ent, "mean_ent": float(np.mean(ents)) if ents else 0.0,
                    "max_con_used": max_con_used, "mean_con": float(np.mean(cons)) if cons else 0.0,
                    "body_success_rate": body_success_rate,
                    "context_mismatch_rate": ctx_mismatch_rate,
                    "update_style_rate": update_style_rate(ranked),
                }

                support_p, contra_p = 0.0, 0.0
                if support_gate:
                    support_p, _ = gate_predict_proba_safe(support_gate, feats, get_gate_feature_list(support_meta, FEATURE_COLUMNS))
                if contra_gate:
                    contra_p, _ = gate_predict_proba_safe(contra_gate, feats, get_gate_feature_list(contra_meta, FEATURE_COLUMNS))

                fallback_supported = (max_ent >= 0.80 and max_rel >= 0.45 and max_con_used < 0.15)
                fallback_contradicted = (max_con_used >= 0.35 and max_rel >= 0.45)
                
                if contra_p >= contra_thr or fallback_contradicted:
                    decision = "CONTRADICTED"
                elif support_p >= support_thr or fallback_supported:
                    decision = "SUPPORTED"
                else:
                    decision = "UNVERIFIED"

                status.update(label="Verification Complete!", state="complete", expanded=False)

        st.session_state.last_result = {
            "T_raw": T_raw, "lang": lang, "claim_en": claim_en, "image": image,
            "decision": decision, "max_ent": max_ent, "scored": scored, "ranked": ranked,
            "g_clip": g_clip, "clip_note": clip_note, "df_display": df_display,
            "rek_display": rek_display, "mismatch_msg": mismatch_msg,
            "p_real_style": p_real_style, "p_fake_style": p_fake_style, "language_hint": language_hint,
            "ctx_mismatch_rate": ctx_mismatch_rate, "rep_locs": rep_locs, "rep_words": rep_words
        }

    # ==========================================
    # RENDER FINAL RESULTS
    # ==========================================
    if "last_result" in st.session_state:
        res = st.session_state.last_result
        
        with results_container:
            # Re-display claim info nicely
            left, right = st.columns([1.25, 0.9], gap="large")
            with left:
                st.markdown("## Claim")
                st.markdown(f"<div class='smalllabel'>Original</div><div class='card'><b>{res['T_raw']}</b></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='margin-top:10px;' class='card'><div class='smalllabel'>English</div><div><b>{res['claim_en']}</b></div></div>", unsafe_allow_html=True)
            with right:
                st.markdown("## Image")
                if res['image']:
                    st.image(res['image'], caption="Analyzed Image", use_container_width=True)
                else:
                    st.markdown("<div class='card'>No image analyzed.</div>", unsafe_allow_html=True)

            # Verdict Section
            st.markdown("## Verdict & Signals")
            vcol, scol = st.columns([0.62, 1.38], gap="large")

            with vcol:
                st.markdown(
                    f"""
                    <div class="verdict">
                      <div class="k">Final verdict</div>
                      <p class="v">{res['decision']}</p>
                    </div>
                    """, unsafe_allow_html=True
                )
                st.markdown(
                    f"""
                    <div class="card" style="margin-top:12px;">
                      <div class="smalllabel">Evidence strength</div>
                      <div style="font-size:1.25rem; font-weight:750;">{res['max_ent']:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True
                )
                
                # Language Style
                style_html = "n/a"
                if res['p_fake_style'] is not None:
                     style_html = f"<b>{res['language_hint']}</b><br/><span style='opacity:0.8'>P(real)={res['p_real_style']:.2f} · P(fake)={res['p_fake_style']:.2f}</span>"
                
                st.markdown(f"<div class='card' style='margin-top:12px;'><div class='smalllabel'>Language Style</div><div>{style_html}</div></div>", unsafe_allow_html=True)

            with scol:
                st.markdown(
                    f"""
                    <div class="card">
                      <div style="font-weight:800; margin-bottom:10px;">Visual signals</div>
                      <div>
                        <span class="badge"><strong>CLIP:</strong> {res['g_clip']:.3f} ({res['clip_note']})</span>
                        <span class="badge"><strong>Deepfake:</strong> {res['df_display']}</span>
                        <span class="badge"><strong>Rekognition:</strong> {res['rek_display']}</span>
                      </div>
                    </div>
                    """, unsafe_allow_html=True
                )
                if res['mismatch_msg']:
                    st.error(res['mismatch_msg'])

            # Explanation
            st.markdown("## Explanation")
            expl = []
            if res['decision'] == "SUPPORTED": expl.append("Multiple relevant sources align with the claim.")
            elif res['decision'] == "CONTRADICTED": expl.append("Relevant sources contain conflicting information.")
            else: expl.append("Evidence is insufficient to verify this claim.")
            
            if res['ctx_mismatch_rate'] >= 0.5: expl.append("Warning: Evidence discusses different context details than the claim.")
            
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            for line in expl: st.markdown(f"- {line}")
            st.markdown("</div>", unsafe_allow_html=True)

            # Evidence List
            st.markdown("## Evidence")
            shown = res['scored'][:10] if res['scored'] else res['ranked'][:10]
            for it in shown:
                 st.markdown(
                    f"""
                    <div class="evidence">
                      <div class="title">{it.title}</div>
                      <div class="meta">Source: {it.source} · Rel: {it.relevance:.2f} · {it.nli_label}</div>
                      <div class="desc">{(it.desc or "")[:200]}...</div>
                      <div style="margin-top:6px;"><a href="{it.url}" target="_blank" style="color:#4da6ff;text-decoration:none;">Read Source →</a></div>
                    </div>
                    """, unsafe_allow_html=True
                )


# ===========================
# ROUTER
# ===========================
if st.session_state.page == "Landing":
    render_landing()
else:
    render_system()