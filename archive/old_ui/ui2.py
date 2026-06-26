# app_streamlit_v5_ui_polished.py
# V5 UI polish (requested):
# - Title: "AI-based Regional Fake News Detection"
# - Consistent typography (headings + body text same scale via custom CSS)
# - Remove deprecated use_column_width warnings (use width="container")
# - Put all image alignment / visual signals (CLIP, Deepfake CNN, AWS Rekognition) together in one "Visual Evidence" card
# - Cleaner, more professional layout with cards, separators, consistent spacing
#
# NOTE: Core pipeline logic kept the same as your v4_fixed (with ceid fix + query sanitization + rekognition face fallback).

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
st.set_page_config(
    page_title="AI-based Regional Fake News Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===========================
# UI: GLOBAL STYLES
# ===========================
st.markdown(
    """
<style>
/* ---- Typography ---- */
:root{
  --app-font: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
  --h1: 1.55rem;
  --h2: 1.15rem;
  --body: 0.98rem;
  --muted: 0.86rem;
}

html, body, [class*="css"]  { font-family: var(--app-font); }

/* Streamlit default titles can be huge; keep consistent */
h1 { font-size: var(--h1) !important; font-weight: 700 !important; letter-spacing: 0.2px; }
h2, h3 { font-size: var(--h2) !important; font-weight: 650 !important; }
p, li, div { font-size: var(--body); }

/* Section headers */
.section-title{
  font-size: var(--h2);
  font-weight: 700;
  margin: 0.2rem 0 0.6rem 0;
}

/* Subtext */
.muted{
  font-size: var(--muted);
  opacity: 0.78;
}

/* Card */
.card{
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.03);
  border-radius: 14px;
  padding: 14px 16px;
  margin-bottom: 12px;
}

/* Small badge */
.badge{
  display:inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.05);
  font-size: 0.84rem;
}

/* Evidence item */
.evi-title{
  font-weight: 650;
  margin-bottom: 4px;
}
.evi-meta{
  font-size: var(--muted);
  opacity: 0.82;
  margin-bottom: 6px;
}
hr{
  border: none;
  border-top: 1px solid rgba(255,255,255,0.09);
  margin: 10px 0;
}
</style>
""",
    unsafe_allow_html=True,
)

# Header
st.title("AI-based Regional Fake News Detection")
st.markdown('<div class="muted">Decision-support system that cross-checks a claim against retrieved evidence and visual signals.</div>', unsafe_allow_html=True)
st.write("")


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


# ===========================
# EVIDENCE STRUCT
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
# SIDEBAR (polished)
# ===========================
st.sidebar.markdown("### Input")
uploaded_image = st.sidebar.file_uploader("Upload image (optional)", type=["jpg", "jpeg", "png"])
news_text = st.sidebar.text_area("Enter headline / claim (any language)", height=140)
show_debug = st.sidebar.checkbox("Show debug details", value=False)
with st.sidebar.expander("Run command", expanded=False):
    st.code("streamlit run app_streamlit_v5_ui_polished.py", language="bash")


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
# IMAGE PREVIEW (no deprecated use_column_width)
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

    # Top layout: Claim + Image preview
    left, right = st.columns([1.4, 1.0], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Claim</div>', unsafe_allow_html=True)
        st.markdown(f"**Original:** {T_raw}")
        st.markdown(f"**Detected language:** <span class='badge'>{lang}</span>", unsafe_allow_html=True)
        st.markdown(f"**English (for retrieval/NLI):** {claim_en}")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Image (optional)</div>', unsafe_allow_html=True)
        if image is not None:
            st.image(image, caption="Uploaded image", width=None, use_container_width=True)
        else:
            st.markdown('<div class="muted">No image uploaded.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

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
    # VISUAL EVIDENCE (ALL IN ONE PLACE)
    # ===========================
    g_clip = 0.0
    deepfake_label = "n/a"
    deepfake_conf = None
    p_img_real = None

    rek_name = None
    rek_conf = None
    rek_note = None
    rek_faces_n = None
    rek_faces_note = None

    if image is not None:
        _, g_clip = clip_gate_score(image, claim_en, device, clip_model, clip_processor)

        if deepfake_model is not None:
            p_img_real, deepfake_label, deepfake_conf = deepfake_predict_pil(image, deepfake_model)

        if rek_client is not None:
            rek_name, rek_conf, rek_note = rekognition_detect_celebrity_from_image(image, rek_client)
            if rek_name is None:
                rek_faces_n, rek_faces_note = rekognition_detect_faces_count(image, rek_client)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Visual Evidence</div>', unsafe_allow_html=True)

    v1, v2, v3 = st.columns(3, gap="large")

    with v1:
        st.metric("CLIP alignment (0–1)", f"{g_clip:.3f}")
        if image is None:
            st.markdown('<div class="muted">Upload an image to compute CLIP alignment.</div>', unsafe_allow_html=True)
        else:
            if g_clip >= 0.65:
                st.markdown('<div class="muted">Strong match between image & claim topic.</div>', unsafe_allow_html=True)
            elif g_clip <= 0.35:
                st.markdown('<div class="muted">Weak match — possible out-of-context image.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="muted">Partial match between image & claim topic.</div>', unsafe_allow_html=True)

    with v2:
        if image is None:
            st.metric("Deepfake CNN", "n/a")
            st.markdown('<div class="muted">Upload an image to run deepfake detection.</div>', unsafe_allow_html=True)
        else:
            if deepfake_model is None:
                st.metric("Deepfake CNN", "unavailable")
                st.markdown(f'<div class="muted">{deepfake_status}</div>', unsafe_allow_html=True)
            else:
                st.metric("Deepfake CNN", deepfake_label)
                st.markdown(f'<div class="muted">confidence: {deepfake_conf:.2f} · p_real: {p_img_real:.2f}</div>', unsafe_allow_html=True)

    with v3:
        if image is None:
            st.metric("AWS Rekognition", "n/a")
            st.markdown('<div class="muted">Upload an image to run celebrity recognition.</div>', unsafe_allow_html=True)
        else:
            if rek_client is None:
                st.metric("AWS Rekognition", "unavailable")
                st.markdown(f'<div class="muted">{rek_status}</div>', unsafe_allow_html=True)
                if show_debug:
                    st.write(rek_debug)
            else:
                if rek_name is None:
                    st.metric("AWS Rekognition", "no celebrity")
                    st.markdown(f'<div class="muted">{rek_note}</div>', unsafe_allow_html=True)
                    if rek_faces_n is not None:
                        st.markdown(f'<div class="muted">Faces detected (fallback): {rek_faces_n} ({rek_faces_note})</div>', unsafe_allow_html=True)
                else:
                    st.metric("AWS Rekognition", rek_name)
                    st.markdown(f'<div class="muted">match confidence: {rek_conf:.1f}%</div>', unsafe_allow_html=True)
                    claim_person = extract_person_from_claim(claim_en, nlp)
                    if claim_person:
                        same = rek_name.lower() in claim_person.lower() or claim_person.lower() in rek_name.lower()
                        if (not same) and rek_conf >= 90.0:
                            st.warning(f"Possible mismatch: claim mentions **{claim_person}**, but image matches **{rek_name}** with high confidence.")
                    else:
                        st.markdown('<div class="muted">No PERSON entity detected in the claim.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ===========================
    # Language-only hint (TF-IDF style)
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
    # Evidence retrieval
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

    queries = build_queries(T_raw, claim_en)

    api_queries = []
    ce = sanitize_api_query(claim_en)
    if ce:
        api_queries.append(ce)
    ce_short = sanitize_api_query(" ".join(claim_en.split()[:9]))
    if ce_short and ce_short not in api_queries:
        api_queries.append(ce_short)

    source_status: Dict[str, str] = {}
    items: List[EvidenceItem] = []

    with st.spinner("Fetching evidence (APIs + RSS)..."):
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

        # RSS always
        rss_any_ok = False
        last_err = ""
        for q in queries:
            its, err = google_news_rss(q, hl="en-IN", gl="IN", ceid=CEID_MAP["en"])
            if err:
                last_err = err
            else:
                rss_any_ok = True
                items.extend(its)
        source_status["GoogleNewsRSS(en-IN)"] = "OK" if rss_any_ok else (last_err or "No results")

        # Local language RSS
        if lang in {"hi", "kn", "te"}:
            hl_map = {"hi": "hi-IN", "kn": "kn-IN", "te": "te-IN"}
            hl = hl_map.get(lang, "en-IN")
            ceid_local = CEID_MAP.get(lang, "IN:en")
            rss2_any_ok = False
            last_err2 = ""
            for q in queries:
                its, err = google_news_rss(q, hl=hl, gl="IN", ceid=ceid_local)
                if err:
                    last_err2 = err
                else:
                    rss2_any_ok = True
                    items.extend(its)
            source_status[f"GoogleNewsRSS({hl})"] = "OK" if rss2_any_ok else (last_err2 or "No results")

    items = dedup_items(items)
    items = items[:MAX_ITEMS_FETCH]

    # ===========================
    # Source status (clean)
    # ===========================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Evidence Sources</div>', unsafe_allow_html=True)
    s1, s2 = st.columns([1.0, 1.2], gap="large")
    with s1:
        st.markdown('<div class="muted">Status</div>', unsafe_allow_html=True)
        st.json(source_status)
    with s2:
        counts: Dict[str, int] = {}
        for it in items:
            counts[it.source] = counts.get(it.source, 0) + 1
        st.markdown('<div class="muted">Items per source</div>', unsafe_allow_html=True)
        if counts:
            st.write(" · ".join([f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])]))
        else:
            st.write("No items returned.")
        st.markdown(f'<div class="muted">Total unique items: {len(items)}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if not items:
        st.error("No evidence items were found. Try a shorter claim or check API quotas.")
        st.stop()

    # ===========================
    # Rank + consensus + NLI
    # ===========================
    ranked = rank_by_relevance(items, claim_en, labse)
    rep_locs, rep_words, ctx_mismatch_rate = consensus_context(ranked, claim_en, nlp)

    max_ent, max_con_used, scored, body_success_rate = aggregate_topk_nli(
        ranked, claim_en, device, nli_tok, nli_model
    )

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
    # RESULT (consistent sizing)
    # ===========================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3, gap="large")
    with r1:
        st.metric("Decision", decision)
    with r2:
        st.metric("Evidence strength (max entail)", f"{max_ent:.2f}")
    with r3:
        st.metric("Language hint", language_hint if p_fake is not None else "n/a")

    if p_fake is not None:
        st.markdown(f'<div class="muted">Language probs: P(real-style)={p_real:.3f} · P(fake-style)={p_fake:.3f}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ===========================
    # EXPLANATION (same font size as evidence titles)
    # ===========================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Explanation</div>', unsafe_allow_html=True)

    expl = []
    if decision == "SUPPORTED":
        expl.append("Multiple relevant sources align with the claim, so it is marked supported.")
    elif decision == "CONTRADICTED":
        expl.append("Relevant sources contain conflicting information, so it is marked contradicted.")
    else:
        expl.append("Evidence is insufficient to strongly support or contradict the claim, so it is marked unverified.")

    if decision == "UNVERIFIED" and p_fake is not None:
        if "Likely True" in language_hint:
            expl.append("Language patterns look closer to standard reporting, but evidence is still insufficient.")
        elif "Likely False" in language_hint:
            expl.append("Language patterns look misinformation-like, but evidence is still insufficient.")
        else:
            expl.append("Language patterns are not decisive for this headline.")

    if ctx_mismatch_rate >= 0.50 and rep_words:
        expl.append("Evidence contains repeated context details not present in the claim, which reduces verification confidence.")

    # Consensus summary
    if rep_locs or rep_words:
        locs = ", ".join(rep_locs) if rep_locs else "none"
        words = ", ".join(rep_words) if rep_words else "none"
        expl.append(f"Consensus context (from evidence): locations={locs}; repeated keywords={words}.")

    for line in expl:
        st.markdown(f"- {line}")

    if show_debug and (support_note or contra_note):
        st.markdown(f"<hr><div class='muted'>Gate debug: support={support_note} · contradiction={contra_note}</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ===========================
    # EVIDENCE LIST (titles + explanation same size)
    # ===========================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Top Evidence</div>', unsafe_allow_html=True)

    shown = scored[: min(10, len(scored))] if scored else ranked[: min(10, len(ranked))]
    for it in shown:
        desc_line = (it.desc or "").strip()
        if len(desc_line) > 240:
            desc_line = desc_line[:240] + "..."

        st.markdown(
            f"""
<div class="evi-title">{it.title}</div>
<div class="evi-meta">Source: {it.source} · Relevance: {it.relevance:.3f} · NLI: {it.nli_label} (ent={it.f_ent:.2f}, con={it.f_con:.2f}, neu={it.f_neu:.2f}) · Contradiction used: {it.weighted_con_used:.2f}</div>
<div>{desc_line}</div>
<div class="muted"><a href="{it.url}" target="_blank">Read article</a></div>
<hr>
""",
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

    if show_debug:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Debug</div>', unsafe_allow_html=True)
        st.write("RSS Queries:", queries)
        st.write("API Queries:", api_queries)
        st.write("Features:", feats)
        st.write("Gates:", {"support_p": support_p, "contra_p": contra_p, "support_note": support_note, "contra_note": contra_note})
        st.markdown('</div>', unsafe_allow_html=True)

st.write("")
st.markdown('<div class="muted">Decision-support only. Outputs are based on retrieved headlines/snippets, NLI, and learned gating; verify using full articles when needed.</div>', unsafe_allow_html=True)

