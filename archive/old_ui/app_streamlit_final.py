# app_streamlit_final.py
# Final integrated Streamlit app:
# - Multilingual claim (HI/KN/TE/EN + mixed) + optional image
# - Retrieval: Google News RSS (primary) + optional APIs if keys exist in .streamlit/secrets.toml
# - Relevance: LaBSE
# - NLI: facebook/bart-large-mnli (premise = title + description)
# - Gates:
#     support_gate.joblib (+ optional meta json) -> decides when to assert SUPPORTED
#     contradiction_gate.joblib (+ optional meta json) -> decides when to assert CONTRADICTED
# - Language-only risk model:
#     tfidf_style_model.joblib -> fake-style risk when evidence is insufficient
# - Non-negotiable: "Consensus context found in evidence" section included.
#
# Files expected in the same folder as this app (C:\ML\fake_news_project):
#   support_gate.joblib
#   contradiction_gate.joblib
#   tfidf_style_model.joblib
# Optional:
#   support_gate_meta.json
#   contradiction_gate_meta.json
#   .streamlit/secrets.toml (optional API keys)

import os
import re
import json
import time
import math
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import streamlit as st
import requests
import torch
from PIL import Image

# Optional but recommended
try:
    import feedparser
except Exception:
    feedparser = None

try:
    import spacy
except Exception:
    spacy = None

from langdetect import detect
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    CLIPProcessor,
    CLIPModel,
)
from sentence_transformers import SentenceTransformer, util

# joblib model loading
import joblib

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Fake News Detection", layout="wide")
st.title("Fake News Detection")

# ---------------- CONSTANTS (universal; no slider wall) ----------------
TOP_K_NLI = 5
MAX_ITEMS_FETCH = 60
RSS_PER_QUERY = 12
HTTP_TIMEOUT = 12

# Gate thresholds: if meta JSON exists, we use meta["best_threshold"]; else defaults.
DEFAULT_SUPPORT_THR = 0.35
DEFAULT_CONTRA_THR = 0.70

# Relevance gate for contradiction usage (soft; universal)
REL_GATE_R0 = 0.45
REL_GATE_R1 = 0.70
MIN_REL_FOR_NLI = 0.18

# CLIP gate mapping (universal)
CLIP_A = 0.20
CLIP_B = 0.35

# Language-risk bands
RISK_HIGH = 0.85
RISK_MED = 0.65

SUPPORTED_LANGS = {"en", "hi", "kn", "te"}  # detection fallback

# NLLB mapping (for translating to English)
LANG_MAP = {"hi": "hin_Deva", "kn": "kan_Knda", "te": "tel_Telu"}


# ---------------- UTILS ----------------
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
        if lang in SUPPORTED_LANGS:
            return lang
        return "en"
    except Exception:
        return "en"


_EN_STOP = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "from", "by", "at",
    "as", "is", "are", "was", "were", "be", "been", "it", "this", "that", "these", "those",
    "after", "before", "new", "latest", "today", "yesterday", "tomorrow", "over", "into",
    "near", "around", "amid", "says", "say", "said", "will", "would", "can", "could", "may",
    "might", "has", "have", "had", "up", "down", "out", "about", "more", "most", "very",
}

def keyword_set_en(text: str) -> set:
    text = normalize_text(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    toks = [t for t in text.split() if len(t) > 2 and t not in _EN_STOP]
    return set(toks)


def rel_gate(relevance: float, r0: float = REL_GATE_R0, r1: float = REL_GATE_R1) -> float:
    if r1 <= r0:
        return 1.0
    return clamp01((relevance - r0) / (r1 - r0))


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


# ---------------- Models ----------------
@st.cache_resource
def load_spacy():
    # Try transformer model if installed, else small model, else None.
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

    # CLIP
    clip_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_name)

    # Relevance encoder
    labse = SentenceTransformer("sentence-transformers/LaBSE")

    # Translator (NLLB) for claim + optionally evidence -> English
    translator_name = "facebook/nllb-200-distilled-600M"
    nllb_tokenizer = AutoTokenizer.from_pretrained(translator_name)
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained(translator_name).to(device)

    # NLI
    nli_name = "facebook/bart-large-mnli"
    nli_tokenizer = AutoTokenizer.from_pretrained(nli_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_name).to(device)

    return device, clip_model, clip_processor, labse, nllb_tokenizer, nllb_model, nli_tokenizer, nli_model


def translate_to_english(text: str, lang: str, device, nllb_tokenizer, nllb_model) -> str:
    text = normalize_text(text)
    if not text:
        return text
    if lang == "en":
        return text
    if lang not in LANG_MAP:
        return text
    try:
        inputs = nllb_tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
        eng_token_id = nllb_tokenizer.convert_tokens_to_ids("eng_Latn")
        with torch.no_grad():
            out = nllb_model.generate(**inputs, forced_bos_token_id=eng_token_id, max_length=128)
        return nllb_tokenizer.decode(out[0], skip_special_tokens=True)
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


def nli_probs(premise: str, hypothesis: str, device, nli_tokenizer, nli_model) -> Tuple[float, float, float]:
    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
    # BART MNLI order: contradiction, neutral, entailment
    p_con, p_neu, p_ent = probs[0], probs[1], probs[2]
    return p_ent, p_con, p_neu


# ---------------- Retrieval: Google News RSS (primary) ----------------
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
        desc = normalize_text(getattr(e, "summary", "") or "")
        published = normalize_text(getattr(e, "published", "") or "")
        if title and link:
            items.append(EvidenceItem(title=title, url=link, source="GoogleNewsRSS", desc=desc, published=published))
    return items, ""


# ---------------- Optional API fetchers (if keys exist) ----------------
def load_keys_from_secrets() -> Dict[str, str]:
    keys = {}
    try:
        # If user put these in .streamlit/secrets.toml
        keys["NEWSAPI_KEY"] = st.secrets.get("NEWSAPI_KEY", "")
        keys["GNEWS_KEY"] = st.secrets.get("GNEWS_KEY", "")
        keys["NEWSDATA_KEY"] = st.secrets.get("NEWSDATA_KEY", "")
        keys["EVENTREGISTRY_KEY"] = st.secrets.get("EVENTREGISTRY_KEY", "")
    except Exception:
        pass
    # Drop empties
    return {k: v for k, v in keys.items() if v}


def fetch_newsapi_org(query: str, api_key: str) -> Tuple[List[EvidenceItem], str]:
    url = (
        "https://newsapi.org/v2/everything?"
        f"q={requests.utils.quote(query)}&language=en&pageSize=20&sortBy=publishedAt&apiKey={api_key}"
    )
    res, err = safe_get_json(url)
    if err:
        return [], err
    items = []
    for a in res.get("articles", [])[:20]:
        title = normalize_text(a.get("title", "") or "")
        link = a.get("url", "") or ""
        desc = normalize_text(a.get("description", "") or "")
        if title and link:
            items.append(EvidenceItem(title=title, url=link, source="NewsAPI.org", desc=desc))
    return items, ""


def fetch_gnews(query: str, api_key: str) -> Tuple[List[EvidenceItem], str]:
    url = f"https://gnews.io/api/v4/search?q={requests.utils.quote(query)}&lang=en&max=20&token={api_key}"
    res, err = safe_get_json(url)
    if err:
        return [], err
    items = []
    for a in res.get("articles", [])[:20]:
        title = normalize_text(a.get("title", "") or "")
        link = a.get("url", "") or ""
        desc = normalize_text(a.get("description", "") or "")
        if title and link:
            items.append(EvidenceItem(title=title, url=link, source="GNews(en)", desc=desc))
    return items, ""


def fetch_newsdata(query: str, api_key: str) -> Tuple[List[EvidenceItem], str]:
    url = f"https://newsdata.io/api/1/news?q={requests.utils.quote(query)}&language=en&apikey={api_key}"
    res, err = safe_get_json(url)
    if err:
        return [], err
    items = []
    for a in res.get("results", [])[:20]:
        title = normalize_text(a.get("title", "") or "")
        link = a.get("link", "") or ""
        desc = normalize_text(a.get("description", "") or "")
        if title and link:
            items.append(EvidenceItem(title=title, url=link, source="NewsData.io", desc=desc))
    return items, ""


def fetch_eventregistry(query: str, api_key: str) -> Tuple[List[EvidenceItem], str]:
    url = "https://eventregistry.org/api/v1/article/getArticles"
    payload = {"action": "getArticles", "keyword": query, "lang": "eng", "articlesCount": 20, "apiKey": api_key}
    res, err = safe_post_json(url, payload)
    if err:
        return [], err
    items = []
    for a in res.get("articles", {}).get("results", [])[:20]:
        title = normalize_text(a.get("title", "") or "")
        link = a.get("url", "") or ""
        desc = normalize_text(a.get("body", "") or "")[:240]
        if title and link:
            items.append(EvidenceItem(title=title, url=link, source="EventRegistry(en)", desc=desc))
    return items, ""


def dedup_items(items: List[EvidenceItem]) -> List[EvidenceItem]:
    seen = set()
    out = []
    for it in items:
        key = normalize_text(it.title).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


# ---------------- Query builder ----------------
def build_queries(original_text: str, claim_en: str) -> List[str]:
    original_text = normalize_text(original_text)
    claim_en = normalize_text(claim_en)

    queries = []
    # Native query first (helps Kannada/Telugu/Hindi)
    if len(original_text) >= 5:
        queries.append(original_text)

    # English query (translation)
    if len(claim_en) >= 5 and claim_en.lower() != original_text.lower():
        queries.append(claim_en)

    # Entity-ish chunks from English claim (very simple; spaCy optional)
    nlp, _ = load_spacy()
    if nlp is not None:
        doc = nlp(claim_en)
        ents = [e.text.strip() for e in doc.ents if e.label_ in {"PERSON", "ORG", "GPE", "LOC", "EVENT"}]
        ents = [e for i, e in enumerate(ents) if e and e.lower() not in {x.lower() for x in ents[:i]}]
        if ents:
            queries.append(" ".join(ents[:3]))
        # verb + first entity
        rootv = ""
        for t in doc:
            if t.dep_ == "ROOT" and t.pos_ in {"VERB", "AUX"}:
                rootv = t.lemma_
                break
        if ents and rootv:
            queries.append(f"{ents[0]} {rootv}")
    else:
        # fallback: take first 8 words
        queries.append(" ".join(claim_en.split()[:8]))

    # Dedup, limit
    out, seen = [], set()
    for q in queries:
        qn = q.lower().strip()
        if len(qn) < 4:
            continue
        if qn not in seen:
            out.append(q.strip())
            seen.add(qn)
    return out[:6]


# ---------------- Ranking + NLI aggregation ----------------
def rank_by_relevance(items: List[EvidenceItem], claim_en: str, labse: SentenceTransformer) -> List[EvidenceItem]:
    claim_emb = labse.encode(claim_en, convert_to_tensor=True)
    ranked = []
    for it in items:
        text = normalize_text((it.title + " " + (it.desc or "")).strip())
        emb = labse.encode(text, convert_to_tensor=True)
        it.relevance = float(util.cos_sim(claim_emb, emb).item())
        ranked.append(it)
    ranked.sort(key=lambda x: x.relevance, reverse=True)
    return ranked


def update_style_rate(items: List[EvidenceItem]) -> float:
    # "update" / evolving-coverage headlines often reduce entailment reliability
    pat = re.compile(r"\b(death toll|toll|rises|rising|latest|update|updates|after|amid|as it happened|live)\b", re.I)
    if not items:
        return 0.0
    hits = 0
    for it in items[:TOP_K_NLI]:
        if pat.search(it.title):
            hits += 1
    return hits / max(1, min(TOP_K_NLI, len(items)))


def consensus_context(items: List[EvidenceItem], claim_en: str, nlp) -> Tuple[List[str], List[str], float]:
    """
    Returns:
      - repeated_locations: location strings mentioned by multiple sources
      - repeated_context_words: repeated non-stop tokens
      - context_mismatch_rate: fraction of repeated context tokens NOT found in claim
    """
    top = items[: min(12, len(items))]
    if not top:
        return [], [], 0.0

    # Locations via spaCy on English-ish titles (best effort)
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

    # Context words repeated across evidence
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
    nli_tokenizer,
    nli_model,
    nllb_tokenizer,
    nllb_model,
) -> Tuple[float, float, List[EvidenceItem], float]:
    """
    Produces:
      max_ent, max_con_used (relevance-gated), scored_items, body_success_rate placeholder (1.0 if we have desc)
    """
    scored = []
    max_ent = 0.0
    max_con_used = 0.0

    # body_success_rate: how many items have a non-empty description
    top_for_body = ranked[: min(12, len(ranked))]
    body_success_rate = 0.0
    if top_for_body:
        body_success_rate = sum(1 for it in top_for_body if (it.desc or "").strip()) / len(top_for_body)

    # NLI on top-K
    top = ranked[: min(TOP_K_NLI, len(ranked))]
    for it in top:
        if it.relevance < MIN_REL_FOR_NLI:
            continue

        premise = normalize_text((it.title + ". " + (it.desc or "")).strip())
        if not premise:
            continue

        # NLI in English. If premise contains lots of non-latin, attempt to translate.
        # (LaBSE handles multilingual relevance, but BART MNLI is English-focused.)
        if re.search(r"[\u0900-\u0D7F]", premise):  # Indic ranges
            premise_en = translate_to_english(premise, "hi", device, nllb_tokenizer, nllb_model)  # best-effort
        else:
            premise_en = premise

        f_ent, f_con, f_neu = nli_probs(premise_en, claim_en, device, nli_tokenizer, nli_model)

        # weighted contradiction = con * (1 - ent), then relevance gate
        w_con = f_con * (1.0 - f_ent)
        g_rel = rel_gate(it.relevance)
        w_con_used = w_con * g_rel

        # Store
        it.f_ent, it.f_con, it.f_neu = float(f_ent), float(f_con), float(f_neu)
        it.rel_gate = float(g_rel)
        it.weighted_con_used = float(w_con_used)

        if f_ent >= f_con and f_ent >= f_neu:
            it.nli_label = "ENTAILS"
        elif f_con >= f_neu:
            it.nli_label = "CONTRADICTS"
        else:
            it.nli_label = "NEUTRAL"

        max_ent = max(max_ent, float(f_ent))
        max_con_used = max(max_con_used, float(w_con_used))
        scored.append(it)

    # Sort evidence for display: best support first
    scored.sort(key=lambda x: (x.f_ent, x.relevance), reverse=True)
    return float(max_ent), float(max_con_used), scored, float(body_success_rate)


# ---------------- Gate loading + feature alignment ----------------
def unwrap_gate(obj):
    # Returns (model, meta_dict_or_None)
    if hasattr(obj, "predict_proba"):
        return obj, None
    if isinstance(obj, dict):
        for k in ["model", "pipeline", "clf", "estimator"]:
            if k in obj and hasattr(obj[k], "predict_proba"):
                return obj[k], obj
    raise TypeError("Gate joblib is not a sklearn estimator or known wrapper dict.")


def load_gate_with_meta(gate_path: str, meta_path: Optional[str]) -> Tuple[object, Dict]:
    model, meta = unwrap_gate(joblib.load(gate_path))
    meta_dict = meta if isinstance(meta, dict) else {}
    if meta_path and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_json = json.load(f)
            meta_dict.update(meta_json)
        except Exception:
            pass
    return model, meta_dict


def gate_features_from_meta(meta: Dict, fallback: List[str]) -> List[str]:
    for k in ["required_features", "feature_names", "features", "cols"]:
        if k in meta and isinstance(meta[k], list) and len(meta[k]) > 0:
            return list(meta[k])
    return list(fallback)


def build_feature_row(features: Dict[str, float], ordered_cols: List[str]) -> List[float]:
    return [float(features.get(c, 0.0)) for c in ordered_cols]


# ---------------- Language risk model ----------------
@st.cache_resource
def load_tfidf_model(path: str):
    return joblib.load(path)


# ---------------- SIDEBAR ----------------
st.sidebar.header("Input")
uploaded_image = st.sidebar.file_uploader("Upload image (optional)", type=["jpg", "jpeg", "png"])
news_text = st.sidebar.text_area("Enter headline / claim (any language)", height=120)

st.sidebar.header("Files (expected)")
st.sidebar.caption("support_gate.joblib, contradiction_gate.joblib, tfidf_style_model.joblib should be in this folder.")

with st.sidebar.expander("Install requirements (copy-paste)", expanded=False):
    st.code(
        "pip install -U streamlit torch transformers sentence-transformers langdetect feedparser scikit-learn joblib pillow requests\n"
        "python -m spacy download en_core_web_sm\n",
        language="bash",
    )

with st.sidebar.expander("API keys (optional)", expanded=False):
    st.markdown(
        "If you have API keys, put them in **.streamlit/secrets.toml** like:\n\n"
        "```\nNEWSAPI_KEY=\"...\"\nGNEWS_KEY=\"...\"\nNEWSDATA_KEY=\"...\"\nEVENTREGISTRY_KEY=\"...\"\n```\n\n"
        "This app works even without keys (RSS fallback)."
    )

show_debug = st.sidebar.checkbox("Show debug details", value=False)

# ---------------- Load heavy resources once ----------------
device, clip_model, clip_processor, labse, nllb_tokenizer, nllb_model, nli_tokenizer, nli_model = load_models()
nlp, spacy_name = load_spacy()

# Try loading gates + TF-IDF
GATE_BASE_FALLBACK = [
    "num_evidence",
    "max_rel", "mean_rel", "p90_rel",
    "max_ent", "mean_ent",
    "max_con_used", "mean_con",
    "body_success_rate",
    "context_mismatch_rate",
    "update_style_rate",
]

support_gate_model = None
contra_gate_model = None
support_meta = {}
contra_meta = {}
tfidf_model = None

# Paths (local folder)
SUPPORT_GATE_PATH = "support_gate.joblib"
CONTRA_GATE_PATH = "contradiction_gate.joblib"
TFIDF_PATH = "tfidf_style_model.joblib"

SUPPORT_META_PATH = "support_gate_meta.json"
CONTRA_META_PATH = "contradiction_gate_meta.json"

# Load artifacts if present
if os.path.exists(SUPPORT_GATE_PATH):
    try:
        support_gate_model, support_meta = load_gate_with_meta(SUPPORT_GATE_PATH, SUPPORT_META_PATH)
    except Exception as e:
        st.error(f"Failed to load support gate: {e}")
if os.path.exists(CONTRA_GATE_PATH):
    try:
        contra_gate_model, contra_meta = load_gate_with_meta(CONTRA_GATE_PATH, CONTRA_META_PATH)
    except Exception as e:
        st.error(f"Failed to load contradiction gate: {e}")
if os.path.exists(TFIDF_PATH):
    try:
        tfidf_model = load_tfidf_model(TFIDF_PATH)
    except Exception as e:
        st.error(f"Failed to load TF-IDF model: {e}")

# Thresholds
support_thr = float(support_meta.get("best_threshold", DEFAULT_SUPPORT_THR)) if support_meta else DEFAULT_SUPPORT_THR
contra_thr = float(contra_meta.get("best_threshold", DEFAULT_CONTRA_THR)) if contra_meta else DEFAULT_CONTRA_THR


# ---------------- Display uploaded image ----------------
image = None
if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

# ---------------- MAIN ----------------
if news_text and len(news_text.strip()) > 0:
    st.subheader("Processing")

    T_raw = normalize_text(news_text)
    lang = detect_language(T_raw)
    claim_en = translate_to_english(T_raw, lang, device, nllb_tokenizer, nllb_model)

    c1, c2 = st.columns(2)
    with c1:
        st.write(f"Detected language: **{lang}**")
        st.write(f"spaCy: **{spacy_name}**")
    with c2:
        st.write(f"Claim (English): **{claim_en}**")

    # Image explanation
    g_clip = 0.0
    if image is not None:
        _, g_clip = clip_gate_score(image, claim_en, device, clip_model, clip_processor)
        st.markdown("### Image explanation")
        if g_clip >= 0.65:
            st.write("The image appears consistent with the claim topic.")
        elif g_clip <= 0.35:
            st.write("The image appears weakly related to the claim topic (possible misuse).")
        else:
            st.write("The image appears somewhat related to the claim topic.")
    else:
        st.markdown("### Image explanation")
        st.write("No image provided, so image–text consistency was not evaluated.")

    # Language-only risk
    p_fake_style = None
    if tfidf_model is not None:
        try:
            p_fake_style = float(tfidf_model.predict_proba([T_raw])[0][1])
        except Exception:
            p_fake_style = None

    # Build queries and fetch
    queries = build_queries(T_raw, claim_en)

    keys = load_keys_from_secrets()
    source_status = {}

    all_items: List[EvidenceItem] = []

    with st.spinner("Fetching evidence (RSS primary)..."):
        # RSS: use both English and local ceid for better regional coverage
        # English India
        for q in queries:
            items, err = google_news_rss(q, hl="en-IN", gl="IN", ceid="IN:en")
            source_status["GoogleNewsRSS(en-IN)"] = "OK" if not err else err
            all_items.extend(items)

        # If claim language is not English, also try RSS with local language params (helps indexing)
        if lang in {"hi", "kn", "te"}:
            # ceid language variants are limited; we still use hl to bias results.
            hl_map = {"hi": "hi-IN", "kn": "kn-IN", "te": "te-IN"}
            hl = hl_map.get(lang, "en-IN")
            for q in queries[:3]:
                items, err = google_news_rss(q, hl=hl, gl="IN", ceid="IN:en")
                source_status[f"GoogleNewsRSS({hl})"] = "OK" if not err else err
                all_items.extend(items)

        # Optional APIs (only if keys exist)
        if keys.get("NEWSAPI_KEY"):
            items, err = fetch_newsapi_org(claim_en, keys["NEWSAPI_KEY"])
            source_status["NewsAPI.org"] = "OK" if not err else err
            all_items.extend(items)

        if keys.get("GNEWS_KEY"):
            items, err = fetch_gnews(claim_en, keys["GNEWS_KEY"])
            source_status["GNews"] = "OK" if not err else err
            all_items.extend(items)

        if keys.get("NEWSDATA_KEY"):
            items, err = fetch_newsdata(claim_en, keys["NEWSDATA_KEY"])
            source_status["NewsData.io"] = "OK" if not err else err
            all_items.extend(items)

        if keys.get("EVENTREGISTRY_KEY"):
            items, err = fetch_eventregistry(claim_en, keys["EVENTREGISTRY_KEY"])
            source_status["EventRegistry"] = "OK" if not err else err
            all_items.extend(items)

    st.markdown("### Source status")
    st.json(source_status)

    all_items = dedup_items(all_items)[:MAX_ITEMS_FETCH]
    st.write(f"Fetched **{len(all_items)}** unique items (before ranking).")

    if len(all_items) == 0:
        st.warning("No evidence items were found. Try rephrasing the claim.")
        st.stop()

    # Rank
    with st.spinner("Ranking by relevance..."):
        ranked = rank_by_relevance(all_items, claim_en, labse)

    # Consensus context (non-negotiable)
    rep_locs, rep_words, ctx_mismatch_rate = consensus_context(ranked, claim_en, nlp)

    st.markdown("## Consensus context found in evidence")
    if rep_locs:
        st.write("Locations mentioned by multiple sources: " + ", ".join(rep_locs))
    else:
        st.write("Locations mentioned by multiple sources: (none)")

    if rep_words:
        st.write("Other repeated context words: " + ", ".join(rep_words))
    else:
        st.write("Other repeated context words: (none)")

    # NLI
    with st.spinner("Running NLI on top evidence..."):
        max_ent, max_con_used, scored, body_success_rate = aggregate_topk_nli(
            ranked,
            claim_en,
            device,
            nli_tokenizer,
            nli_model,
            nllb_tokenizer,
            nllb_model,
        )

    # Additional aggregate stats
    rels = [it.relevance for it in ranked[: min(30, len(ranked))]]
    max_rel = float(max(rels)) if rels else 0.0
    mean_rel = float(sum(rels) / len(rels)) if rels else 0.0
    p90_rel = float(sorted(rels)[max(0, int(0.9 * (len(rels) - 1)))]) if rels else 0.0

    ents = [it.f_ent for it in scored] if scored else [0.0]
    cons = [it.weighted_con_used for it in scored] if scored else [0.0]

    mean_ent = float(sum(ents) / max(1, len(ents)))
    mean_con = float(sum(cons) / max(1, len(cons)))
    upd_rate = float(update_style_rate(ranked))

    # Prepare feature dict
    features = {
        "num_evidence": float(len(ranked)),
        "max_rel": float(max_rel),
        "mean_rel": float(mean_rel),
        "p90_rel": float(p90_rel),
        "max_ent": float(max_ent),
        "mean_ent": float(mean_ent),
        "max_con_used": float(max_con_used),
        "mean_con": float(mean_con),
        "body_success_rate": float(body_success_rate),
        "context_mismatch_rate": float(ctx_mismatch_rate),
        "update_style_rate": float(upd_rate),
    }

    # Gates
    p_support_assert = 0.0
    p_contra_assert = 0.0

    if support_gate_model is not None:
        support_feats = gate_features_from_meta(support_meta, GATE_BASE_FALLBACK)
        Xs = [build_feature_row(features, support_feats)]
        try:
            p_support_assert = float(support_gate_model.predict_proba(Xs)[0][1])
        except Exception:
            p_support_assert = 0.0

    if contra_gate_model is not None:
        contra_feats = gate_features_from_meta(contra_meta, GATE_BASE_FALLBACK)
        Xc = [build_feature_row(features, contra_feats)]
        try:
            p_contra_assert = float(contra_gate_model.predict_proba(Xc)[0][1])
        except Exception:
            p_contra_assert = 0.0

    # Final decision A (SUPPORTED / UNVERIFIED / CONTRADICTED)
    if p_contra_assert >= contra_thr:
        decision = "CONTRADICTED"
    elif p_support_assert >= support_thr:
        decision = "SUPPORTED"
    else:
        decision = "UNVERIFIED"

    # ---------------- OUTPUT ----------------
    st.subheader("Result")

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Decision", decision)
    with m2:
        st.metric("Support gate P", f"{p_support_assert:.3f}")
        st.caption(f"thr={support_thr:.2f}")
    with m3:
        st.metric("Contradiction gate P", f"{p_contra_assert:.3f}")
        st.caption(f"thr={contra_thr:.2f}")
    with m4:
        st.metric("Image alignment", f"{g_clip:.3f}")
    with m5:
        if p_fake_style is None:
            st.metric("Language risk", "n/a")
        else:
            st.metric("Language risk (P)", f"{p_fake_style:.3f}")

    st.subheader("Explanation")
    expl = []
    if decision == "SUPPORTED":
        expl.append("Multiple relevant sources align with the claim, so it is marked supported.")
    elif decision == "CONTRADICTED":
        expl.append("Relevant sources contain conflicting information, so it is marked contradicted.")
    else:
        expl.append("Evidence is insufficient to strongly support or contradict the claim, so it is marked unverified.")

    # Language-risk explanation (only a hint, not a verdict)
    if p_fake_style is not None:
        if p_fake_style >= RISK_HIGH:
            expl.append("The headline style looks strongly associated with misinformation in the training data (language-risk: high).")
        elif p_fake_style >= RISK_MED:
            expl.append("The headline style shows some misinformation-like patterns in the training data (language-risk: medium).")
        else:
            expl.append("The headline style looks closer to normal reporting in the training data (language-risk: low).")

    # Context mismatch explanation (not hardcoded to specific cases)
    if ctx_mismatch_rate >= 0.50 and rep_words:
        expl.append("Evidence contains repeated context details that are not present in the claim, which reduces verification confidence.")

    for line in expl:
        st.write("- " + line)

    st.subheader("Evidence")
    shown = scored[: min(8, len(scored))] if scored else ranked[: min(8, len(ranked))]
    for it in shown:
        desc_line = (it.desc or "").strip()
        if len(desc_line) > 240:
            desc_line = desc_line[:240] + "..."

        st.markdown(
            f"""
**{it.title}**  
{desc_line if desc_line else ""}  
Source: {it.source} | Relevance: {it.relevance:.3f}  
NLI: {it.nli_label} (ent={it.f_ent:.2f}, con={it.f_con:.2f}, neu={it.f_neu:.2f})  
Contradiction used: {it.weighted_con_used:.2f} (rel_gate={it.rel_gate:.2f})  
[Read article]({it.url})
---
"""
        )

    if show_debug:
        st.subheader("Debug")
        st.write("Queries:", queries)
        st.write("Features used for gates:", features)
        st.write("Support gate expected features:", gate_features_from_meta(support_meta, GATE_BASE_FALLBACK))
        st.write("Contradiction gate expected features:", gate_features_from_meta(contra_meta, GATE_BASE_FALLBACK))

st.divider()
st.caption("Decision-support system only. Outputs are based on retrieved headlines/snippets and learned gating; verify using full articles when needed.")
