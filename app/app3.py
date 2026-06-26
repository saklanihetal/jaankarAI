# =========================================================
# JAANKAAR AI (UPGRADED): Multimodal Fake News Detection
# Upgrades:
#   - Gemma-based multilingual query generation
#   - Improved Whisper multilingual transcription/translation
#   - Translated text note panel (frontend)
#   - Expanded multi-query evidence retrieval
#   - Retrieval telemetry panel
# All existing systems preserved (CLIP, BART-MNLI, LaBSE,
#   deepfake, audio analysis, verdict engine, TTS, etc.)
# =========================================================

import os
import re
import io
import json
import base64
from dataclasses import d ataclass, field
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import requests
import numpy as np
import pandas as pd
from PIL import Image

# ---------------- Optional deps (graceful fallback) ----------------
try:
    import torch
except Exception:
    torch = None

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None

try:
    import feedparser
except Exception:
    feedparser = None

try:
    import boto3
except Exception:
    boto3 = None

try:
    import tensorflow as tf
except Exception:
    tf = None

try:
    import joblib
except Exception:
    joblib = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None

try:
    from transformers import (
        CLIPProcessor,
        CLIPModel,
        AutoTokenizer,
        AutoModelForSequenceClassification,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        pipeline as hf_pipeline,
    )
except Exception:
    CLIPProcessor = None
    CLIPModel = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    AutoModelForCausalLM = None
    BitsAndBytesConfig = None
    hf_pipeline = None

try:
    import whisper as openai_whisper
except Exception:
    openai_whisper = None

try:
    from sentence_transformers import SentenceTransformer, util as st_util
except Exception:
    SentenceTransformer = None
    st_util = None

try:
    from gtts import gTTS
except ImportError:
    gTTS = None

try:
    import librosa
    import librosa.feature as _librosa_feature
except Exception:
    librosa = None
    _librosa_feature = None

try:
    import spacy as _spacy_mod

    # Load the small English model; fall back gracefully if not installed.
    try:
        _NLP = _spacy_mod.load("en_core_web_sm")
    except OSError:
        _NLP = None
except Exception:
    _spacy_mod = None
    _NLP = None


# ==============================
# 0) CONFIG + SECRET LOADING
# ==============================
st.set_page_config(
    page_title="Jaankaar AI | Fake News Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_secret(name: str, default: str = "") -> str:
    v = default
    try:
        if hasattr(st, "secrets"):
            v = st.secrets.get(name, default)
    except Exception:
        v = default
    if not v:
        v = os.getenv(name, default)
    return (v or "").strip()


NEWS_API_KEY = get_secret("NEWS_API_KEY", "")
NEWSDATA_KEY = get_secret("NEWSDATA_KEY", "")
EVENTREGISTRY_KEY = get_secret("EVENTREGISTRY_KEY", "")
GNEWS_KEY = ""
AWS_KEY = get_secret("AWS_KEY", "")
AWS_SECRET = get_secret("AWS_SECRET", "")
AWS_REGION = get_secret("AWS_REGION", "ap-south-1")
HF_TOKEN = get_secret("HF_TOKEN", "")

GEMMA_MODEL_ID = "google/gemma-2b-it"
APP_VERSION = "JAANKAAR-AI-V1"

SUPPORTED_LANGUAGES = {
    "en": "English",
    "kn": "Kannada",
    "ta": "Tamil",
    "te": "Telugu",
    "hi": "Hindi",
}


# ==============================
# 1) PROFESSIONAL UI STYLES
# ==============================
st.markdown(
    """
<style>
.stApp {
    background-color: #0e1117;
    background-image: radial-gradient(#1c222e 1px, transparent 1px);
    background-size: 20px 20px;
}
h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; font-weight: 700; color: #f0f2f6; }
p, div, span { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #c3c7d0; }

.css-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    transition: transform 0.2s ease;
}
.css-card:hover { border-color: rgba(255,255,255,0.3); transform: translateY(-2px); }

.custom-badge {
    background: linear-gradient(135deg, #0061ff 0%, #60efff 100%);
    color: #000; padding: 4px 10px; border-radius: 12px;
    font-size: 0.75rem; font-weight: 700; display: inline-block; margin-right: 5px;
}
.verdict-badge-supported {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    color: #fff; padding: 5px 12px; border-radius: 4px; font-weight: bold;
}
.verdict-badge-contradicted {
    background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%);
    color: #fff; padding: 5px 12px; border-radius: 4px; font-weight: bold;
}
.team-member {
    text-align: center; background: #161b22; padding: 20px;
    border-radius: 15px; border: 1px solid #30363d; height: 100%;
}
.team-role { color: #58a6ff; font-size: 0.9rem; font-weight: 600;
             text-transform: uppercase; letter-spacing: 1px; }
.note-panel {
    background: rgba(88,166,255,0.07);
    border: 1px solid rgba(88,166,255,0.3);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
}
[data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
</style>
""",
    unsafe_allow_html=True,
)


# ==============================
# 2) DATA STRUCTURES
# ==============================
@dataclass
class EvidenceItem:
    title: str
    url: str
    source: str
    snippet: str = ""
    img_url: Optional[str] = None
    tfidf_sim: float = 0.0
    nli_label: str = "NEUTRAL"
    nli_ent: float = 0.0
    nli_con: float = 0.0
    nli_neu: float = 1.0
    labse_sim: float = 0.0
    entity_overlap: float = 0.0  # fraction of claim entities found in article
    query_used: str = ""  # which query retrieved this article


# ==============================
# 3) SAFE HELPERS
# ==============================
def norm_text(x: str) -> str:
    x = x or ""
    x = x.strip()
    x = re.sub(r"\s+", " ", x)
    return x


def clamp01(v: float) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except Exception:
        return 0.0


def safe_get(url: str, timeout: int = 12) -> Optional[requests.Response]:
    try:
        return requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    except Exception:
        return None


def load_safe_model(path: str):
    if not joblib:
        return None
    if not os.path.exists(path):
        return None
    try:
        obj = joblib.load(path)
        if hasattr(obj, "predict_proba"):
            return obj
        if isinstance(obj, dict):
            for _, v in obj.items():
                if hasattr(v, "predict_proba"):
                    return v
        return None
    except Exception:
        return None


# ==============================
# 4) LOAD CORE MODELS (cached)
# ==============================
@st.cache_resource(show_spinner="Loading CLIP and NLI models…")
def load_core_models():
    device = "cpu"
    if torch and torch.cuda.is_available():
        device = "cuda"

    clip_model, clip_processor = None, None
    if CLIPModel and CLIPProcessor:
        try:
            clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                ignore_mismatched_sizes=True,
            )
            clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            if torch:
                clip_model = clip_model.to(device)
        except Exception as _e:
            print(f"[CLIP load error] {_e}")
            clip_model, clip_processor = None, None

    nli_tok, nli_model = None, None
    if AutoTokenizer and AutoModelForSequenceClassification:
        try:
            nli_tok = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
            nli_model = AutoModelForSequenceClassification.from_pretrained(
                "facebook/bart-large-mnli"
            )
            if torch:
                nli_model = nli_model.to(device)
        except Exception as _e:
            print(f"[NLI load error] {_e}")
            nli_tok, nli_model = None, None

    return device, clip_model, clip_processor, nli_tok, nli_model


DEVICE, CLIP_M, CLIP_P, NLI_TOK, NLI_M = load_core_models()


@st.cache_resource
def load_aux_models():
    # Resolve paths relative to this file's directory so the app works
    # regardless of the working directory it is launched from.
    _here = os.path.dirname(os.path.abspath(__file__))
    _weights = os.path.join(_here, "..", "models", "weights")

    df_model = None
    _df_path = os.path.join(_weights, "deepfake", "deepfake_cnn.h5")
    if tf and os.path.exists(_df_path):
        try:
            df_model = tf.keras.models.load_model(_df_path, compile=False)
        except Exception as _e:
            print(f"[Deepfake load error] {_e}")
            df_model = None

    rek_client = None
    if boto3 and AWS_KEY and AWS_SECRET and AWS_REGION:
        try:
            rek_client = boto3.client(
                "rekognition",
                aws_access_key_id=AWS_KEY,
                aws_secret_access_key=AWS_SECRET,
                region_name=AWS_REGION,
            )
        except Exception:
            rek_client = None

    support_gate = load_safe_model(os.path.join(_weights, "support_gate.joblib"))
    contra_gate = load_safe_model(os.path.join(_weights, "contradiction_gate.joblib"))
    style_tfidf = load_safe_model(os.path.join(_weights, "tfidf_style_model.joblib"))

    return df_model, rek_client, support_gate, contra_gate, style_tfidf


DF_M, REK, GATE_S, GATE_C, STYLE_TFIDF = load_aux_models()


@st.cache_resource(show_spinner=False)
def load_gemma_model():
    if not (torch and AutoTokenizer and AutoModelForCausalLM):
        return None, None

    token = HF_TOKEN or None
    try:
        tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL_ID, token=token)

        load_kwargs: Dict[str, Any] = {"token": token, "low_cpu_mem_usage": True}

        if torch.cuda.is_available():
            if BitsAndBytesConfig is not None:
                try:
                    bnb_cfg = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    load_kwargs["quantization_config"] = bnb_cfg
                    load_kwargs["device_map"] = "auto"
                except Exception:
                    load_kwargs["torch_dtype"] = torch.float16
                    load_kwargs["device_map"] = "auto"
            else:
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float32

        model = AutoModelForCausalLM.from_pretrained(GEMMA_MODEL_ID, **load_kwargs)
        model.eval()
        return tokenizer, model

    except Exception as exc:
        print(f"[Gemma load error] {exc}")
        return None, None


GEMMA_TOK, GEMMA_M = load_gemma_model()


@st.cache_resource(show_spinner=False)
def load_labse_model():
    """LaBSE for semantic filtering — preserved from original."""
    if SentenceTransformer is None:
        return None
    try:
        return SentenceTransformer("sentence-transformers/LaBSE")
    except Exception:
        return None


LABSE_M = load_labse_model()


@st.cache_resource(show_spinner=False)
def load_whisper_model():
    """Whisper for multilingual speech transcription and translation."""
    if openai_whisper is None:
        return None
    try:
        return openai_whisper.load_model("medium")
    except Exception:
        return None


WHISPER_M = load_whisper_model()


# ==============================
# 5) LEXICON + STYLE
# ==============================
FAKE_LEXICON = [
    "shocking",
    "breaking",
    "unbelievable",
    "exposed",
    "secret",
    "miracle",
    "viral",
    "you won't believe",
    "forwarded",
    "whatsapp",
    "must watch",
    "cure",
    "guaranteed",
    "100%",
    "bombshell",
    "truth revealed",
]


def lexicon_score(text: str) -> float:
    t = (text or "").lower()
    hits = sum(1 for w in FAKE_LEXICON if w in t)
    return clamp01(hits / max(1, len(FAKE_LEXICON) * 0.25))


def basic_style_features(text: str) -> Dict[str, float]:
    t = text or ""
    exclam = t.count("!")
    quest = t.count("?")
    upper = sum(1 for c in t if c.isupper())
    letters = sum(1 for c in t if c.isalpha())
    upper_ratio = (upper / letters) if letters else 0.0
    return {
        "exclam": float(exclam),
        "quest": float(quest),
        "upper_ratio": float(upper_ratio),
        "length": float(len(t)),
    }


# ==============================
# 5b) ENTITY & KEYWORD EXTRACTION (spaCy)
# ==============================
# Entity types we consider "important" for overlap checks.
_IMPORTANT_ENT_TYPES = {
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "EVENT",
    "NORP",
    "FAC",
    "PRODUCT",
    "LAW",
    "DATE",
    "TIME",
}


def extract_entities_and_keywords(text: str) -> Dict[str, Any]:
    """
    Extract named entities and important noun-phrase keywords from *text*
    using spaCy (en_core_web_sm).  Returns a dict with:
        - entities: list of (text, label) tuples
        - entity_texts: flat set of lower-cased entity strings
        - keywords: list of unique noun-phrase strings
    Falls back to a simple regex-based set when spaCy is unavailable.
    """
    result: Dict[str, Any] = {
        "entities": [],
        "entity_texts": set(),
        "keywords": [],
    }

    if _NLP is not None:
        try:
            doc = _NLP(text[:1000])  # cap to avoid slow processing on long text
            entities = [
                (ent.text.strip(), ent.label_)
                for ent in doc.ents
                if ent.label_ in _IMPORTANT_ENT_TYPES and ent.text.strip()
            ]
            entity_texts = {e[0].lower() for e in entities}

            # Noun phrases that are not already captured as named entities
            np_set: set = set()
            for chunk in doc.noun_chunks:
                phrase = chunk.text.strip()
                if (
                    phrase
                    and len(phrase) > 2
                    and phrase.lower() not in entity_texts
                    and not all(t.is_stop for t in chunk)
                ):
                    np_set.add(phrase)

            result["entities"] = entities
            result["entity_texts"] = entity_texts
            result["keywords"] = sorted(np_set)
            return result
        except Exception:
            pass

    # ── Regex fallback: capitalised words as pseudo-entities ──
    tokens = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", text)
    pseudo_ents = list(dict.fromkeys(tokens))  # preserve order, deduplicate
    result["entities"] = [(t, "UNKNOWN") for t in pseudo_ents]
    result["entity_texts"] = {t.lower() for t in pseudo_ents}
    result["keywords"] = pseudo_ents
    return result


def compute_entity_overlap(
    claim_entity_texts: set,
    article_text: str,
) -> float:
    """
    Return the fraction of claim entities that appear (case-insensitive) in
    *article_text*.  Returns 1.0 when the claim has no entities (no penalty).
    """
    if not claim_entity_texts:
        return 1.0
    art_lower = article_text.lower()
    hits = sum(1 for e in claim_entity_texts if e in art_lower)
    return hits / len(claim_entity_texts)


def transcribe_audio(
    audio_path: str,
    language: Optional[str] = None,
) -> Dict[str, str]:
    """
    Transcribe and optionally translate audio using Whisper.

    Args:
        audio_path: path to the audio file
        language:   ISO 639-1 code (e.g. 'hi', 'kn', 'ta', 'te', 'en').
                    When provided Whisper skips language detection and uses
                    this directly — dramatically reduces hallucination on
                    non-English audio.

    Returns:
        {
            "transcript_native":  <original language text>,
            "translated_english": <English translation if non-English>,
            "language":           <detected or forced ISO language code>,
        }
    """
    if WHISPER_M is None:
        return {
            "transcript_native": "",
            "translated_english": "",
            "language": "unknown",
            "error": "Whisper model not loaded.",
        }

    try:
        # Build kwargs — force language when user specifies it
        transcribe_kwargs: Dict[str, Any] = {"task": "transcribe"}
        if language and language != "auto":
            transcribe_kwargs["language"] = language

        native_result = WHISPER_M.transcribe(audio_path, **transcribe_kwargs)
        lang = native_result.get("language", language or "en")
        native_text = native_result.get("text", "").strip()

        # Produce English translation for non-English audio
        if lang != "en":
            translate_kwargs: Dict[str, Any] = {"task": "translate"}
            if language and language != "auto":
                translate_kwargs["language"] = language
            translate_result = WHISPER_M.transcribe(audio_path, **translate_kwargs)
            english_text = translate_result.get("text", "").strip()
        else:
            english_text = native_text

        return {
            "transcript_native": native_text,
            "translated_english": english_text,
            "language": lang,
        }

    except Exception as exc:
        return {
            "transcript_native": "",
            "translated_english": "",
            "language": "unknown",
            "error": str(exc),
        }


# ==============================
# 7) GEMMA QUERY GENERATION
# ==============================
def _fallback_queries(claim: str, claim_info: Optional[Dict] = None) -> List[str]:
    """Rule-based fallback if Gemma query generation fails."""
    c = claim.strip()
    # Short keyword version first — most reliable for APIs
    short = " ".join(c.split()[:5])
    # Entity-focused query from spaCy
    ent_query = ""
    if claim_info:
        ents = [e[0] for e in claim_info.get("entities", [])[:3]]
        if ents:
            ent_query = " ".join(ents)

    queries = [
        short,  # 5-word keyword version (most reliable)
        ent_query,  # entity-focused
        c[:80],  # full claim capped at 80 chars
        f"{short} fact check",
        f"{short} news",
        f"{short} debunked",
        f"{ent_query} news" if ent_query else f"{short} viral",
    ]
    # Deduplicate, filter empty, cap each at 100 chars
    seen, result = set(), []
    for q in queries:
        q = q.strip()[:100]
        if q and q.lower() not in seen:
            seen.add(q.lower())
            result.append(q)
    return result


def generate_search_queries(
    claim: str,
    claim_info: Optional[Dict] = None,
) -> List[str]:
    """
    Use Gemma to generate 5–8 semantically diverse search queries.
    The prompt instructs Gemma to produce entity-focused and event-focused
    queries in addition to the standard variants.
    Falls back to rule-based queries if Gemma is unavailable or fails.
    """
    if GEMMA_TOK is None or GEMMA_M is None:
        return _fallback_queries(claim, claim_info)

    # Build an entity hint string for the prompt
    entity_hint = ""
    if claim_info:
        ents = [e[0] for e in claim_info.get("entities", [])[:6]]
        kws = claim_info.get("keywords", [])[:4]
        parts = []
        if ents:
            parts.append("Key entities: " + ", ".join(ents))
        if kws:
            parts.append("Key phrases: " + ", ".join(kws))
        entity_hint = "  ".join(parts)

    prompt = (
        "<start_of_turn>user\n"
        "You are a search query generator for a fact-checking system.\n"
        "Given the CLAIM and its extracted entities/keywords below, produce "
        "6 to 8 search queries that will help retrieve relevant news articles "
        "to verify or debunk the claim.\n\n"
        "Include ALL of the following query types:\n"
        "1. Original claim rephrased as a query\n"
        "2. Fact-check style query (e.g. 'X fact check')\n"
        "3. Rumor / hoax / debunk query\n"
        "4. Entity-focused query — built around the most important named "
        "entities (persons, organisations, locations) in the claim\n"
        "5. Event-focused query — what event or action is described?\n"
        "6. Neutral semantic rewrite\n"
        "7. Short 3–5 word keyword query\n\n"
        "Rules:\n"
        "- Output ONLY the queries, one per line\n"
        "- NO numbering, NO bullet points, NO markdown, NO quotes\n"
        "- Plain text only\n\n"
        f"CLAIM: {claim}\n"
        + (f"CONTEXT: {entity_hint}\n" if entity_hint else "")
        + "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

    try:
        inputs = GEMMA_TOK(prompt, return_tensors="pt")
        model_dev = next(GEMMA_M.parameters()).device
        inputs = {k: v.to(model_dev) for k, v in inputs.items()}

        with torch.no_grad():
            out_ids = GEMMA_M.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=GEMMA_TOK.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        raw = GEMMA_TOK.decode(
            out_ids[0][prompt_len:], skip_special_tokens=True
        ).strip()

        # Clean and deduplicate; cap each query at 100 chars for API compatibility
        lines = []
        for line in raw.splitlines():
            line = re.sub(r"^[\d\.\-\*\#\s]+", "", line)
            line = re.sub(r'["\']', "", line).strip()[:100]
            if line and len(line) > 3:
                lines.append(line)

        seen, unique = set(), []
        for q in lines:
            key = q.lower()
            if key not in seen:
                seen.add(key)
                unique.append(q)

        if not unique:
            return _fallback_queries(claim, claim_info)

        return unique[:8]

    except Exception:
        return _fallback_queries(claim, claim_info)


# ==============================
# 8) CONTEXT CONSENSUS (evidence fetching)
# ==============================
def fetch_newsapi_evidence(query: str, k: int = 5) -> List[EvidenceItem]:
    if not NEWS_API_KEY:
        return []
    try:
        q = requests.utils.quote(query.strip())
        url = (
            f"https://newsapi.org/v2/everything?q={q}&language=en"
            f"&sortBy=relevancy&pageSize={k}&apiKey={NEWS_API_KEY}"
        )
        r = safe_get(url, timeout=14)
        if not r or not r.ok:
            return []
        items = []
        for a in r.json().get("articles", [])[:k]:
            title = norm_text(a.get("title", ""))
            if title:
                items.append(
                    EvidenceItem(
                        title=title,
                        url=a.get("url", ""),
                        source=(a.get("source") or {}).get("name", "NewsAPI"),
                        snippet=norm_text(a.get("description", "") or ""),
                        img_url=a.get("urlToImage"),
                        query_used=query,
                    )
                )
        return items
    except Exception:
        return []


def fetch_google_rss_evidence(query: str, k: int = 5) -> List[EvidenceItem]:
    if not feedparser:
        return []
    q = requests.utils.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(rss_url)
    items = []
    for e in (feed.entries or [])[:k]:
        title = norm_text(getattr(e, "title", "") or "")
        if title:
            items.append(
                EvidenceItem(
                    title=title,
                    url=getattr(e, "link", "") or "",
                    source="Google RSS",
                    query_used=query,
                )
            )
    return items


def fetch_newsdata_evidence(query: str, k: int = 5) -> List[EvidenceItem]:
    if not NEWSDATA_KEY:
        return []
    q = requests.utils.quote(query)
    url = f"https://newsdata.io/api/1/news?apikey={NEWSDATA_KEY}&q={q}&language=en"
    r = safe_get(url, timeout=14)
    if not r or not r.ok:
        return []
    try:
        items = []
        for res in r.json().get("results", [])[:k]:
            title = norm_text(res.get("title", ""))
            if title:
                items.append(
                    EvidenceItem(
                        title=title,
                        url=res.get("link", ""),
                        source="NewsData.io",
                        snippet=norm_text(res.get("description", "")),
                        img_url=res.get("image_url"),
                        query_used=query,
                    )
                )
        return items
    except Exception:
        return []


def fetch_eventregistry_evidence(query: str, k: int = 5) -> List[EvidenceItem]:
    if not EVENTREGISTRY_KEY:
        return []
    q = requests.utils.quote(query)
    url = (
        f"https://eventregistry.org/api/v1/article/getArticles"
        f"?action=getArticles&keyword={q}&articlesPage=1&articlesCount={k}"
        f"&articlesSortBy=rel&apiKey={EVENTREGISTRY_KEY}"
    )
    r = safe_get(url, timeout=14)
    if not r or not r.ok:
        return []
    try:
        articles = r.json().get("articles", {}).get("results", [])
        items = []
        for a in articles[:k]:
            title = norm_text(a.get("title", ""))
            if title:
                items.append(
                    EvidenceItem(
                        title=title,
                        url=a.get("url", ""),
                        source=a.get("source", {}).get("title", "Event Registry"),
                        img_url=a.get("image"),
                        query_used=query,
                    )
                )
        return items
    except Exception:
        return []


def _fetch_all_for_query(query: str, k_per_source: int = 5) -> List[EvidenceItem]:
    """Fetch from all four sources for a single query."""
    results = []
    for fn in [
        fetch_newsapi_evidence,
        fetch_google_rss_evidence,
        fetch_newsdata_evidence,
        fetch_eventregistry_evidence,
    ]:
        try:
            results.extend(fn(query, k=k_per_source))
        except Exception:
            pass
    return results


def build_evidence_pool(
    claim: str,
    claim_info: Optional[Dict] = None,
    k_total: int = 20,
    k_per_source: int = 4,
) -> Tuple[List[EvidenceItem], List[str]]:
    """
    Generate expanded queries via Gemma (entity-aware), fetch evidence for each,
    deduplicate, and return candidate articles.

    Returns:
        (evidence_list, generated_queries)
    """
    queries = generate_search_queries(claim, claim_info=claim_info)

    pool: List[EvidenceItem] = []
    per_query_counts = []
    for q in queries:
        before = len(pool)
        pool.extend(_fetch_all_for_query(q, k_per_source=k_per_source))
        per_query_counts.append((q, len(pool) - before))

    # Surface per-query fetch counts in session state for debugging
    st.session_state["_fetch_debug"] = per_query_counts

    # Deduplicate by title
    seen, deduped = set(), []
    for it in pool:
        key = it.title.lower().strip()
        if key not in seen:
            seen.add(key)
            deduped.append(it)

    return deduped[:k_total], queries


# ==============================
# 9) TF-IDF SIMILARITY (shortlisting)
# ==============================
def compute_tfidf_sims(claim: str, evid: List[EvidenceItem]) -> List[EvidenceItem]:
    """
    Compute TF-IDF cosine similarity between the claim and each article.
    Uses title + snippet as the article representation for richer matching.
    Articles are scored but NOT filtered here — TF-IDF is shortlisting only.
    """
    if not (TfidfVectorizer and cosine_similarity) or not evid:
        return evid
    # Combine title + snippet for a richer article representation
    article_texts = [f"{e.title} {e.snippet}".strip() for e in evid]
    corpus = [claim] + article_texts
    try:
        vect = TfidfVectorizer(stop_words="english")
        mat = vect.fit_transform(corpus)
        sims = cosine_similarity(mat[0:1], mat[1:]).flatten()
        for i, e in enumerate(evid):
            e.tfidf_sim = float(sims[i]) if i < len(sims) else 0.0
    except Exception:
        pass
    return evid


# ==============================
# 10) LABSE SEMANTIC FILTERING (hard filter at 0.60)
# ==============================
def filter_by_labse(
    claim: str,
    evid: List[EvidenceItem],
    threshold: float = 0.60,
) -> List[EvidenceItem]:
    """
    Compute LaBSE semantic similarity between the claim and each article
    (title + snippet).  Articles that fall *below* the threshold are
    REMOVED entirely — they must not proceed to NLI verification.
    Returns the surviving articles sorted by descending combined score.
    """
    if LABSE_M is None or not evid:
        return evid
    try:
        claim_emb = LABSE_M.encode(claim, convert_to_tensor=True)
        # Use title + snippet for a richer semantic representation
        article_texts = [f"{e.title} {e.snippet}".strip() for e in evid]
        art_embs = LABSE_M.encode(article_texts, convert_to_tensor=True)
        sims = st_util.cos_sim(claim_emb, art_embs)[0].cpu().numpy()
        for i, e in enumerate(evid):
            e.labse_sim = float(sims[i])
    except Exception:
        # If LaBSE fails, keep everything but don't crash
        return sorted(
            evid,
            key=lambda e: e.tfidf_sim + e.labse_sim,
            reverse=True,
        )

    # Hard filter: discard articles below the semantic threshold
    filtered = [e for e in evid if e.labse_sim >= threshold]

    # If the hard filter removes everything, fall back to keeping top-5 by
    # combined score so the pipeline always has something to work with.
    if not filtered:
        filtered = sorted(evid, key=lambda e: e.labse_sim + e.tfidf_sim, reverse=True)[
            :5
        ]

    filtered.sort(key=lambda e: e.labse_sim + e.tfidf_sim, reverse=True)
    return filtered


# ==============================
# 10b) ENTITY-OVERLAP FILTERING (pre-NLI gate)
# ==============================
def apply_entity_overlap_filter(
    evid: List[EvidenceItem],
    claim_entity_texts: set,
    discard_threshold: float = 0.0,
) -> List[EvidenceItem]:
    """
    Score each article by how many claim entities appear in its text.
    Articles with zero overlap when the claim has ≥2 entities are discarded.
    All remaining articles receive an entity_overlap score.
    """
    if not claim_entity_texts:
        # No entities extracted — no penalty, keep everything
        for e in evid:
            e.entity_overlap = 1.0
        return evid

    surviving = []
    for e in evid:
        art_text = f"{e.title} {e.snippet}".lower()
        overlap = compute_entity_overlap(claim_entity_texts, art_text)
        e.entity_overlap = overlap
        # Discard only when claim has ≥2 entities and article shares none
        if len(claim_entity_texts) >= 2 and overlap <= discard_threshold:
            continue
        surviving.append(e)

    # Fallback: never leave NLI with nothing
    if not surviving:
        surviving = evid
    return surviving


# ==============================
# 11) NLI (BART-large-MNLI)
# ==============================
def run_nli(premise: str, hypothesis: str) -> Dict[str, float]:
    if not (torch and NLI_TOK and NLI_M):
        return {"contradiction": 0.0, "neutral": 1.0, "entailment": 0.0}
    try:
        inputs = NLI_TOK(premise, hypothesis, return_tensors="pt", truncation=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            probs = (
                torch.softmax(NLI_M(**inputs).logits, dim=1)[0].detach().cpu().numpy()
            )
        return {
            "contradiction": float(probs[0]),
            "neutral": float(probs[1]),
            "entailment": float(probs[2]),
        }
    except Exception:
        return {"contradiction": 0.0, "neutral": 1.0, "entailment": 0.0}


def annotate_nli(
    claim: str, evid: List[EvidenceItem], max_items: int = 8
) -> List[EvidenceItem]:
    """
    Run BART-MNLI on the top-ranked articles.
    Premise is now title + snippet (not just title) for richer context.
    """
    if not evid:
        return evid
    # Sort by combined relevance score before NLI to process best candidates first
    sorted_e = sorted(
        evid,
        key=lambda x: x.tfidf_sim + x.labse_sim + x.entity_overlap,
        reverse=True,
    )[:max_items]
    for e in sorted_e:
        # Use title + snippet as premise for richer context
        premise = f"{e.title}. {e.snippet}".strip(" .")
        if not premise:
            premise = e.title
        out = run_nli(premise=premise, hypothesis=claim)
        e.nli_ent = out["entailment"]
        e.nli_con = out["contradiction"]
        e.nli_neu = out["neutral"]
        if e.nli_ent >= max(e.nli_con, e.nli_neu):
            e.nli_label = "ENTAILMENT"
        elif e.nli_con >= max(e.nli_ent, e.nli_neu):
            e.nli_label = "CONTRADICTION"
        else:
            e.nli_label = "NEUTRAL"
    m = {x.title: x for x in sorted_e}
    for i in range(len(evid)):
        if evid[i].title in m:
            evid[i] = m[evid[i].title]
    return evid


# ==============================
# 12) CLIP (image-text alignment)
# ==============================
def clip_relevance(image: Image.Image, text: str) -> float:
    """
    Returns a 0-1 score for how well the image matches the text.
    Uses normalised cosine similarity between CLIP embeddings (0=unrelated, 1=perfect match).
    """
    if not (torch and CLIP_M and CLIP_P) or image is None or not text:
        return 0.0
    try:
        inputs = CLIP_P(text=[text], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            # CLIPModel forward pass gives image_embeds and text_embeds directly
            out = CLIP_M(**inputs)
            img_emb = out.image_embeds  # (1, D) tensor
            txt_emb = out.text_embeds  # (1, D) tensor
            # L2-normalise
            img_emb = torch.nn.functional.normalize(img_emb, p=2, dim=-1)
            txt_emb = torch.nn.functional.normalize(txt_emb, p=2, dim=-1)
            # Cosine similarity in [-1, 1] -> remap to [0, 1]
            cos_sim = (img_emb * txt_emb).sum(dim=-1).item()
            score = (cos_sim + 1.0) / 2.0
        return clamp01(score)
    except Exception:
        return 0.0


# ==============================
# 13) DEEPFAKE (Keras CNN)
# ==============================
def deepfake_prob_from_image(image: Image.Image) -> float:
    if DF_M is None or tf is None or image is None:
        return 0.0
    try:
        # Model expects 128×128 RGB input
        img = image.convert("RGB").resize((128, 128))
        arr = np.expand_dims(np.array(img).astype("float32") / 255.0, axis=0)
        pred = DF_M.predict(arr, verbose=0)
        # Model outputs high score for REAL, low for FAKE — invert to get fake probability
        return clamp01(1.0 - float(pred.flatten()[0]))
    except Exception:
        return 0.0


# ==============================
# 13b) AUDIO AUTHENTICITY (librosa)
# ==============================
def analyze_audio_authenticity(audio_path: str) -> Dict[str, Any]:
    """
    Lightweight heuristic audio integrity check using librosa.
    Runs on a background thread — does NOT affect the main verdict.

    Returns a dict with:
        available    : bool
        synthetic_risk: float 0-1  (higher = more suspicious)
        label        : LIKELY SYNTHETIC | LIKELY AUTHENTIC | INCONCLUSIVE
        features     : dict of raw feature values for display
    """
    _empty = {
        "available": False,
        "synthetic_risk": 0.0,
        "label": "UNAVAILABLE",
        "features": {},
    }
    if librosa is None or audio_path is None:
        return _empty
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True, duration=30)

        # Spectral flatness: 1.0 = noise/synthetic, 0.0 = tonal/natural
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))

        # MFCC variance: natural speech has high variance; generated audio is uniform
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_var = float(np.mean(np.var(mfccs, axis=1)))

        # Zero-crossing rate: splice artefacts cause spikes
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))

        # Normalise each to a 0-1 risk contribution
        flat_risk = clamp01(flatness / 0.35)  # >0.35 suspicious
        mfcc_risk = clamp01(1.0 - (mfcc_var / 80.0))  # <80 suspicious
        zcr_risk = clamp01(zcr / 0.28)  # >0.28 suspicious

        # Weighted: MFCC carries most signal
        score = clamp01(0.30 * flat_risk + 0.50 * mfcc_risk + 0.20 * zcr_risk)

        if score >= 0.65:
            label = "LIKELY SYNTHETIC"
        elif score <= 0.30:
            label = "LIKELY AUTHENTIC"
        else:
            label = "INCONCLUSIVE"

        return {
            "available": True,
            "synthetic_risk": round(score, 3),
            "label": label,
            "features": {
                "spectral_flatness": round(flatness, 4),
                "mfcc_variance": round(mfcc_var, 2),
                "zero_crossing_rate": round(zcr, 4),
            },
        }
    except Exception:
        return _empty


# ==============================
# 13c) MEDIA INTEGRITY VERDICT HELPERS
# ==============================
def derive_image_verdict(
    df_prob: float, clip_score: float, rek: Dict[str, Any]
) -> Dict[str, Any]:
    """Standalone image integrity verdict. Never affects the news verdict."""
    if df_prob >= 0.60:
        label, color = "LIKELY MANIPULATED", "#ef473a"
        explanation = (
            f"Deepfake model flagged high manipulation probability "
            f"({df_prob:.0%}). The image may be digitally altered."
        )
    elif df_prob <= 0.25:
        label, color = "LIKELY AUTHENTIC", "#38ef7d"
        explanation = (
            f"Deepfake model found no significant manipulation "
            f"({df_prob:.0%}). Image appears unaltered."
        )
    else:
        label, color = "INCONCLUSIVE", "#ffb000"
        explanation = (
            f"Deepfake score ({df_prob:.0%}) is in the uncertain range. "
            f"Cannot confirm or rule out manipulation."
        )
    notes = []
    for c in rek.get("celebrities", [])[:3]:
        notes.append(f"Recognised: {c['name']}")
    for m in rek.get("moderation", [])[:2]:
        notes.append(f"Content flag: {m['name']}")
    if clip_score < 0.58:
        notes.append("Low image-text alignment (CLIP)")
    return {
        "label": label,
        "color": color,
        "explanation": explanation,
        "notes": notes,
        "df_prob": df_prob,
        "clip_score": clip_score,
    }


def derive_audio_verdict(result: Dict[str, Any]) -> Dict[str, Any]:
    """Standalone audio integrity verdict. Never affects the news verdict."""
    if not result.get("available"):
        return {
            "label": "UNAVAILABLE",
            "color": "#8b949e",
            "explanation": "Librosa not installed or audio could not be analysed.",
            "risk": 0.0,
            "features": {},
        }
    label = result["label"]
    risk = result["synthetic_risk"]
    color_map = {
        "LIKELY SYNTHETIC": "#ef473a",
        "LIKELY AUTHENTIC": "#38ef7d",
        "INCONCLUSIVE": "#ffb000",
    }
    color = color_map.get(label, "#8b949e")
    if label == "LIKELY SYNTHETIC":
        explanation = (
            f"Audio analysis flagged high synthetic risk ({risk:.0%}). "
            f"Spectral and MFCC features suggest AI-generated audio."
        )
    elif label == "LIKELY AUTHENTIC":
        explanation = (
            f"Audio analysis found natural speech characteristics "
            f"(risk: {risk:.0%}). No strong synthetic indicators."
        )
    else:
        explanation = (
            f"Audio analysis inconclusive (risk: {risk:.0%}). Treat with caution."
        )
    return {
        "label": label,
        "color": color,
        "explanation": explanation,
        "risk": risk,
        "features": result.get("features", {}),
    }


# ==============================
# 14) AWS REKOGNITION
# ==============================
def rekognition_signals(image_bytes: bytes) -> Dict[str, Any]:
    if REK is None or not image_bytes:
        return {"available": False, "labels": [], "moderation": []}
    out: Dict[str, Any] = {"available": True, "labels": [], "moderation": []}
    try:
        resp_labels = REK.detect_labels(
            Image={"Bytes": image_bytes}, MaxLabels=10, MinConfidence=70
        )
        out["labels"] = [
            {"name": l.get("Name", ""), "confidence": float(l.get("Confidence", 0.0))}
            for l in resp_labels.get("Labels") or []
        ]
        resp_celebs = REK.recognize_celebrities(Image={"Bytes": image_bytes})
        out["celebrities"] = [
            {
                "name": c.get("Name", ""),
                "confidence": float(c.get("MatchConfidence", 0.0)),
            }
            for c in resp_celebs.get("CelebrityFaces") or []
        ]
    except Exception:
        pass
    try:
        resp_mod = REK.detect_moderation_labels(
            Image={"Bytes": image_bytes}, MinConfidence=70
        )
        out["moderation"] = [
            {"name": m.get("Name", ""), "confidence": float(m.get("Confidence", 0.0))}
            for m in resp_mod.get("ModerationLabels") or []
        ]
    except Exception:
        pass
    return out


# ==============================
# 15) GATES
# ==============================
def gate_predict(model, features: Dict[str, float]) -> float:
    if model is None:
        return 0.0
    try:
        cols = list(features.keys())
        X = pd.DataFrame([[features[c] for c in cols]], columns=cols)
        proba = model.predict_proba(X)[0]
        return float(proba[1]) if len(proba) >= 2 else float(proba[0])
    except Exception:
        return 0.0


def build_gate_features(
    claim: str,
    evid: List[EvidenceItem],
    clip_s: float,
    df_p: float,
    lex_s: float,
) -> Dict[str, float]:
    sims = [e.tfidf_sim for e in evid] if evid else [0.0]
    ent = [e.nli_ent for e in evid] if evid else [0.0]
    con = [e.nli_con for e in evid] if evid else [0.0]
    f = basic_style_features(claim)
    return {
        "tfidf_max": float(np.max(sims)),
        "tfidf_mean": float(np.mean(sims)),
        "nli_ent_max": float(np.max(ent)),
        "nli_con_max": float(np.max(con)),
        "clip": float(clip_s),
        "deepfake": float(df_p),
        "lex": float(lex_s),
        "exclam": float(f["exclam"]),
        "quest": float(f["quest"]),
        "upper_ratio": float(f["upper_ratio"]),
        "length": float(f["length"]),
    }


# ==============================
# 15b) EVIDENCE QUALITY ASSESSMENT
# ==============================
def assess_evidence_quality(evid: List[EvidenceItem]) -> Dict[str, Any]:
    """
    Check whether any articles were retrieved at all.
    If the pool is empty, the system cannot make a verdict and returns
    NOT ENOUGH PROOF immediately — no Gemma call is made.
    All other scoring (LaBSE, NLI signal, confidence) is left to Gemma.
    """
    n = len(evid)
    return {
        "sufficient": n > 0,
        "reason": "no relevant news articles were found for this claim"
        if n == 0
        else "",
        "article_count": n,
        "labse_max": max((e.labse_sim for e in evid), default=0.0),
        "nli_signal_max": max((max(e.nli_ent, e.nli_con) for e in evid), default=0.0),
    }


def _nli_based_verdict(evidence_data: List[Dict]) -> Tuple[str, float, str]:
    """
    Compute verdict deterministically from NLI scores.
    This is the primary verdict signal — Gemma is used only for the rationale.

    Logic:
      - Count ENTAILMENT vs CONTRADICTION labels across all articles.
      - Weight by each article's max(nli_ent, nli_con) score so high-confidence
        articles count more.
      - If weighted contradictions > weighted entailments → CONTRADICTED
      - If weighted entailments > weighted contradictions → SUPPORTED
      - Otherwise → UNVERIFIED
    """
    w_ent = 0.0
    w_con = 0.0
    n_ent = 0
    n_con = 0

    for e in evidence_data:
        nli = e.get("nli", {})
        label = nli.get("label", "NEUTRAL")
        ent_score = float(nli.get("entailment", 0.0))
        con_score = float(nli.get("contradiction", 0.0))

        if label == "ENTAILMENT":
            w_ent += ent_score
            n_ent += 1
        elif label == "CONTRADICTION":
            w_con += con_score
            n_con += 1

    total_w = w_ent + w_con
    if total_w == 0:
        return "UNVERIFIED", 0.0, "nli"

    if w_con > w_ent:
        confidence = round(w_con / total_w, 3)
        return "CONTRADICTED", confidence, "nli"
    elif w_ent > w_con:
        confidence = round(w_ent / total_w, 3)
        return "SUPPORTED", confidence, "nli"
    else:
        return "UNVERIFIED", 0.0, "nli"


def _build_gemma_rationale_prompt(
    claim: str,
    verdict: str,
    evidence_data: List[Dict],
) -> str:
    """Asks Gemma only to explain the verdict — not to re-decide it."""
    lines = []
    for e in evidence_data[:5]:
        title = e.get("title", "")
        label = e.get("nli", {}).get("label", "NEUTRAL")
        if title:
            lines.append(f"- [{label}] {title}")
    ev_block = "\n".join(lines) or "No evidence retrieved."

    return (
        "<start_of_turn>user\n"
        f"The fact-checking system has determined the verdict is: {verdict}\n\n"
        f'CLAIM: "{claim}"\n\n'
        f"EVIDENCE:\n{ev_block}\n\n"
        "Write ONE concise sentence explaining why this verdict was reached. "
        "Be factual. No preamble. Plain text only.\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def _build_gemma_prompt(claim: str, evidence_data: List[Dict]) -> str:
    """
    Build the Gemma verdict prompt.
    evidence_data is a list of dicts with 'title' and 'nli' -> 'label'.
    NLI labels are passed explicitly so Gemma treats CONTRADICTION articles
    as strong evidence the claim is false.
    """
    lines = []
    for e in evidence_data[:8]:
        title = e.get("title", "")
        label = e.get("nli", {}).get("label", "NEUTRAL")
        if title:
            lines.append(f"- [{label}] {title}")
    ev_block = "\n".join(lines) or "No evidence retrieved."

    return (
        "<start_of_turn>user\n"
        "You are a strict fact-checking assistant (knowledge cutoff: January 2026).\n"
        "Each evidence headline is prefixed with its NLI verification status: "
        "[ENTAILMENT] means the headline supports the claim, "
        "[CONTRADICTION] means it contradicts it, "
        "[NEUTRAL] means it is unrelated.\n\n"
        "Rules:\n"
        "1. If the majority of headlines are [CONTRADICTION], the verdict is CONTRADICTED.\n"
        "2. If the majority of headlines are [ENTAILMENT], the verdict is SUPPORTED.\n"
        "3. IDENTITY \u2014 if news is about a replica/tribute but the claim refers to the "
        "original, the verdict is CONTRADICTED.\n"
        "4. ACTIVITY \u2014 if evidence shows a person is active in Jan 2026, any retirement "
        "claim for that person is CONTRADICTED.\n"
        "5. Never use the label MISLEADING \u2014 only SUPPORTED or CONTRADICTED.\n"
        "6. Respond ONLY with a valid JSON object \u2014 no preamble, no markdown fences.\n\n"
        f'CLAIM: "{claim}"\n\n'
        f"EVIDENCE HEADLINES:\n{ev_block}\n\n"
        "Return exactly:\n"
        '{"verdict": "SUPPORTED" or "CONTRADICTED", '
        '"confidence": <float 0.0\u20131.0>, '
        '"rationale": "<one concise sentence>"}\n'
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def backend_reasoner(
    payload: Dict[str, Any],
    evidence_quality: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any], str]:
    """
    Verdict pipeline:
      1. Pre-flight: if no articles retrieved → NOT ENOUGH PROOF immediately
      2. NLI-based verdict: deterministic, computed from BART-MNLI scores
         (weighted entailment vs contradiction across all evidence articles)
      3. Gemma rationale: if Gemma is available, ask it to explain the verdict
         in one sentence. If not, generate a rule-based rationale from NLI counts.
    """
    # ── Pre-flight: evidence quality gate ───────────────────────
    if evidence_quality is not None and not evidence_quality.get("sufficient", True):
        return (
            "NOT ENOUGH PROOF",
            {"confidence": 0.0},
            "No relevant news articles were found for this claim — not enough proof to verify the headline.",
        )

    claim = payload.get("claim", "")
    evidence_data = payload.get("evidence", [])

    # ── Step 1: NLI-based verdict (primary, deterministic) ───────
    nli_verdict, nli_conf, _ = _nli_based_verdict(evidence_data)

    # Count labels for rationale
    n_ent = sum(
        1 for e in evidence_data if e.get("nli", {}).get("label") == "ENTAILMENT"
    )
    n_con = sum(
        1 for e in evidence_data if e.get("nli", {}).get("label") == "CONTRADICTION"
    )
    n_neu = sum(1 for e in evidence_data if e.get("nli", {}).get("label") == "NEUTRAL")
    n_total = len(evidence_data)

    # ── Step 2: Gemma rationale (explanatory only) ────────────────
    rationale = ""
    if GEMMA_TOK is not None and GEMMA_M is not None and nli_verdict != "UNVERIFIED":
        try:
            prompt_text = _build_gemma_rationale_prompt(
                claim, nli_verdict, evidence_data
            )
            inputs = GEMMA_TOK(prompt_text, return_tensors="pt")
            model_dev = next(GEMMA_M.parameters()).device
            inputs = {k: v.to(model_dev) for k, v in inputs.items()}
            with torch.no_grad():
                out_ids = GEMMA_M.generate(
                    **inputs,
                    max_new_tokens=80,
                    do_sample=False,
                    pad_token_id=GEMMA_TOK.eos_token_id,
                )
            prompt_len = inputs["input_ids"].shape[-1]
            rationale = GEMMA_TOK.decode(
                out_ids[0][prompt_len:], skip_special_tokens=True
            ).strip()
            # Clean up any JSON fragments Gemma might still output
            rationale = re.sub(r'^\s*["\{].*?["\}]\s*', "", rationale).strip()
            rationale = rationale.split("\n")[0].strip()
        except Exception:
            rationale = ""

    # ── Fallback rationale from NLI counts ───────────────────────
    if not rationale:
        if nli_verdict == "CONTRADICTED":
            rationale = (
                f"{n_con} of {n_total} retrieved articles contradict this claim "
                f"({n_ent} support, {n_neu} neutral)."
            )
        elif nli_verdict == "SUPPORTED":
            rationale = (
                f"{n_ent} of {n_total} retrieved articles support this claim "
                f"({n_con} contradict, {n_neu} neutral)."
            )
        else:
            rationale = (
                f"Evidence is mixed or insufficient — {n_ent} support, "
                f"{n_con} contradict, {n_neu} neutral out of {n_total} articles."
            )

    data = {
        "verdict": nli_verdict,
        "confidence": nli_conf,
        "rationale": rationale,
        "nli_counts": {"entailment": n_ent, "contradiction": n_con, "neutral": n_neu},
    }
    return nli_verdict, data, rationale


# ==============================
# 16b) TTS — VERDICT AUDIO
# ==============================
def build_verdict_tts_text(
    verdict: str,
    confidence: float,
    rationale: str,
) -> str:
    """Compose a natural-language sentence suitable for TTS."""
    conf_pct = int(round(confidence * 100))
    conf_phrase = f"with {conf_pct} percent confidence" if confidence > 0 else ""
    if verdict == "SUPPORTED":
        opener = "The claim is SUPPORTED"
    elif verdict == "CONTRADICTED":
        opener = "The claim is CONTRADICTED"
    elif verdict == "NOT ENOUGH PROOF":
        opener = "There is NOT ENOUGH PROOF to make a verdict"
    else:
        opener = f"The verdict is {verdict}"
    parts = [opener]
    if conf_phrase:
        parts.append(conf_phrase)
    if rationale:
        parts.append(rationale.rstrip(".") + ".")
    return ". ".join(parts)


def render_verdict_audio(verdict: str, confidence: float, rationale: str) -> None:
    """
    Render a 'Listen to Verdict' button. When clicked, synthesise TTS via
    gTTS, encode to base64, and embed an HTML <audio> element so the browser
    plays it inline without a page reload.
    """
    if gTTS is None:
        st.caption("TTS unavailable — install gtts: `pip install gtts`")
        return
    if st.button("🔊 Listen to Verdict", key="tts_verdict_btn"):
        tts_text = build_verdict_tts_text(verdict, confidence, rationale)
        try:
            buf = io.BytesIO()
            gTTS(text=tts_text, lang="en", slow=False).write_to_fp(buf)
            buf.seek(0)
            audio_b64 = base64.b64encode(buf.read()).decode()
            st.markdown(
                f'<audio autoplay controls style="width:100%; margin-top:8px;">'
                f'<source src="data:audio/mpeg;base64,{audio_b64}" type="audio/mpeg">'
                f"</audio>",
                unsafe_allow_html=True,
            )
        except Exception as exc:
            st.warning(f"TTS error: {exc}")


# ==============================
# 17) TRANSLATION HELPER
# ==============================
def translate_to_english(text: str) -> str:
    if not text or not GoogleTranslator:
        return text
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text


# ==============================
# 18) LANDING PAGE
# ==============================
def show_landing_page():
    st.markdown("<br/><br/>", unsafe_allow_html=True)
    _, c2, _ = st.columns([1, 4, 1])
    with c2:
        st.markdown(
            """
        <div style="text-align: center;">
            <h1 style="font-size: 5rem; color: #58a6ff; margin-bottom: 0;">Jaankaar AI</h1>
            <p style="font-size: 1.2rem; letter-spacing: 1px; color: #8b949e; margin-top: 10px;">
                AI-based Multimodal Multilingual Regional Fake News Detection.
                Supports regional languages including Kannada, Tamil, Telugu, and Hindi.
                Integrates text-based fake news detection, text-image consistency analysis,
                and image-based deepfake detection.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.markdown(
            "<hr style='border-color: #30363d; margin: 40px 0;'>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<h3 style='text-align: center; margin-bottom: 30px;'>TEAM DETAILS</h3>",
        unsafe_allow_html=True,
    )
    team_names = ["Kartik Shekhar", "K V Nihal Mouni", "Kartik Kaul", "Hetal Salanki"]
    cols = st.columns(len(team_names))
    for i, name in enumerate(team_names):
        with cols[i]:
            st.markdown(
                f"""
            <div class="team-member">
                <div style="font-size: 1.5rem; margin-bottom: 10px; color: #8b949e;">Team Member</div>
                <div style="font-size: 1.2rem; font-weight: bold; color: #f0f6fc;">{name}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("<br/><br/><br/>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([3, 2, 3])
    with btn_col:
        st.info("Select 'Run Detector' from the sidebar to begin.")


# ==============================
# 19) TRANSLATED NOTE PANEL
# ==============================
def show_translated_note_panel(
    native_text: str = "",
    english_text: str = "",
    lang_code: str = "en",
) -> Optional[str]:
    """
    Display an editable translated text panel.
    Returns the text to use as claim if user clicks "Use as Claim", else None.

    This panel is shown whenever audio/video content has been processed.
    """
    lang_name = SUPPORTED_LANGUAGES.get(lang_code, lang_code.upper())

    st.markdown(
        """
    <div class="note-panel">
        <h4 style="color:#58a6ff; margin-top:0;">
            🎙️ Whisper Transcription — Translated Text Panel
        </h4>
        <p style="opacity:0.8; font-size:0.9rem;">
            Review and correct the Whisper output below before running analysis.
            You can edit the translated text and click <b>Use as Claim</b> to populate the claim field.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col_native, col_english = st.columns(2)

    with col_native:
        st.markdown(f"**Original ({lang_name})**")
        st.text_area(
            label="native_transcript_display",
            value=native_text,
            height=120,
            disabled=True,
            label_visibility="collapsed",
            key="native_transcript_view",
        )

    with col_english:
        st.markdown("**Translated English** *(editable)*")
        edited_english = st.text_area(
            label="translated_english_edit",
            value=english_text,
            height=120,
            label_visibility="collapsed",
            key="translated_english_edit",
            help="Edit the translation if Whisper made errors before running analysis.",
        )

    use_as_claim = st.button(
        "✅ Use as Claim",
        key="use_as_claim_btn",
        help="Populate the claim field with the translated English text above.",
    )

    if use_as_claim and edited_english.strip():
        st.success("Claim field populated with translated text.")
        return edited_english.strip()

    return None


# ==============================
# 20) RETRIEVAL TELEMETRY PANEL
# ==============================
def show_retrieval_telemetry(
    queries: List[str],
    evid_sorted: List[EvidenceItem],
    claim_info: Optional[Dict] = None,
):
    """Expandable panel showing extracted entities, generated queries,
    and per-article retrieval metadata."""
    with st.expander("🔍 Retrieval Diagnostics", expanded=False):
        # ── Extracted entities & keywords ──
        if claim_info:
            st.markdown("#### Extracted Entities")
            ents = claim_info.get("entities", [])
            if ents:
                badges = " ".join(
                    f'<span style="background:#1f3a5f; color:#58a6ff; '
                    f"padding:2px 8px; border-radius:10px; font-size:0.8rem; "
                    f'margin-right:4px;">{text} <em style="opacity:0.6">({label})</em></span>'
                    for text, label in ents
                )
                st.markdown(badges, unsafe_allow_html=True)
            else:
                st.caption("No named entities detected.")

            kws = claim_info.get("keywords", [])
            if kws:
                st.markdown("**Key Phrases:** " + " · ".join(f"`{k}`" for k in kws[:8]))

            st.divider()

        # ── Generated queries ──
        st.markdown("#### Generated Search Queries")
        for i, q in enumerate(queries, 1):
            st.markdown(f"`{i}.` {q}")

        st.divider()

        # ── Per-article scores ──
        st.markdown("#### Article Retrieval Scores")
        rows = []
        for e in evid_sorted[:15]:
            rows.append(
                {
                    "Article": e.title[:55] + "…" if len(e.title) > 55 else e.title,
                    "Source": e.source,
                    "Query Used": (e.query_used[:45] + "…")
                    if len(e.query_used) > 45
                    else e.query_used,
                    "TF-IDF": round(e.tfidf_sim, 3),
                    "LaBSE": round(e.labse_sim, 3),
                    "Entity Overlap": round(e.entity_overlap, 2),
                    "NLI": e.nli_label,
                }
            )

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ==============================
# 21) DETECTOR PAGE
# ==============================
def show_detector_page():
    if "prefilled_claim" not in st.session_state:
        st.session_state["prefilled_claim"] = ""

    # ════════════════════════════════════════════════════════════
    #  PAGE HEADER
    # ════════════════════════════════════════════════════════════
    st.markdown(
        """<div style="padding:32px 0 24px 0;">
        <h1 style="font-size:2.2rem;font-weight:800;color:#f0f2f6;margin:0;">
            🔍 Jaankaar AI
        </h1>
        <p style="color:#8b949e;font-size:1rem;margin:6px 0 0 0;letter-spacing:0.3px;">
            AI-based Multimodal Fake News Detector &nbsp;&middot;&nbsp;
            Supports English, Hindi, Kannada, Tamil &amp; Telugu
        </p>
        </div>""",
        unsafe_allow_html=True,
    )

    # ════════════════════════════════════════════════════════════
    #  SYSTEM STATUS  (compact badge row)
    # ════════════════════════════════════════════════════════════
    _modules = [
        ("CLIP", CLIP_M and CLIP_P and torch),
        ("NLI (BART)", NLI_M and NLI_TOK and torch),
        ("LaBSE", bool(LABSE_M)),
        ("Whisper", bool(WHISPER_M)),
        ("Gemma", bool(GEMMA_M and GEMMA_TOK)),
        ("Rekognition", bool(REK)),
        ("Librosa", bool(librosa)),
        ("NewsAPI", bool(NEWS_API_KEY)),
    ]
    _badges = ""
    for name, ok in _modules:
        _c = "#38ef7d" if ok else "#ef473a"
        _dot = "●"
        _badges += (
            f'<span style="display:inline-flex;align-items:center;gap:5px;'
            f"background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);"
            f'border-radius:20px;padding:4px 12px;margin:3px;font-size:0.78rem;color:#c3c7d0;">'
            f'<span style="color:{_c};font-size:0.6rem;">{_dot}</span>{name}'
            f"</span>"
        )
    st.markdown(
        f'<div style="margin-bottom:28px;line-height:2;">{_badges}</div>',
        unsafe_allow_html=True,
    )

    # ════════════════════════════════════════════════════════════
    #  INPUT TABS
    # ════════════════════════════════════════════════════════════
    tab_text, tab_audio, tab_image = st.tabs(
        [
            "📝  Text Claim",
            "🎧  Audio Input",
            "🖼️  Image Evidence",
        ]
    )

    audio_file = None
    img_file = None
    whisper_result: Optional[Dict[str, str]] = None

    # ── Tab 1: Text ──────────────────────────────────────────────
    with tab_text:
        st.markdown(
            '<p style="color:#8b949e;font-size:0.88rem;margin-bottom:16px;">'
            "Enter a suspicious headline, rumour, or claim. "
            "Supports all five languages — the system auto-translates to English before analysis."
            "</p>",
            unsafe_allow_html=True,
        )
        claim_input = st.text_area(
            "Claim or Headline",
            height=120,
            value=st.session_state.get("prefilled_claim", ""),
            placeholder="e.g.  Virat Kohli retires from Test cricket…",
            key="claim_input_area",
            label_visibility="collapsed",
        )
        st.markdown(
            '<p style="font-size:0.75rem;color:#555;margin-top:4px;">'
            "Tip: paste the exact headline for best results. Short keyword queries work too."
            "</p>",
            unsafe_allow_html=True,
        )

    # ── Tab 2: Audio ─────────────────────────────────────────────
    with tab_audio:
        st.markdown(
            '<p style="color:#8b949e;font-size:0.88rem;margin-bottom:16px;">'
            "Upload an audio clip. Whisper will transcribe it (with translation if needed) "
            "and populate the claim field. Librosa will also analyse audio authenticity."
            "</p>",
            unsafe_allow_html=True,
        )

        # ── Language + model selectors ───────────────────────────
        _ac1, _ac2 = st.columns(2)
        with _ac1:
            _audio_lang = st.selectbox(
                "Audio language",
                options=[
                    ("auto", "🔍 Auto-detect"),
                    ("hi", "🇮🇳 Hindi"),
                    ("kn", "🇮🇳 Kannada"),
                    ("ta", "🇮🇳 Tamil"),
                    ("te", "🇮🇳 Telugu"),
                    ("en", "🇬🇧 English"),
                ],
                format_func=lambda x: x[1],
                key="whisper_lang_select",
                help="Specifying the language prevents Whisper from hallucinating on non-English audio.",
            )
            _lang_code = _audio_lang[0]

        with _ac2:
            _model_choice = st.selectbox(
                "Whisper model",
                options=[
                    ("medium", "Medium  (recommended — good accuracy)"),
                    ("large-v2", "Large v2 (best accuracy, slower)"),
                    ("base", "Base    (fast, may hallucinate)"),
                    ("small", "Small   (faster than medium)"),
                ],
                format_func=lambda x: x[1],
                key="whisper_model_select",
                help="Larger models are more accurate for regional languages but take longer to load.",
            )
            _chosen_model = _model_choice[0]

        # Reload Whisper if the user picks a different model size
        if st.session_state.get("_whisper_model_loaded") != _chosen_model:
            if st.button("Load selected model", key="load_whisper_btn"):
                with st.spinner(f"Loading Whisper {_chosen_model}…"):
                    try:
                        import whisper as _w

                        new_m = _w.load_model(_chosen_model)
                        # Monkey-patch the global — cache_resource won't re-run
                        import app4 as _self

                        _self.WHISPER_M = new_m
                    except Exception:
                        pass
                st.session_state["_whisper_model_loaded"] = _chosen_model
                st.success(f"Whisper {_chosen_model} loaded.")

        st.markdown("<div style='margin-top:12px;'>", unsafe_allow_html=True)
        audio_file = st.file_uploader(
            "Audio file",
            type=["mp3", "wav", "m4a", "ogg"],
            key="audio_upload",
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if audio_file is not None:
            with st.spinner(
                f"Transcribing with Whisper {_chosen_model} ({_audio_lang[1]})…"
            ):
                import tempfile

                suffix = os.path.splitext(audio_file.name)[-1] or ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(audio_file.read())
                    tmp_path = tmp.name
                whisper_result = transcribe_audio(
                    tmp_path,
                    language=_lang_code if _lang_code != "auto" else None,
                )
                st.session_state["_audio_tmp_path"] = tmp_path

            if whisper_result.get("error"):
                st.warning(f"Whisper error: {whisper_result['error']}")
            else:
                claim_from_panel = show_translated_note_panel(
                    native_text=whisper_result.get("transcript_native", ""),
                    english_text=whisper_result.get("translated_english", ""),
                    lang_code=whisper_result.get("language", "en"),
                )
                if claim_from_panel:
                    st.session_state["prefilled_claim"] = claim_from_panel
                    st.rerun()
        else:
            st.markdown(
                '<div style="border:2px dashed rgba(255,255,255,0.1);border-radius:12px;'
                'padding:32px;text-align:center;color:#555;font-size:0.9rem;">'
                "🎤 Drag &amp; drop an audio file here, or click Browse"
                "</div>",
                unsafe_allow_html=True,
            )

    # ── Tab 3: Image ─────────────────────────────────────────────
    with tab_image:
        st.markdown(
            '<p style="color:#8b949e;font-size:0.88rem;margin-bottom:16px;">'
            "Upload an image related to the claim. "
            "The system will run deepfake detection (CNN), image-text alignment (CLIP), "
            "and AWS Rekognition for labels and celebrity detection."
            "</p>",
            unsafe_allow_html=True,
        )
        img_file = st.file_uploader(
            "Image file",
            type=["jpg", "jpeg", "png"],
            key="image_upload",
            label_visibility="collapsed",
        )
        if img_file is not None:
            st.image(
                img_file, caption="Uploaded image", use_container_width=False, width=320
            )
        else:
            st.markdown(
                '<div style="border:2px dashed rgba(255,255,255,0.1);border-radius:12px;'
                'padding:32px;text-align:center;color:#555;font-size:0.9rem;">'
                "🖼️ Drag &amp; drop an image here, or click Browse"
                "</div>",
                unsafe_allow_html=True,
            )

    # ════════════════════════════════════════════════════════════
    #  ANALYSE BUTTON
    # ════════════════════════════════════════════════════════════
    st.markdown("<div style='margin-top:24px;'>", unsafe_allow_html=True)
    run_btn = st.button(
        "🔍  Analyse Claim",
        type="primary",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Analysis ─────────────────────────────────────────────────
    if run_btn:
        claim_original = norm_text(claim_input)
        if not claim_original:
            st.error("Input required: Please enter a claim.")
            st.stop()

        with st.spinner("Translating input stream…"):
            claim_en = translate_to_english(claim_original)
            claim = claim_en
        if claim != claim_original:
            st.info(f"Translated: {claim}")

        # ── Stage 1: Entity & keyword extraction ────────────────
        with st.spinner("Extracting entities and keywords…"):
            claim_info = extract_entities_and_keywords(claim)

        with st.spinner("Generating expanded search queries via Gemma…"):
            evid, queries = build_evidence_pool(
                claim, claim_info=claim_info, k_total=20, k_per_source=4
            )

        st.caption(
            f"Generated {len(queries)} search queries → retrieved {len(evid)} raw articles"
        )

        # ── Fetch debug: show per-query article counts ───────────
        fetch_debug = st.session_state.get("_fetch_debug", [])
        if fetch_debug and len(evid) == 0:
            with st.expander(
                "⚠️ Fetch Debug — click to see why 0 articles returned", expanded=True
            ):
                for q, cnt in fetch_debug:
                    st.markdown(f"- `{q[:80]}` → **{cnt}** articles")

        with st.spinner("TF-IDF shortlisting…"):
            evid = compute_tfidf_sims(claim, evid)
            # Keep only top-N by TF-IDF before semantic filtering
            evid = sorted(evid, key=lambda e: e.tfidf_sim, reverse=True)[:30]

        with st.spinner("LaBSE semantic filtering (threshold 0.60)…"):
            evid = filter_by_labse(claim, evid, threshold=0.60)

        st.caption(f"{len(evid)} articles passed LaBSE semantic filter")

        with st.spinner("Entity-overlap filtering…"):
            evid = apply_entity_overlap_filter(
                evid, claim_info.get("entity_texts", set())
            )

        st.caption(f"{len(evid)} articles passed entity-overlap filter")

        with st.spinner("Running NLI verification (BART-MNLI)…"):
            evid = annotate_nli(claim, evid)

        # Audio integrity analysis (fast, <2s for 30s clip)
        _raw_audio_result = {"available": False}
        _audio_tmp_path = (
            st.session_state.get("_audio_tmp_path") if audio_file is not None else None
        )
        if _audio_tmp_path and librosa is not None:
            with st.spinner("Analysing audio integrity…"):
                _raw_audio_result = analyze_audio_authenticity(_audio_tmp_path)
            try:
                os.unlink(_audio_tmp_path)
            except Exception:
                pass
            st.session_state["_audio_tmp_path"] = None

        image, img_bytes = None, b""
        if img_file:
            try:
                img_bytes = img_file.read()
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                image = None

        with st.spinner("Computing multimodal vectors…"):
            clip_s = clip_relevance(image, claim) if image else 0.0
            df_p = deepfake_prob_from_image(image) if image else 0.0
            rek = rekognition_signals(img_bytes) if img_bytes else {"available": False}
            lex_s = lexicon_score(claim)
            feats = build_gate_features(claim, evid, clip_s, df_p, lex_s)
            p_support = gate_predict(GATE_S, feats)
            p_contra = gate_predict(GATE_C, feats)

        evid_sorted = sorted(
            evid, key=lambda x: (x.tfidf_sim + x.labse_sim, x.nli_ent), reverse=True
        )
        best_evidence_image = next((e.img_url for e in evid_sorted if e.img_url), None)

        evid_pack = [
            {
                "title": e.title,
                "source": e.source,
                "url": e.url,
                "snippet": e.snippet,
                "tfidf_sim": round(e.tfidf_sim, 4),
                "labse_sim": round(e.labse_sim, 4),
                "entity_overlap": round(e.entity_overlap, 4),
                "nli": {
                    "label": e.nli_label,
                    "entailment": round(e.nli_ent, 4),
                    "contradiction": round(e.nli_con, 4),
                },
            }
            for e in evid_sorted[:8]
        ]

        payload = {
            "claim": claim,
            "evidence": evid_pack,
            "signals": {
                "context": {
                    "tfidf_max": float(np.max([x.tfidf_sim for x in evid] or [0])),
                    "tfidf_mean": float(np.mean([x.tfidf_sim for x in evid] or [0])),
                    "nli_ent_max": float(np.max([x.nli_ent for x in evid] or [0])),
                    "nli_con_max": float(np.max([x.nli_con for x in evid] or [0])),
                    "labse_max": float(np.max([x.labse_sim for x in evid] or [0])),
                    "entity_overlap_mean": float(
                        np.mean([x.entity_overlap for x in evid] or [0])
                    ),
                },
                "image": {
                    "has_image": bool(image),
                    "clip_relevance": float(clip_s),
                    "deepfake_prob": float(df_p),
                    "rekognition": rek,
                },
                "style": {
                    "lexicon": float(lex_s),
                    "basic": basic_style_features(claim),
                },
                "gates": {
                    "support_gate_prob": float(p_support),
                    "contradiction_gate_prob": float(p_contra),
                },
                "extraction": {
                    "entities": [
                        {"text": t, "label": l}
                        for t, l in claim_info.get("entities", [])
                    ],
                    "keywords": claim_info.get("keywords", []),
                },
            },
        }

        # ── Evidence quality gate (pre-Gemma) ───────────────────
        ev_quality = assess_evidence_quality(evid)

        with st.spinner("Deliberating final verdict via Gemma…"):
            verdict, structured, rationale = backend_reasoner(
                {"claim": claim, "evidence": evid_pack},
                evidence_quality=ev_quality,
            )

        confidence = float(structured.get("confidence", 0.0))

        # ── Results UI ────────────────────────────────────────────────────────
        st.markdown("---")

        # ════════════════════════════════════════════════════════
        #  PRIMARY VERDICT BANNER
        # ════════════════════════════════════════════════════════
        _vcolor = {
            "SUPPORTED": "#38ef7d",
            "CONTRADICTED": "#ef473a",
            "NOT ENOUGH PROOF": "#ffb000",
            "UNVERIFIED": "#ffb000",
        }.get(verdict, "#8b949e")

        _icon = {
            "SUPPORTED": "✅",
            "CONTRADICTED": "❌",
            "NOT ENOUGH PROOF": "⚠️",
            "UNVERIFIED": "❓",
        }.get(verdict, "ℹ️")

        _conf_bar = ""
        if verdict in {"SUPPORTED", "CONTRADICTED"} and confidence > 0:
            _cp = int(round(confidence * 100))
            _conf_bar = (
                f'<div style="margin-top:14px;">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:4px;">'
                f'<span style="font-size:0.8rem;color:#8b949e;letter-spacing:1px;text-transform:uppercase;">Confidence</span>'
                f'<span style="font-size:0.8rem;color:{_vcolor};font-weight:700;">{_cp}%</span>'
                f"</div>"
                f'<div style="background:#1c222e;border-radius:8px;height:10px;overflow:hidden;">'
                f'<div style="background:linear-gradient(90deg,{_vcolor}88,{_vcolor});width:{_cp}%;height:10px;border-radius:8px;transition:width 0.6s ease;"></div>'
                f"</div></div>"
            )

        _why = ""
        if verdict in {"NOT ENOUGH PROOF", "UNVERIFIED"}:
            _qr = ev_quality.get("reason", "")
            _why = (
                f'<div style="margin-top:12px;padding:10px 14px;background:rgba(255,176,0,0.08);'
                f'border-radius:8px;border:1px solid rgba(255,176,0,0.25);">'
                f'<span style="font-size:0.78rem;color:#ffb000;font-weight:600;">Why no verdict? </span>'
                f'<span style="font-size:0.78rem;color:#8b949e;">{_qr if _qr else rationale}</span>'
                f"</div>"
            )

        st.markdown(
            f'<div style="background:linear-gradient(135deg,rgba(255,255,255,0.04),rgba(255,255,255,0.02));'
            f'border:1px solid {_vcolor}44;border-left:6px solid {_vcolor};border-radius:14px;padding:24px 28px;margin-bottom:6px;">'
            f'<div style="display:flex;align-items:center;gap:12px;">'
            f'<span style="font-size:2.4rem;">{_icon}</span>'
            f"<div>"
            f'<div style="font-size:0.7rem;color:#8b949e;text-transform:uppercase;letter-spacing:2px;margin-bottom:2px;">News Verification Verdict</div>'
            f'<h1 style="color:{_vcolor};margin:0;font-size:2rem;font-weight:800;letter-spacing:-0.5px;">{verdict}</h1>'
            f"</div></div>"
            f'<p style="margin:12px 0 0 0;font-size:0.95rem;color:#c3c7d0;line-height:1.6;">{rationale}</p>'
            f"{_conf_bar}"
            f"{_why}"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Voice button — prominent, right below the verdict
        st.markdown(
            '<div style="margin-top:8px;margin-bottom:20px;">', unsafe_allow_html=True
        )
        render_verdict_audio(verdict, confidence, rationale)
        st.markdown("</div>", unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════
        #  MEDIA INTEGRITY ROW  (image + audio, side by side)
        # ════════════════════════════════════════════════════════
        _show_img = image is not None
        _show_aud = audio_file is not None

        if _show_img or _show_aud:
            st.markdown(
                '<p style="font-size:0.7rem;color:#8b949e;text-transform:uppercase;'
                'letter-spacing:2px;margin-bottom:8px;">Media Integrity Checks</p>',
                unsafe_allow_html=True,
            )
            _mi_cols = st.columns(2 if (_show_img and _show_aud) else 1)

            if _show_img:
                iv = derive_image_verdict(df_p, clip_s, rek)
                _iv_notes_html = "".join(
                    f'<li style="font-size:0.78rem;color:#8b949e;margin-top:3px;">{n}</li>'
                    for n in iv["notes"]
                )
                with _mi_cols[0]:
                    st.markdown(
                        f'<div style="background:rgba(255,255,255,0.03);border:1px solid {iv["color"]}33;'
                        f'border-left:4px solid {iv["color"]};border-radius:12px;padding:16px 18px;">'
                        f'<div style="font-size:0.65rem;color:#8b949e;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px;">Image Integrity</div>'
                        f'<div style="color:{iv["color"]};font-size:1.15rem;font-weight:700;margin-bottom:6px;">{iv["label"]}</div>'
                        f'<p style="font-size:0.82rem;color:#c3c7d0;margin:0 0 8px 0;">{iv["explanation"]}</p>'
                        + (
                            f'<ul style="margin:4px 0;padding-left:16px;">{_iv_notes_html}</ul>'
                            if _iv_notes_html
                            else ""
                        )
                        + f'<div style="display:flex;gap:16px;margin-top:8px;">'
                        f'<span style="font-size:0.72rem;color:#8b949e;">Deepfake <strong style="color:{iv["color"]};">{iv["df_prob"]:.0%}</strong></span>'
                        f'<span style="font-size:0.72rem;color:#8b949e;">CLIP align <strong style="color:#c3c7d0;">{iv["clip_score"]:.2f}</strong></span>'
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )

            if _show_aud:
                av = derive_audio_verdict(_raw_audio_result)
                _aud_feats = av.get("features", {})
                _feat_rows = ""
                if _aud_feats:
                    _feat_rows = (
                        f'<div style="display:flex;gap:16px;margin-top:8px;flex-wrap:wrap;">'
                        f'<span style="font-size:0.72rem;color:#8b949e;">Flatness <strong style="color:#c3c7d0;">{_aud_feats.get("spectral_flatness", 0):.3f}</strong></span>'
                        f'<span style="font-size:0.72rem;color:#8b949e;">MFCC var <strong style="color:#c3c7d0;">{_aud_feats.get("mfcc_variance", 0):.1f}</strong></span>'
                        f'<span style="font-size:0.72rem;color:#8b949e;">ZCR <strong style="color:#c3c7d0;">{_aud_feats.get("zero_crossing_rate", 0):.3f}</strong></span>'
                        f"</div>"
                    )
                    _risk_pct = int(av.get("risk", 0) * 100)
                    _risk_bar = (
                        f'<div style="margin-top:8px;">'
                        f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
                        f'<span style="font-size:0.7rem;color:#8b949e;">Synthetic Risk</span>'
                        f'<span style="font-size:0.7rem;color:{av["color"]};font-weight:700;">{_risk_pct}%</span>'
                        f"</div>"
                        f'<div style="background:#1c222e;border-radius:6px;height:6px;">'
                        f'<div style="background:{av["color"]};width:{_risk_pct}%;height:6px;border-radius:6px;"></div>'
                        f"</div></div>"
                    )
                else:
                    _risk_bar = ""

                _aud_col_idx = 1 if _show_img else 0
                with _mi_cols[_aud_col_idx]:
                    st.markdown(
                        f'<div style="background:rgba(255,255,255,0.03);border:1px solid {av["color"]}33;'
                        f'border-left:4px solid {av["color"]};border-radius:12px;padding:16px 18px;">'
                        f'<div style="font-size:0.65rem;color:#8b949e;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px;">Audio Integrity (librosa)</div>'
                        f'<div style="color:{av["color"]};font-size:1.15rem;font-weight:700;margin-bottom:6px;">{av["label"]}</div>'
                        f'<p style="font-size:0.82rem;color:#c3c7d0;margin:0 0 4px 0;">{av["explanation"]}</p>'
                        + _feat_rows
                        + _risk_bar
                        + f'<p style="font-size:0.68rem;color:#555;margin-top:8px;margin-bottom:0;">'
                        f"Heuristic signal — low weight. Does not affect the news verdict.</p>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # ════════════════════════════════════════════════════════
        #  EVIDENCE IMAGE  +  KEY SOURCES
        # ════════════════════════════════════════════════════════
        st.markdown("")
        ev_img_col, ev_src_col = st.columns([1, 2])

        with ev_img_col:
            if best_evidence_image:
                st.image(
                    best_evidence_image,
                    caption="Top evidence source image",
                    use_container_width=True,
                )
            elif image:
                st.image(image, caption="Uploaded image", use_container_width=True)

        with ev_src_col:
            st.markdown(
                '<p style="font-size:0.7rem;color:#8b949e;text-transform:uppercase;'
                'letter-spacing:2px;margin-bottom:8px;">Key Evidence Sources</p>',
                unsafe_allow_html=True,
            )
            for e in evid_sorted[:4]:
                _nlc = (
                    "#38ef7d"
                    if e.nli_label == "ENTAILMENT"
                    else "#ef473a"
                    if e.nli_label == "CONTRADICTION"
                    else "#8b949e"
                )
                _snip = (e.snippet[:120] + "…") if len(e.snippet) > 120 else e.snippet
                st.markdown(
                    f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);'
                    f'border-radius:10px;padding:12px 14px;margin-bottom:8px;">'
                    f'<a href="{e.url}" target="_blank" style="text-decoration:none;color:#58a6ff;'
                    f'font-size:0.9rem;font-weight:600;">{e.title[:80]}{"…" if len(e.title) > 80 else ""}</a>'
                    f'<div style="display:flex;gap:8px;align-items:center;margin-top:6px;flex-wrap:wrap;">'
                    f'<span style="background:rgba(0,97,255,0.2);color:#60efff;font-size:0.68rem;'
                    f'padding:2px 8px;border-radius:10px;font-weight:700;">{e.source}</span>'
                    f'<span style="color:{_nlc};font-size:0.7rem;border:1px solid {_nlc};'
                    f'padding:1px 6px;border-radius:6px;">{e.nli_label}</span>'
                    f'<span style="font-size:0.68rem;color:#555;">LaBSE {e.labse_sim:.2f}</span>'
                    f"</div>"
                    + (
                        f'<p style="font-size:0.78rem;color:#8b949e;margin:6px 0 0 0;">{_snip}</p>'
                        if _snip
                        else ""
                    )
                    + f"</div>",
                    unsafe_allow_html=True,
                )

        # ════════════════════════════════════════════════════════
        #  TELEMETRY (collapsed by default)
        # ════════════════════════════════════════════════════════
        show_retrieval_telemetry(queries, evid_sorted, claim_info=claim_info)

        with st.expander("📊 Signal Telemetry", expanded=False):
            st.markdown("#### Context Metrics")
            m1, m2, m3, m4, m5 = st.columns(5)
            ctx = payload["signals"]["context"]
            m1.metric("TF-IDF Max", f"{ctx['tfidf_max']:.2f}")
            m2.metric("TF-IDF Mean", f"{ctx['tfidf_mean']:.2f}")
            m3.metric("NLI Ent Max", f"{ctx['nli_ent_max']:.2f}")
            m4.metric("NLI Con Max", f"{ctx['nli_con_max']:.2f}")
            m5.metric("LaBSE Max", f"{ctx['labse_max']:.2f}")
            st.divider()
            st.markdown("#### Style & Visual")
            s1, s2, s3, s4 = st.columns(4)
            style = payload["signals"]["style"]
            img_sig = payload["signals"]["image"]
            rek_data = img_sig.get("rekognition", {})
            rek_celebs = rek_data.get("celebrities", [])
            rek_labels = rek_data.get("labels", [])
            rek_val = (
                rek_celebs[0]["name"]
                if rek_celebs
                else rek_labels[0]["name"]
                if rek_labels
                else "None"
            )
            s1.metric("Lexicon", f"{style['lexicon']:.2f}")
            s2.metric("Uppercase Ratio", f"{style['basic']['upper_ratio']:.2f}")
            s3.metric(
                "CLIP Relevance",
                f"{img_sig['clip_relevance']:.2f}" if img_sig["has_image"] else "N/A",
            )
            s4.metric(
                "Deepfake Prob",
                f"{img_sig['deepfake_prob']:.2f}" if img_sig["has_image"] else "N/A",
            )
            st.metric("Entity Recognised", rek_val)


# ==============================
# 22) MAIN APP ROUTER
# ==============================
def main():
    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio("Go to", ["Home", "Run Detector"], label_visibility="collapsed")
        st.markdown("---")
        st.markdown("### About")
        st.info(
            "Jaankaar AI — Multi-modal Multilingual Regional Fake News Detection.\n\n"
            "Supports English, Kannada, Tamil, Telugu, and Hindi.\n\n"
            "Pipeline: Whisper → LaBSE → Gemma queries → BART-MNLI → Verdict."
        )
        st.markdown("---")
        st.markdown("### Supported Languages")
        for code, name in SUPPORTED_LANGUAGES.items():
            st.markdown(f"- **{code.upper()}**: {name}")

    if page == "Home":
        show_landing_page()
    else:
        show_detector_page()


if __name__ == "__main__":
    main()
