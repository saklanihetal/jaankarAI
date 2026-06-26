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
from dataclasses import dataclass, field
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
APP_VERSION = "JAANKAAR-AI-V2"

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
    retrieval_score: float = 0.0  # combined: 0.4*LaBSE + 0.3*TF-IDF + 0.3*entity
    query_used: str = ""  # which query retrieved this article
    rejection_reason: str = ""  # why this article was filtered (for logs)


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
            st.session_state["_nli_load_error"] = str(_e)
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
        st.session_state["_gemma_load_error"] = str(exc)
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
    "MONEY",
    "PERCENT",
    "CARDINAL",
    "ORDINAL",
    "QUANTITY",
    "WORK_OF_ART",
    "LANGUAGE",
}

# Higher-priority types drive entity-focused search queries
_CORE_ENT_TYPES = {"PERSON", "ORG", "GPE", "LOC", "EVENT", "NORP", "FAC"}


def extract_entities_and_keywords(text: str) -> Dict[str, Any]:
    """
    Extract named entities and noun-phrase keywords from *text* using spaCy.

    Returns:
        entities       : list of (text, label) — all extracted entities
        entity_texts   : flat set of lower-cased entity strings (for overlap)
        core_entities  : list of (text, label) for PERSON/ORG/GPE/LOC/EVENT only
        by_type        : dict mapping label → list of entity strings
        keywords       : list of unique noun-phrase strings
        person_names   : list of PERSON entity strings (for mismatch detection)
        locations      : list of GPE/LOC entity strings
        dates          : list of DATE/TIME entity strings
    """
    result: Dict[str, Any] = {
        "entities": [],
        "entity_texts": set(),
        "core_entities": [],
        "by_type": {},
        "keywords": [],
        "person_names": [],
        "locations": [],
        "dates": [],
    }

    if _NLP is not None:
        try:
            # Use up to 2000 chars — enough for headlines, avoids slow processing
            doc = _NLP(text[:2000])
            entities: List[Tuple[str, str]] = []
            seen_ent: set = set()
            for ent in doc.ents:
                txt = ent.text.strip()
                if not txt or ent.label_ not in _IMPORTANT_ENT_TYPES:
                    continue
                key = txt.lower()
                if key in seen_ent:
                    continue
                seen_ent.add(key)
                entities.append((txt, ent.label_))

            entity_texts: set = {e[0].lower() for e in entities}
            core_entities = [(t, l) for t, l in entities if l in _CORE_ENT_TYPES]

            # Group by type
            by_type: Dict[str, List[str]] = {}
            for txt, label in entities:
                by_type.setdefault(label, []).append(txt)

            # Noun phrases not already in entities
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
            result["core_entities"] = core_entities
            result["by_type"] = by_type
            result["keywords"] = sorted(np_set)
            result["person_names"] = [t for t, l in entities if l == "PERSON"]
            result["locations"] = [t for t, l in entities if l in ("GPE", "LOC")]
            result["dates"] = [t for t, l in entities if l in ("DATE", "TIME")]
            return result
        except Exception:
            pass

    # ── Regex fallback: capitalised words as pseudo-entities ─────
    tokens = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", text)
    pseudo_ents = list(dict.fromkeys(tokens))
    result["entities"] = [(t, "UNKNOWN") for t in pseudo_ents]
    result["entity_texts"] = {t.lower() for t in pseudo_ents}
    result["core_entities"] = result["entities"]
    result["by_type"] = {"UNKNOWN": pseudo_ents}
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
    Validates the audio has content before transcribing to prevent
    the 'cannot reshape tensor of 0 elements' error on empty/silent files.
    """
    _empty_error = {
        "transcript_native": "",
        "translated_english": "",
        "language": "unknown",
    }

    if WHISPER_M is None:
        return {**_empty_error, "error": "Whisper model not loaded."}

    # ── Validate audio has actual content ────────────────────────
    try:
        import librosa as _lb

        _y, _ = _lb.load(audio_path, sr=16000, mono=True, duration=10)
        if len(_y) < 3200:  # less than 0.2 seconds of audio
            return {
                **_empty_error,
                "error": "Audio is too short or silent. Please upload a clip of at least 1 second.",
            }
        # Check if it's just silence (RMS energy near zero)
        if float((_y**2).mean() ** 0.5) < 1e-4:
            return {
                **_empty_error,
                "error": "Audio appears to be silent. Please check your recording.",
            }
    except Exception:
        pass  # librosa not available or load failed — let Whisper try anyway

    try:
        # fp16=False: avoids UserWarning on CPU and prevents ambiguous reshape errors
        _kw: Dict[str, Any] = {"task": "transcribe", "fp16": False}
        if language and language != "auto":
            _kw["language"] = language

        native_result = WHISPER_M.transcribe(audio_path, **_kw)
        lang = native_result.get("language", language or "en")
        native_text = native_result.get("text", "").strip()

        if lang != "en":
            _kw_t: Dict[str, Any] = {"task": "translate", "fp16": False}
            if language and language != "auto":
                _kw_t["language"] = language
            translate_result = WHISPER_M.transcribe(audio_path, **_kw_t)
            english_text = translate_result.get("text", "").strip()
        else:
            english_text = native_text

        return {
            "transcript_native": native_text,
            "translated_english": english_text,
            "language": lang,
        }

    except Exception as exc:
        msg = str(exc)
        if "reshape tensor of 0 elements" in msg or "ambiguous" in msg:
            msg = (
                "Audio file could not be processed — it may be too short, "
                "silent, or in an unsupported format. Try a WAV or MP3 file "
                "with at least 1 second of speech."
            )
        return {**_empty_error, "error": msg}


# ==============================
# 7) GEMMA QUERY GENERATION
# ==============================
def _build_baseline_queries(claim: str, claim_info: Optional[Dict] = None) -> List[str]:
    """
    Build guaranteed baseline queries from the claim itself.
    These always run — they do NOT depend on Gemma.
    Covers: original claim, entity query, keyword query, fact-check, debunk.
    """
    c = claim.strip()
    queries: List[str] = []

    # 1. Original claim (capped for API compatibility)
    queries.append(c[:120])

    # 2. Entity-focused: join core entity names
    if claim_info:
        core = [t for t, _ in claim_info.get("core_entities", [])[:4]]
        if core:
            queries.append(" ".join(core))
        # Sub-queries per entity type for richer retrieval
        persons = claim_info.get("person_names", [])[:2]
        locs = claim_info.get("locations", [])[:2]
        dates = claim_info.get("dates", [])[:1]
        if persons:
            queries.append(" ".join(persons))
        if locs and persons:
            queries.append(f"{persons[0]} {locs[0]}" if persons else locs[0])
        if dates and persons:
            queries.append(f"{persons[0]} {dates[0]}" if persons else "")

    # 3. Short keyword version (first 5 meaningful words)
    stop = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "has",
        "have",
        "had",
        "will",
        "would",
        "could",
        "should",
        "in",
        "on",
        "at",
        "to",
        "of",
        "for",
        "and",
        "or",
        "but",
        "that",
        "this",
        "it",
        "he",
        "she",
        "they",
        "we",
        "i",
    }
    kw_words = [w for w in c.split() if w.lower() not in stop][:5]
    if kw_words:
        queries.append(" ".join(kw_words))

    # 4. Fact-check style
    short = " ".join(c.split()[:6])
    queries.append(f"{short} fact check")

    # 5. Debunk style
    queries.append(f"{short} fake news debunked")

    # 6. News event style
    queries.append(f"{short} news")

    # Deduplicate + filter empty
    seen: set = set()
    result: List[str] = []
    for q in queries:
        q = q.strip()[:120]
        if q and q.lower() not in seen:
            seen.add(q.lower())
            result.append(q)
    return result


def _fallback_queries(claim: str, claim_info: Optional[Dict] = None) -> List[str]:
    """Rule-based fallback if Gemma query generation fails."""
    return _build_baseline_queries(claim, claim_info)


def generate_search_queries(
    claim: str,
    claim_info: Optional[Dict] = None,
) -> List[str]:
    """
    Always start with guaranteed baseline queries derived directly from the
    claim and its entities, then AUGMENT with Gemma-generated queries if
    Gemma is available.  Gemma is supplementary — not the sole source.
    """
    # Always build baseline queries first
    baseline = _build_baseline_queries(claim, claim_info)

    if GEMMA_TOK is None or GEMMA_M is None:
        return baseline

    # Build an entity hint string for the Gemma prompt
    entity_hint = ""
    if claim_info:
        ents = [e[0] for e in claim_info.get("core_entities", [])[:6]]
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
        "4 to 6 ADDITIONAL search queries that are NOT generic.\n"
        "Focus on: specific events, date-specific queries, location+person combos,\n"
        "and source-specific searches that would find relevant news.\n\n"
        "Rules:\n"
        "- Output ONLY the queries, one per line\n"
        "- NO numbering, NO bullet points, NO markdown, NO quotes\n"
        "- Plain text only\n"
        "- Each query must be specific and distinct\n\n"
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
                max_new_tokens=160,
                do_sample=False,
                pad_token_id=GEMMA_TOK.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        raw = GEMMA_TOK.decode(
            out_ids[0][prompt_len:], skip_special_tokens=True
        ).strip()

        gemma_queries: List[str] = []
        for line in raw.splitlines():
            line = re.sub(r"^[\d\.\-\*\#\s]+", "", line)
            line = re.sub(r'["\']', "", line).strip()[:120]
            if line and len(line) > 3:
                gemma_queries.append(line)

        # Merge: baseline first (guaranteed), then unique Gemma additions
        seen: set = {q.lower() for q in baseline}
        merged = list(baseline)
        for q in gemma_queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                merged.append(q)

        return merged[:8]  # cap total queries

    except Exception:
        return baseline


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
    k_total: int = 30,
    k_per_source: int = 5,
) -> Tuple[List[EvidenceItem], List[str]]:
    """
    Generate expanded queries, fetch evidence from all sources for every query,
    deduplicate, rank by TF-IDF relevance, return top-k most relevant articles.

    Budget: 8 queries × 3 sources × 5 per source = ~120 raw → deduped → top 30.
    """
    queries = generate_search_queries(claim, claim_info=claim_info)

    # ── Collect from all sources for every query ─────────────────
    raw_pool: List[EvidenceItem] = []
    per_query_counts = []
    for q in queries:
        before = len(raw_pool)
        raw_pool.extend(_fetch_all_for_query(q, k_per_source=k_per_source))
        per_query_counts.append((q, len(raw_pool) - before))

    st.session_state["_fetch_debug"] = per_query_counts

    # ── Deduplicate by normalised title ───────────────────────────
    seen: set = set()
    deduped: List[EvidenceItem] = []
    for it in raw_pool:
        key = re.sub(r"\s+", " ", it.title.lower().strip())
        if key and key not in seen:
            seen.add(key)
            deduped.append(it)

    if not deduped:
        return [], queries

    # ── Rank full pool by TF-IDF relevance before capping ─────────
    # This ensures the cap cuts the *least* relevant articles, not
    # random ones that happened to arrive last.
    if TfidfVectorizer and cosine_similarity and len(deduped) > 1:
        try:
            article_texts = [f"{e.title} {e.snippet}".strip() for e in deduped]
            corpus = [claim] + article_texts
            vect = TfidfVectorizer(stop_words="english", max_features=20000)
            mat = vect.fit_transform(corpus)
            sims = cosine_similarity(mat[0:1], mat[1:]).flatten()
            for i, e in enumerate(deduped):
                e.tfidf_sim = float(sims[i]) if i < len(sims) else 0.0
            deduped.sort(key=lambda e: e.tfidf_sim, reverse=True)
        except Exception:
            pass  # if TF-IDF fails, keep fetch order

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
    article_texts = [f"{e.title} {e.snippet}".strip() for e in evid]
    corpus = [claim] + article_texts
    try:
        vect = TfidfVectorizer(stop_words="english", max_features=20000)
        mat = vect.fit_transform(corpus)
        sims = cosine_similarity(mat[0:1], mat[1:]).flatten()
        for i, e in enumerate(evid):
            e.tfidf_sim = float(sims[i]) if i < len(sims) else 0.0
    except Exception:
        pass
    return evid


# ==============================
# 10) SEMANTIC SCORING + COMBINED RANKING
# ==============================
def score_and_rank_evidence(
    claim: str,
    evid: List[EvidenceItem],
    claim_entity_texts: set,
    top_k: int = 40,
) -> Tuple[List[EvidenceItem], List[EvidenceItem]]:
    """
    Combined scoring pipeline for retrieval ranking.

    DESIGN RATIONALE — why entity overlap is intentionally weak here:
    Contradictory articles frequently describe the SAME EVENT with different
    entities (wrong location, wrong person, different date). If entity overlap
    drives ranking, these articles score low and never reach NLI — which is
    exactly the failure mode that produces UNVERIFIED on contradicted claims.

    Formula:
        retrieval_score = 0.55 × LaBSE + 0.35 × TF-IDF + 0.10 × entity_overlap

    Entity overlap is kept at 0.10 as a tiebreaker only.
    LaBSE + TF-IDF together cover semantic + keyword relevance.
    An article with labse=0.55, tfidf=0.30, entity=0.0 still scores 0.41 — enough
    to pass the soft floor and reach NLI where it can fire as CONTRADICTION.

    Returns: (accepted, rejected) — rejected items carry a rejection_reason.
    """
    if not evid:
        return [], []

    # ── LaBSE scores ──────────────────────────────────────────────
    labse_available = False
    if LABSE_M is not None:
        try:
            claim_emb = LABSE_M.encode(claim, convert_to_tensor=True)
            article_texts = [f"{e.title} {e.snippet}".strip() for e in evid]
            art_embs = LABSE_M.encode(article_texts, convert_to_tensor=True)
            sims = st_util.cos_sim(claim_emb, art_embs)[0].cpu().numpy()
            for i, e in enumerate(evid):
                e.labse_sim = float(sims[i])
            labse_available = True
        except Exception:
            pass

    # ── Entity overlap scores (tiebreaker only) ───────────────────
    for e in evid:
        art_text = f"{e.title} {e.snippet}".lower()
        if claim_entity_texts:
            hits = sum(1 for ent in claim_entity_texts if ent in art_text)
            e.entity_overlap = hits / len(claim_entity_texts)
        else:
            e.entity_overlap = 1.0  # no entities → neutral

    # ── Combined score ─────────────────────────────────────────────
    if labse_available:
        w_labse, w_tfidf, w_entity = 0.55, 0.35, 0.10
    else:
        # LaBSE unavailable — shift weight to TF-IDF, keep entity minimal
        w_labse, w_tfidf, w_entity = 0.00, 0.85, 0.15

    for e in evid:
        e.retrieval_score = (
            w_labse * e.labse_sim + w_tfidf * e.tfidf_sim + w_entity * e.entity_overlap
        )

    evid.sort(key=lambda e: e.retrieval_score, reverse=True)

    # ── Soft floor + top-k cap ────────────────────────────────────
    # Floor is deliberately low (0.05) — we want to pass articles to NLI
    # even if they have low entity overlap, as long as they are semantically
    # relevant to the claim topic.
    SOFT_FLOOR = 0.05
    accepted: List[EvidenceItem] = []
    rejected: List[EvidenceItem] = []

    for e in evid:
        if len(accepted) >= top_k:
            e.rejection_reason = (
                f"outside top-{top_k} "
                f"[LaBSE={e.labse_sim:.3f} TF-IDF={e.tfidf_sim:.3f} "
                f"entity={e.entity_overlap:.3f} combined={e.retrieval_score:.3f}]"
            )
            rejected.append(e)
        elif e.retrieval_score < SOFT_FLOOR:
            e.rejection_reason = (
                f"combined={e.retrieval_score:.3f} < floor={SOFT_FLOOR} "
                f"[LaBSE={e.labse_sim:.3f} TF-IDF={e.tfidf_sim:.3f} "
                f"entity={e.entity_overlap:.3f}]"
            )
            rejected.append(e)
        else:
            accepted.append(e)

    # Safety: never return empty
    if not accepted:
        for e in rejected:
            e.rejection_reason = ""
        accepted = sorted(evid, key=lambda e: e.labse_sim + e.tfidf_sim, reverse=True)[
            :5
        ]
        rejected = []

    return accepted, rejected


# ==============================
# 10b) MISMATCH DETECTION (location / person / date / org)
# ==============================
def detect_entity_mismatches(
    claim_info: Dict,
    evid: List[EvidenceItem],
) -> List[EvidenceItem]:
    """
    Run BEFORE NLI verdict aggregation (after NLI scoring).

    Detects articles where the claim's key entities are ABSENT but the
    article describes the same event with DIFFERENT entities — the classic
    contradiction pattern (claim: "event in Jharkhand", article: "Indonesia").

    When a high-LaBSE article (semantically very relevant) has zero entity
    overlap with the claim, that is strong contradiction evidence — not a
    reason to discard the article.

    Boost logic:
    - Each missing entity type (person / location / org) adds a boost
    - Boost is proportional to LaBSE similarity (high LaBSE + low entity
      overlap = high confidence contradiction)
    - Only fires when NLI hasn't already produced a strong signal
    """
    persons = {p.lower() for p in claim_info.get("person_names", [])}
    locations = {loc.lower() for loc in claim_info.get("locations", [])}
    orgs = {t.lower() for t, label in claim_info.get("entities", []) if label == "ORG"}

    for e in evid:
        art = f"{e.title} {e.snippet}".lower()
        mismatch_count = 0
        mismatch_log: List[str] = []

        if persons:
            missing = [p for p in persons if p not in art]
            if len(missing) == len(persons):
                mismatch_count += 1
                mismatch_log.append(f"person absent: {', '.join(missing)}")

        if locations:
            missing_locs = [loc for loc in locations if loc not in art]
            if len(missing_locs) == len(locations):
                mismatch_count += 1
                mismatch_log.append(f"location absent: {', '.join(missing_locs)}")

        if orgs:
            missing_orgs = [o for o in orgs if o not in art]
            if len(missing_orgs) == len(orgs):
                mismatch_count += 1
                mismatch_log.append(f"org absent: {', '.join(missing_orgs)}")

        if mismatch_count > 0:
            # Boost scales with LaBSE (semantic relevance) × mismatch count
            # High LaBSE + all entities absent = strong contradiction signal
            boost = min(0.35, e.labse_sim * 0.25 * mismatch_count)
            if boost > 0 and e.nli_ent < 0.55:
                e.nli_con = min(1.0, e.nli_con + boost)
                if e.nli_con > e.nli_ent and e.nli_con > e.nli_neu:
                    e.nli_label = "CONTRADICTION"

    return evid


# ==============================
# 11) NLI (BART-large-MNLI)
# ==============================
# 11) NLI (BART-large-MNLI) + keyword fallback
# ==============================
_CONTRADICT_WORDS = {
    "denies",
    "denied",
    "refutes",
    "refuted",
    "debunks",
    "debunked",
    "false",
    "fake",
    "hoax",
    "misleading",
    "wrong",
    "incorrect",
    "no evidence",
    "not true",
    "ruled out",
    "dismisses",
    "dismissed",
    "acquitted",
    "cleared",
    "did not",
    "hasn't",
    "have not",
    "has not",
    "is not",
    "are not",
    "was not",
    "were not",
}
_SUPPORT_WORDS = {
    "confirms",
    "confirmed",
    "announces",
    "announced",
    "reveals",
    "revealed",
    "wins",
    "won",
    "signs",
    "signed",
    "launches",
    "launched",
    "achieves",
    "achieved",
    "elected",
    "appointed",
    "resigns",
    "resigned",
    "retires",
    "retired",
    "passes",
    "passed",
    "dies",
    "dead",
    "arrested",
    "charged",
    "convicted",
    "found guilty",
    "scores",
    "sets record",
}


def _keyword_nli_fallback(claim: str, article_text: str) -> Dict[str, float]:
    """
    Lightweight keyword-based NLI used when BART-MNLI is unavailable.
    Much better than returning neutral=1.0 for everything.
    """
    art = article_text.lower()
    claim_l = claim.lower()
    con_hits = sum(1 for w in _CONTRADICT_WORDS if w in art)
    sup_hits = sum(1 for w in _SUPPORT_WORDS if w in art)
    claim_words = set(re.findall(r"\b[a-z]{4,}\b", claim_l))
    art_words = set(re.findall(r"\b[a-z]{4,}\b", art))
    overlap = len(claim_words & art_words) / max(len(claim_words), 1)
    ent = min(0.85, 0.20 + overlap * 0.45 + sup_hits * 0.07)
    con = min(0.85, con_hits * 0.15)
    if ent < 0.35 and con < 0.20 and overlap > 0.3:
        ent = 0.45
    neu = max(0.05, 1.0 - ent - con)
    tot = ent + con + neu
    return {
        "entailment": round(ent / tot, 4),
        "contradiction": round(con / tot, 4),
        "neutral": round(neu / tot, 4),
    }


def run_nli(premise: str, hypothesis: str) -> Optional[Dict[str, float]]:
    """Single NLI pass. Returns probs dict or None if model unavailable."""
    if not (torch and NLI_TOK and NLI_M):
        return None
    try:
        inputs = NLI_TOK(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
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
        return None


def _score_article_nli(claim: str, article_text: str) -> Dict[str, float]:
    """
    Two-pass NLI scoring. Uses the article as premise and asks two
    targeted yes/no questions about the claim. This avoids the structural
    neutral problem of single-pass headline-vs-headline NLI.

    Pass 1: "This text supports the claim: {claim}"  → ent = support evidence
    Pass 2: "This text contradicts the claim: {claim}" → ent = contradict evidence

    Falls back to keyword scoring if BART is unavailable.
    """
    bart_ok = bool(torch and NLI_TOK and NLI_M)

    if bart_ok:
        h_sup = f"This text supports the claim: {claim}"
        h_con = f"This text contradicts the claim: {claim}"
        s1 = run_nli(premise=article_text, hypothesis=h_sup)
        s2 = run_nli(premise=article_text, hypothesis=h_con)
        if s1 is not None and s2 is not None:
            ent = s1["entailment"]
            con = s2["entailment"]
            neu = max(0.05, 1.0 - ent - con)
            tot = ent + con + neu
            return {
                "entailment": round(ent / tot, 4),
                "contradiction": round(con / tot, 4),
                "neutral": round(neu / tot, 4),
            }

    # Keyword fallback
    return _keyword_nli_fallback(claim, article_text)


def annotate_nli(
    claim: str, evid: List[EvidenceItem], max_items: int = 12
) -> List[EvidenceItem]:
    """
    Score articles using two-pass BART-MNLI (or keyword fallback).
    Processes the top max_items by retrieval_score.
    Stores per-article debug info in session state for the debug panel.
    """
    if not evid:
        return evid

    if not (torch and NLI_TOK and NLI_M):
        _nli_err = st.session_state.get("_nli_load_error", "")
        st.warning(
            "⚠️ BART-MNLI not loaded — using keyword-based NLI fallback. "
            "Verdicts will be less precise but still directional."
            + (f"\n\nLoad error: `{_nli_err}`" if _nli_err else "")
        )

    sorted_e = sorted(evid, key=lambda x: x.retrieval_score, reverse=True)[:max_items]
    # NEUTRAL only wins when it leads both ent and con by > 15pp
    _NEUTRAL_MARGIN = 0.15

    nli_debug: List[Dict] = []

    for e in sorted_e:
        article_text = f"{e.title}. {e.snippet}".strip(" .")
        if not article_text:
            article_text = e.title

        out = _score_article_nli(claim, article_text)

        e.nli_ent = out["entailment"]
        e.nli_con = out["contradiction"]
        e.nli_neu = out["neutral"]

        best = max(e.nli_ent, e.nli_con)
        if e.nli_neu - best >= _NEUTRAL_MARGIN:
            e.nli_label = "NEUTRAL"
        elif e.nli_ent >= e.nli_con:
            e.nli_label = "ENTAILMENT"
        else:
            e.nli_label = "CONTRADICTION"

        nli_debug.append(
            {
                "title": e.title[:70],
                "retrieval": round(e.retrieval_score, 3),
                "LaBSE": round(e.labse_sim, 3),
                "TF-IDF": round(e.tfidf_sim, 3),
                "ent": round(e.nli_ent, 3),
                "con": round(e.nli_con, 3),
                "neu": round(e.nli_neu, 3),
                "label": e.nli_label,
            }
        )

    st.session_state["_nli_debug"] = nli_debug

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
    Compute verdict from NLI-scored evidence.

    Aggregation rules:
    - ENTAILMENT  → full ent_score added to w_ent
    - CONTRADICTION → full con_score added to w_con
    - NEUTRAL → partial bleed only if raw score >= 0.30, at 0.25× weight
      (reduced from 0.5× — prevents many weak neutral articles drowning
       a single strong support/contradiction signal)

    Single-article high-confidence override:
    - If any article has ent >= 0.65 or con >= 0.65, that alone produces
      a verdict regardless of other articles. One very relevant article
      that strongly supports or contradicts is conclusive.

    Minimum-signal override:
    - If total_w < 0.10 (nearly all neutral), but at least one article
      has max(ent, con) >= 0.40, use the stronger of the two.
    """
    w_ent = 0.0
    w_con = 0.0
    max_ent = 0.0
    max_con = 0.0

    for e in evidence_data:
        nli = e.get("nli", {})
        label = nli.get("label", "NEUTRAL")
        ent = float(nli.get("entailment", 0.0))
        con = float(nli.get("contradiction", 0.0))
        max_ent = max(max_ent, ent)
        max_con = max(max_con, con)

        if label == "ENTAILMENT":
            w_ent += ent
        elif label == "CONTRADICTION":
            w_con += con
        else:
            # Neutral: only partial bleed for meaningful raw scores
            if ent >= 0.30:
                w_ent += ent * 0.25
            if con >= 0.30:
                w_con += con * 0.25

    total_w = w_ent + w_con

    # ── High-confidence single-article override ───────────────────
    HIGH = 0.65
    if max_con >= HIGH and max_con > max_ent:
        return "CONTRADICTED", round(max_con, 3), "high_conf"
    if max_ent >= HIGH and max_ent > max_con:
        return "SUPPORTED", round(max_ent, 3), "high_conf"

    # ── Minimum-signal override (nearly-all-neutral pool) ─────────
    if total_w < 0.10:
        MID = 0.40
        if max_con >= MID and max_con > max_ent:
            return "CONTRADICTED", round(max_con, 3), "min_signal"
        if max_ent >= MID and max_ent > max_con:
            return "SUPPORTED", round(max_ent, 3), "min_signal"
        return "UNVERIFIED", 0.0, "nli"

    # ── Weighted majority ─────────────────────────────────────────
    if w_con > w_ent:
        return "CONTRADICTED", round(w_con / total_w, 3), "nli"
    elif w_ent > w_con:
        return "SUPPORTED", round(w_ent / total_w, 3), "nli"
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
            rationale = "Retrieved news articles contradict this claim."
        elif nli_verdict == "SUPPORTED":
            rationale = "Retrieved news articles support this claim."
        else:
            rationale = "Evidence is mixed or insufficient to verify this claim."

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
    rejected: Optional[List[EvidenceItem]] = None,
):
    """Expandable panel showing extracted entities, generated queries,
    per-article retrieval scores, and rejection log."""
    with st.expander("🔍 Retrieval Diagnostics", expanded=False):
        # ── Extracted entities & keywords ──
        if claim_info:
            ents = claim_info.get("entities", [])
            kws = claim_info.get("keywords", [])
            all_terms = [t for t, _ in ents] + kws
            if all_terms:
                st.markdown(
                    "**Keywords:** " + " · ".join(f"`{t}`" for t in all_terms[:15])
                )
            else:
                st.caption("No keywords detected.")
            st.divider()

        # ── Generated queries ──
        st.markdown("#### Generated Search Queries")
        for i, q in enumerate(queries, 1):
            st.markdown(f"`{i}.` {q}")

        st.divider()

        # ── Per-article accepted scores ──
        st.markdown("#### Accepted Articles — Retrieval Scores")
        rows = []
        for e in evid_sorted[:20]:
            rows.append(
                {
                    "Article": e.title[:55] + "…" if len(e.title) > 55 else e.title,
                    "Source": e.source,
                    "Combined": round(e.retrieval_score, 3),
                    "LaBSE": round(e.labse_sim, 3),
                    "TF-IDF": round(e.tfidf_sim, 3),
                    "Entity": round(e.entity_overlap, 2),
                    "NLI": e.nli_label,
                    "Query": (e.query_used[:40] + "…")
                    if len(e.query_used) > 40
                    else e.query_used,
                }
            )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── Rejection log ──
        if rejected:
            st.divider()
            st.markdown("#### Rejected Articles — Why They Were Filtered")
            rej_rows = []
            for e in rejected[:20]:
                rej_rows.append(
                    {
                        "Article": e.title[:55] + "…" if len(e.title) > 55 else e.title,
                        "Source": e.source,
                        "Combined": round(e.retrieval_score, 3),
                        "LaBSE": round(e.labse_sim, 3),
                        "TF-IDF": round(e.tfidf_sim, 3),
                        "Entity": round(e.entity_overlap, 2),
                        "Reason": e.rejection_reason,
                    }
                )
            st.dataframe(
                pd.DataFrame(rej_rows), use_container_width=True, hide_index=True
            )


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
    # Show load errors for key models
    _gemma_err = st.session_state.get("_gemma_load_error", "")
    _nli_err = st.session_state.get("_nli_load_error", "")
    if _gemma_err:
        st.error(f"Gemma failed to load: `{_gemma_err}`")
    if _nli_err:
        st.error(f"BART-MNLI failed to load: `{_nli_err}`")

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
            # Use file name + size as a stable identity key — if the same file
            # is still in the widget after a rerun (e.g. user edited the claim
            # text), skip transcription entirely and show the cached result.
            _file_key = f"{audio_file.name}_{audio_file.size}"
            _cached = st.session_state.get("_whisper_cache", {})

            if _file_key in _cached:
                # Already transcribed — just surface the result, no Whisper call
                whisper_result = _cached[_file_key]
            else:
                with st.spinner(
                    f"Transcribing with Whisper {_chosen_model} ({_audio_lang[1]})…"
                ):
                    import tempfile

                    suffix = os.path.splitext(audio_file.name)[-1] or ".wav"
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix
                    ) as tmp:
                        tmp.write(audio_file.read())
                        tmp_path = tmp.name
                    whisper_result = transcribe_audio(
                        tmp_path,
                        language=_lang_code if _lang_code != "auto" else None,
                    )
                    st.session_state["_audio_tmp_path"] = tmp_path
                    # Cache so reruns don't re-transcribe the same file
                    st.session_state["_whisper_cache"] = {_file_key: whisper_result}

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

        # ── Entity debug — show extracted keywords ──
        _ents = claim_info.get("entities", [])
        _kws = claim_info.get("keywords", [])
        _all_terms = [t for t, _ in _ents] + _kws
        if _all_terms:
            st.caption("🏷️ Keywords: " + " · ".join(f"`{t}`" for t in _all_terms[:12]))
        else:
            st.warning(
                "⚠️ No keywords extracted — entity-overlap filter will be skipped. "
                + (
                    "spaCy loaded OK but found nothing."
                    if _NLP is not None
                    else "spaCy not available, using regex fallback."
                )
            )

        with st.spinner("Generating expanded search queries via Gemma…"):
            evid, queries = build_evidence_pool(claim, claim_info=claim_info)

        st.caption(
            f"Generated {len(queries)} search queries → retrieved {len(evid)} articles (relevance-ranked)"
        )

        # ── Fetch debug: show per-query article counts ───────────
        fetch_debug = st.session_state.get("_fetch_debug", [])
        if fetch_debug and len(evid) == 0:
            with st.expander(
                "⚠️ Fetch Debug — click to see why 0 articles returned", expanded=True
            ):
                for q, cnt in fetch_debug:
                    st.markdown(f"- `{q[:80]}` → **{cnt}** articles")

        with st.spinner("TF-IDF scoring…"):
            evid = compute_tfidf_sims(claim, evid)

        with st.spinner("Combined scoring: LaBSE + TF-IDF + Entity Overlap…"):
            evid, _rejected_evid = score_and_rank_evidence(
                claim,
                evid,
                claim_info.get("entity_texts", set()),
                top_k=30,
            )
            st.session_state["_rejected_evid"] = _rejected_evid

        st.caption(
            f"{len(evid)} articles selected by combined score "
            f"({len(_rejected_evid)} rejected — see Retrieval Diagnostics)"
        )

        with st.spinner("Running NLI verification (BART-MNLI)…"):
            evid = annotate_nli(claim, evid)

        with st.spinner("Mismatch detection (person / location / date)…"):
            evid = detect_entity_mismatches(claim_info, evid)

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

        # Sort by strongest NLI signal first, then by retrieval score as tiebreak.
        # This ensures the articles that most clearly support or contradict
        # are at the top of evid_pack — not just the highest LaBSE/TF-IDF ones.
        evid_sorted = sorted(
            evid,
            key=lambda x: (max(x.nli_ent, x.nli_con), x.retrieval_score),
            reverse=True,
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
                "retrieval_score": round(e.retrieval_score, 4),
                "nli": {
                    "label": e.nli_label,
                    "entailment": round(e.nli_ent, 4),
                    "contradiction": round(e.nli_con, 4),
                    "neutral": round(e.nli_neu, 4),
                },
            }
            for e in evid_sorted[:12]
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
                        f'<div style="fnt-size:0.65rem;color:#8b949e;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px;">Image Integrity</div>'
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
        show_retrieval_telemetry(
            queries,
            evid_sorted,
            claim_info=claim_info,
            rejected=st.session_state.get("_rejected_evid", []),
        )

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
