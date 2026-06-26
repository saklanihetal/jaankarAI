# =========================================================
# JAANKAAR AI (FINAL): Multimodal Fake News Detection
# - CLIP + NLI + Deepfake + AWS Rekognition
# - NewsAPI + Google RSS (Context Consensus)
# - Gates + TF-IDF + Lexicon
# - FINAL VERDICT: Gemma 2B Instruct (local, Hugging Face)
# =========================================================

import os
import re
import io
import json
import base64
from dataclasses import dataclass
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
    )
except Exception:
    CLIPProcessor = None
    CLIPModel = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    AutoModelForCausalLM = None
    BitsAndBytesConfig = None

# Audio lib kept (not called in this update)
try:
    from gtts import gTTS
except ImportError:
    gTTS = None


# ==============================
# 0) CONFIG + SECRET LOADING
# ==============================
st.set_page_config(
    page_title="Jaankaar AI | Fake News Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_secret(name: str, default: str = "") -> str:
    """Streamlit secrets first, then environment variables."""
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
APP_VERSION = "JAANKAAR-AI-V0"


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
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    transition: transform 0.2s ease;
}
.css-card:hover { border-color: rgba(255, 255, 255, 0.3); transform: translateY(-2px); }

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
@st.cache_resource
def load_core_models():
    device = "cpu"
    if torch and torch.cuda.is_available():
        device = "cuda"

    clip_model, clip_processor = None, None
    if CLIPModel and CLIPProcessor:
        try:
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            if torch:
                clip_model = clip_model.to(device)
        except Exception:
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
        except Exception:
            nli_tok, nli_model = None, None

    return device, clip_model, clip_processor, nli_tok, nli_model


DEVICE, CLIP_M, CLIP_P, NLI_TOK, NLI_M = load_core_models()


@st.cache_resource
def load_aux_models():
    df_model = None
    model_path = "models/weights/deepfake/deepfake_cnn.h5"
    if tf and os.path.exists(model_path):
        try:
            try:
                tf.keras.config.enable_legacy_deserialization()
            except Exception:
                pass
            df_model = tf.keras.models.load_model(model_path, compile=False)
        except Exception as e:
            print("Deepfake load error:", e)

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

    support_gate = load_safe_model("support_gate.joblib")
    contra_gate = load_safe_model("contradiction_gate.joblib")
    style_tfidf = load_safe_model("tfidf_style_model.joblib")

    return df_model, rek_client, support_gate, contra_gate, style_tfidf


DF_M, REK, GATE_S, GATE_C, STYLE_TFIDF = load_aux_models()
print("Deepfake model loaded:", DF_M is not None)


# ==============================
# 4b) LOAD GEMMA 2B INSTRUCT
# ==============================
@st.cache_resource(show_spinner=False)
def load_gemma_model():
    """
    Load Gemma 2B Instruct from Hugging Face.

    Authentication
    --------------
    The model is gated on Hugging Face. You must:
      1. Accept the licence at https://huggingface.co/google/gemma-2b-it
      2. Create a read-token at https://huggingface.co/settings/tokens
      3. Set it as HF_TOKEN in .streamlit/secrets.toml  OR  as an env var.

    Memory notes
    ------------
    - GPU with bitsandbytes  : loaded in 4-bit (~1.5 GB VRAM)
    - GPU without bitsandbytes: loaded in float16 (~5 GB VRAM)
    - CPU only               : loaded in float32 (~8 GB RAM, slow)
    """
    if not (torch and AutoTokenizer and AutoModelForCausalLM):
        return None, None

    token = HF_TOKEN or None  # None → unauthenticated (will fail for gated models)

    try:
        tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL_ID, token=token)

        load_kwargs: Dict[str, Any] = {
            "token": token,
            "low_cpu_mem_usage": True,
        }

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
            # CPU — full precision
            load_kwargs["torch_dtype"] = torch.float32

        model = AutoModelForCausalLM.from_pretrained(GEMMA_MODEL_ID, **load_kwargs)
        model.eval()
        return tokenizer, model

    except Exception as exc:
        st.warning(
            f"⚠️ Gemma model could not be loaded: {exc}\n\n"
            "Make sure HF_TOKEN is set and you have accepted the Gemma licence on "
            "huggingface.co. The reasoning engine will show UNVERIFIED until fixed."
        )
        return None, None


GEMMA_TOK, GEMMA_M = load_gemma_model()


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
# 6) CONTEXT CONSENSUS
# ==============================
def fetch_newsapi_evidence(query: str, k: int = 10) -> List[EvidenceItem]:
    if not NEWS_API_KEY:
        return []
    q = requests.utils.quote(query)
    url = (
        f"https://newsapi.org/v2/everything?q={q}&language=en"
        f"&sortBy=relevancy&pageSize={k}&apiKey={NEWS_API_KEY}"
    )
    r = safe_get(url, timeout=14)
    if not r or not r.ok:
        return []
    try:
        data = r.json()
        items = []
        for a in data.get("articles", [])[:k]:
            title = norm_text(a.get("title", ""))
            link = a.get("url", "")
            source = (a.get("source") or {}).get("name", "NewsAPI")
            desc = norm_text(a.get("description", "") or "")
            img = a.get("urlToImage", None)
            if title:
                items.append(
                    EvidenceItem(
                        title=title, url=link, source=source, snippet=desc, img_url=img
                    )
                )
        return items
    except Exception:
        return []


def fetch_google_rss_evidence(query: str, k: int = 10) -> List[EvidenceItem]:
    if not feedparser:
        return []
    q = requests.utils.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(rss_url)
    items = []
    for e in (feed.entries or [])[:k]:
        title = norm_text(getattr(e, "title", "") or "")
        link = getattr(e, "link", "") or ""
        if title:
            items.append(EvidenceItem(title=title, url=link, source="Google RSS"))
    return items


def fetch_newsdata_evidence(query: str, k: int = 10) -> List[EvidenceItem]:
    if not NEWSDATA_KEY:
        return []
    q = requests.utils.quote(query)
    url = f"https://newsdata.io/api/1/news?apikey={NEWSDATA_KEY}&q={q}&language=en"
    r = safe_get(url, timeout=14)
    if not r or not r.ok:
        return []
    try:
        data = r.json()
        items = []
        for res in data.get("results", [])[:k]:
            title = norm_text(res.get("title", ""))
            img = res.get("image_url", None)
            if title:
                items.append(
                    EvidenceItem(
                        title=title,
                        url=res.get("link", ""),
                        source="NewsData.io",
                        snippet=norm_text(res.get("description", "")),
                        img_url=img,
                    )
                )
        return items
    except Exception:
        return []


def fetch_eventregistry_evidence(query: str, k: int = 10) -> List[EvidenceItem]:
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
        data = r.json()
        articles = data.get("articles", {}).get("results", [])
        items = []
        for a in articles[:k]:
            title = norm_text(a.get("title", ""))
            img = a.get("image", None)
            if title:
                items.append(
                    EvidenceItem(
                        title=title,
                        url=a.get("url", ""),
                        source=a.get("source", {}).get("title", "Event Registry"),
                        img_url=img,
                    )
                )
        return items
    except Exception:
        return []


def build_evidence_pool(claim: str, k_total: int = 12) -> List[EvidenceItem]:
    pool = (
        fetch_newsapi_evidence(claim, k=10)
        + fetch_google_rss_evidence(claim, k=10)
        + fetch_newsdata_evidence(claim, k=10)
        + fetch_eventregistry_evidence(claim, k=10)
    )
    seen, out = set(), []
    for it in pool:
        key = it.title.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
        if len(out) >= k_total:
            break
    return out


# ==============================
# 7) TF-IDF similarity
# ==============================
def compute_tfidf_sims(claim: str, evid: List[EvidenceItem]) -> List[EvidenceItem]:
    if not (TfidfVectorizer and cosine_similarity) or not evid:
        return evid
    corpus = [claim] + [e.title for e in evid]
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
# 8) NLI (claim vs evidence title)
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
    if not evid:
        return evid
    sorted_e = sorted(evid, key=lambda x: x.tfidf_sim, reverse=True)[:max_items]
    for e in sorted_e:
        out = run_nli(premise=e.title, hypothesis=claim)
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
# 9) CLIP (image-text alignment)
# ==============================
def clip_relevance(image: Image.Image, text: str) -> float:
    if not (torch and CLIP_M and CLIP_P) or image is None or not text:
        return 0.0
    try:
        inputs = CLIP_P(text=[text], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            score = torch.sigmoid(CLIP_M(**inputs).logits_per_image)[0][0].item()
        return clamp01(score)
    except Exception:
        return 0.0


# ==============================
# 10) DEEPFAKE (keras cnn)
# ==============================
def deepfake_prob_from_image(image: Image.Image) -> float:
    if DF_M is None or tf is None or image is None:
        return 0.0
    try:
        img = image.convert("RGB").resize((224, 224))
        arr = np.expand_dims(np.array(img).astype("float32") / 255.0, axis=0)
        pred = DF_M.predict(arr, verbose=0)
        return clamp01(float(pred.flatten()[0]))
    except Exception:
        return 0.0


# ==============================
# 11) AWS Rekognition
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
# 12) GATES (support/contradiction)
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
    claim: str, evid: List[EvidenceItem], clip_s: float, df_p: float, lex_s: float
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
# 13) GEMMA REASONING ENGINE
# ==============================


def _build_gemma_prompt(claim: str, evidence_titles: List[str]) -> str:
    """
    Construct the Gemma Instruct-format prompt.
    The model is asked to respond with a strict JSON object so the rest of the
    pipeline can parse it identically to the previous Gemini integration.
    """
    ev_block = (
        "\n".join(f"- {t}" for t in evidence_titles[:8] if t)
        or "No evidence retrieved."
    )

    return (
        "<start_of_turn>user\n"
        "You are a strict fact-checking assistant (knowledge cutoff: January 2026).\n"
        "Analyse the CLAIM against the EVIDENCE HEADLINES and decide whether the claim "
        "is SUPPORTED or CONTRADICTED.\n\n"
        "Rules:\n"
        "1. IDENTITY — if the news is about a replica/store model/tribute (e.g. a "
        "Statue of Liberty replica in Brazil) but the claim refers to the original "
        "landmark, the verdict is CONTRADICTED.\n"
        "2. ACTIVITY — if evidence shows a person is active in a sport/role in "
        "Jan 2026, any retirement claim for that person is CONTRADICTED.\n"
        "3. PROXIMITY — if an explosion occurred at the gates/grounds of a site, "
        "'blast at [site]' is SUPPORTED.\n"
        "4. Never use the label MISLEADING — only SUPPORTED or CONTRADICTED.\n"
        "5. Respond ONLY with a valid JSON object — no preamble, no markdown fences.\n\n"
        f'CLAIM: "{claim}"\n\n'
        f"EVIDENCE HEADLINES:\n{ev_block}\n\n"
        "Return exactly:\n"
        '{"verdict": "SUPPORTED" or "CONTRADICTED", '
        '"confidence": <float 0.0–1.0>, '
        '"rationale": "<one concise sentence stating the factual reason>"}\n'
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def backend_reasoner(payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any], str]:
    """
    Final reasoning layer powered by Gemma 2B Instruct (local Hugging Face).

    Return signature (unchanged from the previous Gemini integration):
        (verdict: str, structured: dict, rationale: str)
    """
    if GEMMA_TOK is None or GEMMA_M is None:
        return (
            "UNVERIFIED",
            {},
            "Reasoning engine offline — Gemma model not loaded. "
            "Set HF_TOKEN and accept the Gemma licence on huggingface.co.",
        )

    claim = payload.get("claim", "")
    evidence_titles = [e.get("title", "") for e in payload.get("evidence", [])]
    prompt_text = _build_gemma_prompt(claim, evidence_titles)

    try:
        inputs = GEMMA_TOK(prompt_text, return_tensors="pt")
        model_device = next(GEMMA_M.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = GEMMA_M.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,  # greedy → deterministic
                pad_token_id=GEMMA_TOK.eos_token_id,
            )

        # Decode only the newly generated portion (skip echoed prompt)
        prompt_len = inputs["input_ids"].shape[-1]
        raw_text = GEMMA_TOK.decode(
            output_ids[0][prompt_len:], skip_special_tokens=True
        ).strip()

        # Strip accidental markdown fences that some quantised models add
        clean = re.sub(r"```(?:json)?|```", "", raw_text).strip()

        # Parse JSON
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            # Try to extract a JSON object from surrounding text
            match = re.search(r"\{.*?\}", clean, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
            else:
                # Last-resort heuristic from raw text
                if "supported" in clean.lower():
                    verdict_guess = "SUPPORTED"
                elif "contradict" in clean.lower():
                    verdict_guess = "CONTRADICTED"
                else:
                    verdict_guess = "UNVERIFIED"
                return verdict_guess, {}, clean[:300]

        verdict = data.get("verdict", "UNVERIFIED").upper().strip()
        if verdict not in {"SUPPORTED", "CONTRADICTED"}:
            verdict = "UNVERIFIED"

        return verdict, data, data.get("rationale", raw_text[:200])

    except Exception as exc:
        return "ERROR", {}, f"Gemma inference error: {str(exc)}"


def translate_to_english(text: str) -> str:
    if not text or not GoogleTranslator:
        return text
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text


# ==============================
# 14) LANDING PAGE
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
                    This system supports regional languages and integrates text based fake news detection,
                    text-image consistency analysis and image based deepfake detection.
                    By jointly analyzing text and images the proposed system improves the reliability
                    of fake news detection in regional contexts.
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

    # ── Updated team list ──────────────────────────────────────────────
    team_names = [
        "Kartik Shekhar",
        "K V Nihal Mouni",
        "Kartik Kaul",
        "Hetal Saklani",
    ]
    # ──────────────────────────────────────────────────────────────────

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
# 15) DETECTOR PAGE
# ==============================
def show_detector_page():
    st.title("AI based Regional Fake News Detection")
    st.caption(f"System Version: {APP_VERSION}")

    left, right = st.columns([1.5, 1], gap="large")

    with left:
        st.markdown(
            '<div class="css-card"><h3 style="margin-top:0">Analysis Input</h3>'
            '<p style="opacity:0.8">Enter a suspicious headline or claim. '
            "You may also upload a related image.</p>",
            unsafe_allow_html=True,
        )
        claim = st.text_area(
            "Claim / Headline",
            height=100,
            placeholder="e.g. Statue of Liberty collapsed in NY storm...",
        )
        img_file = st.file_uploader(
            "Evidence Image (Optional)", type=["jpg", "jpeg", "png"]
        )
        st.markdown("</div>", unsafe_allow_html=True)
        run_btn = st.button(
            "INITIATE ANALYSIS", type="primary", use_container_width=True
        )

    with right:
        st.markdown(
            '<div class="css-card"><h3 style="margin-top:0">System Status</h3>',
            unsafe_allow_html=True,
        )
        status_rows = [
            ("CLIP", "ONLINE" if (CLIP_M and CLIP_P and torch) else "OFFLINE"),
            ("NLI Engine", "ONLINE" if (NLI_M and NLI_TOK and torch) else "OFFLINE"),
            ("NewsAPI", "KEY LOADED" if NEWS_API_KEY else "NO KEY"),
            (
                "Decision Engine (Gemma)",
                "READY" if (GEMMA_M and GEMMA_TOK) else "UNAVAILABLE",
            ),
        ]
        st.dataframe(
            pd.DataFrame(status_rows, columns=["Module", "State"]),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if run_btn:
        claim_original = norm_text(claim)
        if not claim_original:
            st.error("Input required: Please enter a claim.")
            st.stop()

        with st.spinner("Translating input stream..."):
            claim_en = translate_to_english(claim_original)
            claim = claim_en
        if claim != claim_original:
            st.info(f"Translated: {claim}")

        with st.spinner("Scanning global media sources..."):
            evid = build_evidence_pool(claim, k_total=12)
            evid = compute_tfidf_sims(claim, evid)
            evid = annotate_nli(claim, evid)

        image, img_bytes = None, b""
        if img_file:
            try:
                img_bytes = img_file.read()
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                image = None

        with st.spinner("Computing multimodal vectors..."):
            clip_s = clip_relevance(image, claim) if image else 0.0
            df_p = deepfake_prob_from_image(image) if image else 0.0
            rek = rekognition_signals(img_bytes) if img_bytes else {"available": False}
            lex_s = lexicon_score(claim)
            feats = build_gate_features(claim, evid, clip_s, df_p, lex_s)
            p_support = gate_predict(GATE_S, feats)
            p_contra = gate_predict(GATE_C, feats)

        evid_sorted = sorted(evid, key=lambda x: (x.tfidf_sim, x.nli_ent), reverse=True)

        best_evidence_image = next((e.img_url for e in evid_sorted if e.img_url), None)

        evid_pack = [
            {
                "title": e.title,
                "source": e.source,
                "url": e.url,
                "snippet": e.snippet,
                "tfidf_sim": round(e.tfidf_sim, 4),
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
            },
        }

        with st.spinner("Deliberating final verdict via Gemma..."):
            verdict, structured, rationale = backend_reasoner(
                {"claim": claim, "evidence": evid_pack}
            )

        # ── Results UI ────────────────────────────────────────────────
        st.markdown("---")
        res_col1, res_col2 = st.columns([2, 1])

        with res_col1:
            st.subheader("Final Verdict")
            color_map = {
                "SUPPORTED": "#38ef7d",
                "CONTRADICTED": "#ef473a",
            }
            color = color_map.get(verdict, "#ffb000")
            st.markdown(
                f'<div class="css-card" style="border-left: 5px solid {color};">'
                f'<h2 style="color:{color}; margin:0;">{verdict}</h2>'
                f"<p>{rationale}</p></div>",
                unsafe_allow_html=True,
            )

        with res_col2:
            st.subheader("Evidence Context Image")
            if best_evidence_image:
                st.image(
                    best_evidence_image,
                    caption="Retrieved from Top Evidence Source",
                    use_container_width=True,
                )
            elif image:
                st.image(image, caption="User Uploaded Image", use_container_width=True)
            else:
                st.info("No visual context available.")

        st.subheader("Key Evidence Sources")
        if evid_sorted:
            ec1, ec2 = st.columns(2)
            for i, e in enumerate(evid_sorted[:6]):
                col = ec1 if i % 2 == 0 else ec2
                nli_color = (
                    "#38ef7d"
                    if e.nli_label == "ENTAILMENT"
                    else "#ef473a"
                    if e.nli_label == "CONTRADICTION"
                    else "#8b949e"
                )
                snip = (e.snippet[:150] + "...") if len(e.snippet) > 150 else e.snippet
                col.markdown(
                    f"""
                    <div class="css-card">
                        <div style="font-weight:bold; font-size:1.1rem; color:#fff; margin-bottom:5px;">
                            <a href="{e.url}" target="_blank"
                               style="text-decoration:none; color:#58a6ff;">{e.title}</a>
                        </div>
                        <div style="margin-bottom:8px;">
                            <span class="custom-badge">{e.source}</span>
                            <span style="font-size:0.8rem; color:{nli_color};
                                         border:1px solid {nli_color};
                                         padding:2px 6px; border-radius:4px;">{e.nli_label}</span>
                        </div>
                        <div style="font-size:0.9rem; color:#8b949e;">{snip}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.subheader("Signal Telemetry")
        with st.expander("View Underlying Data (Explainability)", expanded=False):
            st.markdown("#### Context Metrics")
            m1, m2, m3, m4 = st.columns(4)
            ctx = payload["signals"]["context"]
            m1.metric("TF-IDF Max", f"{ctx['tfidf_max']:.2f}")
            m2.metric("TF-IDF Mean", f"{ctx['tfidf_mean']:.2f}")
            m3.metric("NLI Entailment Max", f"{ctx['nli_ent_max']:.2f}")
            m4.metric("NLI Contradiction Max", f"{ctx['nli_con_max']:.2f}")

            st.divider()
            st.markdown("#### Style & Visual Analysis")
            s1, s2, s3, s4 = st.columns(4)
            style = payload["signals"]["style"]
            img_sig = payload["signals"]["image"]

            clip_val = (
                f"{img_sig['clip_relevance']:.2f}" if img_sig["has_image"] else "N/A"
            )
            deepfake_val = (
                f"{img_sig['deepfake_prob']:.2f}" if img_sig["has_image"] else "N/A"
            )

            rek_data = img_sig.get("rekognition", {})
            rek_celebs = rek_data.get("celebrities", [])
            rek_labels = rek_data.get("labels", [])
            rek_val = (
                rek_celebs[0]["name"]
                if rek_celebs
                else rek_labels[0]["name"]
                if rek_labels
                else "None Detected"
            )

            s1.metric("Lexicon Score", f"{style['lexicon']:.2f}")
            s2.metric("Uppercase Ratio", f"{style['basic']['upper_ratio']:.2f}")
            s3.metric("CLIP Relevance", clip_val)
            s4.metric("Deepfake Prob", deepfake_val)
            st.metric("Entity Recognized", rek_val)


# ==============================
# 16) MAIN APP ROUTER
# ==============================
def main():
    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio("Go to", ["Home", "Run Detector"], label_visibility="collapsed")
        st.markdown("---")
        st.markdown("### About")
        st.info(
            "Jaankaar AI proposes a Multi-modal Multilingual Regional Fake News Detection System. "
            "It integrates text based fake news detection, text-image consistency analysis and "
            "image based deepfake detection to improve reliability in regional contexts."
        )

    if page == "Home":
        show_landing_page()
    else:
        show_detector_page()


if __name__ == "__main__":
    main()
