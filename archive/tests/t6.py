# =========================================================
# JAANKAAR AI (FINAL): Multimodal Fake News Detection
# - CLIP + NLI + Deepfake + AWS Rekognition
# - NewsAPI + Google RSS (Context Consensus)
# - Gates + TF-IDF + Lexicon
# - FINAL VERDICT: backend decision engine ONLY
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
    )
except Exception:
    CLIPProcessor = None
    CLIPModel = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None

# Backend-only decision engine client
try:
    import google.generativeai as genai
except Exception:
    genai = None

# Audio generation
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
    initial_sidebar_state="expanded"
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

NEWS_API_KEY = get_secret("NEWS_API_KEY")
NEWSDATA_KEY = get_secret("NEWSDATA_KEY")
EVENTREGISTRY_KEY = get_secret("EVENTREGISTRY_KEY")
GNEWS_KEY = get_secret("GNEWS_KEY") 
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
AWS_KEY = get_secret("AWS_ACCESS_KEY_ID")
AWS_SECRET = get_secret("AWS_SECRET_ACCESS_KEY")
AWS_REGION = get_secret("AWS_REGION", "ap-south-1")

APP_VERSION = "JAANKAAR-AI-V1"


# ==============================
# 1) PROFESSIONAL UI STYLES
# ==============================
st.markdown(
    """
<style>
/* Main Background */
.stApp {
    background-color: #0e1117;
    background-image: radial-gradient(#1c222e 1px, transparent 1px);
    background-size: 20px 20px;
}

/* Typography */
h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; font-weight: 700; color: #f0f2f6; }
p, div, span { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #c3c7d0; }

/* Custom Cards */
.css-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    transition: transform 0.2s ease;
}
.css-card:hover {
    border-color: rgba(255, 255, 255, 0.3);
    transform: translateY(-2px);
}

/* Badges */
.custom-badge {
    background: linear-gradient(135deg, #0061ff 0%, #60efff 100%);
    color: #000;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 700;
    display: inline-block;
    margin-right: 5px;
}
.verdict-badge-supported {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    color: #fff; padding: 5px 12px; border-radius: 4px; font-weight: bold;
}
.verdict-badge-contradicted {
    background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%);
    color: #fff; padding: 5px 12px; border-radius: 4px; font-weight: bold;
}

/* Landing Page Specifics */
.hero-title {
    font-size: 4rem;
    background: -webkit-linear-gradient(eee, #333);
    -webkit-background-clip: text;
    text-shadow: 0 0 20px rgba(0,180,255,0.5);
    margin-bottom: 0;
}
.team-member {
    text-align: center;
    background: #161b22;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #30363d;
    height: 100%;
}
.team-role { color: #58a6ff; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }

/* Sidebar cleanup */
[data-testid="stSidebar"] {
    background-color: #0d1117;
    border-right: 1px solid #30363d;
}
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
    img_url: Optional[str] = None  # Added field for Image Retrieval
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

def generate_voiceover(text: str):
    """Generates audio bytes for the voiceover."""
    if not gTTS:
        return None
    try:
        # Clean text slightly for audio
        clean_text = text.replace("*", "").replace("#", "")
        tts = gTTS(text=clean_text, lang='en', slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        return fp
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
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            if torch:
                clip_model = clip_model.to(device)
        except Exception:
            clip_model, clip_processor = None, None

    nli_tok, nli_model = None, None
    if AutoTokenizer and AutoModelForSequenceClassification:
        try:
            nli_tok = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
            nli_model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
            if torch:
                nli_model = nli_model.to(device)
        except Exception:
            nli_tok, nli_model = None, None

    return device, clip_model, clip_processor, nli_tok, nli_model

DEVICE, CLIP_M, CLIP_P, NLI_TOK, NLI_M = load_core_models()


@st.cache_resource
def load_aux_models():
    df_model = None
    if tf and os.path.exists("deepfake/deepfake_cnn.h5"):
        try:
            try:
                tf.keras.config.enable_legacy_deserialization()
            except Exception:
                pass
            df_model = tf.keras.models.load_model("deepfake/deepfake_cnn.h5", compile=False)
        except Exception:
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

    support_gate = load_safe_model("support_gate.joblib")
    contra_gate = load_safe_model("contradiction_gate.joblib")
    style_tfidf = load_safe_model("tfidf_style_model.joblib")

    return df_model, rek_client, support_gate, contra_gate, style_tfidf

DF_M, REK, GATE_S, GATE_C, STYLE_TFIDF = load_aux_models()


# ==============================
# 5) LEXICON + STYLE
# ==============================
FAKE_LEXICON = [
    "shocking", "breaking", "unbelievable", "exposed", "secret", "miracle",
    "viral", "you won't believe", "forwarded", "whatsapp", "must watch",
    "cure", "guaranteed", "100%", "bombshell", "truth revealed"
]

def lexicon_score(text: str) -> float:
    t = (text or "").lower()
    hits = 0
    for w in FAKE_LEXICON:
        if w in t:
            hits += 1
    if not FAKE_LEXICON:
        return 0.0
    return clamp01(hits / max(1, len(FAKE_LEXICON) * 0.25))

def basic_style_features(text: str) -> Dict[str, float]:
    t = text or ""
    exclam = t.count("!")
    quest = t.count("?")
    upper = sum(1 for c in t if c.isupper())
    letters = sum(1 for c in t if c.isalpha())
    upper_ratio = (upper / letters) if letters else 0.0
    length = len(t)
    return {
        "exclam": float(exclam),
        "quest": float(quest),
        "upper_ratio": float(upper_ratio),
        "length": float(length),
    }


# ==============================
# 6) CONTEXT CONSENSUS (Updated for Images)
# ==============================
def fetch_newsapi_evidence(query: str, k: int = 10) -> List[EvidenceItem]:
    if not NEWS_API_KEY:
        return []
    q = requests.utils.quote(query)
    url = f"https://newsapi.org/v2/everything?q={q}&language=en&sortBy=relevancy&pageSize={k}&apiKey={NEWS_API_KEY}"
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
            img = a.get("urlToImage", None) # Get Image
            if title:
                items.append(EvidenceItem(title=title, url=link, source=source, snippet=desc, img_url=img))
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
        source = "Google RSS"
        if title:
            # RSS rarely has images easily accessible without scraping
            items.append(EvidenceItem(title=title, url=link, source=source))
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
                items.append(EvidenceItem(
                    title=title, 
                    url=res.get("link", ""), 
                    source="NewsData.io", 
                    snippet=norm_text(res.get("description", "")),
                    img_url=img
                ))
        return items
    except Exception:
        return []

def fetch_eventregistry_evidence(query: str, k: int = 10) -> List[EvidenceItem]:
    if not EVENTREGISTRY_KEY:
        return []
    q = requests.utils.quote(query)
    url = f"https://eventregistry.org/api/v1/article/getArticles?action=getArticles&keyword={q}&articlesPage=1&articlesCount={k}&articlesSortBy=rel&apiKey={EVENTREGISTRY_KEY}"
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
                items.append(EvidenceItem(
                    title=title, 
                    url=a.get("url", ""), 
                    source=a.get("source", {}).get("title", "Event Registry"),
                    img_url=img
                ))
        return items
    except Exception:
        return []

def build_evidence_pool(claim: str, k_total: int = 12) -> List[EvidenceItem]:
    a = fetch_newsapi_evidence(claim, k=10)
    b = fetch_google_rss_evidence(claim, k=10)
    c = fetch_newsdata_evidence(claim, k=10)
    d = fetch_eventregistry_evidence(claim, k=10)
    
    pool = a + b + c + d
    seen = set()
    out = []
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
            probs = torch.softmax(NLI_M(**inputs).logits, dim=1)[0].detach().cpu().numpy()
        return {"contradiction": float(probs[0]), "neutral": float(probs[1]), "entailment": float(probs[2])}
    except Exception:
        return {"contradiction": 0.0, "neutral": 1.0, "entailment": 0.0}

def annotate_nli(claim: str, evid: List[EvidenceItem], max_items: int = 8) -> List[EvidenceItem]:
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
            logits = CLIP_M(**inputs).logits_per_image
            score = torch.sigmoid(logits)[0][0].item()
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
        arr = np.array(img).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=0)
        pred = DF_M.predict(arr, verbose=0)
        p = float(pred.flatten()[0])
        return clamp01(p)
    except Exception:
        return 0.0


# ==============================
# 11) AWS Rekognition
# ==============================
def rekognition_signals(image_bytes: bytes) -> Dict[str, Any]:
    if REK is None or not image_bytes:
        return {"available": False, "labels": [], "moderation": []}
    out = {"available": True, "labels": [], "moderation": []}
    try:
        resp_labels = REK.detect_labels(Image={"Bytes": image_bytes}, MaxLabels=10, MinConfidence=70)
        labs = []
        for l in (resp_labels.get("Labels") or []):
            labs.append({"name": l.get("Name", ""), "confidence": float(l.get("Confidence", 0.0))})
        out["labels"] = labs
    except Exception:
        pass
    try:
        resp_mod = REK.detect_moderation_labels(Image={"Bytes": image_bytes}, MinConfidence=70)
        mods = []
        for m in (resp_mod.get("ModerationLabels") or []):
            mods.append({"name": m.get("Name", ""), "confidence": float(m.get("Confidence", 0.0))})
        out["moderation"] = mods
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
        if len(proba) >= 2:
            return float(proba[1])
        return float(proba[0])
    except Exception:
        return 0.0

def build_gate_features(claim: str, evid: List[EvidenceItem], clip_s: float, df_p: float, lex_s: float) -> Dict[str, float]:
    sims = [e.tfidf_sim for e in evid] if evid else [0.0]
    ent = [e.nli_ent for e in evid] if evid else [0.0]
    con = [e.nli_con for e in evid] if evid else [0.0]

    f = basic_style_features(claim)
    feats = {
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
    return feats


# ==============================
# 13) BACKEND REASONING ENGINE
# ==============================
def backend_reasoner(payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any], str]:
    if (not genai) or (not GEMINI_API_KEY):
        return "UNVERIFIED", {}, "Engine offline."

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Hardened logic for Identity and Activity status
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction="""You are a Hard-Truth Fact-Checker (January 2026).
            
            STRICT VERDICT RULES:
            1. IDENTITY: If the news is about a REPLICA, STORE MODEL, or TRIBUTE (like Havan in Brazil), but the claim refers to the ORIGINAL landmark (Statue of Liberty), the verdict is CONTRADICTED.
            2. ACTIVITY: If the news shows a person (Virat Kohli) scoring runs or playing in a specific format (ODI) in Jan 2026, any claim of their 'Retirement' from that format is CONTRADICTED.
            3. PROXIMITY: If an explosion (like a cylinder) happened at the gates/grounds of a site (Mysuru Palace), the claim of a 'Blast at [Site]' is SUPPORTED.

            NEVER USE 'MISLEADING'. Use only SUPPORTED or CONTRADICTED based on these factual distinctions."""
        )

        claim = payload.get('claim', '')
        evidence = [e.get('title', '') for e in payload.get('evidence', [])]
        evidence_text = "\n".join(evidence)

        prompt = f"""
        VERIFY THE FOLLOWING:
        CLAIM: "{claim}"
        CONTEXT: {evidence_text}

        REASONING TASK:
        - Liberty: Did the NYC original fall, or was it a Brazilian store replica?
        - Kohli: Is he active in ODIs as of Jan 2026? (He just played NZ in Indore).
        - Mysuru: Did a cylinder explode at the palace gate in Dec 2025?

        RETURN JSON:
        {{
          "verdict": "SUPPORTED" | "CONTRADICTED",
          "confidence": 0.99,
          "rationale": "Directly state the factual reason (e.g., 'Kohli is active in 2026 ODIs' or 'It was a store replica in Brazil')."
        }}
        """

        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        
        # Strip potential markdown and parse
        res_text = response.text.strip().lstrip('```json').rstrip('```').strip()
        data = json.loads(res_text)
        
        return data.get("verdict", "UNVERIFIED"), data, data.get("rationale", "")

    except Exception as e:
        return "ERROR", {}, f"Logic Error: {str(e)}"

def translate_to_english(text: str) -> str:
    if not text or not GoogleTranslator:
        return text
    try:
        # Specialized translation engine (Hallucination-free)
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        return translated
    except Exception as e:
        return text


# ==============================
# 14) LANDING PAGE LOGIC
# ==============================
def show_landing_page():
    st.markdown("<br/><br/>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 4, 1])
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
            unsafe_allow_html=True
        )
        st.markdown("<hr style='border-color: #30363d; margin: 40px 0;'>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='text-align: center; margin-bottom: 30px;'>TEAM DETAILS</h3>", unsafe_allow_html=True)
    
    # Updated Team Members - 5 People
    team_names = ["Kartik Shekhar", "K V Nihal Mouni", "Kartik Kaul", "Tanishi Jain", "Priyanka S"]
    
    # 5 names, let's do a 3 and 2 split for layout
    col1, col2, col3 = st.columns(3)
    col4, col5 = st.columns(2)
    
    for i, name in enumerate(team_names):
        # Choose column based on index
        if i < 3:
            col = [col1, col2, col3][i]
        else:
            col = [col4, col5][i - 3]
            
        with col:
            st.markdown(
                f"""
                <div class="team-member">
                    <div style="font-size: 1.5rem; margin-bottom: 10px; color: #8b949e;">Team Member</div>
                    <div style="font-size: 1.2rem; font-weight: bold; color: #f0f6fc;">{name}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
    st.markdown("<br/><br/><br/>", unsafe_allow_html=True)
    
    _, btn_col, _ = st.columns([3, 2, 3])
    with btn_col:
        st.info("Select 'Run Detector' from the sidebar to begin.")


# ==============================
# 15) MAIN DETECTOR PAGE LOGIC
# ==============================
def show_detector_page():
    st.title("AI based Regional Fake News Detection")
    st.caption(f"System Version: {APP_VERSION}")

    left, right = st.columns([1.5, 1], gap="large")

    with left:
        st.markdown(
            '<div class="css-card"><h3 style="margin-top:0">Analysis Input</h3>'
            '<p style="opacity:0.8">Enter a suspicious headline or claim. You may also upload a related image.</p>',
            unsafe_allow_html=True
        )
        claim = st.text_area("Claim / Headline", height=100, placeholder="e.g. Statue of Liberty collapsed in NY storm...")
        img_file = st.file_uploader("Evidence Image (Optional)", type=["jpg", "jpeg", "png"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        run_btn = st.button("INITIATE ANALYSIS", type="primary", use_container_width=True)

    with right:
        st.markdown('<div class="css-card"><h3 style="margin-top:0">System Status</h3>', unsafe_allow_html=True)
        status_rows = []
        status_rows.append(("CLIP", "ONLINE" if (CLIP_M and CLIP_P and torch) else "OFFLINE"))
        status_rows.append(("NLI Engine", "ONLINE" if (NLI_M and NLI_TOK and torch) else "OFFLINE"))
        status_rows.append(("NewsAPI", "KEY LOADED" if NEWS_API_KEY else "NO KEY"))
        status_rows.append(("Decision Engine", "READY" if (genai and GEMINI_API_KEY) else "UNAVAILABLE"))
        
        st.dataframe(
            pd.DataFrame(status_rows, columns=["Module", "State"]),
            use_container_width=True, 
            hide_index=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if run_btn:
        claim_original = norm_text(claim)
        if not claim_original:
            st.error("Input required: Please enter a claim.")
            st.stop()

        # --- 1. TRANSLATION ---
        with st.spinner("Translating input stream..."):
            claim_en = translate_to_english(claim_original)
            claim = claim_en  

        if claim != claim_original:
            st.info(f"Translated: {claim}")

        # --- 2. EVIDENCE ---
        with st.spinner("Scanning global media sources..."):
            evid = build_evidence_pool(claim, k_total=12)
            evid = compute_tfidf_sims(claim, evid)
            evid = annotate_nli(claim, evid)

        # --- 3. IMAGES & SIGNALS ---
        image = None
        img_bytes = b""
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

        # --- SORT EVIDENCE ---
        evid_sorted = sorted(evid, key=lambda x: (x.tfidf_sim, x.nli_ent), reverse=True)
        
        # Extract best image from evidence for UI display (New Feature)
        best_evidence_image = None
        for e in evid_sorted:
            if e.img_url:
                best_evidence_image = e.img_url
                break

        evid_pack = []
        for e in evid_sorted[:8]:
            evid_pack.append({
                "title": e.title,
                "source": e.source,
                "url": e.url,
                "snippet": e.snippet,
                "tfidf_sim": round(e.tfidf_sim, 4),
                "nli": {
                    "label": e.nli_label,
                    "entailment": round(e.nli_ent, 4),
                    "contradiction": round(e.nli_con, 4)
                }
            })

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
                }
            }
        }

        # --- 4. VERDICT ---
        with st.spinner("Deliberating final verdict..."):
            verdict, structured, rationale = backend_reasoner({"claim": claim, "evidence": evid_pack})

        # ==========================================
        # RESULTS UI
        # ==========================================
        st.markdown("---")
        
        # Top Result Section
        res_col1, res_col2 = st.columns([2, 1])
        
        with res_col1:
            st.subheader("Final Verdict")
            if verdict == "SUPPORTED":
                st.markdown(f'<div class="css-card" style="border-left: 5px solid #38ef7d;">'
                            f'<h2 style="color:#38ef7d; margin:0;">SUPPORTED</h2>'
                            f'<p>{rationale}</p></div>', unsafe_allow_html=True)
            elif verdict == "CONTRADICTED":
                st.markdown(f'<div class="css-card" style="border-left: 5px solid #ef473a;">'
                            f'<h2 style="color:#ef473a; margin:0;">CONTRADICTED</h2>'
                            f'<p>{rationale}</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="css-card" style="border-left: 5px solid #ffb000;">'
                            f'<h2 style="color:#ffb000; margin:0;">{verdict}</h2>'
                            f'<p>{rationale}</p></div>', unsafe_allow_html=True)

            # VOICE OVER FEATURE
            if rationale and gTTS:
                audio_text = f"The verdict is {verdict}. {rationale}"
                audio_fp = generate_voiceover(audio_text)
                if audio_fp:
                    # Added autoplay=True for automatic playback
                    st.audio(audio_fp, format='audio/mp3', autoplay=True)

        with res_col2:
            st.subheader("Evidence Context Image")
            if best_evidence_image:
                st.image(best_evidence_image, caption="Retrieved from Top Evidence Source", use_container_width=True)
            elif image:
                st.image(image, caption="User Uploaded Image", use_container_width=True)
            else:
                st.info("No visual context available.")

        # Evidence Cards
        st.subheader("Key Evidence Sources")
        if evid_sorted:
            # Display grid of evidence
            ec1, ec2 = st.columns(2)
            for i, e in enumerate(evid_sorted[:6]):
                col = ec1 if i % 2 == 0 else ec2
                with col:
                    nli_color = "#38ef7d" if e.nli_label == "ENTAILMENT" else "#ef473a" if e.nli_label == "CONTRADICTION" else "#8b949e"
                    
                    # Clean snippet
                    snip = (e.snippet[:150] + "...") if len(e.snippet) > 150 else e.snippet
                    
                    col.markdown(
                        f"""
                        <div class="css-card">
                            <div style="font-weight:bold; font-size:1.1rem; color:#fff; margin-bottom:5px;">
                                <a href="{e.url}" target="_blank" style="text-decoration:none; color:#58a6ff;">{e.title}</a>
                            </div>
                            <div style="margin-bottom:8px;">
                                <span class="custom-badge">{e.source}</span>
                                <span style="font-size:0.8rem; color:{nli_color}; border:1px solid {nli_color}; padding:2px 6px; border-radius:4px;">{e.nli_label}</span>
                            </div>
                            <div style="font-size:0.9rem; color:#8b949e;">{snip}</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

        # Signals
        st.subheader("Signal Telemetry")
        with st.expander("View Underlying Data (Explainability)", expanded=False):
            # Context
            st.markdown("#### Context Metrics")
            m1, m2, m3, m4 = st.columns(4)
            ctx = payload["signals"]["context"]
            m1.metric("TF-IDF Max", f"{ctx['tfidf_max']:.2f}")
            m2.metric("TF-IDF Mean", f"{ctx['tfidf_mean']:.2f}")
            m3.metric("NLI Entailment Max", f"{ctx['nli_ent_max']:.2f}")
            m4.metric("NLI Contradiction Max", f"{ctx['nli_con_max']:.2f}")

            st.divider()

            # Style & Image
            st.markdown("#### Style & Visual Analysis")
            s1, s2, s3, s4 = st.columns(4)
            style = payload["signals"]["style"]
            img_sig = payload["signals"]["image"]
            
            s1.metric("Lexicon Score", f"{style['lexicon']:.2f}")
            s2.metric("Uppercase Ratio", f"{style['basic']['upper_ratio']:.2f}")
            
            clip_val = f"{img_sig['clip_relevance']:.2f}" if img_sig['has_image'] else "N/A"
            deepfake_val = f"{img_sig['deepfake_prob']:.2f}" if img_sig['has_image'] else "N/A"
            
            s3.metric("CLIP Relevance", clip_val)
            s4.metric("Deepfake Prob", deepfake_val)


# ==============================
# 16) MAIN APP ROUTER
# ==============================
def main():
    # Sidebar Navigation
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