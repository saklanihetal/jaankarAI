# =========================================================
# EL MASTER (FINAL - PATCHED): Multimodal Fake News Detection
# - CLIP + NLI + Deepfake + AWS Rekognition
# - NewsAPI + Google RSS (Context Consensus)
# - Gates + TF-IDF + Lexicon
# - FINAL VERDICT: backend decision engine ONLY (no UI mention)
# =========================================================

import os
import re
import io
import json
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

# Backend-only decision engine client (kept invisible to UI)
try:
    import google.generativeai as genai
except Exception:
    genai = None


# ==============================
# 0) CONFIG + SECRET LOADING (PATCHED)
# ==============================
st.set_page_config(page_title="Multimodal FactCheck", page_icon="🛡️", layout="wide")


def get_secret(name: str, default: str = "") -> str:
    """
    Streamlit secrets first, then environment variables.
    Never prints the value. Returns stripped string.
    """
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
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")

AWS_KEY = get_secret("AWS_ACCESS_KEY_ID")
AWS_SECRET = get_secret("AWS_SECRET_ACCESS_KEY")
AWS_REGION = get_secret("AWS_REGION", "ap-south-1")

APP_VERSION = "EL-MASTER-FINAL-PATCHED"


# ==============================
# 1) UI STYLES (high contrast)
# ==============================
st.markdown(
    """
<style>
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1200px; }
h1, h2, h3, h4, h5, h6 { letter-spacing: 0.2px; color: #e5e7eb; }
p, li, span, label, div { color: #e5e7eb; }
.card {
  background: #0f172a;
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 16px 16px;
  box-shadow: 0 12px 30px rgba(0,0,0,0.18);
}
.card-title { font-size: 1.05rem; font-weight: 700; color: #e5e7eb; }
.card-muted { color: rgba(229,231,235,0.75); font-size: 0.92rem; }
.badge {
  display:inline-block; padding: 6px 10px; border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.16);
  background: rgba(255,255,255,0.06);
  color: #e5e7eb; font-size: 0.85rem; font-weight: 600;
}
hr { border-color: rgba(255,255,255,0.10); }
.small { font-size: 0.92rem; color: rgba(229,231,235,0.80); }
.footer { margin-top: 18px; color: rgba(229,231,235,0.55); font-size: 0.85rem; }
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
#    PATCHED: NLI uses stable public model
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

    # ✅ PATCH: use guaranteed public NLI model
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
    # Deepfake CNN (your file path)
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

    # AWS Rekognition
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

    # Gates & style models
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
# 6) CONTEXT CONSENSUS (NewsAPI + RSS)
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
            if title:
                items.append(EvidenceItem(title=title, url=link, source=source, snippet=desc))
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
            items.append(EvidenceItem(title=title, url=link, source=source))
    return items

def build_evidence_pool(claim: str, k_total: int = 12) -> List[EvidenceItem]:
    a = fetch_newsapi_evidence(claim, k=10)
    b = fetch_google_rss_evidence(claim, k=10)
    pool = a + b
    seen = set()
    out = []
    for it in pool:
        key = it.title.lower()
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

        # For bart-large-mnli: labels are usually [contradiction, neutral, entailment]
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
# 13) BACKEND REASONING ENGINE (hidden)
# ==============================
def backend_reasoner(payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any], str]:
    """
    Returns:
      (verdict_label, structured_json, reasoning_text_for_UI)

    Allowed verdict labels (exact spellings):
      SUPPORTED
      CONTRADICTED
      UNVERIFIED — Likely True
      UNVERIFIED — Likely Fake
      UNVERIFIED — Neutral
    """
    # Fallback: conservative + fast if decision engine missing
    if (not genai) or (not GEMINI_API_KEY):
        sims = payload.get("signals", {}).get("context", {}).get("tfidf_max", 0.0)
        ent = payload.get("signals", {}).get("context", {}).get("nli_ent_max", 0.0)
        con = payload.get("signals", {}).get("context", {}).get("nli_con_max", 0.0)
        lex = payload.get("signals", {}).get("style", {}).get("lexicon", 0.0)

        if con > 0.60 and sims > 0.25:
            return "CONTRADICTED", {"verdict": "CONTRADICTED", "confidence": 0.65}, "Summary generated from available evidence signals."
        if ent > 0.60 and sims > 0.25:
            return "SUPPORTED", {"verdict": "SUPPORTED", "confidence": 0.65}, "Summary generated from available evidence signals."
        if lex > 0.65:
            return "UNVERIFIED — Likely Fake", {"verdict": "UNVERIFIED — Likely Fake", "confidence": 0.55}, "Summary generated from available evidence signals."
        if sims > 0.45:
            return "UNVERIFIED — Likely True", {"verdict": "UNVERIFIED — Likely True", "confidence": 0.55}, "Summary generated from available evidence signals."
        return "UNVERIFIED — Neutral", {"verdict": "UNVERIFIED — Likely Neutral", "confidence": 0.50}, "Summary generated from available evidence signals."

    # Decision engine path (kept invisible)
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        return "UNVERIFIED — Neutral", {"verdict": "UNVERIFIED — Neutral", "confidence": 0.50}, "Summary generated from available evidence signals."

    # Robust model choice
    model = None
    try:
        avail = [m.name for m in genai.list_models() if "generateContent" in (m.supported_generation_methods or [])]
        chosen = None
        for p in ["flash", "pro", "1.5", "1.0"]:
            chosen = next((x for x in avail if p in x.lower()), None)
            if chosen:
                break
        model = genai.GenerativeModel(chosen or avail[0])
    except Exception:
        try:
            model = genai.GenerativeModel("gemini-pro")
        except Exception:
            model = None

    if not model:
        return "UNVERIFIED — Neutral", {"verdict": "UNVERIFIED — Neutral", "confidence": 0.50}, "Summary generated from available evidence signals."

    prompt = f"""
You are a professional fact-checker.

You MUST decide the final verdict based ONLY on the evidence and signals given.
Do NOT mention any model names, APIs, or internal tools.
Return ONLY valid JSON (no markdown, no extra text).

Allowed final verdict labels (exact spellings):
- "SUPPORTED"
- "CONTRADICTED"
- "UNVERIFIED — Likely True"
- "UNVERIFIED — Likely Fake"
- "UNVERIFIED — Neutral"

Decision rules:
- If evidence strongly supports the claim: SUPPORTED
- If evidence strongly contradicts the claim: CONTRADICTED
- If evidence is insufficient or mixed: UNVERIFIED with the best sublabel
- Use image signals only as supporting context, not as sole truth evidence.

Now analyze this payload:

{json.dumps(payload, ensure_ascii=False)}
""".strip()

    try:
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
    except Exception:
        return "UNVERIFIED — Neutral", {"verdict": "UNVERIFIED — Neutral", "confidence": 0.50}, "Summary generated from available evidence signals."

    parsed = None
    try:
        parsed = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except Exception:
                parsed = None

    allowed = {
        "SUPPORTED",
        "CONTRADICTED",
        "UNVERIFIED — Likely True",
        "UNVERIFIED — Likely Fake",
        "UNVERIFIED — Neutral",
    }

    if not isinstance(parsed, dict):
        return "UNVERIFIED — Neutral", {"verdict": "UNVERIFIED — Neutral", "confidence": 0.50}, "Summary generated from available evidence signals."

    verdict = str(parsed.get("verdict", "")).strip()
    if verdict not in allowed:
        verdict = "UNVERIFIED — Neutral"

    confidence = parsed.get("confidence", None)
    try:
        confidence = float(confidence) if confidence is not None else (0.60 if verdict in ("SUPPORTED", "CONTRADICTED") else 0.55)
    except Exception:
        confidence = 0.55

    rationale = str(parsed.get("rationale", "") or "").strip()
    if not rationale:
        rationale = "Summary generated from available evidence signals."

    structured = {"verdict": verdict, "confidence": confidence}
    return verdict, structured, rationale


# ==============================
# 14) APP UI
# ==============================
st.title("🛡️ Multimodal FactCheck")
st.caption(f"Build: {APP_VERSION}")

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown(
        '<div class="card"><div class="card-title">Input</div>'
        '<div class="card-muted">Enter a claim/headline and optionally add an image.</div></div>',
        unsafe_allow_html=True
    )
    claim = st.text_area("Claim / Headline", height=120, placeholder="Paste the headline/claim here...")
    img_file = st.file_uploader("Optional image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    st.markdown("<hr/>", unsafe_allow_html=True)
    run_btn = st.button("Analyze", type="primary", use_container_width=True)

with right:
    st.markdown('<div class="card"><div class="card-title">System Status</div></div>', unsafe_allow_html=True)

    status_rows = []
    status_rows.append(("CLIP", "OK" if (CLIP_M and CLIP_P and torch) else "Unavailable"))
    status_rows.append(("NLI", "OK" if (NLI_M and NLI_TOK and torch) else "Unavailable"))
    status_rows.append(("NewsAPI", "OK" if NEWS_API_KEY else "No key"))
    status_rows.append(("Google RSS", "OK" if feedparser else "Unavailable"))
    status_rows.append(("Deepfake", "OK" if DF_M else "Unavailable"))
    status_rows.append(("AWS Rekognition", "OK" if REK else "Unavailable"))
    status_rows.append(("Support Gate", "OK" if GATE_S else "Unavailable"))
    status_rows.append(("Contradiction Gate", "OK" if GATE_C else "Unavailable"))
    status_rows.append(("Style TF-IDF", "OK" if STYLE_TFIDF else "Unavailable"))
    status_rows.append(("Decision Engine", "OK" if (genai and GEMINI_API_KEY) else "No key / Unavailable"))

    st.dataframe(pd.DataFrame(status_rows, columns=["Component", "Status"]),
                 use_container_width=True, hide_index=True)


# ==============================
# 15) RUN PIPELINE
# ==============================
if run_btn:
    claim = norm_text(claim)
    if not claim:
        st.error("Please enter a claim/headline.")
        st.stop()

    with st.spinner("Collecting context evidence..."):
        evid = build_evidence_pool(claim, k_total=12)
        evid = compute_tfidf_sims(claim, evid)

    with st.spinner("Running textual consistency checks..."):
        evid = annotate_nli(claim, evid, max_items=8)

    # Image processing (optional)
    image = None
    img_bytes = b""
    if img_file:
        try:
            img_bytes = img_file.read()
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            image = None
            img_bytes = b""

    with st.spinner("Computing multimodal signals..."):
        clip_s = clip_relevance(image, claim) if image else 0.0
        df_p = deepfake_prob_from_image(image) if image else 0.0
        rek = rekognition_signals(img_bytes) if img_bytes else {"available": False, "labels": [], "moderation": []}

        lex_s = lexicon_score(claim)

        feats = build_gate_features(claim, evid, clip_s, df_p, lex_s)
        p_support = gate_predict(GATE_S, feats)
        p_contra = gate_predict(GATE_C, feats)

        style_pred = None
        if STYLE_TFIDF is not None:
            try:
                style_pred = float(STYLE_TFIDF.predict_proba([claim])[0][1]) if hasattr(STYLE_TFIDF, "predict_proba") else None
            except Exception:
                style_pred = None

    evid_sorted = sorted(evid, key=lambda x: (x.tfidf_sim, x.nli_ent), reverse=True)
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
                "contradiction": round(e.nli_con, 4),
                "neutral": round(e.nli_neu, 4),
            }
        })

    sims = [x.tfidf_sim for x in evid] or [0.0]
    ent = [x.nli_ent for x in evid] or [0.0]
    con = [x.nli_con for x in evid] or [0.0]

    payload = {
        "claim": claim,
        "evidence": evid_pack,
        "signals": {
            "context": {
                "tfidf_max": float(np.max(sims)),
                "tfidf_mean": float(np.mean(sims)),
                "nli_ent_max": float(np.max(ent)),
                "nli_con_max": float(np.max(con)),
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
                "style_model_prob": style_pred,
            },
            "gates": {
                "support_gate_prob": float(p_support),
                "contradiction_gate_prob": float(p_contra),
                "feature_vector": feats,
            }
        }
    }

    with st.spinner("Finalizing verdict..."):
        verdict, structured, rationale = backend_reasoner(payload)

    # ---------------- OUTPUT UI (no mention of backend engine name) ----------------
    st.markdown("<hr/>", unsafe_allow_html=True)

    st.subheader("✅ Final Verdict")
    if verdict == "SUPPORTED":
        st.success("SUPPORTED")
    elif verdict == "CONTRADICTED":
        st.error("CONTRADICTED")
    else:
        st.warning(verdict)

    conf = structured.get("confidence", None)
    if conf is not None:
        try:
            st.caption(f"Confidence (estimated): {float(conf):.2f}")
        except Exception:
            pass

    st.subheader("🧾 Reasoning Summary")
    st.write(rationale)

    st.subheader("📰 Context Evidence (Top)")
    if evid_sorted:
        for e in evid_sorted[:6]:
            st.markdown(
                f"""
<div class="card">
  <div class="card-title">{e.title}</div>
  <div class="card-muted">
    <span class="badge">{e.source}</span>
    &nbsp;&nbsp;<span class="badge">TF-IDF {e.tfidf_sim:.2f}</span>
    &nbsp;&nbsp;<span class="badge">NLI {e.nli_label}</span>
    &nbsp;&nbsp;<span class="badge">Ent {e.nli_ent:.2f}</span>
    &nbsp;&nbsp;<span class="badge">Con {e.nli_con:.2f}</span>
  </div>
  <div class="small">{(e.snippet or "")}</div>
  <div class="small">{(e.url or "")}</div>
</div>
<br/>
""",
                unsafe_allow_html=True,
            )
    else:
        st.info("No evidence sources retrieved. Check your internet / keys.")

    st.subheader("📊 Signals Dashboard (Explainability)")
    colA, colB, colC = st.columns(3)

    with colA:
        st.markdown("**Context**")
        st.json(payload["signals"]["context"], expanded=False)

    with colB:
        st.markdown("**Style & Lexicon**")
        st.json(payload["signals"]["style"], expanded=False)

    with colC:
        st.markdown("**Image**")
        st.json(payload["signals"]["image"], expanded=False)

    with st.expander("Gates (signals only — not used for final verdict)"):
        st.json(payload["signals"]["gates"], expanded=False)

    if image:
        st.subheader("🖼️ Uploaded Image")
        st.image(image, use_container_width=True)

    

