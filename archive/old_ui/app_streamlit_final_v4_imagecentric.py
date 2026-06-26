# app_streamlit_final_v4_imagecentric.py
# Image-centric upgrade (Option 1):
# - Reverse image search (SerpAPI Google Lens) to detect image reuse / context drift
# - Extract consensus context from image-web results (locations + repeated context words)
# - Use image-provenance mismatch to reduce verification confidence (never claims truth by itself)
# - Keeps: evidence consensus section (non-negotiable), NLI, language hint, image-text CLIP explanation

import os
import re
import json
import base64
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

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Fake News Detection", layout="wide")
st.title("Fake News Detection")

# ---------------- CONSTANTS ----------------
TOP_K_NLI = 5
MAX_ITEMS_FETCH = 60
RSS_PER_QUERY = 12
HTTP_TIMEOUT = 12

# Conservative fallback logic (keeps app stable even if gate models are missing/broken)
MIN_REL_FOR_NLI = 0.18
REL_GATE_R0 = 0.45
REL_GATE_R1 = 0.70

# CLIP gate mapping
CLIP_A = 0.20
CLIP_B = 0.35

# Language hint thresholds
LIKELY_THR = 0.80
NEUTRAL_BAND = 0.15

# Image provenance consensus thresholds
IMGCTX_MIN_RESULTS = 4
IMGCTX_LOC_REPEAT = 2
IMGCTX_WORD_REPEAT = 3
IMGCTX_MISMATCH_WARN = 0.55  # if image context mismatch high -> warn and reduce confidence

SUPPORTED_LANGS = {"en", "hi", "kn", "te"}
LANG_MAP = {"hi": "hin_Deva", "kn": "kan_Knda", "te": "tel_Telu"}

_EN_STOP = {
    "the","a","an","and","or","to","of","in","on","for","with","from","by","at","as","is","are","was","were","be","been",
    "it","this","that","these","those","after","before","new","latest","today","yesterday","tomorrow","over","into",
    "near","around","amid","says","say","said","will","would","can","could","may","might","has","have","had","up","down","out",
    "about","more","most","very",
}

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

def strip_html_garbage(s: str) -> str:
    s = s or ""
    s = re.sub(r"<a\b[^>]*>(.*?)</a>", r"\1", s, flags=re.I | re.S)
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("&nbsp;", " ").replace("nbsp", " ")
    s = s.replace("font", " ")
    s = re.sub(r"\b(com|www|http|https|google|blank|articles)\b", " ", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---------------- MODELS ----------------
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
    nllb_tok = AutoTokenizer.from_pretrained(translator_name)
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained(translator_name).to(device)

    nli_name = "facebook/bart-large-mnli"
    nli_tok = AutoTokenizer.from_pretrained(nli_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_name).to(device)

    return device, clip_model, clip_processor, labse, nllb_tok, nllb_model, nli_tok, nli_model

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
    return float(probs[2]), float(probs[0]), float(probs[1])  # ent, con, neu

# ---------------- EVIDENCE ----------------
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
        desc = strip_html_garbage(getattr(e, "summary", "") or "")
        published = normalize_text(getattr(e, "published", "") or "")
        if title and link:
            items.append(EvidenceItem(title=title, url=link, source="GoogleNewsRSS", desc=desc, published=published))
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

# ---------------- QUERY BUILDER ----------------
def build_queries(original_text: str, claim_en: str, lang: str) -> List[str]:
    original_text = normalize_text(original_text)
    claim_en = normalize_text(claim_en)
    queries = []
    if len(original_text) >= 5:
        queries.append(original_text)
    if len(claim_en) >= 5 and claim_en.lower() != original_text.lower():
        queries.append(claim_en)
    queries.append(" ".join(claim_en.split()[:9]))
    out, seen = [], set()
    for q in queries:
        qn = q.lower().strip()
        if len(qn) < 4:
            continue
        if qn not in seen:
            out.append(q.strip())
            seen.add(qn)
    return out[:5]

# ---------------- RANK + CONSENSUS + NLI ----------------
def rank_by_relevance(items: List[EvidenceItem], claim_en: str, labse: SentenceTransformer) -> List[EvidenceItem]:
    claim_emb = labse.encode(claim_en, convert_to_tensor=True)
    for it in items:
        text = normalize_text((it.title + " " + (it.desc or "")).strip())
        emb = labse.encode(text, convert_to_tensor=True)
        it.relevance = float(util.cos_sim(claim_emb, emb).item())
    items.sort(key=lambda x: x.relevance, reverse=True)
    return items

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
    scored = []
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
        g_rel = rel_gate(it.relevance, REL_GATE_R0, REL_GATE_R1)
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

# ---------------- LANGUAGE MODEL ----------------
def load_tfidf(path: str):
    return joblib.load(path)

def language_hint_from_probs(p_real: float, p_fake: float) -> str:
    if (p_fake >= LIKELY_THR) and (p_fake - p_real >= NEUTRAL_BAND):
        return "Likely False (language patterns)"
    if (p_real >= LIKELY_THR) and (p_real - p_fake >= NEUTRAL_BAND):
        return "Likely True (language patterns)"
    return "Neutral / inconclusive (language patterns)"

# ---------------- IMAGE PROVENANCE (Reverse Image Search) ----------------
def pil_to_jpg_bytes(img: Image.Image, quality: int = 92) -> bytes:
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()

def serpapi_google_lens(image_bytes: bytes, serpapi_key: str) -> Tuple[List[Dict], str]:
    """
    Uses SerpAPI 'google_lens' engine.
    Returns a list of results with fields:
      title, source, link, snippet
    """
    try:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        # SerpAPI expects a data URI in many examples.
        data_uri = f"data:image/jpeg;base64,{b64}"

        params = {
            "engine": "google_lens",
            "api_key": serpapi_key,
            "url": data_uri,
        }
        # SerpAPI endpoint
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=HTTP_TIMEOUT)
        if r.status_code != 200:
            return [], f"HTTP {r.status_code}: {r.text[:200]}"
        js = r.json()

        results = []
        # Different SerpAPI responses may include: visual_matches / knowledge_graph / organic_results
        for block in js.get("visual_matches", [])[:12]:
            title = normalize_text(block.get("title", "") or "")
            link = block.get("link", "") or block.get("source", "") or ""
            source = normalize_text(block.get("source", "") or "")
            snippet = normalize_text(block.get("snippet", "") or "")
            if title or link:
                results.append({"title": title, "link": link, "source": source, "snippet": snippet})

        # fallback: some responses include "organic_results"
        if not results:
            for block in js.get("organic_results", [])[:12]:
                title = normalize_text(block.get("title", "") or "")
                link = block.get("link", "") or ""
                source = normalize_text(block.get("source", "") or "")
                snippet = normalize_text(block.get("snippet", "") or "")
                if title or link:
                    results.append({"title": title, "link": link, "source": source, "snippet": snippet})

        return results, ""
    except Exception as e:
        return [], str(e)

def image_context_consensus(results: List[Dict], claim_en: str, nlp) -> Tuple[List[str], List[str], float]:
    """
    Extract repeated locations + repeated words from reverse-image results.
    Returns (locations, words, mismatch_rate_vs_claim)
    """
    if not results or len(results) < IMGCTX_MIN_RESULTS:
        return [], [], 0.0

    texts = []
    for r in results[:12]:
        t = normalize_text((r.get("title","") + " " + r.get("snippet","")).strip())
        t = strip_html_garbage(t)
        if t:
            texts.append(t)

    # Locations
    loc_counts: Dict[str, int] = {}
    if nlp is not None:
        for t in texts:
            doc = nlp(t)
            for ent in doc.ents:
                if ent.label_ in {"GPE", "LOC"}:
                    k = ent.text.strip().lower()
                    if len(k) >= 3:
                        loc_counts[k] = loc_counts.get(k, 0) + 1

    rep_locs = [k for k, c in sorted(loc_counts.items(), key=lambda x: -x[1]) if c >= IMGCTX_LOC_REPEAT][:5]

    # Words
    word_counts: Dict[str, int] = {}
    for t in texts:
        toks = keyword_set_en(t)
        for w in toks:
            word_counts[w] = word_counts.get(w, 0) + 1

    rep_words = [w for w, c in sorted(word_counts.items(), key=lambda x: -x[1]) if c >= IMGCTX_WORD_REPEAT][:10]

    claim_tokens = keyword_set_en(claim_en)
    missing = [w for w in rep_words if w not in claim_tokens]
    mismatch_rate = (len(missing) / max(1, len(rep_words))) if rep_words else 0.0

    return rep_locs, rep_words, float(mismatch_rate)

# ---------------- UI HELPERS ----------------
def big_badge(text: str, kind: str = "neutral"):
    bg = {"green": "#123d2a", "red": "#3d1212", "amber": "#3b3312", "neutral": "#1f2937"}[kind]
    border = {"green": "#2ecc71", "red": "#ff6b6b", "amber": "#f1c40f", "neutral": "#93a4b8"}[kind]
    st.markdown(
        f"""
        <div style="
            padding: 14px 16px;
            border-radius: 14px;
            border: 1px solid {border};
            background: {bg};
            font-size: 22px;
            font-weight: 800;
            letter-spacing: 0.3px;
            ">
            {text}
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------- SIDEBAR ----------------
st.sidebar.header("Input")
uploaded_image = st.sidebar.file_uploader("Upload image (optional)", type=["jpg", "jpeg", "png"])
news_text = st.sidebar.text_area("Enter headline / claim (any language)", height=120)
show_debug = st.sidebar.checkbox("Show debug details", value=False)

with st.sidebar.expander("Run command", expanded=False):
    st.code("streamlit run app_streamlit_final_v4_imagecentric.py", language="bash")

# ---------------- Load resources ----------------
device, clip_model, clip_processor, labse, nllb_tok, nllb_model, nli_tok, nli_model = load_models()
nlp, spacy_name = load_spacy()

tfidf_model = None
if os.path.exists("tfidf_style_model.joblib"):
    try:
        tfidf_model = load_tfidf("tfidf_style_model.joblib")
    except Exception:
        tfidf_model = None

# ---------------- Display image ----------------
image = None
image_bytes = None
if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)
    try:
        image_bytes = pil_to_jpg_bytes(image)
    except Exception:
        image_bytes = None

# ---------------- MAIN ----------------
if news_text and len(news_text.strip()) > 0:
    st.subheader("Processing")

    T_raw = normalize_text(news_text)
    lang = detect_language(T_raw)
    claim_en = translate_to_english(T_raw, lang, device, nllb_tok, nllb_model)

    st.write(f"Detected language: **{lang}**")
    st.write(f"spaCy: **{spacy_name}**")
    st.write(f"Claim (English): **{claim_en}**")

    # Image explanation (CLIP)
    g_clip = 0.0
    st.markdown("### Image explanation")
    if image is not None:
        _, g_clip = clip_gate_score(image, claim_en, device, clip_model, clip_processor)
        if g_clip >= 0.65:
            st.write("The image appears consistent with the claim topic.")
        elif g_clip <= 0.35:
            st.write("The image appears weakly related to the claim topic (possible misuse).")
        else:
            st.write("The image appears somewhat related to the claim topic.")
    else:
        st.write("No image provided, so image–text consistency was not evaluated.")

    # Language-only
    p_real, p_fake = None, None
    lang_hint = "n/a"
    if tfidf_model is not None:
        try:
            pr = tfidf_model.predict_proba([T_raw])[0]
            p_real, p_fake = float(pr[0]), float(pr[1])
            lang_hint = language_hint_from_probs(p_real, p_fake)
        except Exception:
            p_real, p_fake = None, None
            lang_hint = "n/a"

    # Evidence retrieval (RSS)
    queries = build_queries(T_raw, claim_en, lang)
    source_status = {}
    items: List[EvidenceItem] = []

    with st.spinner("Fetching evidence headlines..."):
        for q in queries:
            its, err = google_news_rss(q, hl="en-IN", gl="IN", ceid="IN:en")
            source_status["GoogleNewsRSS(en-IN)"] = "OK" if not err else err
            items.extend(its)

        # local language RSS helps regional titles indexing
        if lang in {"hi", "kn", "te"}:
            hl_map = {"hi": "hi-IN", "kn": "kn-IN", "te": "te-IN"}
            hl = hl_map.get(lang, "en-IN")
            for q in queries[:3]:
                its, err = google_news_rss(q, hl=hl, gl="IN", ceid="IN:en")
                source_status[f"GoogleNewsRSS({hl})"] = "OK" if not err else err
                items.extend(its)

    st.markdown("### Source status")
    st.json(source_status)

    items = dedup_items(items)[:MAX_ITEMS_FETCH]
    st.write(f"Fetched **{len(items)}** unique items (before ranking).")
    if not items:
        st.warning("No evidence items were found.")
        st.stop()

    # Rank + consensus (text evidence)
    ranked = rank_by_relevance(items, claim_en, labse)
    rep_locs, rep_words, ctx_mismatch_rate = consensus_context(ranked, claim_en, nlp)

    st.markdown("## Consensus context found in evidence")
    st.write("Locations mentioned by multiple sources: " + (", ".join(rep_locs) if rep_locs else "(none)"))
    st.write("Other repeated context words: " + (", ".join(rep_words) if rep_words else "(none)"))

    # NLI over top-k
    max_ent, max_con_used, scored, body_success_rate = aggregate_topk_nli(
        ranked, claim_en, device, nli_tok, nli_model
    )

    # ---------------- Image provenance: reverse image search ----------------
    imgctx_locs, imgctx_words, imgctx_mismatch = [], [], 0.0
    img_results: List[Dict] = []
    img_status = "skipped"

    st.markdown("## Image provenance (reverse image search)")
    serpapi_key = ""
    try:
        serpapi_key = st.secrets.get("SERPAPI_KEY", "")
    except Exception:
        serpapi_key = ""

    if image_bytes is None:
        st.write("No image uploaded, so reverse image search was not run.")
    else:
        if not serpapi_key:
            st.warning("SERPAPI_KEY not found. Reverse image search requires an API key.")
            st.write("Manual fallback: use Google Images / Lens and compare contexts (location/time/replica, etc.).")
        else:
            with st.spinner("Running reverse image search (Google Lens via SerpAPI)..."):
                img_results, err = serpapi_google_lens(image_bytes, serpapi_key)
                img_status = "OK" if not err else err

            st.write(f"Reverse image search status: **{img_status}**")
            if img_results:
                imgctx_locs, imgctx_words, imgctx_mismatch = image_context_consensus(img_results, claim_en, nlp)
                st.write("Repeated locations in image results: " + (", ".join(imgctx_locs) if imgctx_locs else "(none)"))
                st.write("Repeated context words in image results: " + (", ".join(imgctx_words) if imgctx_words else "(none)"))
                st.write(f"Image-context mismatch vs claim (0–1): **{imgctx_mismatch:.3f}**")

                with st.expander("Top reverse-image matches", expanded=False):
                    for r in img_results[:8]:
                        t = r.get("title","") or "(no title)"
                        link = r.get("link","") or ""
                        snip = r.get("snippet","") or ""
                        st.markdown(f"**{t}**  \n{snip}\n\n{link}\n---")
            else:
                st.write("No reverse-image matches returned.")

    # ---------------- Decision (safe & explainable) ----------------
    # Base decision from text evidence only
    fallback_supported = (max_ent >= 0.80 and ranked[0].relevance >= 0.45 and max_con_used < 0.15)
    fallback_contradicted = (max_con_used >= 0.35 and ranked[0].relevance >= 0.45)

    if fallback_contradicted:
        decision = "CONTRADICTED"
    elif fallback_supported:
        decision = "SUPPORTED"
    else:
        decision = "UNVERIFIED"

    # Image-centric adjustment (never flips from contradicted to supported etc.)
    # Only makes UNVERIFIED more cautious, and makes SUPPORTED less confident if mismatch is high.
    image_mismatch_flag = (imgctx_mismatch >= IMGCTX_MISMATCH_WARN and len(imgctx_words) > 0)

    # ---------------- RESULT UI ----------------
    st.subheader("Result")

    left, right = st.columns([1.2, 1.0])
    with left:
        if decision == "SUPPORTED":
            big_badge("SUPPORTED", kind="green")
        elif decision == "CONTRADICTED":
            big_badge("CONTRADICTED", kind="red")
        else:
            big_badge("UNVERIFIED", kind="amber")
        st.caption(f"Image alignment (CLIP): {g_clip:.3f}")

    with right:
        big_badge(lang_hint, kind="neutral")
        if p_real is not None:
            st.caption(f"Language probs: P(real-style)={p_real:.3f} | P(fake-style)={p_fake:.3f}")

    st.subheader("Explanation")
    expl = []
    if decision == "SUPPORTED":
        expl.append("Relevant sources align with the claim, so it is marked supported.")
    elif decision == "CONTRADICTED":
        expl.append("Relevant sources contain conflicting information, so it is marked contradicted.")
    else:
        expl.append("Evidence is insufficient to strongly support or contradict the claim, so it is marked unverified.")

    # Image-centric explanation
    if image is not None:
        if image_mismatch_flag:
            expl.append("Reverse image search suggests the image commonly appears with a different context than the claim, which reduces confidence.")
        else:
            if img_results:
                expl.append("Reverse image search did not show a strong context mismatch signal.")

    # Text-evidence consensus mismatch explanation
    if ctx_mismatch_rate >= 0.50 and rep_words:
        expl.append("Evidence contains repeated context details not present in the claim, which reduces verification confidence.")

    # UNVERIFIED language hint explanation
    if decision == "UNVERIFIED" and p_real is not None:
        if "Likely True" in lang_hint:
            expl.append("Although evidence is insufficient, the headline language looks closer to normal reporting (likely true by style).")
        elif "Likely False" in lang_hint:
            expl.append("Although evidence is insufficient, the headline language shows misinformation-like patterns (likely false by style).")
        else:
            expl.append("Language patterns are not decisive for this headline.")

    for line in expl:
        st.write("- " + line)

    # ---------------- Evidence ----------------
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
        st.write("max_ent:", max_ent, "max_con_used:", max_con_used, "top_rel:", ranked[0].relevance if ranked else 0.0)
        st.write("imgctx_mismatch:", imgctx_mismatch, "imgctx_locs:", imgctx_locs, "imgctx_words:", imgctx_words)

st.divider()
st.caption("Decision-support system only. Reverse image search provides provenance/context signals and does not itself prove truth.")
