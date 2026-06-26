# app_master_v31_smart_loader.py
# MASTER VERSION (v31) - "The Robust Monolith"
#
# BUG FIX (CRITICAL): 
# - Adds 'load_safe_model' helper to fix AttributeError.
# - If a .joblib file loads as a dictionary, it automatically finds the model inside.
#
# FEATURES:
# - Full Retrieval (Parallel APIs + RSS)
# - Full Analytics (NLI, CLIP, Deepfake, Gates, Style)
# - Stealth Reasoning (Gemini logic hidden)
# - UI (Dark Mode + Debug Console)

import os
import re
import io
import json
import time
import unicodedata
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

# --- CORE IMPORTS (GLOBAL) ---
import streamlit as st
import requests
import numpy as np
import pandas as pd
from PIL import Image
import torch
import google.generativeai as genai

# --- OPTIONAL IMPORTS ---
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
try:
    import tensorflow as tf
except Exception:
    tf = None

# ===========================
# 1. CONFIG & STYLES
# ===========================
st.set_page_config(page_title="FactCheck Ultimate", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 4rem; }
    .verdict-box { 
        text-align: center; padding: 25px; border-radius: 12px; 
        border: 1px solid rgba(255,255,255,0.15); 
        background: linear-gradient(135deg, rgba(255,255,255,0.09) 0%, rgba(255,255,255,0.03) 100%); 
        margin-bottom: 25px;
        box-shadow: 0 4px 25px rgba(0,0,0,0.25);
    }
    .v-label { font-size: 0.9rem; letter-spacing: 2px; text-transform: uppercase; opacity: 0.7; margin-bottom: 5px; }
    .v-val { font-size: 3.2rem; font-weight: 800; margin: 0; text-shadow: 0 2px 10px rgba(0,0,0,0.5); }
    .card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 15px; margin-bottom: 12px; }
    .evidence-item { 
        background: rgba(255,255,255,0.025); border-radius: 8px; padding: 15px; 
        margin-bottom: 12px; border-left: 4px solid rgba(255,255,255,0.1);
        transition: transform 0.1s;
    }
    .evidence-item:hover { background: rgba(255,255,255,0.05); }
    .badge { display: inline-block; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; margin-right: 5px; }
    .term-tag { display: inline-block; padding: 4px 10px; background: rgba(33, 150, 243, 0.15); border: 1px solid rgba(33, 150, 243, 0.3); color: #bbdefb; border-radius: 15px; font-size: 0.8rem; margin: 3px; }
    .analysis-text { font-size: 1.05rem; line-height: 1.7; color: #e0e0e0; background: rgba(0,0,0,0.2); padding: 20px; border-radius: 8px; border-left: 4px solid #2196f3; }
</style>
""", unsafe_allow_html=True)

# KEYS
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY")
NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY")
GNEWS_KEY = st.secrets.get("GNEWS_KEY")
NEWSDATA_KEY = st.secrets.get("NEWSDATA_KEY")
EVENTREGISTRY_KEY = st.secrets.get("EVENTREGISTRY_KEY")
AWS_KEY = st.secrets.get("AWS_ACCESS_KEY_ID")
AWS_SECRET = st.secrets.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = st.secrets.get("AWS_REGION", "ap-south-1")

TOP_K_NLI = 5
MAX_ITEMS_FETCH = 140
RSS_PER_QUERY = 12
_EN_STOP = {"the","a","an","and","or","to","of","in","on","for","with","from","by","at","as","is","are","was","were","be","been","it","this","that","these","those","after","before","new","latest","today","yesterday","tomorrow","over","into","near","around","amid","says","say","said","will","would","can","could","may","might","has","have","had","up","down","out","about","more","most","very","breaking","news"}

# INDIA LEXICON
INDIA_ALIASES = [
    (r"\bmysuru\b", "mysore"), (r"\bmysore\b", "mysuru"),
    (r"\bbengaluru\b", "bangalore"), (r"\bbangalore\b", "bengaluru"),
    (r"\bnew delhi\b", "delhi"), (r"\bdelhi\b", "new delhi"),
    (r"\bmumbai\b", "bombay"), (r"\bchennai\b", "madras"),
    (r"\bkolkata\b", "calcutta"), (r"\bpm\b", "prime minister"),
    (r"\bcm\b", "chief minister"),
]

# ===========================
# 2. UTILITY FUNCTIONS
# ===========================
def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", " ", s)

def detect_lang_safe(text: str) -> str:
    try: return detect(text) if detect(text) in {"en", "hi", "kn", "te"} else "en"
    except: return "en"

def keyword_set_en(text: str) -> List[str]:
    text = normalize_text(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 2 and t not in _EN_STOP]

def strip_html(s: str) -> str:
    s = s or ""
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def translate_to_english(text: str) -> str:
    if not text: return ""
    if len(text.encode('utf-8')) == len(text): return text
    if GoogleTranslator:
        try: return GoogleTranslator(source='auto', target='en').translate(text)
        except: pass
    return text

def safe_get_json(url: str) -> dict:
    try:
        r = requests.get(url, timeout=8, headers={"User-Agent": "FactCheckBot/1.0"})
        if r.status_code == 200: return r.json()
    except: pass
    return {}

def safe_post_json(url: str, payload: dict) -> dict:
    try:
        r = requests.post(url, json=payload, timeout=8, headers={"User-Agent": "FactCheckBot/1.0"})
        if r.status_code == 200: return r.json()
    except: pass
    return {}

# ===========================
# 3. MODEL LOADING (Cached)
# ===========================
@st.cache_resource
def load_core_models():
    if 'torch' not in globals(): import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    labse = SentenceTransformer("sentence-transformers/LaBSE")
    
    nli_tok = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
    nli_model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli").to(device)
    
    nlp = None
    try: nlp = spacy.load("en_core_web_trf")
    except: 
        try: nlp = spacy.load("en_core_web_sm")
        except: pass
        
    return device, clip_model, clip_processor, labse, nli_tok, nli_model, nlp

def load_safe_model(path):
    """Robust loader that handles both direct model objects and dictionary wrappers."""
    if not os.path.exists(path): return None
    try:
        obj = joblib.load(path)
        # Case 1: Object is the model itself
        if hasattr(obj, "predict_proba"): return obj
        # Case 2: Object is a dictionary containing the model
        if isinstance(obj, dict):
            # Try to find a value that looks like a model
            for k, v in obj.items():
                if hasattr(v, "predict_proba"): return v
        return None
    except: return None

@st.cache_resource
def load_auxiliary_models():
    # Deepfake
    df_model = None
    if tf and os.path.exists("deepfake/deepfake_cnn.h5"):
        try: 
            tf.keras.config.enable_legacy_deserialization()
            df_model = tf.keras.models.load_model("deepfake/deepfake_cnn.h5", compile=False)
        except: pass
        
    # AWS
    rek_client = None
    if boto3 and AWS_KEY:
        try: rek_client = boto3.client("rekognition", aws_access_key_id=AWS_KEY, aws_secret_access_key=AWS_SECRET, region_name=AWS_REGION)
        except: pass
        
    # Gates & Style (Safe Load)
    s_gate = load_safe_model("support_gate.joblib")
    c_gate = load_safe_model("contradiction_gate.joblib")
    tfidf = load_safe_model("tfidf_style_model.joblib")
        
    return df_model, rek_client, s_gate, c_gate, tfidf

@dataclass
class EvidenceItem:
    title: str
    url: str
    source: str
    desc: str = ""
    relevance: float = 0.0
    nli_label: str = "NEUTRAL"
    f_ent: float = 0.0
    f_con: float = 0.0
    title_original: str = ""

# ===========================
# 4. PIPELINE LOGIC
# ===========================

def apply_aliases(text: str) -> List[str]:
    variations = {text}
    for pat, rep in INDIA_ALIASES:
        if re.search(pat, text, re.IGNORECASE):
            variations.add(re.sub(pat, rep, text, flags=re.IGNORECASE))
    return list(variations)

def build_queries(claim_en: str, nlp) -> List[str]:
    claim_en = normalize_text(claim_en)
    base_q = re.sub(r"[^\w\s]", "", claim_en)
    queries = [base_q]
    
    if nlp:
        doc = nlp(claim_en)
        ents = [e.text for e in doc.ents if e.label_ in {"PERSON", "ORG", "GPE", "LOC", "EVENT"}]
        if ents: queries.append(" ".join(ents[:3]))
    
    final_queries = []
    for q in queries:
        final_queries.extend(apply_aliases(q))
    return list(set(final_queries))[:5]

def fetch_evidence_full_stack(queries: List[str]) -> Tuple[List[EvidenceItem], int]:
    raw_count = 0
    items = []
    
    def fetch_rss(q):
        out = []
        try:
            url = f"https://news.google.com/rss/search?q={requests.utils.quote(q)}&hl=en-IN&gl=IN&ceid=IN:en"
            d = feedparser.parse(url)
            for e in d.entries[:RSS_PER_QUERY]:
                out.append(EvidenceItem(title=e.title, url=e.link, source="GoogleRSS", desc=strip_html(getattr(e, "summary", "")), title_original=e.title))
        except: pass
        return out

    def fetch_newsapi(q):
        out = []
        if not NEWSAPI_KEY: return []
        try:
            url = f"https://newsapi.org/v2/everything?q={requests.utils.quote(q)}&apiKey={NEWSAPI_KEY}&pageSize=10&language=en"
            r = safe_get_json(url)
            if r.get("status") == "ok":
                for a in r.get("articles", []):
                    out.append(EvidenceItem(title=a["title"], url=a["url"], source="NewsAPI", desc=a.get("description", ""), title_original=a["title"]))
        except: pass
        return out

    def fetch_gnews(q):
        out = []
        if not GNEWS_KEY: return []
        try:
            url = f"https://gnews.io/api/v4/search?q={requests.utils.quote(q)}&token={GNEWS_KEY}&max=10&lang=en"
            r = safe_get_json(url)
            for a in r.get("articles", []):
                out.append(EvidenceItem(title=a["title"], url=a["url"], source="GNews", desc=a.get("description", ""), title_original=a["title"]))
        except: pass
        return out

    def fetch_newsdata(q):
        out = []
        if not NEWSDATA_KEY: return []
        try:
            url = f"https://newsdata.io/api/1/news?q={requests.utils.quote(q)}&language=en&apikey={NEWSDATA_KEY}"
            r = safe_get_json(url)
            for a in r.get("results", []):
                out.append(EvidenceItem(title=a["title"], url=a["link"], source="NewsData", desc=a.get("description", ""), title_original=a["title"]))
        except: pass
        return out

    def fetch_eventregistry(q):
        out = []
        if not EVENTREGISTRY_KEY: return []
        try:
            url = "https://eventregistry.org/api/v1/article/getArticles"
            payload = {"action": "getArticles", "keyword": q, "lang": "eng", "articlesCount": 10, "apiKey": EVENTREGISTRY_KEY}
            r = safe_post_json(url, payload)
            for a in r.get("articles", {}).get("results", []):
                out.append(EvidenceItem(title=a["title"], url=a["url"], source="EventRegistry", desc=a.get("body", "")[:300], title_original=a["title"]))
        except: pass
        return out

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for q in queries:
            futures.append(executor.submit(fetch_rss, q))
            if NEWSAPI_KEY: futures.append(executor.submit(fetch_newsapi, q))
            if GNEWS_KEY: futures.append(executor.submit(fetch_gnews, q))
            if NEWSDATA_KEY: futures.append(executor.submit(fetch_newsdata, q))
            if EVENTREGISTRY_KEY: futures.append(executor.submit(fetch_eventregistry, q))
            
        for f in as_completed(futures):
            res = f.result()
            raw_count += len(res)
            items.extend(res)
            
    seen = set()
    unique = []
    for it in items:
        k = it.title.lower()[:50]
        if k not in seen:
            seen.add(k)
            unique.append(it)
    return unique[:MAX_ITEMS_FETCH], raw_count

def run_local_analytics(items, claim_en, img, resources, extras):
    dev, clip_mod, clip_proc, labse, nli_tok, nli_mod, nlp = resources
    df_model, rek, s_gate, c_gate, tfidf = extras
    
    if items:
        claim_emb = labse.encode(claim_en, convert_to_tensor=True)
        titles = [it.title + " " + it.desc for it in items]
        embs = labse.encode(titles, convert_to_tensor=True)
        scores = util.cos_sim(claim_emb, embs)[0]
        for i, it in enumerate(items):
            it.relevance = float(scores[i])
        items.sort(key=lambda x: x.relevance, reverse=True)
        
    top_items = items[:TOP_K_NLI]
    for it in top_items:
        if detect_lang_safe(it.title_original) != "en":
            it.title = translate_to_english(it.title_original)
            
        premise = f"{it.title}. {it.desc}"
        inputs = nli_tok(premise, claim_en, return_tensors="pt", truncation=True, max_length=512).to(dev)
        with torch.no_grad():
            probs = torch.softmax(nli_mod(**inputs).logits, dim=-1)[0]
            it.f_con, it.f_ent = float(probs[0]), float(probs[2])
        if it.f_ent > 0.5: it.nli_label = "ENTAILS"
        elif it.f_con > 0.5: it.nli_label = "CONTRADICTED"
        else: it.nli_label = "NEUTRAL"

    clip_val = 0.0
    df_prob = 0.0
    celebs = []
    
    if img:
        inputs = clip_proc(text=[claim_en], images=img, return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            out = clip_mod(**inputs)
            clip_val = (out.image_embeds @ out.text_embeds.T).item()
        
        if df_model:
            try:
                arr = np.array(img.resize((128,128))) / 255.0
                pred = df_model.predict(np.expand_dims(arr,0), verbose=0)[0][0]
                df_prob = 1.0 - pred 
            except: pass
            
        if rek:
            try:
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                r = rek.recognize_celebrities(Image={'Bytes': buf.getvalue()})
                celebs = [c['Name'] for c in r.get('CelebrityFaces', [])]
            except: pass

    gate_signal = "Neutral"
    feat_vec = []
    if top_items and (s_gate or c_gate):
        feat_vec = [
            len(top_items),
            top_items[0].relevance,
            np.mean([x.relevance for x in top_items]),
            max([x.f_ent for x in top_items]),
            np.mean([x.f_ent for x in top_items]),
            max([x.f_con for x in top_items]),
            np.mean([x.f_con for x in top_items])
        ]
        while len(feat_vec) < 11: feat_vec.append(0.0)
        
        # Safe Prediction
        s_prob = 0
        if s_gate:
            try: s_prob = s_gate.predict_proba([feat_vec])[0][1]
            except: pass
            
        c_prob = 0
        if c_gate:
            try: c_prob = c_gate.predict_proba([feat_vec])[0][1]
            except: pass
        
        if s_prob > 0.75: gate_signal = f"High Support ({s_prob:.2f})"
        elif c_prob > 0.75: gate_signal = f"High Contradiction ({c_prob:.2f})"

    style_risk = "Low"
    if tfidf:
        try:
            sp = tfidf.predict_proba([claim_en])[0]
            if sp[1] > 0.75: style_risk = f"High Fake Style ({sp[1]:.2f})"
        except: pass

    common_terms = []
    if items:
        blob = " ".join([it.title for it in top_items])
        common_terms = Counter(keyword_set_en(blob)).most_common(10)

    return top_items, clip_val, df_prob, celebs, gate_signal, common_terms, style_risk, feat_vec

def get_stealth_verdict(claim, evidence, clip, df_prob, celebs, gate, style, api_key):
    if not api_key: return "ERROR", {}, "Config Missing"
    genai.configure(api_key=api_key)
    
    model = None
    try:
        avail = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for pref in ['flash', 'pro', '1.0']:
             found = next((m for m in avail if pref in m), None)
             if found: 
                 model = genai.GenerativeModel(found)
                 break
        if not model: model = genai.GenerativeModel(avail[0])
    except:
        model = genai.GenerativeModel('gemini-pro')

    if not model: return "ERROR", {}, "Reasoning Engine Init Failed"

    ev_text = ""
    for i, it in enumerate(evidence):
        ev_text += f"[{i+1}] {it.title}\n    Source: {it.source} | NLI: {it.nli_label} (Ent:{it.f_ent:.2f}, Con:{it.f_con:.2f})\n\n"

    prompt = f"""
    Act as a professional Fact Checker. Analyze this Claim using the Evidence and Signal Data.
    CLAIM: "{claim}"
    
    SIGNALS:
    - Visual Align: {clip:.2f} (High > 0.25)
    - Manipulation: {df_prob:.2f} (High > 0.5 is FAKE)
    - Entities: {', '.join(celebs) if celebs else 'None'}
    - ML Gate Verdict: {gate}
    - Language Style Risk: {style}
    
    EVIDENCE:
    {ev_text}
    
    INSTRUCTIONS:
    1. **Visuals:** If Manipulation > 0.5, verdict is MISLEADING/FAKE IMAGE.
    2. **Updates:** If NLI says 'Contradicts' but evidence shows a death toll increase (e.g. 1->3), verdict is SUPPORTED.
    3. **Replicas:** If evidence mentions 'Replica', 'Mock Drill', or 'Scene', verdict is MISLEADING.
    4. **Context:** Use the ML Gate and Style signals as confidence boosters.
    
    OUTPUT JSON:
    {{
        "verdict": "SUPPORTED" | "CONTRADICTED" | "MISLEADING" | "UNVERIFIED",
        "confidence": 0.0-1.0,
        "analysis": "Professional explanation referencing specific evidence indices..."
    }}
    """
    try:
        resp = model.generate_content(prompt)
        txt = resp.text.replace("```json", "").replace("```", "").strip()
        return "SUCCESS", json.loads(txt), "Logic Engine (v31)"
    except Exception as e:
        return "ERROR", {}, str(e)

# ===========================
# 5. UI
# ===========================
def render_ui():
    st.title("FactCheck Console")
    st.caption("Integrated Verification System (Legacy Stack + Neuro-Symbolic Logic)")

    with st.sidebar:
        st.header("System Status")
        user_key = st.text_input("System Key (Logic)", value=GEMINI_KEY if GEMINI_KEY else "", type="password")
        
        c1, c2 = st.columns(2)
        with c1:
            if NEWSAPI_KEY: st.success("NewsAPI: ON")
            else: st.warning("NewsAPI: OFF")
            if NEWSDATA_KEY: st.success("NewsData: ON")
            else: st.warning("NewsData: OFF")
        with c2:
            if GNEWS_KEY: st.success("GNews: ON")
            else: st.warning("GNews: OFF")
            if EVENTREGISTRY_KEY: st.success("EventReg: ON")
            else: st.warning("EventReg: OFF")
            
        st.markdown("---")
        show_debug = st.checkbox("Show Debug Console", value=False)

    img_file = st.file_uploader("Upload Media (Optional)", type=["jpg","png"])
    if img_file: st.image(img_file, caption="Analyzed Media", width=300)
    
    claim_input = st.text_area("Investigation Subject", height=100, placeholder="Enter a headline or claim...")
    
    run_btn = st.button("RUN VERIFICATION", type="primary", use_container_width=True)

    if run_btn and claim_input:
        if not user_key:
            st.error("System Key (Logic) is required.")
            st.stop()
            
        status = st.status("Processing...", expanded=True)
        
        status.write("• Loading Neural Models (BART, CLIP, LaBSE)...")
        models = load_core_models()
        extras = load_auxiliary_models()
        
        status.write("• Processing Input & Generating Queries...")
        img = Image.open(img_file).convert("RGB") if img_file else None
        claim_en = translate_to_english(normalize_text(claim_input))
        queries = build_queries(claim_en, models[-1]) 
        
        status.write(f"• Executing Multi-Source Search: {queries}")
        items, raw_count = fetch_evidence_full_stack(queries)
        status.write(f"• Retrieved {raw_count} documents. Filtering & Scoring...")
        
        if not items:
            status.update(label="Process Failed", state="error")
            st.error("No relevant data found in search index.")
            st.stop()
            
        status.write("• Running Local Analytics (NLI, Visuals, Gates)...")
        top_items, clip_val, df_prob, celebs, gate_sig, terms, style_risk, feat_vec = run_local_analytics(items, claim_en, img, models, extras)
        
        status.write("• Synthesizing Final Verdict...")
        code, res, err = get_stealth_verdict(claim_en, top_items, clip_val, df_prob, celebs, gate_sig, style_risk, user_key)
        
        status.update(label="Verification Complete", state="complete", expanded=False)
        
        if code == "ERROR":
            st.error(f"Reasoning Engine Failure: {err}")
            st.stop()
            
        verdict = res.get("verdict", "UNVERIFIED").upper()
        conf = res.get("confidence", 0.0)
        v_color = "#ff9800"
        if verdict == "SUPPORTED": v_color = "#4caf50"
        if verdict == "CONTRADICTED": v_color = "#f44336"
        if verdict == "MISLEADING": v_color = "#ffd700"
        
        col_res, col_sig = st.columns([1.2, 1])
        
        with col_res:
            st.markdown(f"""
            <div class="verdict-box" style="border-color: {v_color}; background: {v_color}15;">
                <div class="v-label" style="color:{v_color}">Global Verdict</div>
                <div class="v-val" style="color:{v_color}">{verdict}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"<div class='analysis-text'><b>System Analysis:</b><br>{res.get('analysis', 'No report.')}</div>", unsafe_allow_html=True)

        with col_sig:
            df_str = f"Authentic ({1-df_prob:.2f})" if df_prob < 0.5 else f"High Risk ({df_prob:.2f})"
            ent_str = ", ".join(celebs) if celebs else "None Detected"
            st.markdown(f"""
            <div class="card">
                <b>Visual Analysis</b><br>
                • Text-Image Align: <code>{clip_val:.2f}</code><br>
                • Media Integrity: <code>{df_str}</code><br>
                • Entities (AWS): <code>{ent_str}</code><br>
                <b>Linguistic Analysis</b><br>
                • ML Gate: <code>{gate_sig}</code><br>
                • Style Risk: <code>{style_risk}</code>
            </div>
            """, unsafe_allow_html=True)
            
            if terms:
                st.markdown("<b>Consensus Context:</b>", unsafe_allow_html=True)
                tags_html = "".join([f"<span class='term-tag'>{t}</span>" for t, _ in terms])
                st.markdown(f"<div>{tags_html}</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader(f"📚 Top {len(top_items)} Verified Sources")
        
        for it in top_items:
            title_html = it.title
            if it.title_original and it.title_original != it.title:
                title_html = f"{it.title_original}<br><span style='font-size:0.85rem; opacity:0.7'>🇬🇧 {it.title}</span>"
            
            nli_bg = "#4caf50aa" if it.nli_label == "ENTAILS" else "#f44336aa" if it.nli_label == "CONTRADICTED" else "rgba(255,255,255,0.1)"
            
            st.markdown(f"""
            <div class="evidence-item">
                <div style="font-weight:700; margin-bottom:4px; line-height:1.2;">{title_html}</div>
                <div style="font-size:0.85rem; opacity:0.7; margin-bottom:6px;">
                    {it.source} • Rel: {it.relevance:.2f} • <span class='badge' style='background:{nli_bg}'>{it.nli_label}</span>
                </div>
                <div style="font-size:0.9rem; opacity:0.9;">{it.desc[:250]}...</div>
                <div style="margin-top:6px;"><a href="{it.url}" target="_blank" style="color:#4da6ff; text-decoration:none; font-size:0.85rem;">Read Full Source →</a></div>
            </div>
            """, unsafe_allow_html=True)
            
        if show_debug:
            st.markdown("---")
            st.subheader("🛠️ System Diagnostics")
            with st.expander("View Raw Logic State", expanded=True):
                st.write("**Queries Generated:**", queries)
                st.write("**Fetch Statistics:**", {"raw": raw_count, "kept": len(items)})
                if 'feat_vec' in locals():
                    st.write("**ML Gate Vector (11-dim):**", feat_vec)
                if 'err' in locals() and err:
                    st.error(f"Last Error: {err}")

if __name__ == "__main__":
    render_ui()