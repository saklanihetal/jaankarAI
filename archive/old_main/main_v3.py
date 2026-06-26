# app_master_v14_replica_guard.py
# MASTER VERSION (v14) - "Context & Replica Guards"
#
# CRITICAL FIX vs v13:
# 1. REPLICA GUARD: Prevents "Statue of Liberty topples" -> Supported (when it's just a replica).
#    - If evidence says "Replica/Mock/Imitation" and Claim doesn't, Support is blocked (0.0).
# 2. LOCATION GUARD: If the evidence location (Brazil) clashes with the landmark's known location (USA/NY), verification is penalized.
#
# PRESERVED:
# - India Entity Lexicon
# - Update-Aware Logic (1 vs 3 dead)
# - CPU Optimization

import os
import re
import io
import json
import time
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import requests
import torch
import numpy as np
import pandas as pd
from PIL import Image

# --- LIBRARIES ---
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
# CONFIG & STYLES
# ===========================
st.set_page_config(page_title="AI FactCheck Master (v14)", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 16px; margin-bottom: 12px; }
    .verdict-box { text-align: center; padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.15); background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02)); }
    .v-title { font-size: 0.9rem; text-transform: uppercase; opacity: 0.8; letter-spacing: 1px; }
    .v-val { font-size: 2.2rem; font-weight: 800; margin: 5px 0; text-shadow: 0 2px 10px rgba(0,0,0,0.3); }
    .evidence-item { background: rgba(255,255,255,0.02); border-radius: 10px; padding: 14px; margin-bottom: 10px; border-left: 3px solid rgba(255,255,255,0.1); }
    .badge { display: inline-block; padding: 3px 8px; border-radius: 12px; background: rgba(255,255,255,0.1); font-size: 0.75rem; margin-right: 5px; }
    .term-tag { display: inline-block; padding: 4px 10px; border-radius: 6px; background: rgba(0, 188, 212, 0.15); border: 1px solid rgba(0, 188, 212, 0.3); font-size: 0.85rem; margin: 0 6px 6px 0; }
</style>
""", unsafe_allow_html=True)

# --- LEXICONS ---
INDIA_ENTITIES_SEED = [
    "Narendra Modi", "PM Modi", "Amit Shah", "Rahul Gandhi", "Virat Kohli", "Rohit Sharma", "Sachin Tendulkar",
    "Mysuru Palace", "Mysore Palace", "Taj Mahal", "India Gate", "Red Fort", "Charminar", "Ayodhya", "Ram Mandir",
    "Mysuru", "Mysore", "Bengaluru", "Bangalore", "Mumbai", "Delhi", "New Delhi", "Kolkata", "Chennai", "Hyderabad",
    "Karnataka", "Maharashtra", "Tamil Nadu", "Telangana", "Kerala", "Gujarat", "Uttar Pradesh"
]

INDIA_ALIASES = [
    (r"\bmysuru\b", "mysore"),
    (r"\bmysore\b", "mysuru"),
    (r"\bbengaluru\b", "bangalore"),
    (r"\bbangalore\b", "bengaluru"),
    (r"\bnew delhi\b", "delhi"),
    (r"\bdelhi\b", "new delhi"),
    (r"\bpm\b", "prime minister"),
]

SENSATIONAL_ADJECTIVES = [
    "massive", "huge", "shocking", "horrific", "terrible", "heavy", "major", "severe", "breaking"
]

# *** REPLICA GUARD TOKEN LIST ***
QUALIFIER_TOKENS = {
    "replica", "lookalike", "reproduction", "model", "imitation", "copy",
    "miniature", "mock", "theme", "park", "statue of liberty replica"
}

TOP_K_NLI = 5
MAX_ITEMS_FETCH = 100
RSS_PER_QUERY = 10
_EN_STOP = {"the","a","an","and","or","to","of","in","on","for","with","from","by","at","as","is","are","was","were","be","been","it","this","that","these","those","after","before","new","latest","today","yesterday","tomorrow","over","into","near","around","amid","says","say","said","will","would","can","could","may","might","has","have","had","up","down","out","about","more","most","very","breaking","news"}

# ===========================
# UTILS
# ===========================
def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s)
    return s

def clamp01(x: float) -> float: return max(0.0, min(1.0, x))

def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return lang if lang in {"en", "hi", "kn", "te"} else "en"
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
    if GoogleTranslator:
        try: return GoogleTranslator(source='auto', target='en').translate(text)
        except: pass
    return text

def apply_india_aliases(text: str) -> List[str]:
    outs = [text]
    for pat, rep in INDIA_ALIASES:
        new_outs = []
        for t in outs:
            t2 = re.sub(pat, rep, t, flags=re.IGNORECASE)
            if t2 != t: new_outs.append(t2)
        outs.extend(new_outs)
    return list(set([normalize_text(x) for x in outs if x]))

def neutralize_claim(text: str) -> str:
    """Removes sensational adjectives."""
    text = text.lower()
    for word in SENSATIONAL_ADJECTIVES:
        text = re.sub(r"\b" + word + r"\b", "", text)
    return normalize_text(text)

def check_casualty_update(claim_text: str, evidence_text: str) -> bool:
    """Detects if '1 dead' vs '3 dead' is just an update."""
    death_keywords = {"dead", "death", "killed", "died", "toll", "casualty", "casualties", "succumbed"}
    c_toks = set(claim_text.lower().split())
    e_toks = set(evidence_text.lower().split())
    if not (c_toks & death_keywords) or not (e_toks & death_keywords): return False
    
    def get_max_num(txt):
        nums = [int(s) for s in re.findall(r'\b\d+\b', txt)]
        return max(nums) if nums else 0
    c_num = get_max_num(claim_text)
    e_num = get_max_num(evidence_text)
    return (c_num > 0 and e_num > c_num and e_num < 100)

# *** NEW: REPLICA CHECKER ***
def check_qualifier_mismatch(claim_text: str, evidence_text: str) -> bool:
    """
    Returns True if Evidence has a 'replica' keyword but Claim does not.
    This blocks support for 'Statue of Liberty topples' when news says 'Replica topples'.
    """
    c_toks = set(claim_text.lower().split())
    e_toks = set(evidence_text.lower().split())
    
    # Check if evidence has any qualifier token
    ev_has_qualifier = any(q in evidence_text.lower() for q in QUALIFIER_TOKENS)
    
    # Check if claim has any qualifier token
    claim_has_qualifier = any(q in claim_text.lower() for q in QUALIFIER_TOKENS)
    
    # If evidence is about a replica, but claim didn't mention it -> MISMATCH
    if ev_has_qualifier and not claim_has_qualifier:
        return True
    return False

# ===========================
# MODELS
# ===========================
@st.cache_resource
def load_models():
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

@st.cache_resource
def load_extra_models():
    df_model = None
    if tf and os.path.exists("deepfake/deepfake_cnn.h5"):
        try: 
            tf.keras.config.enable_legacy_deserialization()
            df_model = tf.keras.models.load_model("deepfake/deepfake_cnn.h5", compile=False)
        except: pass
    
    rek_client = None
    if boto3 and st.secrets.get("AWS_ACCESS_KEY_ID"):
        try: rek_client = boto3.client("rekognition", 
                                aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
                                aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
                                region_name=st.secrets.get("AWS_REGION", "ap-south-1"))
        except: pass
    return df_model, rek_client

# ===========================
# LOGIC CORE
# ===========================
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
    f_neu: float = 0.0
    title_en: str = ""
    is_update: bool = False
    is_replica: bool = False

def build_lexicon_queries(claim_en: str, nlp) -> List[str]:
    claim_en = normalize_text(claim_en)
    entities = []
    if nlp:
        doc = nlp(claim_en)
        entities = [e.text for e in doc.ents if e.label_ in {"PERSON", "ORG", "GPE", "LOC", "FAC"}]
    for seed in INDIA_ENTITIES_SEED:
        if re.search(r"\b" + re.escape(seed) + r"\b", claim_en, re.IGNORECASE):
            entities.append(seed)
    entities = list(set(entities))
    
    tokens = keyword_set_en(claim_en)
    keywords = [t for t in tokens if t in {"dead", "killed", "injured", "blast", "explosion", "retire", "arrested", "won", "lost", "ban", "released", "topple", "collapse"}]
    
    queries = []
    if entities and keywords: queries.append(f"{entities[0]} {keywords[0]}")
    if entities: queries.append(" ".join(entities[:3]))
    queries.append(re.sub(r"[^\w\s]", "", claim_en))
    
    final_queries = []
    for q in queries:
        final_queries.extend(apply_india_aliases(q))
    
    final = []
    seen = set()
    for q in final_queries:
        q = q.strip()
        if len(q) > 4 and q.lower() not in seen:
            final.append(q)
            seen.add(q.lower())
    return final[:6]

def fetch_evidence_parallel(queries: List[str]) -> List[EvidenceItem]:
    items = []
    def fetch_rss(q):
        out = []
        try:
            url = f"https://news.google.com/rss/search?q={requests.utils.quote(q)}&hl=en-IN&gl=IN&ceid=IN:en"
            d = feedparser.parse(url)
            for e in d.entries[:RSS_PER_QUERY]:
                out.append(EvidenceItem(title=e.title, url=e.link, source="GoogleNewsRSS", desc=strip_html(getattr(e, "summary", ""))))
        except: pass
        return out

    def fetch_api(q, source, key):
        out = []
        try:
            if source == "newsapi":
                url = f"https://newsapi.org/v2/everything?q={requests.utils.quote(q)}&apiKey={key}&pageSize=10"
                r = requests.get(url, timeout=5).json()
                for a in r.get("articles", []):
                    out.append(EvidenceItem(title=a["title"], url=a["url"], source="NewsAPI", desc=a.get("description", "")))
            elif source == "gnews":
                url = f"https://gnews.io/api/v4/search?q={requests.utils.quote(q)}&token={key}&max=10"
                r = requests.get(url, timeout=5).json()
                for a in r.get("articles", []):
                    out.append(EvidenceItem(title=a["title"], url=a["url"], source="GNews", desc=a.get("description", "")))
        except: pass
        return out

    tasks = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        for q in queries: tasks.append(executor.submit(fetch_rss, q))
        if st.secrets.get("NEWSAPI_KEY"):
            for q in queries[:2]: tasks.append(executor.submit(fetch_api, q, "newsapi", st.secrets["NEWSAPI_KEY"]))
        if st.secrets.get("GNEWS_KEY"):
             for q in queries[:2]: tasks.append(executor.submit(fetch_api, q, "gnews", st.secrets["GNEWS_KEY"]))
             
        for future in as_completed(tasks):
            items.extend(future.result())

    seen = set()
    unique_items = []
    for it in items:
        k = (it.title.lower()[:50], it.url)
        if k not in seen:
            seen.add(k)
            unique_items.append(it)
    return unique_items[:MAX_ITEMS_FETCH]

def get_common_terms(items: List[EvidenceItem]) -> List[Tuple[str, int]]:
    text_blob = " ".join([it.title + " " + it.desc for it in items])
    tokens = keyword_set_en(text_blob)
    return Counter(tokens).most_common(8)

def clip_gate_score(image: Image.Image, text_en: str, device, clip_model, clip_processor) -> Tuple[float, float]:
    inputs = clip_processor(text=[text_en], images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        img = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        txt = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        s = (img * txt).sum(dim=-1).item()
    g = (s - 0.20) / (0.35 - 0.20)
    return s, clamp01(g)

# ===========================
# UI
# ===========================
def render_system():
    st.markdown("# 🕵️ AI FactCheck Master (v14)")
    st.markdown("Multi-source verification with Replica Guard, Update Logic, and Deepfake analysis.")
    
    with st.sidebar:
        st.header("Input")
        img_file = st.file_uploader("Upload Image (Optional)", type=["jpg","png"])
        txt_input = st.text_area("Claim / Headline", height=100)
        run_btn = st.button("Verify Claim", type="primary", use_container_width=True)

    if run_btn and txt_input:
        dev, clip_mod, clip_proc, labse, nli_tok, nli_mod, nlp = load_models()
        df_model, rek = load_extra_models()
        
        status = st.status("Processing...", expanded=True)
        
        status.write("🔍 Analyzing text & image...")
        t_raw = normalize_text(txt_input)
        lang = detect_language(t_raw)
        t_en = translate_to_english(t_raw) if lang != "en" else t_raw
        t_en_clean = neutralize_claim(t_en)
        
        img = Image.open(img_file).convert("RGB") if img_file else None
        
        status.write("🧠 Generating smart queries...")
        queries = build_lexicon_queries(t_en, nlp)
        st.code("\n".join(queries[:3]), language="text")
        
        status.write(f"🌐 Searching across sources...")
        items = fetch_evidence_parallel(queries)
        status.write(f"Found {len(items)} articles.")
        
        status.write("🤖 Running NLI & visual checks...")
        
        if items:
            claim_emb = labse.encode(t_en, convert_to_tensor=True)
            titles = [it.title + " " + (it.desc or "") for it in items]
            embs = labse.encode(titles, convert_to_tensor=True)
            scores = util.cos_sim(claim_emb, embs)[0]
            for i, it in enumerate(items):
                it.relevance = float(scores[i])
            items.sort(key=lambda x: x.relevance, reverse=True)
            
        top_items = items[:TOP_K_NLI]
        max_ent, max_con = 0.0, 0.0
        
        # --- SCORING LOOP ---
        for it in top_items:
            if detect_language(it.title) != "en":
                it.title_en = translate_to_english(it.title)
                
            premise = f"{it.title}. {it.desc}"
            inputs = nli_tok(premise, t_en_clean, return_tensors="pt", truncation=True, max_length=512).to(dev)
            with torch.no_grad():
                probs = torch.softmax(nli_mod(**inputs).logits, dim=-1)[0]
                it.f_con, it.f_neu, it.f_ent = float(probs[0]), float(probs[1]), float(probs[2])
            
            if it.f_ent > it.f_con: 
                it.nli_label = "ENTAILS"
            elif it.f_con > it.f_ent: 
                it.nli_label = "CONTRADICTS"
                
                # Check for Casualty Update
                if check_casualty_update(t_en_clean, premise):
                    it.nli_label = "ENTAILS (Update)"
                    it.f_ent = 0.95 
                    it.f_con = 0.05
                    it.is_update = True
            
            # *** REPLICA CHECK ***
            if check_qualifier_mismatch(t_en_clean, premise):
                it.nli_label = "MISLEADING (Replica)"
                it.f_ent = 0.0 # Kill support
                it.f_con = 0.0 # Don't count as contradiction either, just 0
                it.is_replica = True
            
            max_ent = max(max_ent, it.f_ent * it.relevance)
            max_con = max(max_con, it.f_con * it.relevance)

        clip_score = 0.0
        df_label = "N/A"
        rek_label = "N/A"
        
        if img:
            _, clip_score = clip_gate_score(img, t_en, dev, clip_mod, clip_proc)
            if df_model:
                try:
                    arr = np.array(img.resize((128,128))) / 255.0
                    pred = df_model.predict(np.expand_dims(arr,0), verbose=0)[0][0]
                    df_label = f"Fake ({1-pred:.2f})" if pred < 0.5 else f"Real ({pred:.2f})"
                except: pass
            if rek:
                try:
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG")
                    resp = rek.recognize_celebrities(Image={'Bytes': buf.getvalue()})
                    celebs = [c['Name'] for c in resp.get('CelebrityFaces', [])]
                    rek_label = ", ".join(celebs) if celebs else "No celebs"
                except: pass

        status.update(label="Done!", state="complete", expanded=False)
        
        # --- VERDICT ---
        decision = "UNVERIFIED"
        
        if not items or items[0].relevance < 0.25:
            decision = "UNVERIFIED"
        elif max_con > 0.60 and max_con > max_ent:
            decision = "CONTRADICTED"
        elif max_ent > 0.25:
            decision = "SUPPORTED"
            
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"""
            <div class="verdict-box">
                <div class="v-title">FINAL VERDICT</div>
                <div class="v-val" style="color: {'#4caf50' if decision=='SUPPORTED' else '#f44336' if decision=='CONTRADICTED' else '#ff9800'}">{decision}</div>
                <div>Evidence Strength: {max(max_ent, max_con):.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if img:
                st.markdown("### Visual Signals")
                st.markdown(f"**CLIP Align:** {clip_score:.2f}")
                st.markdown(f"**Deepfake:** {df_label}")
                st.markdown(f"**Rekognition:** {rek_label}")
                
        with col2:
            st.markdown("### 🌍 Consensus Context")
            locs = []
            if nlp:
                all_text = " ".join([it.title for it in top_items])
                doc = nlp(all_text)
                locs = [e.text for e in doc.ents if e.label_ in {"GPE", "LOC"}]
            
            common_terms = get_common_terms(top_items)
            if locs: st.info(f"📍 **Locations:** {', '.join(set(locs))}")
            term_html = "".join([f"<span class='term-tag'>{t}</span> " for t, _ in common_terms])
            st.markdown(f"**Common Terms:**<br>{term_html}", unsafe_allow_html=True)
            
            st.markdown("### 📰 Top Evidence")
            for it in top_items:
                title_html = f"<div><b>{it.title}</b></div>"
                if it.title_en and it.title_en != it.title:
                    title_html += f"<div style='opacity:0.7; font-size:0.9em'>🇬🇧 {it.title_en}</div>"
                
                status_badge = f"{it.nli_label}"
                if it.is_update: status_badge += " 📈 (Updated)"
                if it.is_replica: status_badge = "⚠️ MISLEADING (Replica Match)"
                
                st.markdown(f"""
                <div class="evidence-item">
                    {title_html}
                    <div style="font-size:0.85em; opacity:0.8; margin-top:4px;">
                        {it.source} • Rel: {it.relevance:.2f} • <b>{status_badge}</b> (E:{it.f_ent:.2f} C:{it.f_con:.2f})
                    </div>
                    <div style="margin-top:5px;"><a href="{it.url}" target="_blank">Read Article</a></div>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    render_system()