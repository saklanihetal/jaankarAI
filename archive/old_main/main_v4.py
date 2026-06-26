# app_master_v18_dynamic_discovery.py
# MASTER VERSION (v18) - "Dynamic Discovery Engine"
#
# CRITICAL FIX:
# - Solves "404 Model Not Found" by asking Google which models are available 
#   to YOUR specific API Key, instead of guessing hardcoded names.
#
# PIPELINE:
# 1. AUTH: Connect with API Key.
# 2. DISCOVER: Fetch list of available models -> Pick best one (Flash > Pro).
# 3. VERIFY: Run RAG pipeline (Search -> Reason).

import os
import re
import time
import json
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Optional

import streamlit as st
import requests
import google.generativeai as genai
from PIL import Image

# --- CONFIG ---
st.set_page_config(page_title="AI FactCheck: Dynamic Engine", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .verdict-box { padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); background: rgba(255,255,255,0.03); text-align: center; margin-bottom: 20px; }
    .v-val { font-size: 2.5rem; font-weight: 800; margin: 10px 0; }
    .reasoning-text { font-size: 1.05rem; line-height: 1.6; opacity: 0.9; background: rgba(0,0,0,0.2); padding: 15px; border-radius: 8px; border-left: 4px solid #4caf50; }
    .evidence-card { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 12px; margin-bottom: 8px; }
    .status-log { font-family: monospace; font-size: 0.85rem; opacity: 0.8; }
</style>
""", unsafe_allow_html=True)

# --- SETUP KEYS ---
API_KEY = st.secrets.get("GEMINI_API_KEY")

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
    from langdetect import detect
except ImportError:
    detect = lambda x: "en"

# --- UTILS ---
def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", " ", s)

def translate_to_english(text: str) -> str:
    if not text: return ""
    if detect(text) != "en" and GoogleTranslator:
        try: return GoogleTranslator(source='auto', target='en').translate(text)
        except: pass
    return text

def strip_html(s: str) -> str:
    s = s or ""
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()

@dataclass
class EvidenceItem:
    title: str
    url: str
    source: str
    desc: str = ""

# --- 1. RETRIEVAL LAYER (Python) ---
def build_search_queries(claim: str) -> List[str]:
    claim = normalize_text(claim)
    queries = [claim]
    words = claim.split()
    entities = [w for w in words if w[0].isupper() and len(w) > 2]
    if entities:
        queries.append(" ".join(entities))
    return list(set(queries))[:3]

def fetch_evidence(queries: List[str]) -> List[EvidenceItem]:
    items = []
    def fetch_rss(q):
        out = []
        try:
            url = f"https://news.google.com/rss/search?q={requests.utils.quote(q)}&hl=en-IN&gl=IN&ceid=IN:en"
            d = feedparser.parse(url)
            for e in d.entries[:5]:
                out.append(EvidenceItem(title=e.title, url=e.link, source="GoogleNews", desc=strip_html(getattr(e, "summary", ""))))
        except: pass
        return out

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(fetch_rss, q) for q in queries]
        for f in as_completed(futures):
            items.extend(f.result())
            
    seen = set()
    unique = []
    for it in items:
        if it.title not in seen:
            seen.add(it.title)
            unique.append(it)
    return unique[:15]

# --- 2. DYNAMIC MODEL LOADER (Critical Fix) ---
def find_working_model(api_key: str):
    """
    Asks Google: 'What models can I use?'
    Returns the best available model name for this specific API Key.
    """
    genai.configure(api_key=api_key)
    try:
        # List all models available to this key
        all_models = list(genai.list_models())
        
        # Filter for models that support 'generateContent' (Chat/Text)
        capable_models = [m for m in all_models if 'generateContent' in m.supported_generation_methods]
        
        if not capable_models:
            return None, "No text-generation models available for this API Key."

        # Priority Selection: Flash -> Pro -> 1.5 -> 1.0
        # We sort by name preference
        model_names = [m.name for m in capable_models]
        
        # Preference Logic
        best_model = None
        for pref in ['flash', 'gemini-1.5', 'gemini-pro', 'gemini-1.0']:
            match = next((name for name in model_names if pref in name), None)
            if match:
                best_model = match
                break
        
        # Fallback to the first available if no preferences match
        if not best_model:
            best_model = model_names[0]
            
        return best_model, None

    except Exception as e:
        return None, str(e)

# --- 3. REASONING LAYER ---
def verify_with_llm_agent(claim: str, evidence: List[EvidenceItem], api_key: str, model_name: str):
    genai.configure(api_key=api_key)
    
    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        return "ERROR", f"Failed to load model {model_name}: {e}"

    evidence_text = ""
    for i, item in enumerate(evidence):
        evidence_text += f"{i+1}. [{item.source}] {item.title} - {item.desc[:200]}...\n"

    prompt = f"""
    You are an expert Fact Checking AI. Verify this CLAIM against the EVIDENCE.
    
    CLAIM: "{claim}"
    
    EVIDENCE:
    {evidence_text}
    
    INSTRUCTIONS:
    1. **Context & Time:** If evidence has a higher death count than claim, it's an UPDATE (Supported).
    2. **Nuance:** If claim implies real event but evidence says "Replica" or "Mock Drill", mark MISLEADING.
    3. **Tone:** Ignore sensational words like "Shocking" or "Massive" if the core event is true.
    
    OUTPUT JSON:
    {{
        "verdict": "SUPPORTED" | "CONTRADICTED" | "MISLEADING" | "UNVERIFIED",
        "confidence": 0.0-1.0,
        "explanation": "concise reasoning..."
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        txt = response.text.replace("```json", "").replace("```", "").strip()
        result = json.loads(txt)
        return "SUCCESS", result
    except Exception as e:
        return "ERROR", str(e)

# ===========================
# UI
# ===========================
def render_ui():
    st.title("🧠 AI FactCheck: Dynamic Engine")
    st.caption("v18 • Auto-Model Discovery • Intelligent Logic")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        user_key = st.text_input("Gemini API Key", value=API_KEY if API_KEY else "", type="password")
        
        if st.button("Check Available Models"):
            if not user_key:
                st.error("No key provided.")
            else:
                model_name, err = find_working_model(user_key)
                if model_name:
                    st.success(f"Best Model Found: {model_name}")
                else:
                    st.error(f"Discovery Failed: {err}")

    # Main Input
    claim_input = st.text_area("Enter News Headline / Claim", height=100)
    run = st.button("Verify with AI Agent", type="primary", use_container_width=True)

    if run and claim_input:
        if not user_key:
            st.error("⚠️ API Key required.")
            st.stop()

        status = st.status("Investigation in progress...", expanded=True)
        
        # 1. Model Check
        status.write("🔌 Connecting to Google AI...")
        model_name, err = find_working_model(user_key)
        if not model_name:
            status.update(label="Connection Failed", state="error")
            st.error(f"Could not find a usable model: {err}")
            st.stop()
        
        status.write(f"✅ Using Model: `{model_name}`")

        # 2. Search
        status.write("🌐 Translating & Searching...")
        claim_en = translate_to_english(claim_input)
        queries = build_search_queries(claim_en)
        evidence_items = fetch_evidence(queries)
        
        if not evidence_items:
            status.update(label="No evidence found!", state="error")
            st.error("No relevant news articles found.")
            st.stop()
            
        status.write(f"📂 Analyzed {len(evidence_items)} articles. Reasoning...")
        
        # 3. Reason
        status_code, result = verify_with_llm_agent(claim_en, evidence_items, user_key, model_name)
        
        status.update(label="Verification Complete", state="complete", expanded=False)

        if status_code == "ERROR":
            st.error(f"AI Reasoning Failed: {result}")
            st.stop()

        # Display
        verdict = result.get("verdict", "UNVERIFIED").upper()
        color = "#ff9800"
        if verdict == "SUPPORTED": color = "#4caf50"
        if verdict == "CONTRADICTED": color = "#f44336"
        if verdict == "MISLEADING": color = "#ffd700"

        st.markdown(f"""
        <div class="verdict-box" style="border-color: {color}; background: {color}10;">
            <div style="opacity: 0.8; letter-spacing: 2px;">AI VERDICT</div>
            <div class="v-val" style="color: {color};">{verdict}</div>
            <div style="font-size: 1rem;">Confidence: {result.get('confidence', 0):.2f} • Model: {model_name}</div>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("📝 AI Analysis")
        st.markdown(f"""
        <div class="reasoning-text">
        {result.get('explanation', 'No explanation provided.')}
        </div>
        """, unsafe_allow_html=True)

        st.subheader("📚 Source Evidence")
        for i, item in enumerate(evidence_items[:5]):
            st.markdown(f"""
            <div class="evidence-card">
                <b>[{i+1}] {item.title}</b><br>
                <span style="opacity:0.7; font-size:0.9rem;">{item.source}</span>
                <div style="font-size:0.9rem; margin-top:5px; opacity:0.9;">{item.desc[:200]}...</div>
                <a href="{item.url}" target="_blank" style="color: #4da6ff; font-size: 0.85rem;">Read Article</a>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    render_ui()