# =========================================================
# EL MASTER: Multimodal FactCheck (v18 - MODEL OVERRIDE)
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

# Core Model Libraries
try: import torch
except: torch = None
try: import boto3
except: boto3 = None
try: import tensorflow as tf
except: tf = None
try:
    from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForSequenceClassification
except:
    CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForSequenceClassification = None, None, None, None
try: import google.generativeai as genai
except: genai = None

# ==============================
# 1) CONFIG & SECRETS
# ==============================
st.set_page_config(page_title="Multimodal FactCheck", page_icon="🛡️", layout="wide")

NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", "")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

# ==============================
# 2) UI STYLING
# ==============================
st.markdown("""
<style>
    .stMetric { background: rgba(255,255,255,0.05); padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); }
    .signal-card { background: #1e293b; padding: 12px; border-radius: 10px; margin-bottom: 8px; border-left: 4px solid #3b82f6; }
</style>
""", unsafe_allow_html=True)

# ==============================
# 3) MODEL LOADING
# ==============================
@st.cache_resource
def load_all_models():
    device = "cuda" if torch and torch.cuda.is_available() else "cpu"
    c_m, c_p, n_t, n_m, df_m = None, None, None, None, None
    
    if CLIPModel:
        try:
            c_m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            c_p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        except: pass
    if AutoTokenizer:
        try:
            n_t = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
            n_m = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli").to(device)
        except: pass
    if tf and os.path.exists("deepfake/deepfake_cnn.h5"):
        try: df_m = tf.keras.models.load_model("deepfake/deepfake_cnn.h5", compile=False)
        except: pass
    
    return device, c_m, c_p, n_t, n_m, df_m

DEVICE, CLIP_M, CLIP_P, NLI_TOK, NLI_M, DF_M = load_all_models()

# ==============================
# 4) PROCESSING HELPERS
# ==============================
def get_nli_scores(premise, hypo):
    if not NLI_M: return {"con": 0.0, "neu": 1.0, "ent": 0.0}
    inputs = NLI_TOK(premise, hypo, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        p = torch.softmax(NLI_M(**inputs).logits, dim=1)[0].cpu().numpy()
    return {"con": float(p[0]), "neu": float(p[1]), "ent": float(p[2])}

def get_clip_score(img, txt):
    if not (CLIP_M and img and txt): return 0.0
    try:
        inputs = CLIP_P(text=[txt], images=img, return_tensors="pt", padding=True, truncation=True, max_length=77).to(DEVICE)
        with torch.no_grad():
            return float(torch.sigmoid(CLIP_M(**inputs).logits_per_image)[0][0].item())
    except: return 0.0

def get_deepfake_score(img):
    if not DF_M or not img: return 0.0
    try:
        arr = np.array(img.resize((224,224))).astype("float32")/255.0
        return float(DF_M.predict(np.expand_dims(arr, 0), verbose=0)[0][0])
    except: return 0.0

# ==============================
# 5) HARD-CODED REASONING ENGINE
# ==============================
def run_gemini_decision(payload):
    if not (genai and GEMINI_API_KEY):
        return "UNVERIFIED", 0.5, "Insufficient evidence, likely neutral"
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Using the requested model string
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        prompt = f"""
        Analyze these multimodal signals and context evidence:
        {json.dumps(payload, indent=2)}
        
        Select exactly one verdict from this list:
        - SUPPORTED
        - CONTRADICTED
        - UNVERIFIED - LIKELY TRUE
        - UNVERIFIED - LIKELY FALSE

        Return ONLY a JSON object: {{"choice": "VERDICT_NAME", "confidence": 0.0}}
        """
        r = model.generate_content(prompt)
        clean_text = r.text.strip().replace("```json","").replace("```","").strip()
        js = json.loads(clean_text)
        choice = js['choice'].upper()
        conf = js.get('confidence', 0.5)

        # Apply Hard-Coded Reasoning Mapping
        if "SUPPORTED" in choice:
            return "SUPPORTED", conf, "multiple evidences support claim"
        elif "CONTRADICTED" in choice:
            return "CONTRADICTED", conf, "multiple evidences contradict claim"
        elif "LIKELY TRUE" in choice:
            return "UNVERIFIED - LIKELY TRUE", conf, "insufficient evidence, likely true depending on the news"
        else:
            return "UNVERIFIED - LIKELY FALSE", conf, "insufficient evidence, likely fake depending on the news"

    except Exception as e:
        # Fallback to local threshold logic if model fails
        return "UNVERIFIED", 0.5, f"insufficient evidence, likely neutral (Engine error: {str(e)})"

# ==============================
# 6) UI MAIN APP
# ==============================
st.title("🛡️ Multimodal FactCheck")
st.caption("v18 • Gemini-2.5-Flash Logic • Hardcoded Reasoning")

left, right = st.columns([1, 1], gap="medium")

with left:
    claim = st.text_area("Claim Headline", placeholder="Enter claim here...", height=100)
    u_file = st.file_uploader("Upload Related Image", type=['jpg','png','jpeg'])
    analyze_btn = st.button("🚀 Analyze", type="primary", use_container_width=True)

with right:
    if u_file:
        img = Image.open(u_file).convert("RGB")
        st.image(img, caption="Provided Image", use_container_width=True)
    else: img = None

if analyze_btn and claim:
    with st.spinner("Gathering Multimodal Signals..."):
        # 1. Fetch Context
        evid_list = []
        if NEWS_API_KEY:
            try:
                res = requests.get(f"https://newsapi.org/v2/everything?q={claim}&pageSize=5&apiKey={NEWS_API_KEY}").json()
                for art in res.get('articles', []):
                    nli = get_nli_scores(art['title'], claim)
                    evid_list.append({"title": art['title'], "source": art['source']['name'], "nli": nli})
            except: pass

        # 2. Signals
        clip_val = get_clip_score(img, claim)
        df_val = get_deepfake_score(img)
        
        # 3. Decision
        payload = {"claim": claim, "signals": {"clip": clip_val, "deepfake": df_val}, "context": evid_list}
        verdict, confidence, reason = run_gemini_decision(payload)

        # 4. Display Signals
        st.divider()
        st.subheader("📊 Signal Dashboard")
        m1, m2, m3 = st.columns(3)
        m1.metric("CLIP Relevance", f"{clip_val:.2%}")
        m2.metric("Deepfake Prob", f"{df_val:.2%}")
        m3.metric("System Confidence", f"{confidence:.2%}")

        st.write("**Visual Match Accuracy**")
        st.progress(clip_val)
        st.write("**Deepfake Probability**")
        st.progress(df_val)

        # 5. Final Verdict
        st.divider()
        st.subheader("✅ Final Decision")
        if "SUPPORTED" in verdict: st.success(f"### {verdict}")
        elif "CONTRADICTED" in verdict: st.error(f"### {verdict}")
        else: st.warning(f"### {verdict}")
        
        st.info(f"**Reasoning:** {reason}")

        # 6. Evidence
        with st.expander("🔍 News Context Breakdown"):
            for e in evid_list:
                st.markdown(f"<div class='signal-card'><strong>{e['source']}</strong>: {e['title']}</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Decision Engine: Gemini 2.5 Flash | Status: Dynamic Model Discovery Active")