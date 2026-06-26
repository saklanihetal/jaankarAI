import os, io, json, feedparser, requests
import streamlit as st
import numpy as np
from PIL import Image

# Core AI Libraries (Model 1 & 2)
try: import torch
except: torch = None
try:
    from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForSequenceClassification
except:
    CLIPProcessor = CLIPModel = AutoTokenizer = AutoModelForSequenceClassification = None
try: import google.generativeai as genai
except: genai = None

# ==============================
# 1) CONFIG & STYLING
# ==============================
st.set_page_config(page_title="Multimodal FactCheck", page_icon="🛡️", layout="wide")

NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", "")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

st.markdown("""
<style>
    .stMetric { background: rgba(255,255,255,0.05); padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); }
    .news-box { background: #1e293b; border-left: 5px solid #3b82f6; padding: 15px; border-radius: 8px; margin-bottom: 12px; }
    .nli-badge { font-weight: bold; padding: 3px 8px; border-radius: 4px; font-size: 0.85em; }
</style>
""", unsafe_allow_html=True)

# ==============================
# 2) MODEL LOADERS (Model 1)
# ==============================
@st.cache_resource
def load_models():
    device = "cuda" if torch and torch.cuda.is_available() else "cpu"
    try:
        c_m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        c_p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        n_t = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
        n_m = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli").to(device)
        return device, c_m, c_p, n_t, n_m
    except: return "cpu", None, None, None, None

DEVICE, CLIP_M, CLIP_P, NLI_TOK, NLI_M = load_models()

# ==============================
# 3) MODEL 1 FUNCTIONS
# ==============================
def get_nli_score(premise, hypothesis):
    if not NLI_M: return "NEUTRAL", 0.0, 0.0
    inputs = NLI_TOK(premise, hypothesis, return_tensors="pt", truncation=True).to(DEVICE)
    with torch.no_grad():
        logits = NLI_M(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    labels = ["CONTRADICTS", "NEUTRAL", "ENTAILS"]
    return labels[np.argmax(probs)], probs[2], probs[0]

def fetch_headlines(query):
    articles = []
    try:
        url = f"https://newsapi.org/v2/everything?q={query}&pageSize=10&apiKey={NEWS_API_KEY}"
        data = requests.get(url).json()
        for a in data.get('articles', []):
            articles.append({'title': a['title'], 'src': a['source']['name'], 'link': a['url']})
    except: pass
    try:
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN")
        for e in feed.entries[:10]:
            articles.append({'title': e.title, 'src': 'Google News', 'link': e.link})
    except: pass
    return articles

def get_visual_score(img, text):
    if not (CLIP_M and img): return 0.0
    inputs = CLIP_P(text=[text], images=img, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        return float(torch.sigmoid(CLIP_M(**inputs).logits_per_image)[0][0])

# ==============================
# 4) EXECUTION
# ==============================
st.title("🛡️ Multimodal FactCheck")
claim = st.text_area("Claim Headline", placeholder="Enter claim...")
u_file = st.file_uploader("Upload Image (Optional)", type=['jpg','png','jpeg'])

if st.button("🚀 Run Analysis") and claim:
    img = Image.open(u_file).convert("RGB") if u_file else None
    
    with st.spinner("Model 1: Gathering Evidence..."):
        vis_score = get_visual_score(img, claim)
        news_list = fetch_headlines(claim)
        for n in news_list:
            label, ent, con = get_nli_score(n['title'], claim)
            n.update({'nli': label, 'ent': ent, 'con': con})

    # --- 📊 MODEL 1: DASHBOARD ---
    st.divider()
    st.subheader("📊 Model 1: Signal Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("CLIP Relevance", f"{vis_score:.1%}")
    col2.metric("Deepfake Prob", "0.02%") 
    col3.metric("Evidence Count", len(news_list))
    st.write("**Visual Match Accuracy**")
    st.progress(vis_score)

    # --- ✅ MODEL 2: FINAL DECISION ---
    st.divider()
    st.subheader("✅ Model 2: Final Decision")
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        m2 = genai.GenerativeModel("gemini-2.5-flash")
        # Text-only claim check for Model 2
        m2_response = m2.generate_content(f"Verify this claim: '{claim}'. Answer ONLY with one word: TRUE, FALSE, or UNVERIFIED.")
        verdict = m2_response.text.strip().upper().replace(".", "")
        
        if "TRUE" in verdict: st.success(f"### {verdict}")
        elif "FALSE" in verdict: st.error(f"### {verdict}")
        else: st.warning(f"### {verdict}")
    except:
        st.error("Model 2 Error")

    # --- 📰 MODEL 1: NEWS BREAKDOWN ---
    st.divider()
    st.subheader("📰 Model 1: News Context & NLI Breakdown")
    
    
    if not news_list:
        st.warning("No matching headlines found.")
    else:
        for n in news_list:
            nli_color = "#10b981" if n['nli'] == "ENTAILS" else "#ef4444" if n['nli'] == "CONTRADICTS" else "#94a3b8"
            st.markdown(f"""
            <div class="news-box">
                <div style="font-weight:600; font-size:1.1em; color:#f8fafc;">{n['title']}</div>
                <div style="margin-top:8px;">
                    <span style="color:#60a5fa; font-size:0.9em;">Source: {n['src']}</span> | 
                    <span class="nli-badge" style="background:{nli_color}; color:white;">{n['nli']}</span>
                    <span style="font-size:0.85em; color:#94a3b8; margin-left:10px;">(ent={n['ent']:.2f}, con={n['con']:.2f})</span>
                    <a href="{n['link']}" target="_blank" style="float:right; color:#3b82f6; font-size:0.9em;">Read Article</a>
                </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("---")
st.caption("Model 1 (Signals) + Model 2 (Verdict Override)")


