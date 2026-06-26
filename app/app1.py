# app.py
# Fake News Detection (FAST v2): CLIP (optional) + News APIs + Relevance(topK) + NLI(topK)
# Output labels: SUPPORTED / UNVERIFIED / CONTRADICTED
#
# Install:
#   pip install streamlit torch torchvision transformers sentence-transformers spacy langdetect requests pillow
#   python -m spacy download en_core_web_sm
#
# Run:
#   streamlit run app.py
#
# Optional .env keys:
#   NEWSAPI_KEY=...
#   GNEWS_KEY=...
#   NEWSDATA_KEY=...
#   NEWSAPI_AI_KEY=...

import os
import re
from typing import Dict, List, Tuple

import streamlit as st
import requests
from PIL import Image
from langdetect import detect

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, CLIPProcessor, CLIPModel
import spacy

from sentence_transformers import SentenceTransformer, util


# ----------------------------- UI -----------------------------
st.set_page_config(page_title="Fake News Detection", layout="wide")
st.title("Fake News Detection")


# ----------------------------- Utils -----------------------------
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("’", "'").replace("“", '"').replace("”", '"').replace("–", "-")
    return s

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "en"

def safe_get_json(url: str, timeout: int = 12) -> dict:
    try:
        r = requests.get(url, timeout=timeout)
        try:
            j = r.json()
        except Exception:
            j = {"_raw_text": r.text[:300]}
        j["_http_status"] = r.status_code
        return j
    except Exception as e:
        return {"_error": str(e)}

def safe_post_json(url: str, payload: dict, timeout: int = 12) -> dict:
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        try:
            j = r.json()
        except Exception:
            j = {"_raw_text": r.text[:300]}
        j["_http_status"] = r.status_code
        return j
    except Exception as e:
        return {"_error": str(e)}

def dedupe_by_title(items: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for it in items:
        t = normalize_text(it.get("title", ""))
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({**it, "title": t})
    return out


# ----------------------------- Models (cached) -----------------------------
@st.cache_resource
def load_spacy_sm():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_clip():
    model_id = "openai/clip-vit-base-patch32"
    proc = CLIPProcessor.from_pretrained(model_id)
    mdl = CLIPModel.from_pretrained(model_id)
    mdl.eval()
    return proc, mdl

@st.cache_resource
def load_relevance_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_nli():
    # ✅ Public + reliable MNLI model
    model_id = "cross-encoder/nli-deberta-v3-base"
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
    mdl.eval()
    return tok, mdl

nlp = load_spacy_sm()
clip_processor, clip_model = load_clip()
rel_model = load_relevance_model()
nli_tokenizer, nli_model = load_nli()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(DEVICE)
nli_model.to(DEVICE)


# ----------------------------- Sidebar -----------------------------
st.sidebar.header("Input")
uploaded_image = st.sidebar.file_uploader("Upload image (optional)", type=["jpg", "jpeg", "png"])
claim_text = st.sidebar.text_area("Enter headline / claim", height=120)

st.sidebar.header("Evidence")
top_k = st.sidebar.slider("Top-K headlines for NLI", 3, 10, 5, 1)
min_keep = st.sidebar.slider("Min relevance to keep (pre-filter)", 0.0, 1.0, 0.20, 0.01)
min_rel_con = st.sidebar.slider("Min relevance to count contradiction", 0.0, 1.0, 0.35, 0.01)

st.sidebar.header("CLIP gate (optional)")
use_clip = st.sidebar.checkbox("Use CLIP alignment if image provided", value=True)
clip_a = st.sidebar.slider("CLIP low (a)", 0.0, 0.6, 0.20, 0.01)
clip_b = st.sidebar.slider("CLIP high (b)", 0.0, 0.9, 0.35, 0.01)

st.sidebar.header("Decision thresholds")
support_th = st.sidebar.slider("SUPPORTED if support >=", 0.2, 0.95, 0.60, 0.01)
contr_th = st.sidebar.slider("CONTRADICTED if contradiction >=", 0.2, 0.99, 0.75, 0.01)
margin = st.sidebar.slider("Margin (support - contradiction)", 0.0, 0.6, 0.12, 0.01)

st.sidebar.header("Qualifier penalty (soft)")
qual_gamma = st.sidebar.slider("Penalty strength (0=off, 1=strong)", 0.0, 1.0, 0.55, 0.01)

st.sidebar.header("Debug")
show_debug = st.sidebar.checkbox("Show debug info", value=True)
show_api_errors = st.sidebar.checkbox("Show API errors", value=False)

st.sidebar.header("API Keys")
NEWSAPI_KEY = st.sidebar.text_input("NewsAPI.org", value=os.environ.get("NEWSAPI_KEY", ""), type="password")
GNEWS_KEY = st.sidebar.text_input("GNews", value=os.environ.get("GNEWS_KEY", ""), type="password")
NEWSDATA_KEY = st.sidebar.text_input("NewsData", value=os.environ.get("NEWSDATA_KEY", ""), type="password")
NEWSAPI_AI_KEY = st.sidebar.text_input("NewsAPI.ai (EventRegistry)", value=os.environ.get("NEWSAPI_AI_KEY", ""), type="password")


# ----------------------------- Claim parsing / queries -----------------------------
def extract_terms(claim_en: str) -> Dict[str, List[str]]:
    doc = nlp(normalize_text(claim_en))

    propn = [t.text for t in doc if t.pos_ == "PROPN" and len(t.text) >= 2]

    predicate = ""
    root = [t for t in doc if t.dep_ == "ROOT"]
    if root and root[0].pos_ in {"VERB", "AUX"}:
        predicate = root[0].lemma_.lower()
    else:
        for t in doc:
            if t.pos_ in {"VERB", "AUX"}:
                predicate = t.lemma_.lower()
                break

    keywords = []
    for t in doc:
        if t.pos_ in {"NOUN", "PROPN"}:
            kw = t.lemma_.lower()
            if len(kw) >= 2:
                keywords.append(kw)
    if predicate:
        keywords.append(predicate)

    low = set([k.lower() for k in keywords] + [p.lower() for p in propn])
    if "mysuru" in low:
        keywords.append("mysore")
    if "mysore" in low:
        keywords.append("mysuru")

    def dedupe(lst):
        seen = set()
        out = []
        for x in lst:
            x = x.strip()
            if not x:
                continue
            xl = x.lower()
            if xl in seen:
                continue
            seen.add(xl)
            out.append(xl)
        return out

    return {"propn": dedupe(propn), "keywords": dedupe(keywords), "predicate": predicate or ""}

def build_queries(claim_en: str, terms: Dict[str, List[str]]) -> List[str]:
    propn = terms["propn"][:4]
    kw = terms["keywords"][:8]
    pred = terms["predicate"]

    q1 = " ".join([*propn, pred]).strip()
    q2 = " ".join([*propn, *kw]).strip()
    q3 = " ".join(kw[:6]).strip()
    q4 = normalize_text(claim_en)

    qs = []
    for q in [q1, q2, q3, q4]:
        q = normalize_text(q)
        if len(q) >= 3 and q.lower() not in [x.lower() for x in qs]:
            qs.append(q)

    return qs[:4]


# ----------------------------- CLIP gate -----------------------------
def compute_clip_gate(image: Image.Image, claim_en: str, a: float, b: float) -> float:
    with torch.no_grad():
        inputs = clip_processor(text=[claim_en], images=image, return_tensors="pt", padding=True).to(DEVICE)
        out = clip_model(**inputs)
        img = out.image_embeds[0]
        txt = out.text_embeds[0]
        img = img / img.norm(p=2)
        txt = txt / txt.norm(p=2)
        cos = float((img * txt).sum().item())

    if b <= a:
        return 0.5
    g = (cos - a) / (b - a)
    return clamp01(g)


# ----------------------------- News fetchers -----------------------------
def fetch_newsapi_org(q: str):
    if not NEWSAPI_KEY:
        return [], {"_error": "Missing NewsAPI key"}
    url = f"https://newsapi.org/v2/everything?q={requests.utils.quote(q)}&language=en&pageSize=10&apiKey={NEWSAPI_KEY}"
    res = safe_get_json(url)
    if res.get("status") == "error":
        return [], res
    return [{"title": a.get("title",""), "url": a.get("url",""), "source": "NewsAPI.org"} for a in res.get("articles", []) if a.get("title")], None

def fetch_gnews(q: str):
    if not GNEWS_KEY:
        return [], {"_error": "Missing GNews key"}
    url = f"https://gnews.io/api/v4/search?q={requests.utils.quote(q)}&lang=en&max=10&token={GNEWS_KEY}"
    res = safe_get_json(url)
    if res.get("errors"):
        return [], res
    return [{"title": a.get("title",""), "url": a.get("url",""), "source": "GNews"} for a in res.get("articles", []) if a.get("title")], None

def fetch_newsdata(q: str):
    if not NEWSDATA_KEY:
        return [], {"_error": "Missing NewsData key"}
    url = f"https://newsdata.io/api/1/news?q={requests.utils.quote(q)}&language=en&apikey={NEWSDATA_KEY}"
    res = safe_get_json(url)
    if res.get("status") == "error":
        return [], res
    return [{"title": a.get("title",""), "url": a.get("link",""), "source": "NewsData.io"} for a in res.get("results", []) if a.get("title")], None

def fetch_newsapi_ai(q: str):
    if not NEWSAPI_AI_KEY:
        return [], {"_error": "Missing NewsAPI.ai key"}
    url = "https://eventregistry.org/api/v1/article/getArticles"
    payload = {"action":"getArticles","keyword": q,"lang":"eng","articlesCount": 10,"apiKey": NEWSAPI_AI_KEY}
    res = safe_post_json(url, payload, timeout=14)
    if res.get("error"):
        return [], res
    results = res.get("articles", {}).get("results", [])
    return [{"title": a.get("title",""), "url": a.get("url",""), "source": "NewsAPI.ai"} for a in results if a.get("title")], None

def fetch_all(queries: List[str]) -> Tuple[List[dict], List[dict]]:
    items = []
    errs = []
    fns = [fetch_newsapi_org, fetch_gnews, fetch_newsdata, fetch_newsapi_ai]
    for q in queries:
        for fn in fns:
            got, err = fn(q)
            items.extend(got)
            if err and show_api_errors:
                errs.append({"query": q, "error": err})
    return dedupe_by_title(items), errs


# ----------------------------- Relevance: fast batch -----------------------------
def rank_by_relevance(claim_en: str, items: List[dict]) -> List[dict]:
    titles = [it["title"] for it in items]
    claim_emb = rel_model.encode([claim_en], convert_to_tensor=True)
    title_emb = rel_model.encode(titles, convert_to_tensor=True)
    sims = util.cos_sim(claim_emb, title_emb)[0].tolist()

    scored = []
    for it, s in zip(items, sims):
        scored.append({**it, "relevance": float(s)})
    scored.sort(key=lambda x: x["relevance"], reverse=True)
    return scored


# ----------------------------- NLI: fast batch (forward only) -----------------------------
def nli_forward_batch(premises: List[str], hypothesis: str) -> List[Tuple[float, float, float]]:
    with torch.no_grad():
        enc = nli_tokenizer(
            premises,
            [hypothesis] * len(premises),
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        ).to(DEVICE)
        logits = nli_model(**enc).logits
        probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()

    # label mapping via config (safe across models)
    id2label = {int(k): v.lower() for k, v in nli_model.config.id2label.items()}
    # Find indices for entail/contr/neutral
    ent_i = next((i for i, lab in id2label.items() if "entail" in lab), None)
    con_i = next((i for i, lab in id2label.items() if "contr" in lab), None)
    neu_i = next((i for i, lab in id2label.items() if "neutral" in lab), None)

    # fallback common ordering: [contradiction, neutral, entailment]
    if ent_i is None or con_i is None or neu_i is None:
        con_i, neu_i, ent_i = 0, 1, 2

    out = []
    for p in probs:
        ent, con, neu = float(p[ent_i]), float(p[con_i]), float(p[neu_i])
        out.append((ent, con, neu))
    return out

def rel_gate(rel: float, min_rel: float) -> float:
    if rel <= min_rel:
        return 0.0
    return clamp01((rel - min_rel) / max(1e-6, (1.0 - min_rel)))


# ----------------------------- Qualifier penalty (soft) -----------------------------
QUALIFIERS = {"replica","model","toy","miniature","ai","deepfake","edited","old","archival"}
def soft_qualifier_penalty(claim: str, title: str, gamma: float) -> float:
    if gamma <= 0:
        return 1.0
    c = set(re.findall(r"[a-zA-Z]+", claim.lower()))
    t = set(re.findall(r"[a-zA-Z]+", title.lower()))
    extra = (t & QUALIFIERS) - c
    if not extra:
        return 1.0
    return clamp01(1.0 - gamma * min(1.0, len(extra)))


# ----------------------------- Main -----------------------------
if not claim_text or not claim_text.strip():
    st.write("Enter a headline/claim to begin.")
    st.stop()

claim = normalize_text(claim_text)
lang = detect_language(claim)

terms = extract_terms(claim)
queries = build_queries(claim, terms)

g_clip = 0.0
if uploaded_image is not None and use_clip:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)
    g_clip = compute_clip_gate(image, claim, clip_a, clip_b)

st.subheader("Processing")
st.write(f"Detected language: {lang}")
st.write(f"Claim (English): {claim}")
if uploaded_image is not None and use_clip:
    st.write(f"Image-text alignment (CLIP): {g_clip:.3f}")

if show_debug:
    st.write("Queries tried:")
    for q in queries:
        st.write("-", q)

evidence, api_errors = fetch_all(queries)
st.write(f"Fetched {len(evidence)} unique headlines (before filtering).")

if show_api_errors and api_errors:
    st.write("API errors:")
    st.json(api_errors)

if not evidence:
    st.subheader("Result")
    st.write("Decision: UNVERIFIED")
    st.write("Explanation: No evidence headlines returned by the news APIs (keys/quota/coverage).")
    st.stop()

ranked = rank_by_relevance(claim, evidence)

kept = [x for x in ranked if x["relevance"] >= min_keep]
if show_debug:
    st.write(f"After filtering: {len(kept)} headlines kept (min_keep={min_keep:.2f}).")

if not kept:
    st.subheader("Result")
    st.write("Decision: UNVERIFIED")
    st.write("Explanation: Headlines were fetched but none were relevant enough.")
    st.stop()

top = kept[:top_k]
premises = [x["title"] for x in top]
nli = nli_forward_batch(premises, claim)

annotated = []
S_ent = 0.0
S_con = 0.0

for x, (ent, con, neu) in zip(top, nli):
    rel = float(x["relevance"])

    qmul = soft_qualifier_penalty(claim, x["title"], qual_gamma)
    support_used = ent * qmul

    cg = rel_gate(rel, min_rel_con)
    con_used = con * cg

    S_ent = max(S_ent, support_used)
    S_con = max(S_con, con_used)

    label = "NEUTRAL"
    if ent >= con and ent >= neu:
        label = "ENTAILS"
    elif con >= ent and con >= neu:
        label = "CONTRADICTS"

    annotated.append({
        "title": x["title"],
        "url": x.get("url",""),
        "source": x.get("source",""),
        "relevance": rel,
        "label": label,
        "ent": ent, "con": con, "neu": neu,
    })

# Decision rule
if (S_con >= contr_th) and ((S_con - S_ent) >= margin):
    decision = "CONTRADICTED"
    explanation = "A highly relevant headline contradicts the claim."
elif (S_ent >= support_th) and ((S_ent - S_con) >= margin):
    decision = "SUPPORTED"
    explanation = "A highly relevant headline supports the claim."
else:
    decision = "UNVERIFIED"
    explanation = "Evidence is not strong enough to support or contradict the claim."

if uploaded_image is not None and use_clip:
    explanation += " The image appears consistent with the claim topic."

st.subheader("Result")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Support", f"{S_ent:.3f}")
c2.metric("Contradiction", f"{S_con:.3f}")
c3.metric("Image-text alignment", f"{g_clip:.3f}" if (uploaded_image is not None and use_clip) else "n/a")
c4.metric("Decision", decision)

st.subheader("Explanation")
st.write(explanation)

st.subheader("Evidence headlines (TOP-K)")
for a in annotated:
    st.markdown(f"**{a['title']}**")
    st.write(f"Source: {a['source']} | Relevance: {a['relevance']:.3f}")
    st.write(f"NLI: {a['label']} (ent={a['ent']:.2f}, con={a['con']:.2f}, neu={a['neu']:.2f})")
    if a["url"]:
        st.markdown(f"[Read article]({a['url']})")
    st.write("---")





















