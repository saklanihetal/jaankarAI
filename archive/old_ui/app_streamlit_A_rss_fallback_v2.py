# app_streamlit_A_rss_fallback_v2.py
# A-mode: SUPPORTED / UNVERIFIED / CONTRADICTED
# Fix: universal underspecified-claim guard (replica/location qualifiers) to prevent false SUPPORT.
# Retrieval: APIs when available + Google News RSS fallback.
# Keeps "Consensus context found in evidence" (non-negotiable).
# Adds image explanation when image provided.

import re
import unicodedata
from typing import Dict, List, Tuple
import hashlib

import streamlit as st
import requests
import torch
from PIL import Image
from langdetect import detect
import feedparser

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    CLIPProcessor,
    CLIPModel,
)
from sentence_transformers import SentenceTransformer, util

# ---------------- PAGE ----------------
st.set_page_config(page_title="Fake News Detection", layout="wide")
st.title("Fake News Detection")

# ---------------- CONSTANTS (NO SLIDERS) ----------------
TOP_K_NLI = 5
MAX_ITEMS_FETCH = 80

MIN_REL_FOR_CON = 0.35
R0, R1 = 0.50, 0.70

SUPPORT_TH = 0.55
CONTRADICT_TH = 0.75

# Guard strengths (fixed defaults)
QUALIFIER_PENALTY = 0.70  # how much to down-weight support if evidence adds replica/model/etc.
LOC_GUARD_PENALTY = 0.40  # down-weight support if evidence repeats a location not in claim (landmark-type)

# ---------------- SECRETS ----------------
def get_secret(name: str) -> str:
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""

NEWSAPI_KEY = get_secret("NEWSAPI_KEY")
GNEWS_KEY = get_secret("GNEWS_KEY")
NEWSDATA_KEY = get_secret("NEWSDATA_KEY")
EVENTREGISTRY_KEY = get_secret("EVENTREGISTRY_KEY")

# ---------------- TEXT UTILS ----------------
SUPPORTED_LANGS = {"en", "hi", "kn", "te"}

def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s)
    return s

def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return lang if lang in SUPPORTED_LANGS else "en"
    except Exception:
        return "en"

def keyword_set_en(text: str) -> set:
    text = normalize_text(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return {t for t in text.split() if len(t) > 2}

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def rel_gate(relevance: float) -> float:
    if relevance <= R0:
        return 0.0
    if relevance >= R1:
        return 1.0
    return clamp01((relevance - R0) / (R1 - R0))

def make_cache_key(text_en: str) -> str:
    norm = normalize_text(text_en).lower()
    return hashlib.md5(norm.encode("utf-8")).hexdigest()

# ---------------- MODELS ----------------
LANG_MAP = {"hi": "hin_Deva", "kn": "kan_Knda", "te": "tel_Telu"}

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_name)

    emb = SentenceTransformer("sentence-transformers/LaBSE")

    trans_name = "facebook/nllb-200-distilled-600M"
    nllb_tok = AutoTokenizer.from_pretrained(trans_name)
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained(trans_name).to(device)

    nli_name = "facebook/bart-large-mnli"
    nli_tok = AutoTokenizer.from_pretrained(nli_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_name).to(device)

    return device, clip_model, clip_processor, emb, nllb_tok, nllb_model, nli_tok, nli_model

device, clip_model, clip_processor, emb_model, nllb_tok, nllb_model, nli_tok, nli_model = load_models()

# ---------------- TRANSLATION ----------------
@torch.inference_mode()
def translate_to_english(text: str, lang: str) -> str:
    text = normalize_text(text)
    if lang == "en":
        return text
    if lang not in LANG_MAP:
        return text
    inputs = nllb_tok(text, return_tensors="pt").to(device)
    eng_id = nllb_tok.convert_tokens_to_ids("eng_Latn")
    out = nllb_model.generate(**inputs, forced_bos_token_id=eng_id, max_length=96)
    return nllb_tok.decode(out[0], skip_special_tokens=True)

# ---------------- CLIP GATE ----------------
@torch.inference_mode()
def clip_gate_score(image: Image.Image, text_en: str, a: float = 0.20, b: float = 0.35) -> Tuple[float, float]:
    inputs = clip_processor(text=[text_en], images=image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    img = outputs.image_embeds
    txt = outputs.text_embeds
    img = img / img.norm(dim=-1, keepdim=True)
    txt = txt / txt.norm(dim=-1, keepdim=True)
    s = (img * txt).sum(dim=-1).item()
    g = (s - a) / (b - a) if b != a else 0.0
    return float(s), clamp01(float(g))

# ---------------- NLI ----------------
@torch.inference_mode()
def nli_probs(premise: str, hypothesis: str) -> Tuple[float, float, float]:
    inputs = nli_tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256).to(device)
    logits = nli_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
    p_con, p_neu, p_ent = probs[0], probs[1], probs[2]
    return float(p_ent), float(p_con), float(p_neu)

# ---------------- FETCHERS ----------------
def safe_get_json(url: str, timeout: int = 12) -> Tuple[dict, str]:
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return {}, f"HTTP {r.status_code}: {r.text[:140]}"
        return r.json(), ""
    except Exception as e:
        return {}, str(e)

def safe_post_json(url: str, payload: dict, timeout: int = 14) -> Tuple[dict, str]:
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        if r.status_code != 200:
            return {}, f"HTTP {r.status_code}: {r.text[:140]}"
        return r.json(), ""
    except Exception as e:
        return {}, str(e)

def fetch_newsapi_org(query: str) -> Tuple[List[Dict], str]:
    if not NEWSAPI_KEY:
        return [], "missing key"
    url = (
        "https://newsapi.org/v2/everything?"
        f"q={requests.utils.quote(query)}&language=en&pageSize=25&sortBy=publishedAt&apiKey={NEWSAPI_KEY}"
    )
    res, err = safe_get_json(url)
    arts = [{
        "title": a.get("title",""),
        "description": a.get("description","") or "",
        "url": a.get("url",""),
        "source": "NewsAPI.org",
        "lang": "en"
    } for a in res.get("articles", []) if a.get("title") and a.get("url")]
    return arts, err

def fetch_gnews(query: str) -> Tuple[List[Dict], str]:
    if not GNEWS_KEY:
        return [], "missing key"
    url = f"https://gnews.io/api/v4/search?q={requests.utils.quote(query)}&lang=en&max=25&token={GNEWS_KEY}"
    res, err = safe_get_json(url)
    arts = [{
        "title": a.get("title",""),
        "description": a.get("description","") or "",
        "url": a.get("url",""),
        "source": "GNews",
        "lang": "en"
    } for a in res.get("articles", []) if a.get("title") and a.get("url")]
    return arts, err

def fetch_newsdata(query: str) -> Tuple[List[Dict], str]:
    if not NEWSDATA_KEY:
        return [], "missing key"
    url = f"https://newsdata.io/api/1/news?q={requests.utils.quote(query)}&language=en&apikey={NEWSDATA_KEY}"
    res, err = safe_get_json(url)
    arts = [{
        "title": a.get("title",""),
        "description": a.get("description","") or "",
        "url": a.get("link",""),
        "source": "NewsData.io",
        "lang": "en"
    } for a in res.get("results", []) if a.get("title") and a.get("link")]
    return arts, err

def fetch_eventregistry(query: str) -> Tuple[List[Dict], str]:
    if not EVENTREGISTRY_KEY:
        return [], "missing key"
    url = "https://eventregistry.org/api/v1/article/getArticles"
    payload = {
        "action": "getArticles",
        "keyword": query,
        "lang": "eng",
        "articlesCount": 25,
        "apiKey": EVENTREGISTRY_KEY
    }
    res, err = safe_post_json(url, payload)
    arts = [{
        "title": a.get("title",""),
        "description": (a.get("body") or a.get("summary") or "")[:500],
        "url": a.get("url",""),
        "source": "EventRegistry",
        "lang": "en"
    } for a in res.get("articles", {}).get("results", []) if a.get("title") and a.get("url")]
    return arts, err

def fetch_google_news_rss(query: str) -> Tuple[List[Dict], str]:
    url = "https://news.google.com/rss/search"
    params = {"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"}
    try:
        r = requests.get(url, params=params, timeout=12, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code != 200:
            return [], f"HTTP {r.status_code}"
        feed = feedparser.parse(r.text)
        arts = []
        for e in feed.entries[:30]:
            title = getattr(e, "title", "") or ""
            link = getattr(e, "link", "") or ""
            summary = getattr(e, "summary", "") or ""
            if title and link:
                arts.append({
                    "title": title,
                    "description": re.sub("<.*?>", " ", summary)[:500],
                    "url": link,
                    "source": "GoogleNewsRSS",
                    "lang": "en"
                })
        return arts, ""
    except Exception as e:
        return [], str(e)

def dedup_articles(articles: List[Dict]) -> List[Dict]:
    seen, out = set(), []
    for a in articles:
        t = normalize_text(a.get("title","")).lower()
        u = (a.get("url","") or "").strip()
        if not t or not u:
            continue
        k = (t, u)
        if k in seen:
            continue
        seen.add(k)
        out.append(a)
    return out

# ---------------- QUERIES ----------------
def build_queries(claim_en: str) -> List[str]:
    claim_en = normalize_text(claim_en)
    toks = list(keyword_set_en(claim_en))
    toks = toks[:7]
    qs = []
    if len(toks) >= 5:
        qs.append(" ".join(toks[:5]) + " India")
    if len(toks) >= 3:
        qs.append(" ".join(toks[:3]) + " India")
    qs.append(claim_en + " India")
    out, seen = [], set()
    for q in qs:
        qn = q.lower().strip()
        if qn and qn not in seen:
            out.append(q); seen.add(qn)
    return out[:3]

def fetch_evidence(claim_en: str) -> Tuple[List[Dict], Dict[str, str]]:
    queries = build_queries(claim_en)
    status = {}
    all_items: List[Dict] = []

    for q in queries:
        a, e = fetch_newsapi_org(q); status["NewsAPI.org"] = "OK" if not e else e; all_items += a
        a, e = fetch_gnews(q);       status["GNews"] = "OK" if not e else e;       all_items += a
        a, e = fetch_newsdata(q);    status["NewsData.io"] = "OK" if not e else e; all_items += a
        a, e = fetch_eventregistry(q); status["EventRegistry"] = "OK" if not e else e; all_items += a

    if len(all_items) == 0:
        rss_items = []
        for q in queries:
            a, e = fetch_google_news_rss(q); status["GoogleNewsRSS"] = "OK" if not e else e; rss_items += a
        all_items = rss_items

    all_items = dedup_articles(all_items)
    return all_items[:MAX_ITEMS_FETCH], status

# ---------------- RANK + NLI ----------------
def rank_by_relevance(items: List[Dict], claim_en: str) -> List[Dict]:
    claim_emb = emb_model.encode(claim_en, convert_to_tensor=True, normalize_embeddings=True)
    blobs = [(normalize_text(x["title"]) + ". " + normalize_text(x.get("description",""))).strip() for x in items]
    ev_emb = emb_model.encode(blobs, convert_to_tensor=True, normalize_embeddings=True)
    rels = util.cos_sim(claim_emb, ev_emb).squeeze(0).tolist()

    ranked = []
    for it, rel, blob in zip(items, rels, blobs):
        ranked.append({**it, "relevance": float(rel), "blob": blob})
    ranked.sort(key=lambda x: x["relevance"], reverse=True)
    return ranked

def aggregate_nli(ranked: List[Dict], claim_en: str, top_k: int = TOP_K_NLI):
    top = ranked[:top_k]
    S_ent = 0.0
    S_con = 0.0
    scored = []

    for it in top:
        premise = it["blob"]
        ent, con, neu = nli_probs(premise, claim_en)
        weighted_con = con * (1.0 - ent)
        g_rel = rel_gate(it["relevance"]) if it["relevance"] >= MIN_REL_FOR_CON else 0.0
        con_used = weighted_con * g_rel

        S_ent = max(S_ent, ent)
        S_con = max(S_con, con_used)

        if ent >= con and ent >= neu:
            label = "ENTAILS"
        elif con >= neu:
            label = "CONTRADICTS"
        else:
            label = "NEUTRAL"

        scored.append({**it,
                       "ent": ent, "con": con, "neu": neu,
                       "con_used": con_used, "g_rel": g_rel,
                       "nli_label": label})

    return S_ent, S_con, scored

# ---------------- CONSENSUS (NON-NEGOTIABLE) ----------------
def consensus_context(top_items: List[Dict]) -> Dict[str, List[str]]:
    from collections import Counter

    STOP = {"the","and","for","with","from","into","over","near","after","before","will","has","have","had","was","were","are","is","be",
            "in","on","at","to","of","a","an","as","by","it","this","that","these","those","live","today","latest","news"}

    loc_words = []
    ctx_words = []

    GEO = {
        "india","karnataka","bengaluru","bangalore","mysuru","mysore","delhi","mumbai","chennai","hyderabad",
        "telangana","andhra","kerala","tamil","nadu","kolkata","punjab","gujarat","rajasthan","bihar","odisha","assam","goa",
        "brazil","rio","sao","paulo","new","york","jersey","london","paris","dubai"
    }

    for it in top_items:
        text = normalize_text(it.get("blob","")).lower()
        toks = re.sub(r"[^a-z0-9\s]", " ", text).split()
        toks = [t for t in toks if len(t) > 3 and t not in STOP]
        ctx_words += toks
        for t in toks:
            if t in GEO:
                loc_words.append(t)

    def top_repeated(words, k):
        c = Counter(words)
        items = [(w,n) for w,n in c.items() if n >= 2]
        items.sort(key=lambda x: x[1], reverse=True)
        return [w for w,_ in items[:k]]

    return {
        "locations": top_repeated(loc_words, 8),
        "context_words": top_repeated(ctx_words, 12),
    }

# ---------------- UNIVERSAL UNDERSPECIFIED GUARD ----------------
QUALIFIER_TOKENS = {
    "replica","lookalike","reproduction","model","imitation","copy","miniature","mock","theme","park","toy","fake","statue-replica"
}

LANDMARK_TERMS = {
    "statue of liberty","taj mahal","eiffel tower","colosseum","big ben","white house","buckingham palace","golden gate bridge"
}

def guard_multiplier(claim_en: str, top_items: List[Dict]) -> float:
    """
    Returns multiplier in [0,1] to down-weight SUPPORT when evidence adds qualifiers/locations
    missing from the claim (classic 'replica in brazil' problem).
    No hard-coded explanation text — just decision logic.
    """
    claim_l = normalize_text(claim_en).lower()
    claim_tokens = keyword_set_en(claim_en)

    ev_text = " ".join([normalize_text(it.get("blob","")) for it in top_items[:5]]).lower()
    ev_tokens = keyword_set_en(ev_text)

    mult = 1.0

    # Qualifier guard (replica/model/etc.) — if evidence has it, claim doesn't
    if (ev_tokens & QUALIFIER_TOKENS) and not (claim_tokens & QUALIFIER_TOKENS):
        mult *= (1.0 - QUALIFIER_PENALALTY) if False else (1.0 - QUALIFIER_PENALTY)  # keep stable

    # Landmark + location guard: if claim is a landmark claim and evidence repeats a location not in claim
    is_landmark_claim = any(t in claim_l for t in LANDMARK_TERMS)
    if is_landmark_claim:
        # small location list for the specific failure mode; consensus block will show these anyway
        LOC = {"brazil","rio","sao","paulo","new york","jersey","london","paris","india"}
        ev_has_loc = any(loc in ev_text for loc in LOC)
        claim_has_loc = any(loc in claim_l for loc in LOC)
        if ev_has_loc and not claim_has_loc:
            mult *= (1.0 - LOC_GUARD_PENALTY)

    return clamp01(mult)

def decide_label(S_ent: float, S_con: float, support_mult: float) -> str:
    S_ent_adj = S_ent * support_mult

    if S_con >= CONTRADICT_TH and (S_con - S_ent_adj) >= 0.10:
        return "CONTRADICTED"
    if S_ent_adj >= SUPPORT_TH and (S_ent_adj - S_con) >= 0.05:
        return "SUPPORTED"
    return "UNVERIFIED"

# ---------------- UI ----------------
st.sidebar.header("Input")
uploaded_image = st.sidebar.file_uploader("Upload image (optional)", type=["jpg", "jpeg", "png"])
news_text = st.sidebar.text_area("Enter headline / claim (any language)")

image = None
if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

if news_text and news_text.strip():
    st.subheader("Processing")

    raw = normalize_text(news_text)
    lang = detect_language(raw)
    claim_en = translate_to_english(raw, lang)

    st.write(f"Detected language: {lang}")
    st.write(f"Claim (English): {claim_en}")

    if image is not None:
        _, g_clip = clip_gate_score(image, claim_en)
        st.write(f"Image-text alignment (CLIP): {g_clip:.3f}")
        st.subheader("Image explanation")
        if g_clip >= 0.65:
            st.write("The image appears consistent with the claim topic.")
        elif g_clip <= 0.35:
            st.write("The image appears weakly related to the claim topic (possible misuse or mismatch).")
        else:
            st.write("The image appears somewhat related to the claim topic.")
    else:
        g_clip = 0.0
        st.write("No image uploaded: image-text alignment skipped.")

    st.subheader("Source status")
    with st.spinner("Fetching evidence..."):
        items, status = fetch_evidence(claim_en)
    st.json(status)

    if not items:
        st.warning("No evidence found from APIs or RSS.")
        st.stop()

    st.write(f"Fetched {len(items)} unique items (before ranking).")

    with st.spinner("Ranking by relevance..."):
        ranked = rank_by_relevance(items, claim_en)

    with st.spinner("Running NLI on top evidence..."):
        S_ent, S_con, scored = aggregate_nli(ranked, claim_en)

    # Apply universal guard (no hard-coded explanation lines)
    support_mult = guard_multiplier(claim_en, ranked[:10])
    decision = decide_label(S_ent, S_con, support_mult)
    S_ent_adj = S_ent * support_mult

    st.subheader("Result")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Support", f"{S_ent:.3f}")
    c2.metric("Support (guarded)", f"{S_ent_adj:.3f}")
    c3.metric("Contradiction", f"{S_con:.3f}")
    c4.metric("Image-text", f"{g_clip:.3f}")
    c5.metric("Decision", decision)

    st.subheader("Explanation")
    if decision == "SUPPORTED":
        st.write("- One or more highly relevant items support the claim.")
    elif decision == "CONTRADICTED":
        st.write("- One or more highly relevant items contradict the claim.")
    else:
        st.write("- No item strongly supports the claim, so it is treated as unverified.")

    # ---------------- CONSENSUS (NON-NEGOTIABLE) ----------------
    st.subheader("Consensus context found in evidence")
    cons = consensus_context(ranked[:10])
    st.write("Locations mentioned by multiple sources: " + (", ".join(cons["locations"]) if cons["locations"] else "(none detected)"))
    st.write("Other repeated context words: " + (", ".join(cons["context_words"]) if cons["context_words"] else "(none detected)"))

    st.subheader("Evidence")
    for it in scored[:TOP_K_NLI]:
        st.markdown(
            f"""
**{it['title']}**  
{(it.get('description') or '').strip()}  
Source: {it['source']} | Relevance: {it['relevance']:.3f}  
NLI: {it['nli_label']} (ent={it['ent']:.3f}, con={it['con']:.3f}, neu={it['neu']:.3f})  
[Read article]({it['url']})
---
"""
        )

st.divider()
st.caption("Decision-support system only. Results are based on retrieved headlines/descriptions; verify with full articles when needed.")

