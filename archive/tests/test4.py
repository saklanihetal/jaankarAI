# app_streamlit_universal_v6.py
# Universal Fake News Detection (MECIR v2): minimal UI + auto decisions
#
# Key ideas (universal, not per-news-type):
# 1) Translate claim -> English (for retrieval + scoring)
# 2) Fetch evidence using English queries + native-language queries
# 3) Rank evidence by semantic relevance (LaBSE) using title+description blob
# 4) Run NLI (BART-MNLI) on TOP-K blobs vs multiple claim hypotheses
# 5) Aggregate with generic safeguards:
#    - relevance-gated contradiction (irrelevant headlines can't contradict)
#    - "update-like" contradiction downweight (temporal/number updates)
#    - "context mismatch" support downweight (evidence adds key context missing from claim)
# 6) Automatic thresholds (no tuning sliders in UI)
#
# NOTE: Put API keys in .streamlit/secrets.toml or environment variables.
# secrets names: NEWSAPI_KEY, GNEWS_KEY, NEWSDATA_KEY, EVENTREGISTRY_KEY

import os
import re
import unicodedata
from typing import Dict, List, Tuple, Optional
from collections import Counter

import streamlit as st
import torch
import requests
import spacy
from PIL import Image, ImageFile
from langdetect import detect

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    CLIPProcessor,
    CLIPModel,
)
from sentence_transformers import SentenceTransformer, util

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Fake News Detection", layout="wide")
st.title("Fake News Detection")

# ---------------- spaCy LOAD ----------------
DEFAULT_SPACY = "en_core_web_sm"
PREFERRED_TRF = "en_core_web_trf"

def load_spacy():
    for name in (PREFERRED_TRF, DEFAULT_SPACY):
        try:
            return spacy.load(name), name
        except Exception:
            pass
    return None, ""

nlp, spacy_name = load_spacy()
if nlp is None:
    st.error(
        "spaCy model not found.\n\nRun:\n"
        "  python -m spacy download en_core_web_sm\n\n"
        "Optional (better NER):\n"
        "  python -m spacy download en_core_web_trf\n"
    )
    st.stop()

SUPPORTED_LANGS = {"en", "hi", "kn", "te"}

# ---------------- TEXT UTILS ----------------
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

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def clean_desc(s: str) -> str:
    s = normalize_text(s or "")
    s = re.sub(r"\[\+\d+\s+chars\]$", "", s).strip()
    return s

def short(s: str, n: int = 260) -> str:
    s = normalize_text(s)
    return s if len(s) <= n else (s[: n - 3].rstrip() + "...")

def keyword_set_en(text: str) -> set:
    """English-only keyword set (for debug context terms)."""
    text = normalize_text(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return {t for t in text.split() if len(t) > 2}

def percentile(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    v = sorted(vals)
    if len(v) == 1:
        return v[0]
    k = (len(v) - 1) * p
    f = int(k)
    c = min(f + 1, len(v) - 1)
    if f == c:
        return v[f]
    return v[f] + (k - f) * (v[c] - v[f])

# ---------------- INDIA ENTITY LEXICON ----------------
INDIA_ENTITIES_SEED = [
    "Narendra Modi","PM Modi","Prime Minister Modi","Amit Shah","Rahul Gandhi",
    "Virat Kohli","Rohit Sharma","Sachin Tendulkar","MS Dhoni",
    "Mysuru Palace","Mysore Palace","Taj Mahal","India Gate","Red Fort",
    "Charminar","Gateway of India","Golden Temple","Qutub Minar",
    "Ayodhya","Ram Mandir",
    "Mysuru","Mysore","Bengaluru","Bangalore","Mumbai","Delhi","New Delhi",
    "Kolkata","Chennai","Hyderabad","Pune","Ahmedabad","Jaipur","Lucknow",
    "Patna","Bhopal","Indore","Nagpur","Surat","Kanpur","Guwahati",
    "Srinagar","Jammu","Kochi","Thiruvananthapuram",
    "Karnataka","Maharashtra","Tamil Nadu","Telangana","Kerala","Gujarat",
    "Rajasthan","Uttar Pradesh","Bihar","West Bengal","Punjab","Haryana",
    "Madhya Pradesh","Odisha","Assam","Jammu and Kashmir"
]

def load_entities_file(path: str) -> List[str]:
    try:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            lines = [normalize_text(x) for x in f.read().splitlines()]
        lines = [x for x in lines if x]
        seen = set()
        out = []
        for x in lines:
            k = x.lower()
            if k not in seen:
                out.append(x)
                seen.add(k)
        return out
    except Exception:
        return []

def build_lexicon() -> List[str]:
    lex = INDIA_ENTITIES_SEED[:]
    lex += load_entities_file(os.path.join("lexicon_out", "india_entities.txt"))
    seen = set()
    out = []
    for x in lex:
        k = x.lower().strip()
        if k and k not in seen:
            out.append(x.strip())
            seen.add(k)
    return out

LEXICON = build_lexicon()

# ---------------- CLAIM PARSING + QUERY BUILDER ----------------
INDIA_ALIASES = [
    (r"\bmysuru\b", "mysore"),
    (r"\bmysore\b", "mysuru"),
    (r"\bbengaluru\b", "bangalore"),
    (r"\bbangalore\b", "bengaluru"),
    (r"\bnew delhi\b", "delhi"),
    (r"\bdelhi\b", "new delhi"),
    (r"\bpm\b", "prime minister"),
]

def apply_aliases(text: str) -> List[str]:
    outs = [text]
    for pat, rep in INDIA_ALIASES:
        new_outs = []
        for t in outs:
            new_outs.append(re.sub(pat, rep, t, flags=re.IGNORECASE))
        outs = list(dict.fromkeys([normalize_text(x) for x in new_outs if normalize_text(x)]))
    return outs

def extract_claim_parts(text_en: str, lex: List[str]):
    """phrases, keywords, entities, predicate"""
    text_en = normalize_text(text_en)
    doc = nlp(text_en)

    entities = []
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "EVENT"}:
            entities.append(ent.text.strip())

    # lexicon match (helps India-only entities)
    t_low = " " + text_en.lower() + " "
    for name in lex:
        n = name.strip()
        if n and re.search(r"\b" + re.escape(n.lower()) + r"\b", t_low):
            entities.append(n)

    predicate = ""
    for tok in doc:
        if tok.dep_ == "ROOT" and tok.pos_ in {"VERB", "AUX"}:
            predicate = tok.lemma_.lower()
            break

    noun_phrases = []
    try:
        for chunk in doc.noun_chunks:
            if len(chunk.text.strip().split()) >= 2:
                noun_phrases.append(chunk.text.strip())
    except Exception:
        pass

    keywords = []
    for tok in doc:
        if tok.is_stop or tok.is_punct:
            continue
        if tok.pos_ in {"NOUN", "PROPN", "VERB"}:
            lemma = tok.lemma_.lower()
            if len(lemma) > 2:
                keywords.append(lemma)

    def dedup(seq):
        seen, out = set(), []
        for x in seq:
            k = x.lower().strip()
            if k and k not in seen:
                out.append(x.strip())
                seen.add(k)
        return out

    entities = dedup(entities)
    noun_phrases = dedup(noun_phrases)
    keywords = dedup(keywords)
    phrases = dedup(entities + noun_phrases)[:10]
    return phrases, keywords, entities, predicate

def build_queries_en(claim_en: str, lex: List[str]) -> List[str]:
    claim_en = normalize_text(claim_en)
    phrases, keywords, entities, predicate = extract_claim_parts(claim_en, lex)

    queries = []
    top_phrase = phrases[0] if phrases else ""
    top_kw = keywords[:12]

    if entities:
        if predicate:
            queries.append(f"\"{entities[0]}\" {predicate}".strip())
        queries.append(f"\"{entities[0]}\"".strip())

    if top_kw:
        queries.append(" ".join(top_kw[:7]))
    if top_phrase and predicate:
        queries.append(f"\"{top_phrase}\" {predicate}")
    if top_phrase:
        queries.append(f"\"{top_phrase}\"")

    if len(entities) >= 2:
        queries.append(" ".join(entities[:2] + ([predicate] if predicate else [])))
        queries.append(" ".join(entities[:2]))

    short_claim = " ".join(claim_en.split()[:12])
    if len(short_claim) >= 8:
        queries.append(short_claim)

    out, seen = [], set()
    for q in queries:
        for v in apply_aliases(normalize_text(q)):
            vn = v.lower().strip()
            if len(vn) < 4:
                continue
            if vn not in seen:
                out.append(v)
                seen.add(vn)
    return out[:10]

def build_queries_native(original_text: str) -> List[str]:
    t = normalize_text(original_text)
    if not t:
        return []
    q = [t, " ".join(t.split()[:8])]
    out, seen = [], set()
    for x in q:
        k = x.lower()
        if len(k) >= 4 and k not in seen:
            out.append(x)
            seen.add(k)
    return out[:2]

# ---------------- MULTI-HYPOTHESIS (UNIVERSAL) ----------------
PROMO_PATTERNS_EN = [
    r"\bwhere\b.*$",
    r"\bwhich place\b.*$",
    r"\bhow\b.*$",
    r"\bfree\b.*$",
    r"\bfor free\b.*$",
]

def clean_hypothesis_text(t: str) -> str:
    t = normalize_text(t)
    t = re.sub(r"[,\-:;]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def core_claim(text_en: str) -> str:
    t = normalize_text(text_en)
    t = t.split("?")[0].strip()
    t = t.split(";")[0].strip()
    low = t.lower()
    for pat in PROMO_PATTERNS_EN:
        low2 = re.sub(pat, "", low).strip()
        if len(low2) >= 6:
            low = low2
    return clean_hypothesis_text(low) if len(low.split()) >= 3 else clean_hypothesis_text(text_en)

def event_core_claim(text_en: str) -> str:
    """Removes fragile numeric/quantity details so event-level support can still be detected."""
    t = normalize_text(text_en).lower()
    t = t.split("?")[0].strip()
    t = t.split(";")[0].strip()

    t = re.sub(r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\b", " ", t)
    t = re.sub(r"\b\d+\b", " ", t)

    t = re.sub(r"\b(many|several|multiple|numerous|serious|seriously|severe|severely)\b", " ", t)
    t = re.sub(r"\b(killed|dead|deaths|death|injured|injuries)\b", " ", t)

    t = clean_hypothesis_text(t)
    return t if len(t.split()) >= 4 else clean_hypothesis_text(text_en)

def claim_hypotheses(claim_full_en: str) -> List[str]:
    hs = [
        clean_hypothesis_text(claim_full_en),
        clean_hypothesis_text(core_claim(claim_full_en)),
        clean_hypothesis_text(event_core_claim(claim_full_en)),
    ]
    out, seen = [], set()
    for h in hs:
        k = h.lower().strip()
        if k and k not in seen and len(k.split()) >= 3:
            out.append(h)
            seen.add(k)
    return out[:4]

# ---------------- GENERIC SAFEGUARDS (NO USER-FACING HARDCODED WORDING) ----------------
# A) Context mismatch: evidence contains "special qualifiers" or clear location detail the claim lacks.
# This is generic: model/toy/replica/miniature/etc (and also location mismatch).
CONTEXT_QUALIFIERS = {
    "replica", "model", "toy", "miniature", "lookalike", "imitation", "reproduction",
    "parody", "satire", "prank", "rumor", "alleged", "reportedly", "unconfirmed"
}

def extract_locations(text: str) -> set:
    doc = nlp(normalize_text(text))
    locs = set()
    for ent in doc.ents:
        if ent.label_ in {"GPE", "LOC"}:
            locs.add(ent.text.lower().strip())
    return locs

def context_mismatch_multiplier(claim_en: str, blob_en: str) -> float:
    """
    Returns multiplier in [0,1] to downweight support if evidence adds key missing context.
    This is intentionally generic and silent (no special-case explanation text).
    """
    c = normalize_text(claim_en).lower()
    b = normalize_text(blob_en).lower()

    c_tokens = keyword_set_en(c)
    b_tokens = keyword_set_en(b)

    # Qualifier appears in evidence but not in claim
    q = (b_tokens & CONTEXT_QUALIFIERS)
    if q and not (c_tokens & q):
        return 0.0

    # Location present in evidence but claim lacks it -> risky support
    c_locs = extract_locations(c)
    b_locs = extract_locations(b)
    if b_locs and not c_locs:
        return 0.0
    if b_locs and c_locs and not (b_locs & c_locs):
        return 0.2

    return 1.0

# B) Update-like contradiction downweight (generic temporal updates / counts)
def looks_like_update_style(claim_en: str, blob_any: str) -> bool:
    """
    Detect update-style phrasing across languages (EN/KN/TE/HI).
    We keep this as a generic safeguard: when evidence looks like a subsequent update,
    strict NLI contradictions are downweighted.
    """
    c = normalize_text(claim_en).lower()
    b = normalize_text(blob_any).lower()

    claim_has_number = bool(re.search(r"\b(one|two|three|four|five|\d+)\b", c))
    if not claim_has_number:
        return False

    # English "toll rises/reaches"
    en = bool(re.search(r"\b(toll|count|number)\b.*\b(rises|rise|reaches|hits|climbs|mounts|increases|increased)\b", b))

    # Kannada common update words
    kn = (("ಸಂಖ್ಯೆ" in b or "ಸಾವಿನ ಸಂಖ್ಯೆ" in b) and ("ಏರಿಕೆ" in b or "ಏರಿಕೆಯಾಗಿದೆ" in b or "ಏರಿತು" in b))

    # Telugu
    te = (("సంఖ్య" in b or "మృతుల సంఖ్య" in b) and ("పెరిగ" in b or "ఎక్కువ" in b or "పెరిగింది" in b))

    # Hindi
    hi = (("संख्या" in b) and ("बढ़" in b or "बढ़" in b or "पहुंच" in b))

    return en or kn or te or hi

# ---------------- API KEYS ----------------
def get_key(name: str) -> str:
    if name in st.secrets:
        return str(st.secrets[name])
    return os.getenv(name, "")

NEWSAPI_KEY = get_key("NEWSAPI_KEY")
GNEWS_KEY = get_key("GNEWS_KEY")
NEWSDATA_KEY = get_key("NEWSDATA_KEY")
EVENTREGISTRY_KEY = get_key("EVENTREGISTRY_KEY")

HEADERS = {"User-Agent": "FakeNewsUniversal/1.0"}

def safe_get_json(url: str, timeout: int = 20) -> dict:
    try:
        r = requests.get(url, timeout=timeout, headers=HEADERS)
        if r.status_code != 200:
            return {}
        return r.json()
    except Exception:
        return {}

def safe_post_json(url: str, payload: dict, timeout: int = 25) -> dict:
    try:
        r = requests.post(url, json=payload, timeout=timeout, headers=HEADERS)
        if r.status_code != 200:
            return {}
        return r.json()
    except Exception:
        return {}

GNEWS_LANG = {"en": "en", "hi": "hi", "kn": "kn", "te": "te"}
NEWSDATA_LANG = {"en": "en", "hi": "hi", "kn": "kn", "te": "te"}
EVENT_LANG = {"en": "eng", "hi": "hin", "kn": "kan", "te": "tel"}

def fetch_newsapi_org(query: str) -> List[Dict]:
    if not NEWSAPI_KEY:
        return []
    url = (
        "https://newsapi.org/v2/everything?"
        f"q={requests.utils.quote(query)}&language=en&pageSize=25&sortBy=publishedAt&apiKey={NEWSAPI_KEY}"
    )
    res = safe_get_json(url)
    out = []
    for a in (res.get("articles", []) or []):
        t = a.get("title") or ""
        u = a.get("url") or ""
        d = clean_desc(a.get("description") or a.get("content") or "")
        if t and u:
            out.append({"title": t, "description": d, "url": u, "api": "NewsAPI.org"})
    return out

def fetch_gnews(query: str, lang: str) -> List[Dict]:
    if not GNEWS_KEY:
        return []
    gl = GNEWS_LANG.get(lang, "en")
    url = f"https://gnews.io/api/v4/search?q={requests.utils.quote(query)}&lang={gl}&max=25&token={GNEWS_KEY}"
    res = safe_get_json(url)
    out = []
    for a in (res.get("articles", []) or []):
        t = a.get("title") or ""
        u = a.get("url") or ""
        d = clean_desc(a.get("description") or a.get("content") or "")
        if t and u:
            out.append({"title": t, "description": d, "url": u, "api": f"GNews({gl})"})
    return out

def fetch_newsdata(query: str, lang: str) -> List[Dict]:
    if not NEWSDATA_KEY:
        return []
    nl = NEWSDATA_LANG.get(lang, "en")
    url = f"https://newsdata.io/api/1/news?q={requests.utils.quote(query)}&language={nl}&apikey={NEWSDATA_KEY}"
    res = safe_get_json(url)
    out = []
    for a in (res.get("results", []) or []):
        t = a.get("title") or ""
        u = a.get("link") or ""
        d = clean_desc(a.get("description") or a.get("content") or "")
        if t and u:
            out.append({"title": t, "description": d, "url": u, "api": f"NewsData.io({nl})"})
    return out

def fetch_eventregistry(query: str, lang: str) -> List[Dict]:
    if not EVENTREGISTRY_KEY:
        return []
    el = EVENT_LANG.get(lang, "eng")
    url = "https://eventregistry.org/api/v1/article/getArticles"
    payload = {
        "action": "getArticles",
        "keyword": query,
        "lang": el,
        "articlesPage": 1,
        "articlesCount": 25,
        "articlesSortBy": "date",
        "articlesSortByAsc": False,
        "resultType": "articles",
        "apiKey": EVENTREGISTRY_KEY,
    }
    res = safe_post_json(url, payload)
    results = (((res.get("articles") or {}).get("results")) or [])
    out = []
    for a in results:
        t = a.get("title") or ""
        u = a.get("url") or ""
        d = clean_desc(a.get("summary") or a.get("body") or a.get("snippet") or "")
        if t and u:
            out.append({"title": t, "description": d, "url": u, "api": f"EventRegistry({el})"})
    return out

def dedup_articles(articles: List[Dict]) -> List[Dict]:
    seen, out = set(), []
    for a in articles:
        t = normalize_text(a.get("title", "")).lower()
        u = (a.get("url") or "").strip().lower()
        if not t:
            continue
        key = (t, u)
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out

def fetch_all_sources(queries_en: List[str], queries_native: List[str], lang: str, max_total: int = 220) -> List[Dict]:
    all_arts: List[Dict] = []

    for q in queries_en:
        all_arts.extend(fetch_newsapi_org(q))
        all_arts.extend(fetch_gnews(q, "en"))
        all_arts.extend(fetch_newsdata(q, "en"))
        all_arts.extend(fetch_eventregistry(q, "en"))

    if lang != "en":
        for q in queries_native:
            all_arts.extend(fetch_gnews(q, lang))
            all_arts.extend(fetch_newsdata(q, lang))
            all_arts.extend(fetch_eventregistry(q, lang))

    all_arts = dedup_articles(all_arts)
    return all_arts[:max_total]

# ---------------- MODELS ----------------
LANG_MAP = {"hi": "hin_Deva", "kn": "kan_Knda", "te": "tel_Telu"}

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(clip_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_name)

    labse_model = SentenceTransformer("sentence-transformers/LaBSE")

    translator_name = "facebook/nllb-200-distilled-600M"
    nllb_tokenizer = AutoTokenizer.from_pretrained(translator_name)
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained(translator_name).to(device)

    nli_name = "facebook/bart-large-mnli"
    nli_tokenizer = AutoTokenizer.from_pretrained(nli_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_name).to(device)

    return device, clip_model, clip_processor, labse_model, nllb_tokenizer, nllb_model, nli_tokenizer, nli_model

device, clip_model, clip_processor, labse_model, nllb_tokenizer, nllb_model, nli_tokenizer, nli_model = load_models()

def translate_to_english(text: str, lang: str) -> str:
    text = normalize_text(text)
    if not text:
        return ""
    if lang == "en":
        return text
    if lang not in LANG_MAP:
        return text
    nllb_tokenizer.src_lang = LANG_MAP[lang]
    inputs = nllb_tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    eng_token_id = nllb_tokenizer.convert_tokens_to_ids("eng_Latn")
    with torch.no_grad():
        translated = nllb_model.generate(
            **inputs,
            forced_bos_token_id=eng_token_id,
            max_length=128,
            num_beams=3,
        )
    return nllb_tokenizer.decode(translated[0], skip_special_tokens=True)

def clip_gate_score(image: Image.Image, text_en: str) -> float:
    # fixed calibration (no UI sliders)
    a, b = 0.20, 0.35
    if b <= a:
        b = a + 1e-6
    inputs = clip_processor(text=[text_en], images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        img = outputs.image_embeds
        txt = outputs.text_embeds
        img = img / img.norm(dim=-1, keepdim=True)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        s = (img * txt).sum(dim=-1).item()
    g = (s - a) / (b - a)
    return clamp01(g)

def nli_probs(premise: str, hypothesis: str) -> Tuple[float, float, float]:
    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
    p_con, p_neu, p_ent = probs[0], probs[1], probs[2]
    return p_ent, p_con, p_neu

def rank_by_relevance(evidence: List[Dict], claim_en: str) -> List[Dict]:
    blobs = []
    for a in evidence:
        t = normalize_text(a.get("title", ""))
        d = normalize_text(a.get("description", ""))
        blob = (t + ". " + d).strip() if d else t
        a["blob"] = blob
        blobs.append(blob)

    claim_emb = labse_model.encode([claim_en], convert_to_tensor=True, normalize_embeddings=True)
    ev_emb = labse_model.encode(blobs, convert_to_tensor=True, normalize_embeddings=True)
    rels = util.cos_sim(claim_emb, ev_emb).squeeze(0).tolist()

    ranked = []
    for art, rel in zip(evidence, rels):
        ranked.append({**art, "relevance": float(rel)})
    ranked.sort(key=lambda x: x["relevance"], reverse=True)
    return ranked

def common_context_terms(scored: List[Dict], top_n: int = 10) -> List[Tuple[str, int]]:
    counter = Counter()
    for art in scored[:5]:
        blob = normalize_text(art.get("blob", ""))
        for tok in keyword_set_en(blob):
            if tok in {"near","front","city","said","says","today","report","reports","news"}:
                continue
            counter[tok] += 1
    return counter.most_common(top_n)

# ---------------- SCORING + AUTO DECISION ----------------
def score_topk(ranked: List[Dict], claim_en: str, top_k: int) -> Tuple[List[Dict], float, float, Dict[str, float]]:
    """
    Universal fixed settings (no UI tuning).
    Returns scored evidence, S_ent, S_con, and diagnostic settings.
    """
    # universal settings (fixed)
    MIN_REL_FOR_CON = 0.35
    R0, R1 = 0.50, 0.70
    UPDATE_CON_DOWNWEIGHT = 0.15

    hyps = claim_hypotheses(claim_en)

    S_ent, S_con = 0.0, 0.0
    scored = []

    for art in ranked[:top_k]:
        blob = normalize_text(art.get("blob", "")) or normalize_text(art.get("title",""))
        if not blob:
            continue

        # best entailment across hypotheses
        best = {"ent": 0.0, "con": 0.0, "neu": 1.0, "hyp": hyps[0]}
        for h in hyps:
            fe, fc, fn = nli_probs(blob, h)
            if fe > best["ent"]:
                best = {"ent": fe, "con": fc, "neu": fn, "hyp": h}

        f_ent, f_con, f_neu, best_h = best["ent"], best["con"], best["neu"], best["hyp"]

        # support gets generic context-mismatch multiplier
        support = f_ent * context_mismatch_multiplier(claim_en, blob)

        # contradiction is relevance gated; update-like contradictions downweighted
        weighted_con = f_con * (1.0 - f_ent)
        if looks_like_update_style(claim_en, blob):
            weighted_con *= UPDATE_CON_DOWNWEIGHT

        rel = float(art.get("relevance", 0.0))
        if rel >= MIN_REL_FOR_CON:
            g_rel = clamp01((rel - R0) / (R1 - R0)) if (R1 > R0) else 1.0
            contr_used = weighted_con * g_rel
        else:
            g_rel = 0.0
            contr_used = 0.0

        S_ent = max(S_ent, support)
        S_con = max(S_con, contr_used)

        if f_ent >= f_con and f_ent >= f_neu:
            label = "ENTAILS"
        elif f_con >= f_neu:
            label = "CONTRADICTS"
        else:
            label = "NEUTRAL"

        scored.append({
            **art,
            "nli_label": label,
            "f_ent": f_ent,
            "f_con": f_con,
            "f_neu": f_neu,
            "best_hypothesis": best_h,
            "support": support,
            "contr_used": contr_used,
            "rel_gate": g_rel,
        })

    scored.sort(key=lambda x: (x["support"], x.get("relevance",0.0)), reverse=True)

    diag = {
        "min_rel_for_con": MIN_REL_FOR_CON,
        "r0": R0,
        "r1": R1,
        "update_con_downweight": UPDATE_CON_DOWNWEIGHT,
    }
    return scored, S_ent, S_con, diag

def auto_decide(scored: List[Dict], S_ent: float, S_con: float) -> Tuple[str, Dict[str, float]]:
    """
    Auto thresholds from distributions (universal; no per-claim tuning).
    """
    ent_list = [float(x["support"]) for x in scored] or [0.0]
    con_list = [float(x["contr_used"]) for x in scored] or [0.0]

    ent_p75 = percentile(ent_list, 0.75)
    con_p75 = percentile(con_list, 0.75)

    # universal floors
    support_th = max(0.22, ent_p75 + 0.06)
    contradict_th = max(0.70, con_p75 + 0.10)

    margin = 0.05
    if S_ent > 0.15 and S_con > 0.20:
        margin = 0.08

    if S_con >= contradict_th and S_con > (S_ent + margin):
        return "CONTRADICTED", {"support_th": support_th, "contradict_th": contradict_th, "margin": margin}
    if S_ent >= support_th and S_ent > (S_con + margin):
        return "SUPPORTED", {"support_th": support_th, "contradict_th": contradict_th, "margin": margin}
    return "UNVERIFIED", {"support_th": support_th, "contradict_th": contradict_th, "margin": margin}

# ---------------- UI (MINIMAL) ----------------
st.sidebar.header("Input")
uploaded_image = st.sidebar.file_uploader("Upload image (optional)", type=["jpg", "jpeg", "png"])
news_text = st.sidebar.text_area("Enter headline / claim (any language)")

TOP_K_EVIDENCE = st.sidebar.selectbox("Evidence depth (Top-K)", [3, 5, 7, 10], index=1)
show_debug = st.sidebar.checkbox("Show debug", value=False)

# ---------------- IMAGE DISPLAY ----------------
image: Optional[Image.Image] = None
if uploaded_image:
    try:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)
    except Exception:
        st.warning("Could not read the uploaded image.")
        image = None

# ---------------- MAIN ----------------
if news_text and len(news_text.strip()) > 0:
    st.subheader("Processing")

    T_raw = normalize_text(news_text)
    lang = detect_language(T_raw)
    T_en = translate_to_english(T_raw, lang)

    st.write(f"spaCy: {spacy_name}")
    st.write(f"Detected language: {lang}")
    st.write(f"Claim (English): {T_en}")

    # Image explanation (generic wording)
    if image is not None:
        g_clip = clip_gate_score(image, T_en)
        st.write(f"Image-text alignment (CLIP): {g_clip:.3f}")
        st.subheader("Image explanation")
        if g_clip >= 0.65:
            st.write("- The image appears consistent with the claim topic.")
        elif g_clip <= 0.35:
            st.write("- The image appears inconsistent or weakly related to the claim topic (possible misuse or out-of-context image).")
        else:
            st.write("- The image is somewhat related to the claim topic, but not strongly aligned.")
    else:
        g_clip = 0.0
        st.write("No image uploaded: image-text alignment skipped.")

    queries_en = build_queries_en(T_en, LEXICON)
    queries_native = build_queries_native(T_raw)

    with st.spinner("Fetching evidence (English + native queries)..."):
        evidence = fetch_all_sources(queries_en, queries_native, lang, max_total=220)

    if not evidence:
        st.warning("No evidence returned (keys/quota/language coverage).")
        st.stop()

    st.write(f"Fetched {len(evidence)} unique items (before ranking).")

    with st.spinner("Ranking by relevance..."):
        ranked = rank_by_relevance(evidence, T_en)

    with st.spinner("Running NLI on top evidence..."):
        scored, S_ent, S_con, diag = score_topk(ranked, T_en, top_k=TOP_K_EVIDENCE)

    decision, th = auto_decide(scored, S_ent, S_con)

    st.subheader("Result")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Support", f"{S_ent:.3f}")
    with c2:
        st.metric("Contradiction", f"{S_con:.3f}")
    with c3:
        st.metric("Image-text alignment", f"{g_clip:.3f}")
    with c4:
        st.metric("Decision", decision)

    st.subheader("Explanation")
    if decision == "SUPPORTED":
        st.write("- One or more highly relevant items support the claim.")
        st.write("- Support is based on semantic agreement between the claim and evidence summaries.")
    elif decision == "CONTRADICTED":
        st.write("- One or more highly relevant items contradict the claim.")
        st.write("- Contradiction is only counted when the evidence is highly relevant to the claim.")
    else:
        st.write("- No item strongly supports the claim, so it is treated as unverified.")
        st.write("- This can happen when coverage is limited or when available headlines do not clearly confirm the claim.")

    st.subheader("Evidence")
    for art in scored[:TOP_K_EVIDENCE]:
        title = normalize_text(art.get("title",""))
        desc = normalize_text(art.get("description",""))
        desc_line = short(desc, 260) if desc else "(No description provided by this source.)"
        st.markdown(
            f"""
**{title}**  
{desc_line}  
Source: {art.get("api","")} | Relevance: {art.get("relevance",0.0):.3f}  
NLI: {art.get("nli_label","")} (ent={art.get("f_ent",0.0):.3f}, con={art.get("f_con",0.0):.3f}, neu={art.get("f_neu",0.0):.3f})  
[Read article]({art.get("url","")})
---
"""
        )

    if show_debug:
        st.subheader("Debug")
        st.write("Entities loaded:", len(LEXICON))
        st.write("Auto thresholds:", th)
        st.write("Fixed safeguard settings:", diag)
        st.write("Hypotheses tested:", claim_hypotheses(T_en))
        st.write("English queries:", queries_en)
        st.write("Native queries:", queries_native)
        terms = common_context_terms(scored, top_n=12)
        if terms:
            st.write("Common context terms:", ", ".join([f"{w}({c})" for w, c in terms]))

st.divider()
st.caption("Decision-support system only. Evidence is headline+description; verify full articles when needed.")
