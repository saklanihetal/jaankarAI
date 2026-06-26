# test2.py
# Fake News Detection (MECIR v2): India-aware entities + robust retrieval + fast relevance + robust NLI
# Fixes in this iteration:
# - HuggingFace download timeouts increased (common on slow networks)
# - India entity lexicon fallback (Mysuru Palace, Taj Mahal, Virat Kohli, PM Modi, etc.)
# - Better India alias expansion (Mysuru/Mysore, Bengaluru/Bangalore...)
# - Landmark specificity guard still included (replica in Brazil => UNVERIFIED)

import os
import re
import unicodedata
from typing import Dict, List, Tuple, Optional

# ---------------- HUGGINGFACE NETWORK TIMEOUT FIX ----------------
# Must be set BEFORE transformers/sentence-transformers import/load
os.environ.setdefault("HF_HUB_READ_TIMEOUT", "120")      # default often ~10
os.environ.setdefault("HF_HUB_CONNECT_TIMEOUT", "60")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
# Optional speedup (requires: pip install hf_transfer)
# os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

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
    for name in [PREFERRED_TRF, DEFAULT_SPACY]:
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

# ---------------- INDIA ENTITY LEXICON (lightweight, editable) ----------------
# This is NOT exhaustive; it’s a starter set. You can add more via sidebar.
INDIA_ENTITIES_SEED = [
    # People
    "Narendra Modi", "PM Modi", "Prime Minister Modi", "Amit Shah", "Rahul Gandhi",
    "Virat Kohli", "Rohit Sharma", "Sachin Tendulkar", "MS Dhoni",

    # Places / landmarks
    "Mysuru Palace", "Mysore Palace", "Taj Mahal", "India Gate", "Red Fort",
    "Charminar", "Gateway of India", "Golden Temple", "Qutub Minar",
    "Kedarnath", "Badrinath", "Ayodhya", "Ram Mandir",

    # Cities
    "Mysuru", "Mysore", "Bengaluru", "Bangalore", "Mumbai", "Delhi", "New Delhi",
    "Kolkata", "Chennai", "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
    "Patna", "Bhopal", "Indore", "Nagpur", "Surat", "Kanpur", "Guwahati",
    "Srinagar", "Jammu", "Kochi", "Thiruvananthapuram",

    # States
    "Karnataka", "Maharashtra", "Tamil Nadu", "Telangana", "Kerala", "Gujarat",
    "Rajasthan", "Uttar Pradesh", "Bihar", "West Bengal", "Punjab", "Haryana",
    "Madhya Pradesh", "Odisha", "Assam", "Jammu and Kashmir"
]

INDIA_ALIASES = [
    # city aliases
    (r"\bmysuru\b", "mysore"),
    (r"\bmysore\b", "mysuru"),
    (r"\bbengaluru\b", "bangalore"),
    (r"\bbangalore\b", "bengaluru"),
    (r"\bnew delhi\b", "delhi"),
    (r"\bdelhi\b", "new delhi"),
    # common politician form
    (r"\bpm\b", "prime minister"),
]

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

def keyword_set(text: str) -> set:
    text = normalize_text(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return set(t for t in text.split() if len(t) > 2)

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def rel_gate(relevance: float, r0: float, r1: float) -> float:
    if r1 <= r0:
        return 1.0
    return clamp01((relevance - r0) / (r1 - r0))

def apply_india_aliases(text: str) -> List[str]:
    """Return variants of a query with India-specific aliases swapped."""
    outs = [text]
    for pat, rep in INDIA_ALIASES:
        new_outs = []
        for t in outs:
            t2 = re.sub(pat, rep, t, flags=re.IGNORECASE)
            new_outs.append(t2)
        outs = list(dict.fromkeys(new_outs))  # dedup preserving order
    outs = [normalize_text(x) for x in outs if normalize_text(x)]
    return list(dict.fromkeys(outs))

# ---------------- CLAIM PARSING ----------------
def extract_spacy_entities(text_en: str) -> List[str]:
    doc = nlp(normalize_text(text_en))
    ents = []
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "EVENT"}:
            ents.append(ent.text.strip())
    # dedup
    seen, out = set(), []
    for e in ents:
        k = e.lower()
        if k not in seen:
            out.append(e)
            seen.add(k)
    return out

def extract_india_entities_lexicon(text_en: str, india_list: List[str]) -> List[str]:
    """Regex match India entity phrases inside claim (works even if spaCy misses)."""
    t = " " + normalize_text(text_en).lower() + " "
    found = []
    for name in india_list:
        n = name.strip()
        if not n:
            continue
        pat = r"\b" + re.escape(n.lower()) + r"\b"
        if re.search(pat, t):
            found.append(n)
    # also catch "PM Modi" style if claim says "Modi"
    if re.search(r"\bmodi\b", t) and "Narendra Modi" not in found:
        found.append("Narendra Modi")
    return found

def extract_claim_parts(text_en: str, india_list: List[str]):
    """
    Returns: phrases, keywords, entities, predicate
    entities = spaCy entities + lexicon matches
    """
    text_en = normalize_text(text_en)
    doc = nlp(text_en)

    spacy_ents = extract_spacy_entities(text_en)
    lex_ents = extract_india_entities_lexicon(text_en, india_list)

    # merge
    entities = []
    seen = set()
    for e in spacy_ents + lex_ents:
        k = e.lower().strip()
        if k and k not in seen:
            entities.append(e.strip())
            seen.add(k)

    predicate = ""
    for tok in doc:
        if tok.dep_ == "ROOT" and tok.pos_ in {"VERB", "AUX"}:
            predicate = tok.lemma_.lower()
            break

    noun_phrases = []
    try:
        for chunk in doc.noun_chunks:
            t = chunk.text.strip()
            if len(t.split()) >= 2:
                noun_phrases.append(t)
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
        seen2, out2 = set(), []
        for x in seq:
            k2 = x.lower().strip()
            if k2 and k2 not in seen2:
                out2.append(x.strip())
                seen2.add(k2)
        return out2

    entities = dedup(entities)
    noun_phrases = dedup(noun_phrases)
    keywords = dedup(keywords)

    # phrases: prioritize India entities + noun chunks
    phrases = dedup(entities + noun_phrases)[:10]
    return phrases, keywords, entities, predicate

# ---------------- ANCHOR TOKENS ----------------
def anchor_tokens_from_claim(claim_en: str, india_list: List[str]) -> set:
    anchors = set()
    for ent in extract_spacy_entities(claim_en):
        anchors |= keyword_set(ent)
    for ent in extract_india_entities_lexicon(claim_en, india_list):
        anchors |= keyword_set(ent)
    return anchors

# ---------------- MULTI-QUERY BUILDER ----------------
def build_queries(claim_en: str, india_list: List[str]) -> List[str]:
    claim_en = normalize_text(claim_en)
    phrases, keywords, entities, predicate = extract_claim_parts(claim_en, india_list)

    queries = []
    top_phrase = phrases[0] if phrases else ""
    top_kw = keywords[:12]

    # Strong entity-first queries (India-aware)
    if entities:
        if predicate:
            queries.append(f"\"{entities[0]}\" {predicate}".strip())
        queries.append(f"\"{entities[0]}\"".strip())

    # Keywords query
    if top_kw:
        queries.append(" ".join(top_kw[:7]))

    # Phrase + predicate
    if top_phrase and predicate:
        queries.append(f"\"{top_phrase}\" {predicate}")
    if top_phrase:
        queries.append(f"\"{top_phrase}\"")

    # Entity pair
    if len(entities) >= 2:
        queries.append(" ".join(entities[:2] + ([predicate] if predicate else [])))

    # Full short claim fallback
    short_claim = " ".join(claim_en.split()[:12])
    if len(short_claim) >= 8:
        queries.append(short_claim)

    # Apply India alias swaps + dedup
    out = []
    seen = set()
    for q in queries:
        for v in apply_india_aliases(q):
            vn = v.lower().strip()
            if len(vn) < 4:
                continue
            if vn not in seen:
                out.append(v)
                seen.add(vn)

    return out[:10]

# ---------------- SMART FILTER ----------------
def smart_hard_filter(evidence: List[Dict], claim_en: str, india_list: List[str]) -> List[Dict]:
    phrases, keywords, entities, _ = extract_claim_parts(claim_en, india_list)

    claim_tokens = set(k.lower() for k in keywords)
    for ph in phrases[:5]:
        claim_tokens |= keyword_set(ph)

    entity_tokens = set()
    for e in entities:
        entity_tokens |= keyword_set(e)

    min_overlap = 1 if (len(claim_tokens) <= 6 or len(entity_tokens) >= 2) else 2

    syn = {
        "explode": {"blast", "explosion"},
        "explodes": {"explosion", "blast"},
        "exploded": {"explosion", "blast"},
        "explosion": {"blast", "explodes"},
        "topple": {"collapse", "collapsed", "falls", "fall", "topples"},
        "collapse": {"topple", "topples", "falls", "fall", "crash"},
        "retire": {"retires", "retired", "quit", "resign"},
    }

    def expand(tokens: set) -> set:
        expanded = set(tokens)
        for t in list(tokens):
            if t in syn:
                expanded |= syn[t]
            for k, vs in syn.items():
                if t in vs:
                    expanded.add(k)
                    expanded |= vs
        return expanded

    claim_tokens = expand(claim_tokens | entity_tokens)

    filtered = []
    for art in evidence:
        title = normalize_text(art.get("title", ""))
        if not title:
            continue
        tset = keyword_set(title)
        overlap = len(tset & claim_tokens)
        has_entity = len(tset & entity_tokens) > 0
        if overlap >= min_overlap or has_entity:
            filtered.append(art)

    return filtered

# ---------------- QUALIFIER + LANDMARK GUARDS ----------------
QUALIFIER_TOKENS = {
    "replica", "lookalike", "reproduction", "model", "imitation", "copy",
    "miniature", "mock", "theme", "park"
}

LANDMARK_TERMS = {
    "statue of liberty", "eiffel tower", "taj mahal", "colosseum",
    "big ben", "golden gate bridge", "white house", "buckingham palace"
}

def extract_locations(text: str) -> set:
    doc = nlp(normalize_text(text))
    locs = set()
    for ent in doc.ents:
        if ent.label_ in {"GPE", "LOC"}:
            locs.add(ent.text.lower().strip())
    return locs

def contains_landmark(text: str) -> Optional[str]:
    t = normalize_text(text).lower()
    for lm in LANDMARK_TERMS:
        if lm in t:
            return lm
    return None

def landmark_specificity_support_multiplier(claim_en: str, title: str) -> float:
    claim_l = normalize_text(claim_en).lower()
    title_l = normalize_text(title).lower()

    lm = contains_landmark(claim_l)
    if not lm:
        return 1.0

    claim_tokens = keyword_set(claim_l)
    title_tokens = keyword_set(title_l)

    ev_qual = title_tokens & QUALIFIER_TOKENS
    if ev_qual and len(claim_tokens & ev_qual) == 0:
        return 0.0  # block support if evidence says replica/model but claim doesn't

    claim_locs = extract_locations(claim_l)
    ev_locs = extract_locations(title_l)

    if ev_locs and not claim_locs:
        return 0.0  # block support when evidence has location and claim has none

    if ev_locs and claim_locs and len(ev_locs & claim_locs) == 0:
        return 0.2  # downweight if locations conflict

    return 1.0

# ---------------- CLAIM VARIANTS (ROBUST NLI) ----------------
def claim_variants_for_nli(claim_en: str) -> List[str]:
    orig = normalize_text(claim_en)
    c = orig.lower()

    variants = set([orig])

    c1 = c
    c1 = re.sub(r"\bexplodes\b", "explosion", c1)
    c1 = re.sub(r"\bexploded\b", "explosion", c1)
    c1 = re.sub(r"\bblast\b", "explosion", c1)
    c1 = re.sub(r"\btopples\b", "collapses", c1)
    c1 = re.sub(r"\btopple\b", "collapse", c1)
    variants.add(normalize_text(c1))

    variants.add(normalize_text(re.sub(r"\bnear\b", "at", c1)))

    out = [normalize_text(v) for v in variants if len(v.split()) >= 3]
    return out[:6]

# ---------------- API KEYS ----------------
def get_key(name: str) -> str:
    if name in st.secrets:
        return str(st.secrets[name])
    return os.getenv(name, "")

NEWSAPI_KEY = get_key("NEWSAPI_KEY")
GNEWS_KEY = get_key("GNEWS_KEY")
NEWSDATA_KEY = get_key("NEWSDATA_KEY")
EVENTREGISTRY_KEY = get_key("EVENTREGISTRY_KEY")

# ---------------- REQUEST HELPERS ----------------
HEADERS = {"User-Agent": "FakeNewsMECIRv2/1.0"}

def safe_get_json(url: str, timeout: int = 20) -> Tuple[dict, str]:
    try:
        r = requests.get(url, timeout=timeout, headers=HEADERS)
        if r.status_code != 200:
            return {}, f"HTTP {r.status_code}: {r.text[:200]}"
        return r.json(), ""
    except Exception as e:
        return {}, str(e)

def safe_post_json(url: str, payload: dict, timeout: int = 25) -> Tuple[dict, str]:
    try:
        r = requests.post(url, json=payload, timeout=timeout, headers=HEADERS)
        if r.status_code != 200:
            return {}, f"HTTP {r.status_code}: {r.text[:200]}"
        return r.json(), ""
    except Exception as e:
        return {}, str(e)

# ---------------- NEWS FETCHERS ----------------
def fetch_newsapi_org(query: str) -> List[Dict]:
    if not NEWSAPI_KEY:
        return []
    url = (
        "https://newsapi.org/v2/everything?"
        f"q={requests.utils.quote(query)}&language=en&pageSize=25&sortBy=publishedAt&apiKey={NEWSAPI_KEY}"
    )
    res, _ = safe_get_json(url)
    return [{"title": a["title"], "url": a["url"], "api": "NewsAPI.org"}
            for a in (res.get("articles", []) or []) if a.get("title") and a.get("url")]

def fetch_gnews(query: str) -> List[Dict]:
    if not GNEWS_KEY:
        return []
    url = f"https://gnews.io/api/v4/search?q={requests.utils.quote(query)}&lang=en&max=25&token={GNEWS_KEY}"
    res, _ = safe_get_json(url)
    return [{"title": a["title"], "url": a["url"], "api": "GNews"}
            for a in (res.get("articles", []) or []) if a.get("title") and a.get("url")]

def fetch_newsdata(query: str) -> List[Dict]:
    if not NEWSDATA_KEY:
        return []
    url = f"https://newsdata.io/api/1/news?q={requests.utils.quote(query)}&language=en&apikey={NEWSDATA_KEY}"
    res, _ = safe_get_json(url)
    return [{"title": a["title"], "url": a["link"], "api": "NewsData.io"}
            for a in (res.get("results", []) or []) if a.get("title") and a.get("link")]

def fetch_eventregistry(query: str) -> List[Dict]:
    if not EVENTREGISTRY_KEY:
        return []
    url = "https://eventregistry.org/api/v1/article/getArticles"
    payload = {
        "action": "getArticles",
        "keyword": query,
        "lang": "eng",
        "articlesPage": 1,
        "articlesCount": 25,
        "articlesSortBy": "date",
        "articlesSortByAsc": False,
        "resultType": "articles",
        "apiKey": EVENTREGISTRY_KEY,
    }
    res, _ = safe_post_json(url, payload)
    results = (((res.get("articles") or {}).get("results")) or [])
    return [{"title": a["title"], "url": a["url"], "api": "EventRegistry"}
            for a in results if a.get("title") and a.get("url")]

def dedup_articles(articles: List[Dict]) -> List[Dict]:
    seen, out = set(), []
    for a in articles:
        t = normalize_text(a.get("title", "")).lower()
        if not t:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(a)
    return out

def fetch_all_sources(queries: List[str], max_total: int = 200) -> List[Dict]:
    all_arts: List[Dict] = []
    for q in queries:
        all_arts.extend(fetch_newsapi_org(q))
        all_arts.extend(fetch_gnews(q))
        all_arts.extend(fetch_newsdata(q))
        all_arts.extend(fetch_eventregistry(q))
    all_arts = dedup_articles(all_arts)
    return all_arts[:max_total]

# ---------------- LOAD MODELS ----------------
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

# ---------------- TRANSLATION ----------------
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

# ---------------- CLIP GATE ----------------
def clip_gate_score(image: Image.Image, text_en: str, a: float, b: float) -> Tuple[float, float]:
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
    return s, clamp01(g)

# ---------------- NLI ----------------
def nli_probs(premise: str, hypothesis: str) -> Tuple[float, float, float]:
    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
    # BART MNLI order: contradiction, neutral, entailment
    p_con, p_neu, p_ent = probs[0], probs[1], probs[2]
    return p_ent, p_con, p_neu

# ---------------- RELEVANCE (BATCH) ----------------
def rank_by_relevance(evidence: List[Dict], claim_en: str) -> List[Dict]:
    titles = [normalize_text(a["title"]) for a in evidence]
    claim_emb = labse_model.encode([claim_en], convert_to_tensor=True, normalize_embeddings=True)
    title_emb = labse_model.encode(titles, convert_to_tensor=True, normalize_embeddings=True)
    rels = util.cos_sim(claim_emb, title_emb).squeeze(0).tolist()

    ranked = []
    for art, rel in zip(evidence, rels):
        ranked.append({**art, "relevance": float(rel)})
    ranked.sort(key=lambda x: x["relevance"], reverse=True)
    return ranked

# ---------------- AGGREGATION ----------------
def aggregate_nli_topk(
    evidence_ranked: List[Dict],
    claim_en: str,
    top_k: int,
    min_rel_for_con: float,
    rcon_block: float,
    r0: float,
    r1: float,
):
    top = evidence_ranked[:top_k]
    variants = claim_variants_for_nli(claim_en)

    S_ent, S_con = 0.0, 0.0
    scored = []

    for art in top:
        title = normalize_text(art["title"])

        # Forward: best entailment across variants
        best_f_ent, best_f_con, best_f_neu = 0.0, 0.0, 1.0
        best_h = claim_en
        for h in variants:
            fe, fc, fn = nli_probs(title, h)
            if fe > best_f_ent:
                best_f_ent, best_f_con, best_f_neu = fe, fc, fn
                best_h = h
        f_ent, f_con, f_neu = best_f_ent, best_f_con, best_f_neu

        # Reverse: block support if evidence adds specificity
        r_ent, r_con, r_neu = nli_probs(claim_en, title)

        support_raw = f_ent if r_con < rcon_block else 0.0

        # Landmark specificity guard (replica/location)
        lm_mult = landmark_specificity_support_multiplier(claim_en, title)
        support = support_raw * lm_mult

        weighted_con = f_con * (1.0 - f_ent)

        if art["relevance"] >= min_rel_for_con:
            g_rel = rel_gate(art["relevance"], r0, r1)
            weighted_con_gated = weighted_con * g_rel
        else:
            g_rel = 0.0
            weighted_con_gated = 0.0

        S_ent = max(S_ent, support)
        S_con = max(S_con, weighted_con_gated)

        # label for UI
        if f_ent >= f_con and f_ent >= f_neu:
            f_label = "ENTAILS"
        elif f_con >= f_neu:
            f_label = "CONTRADICTS"
        else:
            f_label = "NEUTRAL"

        scored.append({
            **art,
            "relevance": art["relevance"],
            "f_ent": f_ent, "f_con": f_con, "f_neu": f_neu,
            "r_con": r_con,
            "support_raw": support_raw,
            "support": support,
            "landmark_mult": lm_mult,
            "g_rel": g_rel,
            "weighted_con_gated": weighted_con_gated,
            "nli_label": f_label,
            "best_hypothesis": best_h,
        })

    scored.sort(key=lambda x: (x["support"], x["relevance"]), reverse=True)
    return S_ent, S_con, scored

# ---------------- SIDEBAR ----------------
st.sidebar.header("Input")
uploaded_image = st.sidebar.file_uploader("Upload image (optional)", type=["jpg", "jpeg", "png"])
news_text = st.sidebar.text_area("Enter headline / claim (any language)")

st.sidebar.header("India entity boost")
extra_india_entities = st.sidebar.text_area(
    "Add India entities (one per line) - optional",
    value="",
    help="Example:\nNirmala Sitharaman\nUdupi\nWankhede Stadium"
)

india_list = INDIA_ENTITIES_SEED[:]
for line in extra_india_entities.splitlines():
    line = line.strip()
    if line:
        india_list.append(line)
# dedup
seen = set()
india_list = [x for x in india_list if not (x.lower() in seen or seen.add(x.lower()))]

st.sidebar.header("Evidence selection")
TOP_K_EVIDENCE = st.sidebar.slider("Top-K evidence (NLI runs on these)", 3, 10, 5)
MIN_REL_FOR_CON = st.sidebar.slider("Min relevance to count contradiction", 0.0, 1.0, 0.35, 0.01)

st.sidebar.subheader("Relevance-gated contradiction")
r0 = st.sidebar.slider("Contradiction gate starts (r0)", 0.0, 1.0, 0.50, 0.01)
r1 = st.sidebar.slider("Contradiction gate full (r1)",   0.0, 1.0, 0.70, 0.01)

st.sidebar.header("CLIP settings")
a = st.sidebar.slider("CLIP low threshold (a)", 0.00, 0.60, 0.20, 0.01)
b = st.sidebar.slider("CLIP high threshold (b)", 0.00, 0.60, 0.35, 0.01)

st.sidebar.header("NLI support rule")
rcon_block = st.sidebar.slider("Block support if reverse-contradiction ≥", 0.0, 1.0, 0.70, 0.01)

st.sidebar.header("Decision thresholds")
support_th = st.sidebar.slider("SUPPORTED if support ≥", 0.3, 0.95, 0.55, 0.01)
contradict_th = st.sidebar.slider("CONTRADICTED if contradiction ≥", 0.3, 0.95, 0.75, 0.01)

show_debug = st.sidebar.checkbox("Show debug", value=False)
show_evidence = st.sidebar.checkbox("Show evidence details", value=True)
headlines_to_show = st.sidebar.slider("Headlines to show", 3, 10, 5)

# ---------------- DISPLAY IMAGE ----------------
image: Optional[Image.Image] = None
if uploaded_image:
    try:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)
    except Exception:
        st.warning("Could not read the uploaded image. Try a different file.")
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

    if image is not None:
        _, g_clip = clip_gate_score(image, T_en, a=a, b=b)
        st.write(f"Image-text alignment (CLIP): {g_clip:.3f}")
    else:
        g_clip = 0.0
        st.write("No image uploaded: image-text alignment skipped.")

    queries = build_queries(T_en, india_list)
    with st.spinner("Fetching evidence headlines..."):
        evidence = fetch_all_sources(queries, max_total=250)

    if not evidence:
        st.warning("No evidence headlines returned by the news APIs (keys, quota, or query too strict).")
        st.stop()

    st.write(f"Fetched {len(evidence)} unique headlines (before filtering).")

    evidence_f = smart_hard_filter(evidence, T_en, india_list)

    anchors = anchor_tokens_from_claim(T_en, india_list)
    if anchors:
        anchored = [a_ for a_ in evidence_f if len(keyword_set(a_["title"]) & anchors) > 0]
        if len(anchored) >= max(5, TOP_K_EVIDENCE):
            evidence_f = anchored

    st.write(f"After filtering: {len(evidence_f)} headlines kept.")
    if not evidence_f:
        st.warning("Filtering removed all headlines. Try rephrasing or adding India entities in sidebar.")
        st.stop()

    with st.spinner("Ranking by relevance..."):
        ranked = rank_by_relevance(evidence_f, T_en)

    with st.spinner("Running NLI on top evidence..."):
        S_ent, S_con, scored_top = aggregate_nli_topk(
            ranked, T_en,
            top_k=TOP_K_EVIDENCE,
            min_rel_for_con=MIN_REL_FOR_CON,
            rcon_block=rcon_block,
            r0=r0, r1=r1
        )

    if S_con >= contradict_th:
        decision = "CONTRADICTED"
    elif S_ent >= support_th:
        decision = "SUPPORTED"
    else:
        decision = "UNVERIFIED"

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
        st.write("- One or more highly relevant headlines support the claim.")
    elif decision == "CONTRADICTED":
        st.write("- One or more highly relevant headlines contradict the claim.")
    else:
        st.write("- No headline strongly supports the claim, so it is treated as unverified.")
    if contains_landmark(T_en):
        st.write("- Landmark guard is active for this claim (replica/location specificity handled).")

    if show_evidence and scored_top:
        st.subheader("Evidence headlines")
        for art in scored_top[:headlines_to_show]:
            st.markdown(
                f"""
**{art['title']}**  
Source: {art['api']} | Relevance: {art['relevance']:.3f}  
NLI: {art['nli_label']} (ent={art['f_ent']:.2f}, con={art['f_con']:.2f}, neu={art['f_neu']:.2f})  
Support used: {art.get('support',0.0):.2f} (landmark_mult={art.get('landmark_mult',1.0):.2f}, reverse_con={art.get('r_con',0.0):.2f})  
Contradiction used: {art.get('weighted_con_gated',0.0):.2f} (rel_gate={art.get('g_rel',0.0):.2f})  
[Read article]({art['url']})
---
"""
            )

    if show_debug:
        st.subheader("Debug")
        st.write("Queries:", queries)
        st.write("India entities matched (lexicon):", extract_india_entities_lexicon(T_en, india_list))
        st.write("spaCy entities:", extract_spacy_entities(T_en))
        st.write("Anchor tokens:", sorted(list(anchors))[:50])

st.divider()
st.caption("Decision-support system only. Headline-level evidence; verify with full articles when needed.")
