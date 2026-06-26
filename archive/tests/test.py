# app.py
# Fake News Detection (MECIR v2): CLIP soft-gate + robust retrieval + anchor entity filter + NLI
# Immediate fix included: relevance-gated contradiction (soft fusion)
# Output labels: SUPPORTED / UNVERIFIED / CONTRADICTED
# UI: no emojis, no "None" spam, human-readable explanation.

import re
import unicodedata
from typing import Dict, List, Tuple

import streamlit as st
import torch
import requests
import spacy
from PIL import Image
from langdetect import detect

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    CLIPProcessor,
    CLIPModel
)
from sentence_transformers import SentenceTransformer, util

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Fake News Detection", layout="wide")
st.title("Fake News Detection")

# ---------------- spaCy LOAD ----------------
SPACY_MODEL = "en_core_web_sm"  # can switch later to "en_core_web_trf"
try:
    nlp = spacy.load(SPACY_MODEL)
except Exception:
    st.error(
        f"spaCy model '{SPACY_MODEL}' not found.\n\n"
        f"Run:\n  python -m spacy download {SPACY_MODEL}\n"
        f"Then restart Streamlit."
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

def keyword_set(text: str) -> set:
    text = normalize_text(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return set(t for t in text.split() if len(t) > 2)

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def rel_gate(relevance: float, r0: float, r1: float) -> float:
    """
    Soft gate for contradiction:
    - relevance <= r0  -> 0 (ignore contradiction)
    - relevance >= r1  -> 1 (fully count contradiction)
    linear in between
    """
    if r1 <= r0:
        return 1.0
    return clamp01((relevance - r0) / (r1 - r0))

# ---------------- CLAIM PARSING ----------------
def extract_claim_parts(text_en: str):
    """
    Returns: phrases, keywords, entities, predicate
    - phrases: multi-word chunks/entities
    - keywords: nouns/propns/verbs lemmas
    - entities: PERSON/ORG/GPE/LOC/EVENT strings
    - predicate: root verb lemma (if any)
    """
    text_en = normalize_text(text_en)
    doc = nlp(text_en)

    entities = []
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "GPE", "LOC", "EVENT"}:
            entities.append(ent.text.strip())

    predicate = ""
    for tok in doc:
        if tok.dep_ == "ROOT" and tok.pos_ in {"VERB", "AUX"}:
            predicate = tok.lemma_.lower()
            break

    propn_phrases = []
    buff = []
    for tok in doc:
        if tok.pos_ == "PROPN":
            buff.append(tok.text)
        else:
            if len(buff) >= 2:
                propn_phrases.append(" ".join(buff))
            buff = []
    if len(buff) >= 2:
        propn_phrases.append(" ".join(buff))

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
    propn_phrases = dedup(propn_phrases)
    noun_phrases = dedup(noun_phrases)
    keywords = dedup(keywords)
    phrases = dedup(propn_phrases + noun_phrases + entities)[:8]

    return phrases, keywords, entities, predicate

# ---------------- ANCHOR TOKENS (stops entity drift) ----------------
def anchor_tokens_from_claim(claim_en: str) -> set:
    doc = nlp(normalize_text(claim_en))
    anchors = set()
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "GPE", "LOC"}:
            anchors |= keyword_set(ent.text)
    return anchors

# ---------------- MULTI-QUERY BUILDER ----------------
def build_queries(claim_en: str) -> List[str]:
    claim_en = normalize_text(claim_en)
    phrases, keywords, entities, predicate = extract_claim_parts(claim_en)

    alias_pairs = [(r"\bmysuru\b", "mysore")]

    def apply_alias(q: str) -> List[str]:
        outs = [q]
        for pat, rep in alias_pairs:
            q2 = re.sub(pat, rep, q, flags=re.IGNORECASE)
            if q2.lower().strip() != q.lower().strip():
                outs.append(q2)
        return outs

    queries = []
    top_phrase = phrases[0] if phrases else ""
    top_kw = keywords[:10]

    # Force top entity phrase (helps Modi-Putin)
    if entities:
        if predicate:
            queries.append(f"\"{entities[0]}\" {predicate}".strip())
        queries.append(f"\"{entities[0]}\"".strip())

    if top_kw:
        queries.append(" ".join(top_kw[:6]))
    if top_phrase and predicate:
        queries.append(f"\"{top_phrase}\" {predicate}")
    if top_phrase:
        queries.append(f"\"{top_phrase}\"")
    if entities and predicate:
        queries.append(" ".join(entities[:2] + [predicate]))
    if entities:
        queries.append(" ".join(entities[:2]))
    if predicate and len(top_kw) >= 4:
        queries.append(f"{predicate} {top_kw[-2]} {top_kw[-1]}")
    if len(top_kw) >= 2:
        queries.append(f"{top_kw[0]} {top_kw[1]} {predicate}".strip())

    short_claim = " ".join(claim_en.split()[:10])
    if len(short_claim) >= 8:
        queries.append(short_claim)

    # Dedup + aliases
    out, seen = [], set()
    for q in queries:
        q = normalize_text(q)
        if len(q) < 4:
            continue
        for v in apply_alias(q):
            vn = v.lower().strip()
            if vn not in seen:
                out.append(v)
                seen.add(vn)

    return out[:10]

# ---------------- SMART FILTER (adaptive + synonyms) ----------------
def smart_hard_filter(evidence: List[Dict], claim_en: str) -> List[Dict]:
    claim_en = normalize_text(claim_en)
    phrases, keywords, entities, predicate = extract_claim_parts(claim_en)

    claim_tokens = set(k.lower() for k in keywords)
    for ph in phrases[:4]:
        claim_tokens |= keyword_set(ph)

    entity_tokens = set()
    for e in entities:
        entity_tokens |= keyword_set(e)

    # Adaptive overlap
    if len(claim_tokens) <= 6 or len(entity_tokens) >= 2:
        min_overlap = 1
    else:
        min_overlap = 2

    # Generic synonyms
    syn = {
        "wish": {"greet", "greets", "congratulate", "congratulates", "extends"},
        "birthday": {"bday", "anniversary"},
        "explode": {"blast", "explosion"},
        "topple": {"collapse", "collapsed", "falls", "fall", "tilts", "topples"},
        "collapse": {"topple", "topples", "falls", "fall", "crash", "caved"},
        "retire": {"retires", "retired", "quit", "quits", "resign", "resigns"},
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

# ---------------- QUALIFIER + UNDERSPECIFIED GUARDS ----------------
QUALIFIER_TOKENS = {
    "replica", "lookalike", "reproduction", "model", "imitation", "copy",
    "miniature", "mock", "store", "mall", "theme", "park"
}

LANDMARK_TERMS = {
    "statue of liberty", "eiffel tower", "taj mahal", "colosseum",
    "big ben", "golden gate bridge", "white house", "buckingham palace"
}

def qualifier_penalty(claim_en: str, top_titles: List[str]) -> float:
    claim_tokens = keyword_set(claim_en)
    ev_tokens = set()
    for t in top_titles:
        ev_tokens |= keyword_set(t)
    present = ev_tokens & QUALIFIER_TOKENS
    if not present:
        return 0.0
    return 1.0 if len(claim_tokens & present) == 0 else 0.0

def underspecified_guard(claim_en: str, top_titles: List[str]) -> bool:
    claim_l = normalize_text(claim_en).lower()
    if not any(t in claim_l for t in LANDMARK_TERMS):
        return False

    claim_tokens = keyword_set(claim_en)

    ev_tokens = set()
    ev_locs = set()
    for t in top_titles:
        t_norm = normalize_text(t)
        ev_tokens |= keyword_set(t_norm)
        doc = nlp(t_norm)
        for ent in doc.ents:
            if ent.label_ in {"GPE", "LOC"}:
                ev_locs.add(ent.text.lower())

    # qualifier missing
    ev_qual = ev_tokens & QUALIFIER_TOKENS
    if ev_qual and len(claim_tokens & ev_qual) == 0:
        return True

    # location missing
    if ev_locs:
        claim_has_loc = any(loc in claim_l for loc in ev_locs)
        if not claim_has_loc:
            return True

    return False

# ---------------- API KEYS ----------------
NEWSAPI_AI_KEY = ""
NEWSAPI_KEY = ""
GNEWS_KEY = ""
NEWSDATA_KEY = ""

# ---------------- NEWS FETCHERS ----------------
def safe_get_json(url: str, timeout: int = 12) -> Tuple[dict, str]:
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return {}, f"HTTP {r.status_code}"
        return r.json(), ""
    except Exception as e:
        return {}, str(e)

def safe_post_json(url: str, payload: dict, timeout: int = 14) -> Tuple[dict, str]:
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        if r.status_code != 200:
            return {}, f"HTTP {r.status_code}"
        return r.json(), ""
    except Exception as e:
        return {}, str(e)

def fetch_newsapi_org(query: str) -> Tuple[List[Dict], str]:
    url = (
        "https://newsapi.org/v2/everything?"
        f"q={requests.utils.quote(query)}&language=en&pageSize=25&sortBy=publishedAt&apiKey={NEWSAPI_KEY}"
    )
    res, err = safe_get_json(url)
    arts = [{"title": a["title"], "url": a["url"], "api": "NewsAPI.org"}
            for a in res.get("articles", []) if a.get("title") and a.get("url")]
    return arts, err

def fetch_gnews(query: str) -> Tuple[List[Dict], str]:
    url = f"https://gnews.io/api/v4/search?q={requests.utils.quote(query)}&lang=en&max=25&token={GNEWS_KEY}"
    res, err = safe_get_json(url)
    arts = [{"title": a["title"], "url": a["url"], "api": "GNews"}
            for a in res.get("articles", []) if a.get("title") and a.get("url")]
    return arts, err

def fetch_newsdata(query: str) -> Tuple[List[Dict], str]:
    url = f"https://newsdata.io/api/1/news?q={requests.utils.quote(query)}&language=en&apikey={NEWSDATA_KEY}"
    res, err = safe_get_json(url)
    arts = [{"title": a["title"], "url": a["link"], "api": "NewsData.io"}
            for a in res.get("results", []) if a.get("title") and a.get("link")]
    return arts, err

def fetch_newsapi_ai(query: str) -> Tuple[List[Dict], str]:
    url = "https://eventregistry.org/api/v1/article/getArticles"
    payload = {"action": "getArticles", "keyword": query, "lang": "eng", "articlesCount": 25, "apiKey": NEWSAPI_AI_KEY}
    res, err = safe_post_json(url, payload)
    arts = [{"title": a["title"], "url": a.get("url", ""), "api": "NewsAPI.ai"}
            for a in res.get("articles", {}).get("results", []) if a.get("title") and a.get("url")]
    return arts, err

def dedup_articles(articles: List[Dict]) -> List[Dict]:
    seen, out = set(), []
    for a in articles:
        t = normalize_text(a.get("title", "")).lower()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(a)
    return out

def fetch_all_sources(queries: List[str], max_total: int = 200) -> List[Dict]:
    all_arts: List[Dict] = []
    for q in queries:
        a, _ = fetch_newsapi_org(q); all_arts.extend(a)
        a, _ = fetch_gnews(q);       all_arts.extend(a)
        a, _ = fetch_newsdata(q);    all_arts.extend(a)
        a, _ = fetch_newsapi_ai(q);  all_arts.extend(a)

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
    if lang == "en":
        return text
    if lang not in LANG_MAP:
        return text
    inputs = nllb_tokenizer(text, return_tensors="pt").to(device)
    eng_token_id = nllb_tokenizer.convert_tokens_to_ids("eng_Latn")
    with torch.no_grad():
        translated = nllb_model.generate(**inputs, forced_bos_token_id=eng_token_id, max_length=96)
    return nllb_tokenizer.decode(translated[0], skip_special_tokens=True)

# ---------------- CLIP GATE ----------------
def clip_gate_score(image: Image.Image, text_en: str, a: float, b: float) -> Tuple[float, float]:
    inputs = clip_processor(text=[text_en], images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        img = outputs.image_embeds
        txt = outputs.text_embeds
        img = img / img.norm(dim=-1, keepdim=True)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        s = (img * txt).sum(dim=-1).item()
    g = (s - a) / (b - a) if b != a else 0.0
    g = clamp01(g)
    return s, g

# ---------------- NLI ----------------
def nli_probs(premise: str, hypothesis: str) -> Tuple[float, float, float]:
    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
    p_con, p_neu, p_ent = probs[0], probs[1], probs[2]
    return p_ent, p_con, p_neu

def rank_by_relevance(evidence: List[Dict], claim_en: str) -> List[Dict]:
    claim_emb = labse_model.encode(claim_en, convert_to_tensor=True)
    ranked = []
    for art in evidence:
        title = normalize_text(art["title"])
        t_emb = labse_model.encode(title, convert_to_tensor=True)
        rel = util.cos_sim(claim_emb, t_emb).item()
        ranked.append({**art, "relevance": rel})
    ranked.sort(key=lambda x: x["relevance"], reverse=True)
    return ranked

def aggregate_nli_topk_FIXED(
    evidence_ranked: List[Dict],
    claim_en: str,
    top_k: int,
    min_rel_for_con: float,
    rcon_block: float,
    r0: float,
    r1: float,
):
    """
    Support:
      - forward NLI: title -> claim
      - reverse NLI: claim -> title
      - support = forward entailment only if reverse contradiction is low, else 0
    Contradiction:
      - weighted_con = f_con * (1 - f_ent)
      - apply relevance gate: weighted_con_gated = weighted_con * rel_gate(relevance, r0, r1)
      - only consider if relevance >= min_rel_for_con
    """
    top = evidence_ranked[:top_k]

    S_ent, S_con = 0.0, 0.0
    scored = []

    for art in top:
        title = normalize_text(art["title"])

        f_ent, f_con, f_neu = nli_probs(title, claim_en)
        r_ent, r_con, r_neu = nli_probs(claim_en, title)

        support = f_ent if r_con < rcon_block else 0.0
        weighted_con = f_con * (1.0 - f_ent)

        # soft-gated contradiction
        if art["relevance"] >= min_rel_for_con:
            g_rel = rel_gate(art["relevance"], r0, r1)
            weighted_con_gated = weighted_con * g_rel
        else:
            g_rel = 0.0
            weighted_con_gated = 0.0

        S_ent = max(S_ent, support)
        S_con = max(S_con, weighted_con_gated)

        if f_ent >= f_con and f_ent >= f_neu:
            f_label = "ENTAILS"
        elif f_con >= f_neu:
            f_label = "CONTRADICTS"
        else:
            f_label = "NEUTRAL"

        scored.append({
            **art,
            "f_ent": f_ent, "f_con": f_con, "f_neu": f_neu,
            "r_ent": r_ent, "r_con": r_con, "r_neu": r_neu,
            "support": support,
            "weighted_con": weighted_con,
            "g_rel": g_rel,
            "weighted_con_gated": weighted_con_gated,
            "nli_label": f_label
        })

    scored.sort(key=lambda x: (x["support"], x["relevance"]), reverse=True)
    return S_ent, S_con, scored

# ---------------- SIDEBAR ----------------
st.sidebar.header("Input")
uploaded_image = st.sidebar.file_uploader("Upload image (optional)", type=["jpg", "jpeg", "png"])
news_text = st.sidebar.text_area("Enter headline / claim (any language)")

st.sidebar.header("Evidence selection")
TOP_K_EVIDENCE = st.sidebar.slider("Top-K evidence (NLI runs on these)", 3, 10, 5)
MIN_REL_FOR_CON = st.sidebar.slider("Min relevance to count contradiction", 0.0, 1.0, 0.35, 0.01)

st.sidebar.subheader("Relevance-gated contradiction (recommended)")
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

st.sidebar.header("Advanced")
gamma_qual = st.sidebar.slider("Qualifier penalty strength", 0.0, 1.0, 0.70, 0.05)

show_debug = st.sidebar.checkbox("Show debug", value=False)
show_evidence = st.sidebar.checkbox("Show evidence details", value=True)
headlines_to_show = st.sidebar.slider("Headlines to show", 3, 10, 5)

# ---------------- DISPLAY IMAGE ----------------
image = None
if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

# ---------------- MAIN ----------------
if news_text and len(news_text.strip()) > 0:
    st.subheader("Processing")

    T_raw = normalize_text(news_text)
    lang = detect_language(T_raw)
    T_en = translate_to_english(T_raw, lang)

    st.write(f"Detected language: {lang}")
    st.write(f"Claim (English): {T_en}")

    # CLIP gate
    if image is not None:
        _, g_clip = clip_gate_score(image, T_en, a=a, b=b)
        st.write(f"Image-text alignment (CLIP): {g_clip:.3f}")
    else:
        g_clip = 0.0
        st.write("No image uploaded: image-text alignment skipped.")

    # Queries + evidence
    queries = build_queries(T_en)
    with st.spinner("Fetching evidence headlines..."):
        evidence = fetch_all_sources(queries, max_total=250)

    if not evidence:
        st.warning("No evidence headlines returned by the news APIs.")
        st.stop()

    st.write(f"Fetched {len(evidence)} unique headlines (before filtering).")

    # Smart filter
    evidence_f = smart_hard_filter(evidence, T_en)

    # Anchor filter (prevents drift)
    anchors = anchor_tokens_from_claim(T_en)
    if anchors:
        anchored = [a_ for a_ in evidence_f if len(keyword_set(a_["title"]) & anchors) > 0]
        if len(anchored) >= max(5, TOP_K_EVIDENCE):
            evidence_f = anchored

    st.write(f"After filtering: {len(evidence_f)} headlines kept.")

    if len(evidence_f) == 0:
        st.warning("Filtering removed all headlines. Try a different wording.")
        st.stop()

    # Rank by relevance
    with st.spinner("Ranking by relevance..."):
        ranked = rank_by_relevance(evidence_f, T_en)

    # NLI top-k
    with st.spinner("Running NLI on top evidence..."):
        S_ent, S_con, scored_top = aggregate_nli_topk_FIXED(
            ranked, T_en,
            top_k=TOP_K_EVIDENCE,
            min_rel_for_con=MIN_REL_FOR_CON,
            rcon_block=rcon_block,
            r0=r0, r1=r1
        )

    top_titles = [a_["title"] for a_ in scored_top[:3]] if scored_top else []
    qual_pen = qualifier_penalty(T_en, top_titles)
    S_ent_adj = S_ent * (1.0 - gamma_qual * qual_pen)
    is_underspecified = underspecified_guard(T_en, top_titles)

    # Decision (simple, now stable due to relevance-gated contradiction)
    if is_underspecified and S_ent_adj >= support_th and S_con < contradict_th:
        decision = "UNVERIFIED"
    elif S_con >= contradict_th:
        decision = "CONTRADICTED"
    elif S_ent_adj >= support_th:
        decision = "SUPPORTED"
    else:
        decision = "UNVERIFIED"

    # ---------------- OUTPUT ----------------
    st.subheader("Result")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Support (after guard)", f"{S_ent_adj:.3f}")
    with c2:
        st.metric("Contradiction (relevance-gated)", f"{S_con:.3f}")
    with c3:
        st.metric("Image-text alignment", f"{g_clip:.3f}")
    with c4:
        st.metric("Decision", decision)

    st.subheader("Explanation")
    lines = []

    if decision == "SUPPORTED":
        lines.append("One or more highly relevant headlines support the claim.")
    elif decision == "CONTRADICTED":
        lines.append("One or more highly relevant headlines contradict the claim.")
    else:
        lines.append("No headline strongly supports the claim, so it is treated as unverified.")

    if is_underspecified:
        lines.append(
            "The claim is missing key context compared to the evidence (for example, location or qualifier like 'replica'), "
            "so it is marked unverified."
        )

    if image is not None:
        if g_clip >= 0.65:
            lines.append("The image content appears consistent with the claim topic.")
        elif g_clip <= 0.35:
            lines.append("The image content appears weakly related to the claim topic (possible misuse).")
        else:
            lines.append("The image content is somewhat related to the claim topic.")

    for line in lines:
        st.write("- " + line)

    if show_evidence and scored_top:
        st.subheader("Evidence headlines")
        for art in scored_top[:headlines_to_show]:
            st.markdown(
                f"""
**{art['title']}**  
Source: {art['api']} | Relevance: {art['relevance']:.3f}  
NLI: {art['nli_label']} (ent={art['f_ent']:.2f}, con={art['f_con']:.2f}, neu={art['f_neu']:.2f})  
Contradiction used: {art.get('weighted_con_gated',0.0):.2f} (rel_gate={art.get('g_rel',0.0):.2f})  
[Read article]({art['url']})
---
"""
            )

    if show_debug:
        st.subheader("Debug")
        phrases, keywords, entities, predicate = extract_claim_parts(T_en)
        if predicate:
            st.write(f"Predicate: {predicate}")
        if entities:
            st.write("Entities:", entities)
        if anchors:
            st.write("Anchor tokens:", sorted(list(anchors)))
        st.write("Queries:", queries)

st.divider()
st.caption("Decision-support system only. Results are based on headline-level evidence; verify with full articles when needed.")


