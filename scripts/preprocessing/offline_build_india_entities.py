#!/usr/bin/env python
"""
offline_build_india_entities.py

Goal:
- Read your multilingual Indian news dataset (en/hi/kn/te)
- Translate non-English text to English (NLLB) in batches
- Run spaCy NER on English text
- Produce:
    1) india_entities.txt              (one entity per line, sorted by freq desc)
    2) india_entities_with_counts.csv  (entity,count)
    3) translated_cache.csv            (text,lang,text_en)  [optional cache]

Designed to run locally on Windows (VS Code terminal) ONCE to build an India entity lexicon.

USAGE (examples):
  python offline_build_india_entities.py ^
    --input "fake_news_simplified_LANGFILTER_labels01.csv" ^
    --text-cols headline,title,text,claim,news ^
    --lang-col language ^
    --out-dir "lexicon_out" ^
    --min-count 2 ^
    --spacy-model en_core_web_trf

If your dataset already has a language column with values like: en/hi/kn/te, set --lang-col.
If not, omit --lang-col and we will detect script-based language.

Notes:
- Translation uses NLLB: facebook/nllb-200-distilled-600M
- spaCy NER uses en_core_web_trf if installed, else fallback to en_core_web_sm
- Batch size defaults conservative to avoid OOM on CPU.

"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from tqdm import tqdm

import spacy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ----------------------------- Defaults -----------------------------

DEFAULT_TRANSLATOR = "facebook/nllb-200-distilled-600M"
DEFAULT_SPACY_TRF = "en_core_web_trf"
DEFAULT_SPACY_SM = "en_core_web_sm"

# NLLB language tags for your supported languages
NLLB_LANG = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "kn": "kan_Knda",
    "te": "tel_Telu",
}

# Entities to keep (tune if you want)
KEEP_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "EVENT"}

# Basic stoplist for junk entities that often appear from headlines
JUNK_ENTITIES = {
    "today", "yesterday", "tomorrow", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
    "government", "police", "court", "india", "indian", "state", "district"
}

# Script ranges (rough but effective)
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
KANNADA_RE    = re.compile(r"[\u0C80-\u0CFF]")
TELUGU_RE     = re.compile(r"[\u0C00-\u0C7F]")

# ----------------------------- Utils -----------------------------

def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s)
    return s

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def guess_lang_by_script(text: str) -> str:
    """Fallback if your dataset doesn't have a language column."""
    t = text or ""
    if DEVANAGARI_RE.search(t):
        return "hi"
    if KANNADA_RE.search(t):
        return "kn"
    if TELUGU_RE.search(t):
        return "te"
    # default
    return "en"

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def pick_first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None

def clean_entity(ent: str) -> str:
    ent = normalize_text(ent)
    ent = ent.strip(" ,.;:()[]{}\"'`")
    ent = re.sub(r"\s+", " ", ent)
    # Drop weird leftovers
    if not ent:
        return ""
    if len(ent) < 3:
        return ""
    if ent.isnumeric():
        return ""
    return ent

def looks_like_junk(ent: str) -> bool:
    t = ent.lower().strip()
    if t in JUNK_ENTITIES:
        return True
    # only digits/punct
    if re.fullmatch(r"[\W_]+", t or ""):
        return True
    return False

# ----------------------------- Translation Cache -----------------------------

@dataclass
class CacheItem:
    text: str
    lang: str
    text_en: str

class TranslationCache:
    """
    Disk-backed cache in a CSV file:
      cache.csv columns: key,text,lang,text_en
    key = sha1(lang + "||" + text)
    """
    def __init__(self, cache_csv_path: str):
        self.cache_csv_path = cache_csv_path
        self.map: Dict[str, CacheItem] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.cache_csv_path):
            return
        try:
            df = pd.read_csv(self.cache_csv_path)
            for _, r in df.iterrows():
                k = str(r["key"])
                self.map[k] = CacheItem(
                    text=str(r["text"]),
                    lang=str(r["lang"]),
                    text_en=str(r["text_en"]),
                )
        except Exception as e:
            print(f"[WARN] Could not load cache CSV: {e}")

    def get(self, text: str, lang: str) -> Optional[str]:
        k = sha1(f"{lang}||{text}")
        item = self.map.get(k)
        return item.text_en if item else None

    def put_many(self, items: List[CacheItem]) -> None:
        if not items:
            return
        # Update in-memory
        for it in items:
            k = sha1(f"{it.lang}||{it.text}")
            self.map[k] = it

        # Append to disk
        file_exists = os.path.exists(self.cache_csv_path)
        with open(self.cache_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["key", "text", "lang", "text_en"])
            for it in items:
                writer.writerow([sha1(f"{it.lang}||{it.text}"), it.text, it.lang, it.text_en])

# ----------------------------- NLLB Translator -----------------------------

class NLLBTranslator:
    def __init__(self, model_name: str, device: str):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def translate_batch(self, texts: List[str], src_lang: str) -> List[str]:
        """
        Translate a list of texts in the same src_lang to English.
        """
        if src_lang == "en":
            return [normalize_text(t) for t in texts]

        if src_lang not in NLLB_LANG:
            # unknown, return as-is
            return [normalize_text(t) for t in texts]

        self.tokenizer.src_lang = NLLB_LANG[src_lang]
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(self.device)

        eng_token_id = self.tokenizer.convert_tokens_to_ids("eng_Latn")
        out = self.model.generate(
            **inputs,
            forced_bos_token_id=eng_token_id,
            max_length=128,
            num_beams=3,
        )
        decoded = [self.tokenizer.decode(x, skip_special_tokens=True) for x in out]
        return [normalize_text(d) for d in decoded]

# ----------------------------- spaCy NER -----------------------------

def load_spacy(model_name: str) -> Tuple[spacy.language.Language, str]:
    """
    Tries requested model; if fails, tries en_core_web_sm.
    """
    try:
        nlp = spacy.load(model_name)
        return nlp, model_name
    except Exception:
        try:
            nlp = spacy.load(DEFAULT_SPACY_SM)
            return nlp, DEFAULT_SPACY_SM
        except Exception as e:
            raise RuntimeError(
                "spaCy model not found. Install one of:\n"
                "  python -m spacy download en_core_web_sm\n"
                "  python -m spacy download en_core_web_trf\n"
            ) from e

def extract_entities(nlp: spacy.language.Language, text_en: str) -> List[str]:
    doc = nlp(text_en)
    out = []
    for ent in doc.ents:
        if ent.label_ in KEEP_ENTITY_LABELS:
            ce = clean_entity(ent.text)
            if ce and not looks_like_junk(ce):
                out.append(ce)
    return out

# ----------------------------- Main Pipeline -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input dataset CSV/XLSX")
    parser.add_argument("--out-dir", default="lexicon_out", help="Output folder")
    parser.add_argument("--lang-col", default="", help="Language column name (optional)")
    parser.add_argument(
        "--text-cols",
        default="headline,title,text,claim,news,content,description",
        help="Comma-separated candidate text columns to use (first found is used)",
    )
    parser.add_argument("--min-count", type=int, default=2, help="Keep entities with count >= this")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional cap for testing (0 = no cap)")
    parser.add_argument("--batch-size", type=int, default=16, help="Translation batch size")
    parser.add_argument("--spacy-model", default=DEFAULT_SPACY_TRF, help="spaCy English model")
    parser.add_argument("--translator", default=DEFAULT_TRANSLATOR, help="NLLB model name")
    parser.add_argument("--cache-csv", default="", help="Optional cache CSV path (default: out-dir/translated_cache.csv)")
    parser.add_argument("--device", default="", help="cuda or cpu (default auto)")
    args = parser.parse_args()

    in_path = args.input
    out_dir = args.out_dir
    safe_mkdir(out_dir)

    cache_csv = args.cache_csv or os.path.join(out_dir, "translated_cache.csv")

    device = args.device.strip().lower()
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Input: {in_path}")
    print(f"[INFO] Output dir: {out_dir}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Cache: {cache_csv}")

    # ----- Load dataset -----
    if in_path.lower().endswith(".xlsx") or in_path.lower().endswith(".xls"):
        df = pd.read_excel(in_path)
    else:
        df = pd.read_csv(in_path)

    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows)

    text_cols = [c.strip() for c in args.text_cols.split(",") if c.strip()]
    text_col = pick_first_existing_col(df, text_cols)
    if not text_col:
        raise ValueError(f"None of the text columns found. Tried: {text_cols}\nColumns: {list(df.columns)}")

    lang_col = args.lang_col.strip()
    if lang_col:
        # try case-insensitive match
        lang_col_real = pick_first_existing_col(df, [lang_col])
        if not lang_col_real:
            raise ValueError(f"Language column '{lang_col}' not found in dataset.")
        lang_col = lang_col_real

    print(f"[INFO] Using text column: {text_col}")
    print(f"[INFO] Using language column: {lang_col if lang_col else '(script detect)'}")

    # ----- Normalize + dedup texts -----
    rows: List[Tuple[str, str]] = []
    for _, r in df.iterrows():
        txt = normalize_text(str(r.get(text_col, "") or ""))
        if not txt:
            continue
        if lang_col:
            lg = normalize_text(str(r.get(lang_col, "") or "")).lower()
            lg = lg[:2]  # en/hi/kn/te style
            if lg not in {"en", "hi", "kn", "te"}:
                lg = guess_lang_by_script(txt)
        else:
            lg = guess_lang_by_script(txt)
        rows.append((txt, lg))

    # Dedup on (lang,text)
    seen = set()
    uniq: List[Tuple[str, str]] = []
    for txt, lg in rows:
        k = (lg, txt.lower())
        if k not in seen:
            uniq.append((txt, lg))
            seen.add(k)

    print(f"[INFO] Total rows: {len(rows)} | Unique texts: {len(uniq)}")

    # ----- Load models -----
    print("[INFO] Loading spaCy...")
    nlp, used_spacy = load_spacy(args.spacy_model)
    print(f"[INFO] spaCy model: {used_spacy}")

    print("[INFO] Loading NLLB translator (first time may take a while)...")
    translator = NLLBTranslator(args.translator, device=device)

    cache = TranslationCache(cache_csv)

    # ----- Translate to English -----
    translated_en: List[str] = []
    lang_list: List[str] = []
    orig_list: List[str] = []

    # Group by language for efficient batching
    by_lang: Dict[str, List[str]] = {"en": [], "hi": [], "kn": [], "te": []}
    for txt, lg in uniq:
        if lg not in by_lang:
            lg = "en"
        by_lang[lg].append(txt)

    def process_lang(lg: str, texts: List[str]):
        nonlocal translated_en, lang_list, orig_list
        if not texts:
            return

        # translation in batches with cache
        for i in tqdm(range(0, len(texts), args.batch_size), desc=f"Translate {lg} -> en"):
            batch = texts[i : i + args.batch_size]
            batch_out: List[Optional[str]] = []
            to_translate: List[str] = []
            translate_indices: List[int] = []

            # cache lookup
            for j, t in enumerate(batch):
                cached = cache.get(t, lg)
                if cached is not None:
                    batch_out.append(cached)
                else:
                    batch_out.append(None)
                    to_translate.append(t)
                    translate_indices.append(j)

            # translate missing
            if to_translate:
                outs = translator.translate_batch(to_translate, src_lang=lg)
                cache_items = []
                for src_text, en_text in zip(to_translate, outs):
                    cache_items.append(CacheItem(text=src_text, lang=lg, text_en=en_text))
                cache.put_many(cache_items)

                for idx, en_text in zip(translate_indices, outs):
                    batch_out[idx] = en_text

            # collect
            for src_text, en_text in zip(batch, batch_out):
                en_text = normalize_text(en_text or "")
                if not en_text:
                    continue
                orig_list.append(src_text)
                lang_list.append(lg)
                translated_en.append(en_text)

    for lg in ["en", "hi", "kn", "te"]:
        process_lang(lg, by_lang[lg])

    print(f"[INFO] Translated rows ready for NER: {len(translated_en)}")

    # ----- spaCy NER over translated English -----
    entity_counter = Counter()

    # Use spaCy pipe for speed
    print("[INFO] Running NER (spaCy pipe)...")
    for doc in tqdm(nlp.pipe(translated_en, batch_size=32), total=len(translated_en), desc="NER"):
        for ent in doc.ents:
            if ent.label_ in KEEP_ENTITY_LABELS:
                ce = clean_entity(ent.text)
                if ce and not looks_like_junk(ce):
                    entity_counter[ce] += 1

    print(f"[INFO] Total unique entities (raw): {len(entity_counter)}")

    # ----- Filter by min-count -----
    min_count = max(1, int(args.min_count))
    kept = [(e, c) for e, c in entity_counter.items() if c >= min_count]
    kept.sort(key=lambda x: x[1], reverse=True)

    print(f"[INFO] Entities kept (count >= {min_count}): {len(kept)}")

    # ----- Save outputs -----
    out_counts_csv = os.path.join(out_dir, "india_entities_with_counts.csv")
    out_txt = os.path.join(out_dir, "india_entities.txt")

    pd.DataFrame(kept, columns=["entity", "count"]).to_csv(out_counts_csv, index=False, encoding="utf-8")
    with open(out_txt, "w", encoding="utf-8") as f:
        for ent, _ in kept:
            f.write(ent + "\n")

    # Optional: Save a small sample for sanity checking
    sample_path = os.path.join(out_dir, "sample_translations.csv")
    sample_df = pd.DataFrame({
        "text": orig_list[:200],
        "lang": lang_list[:200],
        "text_en": translated_en[:200]
    })
    sample_df.to_csv(sample_path, index=False, encoding="utf-8")

    print("\n[DONE]")
    print(f"  - {out_txt}")
    print(f"  - {out_counts_csv}")
    print(f"  - {cache_csv}")
    print(f"  - {sample_path}")
    print("\nNext: In Streamlit, load india_entities.txt and merge with your seed list.")

if __name__ == "__main__":
    main()
