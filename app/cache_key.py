# cache_key.py
import hashlib
import unicodedata
import re

def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def make_cache_key(text: str) -> str:
    norm = normalize_text(text)
    return hashlib.md5(norm.encode("utf-8")).hexdigest()
