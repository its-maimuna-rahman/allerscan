"""
AllerScan Multilingual Text Normalisation, Script Identification,
and Route A Glossary Matcher.

Implements Steps 3, 5, and 6 of docs/05_ingredient_translation.md:
  - Unicode NFC normalisation and zero-width character removal.
  - Native digit and punctuation mapping across all Phase 1 scripts.
  - Script identification and short-fragment language detection.
  - Sub-ingredient list splitting preserving brackets.
  - Longest-match Route A glossary search for both spaced and unspaced scripts.
  - Multilingual negation ("free from") and trace ("may contain") detection.
"""
from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data" / "translation"

# -----------------------------------------------------------------------------
# 1. Native Digit & Punctuation Translation Tables
# -----------------------------------------------------------------------------
DIGIT_MAP = str.maketrans({
    # Bengali (০-৯)
    '০': '0', '১': '1', '২': '2', '৩': '3', '৪': '4',
    '৫': '5', '৬': '6', '৭': '7', '৮': '8', '৯': '9',
    # Devanagari (०-९)
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9',
    # Arabic-Indic (٠-٩)
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
    # Eastern Arabic-Indic / Persian / Urdu (۰-۹)
    '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
    '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9',
})

PUNCT_MAP = str.maketrans({
    '।': '.',    # Bengali / Hindi danda
    '॥': '.',    # Double danda
    '،': ',',    # Arabic comma
    '؛': ';',    # Arabic semicolon
    '؟': '?',    # Arabic question mark
    '，': ',',    # Fullwidth comma (Chinese / Japanese)
    '、': ',',    # Ideographic comma (Japanese / Chinese)
    '；': ';',    # Fullwidth semicolon
    '：': ':',    # Fullwidth colon
    '・': ',',    # Japanese middle dot separator
    '·': ',',    # Middle dot
    '（': '(',    # Fullwidth left bracket
    '）': ')',    # Fullwidth right bracket
    '【': '[',    # Fullwidth black lenticular bracket
    '】': ']',
    '「': '"',    # Japanese quote marks
    '」': '"',
})

ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200F\uFEFF\u202A-\u202E]")


def normalize_multilingual_text(text: str) -> str:
    """
    Step 5 Normalisation:
      1. Unicode NFC normalisation.
      2. Strip zero-width and bidirectional control characters.
      3. Map native digits to Western digits (0-9).
      4. Map native script punctuation to standard separators (, . : ; ()).
      5. Consolidate whitespace while preserving structure.
    """
    if not text:
        return ""
    # 1. Unicode NFC
    s = unicodedata.normalize("NFC", text)
    # 2. Strip zero-width / bidi controls
    s = ZERO_WIDTH_RE.sub("", s)
    # 3. Native digits
    s = s.translate(DIGIT_MAP)
    # 4. Native punctuation
    s = s.translate(PUNCT_MAP)
    # 5. Native E-number prefix mapping (e.g. Bengali ই-102 / ই১০২ -> E102)
    s = re.sub(r"(?:^|(?<=[\s,;.(/]))[ই][\s\-]*(?=[0-9]{3,4})", "E", s)
    # 6. Collapse spaces
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s+", "\n", s)
    return s.strip()


# -----------------------------------------------------------------------------
# 2. Script Identification
# -----------------------------------------------------------------------------
SCRIPT_RANGES = {
    "bengali": re.compile(r"[\u0980-\u09FF]"),
    "devanagari": re.compile(r"[\u0900-\u097F]"),
    "arabic": re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]"),
    "thai": re.compile(r"[\u0E00-\u0E7F]"),
    "hangul": re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]"),
    "japanese_kana": re.compile(r"[\u3040-\u309F\u30A0-\u30FF]"),
    "han": re.compile(r"[\u4E00-\u9FFF]"),
    "cyrillic": re.compile(r"[\u0400-\u04FF]"),
    "latin": re.compile(r"[a-zA-Z\u00C0-\u024F]"),
}


def detect_script(text: str) -> str:
    """Detect dominant writing script of a text segment."""
    counts = {script: len(pat.findall(text)) for script, pat in SCRIPT_RANGES.items()}
    # Disregard single stray characters if another script dominates
    top_script, top_count = max(counts.items(), key=lambda x: x[1])
    if top_count == 0:
        return "unknown"
    return top_script


# -----------------------------------------------------------------------------
# 3. Short-Fragment Language Identification
# -----------------------------------------------------------------------------
# Characteristic letter sets for Latin-script disambiguation
LATIN_DIACRITICS = {
    "fr": re.compile(r"[éèêëàâùûîïôçœæ]", re.I),
    "de": re.compile(r"[äöüß]", re.I),
    "es": re.compile(r"[ñáéíóúü¡¿]", re.I),
    "it": re.compile(r"[àèéìíîòóùú]", re.I),
    "pt": re.compile(r"[ãõçáéíóúâêôà]", re.I),
    "tr": re.compile(r"[çğıöşüÇĞİÖŞÜ]", re.I),
    "ru": re.compile(r"[а-яёА-ЯЁ]"),
}

# Distinctive Urdu vs Arabic letters
URDU_LETTERS = re.compile(r"[ٹڈڑںےہھپچژگ]")


def detect_language(text: str, default_lang: Optional[str] = None) -> Tuple[str, float]:
    """
    Step 6: Script-first, rule-and-heuristic language identification
    optimized specifically for short packaging ingredient fragments.
    Returns: (iso_lang_code, confidence)
    """
    clean = normalize_multilingual_text(text)
    if not clean:
        return default_lang or "unknown", 0.0

    alpha_count = sum(c.isalpha() for c in clean)
    if alpha_count < 3:
        return "unknown", 0.0

    script = detect_script(clean)
    if script == "unknown":
        return "unknown", 0.0

    if script == "bengali":
        return "bn", 0.98

    if script == "devanagari":
        return "hi", 0.95

    if script == "thai":
        return "th", 0.99

    if script == "hangul":
        return "ko", 0.99

    if script == "japanese_kana":
        return "ja", 0.99

    if script == "han":
        # Pure Han without Kana: distinguish Chinese Simplified vs Traditional
        has_trad = any(c in clean for c in "產產品製配料點過敏原資訊麩質")
        return ("zh-TW" if has_trad else "zh-CN"), 0.92

    if script == "cyrillic":
        return "ru", 0.96

    if script == "arabic":
        if URDU_LETTERS.search(clean):
            return "ur", 0.92
        return "ar", 0.94

    if script == "latin":
        low = clean.lower()
        # 1. Check Turkish unique letters
        if any(c in low for c in "ğış"):
            return "tr", 0.95
        # 2. Check German specific
        if "ß" in low or any(c in low for c in "äöü"):
            return "de", 0.92
        # 3. Check Spanish specific
        if "ñ" in low or "¿" in clean or "¡" in clean:
            return "es", 0.92
        # 4. Check Portuguese specific
        if "ã" in low or "õ" in low:
            return "pt", 0.92
        # 5. Check French specific
        if any(c in low for c in "œæ") or ("é" in low and "de" in low):
            return "fr", 0.90

        # Check vocabulary statement keywords
        if any(w in low for w in ["ingrédients", "ingredients", "lait", "blé", "sucre", "soja", "arachide"]):
            return "fr", 0.88
        if any(w in low for w in ["zutaten", "weizen", "milch", "zucker", "erdnuss", "inhaltsstoffe"]):
            return "de", 0.88
        if any(w in low for w in ["ingredientes", "leche", "azúcar", "cacahuete", "trigo"]):
            return "es", 0.88
        if any(w in low for w in ["ingredienti", "latte", "zucchero", "farina", "grano"]):
            return "it", 0.88
        if any(w in low for w in ["komposisi", "gula", "minyak", "tepung", "garam"]):
            return "id", 0.88
        if any(w in low for w in ["ramuan", "gula", "minyak", "tepung", "garam"]):
            return "ms", 0.88

        # Fallback to English for Latin if no distinct markers
        return (default_lang or "en"), 0.80

    return default_lang or "en", 0.50


# -----------------------------------------------------------------------------
# 4. Item Splitting (Preserving Sub-Ingredient Parentheses)
# -----------------------------------------------------------------------------
def split_ingredient_items(text: str, lang: str = "en") -> List[str]:
    """
    Step 5.3: Split ingredient lists into items on commas, semicolons, bullets,
    and newlines, while keeping sub-ingredient lists in brackets attached.
    Example: 'sugar, chocolate (cocoa, milk), vanilla' -> ['sugar', 'chocolate (cocoa, milk)', 'vanilla']
    """
    norm = normalize_multilingual_text(text)
    if not norm:
        return []

    # Strip ingredient header if present at start
    norm = re.sub(r"^(?:ingredients?|ingrédients|zutaten|ingredientes|ingredienti|состав|উপকরণ|উপাদান|सामग्री|المكونات|配料|原材料名|ส่วนประกอบ)\s*[:.\-]\s*", "", norm, flags=re.I)

    items: List[str] = []
    buf: List[str] = []
    depth = 0

    for i, ch in enumerate(norm):
        if ch in "([{\u3010":
            depth += 1
            buf.append(ch)
        elif ch in ")]}\u3011":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch in (",", ";", "\n", "\u2022", "•", "।", "॥") and depth == 0:
            item_text = "".join(buf).strip()
            if item_text:
                items.append(item_text)
            buf = []
        elif ch == "." and depth == 0:
            is_decimal = (i > 0 and norm[i - 1].isdigit() and i + 1 < len(norm) and norm[i + 1].isdigit())
            if not is_decimal:
                item_text = "".join(buf).strip()
                if item_text:
                    items.append(item_text)
                buf = []
            else:
                buf.append(ch)
        else:
            buf.append(ch)

    if buf:
        item_text = "".join(buf).strip()
        if item_text:
            items.append(item_text)

    # Secondary split for no-space Thai or Chinese if single giant item returned
    if len(items) <= 1 and lang in ("th", "zh-CN", "zh-TW", "ja"):
        raw = items[0] if items else norm
        sub = re.split(r"[,;、，・\s]+", raw)
        clean_sub = [s.strip() for s in sub if s.strip()]
        if len(clean_sub) > 1:
            return clean_sub

    return items


# -----------------------------------------------------------------------------
# 5. Route A Multilingual Glossary Matcher
# -----------------------------------------------------------------------------
@dataclass
class GlossaryHit:
    canonical_key: str
    matched_term: str
    language: str
    kind: str  # 'direct', 'trace', 'negated'
    confidence: str  # 'high', 'medium', 'low'
    reviewed: bool
    source: str
    item_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical_key": self.canonical_key,
            "matched_term": self.matched_term,
            "language": self.language,
            "kind": self.kind,
            "confidence": self.confidence,
            "reviewed": self.reviewed,
            "source": self.source,
            "item_text": self.item_text,
        }


class GlossaryMatcher:
    """
    Route A: Multilingual glossary matcher supporting direct exact
    and longest-match trie lookup across all 18 Phase 1 languages.
    """
    def __init__(self):
        self.glossary_by_lang: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        self.statements_by_lang: Dict[str, Dict[str, List[str]]] = {}
        self._load_data()

    def _load_data(self):
        glossary_file = DATA_DIR / "multilingual_glossary.json"
        phrases_file = DATA_DIR / "statement_phrases.json"

        if not glossary_file.exists() or not phrases_file.exists():
            logger.warning("Glossary data missing at %s. Attempting rebuild...", glossary_file)
            try:
                from tools.build_multilingual_glossary import build_multilingual_glossary
                build_multilingual_glossary()
            except Exception as exc:
                logger.error("Failed to build glossary: %s", exc)

        if glossary_file.exists():
            with open(glossary_file, "r", encoding="utf-8") as f:
                records = json.load(f)
            for r in records:
                l = r.get("language")
                norm = r.get("normalized")
                if l and norm:
                    self.glossary_by_lang[l][norm] = r

        if phrases_file.exists():
            with open(phrases_file, "r", encoding="utf-8") as f:
                self.statements_by_lang = json.load(f)

    def match_item(self, item_text: str, lang: str) -> List[GlossaryHit]:
        """
        Match a single ingredient item text against the glossary in language `lang`.
        Supports longest-match search, negation detection, and trace detection.
        """
        norm = normalize_multilingual_text(item_text).lower()
        if not norm:
            return []

        hits: List[GlossaryHit] = []
        lang_dict = self.glossary_by_lang.get(lang, {})
        statements = self.statements_by_lang.get(lang, {})

        # Check statement classification (free from vs may contain)
        is_negated = False
        for free_pat in statements.get("free_from", []):
            if free_pat.lower() in norm:
                is_negated = True
                break

        is_trace = False
        for trace_pat in statements.get("may_contain", []):
            if trace_pat.lower() in norm:
                is_trace = True
                break

        kind = "negated" if is_negated else ("trace" if is_trace else "direct")

        # 1. Exact match on whole item
        if norm in lang_dict:
            entry = lang_dict[norm]
            can_key = entry.get("canonical_key")
            if can_key and can_key != "ingredient_not_allergen":
                hits.append(GlossaryHit(
                    canonical_key=can_key,
                    matched_term=entry.get("term", norm),
                    language=lang,
                    kind=kind,
                    confidence=entry.get("confidence", "high"),
                    reviewed=entry.get("reviewed", False),
                    source=entry.get("source", "glossary"),
                    item_text=item_text
                ))
                return hits

        # 2. Sub-string / longest-match search (sorted descending by length)
        # Allows matching inside unspaced scripts (Thai, Chinese, Japanese)
        # and compound phrases (e.g. 'দুধের গুঁড়া', 'huile d'arachide')
        sorted_terms = sorted(lang_dict.keys(), key=lambda x: -len(x))
        seen_keys: Set[str] = set()

        for term in sorted_terms:
            if len(term) < 2 or term not in norm:
                continue
            # For space-separated languages, ensure word boundary match
            if lang not in ("th", "zh-CN", "zh-TW", "ja"):
                pattern = rf"(?:\b|^){re.escape(term)}(?:\b|$)"
                match_found = bool(re.search(pattern, norm))
            else:
                match_found = True

            if match_found:
                entry = lang_dict[term]
                can_key = entry.get("canonical_key")
                if can_key and can_key != "ingredient_not_allergen" and can_key not in seen_keys:
                    seen_keys.add(can_key)
                    hits.append(GlossaryHit(
                        canonical_key=can_key,
                        matched_term=entry.get("term", term),
                        language=lang,
                        kind=kind,
                        confidence=entry.get("confidence", "high"),
                        reviewed=entry.get("reviewed", False),
                        source=entry.get("source", "glossary"),
                        item_text=item_text
                    ))

        return hits

    def match_full_text(self, text: str, lang: Optional[str] = None) -> Tuple[List[GlossaryHit], str]:
        """
        Match whole ingredient list text. Automatically detects language if not specified.
        Returns: (hits, detected_language)
        """
        if not lang or lang == "auto":
            lang, _ = detect_language(text)

        items = split_ingredient_items(text, lang=lang)
        all_hits: List[GlossaryHit] = []
        for item in items:
            hits = self.match_item(item, lang=lang)
            all_hits.extend(hits)

        return all_hits, lang


# Global singleton instance
_GLOBAL_GLOSSARY_MATCHER: Optional[GlossaryMatcher] = None

def get_glossary_matcher() -> GlossaryMatcher:
    global _GLOBAL_GLOSSARY_MATCHER
    if _GLOBAL_GLOSSARY_MATCHER is None:
        _GLOBAL_GLOSSARY_MATCHER = GlossaryMatcher()
    return _GLOBAL_GLOSSARY_MATCHER
