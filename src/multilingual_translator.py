"""
AllerScan Multilingual Translation Engine (Task 5: Route B).

Implements Step 7 of docs/05_ingredient_translation.md:
  - BaseTranslationEngine interface supporting swappable translation backends.
  - Offline GlossaryTranslationEngine: Item-by-item translation using the 32,000+
    multilingual glossary, with placeholder protection for E-numbers, percentages,
    and weights/measures.
  - Optional NeuralTranslationEngine (Helsinki-NLP / MarianMT) with graceful fallback.
  - Result caching and error handling (continues with Route A if translation fails).
"""
from __future__ import annotations

import html
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from .multilingual_parser import (
        GlossaryHit,
        detect_language,
        get_glossary_matcher,
        normalize_multilingual_text,
        split_ingredient_items,
    )
except ImportError:
    from multilingual_parser import (
        GlossaryHit,
        detect_language,
        get_glossary_matcher,
        normalize_multilingual_text,
        split_ingredient_items,
    )

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data" / "translation"


@dataclass
class TranslatedItem:
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    placeholders_restored: bool
    glossary_override_applied: bool
    detected_allergens: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TranslationBatchResult:
    items: List[TranslatedItem]
    full_translated_text: str
    source_language: str
    engine_name: str
    success: bool
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "items": [it.to_dict() for it in self.items],
            "full_translated_text": self.full_translated_text,
            "source_language": self.source_language,
            "engine_name": self.engine_name,
            "success": self.success,
            "notes": self.notes,
        }


# -----------------------------------------------------------------------------
# Placeholder Protection (Step 7.3)
# -----------------------------------------------------------------------------
ENUM_PATTERN = re.compile(r"\bE\d{3,4}[a-z]?\b", re.I)
PCT_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?%", re.I)
NUM_UNIT_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|kg|ml|cl|oz|lb|g|l|kcal|kj)\b", re.I)


def protect_placeholders(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Replace E-numbers, percentages, numbers, and units with safe placeholders
    before translation so they cannot be corrupted or mistranslated.
    """
    mapping: Dict[str, str] = {}
    counter = 0

    def rep_enum(m: re.Match) -> str:
        nonlocal counter
        ph = f"__ENUM_{counter}__"
        mapping[ph] = m.group(0)
        counter += 1
        return ph

    def rep_pct(m: re.Match) -> str:
        nonlocal counter
        ph = f"__PCT_{counter}__"
        mapping[ph] = m.group(0)
        counter += 1
        return ph

    def rep_num(m: re.Match) -> str:
        nonlocal counter
        ph = f"__UNIT_{counter}__"
        mapping[ph] = m.group(0)
        counter += 1
        return ph

    s = ENUM_PATTERN.sub(rep_enum, text)
    s = PCT_PATTERN.sub(rep_pct, s)
    s = NUM_UNIT_PATTERN.sub(rep_num, s)
    return s, mapping


def restore_placeholders(text: str, mapping: Dict[str, str]) -> str:
    """Restore placeholders back to their original codes and percentages."""
    s = text
    for ph, orig in mapping.items():
        s = s.replace(ph, orig)
    return s


# -----------------------------------------------------------------------------
# Base Translation Engine Interface
# -----------------------------------------------------------------------------
class BaseTranslationEngine(ABC):
    @abstractmethod
    def translate_items(
        self,
        items: List[str],
        source_lang: str,
        target_lang: str = "en"
    ) -> TranslationBatchResult:
        """Translate a list of ingredient items into English."""
        pass


# -----------------------------------------------------------------------------
# 1. Offline Glossary Translation Engine
# -----------------------------------------------------------------------------
class GlossaryTranslationEngine(BaseTranslationEngine):
    """
    Fast, deterministic offline translation engine backed by the AllerScan
    multilingual glossary. Performs longest-match translation of food words,
    allergens, and additives while preserving sentence structure and percentages.
    """
    def __init__(self):
        self.glossary_terms: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._load_glossary()

    def _load_glossary(self):
        glossary_file = DATA_DIR / "multilingual_glossary.json"
        if glossary_file.exists():
            with open(glossary_file, "r", encoding="utf-8") as f:
                records = json.load(f)
            for r in records:
                l = r.get("language")
                norm = r.get("normalized")
                if l and norm:
                    if l not in self.glossary_terms:
                        self.glossary_terms[l] = {}
                    self.glossary_terms[l][norm] = r

    def translate_items(
        self,
        items: List[str],
        source_lang: str,
        target_lang: str = "en"
    ) -> TranslationBatchResult:
        if source_lang == "en":
            # Pass-through for English
            translated_items = [
                TranslatedItem(
                    original_text=it,
                    translated_text=it,
                    source_language="en",
                    target_language="en",
                    placeholders_restored=True,
                    glossary_override_applied=False,
                )
                for it in items
            ]
            return TranslationBatchResult(
                items=translated_items,
                full_translated_text=", ".join(items),
                source_language="en",
                engine_name="GlossaryTranslationEngine (Pass-through)",
                success=True,
            )

        lang_dict = self.glossary_terms.get(source_lang, {})
        matcher = get_glossary_matcher()
        translated_items: List[TranslatedItem] = []

        # Sort glossary terms by length descending for greedy substring replacement
        sorted_terms = sorted(lang_dict.keys(), key=lambda x: -len(x))

        for item in items:
            clean = normalize_multilingual_text(item)
            if not clean:
                continue

            protected_text, ph_map = protect_placeholders(clean)
            work_text = protected_text
            applied_override = False

            # Translate known terms via longest match
            low_text = work_text.lower()
            for term in sorted_terms:
                if len(term) < 2 or term not in low_text:
                    continue
                entry = lang_dict[term]
                can_key = entry.get("canonical_key")
                eng_name = can_key.replace("_", " ") if can_key and can_key != "ingredient_not_allergen" else None

                if not eng_name:
                    continue

                if source_lang not in ("th", "zh-CN", "zh-TW", "ja"):
                    # Word boundary check
                    pat = rf"(?:\b|^){re.escape(term)}(?:\b|$)"
                    if re.search(pat, work_text, flags=re.I):
                        work_text = re.sub(pat, eng_name, work_text, flags=re.I)
                        applied_override = True
                        low_text = work_text.lower()
                else:
                    work_text = work_text.replace(term, f" {eng_name} ")
                    applied_override = True
                    low_text = work_text.lower()

            # Restore placeholders
            final_item_text = restore_placeholders(work_text, ph_map)
            final_item_text = re.sub(r"\s{2,}", " ", final_item_text).strip()

            # Find allergens present in this item via Route A
            item_hits = matcher.match_item(item, lang=source_lang)
            allergens_in_item = [h.canonical_key for h in item_hits if h.kind != "negated"]

            translated_items.append(TranslatedItem(
                original_text=item,
                translated_text=final_item_text,
                source_language=source_lang,
                target_language="en",
                placeholders_restored=True,
                glossary_override_applied=applied_override,
                detected_allergens=allergens_in_item,
            ))

        full_text = ", ".join(t.translated_text for t in translated_items if t.translated_text)
        return TranslationBatchResult(
            items=translated_items,
            full_translated_text=full_text,
            source_language=source_lang,
            engine_name="GlossaryTranslationEngine (Offline)",
            success=True,
        )


# -----------------------------------------------------------------------------
# 2. Neural Machine Translation Engine (MarianMT / Opus-MT)
# -----------------------------------------------------------------------------
class NeuralTranslationEngine(BaseTranslationEngine):
    """
    Helsinki-NLP Opus-MT translation engine. Runs offline when model weights
    are present. Falls back seamlessly to GlossaryTranslationEngine if not installed.
    """
    def __init__(self, fallback_engine: Optional[BaseTranslationEngine] = None):
        self.fallback = fallback_engine or GlossaryTranslationEngine()
        self.loaded_models: Dict[str, Any] = {}

    def translate_items(
        self,
        items: List[str],
        source_lang: str,
        target_lang: str = "en"
    ) -> TranslationBatchResult:
        if source_lang == "en":
            return self.fallback.translate_items(items, source_lang, target_lang)

        # Attempt neural pipeline if transformers installed
        try:
            import torch
            from transformers import MarianMTModel, MarianTokenizer

            model_name = f"Helsinki-NLP/opus-mt-{source_lang}-en"
            if model_name not in self.loaded_models:
                logger.info("Loading MarianMT model %s...", model_name)
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model.to(device)
                self.loaded_models[model_name] = (tokenizer, model, device)

            tokenizer, model, device = self.loaded_models[model_name]

            # Protect placeholders
            protected_items = []
            maps = []
            for it in items:
                prot, p_map = protect_placeholders(it)
                protected_items.append(prot)
                maps.append(p_map)

            # Batched neural inference
            inputs = tokenizer(protected_items, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                gen = model.generate(**inputs, max_length=128)
            decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)

            # Restore placeholders
            translated_items = []
            for orig, dec, p_map in zip(items, decoded, maps):
                restored = restore_placeholders(dec, p_map)
                translated_items.append(TranslatedItem(
                    original_text=orig,
                    translated_text=restored,
                    source_language=source_lang,
                    target_language="en",
                    placeholders_restored=True,
                    glossary_override_applied=False,
                ))

            full_text = ", ".join(t.translated_text for t in translated_items)
            return TranslationBatchResult(
                items=translated_items,
                full_translated_text=full_text,
                source_language=source_lang,
                engine_name=f"MarianMT ({model_name})",
                success=True,
            )

        except Exception as exc:
            logger.info("Neural translation not available (%s); using offline glossary translator.", exc)
            res = self.fallback.translate_items(items, source_lang, target_lang)
            res.notes.append(f"Offline glossary translation used ({exc}).")
            return res


# Global singleton instance
_GLOBAL_TRANSLATOR: Optional[BaseTranslationEngine] = None

def get_translator() -> BaseTranslationEngine:
    global _GLOBAL_TRANSLATOR
    if _GLOBAL_TRANSLATOR is None:
        _GLOBAL_TRANSLATOR = NeuralTranslationEngine(fallback_engine=GlossaryTranslationEngine())
    return _GLOBAL_TRANSLATOR


def translate_label_text(text: str, source_lang: Optional[str] = None) -> TranslationBatchResult:
    """Public helper to translate an ingredient label into English."""
    if not source_lang or source_lang == "auto":
        source_lang, _ = detect_language(text)

    items = split_ingredient_items(text, lang=source_lang)
    translator = get_translator()
    return translator.translate_items(items, source_lang=source_lang, target_lang="en")
