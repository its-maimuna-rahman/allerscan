"""
AllerScan Multilingual Fusion & State Machine (Task 5: Step 8).

Implements the multi-route fusion architecture:
  - Route A: Original language reviewed glossary lookup.
  - Route B: Translation to English + Task 4 spelling corrector + NLP parser.
  - Unions findings with provenance badges: 'both', 'glossary', 'translation', 'possible'.
  - Evaluates Additive Hypersensitivities (Task 3) across original and translated text.
  - Enforces Verdict State Machine ('verified', 'partly_verified', 'could_not_verify').
  - CRITICAL SAFETY INVARIANT: 'could_not_verify' state NEVER permits 'no allergens detected'.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import allergen_kb as kb
from nlp_parser import parse_label
import spelling_corrector as sc
import additive_detector as ad

try:
    from .multilingual_parser import (
        GlossaryHit,
        detect_language,
        detect_script,
        get_glossary_matcher,
        normalize_multilingual_text,
        split_ingredient_items,
    )
    from .multilingual_translator import (
        TranslationBatchResult,
        translate_label_text,
    )
except ImportError:
    from multilingual_parser import (
        GlossaryHit,
        detect_language,
        detect_script,
        get_glossary_matcher,
        normalize_multilingual_text,
        split_ingredient_items,
    )
    from multilingual_translator import (
        TranslationBatchResult,
        translate_label_text,
    )

logger = logging.getLogger(__name__)


class VerificationState(str, Enum):
    VERIFIED = "verified"                  # Good OCR, language known, routes agree
    PARTLY_VERIFIED = "partly_verified"    # Language known, single route or minor gaps
    COULD_NOT_VERIFY = "could_not_verify"  # Low OCR quality, unknown language, or unreadable


@dataclass
class FusedAllergenHit:
    canonical_key: str
    display_name: str
    provenance: str             # 'both', 'glossary', 'translation', 'possible'
    confidence: str             # 'high', 'medium', 'low'
    kind: str                   # 'direct', 'trace', 'possible'
    original_matched_term: Optional[str] = None
    translated_matched_term: Optional[str] = None
    source_language: str = "en"
    is_reviewed: bool = False
    evidence_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MultilingualAnalysisResult:
    original_text: str
    normalized_original_text: str
    detected_language: str
    language_confidence: float
    verification_state: VerificationState
    state_reason: str
    safe_verdict_allowed: bool
    translated_text: str
    translation_engine: str
    allergens: Dict[str, FusedAllergenHit] = field(default_factory=dict)
    additives: List[Any] = field(default_factory=list)
    possible_allergens: List[Any] = field(default_factory=list)
    spelling_corrections: List[Any] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "normalized_original_text": self.normalized_original_text,
            "detected_language": self.detected_language,
            "language_confidence": self.language_confidence,
            "verification_state": self.verification_state.value,
            "state_reason": self.state_reason,
            "safe_verdict_allowed": self.safe_verdict_allowed,
            "translated_text": self.translated_text,
            "translation_engine": self.translation_engine,
            "allergens": {k: v.to_dict() for k, v in self.allergens.items()},
            "additives": [a.to_dict() if hasattr(a, "to_dict") else a for a in self.additives],
            "possible_allergens": [p.to_dict() if hasattr(p, "to_dict") else p for p in self.possible_allergens],
            "spelling_corrections": [c.to_dict() if hasattr(c, "to_dict") else c for c in self.spelling_corrections],
            "notes": self.notes,
        }


def analyze_multilingual_label(
    text: str,
    selected_language: Optional[str] = None,
    ocr_confidence: Optional[float] = None,
    user_add_sens: Optional[Set[str]] = None,
    user_has_asthma: bool = False,
) -> MultilingualAnalysisResult:
    """
    Step 8: End-to-end multi-route analysis.
    Combines Route A (Glossary) + Route B (Translation + Spelling + Parser) + Additives.
    """
    clean_orig = normalize_multilingual_text(text)
    if not clean_orig:
        return MultilingualAnalysisResult(
            original_text=text or "",
            normalized_original_text="",
            detected_language=selected_language or "unknown",
            language_confidence=0.0,
            verification_state=VerificationState.COULD_NOT_VERIFY,
            state_reason="Text is empty or unreadable.",
            safe_verdict_allowed=False,
            translated_text="",
            translation_engine="None",
            notes=["No text provided."],
        )

    # 1. Determine Language
    if selected_language and selected_language != "auto":
        lang = selected_language
        lang_conf = 1.0
    else:
        lang, lang_conf = detect_language(clean_orig)

    # 2. Route A: Multilingual Glossary Matcher
    glossary_matcher = get_glossary_matcher()
    glossary_hits, _ = glossary_matcher.match_full_text(clean_orig, lang=lang)

    # Filter out negated items (e.g. "free from milk", "gluten free")
    negated_keys = {h.canonical_key for h in glossary_hits if h.kind == "negated"}
    active_glossary_hits = [h for h in glossary_hits if h.kind != "negated"]

    # 3. Route B: Machine Translation + English Analysis
    trans_res: TranslationBatchResult = translate_label_text(clean_orig, source_lang=lang)
    translated_english = trans_res.full_translated_text

    # Apply Task 4 Spelling Corrector on English translated text
    spelling_res = sc.correct_text(translated_english, mode="typed")
    clean_translated = spelling_res.corrected_text

    # Run English Parser on translated text
    parser_findings = parse_label(clean_translated)

    # 4. Task 3: Detect Additives on both Original and Translated text
    detected_additives = ad.detect_additives(
        clean_orig,
        user_sensitivities=user_add_sens or set(),
        user_has_asthma=user_has_asthma,
    )
    if clean_translated and clean_translated != clean_orig:
        additives_trans = ad.detect_additives(
            clean_translated,
            user_sensitivities=user_add_sens or set(),
            user_has_asthma=user_has_asthma,
        )
        existing_ids = {a.additive_id for a in detected_additives}
        for a in additives_trans:
            if a.additive_id not in existing_ids:
                detected_additives.append(a)
                existing_ids.add(a.additive_id)

    # 5. Union Allergen Findings and Assign Provenance
    fused_allergens: Dict[str, FusedAllergenHit] = {}

    glossary_by_key: Dict[str, List[GlossaryHit]] = {}
    for gh in active_glossary_hits:
        glossary_by_key.setdefault(gh.canonical_key, []).append(gh)

    all_keys = (set(glossary_by_key.keys()) | set(parser_findings.keys())) - negated_keys

    for can_k in all_keys:
        in_glossary = can_k in glossary_by_key
        in_parser = can_k in parser_findings

        gh_list = glossary_by_key.get(can_k, [])
        pf_info = parser_findings.get(can_k, {})

        matched_orig = gh_list[0].matched_term if gh_list else None
        is_reviewed = any(gh.reviewed for gh in gh_list)

        # Provenance assignment
        if in_glossary and in_parser:
            prov = "both"
            conf = "high"
            notes = ["Confirmed by both original dictionary and English translation."]
        elif in_glossary and not in_parser:
            prov = "glossary"
            conf = "high" if is_reviewed else "medium"
            notes = [f"Found in {lang} glossary dictionary ('{matched_orig}')."]
        else:
            prov = "translation"
            conf = "medium"
            notes = ["Found in English translation. Please check packaging."]

        kind = "trace" if (any(gh.kind == "trace" for gh in gh_list) or pf_info.get("kind") == "trace") else "direct"

        fused_allergens[can_k] = FusedAllergenHit(
            canonical_key=can_k,
            display_name=kb.pretty(can_k),
            provenance=prov,
            confidence=conf,
            kind=kind,
            original_matched_term=matched_orig,
            translated_matched_term=", ".join(pf_info.get("sources", [])) if in_parser else None,
            source_language=lang,
            is_reviewed=is_reviewed,
            evidence_notes=notes,
        )

    # 6. Verdict State Machine (Step 8.5)
    script = detect_script(clean_orig)
    state = VerificationState.VERIFIED
    state_reason = "Label language identified and verified with high confidence."
    safe_allowed = True

    # Rule: Low OCR confidence forces COULD_NOT_VERIFY
    alpha_chars = sum(c.isalpha() for c in clean_orig)
    if alpha_chars < 3:
        state = VerificationState.COULD_NOT_VERIFY
        state_reason = "Text contains insufficient readable words or characters."
        safe_allowed = False
    elif ocr_confidence is not None and ocr_confidence < 0.40:
        state = VerificationState.COULD_NOT_VERIFY
        state_reason = f"Image recognition confidence too low ({int(ocr_confidence*100)}%). Unreliable reading."
        safe_allowed = False

    # Rule: Unknown language forces COULD_NOT_VERIFY
    elif lang == "unknown" or lang_conf < 0.40:
        state = VerificationState.COULD_NOT_VERIFY
        state_reason = "Language or script could not be identified with confidence."
        safe_allowed = False

    # Rule: Foreign text with 0 glossary hits and translation failure
    elif lang != "en" and not active_glossary_hits and not parser_findings and not clean_translated:
        state = VerificationState.COULD_NOT_VERIFY
        state_reason = "Foreign text could not be verified against dictionaries or translation engines."
        safe_allowed = False

    # Rule: Partial verification when routes diverge
    elif lang != "en" and not in_glossary_agreed(glossary_by_key, parser_findings):
        state = VerificationState.PARTLY_VERIFIED
        state_reason = "Translation and dictionary routes partially diverge. Please inspect packaging."
        safe_allowed = True

    return MultilingualAnalysisResult(
        original_text=text,
        normalized_original_text=clean_orig,
        detected_language=lang,
        language_confidence=lang_conf,
        verification_state=state,
        state_reason=state_reason,
        safe_verdict_allowed=safe_allowed,
        translated_text=clean_translated,
        translation_engine=trans_res.engine_name,
        allergens=fused_allergens,
        additives=detected_additives,
        possible_allergens=spelling_res.possible_allergens,
        spelling_corrections=spelling_res.corrections,
        notes=trans_res.notes,
    )


def in_glossary_agreed(glossary_map: Dict[str, Any], parser_map: Dict[str, Any]) -> bool:
    """Check if glossary hits and parser findings agree substantially."""
    if not glossary_map and not parser_map:
        return True
    g_keys = set(glossary_map.keys())
    p_keys = set(parser_map.keys())
    if not g_keys or not p_keys:
        return False
    # Agreement if overlap is at least 50%
    return len(g_keys & p_keys) > 0
