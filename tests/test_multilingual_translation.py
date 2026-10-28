"""
Unit and integration test suite for Task 5: Multilingual ingredient reading,
translation, and multi-route fusion.

Implements Step 10.4 of docs/05_ingredient_translation.md:
  1. Text normalisation (digits, punctuation, zero-width characters).
  2. Sub-ingredient list splitting preserving brackets.
  3. Placeholder protection (E-numbers, percentages, numbers/units).
  4. Script and language identification on short ingredient fragments.
  5. Route A glossary matching (spaced and unspaced scripts).
  6. Negation ('free from') and trace ('may contain') handling.
  7. Verification state machine enforcing COULD_NOT_VERIFY safety invariant.
  8. Product parser non-Latin list acceptance (Step 9.5).
  9. Multilingual Reader Manager LRU cache.
  10. Full pipeline integration across Task 2, Task 3, and Task 4.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from multilingual_parser import (
    DIGIT_MAP,
    PUNCT_MAP,
    GlossaryHit,
    detect_language,
    detect_script,
    get_glossary_matcher,
    normalize_multilingual_text,
    split_ingredient_items,
)
from multilingual_translator import (
    TranslationBatchResult,
    protect_placeholders,
    restore_placeholders,
    translate_label_text,
)
from multilingual_fusion import (
    FusedAllergenHit,
    MultilingualAnalysisResult,
    VerificationState,
    analyze_multilingual_label,
)
from photo_scanner import (
    MultilingualReaderManager,
    extract_clean_ingredient_section,
    find_confirmed_header,
    readability,
)
from food_fetcher import parse_off


# -----------------------------------------------------------------------------
# 1. Normalisation Tests
# -----------------------------------------------------------------------------
def test_normalisation_native_digits():
    # Bengali (০-৯) -> (0-9)
    assert normalize_multilingual_text("চিনি ৫০গ্রাম, লবণ ৫%") == "চিনি 50গ্রাম, লবণ 5%"
    # Bengali transliterated E-number
    assert normalize_multilingual_text("ই১০২") == "E102"
    # Devanagari (०-९) -> (0-9)
    assert normalize_multilingual_text("दूध २५० मिली, चीनी १०%") == "दूध 250 मिली, चीनी 10%"
    # Arabic-Indic (٠-٩) -> (0-9)
    assert normalize_multilingual_text("حليب ١٠٠ مل، سكر ٥٪") == "حليب 100 مل, سكر 5٪"


def test_normalisation_native_punctuation():
    # Bengali danda
    assert normalize_multilingual_text("উপকরণ: দুধ। গম।") == "উপকরণ: দুধ. গম."
    # Arabic comma & semicolon
    assert normalize_multilingual_text("حليب، سكر؛ قمح") == "حليب, سكر; قمح"
    # Chinese/Japanese fullwidth commas, brackets and middle dots
    assert normalize_multilingual_text("配料：牛奶，花生、大豆・白糖") == "配料:牛奶,花生,大豆,白糖"
    assert normalize_multilingual_text("（乳製品）") == "(乳製品)"


def test_normalisation_zero_width_and_bidi_characters():
    text_with_zw = "दू\u200Bध, \uFEFFগম\u200E"
    clean = normalize_multilingual_text(text_with_zw)
    assert "\u200B" not in clean and "\uFEFF" not in clean and "\u200E" not in clean
    assert "दूध" in clean and "গম" in clean


# -----------------------------------------------------------------------------
# 2. Item Splitting Tests (Preserving Sub-Ingredients)
# -----------------------------------------------------------------------------
def test_split_items_preserves_parentheses():
    s = "sugar, chocolate (cocoa mass, milk powder, soy lecithin), vanilla extract"
    items = split_ingredient_items(s, lang="en")
    assert len(items) == 3
    assert items[0] == "sugar"
    assert items[1] == "chocolate (cocoa mass, milk powder, soy lecithin)"
    assert items[2] == "vanilla extract"


def test_split_items_multilingual_brackets():
    s = "উপকরণ: চিনি, চকোলেট (দুধের গুঁড়া, কোকো মাখন), লবণ"
    items = split_ingredient_items(s, lang="bn")
    assert len(items) == 3
    assert "চকোলেট (দুধের গুঁড়া, কোকো মাখন)" in items


# -----------------------------------------------------------------------------
# 3. Placeholder Protection & Restoration
# -----------------------------------------------------------------------------
def test_placeholder_protection_and_restoration():
    sample = "Contains 12.5% cocoa, E322 emulsifier, 250mg calcium, 30g protein"
    protected, mapping = protect_placeholders(sample)

    assert "12.5%" not in protected
    assert "E322" not in protected
    assert "250mg" not in protected
    assert "30g" not in protected
    assert len(mapping) >= 4

    restored = restore_placeholders(protected, mapping)
    assert restored == sample


# -----------------------------------------------------------------------------
# 4. Script & Language Identification
# -----------------------------------------------------------------------------
def test_detect_script():
    assert detect_script("উপকরণ: দুধ, গম") == "bengali"
    assert detect_script("المكونات: حليب، قمح") == "arabic"
    assert detect_script("सामग्री: दूध, गेहूं") == "devanagari"
    assert detect_script("ส่วนประกอบ: นม, แป้งสาลี") == "thai"
    assert detect_script("配料：牛奶，小麦") == "han"
    assert detect_script("原材料名：ミルク、こむぎ") == "japanese_kana"
    assert detect_script("состав: молоко, пшеница") == "cyrillic"
    assert detect_script("Ingredients: milk, wheat") == "latin"


def test_short_fragment_language_identification():
    # Characteristic diacritics / markers
    assert detect_language("Ingrédients: lait, arachides")[0] == "fr"
    assert detect_language("Zutaten: Weizenmehl, Milch, Zucker")[0] == "de"
    assert detect_language("Ingredientes: harina de trigo, leche")[0] == "es"
    assert detect_language("Ingredienti: farina di grano, latte")[0] == "it"
    assert detect_language("İçindekiler: buğday unu, süt")[0] == "tr"
    assert detect_language("উপকরণ: গরুর দুধ, চিনাবাদাম")[0] == "bn"
    assert detect_language("सामग्री: गेहूं का आटा, दूध")[0] == "hi"
    assert detect_language("المكونات: دقيق القمح، حليب")[0] == "ar"
    assert detect_language("اجزاء: گندم کا آٹا، دودھ")[0] == "ur"


# -----------------------------------------------------------------------------
# 5. Route A Glossary Matcher (Spaced & Unspaced Scripts)
# -----------------------------------------------------------------------------
def test_bengali_glossary_matching():
    matcher = get_glossary_matcher()
    text = "উপকরণ: গরুর দুধের গুঁড়া, চিনাবাদাম, গমের আটা, চিংড়ি মাছ।"
    hits, _ = matcher.match_full_text(text, lang="bn")
    keys = {h.canonical_key for h in hits}
    assert {"cow_milk", "peanut", "wheat", "shrimp"} <= keys
    # Verify reviewed status for Bengali terms
    bengali_hits = [h for h in hits if h.language == "bn"]
    assert any(h.reviewed for h in bengali_hits)


def test_chinese_unspaced_glossary_matching():
    matcher = get_glossary_matcher()
    # No spaces between Chinese words
    text = "配料：全脂牛奶，花生酱，小麦粉，大豆磷脂。"
    hits, _ = matcher.match_full_text(text, lang="zh-CN")
    keys = {h.canonical_key for h in hits}
    assert {"cow_milk", "peanut", "wheat", "soybean"} <= keys


def test_thai_unspaced_glossary_matching():
    matcher = get_glossary_matcher()
    text = "ส่วนประกอบ:นมถั่วลิสงแป้งสาลี"
    hits, _ = matcher.match_full_text(text, lang="th")
    keys = {h.canonical_key for h in hits}
    assert {"cow_milk", "peanut", "wheat"} <= keys


# -----------------------------------------------------------------------------
# 6. Negation ('Free From') and Trace Statement Handling
# -----------------------------------------------------------------------------
def test_multilingual_negation_not_reported_as_allergen():
    # French 'sans lait' / 'sans gluten'
    res_fr = analyze_multilingual_label("Ingrédients: sucre, huile. Sans lait. Sans gluten.", selected_language="fr")
    active_allergens = set(res_fr.allergens.keys())
    assert "cow_milk" not in active_allergens
    assert "wheat" not in active_allergens

    # German 'ohne Milch' / 'glutenfrei'
    res_de = analyze_multilingual_label("Zutaten: Zucker, Pflanzenöl. Glutenfrei. Ohne Milch.", selected_language="de")
    active_de = set(res_de.allergens.keys())
    assert "cow_milk" not in active_de
    assert "wheat" not in active_de

    # Bengali 'দুধমুক্ত'
    res_bn = analyze_multilingual_label("উপকরণ: চিনি, পাম তেল। দুধমুক্ত।", selected_language="bn")
    assert "cow_milk" not in set(res_bn.allergens.keys())


def test_multilingual_trace_statement_classified_as_trace():
    # French 'peut contenir des traces d'arachides'
    res_fr = analyze_multilingual_label("Ingrédients: sucre, blé. Peut contenir des traces d'arachides.", selected_language="fr")
    assert "peanut" in res_fr.allergens
    assert res_fr.allergens["peanut"].kind == "trace"
    assert res_fr.allergens["wheat"].kind == "direct"

    # Bengali 'চিনাবাদামের উপস্থিতি থাকতে পারে'
    res_bn = analyze_multilingual_label("উপকরণ: গমের আটা, চিনি। চিনাবাদামের উপস্থিতি থাকতে পারে।", selected_language="bn")
    assert "peanut" in res_bn.allergens
    assert res_bn.allergens["peanut"].kind == "trace"


# -----------------------------------------------------------------------------
# 7. Verification State Machine & Safety Invariant (Rule 1)
# -----------------------------------------------------------------------------
def test_unreadable_or_empty_text_forces_could_not_verify():
    # Empty
    res_empty = analyze_multilingual_label("")
    assert res_empty.verification_state == VerificationState.COULD_NOT_VERIFY
    assert res_empty.safe_verdict_allowed is False

    # Random punctuation / numbers / unreadable noise
    res_noise = analyze_multilingual_label("12345 !@#$%^&*() _+=-")
    assert res_noise.verification_state == VerificationState.COULD_NOT_VERIFY
    assert res_noise.safe_verdict_allowed is False


def test_low_ocr_confidence_forces_could_not_verify():
    # When OCR confidence is below 40%, system must refuse safe verdict
    res_low = analyze_multilingual_label("উপকরণ: দুধ, গম", ocr_confidence=0.32)
    assert res_low.verification_state == VerificationState.COULD_NOT_VERIFY
    assert res_low.safe_verdict_allowed is False
    assert "confidence too low" in res_low.state_reason.lower()


def test_verified_state_allows_safe_verdict():
    res_good = analyze_multilingual_label("উপকরণ: গরুর দুধ, চিনি, গম।", selected_language="bn", ocr_confidence=0.88)
    assert res_good.verification_state in (VerificationState.VERIFIED, VerificationState.PARTLY_VERIFIED)
    assert res_good.safe_verdict_allowed is True
    assert "cow_milk" in res_good.allergens
    assert "wheat" in res_good.allergens


# -----------------------------------------------------------------------------
# 8. Product Database Non-Latin Filter Removal (Step 9.5)
# -----------------------------------------------------------------------------
def test_product_database_accepts_non_latin_products():
    sample_off_response = {
        "products": [
            {
                "code": "894110000001",
                "product_name": "Bengali Milk Biscuit",
                "brands": "Pran",
                "unique_scans_n": 50,
                "ingredients_text": "উপকরণ: ময়দা, চিনি, গরুর দুধের গুঁড়া, পাম তেল।",
            },
            {
                "code": "622110000002",
                "product_name": "Arabic Wafer",
                "brands": "Halwani",
                "unique_scans_n": 40,
                "ingredients_text": "المكونات: دقيق القمح، سكر، زيت نباتي، حليب مجفف.",
            }
        ]
    }
    parsed = parse_off(sample_off_response)
    assert len(parsed) == 2
    codes = {p["code"] for p in parsed}
    assert "894110000001" in codes
    assert "622110000002" in codes


# -----------------------------------------------------------------------------
# 9. Multilingual Reader Manager & Script-Aware Readability
# -----------------------------------------------------------------------------
def test_multilingual_reader_manager_caching():
    mgr = MultilingualReaderManager(max_readers=2)
    assert mgr.max_readers == 2
    reader = mgr.get_reader("en")
    assert reader is not None
    assert "en" in mgr._cache


def test_multilingual_header_detection():
    # Confirmed headers in Bengali, Arabic, French, Hindi
    assert find_confirmed_header("উপকরণ: দুধ, চিনাবাদাম, চিনি") is not None
    assert find_confirmed_header("المكونات: حليب، سكر، قمح") is not None
    assert find_confirmed_header("Ingrédients: lait, sucre, blé") is not None
    assert find_confirmed_header("सामग्री: दूध, चीनी, गेहूं") is not None


# -----------------------------------------------------------------------------
# 10. Multi-Route Fusion & Task 2/3/4 Cross-Integration
# -----------------------------------------------------------------------------
def test_fused_allergens_provenance_and_additives():
    label = "উপকরণ: গরুর দুধ, চিনি, সোডিয়াম বেনজয়েট (E211), চিনাবাদাম।"
    res = analyze_multilingual_label(label, selected_language="bn", ocr_confidence=0.85)

    assert "cow_milk" in res.allergens
    assert "peanut" in res.allergens

    # Verify provenance badges
    milk_hit = res.allergens["cow_milk"]
    assert milk_hit.provenance in ("both", "glossary")
    assert milk_hit.confidence == "high"
    assert milk_hit.original_matched_term in ("দুধ", "গরুর দুধ")

    # Verify Task 3 additive hypersensitivity detection on foreign text
    additive_ids = {a.additive_id for a in res.additives}
    assert "benzoates" in additive_ids
