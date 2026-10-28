"""
Comprehensive Unit and Property Tests for AllerScan Spelling Correction.

Covers all test cases, invariants, and edge cases specified in Step 9 of
docs/04_spelling_correction.md.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import allergen_kb as kb
from nlp_parser import parse_label
from spelling_corrector import correct_text, suggest_name, get_candidates, get_corrector


# -----------------------------------------------------------------------------
# 1. Primary Typo Tests (Step 9 Table)
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("typo", ["mlik", "milc", "miilk"])
def test_milk_variants(typo):
    res = correct_text(typo, mode="typed")
    assert res.corrected_text.lower() == "milk"
    assert any(c.corrected.lower() == "milk" for c in res.corrections)


@pytest.mark.parametrize("typo", ["peenut", "paenut", "peanutt", "pinut"])
def test_peanut_variants(typo):
    res = correct_text(typo, mode="typed")
    assert "peanut" in res.corrected_text.lower()


@pytest.mark.parametrize("typo", ["whaet", "wheet"])
def test_wheat_variants(typo):
    res = correct_text(typo, mode="typed")
    assert "wheat" in res.corrected_text.lower()


@pytest.mark.parametrize("typo", ["glutten", "gluton"])
def test_gluten_variants(typo):
    res = correct_text(typo, mode="typed")
    assert "gluten" in res.corrected_text.lower()


@pytest.mark.parametrize("typo", ["casien", "cassein"])
def test_casein_variants(typo):
    res = correct_text(typo, mode="typed")
    assert "casein" in res.corrected_text.lower()


@pytest.mark.parametrize("typo", ["shrimpp", "srimp"])
def test_shrimp_variants(typo):
    res = correct_text(typo, mode="typed")
    assert "shrimp" in res.corrected_text.lower()


def test_cashw_possible_not_silent():
    """'cashw' must be reported as possible cashew, never applied silently."""
    res = correct_text("cashw", mode="typed")
    # Must NOT silently rewrite to cashew
    assert res.corrected_text.lower() == "cashw"
    assert len(res.corrections) == 0
    # Must be in possible allergens
    assert len(res.possible_allergens) >= 1
    top_p = res.possible_allergens[0]
    assert top_p.candidate_allergen.lower() == "cashew"
    assert top_p.canonical_key == "cashew"


@pytest.mark.parametrize("token", ["nut", "nuts"])
def test_nut_or_nuts_alone_unchanged(token):
    """'nut' or 'nuts' alone: existing behaviour unchanged."""
    res = correct_text(token, mode="typed")
    assert res.corrected_text.lower() == token
    assert len(res.corrections) == 0


@pytest.mark.parametrize("compound,expected", [
    ("wheatflour", "wheat flour"),
    ("skimmedmilk", "skimmed milk"),
    ("peanutbutter", "peanut butter"),
    ("eggwhite", "egg white"),
    ("soyabean", "soya bean"),
])
def test_segmented_run_together_words(compound, expected):
    res = correct_text(compound, mode="typed")
    assert res.corrected_text.lower() == expected


@pytest.mark.parametrize("split_input,expected", [
    ("pea nut", "peanut"),
    ("whe at", "wheat"),
])
def test_joined_split_words(split_input, expected):
    res = correct_text(split_input, mode="typed")
    assert res.corrected_text.lower() == expected


# -----------------------------------------------------------------------------
# 2. Protection and Safety Tests
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("word", ["pearl", "price", "peer", "mild", "sheet", "dish"])
def test_protected_words_unchanged(word):
    """Real English words must NEVER be changed into allergens."""
    res = correct_text(word, mode="typed")
    assert res.corrected_text.lower() == word
    assert len(res.corrections) == 0
    assert word in res.protected_tokens


def test_barely_stays_barely_with_possible_barley():
    """
    'barley' misspelled as 'barely': stays 'barely' (real word);
    if context is ingredient list, report barley as possible, not confirmed.
    """
    text = "wheat flour, sugar, barely, salt"
    res = correct_text(text, mode="typed")
    assert "barely" in res.corrected_text.lower()
    assert "barley" not in res.corrected_text.lower()
    # Must be in possible allergens
    assert any(p.canonical_key == "barley" for p in res.possible_allergens)


@pytest.mark.parametrize("code", ["E102", "E322", "30g", "12%", "250ml", "150kcal"])
def test_codes_and_units_untouched(code):
    res = correct_text(code, mode="typed")
    assert res.corrected_text == code
    assert len(res.corrections) == 0


# -----------------------------------------------------------------------------
# 3. Scanned Mode Tests
# -----------------------------------------------------------------------------
def test_scanned_mode_ocr_confusions():
    scanned_text = "Sait, rnilk, 0il, l actose"
    res = correct_text(scanned_text, mode="scanned")
    assert "salt" in res.corrected_text.lower()
    assert "milk" in res.corrected_text.lower()
    assert "oil" in res.corrected_text.lower()
    assert "lactose" in res.corrected_text.lower()


# -----------------------------------------------------------------------------
# 4. Robustness and Determinism
# -----------------------------------------------------------------------------
def test_empty_and_whitespace_input():
    assert correct_text("").corrected_text == ""
    assert correct_text("   ").corrected_text == "   "


def test_huge_input_no_crash():
    huge = "wheat flour, sugar, salt. " * 2500  # > 50,000 chars
    res = correct_text(huge)
    assert res.corrected_text == huge
    assert len(res.notes) >= 1


def test_emoji_and_non_latin_no_crash():
    text = "🥜 🥛 حليب молоко 牛奶"
    res = correct_text(text)
    assert res.corrected_text == text


def test_deterministic_output():
    text = "sugar, mlik powder, whaet flour, peenut oil"
    res1 = correct_text(text)
    res2 = correct_text(text)
    assert res1.corrected_text == res2.corrected_text
    assert [c.to_dict() for c in res1.corrections] == [c.to_dict() for c in res2.corrections]


# -----------------------------------------------------------------------------
# 5. Property-Style Invariant Tests
# -----------------------------------------------------------------------------
def test_property_already_correct_text_unchanged():
    correct = "wheat flour, skimmed milk powder, soy lecithin, hazelnuts, almonds, cocoa butter"
    res = correct_text(correct)
    assert res.corrected_text == correct
    assert len(res.corrections) == 0


def test_property_idempotency():
    """Running corrector twice gives the same result as once: f(f(x)) == f(x)."""
    text = "sugar, mlik powder, whaet flour, peenut oil, skimmedmilk"
    pass1 = correct_text(text)
    pass2 = correct_text(pass1.corrected_text)
    assert pass1.corrected_text == pass2.corrected_text


def test_property_edit_distance_bound():
    """No correction produces a word beyond allowed edit distance."""
    from rapidfuzz.distance import DamerauLevenshtein
    text = "sugar, mlik, whaet, peenut, shrimpp, allmonds"
    res = correct_text(text)
    for c in res.corrections:
        if c.kind in ("typo_fix", "scanner_heal"):
            dist = DamerauLevenshtein.distance(c.original.lower(), c.corrected.lower())
            assert dist <= 2, f"Correction {c.original} -> {c.corrected} exceeded max distance (dist={dist})"


# -----------------------------------------------------------------------------
# 6. Task 2 Name Mode Suggestions
# -----------------------------------------------------------------------------
def test_suggest_name_mode_behavior():
    suggs = suggest_name("penut")
    assert len(suggs) >= 1
    assert suggs[0]["canonical_key"] == "peanut"

    # Strict name mode does not suggest for tokens < 4 chars
    assert suggest_name("mk") == []
    assert suggest_name("egg") == []


# -----------------------------------------------------------------------------
# 7. End-to-End Section 2 Integration with NLP Parser
# -----------------------------------------------------------------------------
def test_section_2_pasted_text_reproduction():
    s2_text = "sugar, mlik powder, peenut oil, whaet flour, soya lecitin, casien, glutten, sesamee seeds, shrimpp, hazlenuts, allmonds, cashw"
    res = correct_text(s2_text, mode="typed")

    # Fixed text passed to parse_label
    found = parse_label(res.corrected_text)

    # Core allergens are now found
    expected_found = {"cow_milk", "peanut", "wheat", "soybean", "sesame_seed", "shrimp", "hazelnut", "almond"}
    assert expected_found <= set(found.keys()), f"Missing allergens: {expected_found - set(found.keys())}"

    # Cashw must be in possible allergens
    assert any(p.canonical_key == "cashew" for p in res.possible_allergens)
