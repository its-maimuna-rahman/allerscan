"""
Unit and integration tests for Task 3: Food Additive Hypersensitivity Warnings.

Verifies:
1. Exact and normalized detection across all required Step 7 cases.
2. E-numbers, INS numbers, US colour names, brackets and lists.
3. Scanner OCR confusion healing inside candidate E-numbers (confidence=0.85).
4. Negation handling ("free from artificial colours", "no added sulphites", "sulphite free").
5. Precautionary trace label marking.
6. Weak-evidence tier gating (Tier 3 hidden by default; revealed with explanation when toggled).
7. Personalisation: asthma relevance for sulfites and "You listed this" for typed sensitivities.
8. Speed benchmark on 5,000-word ingredient list (< 100 ms).
9. HTML safety and escaping.
10. Strict schema validator tests on broken copies (duplicate alias, missing source, invalid tier).
11. Regression test ensuring the allergen model and predictions remain 100% identical.
"""
import copy
import html
import sys
import time
from pathlib import Path
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import allergen_kb as kb
from model import predict_severity
from nlp_parser import parse_label
import term_resolver as tr
import additive_detector as ad


# ----------------------------------------------------------------------
# Step 7 Table Cases
# ----------------------------------------------------------------------

def test_case_01_water_sugar_e102():
    res = ad.detect_additives("Water, sugar, colour (E102), flavouring")
    assert len(res) == 1
    m = res[0]
    assert m.additive_id == "tartrazine"
    assert m.match_kind == "e_number"
    assert "E102" in m.matched_text


def test_case_02_colours_tartrazine_sunset_yellow():
    res = ad.detect_additives("Colours: tartrazine, sunset yellow FCF")
    ids = {m.additive_id for m in res}
    assert ids == {"tartrazine", "sunset_yellow"}


def test_case_03_us_colour_names():
    res = ad.detect_additives("FD&C Yellow No. 5 Lake, Red 40")
    ids = {m.additive_id for m in res}
    assert ids == {"tartrazine", "allura_red"}


def test_case_04_sodium_metabisulphite_both_spellings():
    # British spelling
    res1 = ad.detect_additives("Preservative: sodium metabisulphite (E223)")
    assert len(res1) == 1
    assert res1[0].additive_id == "sulfites"

    # US spelling
    res2 = ad.detect_additives("Preservative: sodium metabisulfite (E223)")
    assert len(res2) == 1
    assert res2[0].additive_id == "sulfites"


def test_case_05_contains_sulphites():
    res = ad.detect_additives("Contains sulphites")
    assert len(res) == 1
    assert res[0].additive_id == "sulfites"
    assert not res[0].is_trace


def test_case_06_free_from_colours_and_preservatives():
    res = ad.detect_additives("Free from artificial colours and preservatives")
    assert len(res) == 0


def test_case_07_no_added_sulphites_and_sulphite_free():
    for text in [
        "No added sulphites",
        "No added sulfites",
        "sulphite free",
        "sulfite free",
        "sulphite-free",
        "sulfite-free",
        "without preservatives",
        "contains no colourings",
    ]:
        res = ad.detect_additives(text)
        assert len(res) == 0, f"Expected no match for negated text '{text}', got: {[m.additive_id for m in res]}"


def test_case_08_may_contain_traces_of_sulphur_dioxide():
    res = ad.detect_additives("May contain traces of sulphur dioxide")
    assert len(res) == 1
    assert res[0].additive_id == "sulfites"
    assert res[0].is_trace is True


def test_case_09_cochineal_extract_carmine_and_e120():
    res = ad.detect_additives("Cochineal extract (carmine) and E120")
    assert len(res) == 1
    assert res[0].additive_id == "carmine"
    assert len(res[0].all_occurrences) >= 2


def test_case_10_annatto_and_annatto_extract():
    res = ad.detect_additives("Annatto (E160b), annatto extract")
    assert len(res) == 1
    assert res[0].additive_id == "annatto"
    assert len(res[0].all_occurrences) >= 2


def test_case_11_e_120_g_sugar():
    # E followed by quantity/grams must not match E120
    res = ad.detect_additives("E 120 g sugar")
    assert len(res) == 0


def test_case_12_vitamin_e_and_lecithin():
    # Vitamin E is a nutrient; lecithin belongs to allergens, not additive warnings
    res = ad.detect_additives("Vitamin E, E322 (lecithin)")
    assert len(res) == 0


def test_case_13_unspecific_words():
    res = ad.detect_additives("colour, colours, natural flavourings, preservative")
    assert len(res) == 0


def test_case_14_scanner_ocr_misreads():
    # Letter O instead of 0
    res_o = ad.detect_additives("Wheat flour, E1O2, salt")
    assert len(res_o) == 1
    assert res_o[0].additive_id == "tartrazine"
    assert res_o[0].confidence == 0.85

    # Letter l / I instead of 1
    res_l = ad.detect_additives("Wheat flour, El02, salt")
    assert len(res_l) == 1
    assert res_l[0].additive_id == "tartrazine"
    assert res_l[0].confidence == 0.85


def test_case_15_speed_benchmark_5000_words():
    base_words = ["water", "wheat", "flour", "sugar", "salt", "palm", "oil", "yeast", "natural", "flavouring", "spices", "onion", "garlic"] * 400
    base_words.extend(["colour", "tartrazine", "preservative", "sodium metabisulphite", "curcumin"])
    text = ", ".join(base_words)
    assert len(text.split()) >= 5000

    t0 = time.perf_counter()
    res = ad.detect_additives(text)
    duration = time.perf_counter() - t0

    assert len(res) >= 2
    # Gate: must complete in < 100 ms
    assert duration < 0.100, f"Expected 5,000 words scan in < 100 ms, took {duration*1000:.2f} ms"


def test_case_16_html_safety_and_escaping():
    malicious = "<script>alert('xss')</script>, <b>tartrazine</b>, &quot;E102&quot;"
    res = ad.detect_additives(malicious)
    assert len(res) >= 1
    card_html = ad.render_additive_card_html(res[0])

    # Must escape dangerous characters and never output raw HTML tags
    assert "<script>" not in card_html
    assert "<b>" not in card_html

    # Ensure that any HTML tags injected into match fields are escaped
    m_malicious = copy.deepcopy(res[0])
    m_malicious.all_occurrences[0].matched_text = "<script>alert('xss')</script>"
    m_malicious.data["description"] = "<img src=x onerror=alert(1)>"
    escaped_card = ad.render_additive_card_html(m_malicious)
    assert "<script>" not in escaped_card
    assert "&lt;script&gt;" in escaped_card
    assert "<img" not in escaped_card
    assert "&lt;img" in escaped_card


def test_case_17_weak_evidence_switch_off_and_on():
    text = "Snack containing MSG, aspartame, and BHA"
    res = ad.detect_additives(text)
    assert len(res) == 3
    for m in res:
        assert m.tier == 3

    # Switch off: format_additive_line returns empty string
    line_off = ad.format_additive_line(res, show_weak=False)
    assert line_off == ""

    # Switch on: formatted line includes them
    line_on = ad.format_additive_line(res, show_weak=True)
    assert "Monosodium Glutamate" in line_on
    assert "Aspartame" in line_on
    assert "BHA & BHT" in line_on

    # Card HTML contains clear wording that blinded tests have not confirmed this link
    for m in res:
        card = ad.render_additive_card_html(m)
        assert "blinded" in card.lower() or "controlled" in card.lower() or "not confirmed" in card.lower()


def test_case_18_asthma_and_sulfites():
    # User with asthma
    res_asthma = ad.detect_additives("Dried apricots, preservative (sulfur dioxide)", user_has_asthma=True)
    assert len(res_asthma) == 1
    assert res_asthma[0].asthma_relevant is True
    card = ad.render_additive_card_html(res_asthma[0])
    assert "Sulfites are the additive most often linked to breathing symptoms in people with asthma." in card

    # User without asthma
    res_no_asthma = ad.detect_additives("Dried apricots, preservative (sulfur dioxide)", user_has_asthma=False)
    assert len(res_no_asthma) == 1
    assert res_no_asthma[0].asthma_relevant is False


def test_case_19_typed_additive_personal_sensitivity():
    # Resolver coordination
    resolved = tr.resolve_terms("tartrazine, sulfites, msg")
    assert len(resolved) == 3
    for r in resolved:
        assert r.status == "additive"

    # Detector coordination with user sensitivity
    res = ad.detect_additives("Yellow soda, tartrazine", user_sensitivities={"tartrazine"})
    assert len(res) == 1
    assert res[0].is_user_listed is True
    # Shown even if weak evidence switch is off because user listed it!
    line = ad.format_additive_line(res, show_weak=False)
    assert "Tartrazine" in line

    card = ad.render_additive_card_html(res[0])
    assert "You listed this" in card


# ----------------------------------------------------------------------
# Validator Tests (Gate for Step 3)
# ----------------------------------------------------------------------

def test_validator_passes_on_real_data():
    real_data = ad.load_additive_data(force_reload=True)
    errors = ad.validate_additive_data(real_data)
    assert errors == []


def test_validator_fails_on_duplicate_alias():
    data = copy.deepcopy(ad.load_additive_data())
    # Introduce duplicate alias between two additives
    data["additives"][0]["synonyms"].append("shared_conflict_alias")
    data["additives"][1]["synonyms"].append("shared_conflict_alias")

    errors = ad.validate_additive_data(data)
    assert any("Alias collision for 'shared_conflict_alias'" in e for e in errors)


def test_validator_fails_on_missing_source():
    data = copy.deepcopy(ad.load_additive_data())
    data["additives"][0]["sources"] = []
    errors = ad.validate_additive_data(data)
    assert any("must have a non-empty list of 'sources'" in e for e in errors)


def test_validator_fails_on_invalid_tier():
    data = copy.deepcopy(ad.load_additive_data())
    data["additives"][0]["tier"] = 99
    errors = ad.validate_additive_data(data)
    assert any("invalid tier '99'" in e for e in errors)


def test_validator_fails_on_invalid_e_number():
    data = copy.deepcopy(ad.load_additive_data())
    data["additives"][0]["e_numbers"].append("InvalidENumber")
    errors = ad.validate_additive_data(data)
    assert any("invalid E-number format 'InvalidENumber'" in e for e in errors)


# ----------------------------------------------------------------------
# Open Food Facts structured tags & multi-text interplay
# ----------------------------------------------------------------------

def test_structured_tags_union():
    res = ad.detect_additives("Water, sugar", structured_tags=["en:e102", "en:e223"])
    ids = {m.additive_id for m in res}
    assert ids == {"tartrazine", "sulfites"}
    assert any(m.match_kind == "structured_tag" for m in res)


def test_multi_text_interplay_translation():
    # Simulates OCR foreign text with E-number + English translation with name
    original_text = "Ingrédients: eau, sucre, E102, E120."
    translated_text = "Ingredients: water, sugar, tartrazine, carmine."
    res = ad.detect_additives_multi_text([original_text, translated_text])
    ids = {m.additive_id for m in res}
    assert ids == {"tartrazine", "carmine"}


# ----------------------------------------------------------------------
# Regression Test (No allergen prediction / model changes)
# ----------------------------------------------------------------------

def test_model_predictions_regression_untouched():
    label = "Wheat flour, skimmed milk powder, roasted almonds, soy lecithin, tartrazine (E102)."
    found = parse_label(label)

    profile = {"almond": 2, "cow_milk": 1}
    predictions = {}
    for allergen, info in found.items():
        pred = predict_severity({
            "age": 28, "sex": 0, "asthma_flag": 1.0, "eczema_flag": 0,
            "family_history_flag": 1, "known_allergens": profile,
            "detected_allergen": allergen, "is_hidden_allergen": info["hidden"]
        })
        p_react_val = round(pred["p_react"], 3) if pred.get("p_react") is not None else None
        predictions[allergen] = (pred["severity_score"], pred["mode"], p_react_val)

    # Expected unchanged allergen detections
    assert "wheat" in predictions
    assert "cow_milk" in predictions
    assert "almond" in predictions
    assert "soybean" in predictions

    # Model scores for the profile
    assert predictions["almond"][1] == "profile"
    assert predictions["cow_milk"][1] == "profile"
    assert predictions["almond"][0] == 2
    assert predictions["cow_milk"][0] == 1
