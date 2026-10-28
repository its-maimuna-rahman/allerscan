"""
Unit tests for the parts of photo_scanner.py that do not need OCR.
Run:  pytest -q tests/test_scanner_units.py   (or)  python tests/test_scanner_units.py
"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2
import numpy as np

import photo_scanner as ps
from nlp_parser import parse_label


def allergens(text):
    return set(parse_label(text))


# ---------------------------------------------------------------- section extraction
def test_facility_statement_is_kept():
    out = ps.extract_clean_ingredient_section(
        "INGREDIENTS: Sugar, wheat flour, soy lecithin. Made in a facility that also processes peanuts and tree nuts.")
    assert {"peanut", "wheat", "soybean"} <= allergens(out)
    assert "facility" in out.lower()


def test_lot_and_exp_are_not_eaten_inside_words():
    out = ps.extract_clean_ingredient_section(
        "Ingredients: Water, expeller pressed canola oil, lotus seed paste, salt.")
    assert "expeller" in out.lower() and "lotus" in out.lower()


def test_contains_statement_before_ingredients_is_kept():
    out = ps.extract_clean_ingredient_section(
        "Contains: Milk, Soy. Ingredients: Sugar, cocoa butter, vanilla.")
    assert {"cow_milk", "soybean"} <= allergens(out)


def test_may_contain_is_kept_and_stays_trace():
    out = ps.extract_clean_ingredient_section(
        "Ingredients: Sugar, Hazelnuts (13%), Cocoa. Product of Italy. May contain traces of peanuts. Best before: see lid")
    info = parse_label(out)
    assert info["peanut"]["kind"] == "trace"
    assert info["hazelnut"]["kind"] == "direct"
    assert "italy" not in out.lower() and "best before" not in out.lower()


def test_nutrition_numbers_do_not_swallow_first_ingredients():
    out = ps.extract_clean_ingredient_section(
        "Nutrition Facts Serving size 30g Calories 120 Total Fat 5g Ingredients: wheat flour, sugar, milk powder")
    assert {"wheat", "cow_milk"} <= allergens(out)


def test_damaged_header_is_found():
    out = ps.extract_clean_ingredient_section("lngredients: Wheat flour, Skimmed milk powder, Yeast.")
    assert out.lower().startswith("wheat flour")


def test_wrapped_trace_statement_stays_together():
    out = ps.extract_clean_ingredient_section("Ingredients: sugar, oat flour\nMay contain traces of peanuts, tree\nnuts and sesame.")
    info = parse_label(out)
    assert info["peanut"]["kind"] == "trace" and info["sesame_seed"]["kind"] == "trace"


# ---------------------------------------------------------------- healing
REAL_WORDS = ("acidity regulator stabiliser preservative colour turmeric paprika coriander cinnamon pepper chilli "
              "vinegar maize yeast sugars salts barely bread margarine emulsifiers antioxidants thickener mango pasta "
              "noodles seasoning rapeseed sunflower oatmeal tomatoes dextrose caramel nutmeg ginger lime sweetener "
              "flavourings lactic ferment cultures enzymes rennet vitamin riboflavin thiamin niacin folic iron calcium "
              "zinc modified starch glucose syrup raising agents antioxidant ascorbic acid tocopherols polysorbate "
              "cellulose beeswax shellac stevia erythritol inulin chicory raisins sultanas currants lemon orange "
              "cherries grapes gherkin capers pickled mayonnaise ketchup marinade dressing seasoned smoked cured "
              "salted unsalted sweetened bulking gelling sulphites tartrate annatto carotene curcumin cochineal "
              "jalapeno espresso mocha vanillin cornflour wheatgerm wholemeal wholegrain emulsifier firming").split()


def test_real_ingredient_words_are_never_changed():
    changed = [(w, ps.heal_word(w)) for w in REAL_WORDS if ps.heal_word(w) != w]
    assert not changed, changed


def test_typos_are_repaired():
    cases = {"Mik": "milk", "Wney": "whey", "Lecitin": "lecithin", "mlk": "milk", "0il": "oil", "5ugar": "sugar",
             "Mi1k": "milk", "rnilk": "milk", "peanu": "peanut", "hazelnuls": "hazelnuts", "almonds": "almonds"}
    for bad, good in cases.items():
        assert ps.heal_word(bad).lower() == good, (bad, ps.heal_word(bad))


def test_accents_numbers_and_punctuation_survive():
    s = "Ingr\u00e9dients: jalape\u00f1o, cr\u00e8me fra\u00eeche, E322, B12 2.5%, 13%"
    out = ps.heal_line_tokens(s)
    for piece in ("Ingr\u00e9dients", "jalape\u00f1o", "cr\u00e8me", "fra\u00eeche", "E322", "B12", "2.5%", "13%"):
        assert piece in out, (piece, out)


def test_deglue():
    assert ps.heal_word("wheatflour").lower() == "wheat flour"
    assert ps.heal_word("cocoabutter").lower() == "cocoa butter"


def test_hyphen_across_lines():
    lines = [[ps.OCRBox([[0, 0], [90, 0], [90, 20], [0, 20]], "sugar, hazel-", 0.9)],
             [ps.OCRBox([[0, 30], [90, 30], [90, 50], [0, 50]], "nuts, salt", 0.9)]]
    assert "hazelnuts" in ps.merge_multiline_text(lines)


# ---------------------------------------------------------------- layout
def _line_box(x, y, w, text):
    return ps.OCRBox([[x, y], [x + w, y], [x + w, y + 30], [x, y + 30]], text, 0.9)


def test_two_columns_are_read_one_after_the_other():
    left = [_line_box(50, 40 * i, 500, f"L{i}") for i in range(6)]
    right = [_line_box(620, 40 * i, 500, f"R{i}") for i in range(6)]
    lines = ps.group_boxes_into_lines(left + right)
    order = [b.text for ln in lines for b in ln]
    assert order == [f"L{i}" for i in range(6)] + [f"R{i}" for i in range(6)], order


def test_single_column_not_split():
    boxes = [_line_box(50, 40 * i, 900 - 30 * (i % 3), f"T{i}") for i in range(10)]
    assert len(ps.split_columns(boxes)) == 1


# ---------------------------------------------------------------- classification / barcode / skew
def test_front_pack_with_marketing_words_is_not_an_ingredient_list():
    t = "Special K Original natural protein fiber whole grain flavor 350 g"
    assert ps.classify_scan_type(t, []) == "product_front"


def test_ingredient_detection():
    assert ps.classify_scan_type("Ingredients: sugar, milk", []) == "ingredient_list"
    assert ps.classify_scan_type("Contains: peanuts", []) == "ingredient_list"
    assert ps.classify_scan_type("sugar, wheat flour, milk powder, soy lecithin, cocoa butter, salt, egg, almonds", []) == "ingredient_list"


def test_ean_checksum():
    assert ps.ean_checksum_ok("5901234123457")
    assert not ps.ean_checksum_ok("5901234123458")
    assert not ps.ean_checksum_ok("12345")


def test_skew_recovery():
    img = np.full((900, 700), 255, np.uint8)
    for i in range(18):
        cv2.putText(img, "sugar wheat flour milk powder soy", (30, 60 + i * 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    for true in (-6.0, -2.5, 3.0, 7.5):
        m = cv2.getRotationMatrix2D((350, 450), true, 1.0)
        tilted = cv2.warpAffine(img, m, (700, 900), borderValue=255)
        corr, gain = ps.estimate_skew_deg(tilted)
        assert abs(corr + true) <= 0.8, (true, corr)
        assert gain > 1.15


def test_flat_page_is_not_straightened():
    img = np.full((900, 700), 255, np.uint8)
    for i in range(18):
        cv2.putText(img, "sugar wheat flour milk powder soy", (30, 60 + i * 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    corr, gain = ps.estimate_skew_deg(img)
    assert abs(corr) < ps.CONFIG["skew_min_deg"] or gain < ps.CONFIG["skew_min_gain"]


def test_verify_hits_rejects_wrong_product():
    hits = [{"name": "Chocolate Digestive Biscuits", "brand": "McVitie's", "text": "x", "scans": 900},
            {"name": "Nutella", "brand": "Ferrero", "text": "y", "scans": 100}]
    ok = ps.verify_hits(["Nutella Hazelnut Spread"], hits)
    assert [h["name"] for h in ok] == ["Nutella"]
    assert ps.verify_hits(["Completely Different Thing"], hits) == []


# ---------------------------------------------------------------- Step 3 regressions
def test_lost_first_letter_header():
    text = (
        "NGREDIENTS: water, corn syrup, canola oil, sodium caseinate (MILK), soy lecithin.\n"
        "INGREDIENTS\n"
        "Understanding food ingredient lists for healthier choices."
    )
    out = ps.extract_clean_ingredient_section(text)
    out_upper = out.upper()
    assert "CORN SYRUP" in out_upper
    assert "CASEINATE" in out_upper
    assert "LECITHIN" in out_upper
    assert {"cow_milk", "soybean"} <= allergens(out)


def test_stray_word_does_not_cut_off_ingredients():
    text = "sugar, wheat flour, milk powder, salt. See all INGREDIENTS on the back"
    out = ps.extract_clean_ingredient_section(text)
    assert {"wheat", "cow_milk"} <= allergens(out)


def test_heading_sentence_is_not_a_header():
    text = "Understanding food ingredient lists for healthier choices. Water, sugar, soy lecithin."
    out = ps.extract_clean_ingredient_section(text)
    assert "soybean" in allergens(out)


def test_marketing_word_classified_as_product_front():
    text = "Made with natural ingredients and real hazelnuts"
    assert ps.classify_scan_type(text, []) == "product_front"


if __name__ == "__main__":
    import sys
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name)
            except AssertionError as exc:
                failed += 1; print("FAIL", name, exc)
    sys.exit(1 if failed else 0)
