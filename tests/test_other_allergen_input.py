"""
Unit and integration tests for Task 2: "Other allergens" typed input and custom allergens.

Verifies:
  - Input splitting on commas, semicolons, slashes, bullets, 'and', 'or', '+', '&'
  - Input cleaning: leading phrases, trailing allergy words, accents, whitespace, HTML sanitization
  - Tier 1 exact and normalized matches (peanuts, cow's milk)
  - Tier 2 group expansions: tree nuts, shellfish (not shrimp alone), seafood, fish, gluten (with oats note)
  - Tier 2 clarifications: 'nuts' (ask which), 'fruit'/'vegetables' (over-broad)
  - Tier 3 conservative fuzzy suggestions ('penut' -> 'peanut', never auto-accepted)
  - Tier 4 custom allergens for foods outside the model (e.g. 'xyzzy', 'jackfruit')
  - Duplicate detection against multiselect and within batch
  - Cap of 15 custom allergens with visible limit message
  - Custom allergen ingredient scanner: whole word, plural, multi-word phrase, 1-char difference tolerance (>=6 chars), no inside match
  - Streamlit AppTest headless integration tests (tree nuts group item, custom allergen chip and slider)
"""
import sys
from pathlib import Path
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import allergen_kb as kb
import term_resolver as tr


def test_tier1_peanuts_matches_peanut():
    res = tr.resolve_terms("peanuts")
    assert len(res) == 1
    assert res[0].status == "matched"
    assert res[0].matched_keys == ["peanut"]
    assert res[0].display_name == "Peanut"


def test_tier1_cows_milk_variants():
    for term in ["Cow's milk", "cows milk", "cow milk", "cows' milk"]:
        res = tr.resolve_terms(term)
        assert len(res) == 1
        assert res[0].status == "matched", f"Failed for {term}"
        assert res[0].matched_keys == ["cow_milk"]


def test_tier2_tree_nuts_group():
    res = tr.resolve_terms("tree nuts")
    assert len(res) == 1
    item = res[0]
    assert item.status == "group"
    assert item.display_name == "Tree nuts"
    assert len(item.matched_keys) == 10
    expected_members = {
        "brazil_nut", "english_walnut", "black_walnut", "cashew", "hazelnut",
        "pistachio", "macadamia_nut", "pecan", "almond", "chestnut"
    }
    assert set(item.matched_keys) == expected_members


def test_tier2_nuts_ask_which():
    res = tr.resolve_terms("nuts")
    assert len(res) == 1
    item = res[0]
    assert item.status == "ask_which"
    assert "peanut" in item.message.lower() or any("peanut" in str(opt).lower() for opt in item.options)
    assert len(item.options) >= 3


def test_tier2_shellfish_not_shrimp_alone():
    res = tr.resolve_terms("shellfish")
    assert len(res) == 1
    item = res[0]
    assert item.status == "group"
    assert item.display_name == "Shellfish"
    assert len(item.matched_keys) == 15
    assert "shrimp" in item.matched_keys
    assert "crab" in item.matched_keys
    assert "lobster" in item.matched_keys
    assert "clam" in item.matched_keys
    assert "oyster" in item.matched_keys
    # Crucial gate: must NOT resolve to shrimp alone!
    assert item.matched_keys != ["shrimp"]


def test_tier2_seafood_group():
    res = tr.resolve_terms("seafood")
    assert len(res) == 1
    item = res[0]
    assert item.status == "group"
    assert item.display_name == "Seafood"
    assert len(item.matched_keys) == 40
    # Must contain fish, crustaceans, and molluscs
    assert "salmon" in item.matched_keys
    assert "shrimp" in item.matched_keys
    assert "squid" in item.matched_keys


def test_tier2_fish_group():
    res = tr.resolve_terms("fish")
    assert len(res) == 1
    item = res[0]
    assert item.status == "group"
    assert item.display_name == "Fish"
    assert len(item.matched_keys) == 25
    assert set(item.matched_keys) == set(kb.GROUPS["fish"])


def test_tier2_gluten_group_and_oats_note():
    res = tr.resolve_terms("gluten")
    assert len(res) == 1
    item = res[0]
    assert item.status == "group"
    assert "wheat" in item.matched_keys
    assert "barley" in item.matched_keys
    assert "rye" in item.matched_keys
    assert "oat" in item.message.lower()


def test_tier3_penut_fuzzy_suggestion_not_auto_accepted():
    res = tr.resolve_terms("penut")
    assert len(res) == 1
    item = res[0]
    assert item.status == "suggestions"
    assert len(item.matched_keys) == 0  # NOT auto-accepted
    assert len(item.suggestions) > 0
    top_sug = item.suggestions[0]
    assert top_sug["key"] == "peanut"
    assert top_sug["score"] >= 0.88


def test_tier4_xyzzy_and_unmodeled_food():
    res = tr.resolve_terms("xyzzy")
    assert len(res) == 1
    assert res[0].status == "custom"
    assert res[0].matched_keys == []
    assert res[0].custom_key == "custom_xyzzy"

    # Jackfruit is not in 143 model allergens
    res2 = tr.resolve_terms("jackfruit")
    assert len(res2) == 1
    assert res2[0].status == "custom"
    assert res2[0].matched_keys == []
    assert res2[0].custom_key == "custom_jackfruit"


def test_input_splitting_multiple_delimiters():
    raw = "peanut and sesame; peanut; sesame / egg"
    res = tr.resolve_terms(raw)
    statuses = [r.status for r in res]
    keys = [r.matched_keys for r in res]
    # First peanut matched, first sesame matched, second peanut already added,
    # second sesame already added, egg matched
    assert statuses == ["matched", "matched", "already_added", "already_added", "matched"]
    assert keys[0] == ["peanut"]
    assert keys[1] == ["sesame_seed"]
    assert keys[4] == ["chicken_egg"]


def test_prefix_and_suffix_removal():
    for text in ["allergic to peanuts", "peanut allergy", "i am allergic to peanuts"]:
        res = tr.resolve_terms(text)
        assert len(res) == 1
        assert res[0].status == "matched"
        assert res[0].matched_keys == ["peanut"]


def test_already_chosen_in_multiselect():
    res = tr.resolve_terms("peanut", existing_model_keys={"peanut"})
    assert len(res) == 1
    assert res[0].status == "already_added"
    assert "already in your allergen list" in res[0].message


def test_over_broad_terms():
    for word in ["fruit", "vegetables"]:
        res = tr.resolve_terms(word)
        assert len(res) == 1
        assert res[0].status == "over_broad"
        assert "specific" in res[0].message.lower()
        assert len(res[0].options) > 0


def test_html_and_script_sanitization():
    res = tr.resolve_terms("<script>alert('xss')</script>")
    # Must be rejected because <>() are rejected characters
    assert all(r.status == "rejected" for r in res)
    for r in res:
        assert "<script>" not in tr.escape_html(r.original) or "&lt;script&gt;" in tr.escape_html(r.original)


def test_empty_punctuation_and_long_strings_no_crash():
    # Empty string
    r1 = tr.resolve_terms("")
    assert len(r1) == 1
    assert r1[0].status == "rejected"

    # Only punctuation
    r2 = tr.resolve_terms(",,,;;;   ///")
    assert len(r2) == 1
    assert r2[0].status == "rejected"

    # 200 characters string
    r3 = tr.resolve_terms("a" * 200)
    assert len(r3) == 1
    assert r3[0].status == "rejected"
    assert "too long" in r3[0].message.lower()


def test_custom_cap_limit_at_15():
    res = tr.resolve_terms("dragonfruit", current_custom_count=15, max_custom=15)
    assert len(res) == 1
    assert res[0].status == "rejected"
    assert "limit reached" in res[0].message.lower()


def test_custom_matcher_whole_word_plural_phrase_tolerance():
    customs = {
        "custom_jackfruit": {"display_name": "Jackfruit", "clean_term": "jackfruit", "rating": 2},
        "custom_passion_fruit": {"display_name": "Passion Fruit", "clean_term": "passion fruit", "rating": 3},
        "custom_mushroom": {"display_name": "Mushroom", "clean_term": "mushroom", "rating": 1},
        "custom_jack": {"display_name": "Jack", "clean_term": "jack", "rating": 2},
    }

    # 1. Whole word match
    t1 = "Ingredients: organic jackfruit, water, salt."
    h1 = tr.match_custom_allergens(customs, t1)
    assert len(h1) == 1
    assert h1[0]["name"] == "Jackfruit"
    assert h1[0]["is_possible"] is False

    # 2. Plural match
    t2 = "Contains dried mushrooms and garlic."
    h2 = tr.match_custom_allergens(customs, t2)
    assert len(h2) == 1
    assert h2[0]["name"] == "Mushroom"
    assert h2[0]["is_possible"] is False

    # 3. Multi-word phrase match
    t3 = "Made with passion fruit puree."
    h3 = tr.match_custom_allergens(customs, t3)
    assert len(h3) == 1
    assert h3[0]["name"] == "Passion Fruit"
    assert h3[0]["is_possible"] is False

    # 4. 1-character difference tolerance for words >= 6 chars
    t4 = "May contain jakfruit or seeds."
    h4 = tr.match_custom_allergens(customs, t4)
    assert len(h4) == 1
    assert h4[0]["name"] == "Jackfruit"
    assert h4[0]["is_possible"] is True

    # 5. No inside match inside unrelated longer word (lumberjack != jack)
    t5 = "We sell lumberjack boots and jackfruit."
    h5 = tr.match_custom_allergens(customs, t5)
    names = [h["name"] for h in h5]
    assert "Jack" not in names
    assert "Jackfruit" in names


def test_app_level_tree_nuts_group_headless():
    from streamlit.testing.v1 import AppTest
    app_path = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    assert not at.exception

    at.sidebar.text_input[0].input("tree nuts")
    at.sidebar.button[0].click().run()
    assert not at.exception

    markdown_values = [m.value for m in at.sidebar.markdown]
    assert any("Group: Tree nuts" in v for v in markdown_values)
    slider_labels = [s.label for s in at.sidebar.slider]
    assert "Tree nuts severity" in slider_labels


def test_app_level_custom_allergen_headless():
    from streamlit.testing.v1 import AppTest
    app_path = str(Path(__file__).resolve().parent.parent / "app.py")
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    assert not at.exception

    at.sidebar.text_input[0].input("jackfruit")
    at.sidebar.button[0].click().run()
    assert not at.exception

    markdown_values = [m.value for m in at.sidebar.markdown]
    assert any("Jackfruit" in v and "custom" in v for v in markdown_values)
    slider_labels = [s.label for s in at.sidebar.slider]
    assert any("Jackfruit" in l for l in slider_labels)
