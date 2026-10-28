"""
AllerScan Spelling Vocabulary Builder.

Constructs the unified vocabulary, unigram frequencies, and bigram statistics
used by the AllerScan spelling corrector across all features.

Sources:
  1. AllerScan Allergen KB: name_mapping.json (148 canonical allergens + ~1,600 synonyms),
     plus HIDDEN_KB and PLANT_PHRASES from nlp_parser.py.
  2. AllerScan Food Additive Hypersensitivity Database: data/additive_warnings.json.
  3. Open Food Facts Ingredients Taxonomy (ODbL v1.0, https://opendatacommons.org/licenses/odbl/1.0/):
     data/openfoodfacts_ingredients.json.
  4. Packaging and food label lexicons: photo_scanner.py (COMMON_FOODS, ENGLISH_COMMON,
     LABEL_SAFE_WORDS, STRUCTURAL_VOCAB).
  5. Ground truth ingredient labels & culinary recipes: tools/scan_data/*.txt & food_fetcher.py.

Outputs:
  - data/spelling/spelling_vocab.json
  - data/spelling/frequency_dictionary.txt
  - data/spelling/bigram_dictionary.txt
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
SPELLING_DATA_DIR = DATA_DIR / "spelling"

sys.path.insert(0, str(SRC_DIR))
import allergen_kb as kb
from nlp_parser import HIDDEN_KB, PLANT_PHRASES
from photo_scanner import (
    COMMON_FOODS,
    ENGLISH_COMMON,
    FULL_TOKEN_VOCAB,
    LABEL_SAFE_WORDS,
    STEM_EXPANSIONS,
    STRUCTURAL_VOCAB,
)

# Open Food Facts attribution
ODBL_CITATION = (
    "Open Food Facts ingredients taxonomy database, licensed under the Open Database License "
    "(ODbL) v1.0 (https://opendatacommons.org/licenses/odbl/1.0/). "
    "Contributors: Open Food Facts community (https://world.openfoodfacts.org)."
)


def clean_token(s: str) -> str:
    return re.sub(r"[^a-z0-9\-']", "", s.strip().lower())


def pluralize(word: str) -> List[str]:
    """Generate standard English plurals for a food token."""
    w = word.lower().strip()
    if not w or len(w) < 3:
        return [w]
    forms = [w]
    if w.endswith("y") and len(w) > 3 and w[-2] not in "aeiou":
        forms.append(w[:-1] + "ies")
    elif w.endswith(("s", "sh", "ch", "x", "z")):
        forms.append(w + "es")
    else:
        forms.append(w + "s")
    return forms


def singularize(word: str) -> List[str]:
    """Generate standard singular form candidates."""
    w = word.lower().strip()
    cands = [w]
    if w.endswith("ies") and len(w) > 4:
        cands.append(w[:-3] + "y")
    elif w.endswith("es") and len(w) > 4:
        cands.append(w[:-2])
        cands.append(w[:-1])
    elif w.endswith("s") and len(w) > 3 and not w.endswith("ss"):
        cands.append(w[:-1])
    return cands


def build_spelling_vocabulary() -> Dict[str, Any]:
    print("Building AllerScan Spelling Vocabulary...")
    SPELLING_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Allergen Vocabulary
    # -------------------------------------------------------------------------
    name_map = kb.load_name_mapping()
    allergen_terms: Dict[str, Dict[str, Any]] = {}
    allergen_single_words: Set[str] = set()

    for canonical, synonyms in name_map.items():
        can_clean = kb.MERGE.get(canonical, canonical)
        all_terms = [canonical] + list(synonyms)
        for term in all_terms:
            t_clean = term.replace("_", " ").lower().strip()
            if not t_clean:
                continue
            is_phrase = " " in t_clean or "-" in t_clean
            allergen_terms[t_clean] = {
                "canonical": can_clean,
                "is_hidden": False,
                "is_phrase": is_phrase,
            }
            if not is_phrase:
                allergen_single_words.add(t_clean)
                for pl in pluralize(t_clean):
                    allergen_terms[pl] = {
                        "canonical": can_clean,
                        "is_hidden": False,
                        "is_phrase": False,
                    }
                    allergen_single_words.add(pl)

    # Add hidden allergens
    for key, (all_list, conf) in HIDDEN_KB.items():
        k_clean = key.replace("_", " ").lower().strip()
        is_phrase = " " in k_clean or "-" in k_clean
        allergen_terms[k_clean] = {
            "canonical": all_list[0] if len(all_list) == 1 else "derived_multiple",
            "allergens": all_list,
            "is_hidden": True,
            "is_phrase": is_phrase,
        }
        if not is_phrase:
            allergen_single_words.add(k_clean)
            for pl in pluralize(k_clean):
                allergen_terms[pl] = {
                    "canonical": all_list[0] if len(all_list) == 1 else "derived_multiple",
                    "allergens": all_list,
                    "is_hidden": True,
                    "is_phrase": False,
                }
                allergen_single_words.add(pl)

    # Add plant milk phrases
    for key, plant_target in PLANT_PHRASES.items():
        k_clean = key.replace("_", " ").lower().strip()
        allergen_terms[k_clean] = {
            "canonical": plant_target,
            "is_hidden": False,
            "is_phrase": True,
            "is_plant_phrase": True,
        }

    # -------------------------------------------------------------------------
    # 2. Additive Vocabulary (Task 3)
    # -------------------------------------------------------------------------
    additive_terms: Dict[str, Dict[str, Any]] = {}
    protected_codes: Set[str] = set()

    additive_path = DATA_DIR / "additive_warnings.json"
    if additive_path.exists():
        with open(additive_path, "r", encoding="utf-8") as f:
            add_data = json.load(f)
        for entry in add_data.get("additives", []):
            add_id = entry.get("id", "")
            disp = entry.get("display_name", "")
            for e_code in entry.get("e_numbers", []):
                protected_codes.add(e_code.upper())
                protected_codes.add(e_code.lower())
            for us_name in entry.get("us_names", []):
                u_clean = us_name.lower().strip()
                additive_terms[u_clean] = {"id": add_id, "display_name": disp}
            for syn in entry.get("synonyms", []):
                s_clean = syn.lower().strip()
                additive_terms[s_clean] = {"id": add_id, "display_name": disp}
                for pl in pluralize(s_clean):
                    additive_terms[pl] = {"id": add_id, "display_name": disp}

    # -------------------------------------------------------------------------
    # 3. Harmless Vocabulary & Label Vocabulary
    # -------------------------------------------------------------------------
    harmless_terms: Set[str] = set()

    # Add photo scanner built-ins
    for word in COMMON_FOODS | ENGLISH_COMMON | LABEL_SAFE_WORDS | STRUCTURAL_VOCAB | FULL_TOKEN_VOCAB:
        w_clean = word.lower().strip()
        if w_clean and len(w_clean) >= 2:
            harmless_terms.add(w_clean)
            for pl in pluralize(w_clean):
                harmless_terms.add(pl)

    # Add stem expansions targets
    for stem, target in STEM_EXPANSIONS.items():
        harmless_terms.add(target.lower().strip())

    # Add Open Food Facts ingredient taxonomy terms
    off_path = DATA_DIR / "openfoodfacts_ingredients.json"
    off_ingredient_count = 0
    if off_path.exists():
        with open(off_path, "r", encoding="utf-8") as f:
            off_data = json.load(f)
        for _, entry in off_data.items():
            names_to_add = []
            if "name" in entry and "en" in entry["name"]:
                names_to_add.append(entry["name"]["en"])
            if "synonyms" in entry and "en" in entry["synonyms"]:
                names_to_add.extend(entry["synonyms"]["en"])
            for name in names_to_add:
                n_clean = name.lower().strip()
                if not n_clean:
                    continue
                off_ingredient_count += 1
                harmless_terms.add(n_clean)
                for part in re.split(r"[\s,\-\(\)]+", n_clean):
                    p_clean = clean_token(part)
                    if len(p_clean) >= 3 and p_clean.isalpha():
                        harmless_terms.add(p_clean)

    # Function words and label structural tokens
    FUNCTION_WORDS = {
        "and", "or", "of", "with", "in", "from", "for", "as", "by", "on", "at", "to",
        "contains", "contain", "may", "might", "could", "can", "traces", "trace",
        "produced", "processed", "made", "manufactured", "packed", "packaged",
        "handled", "facility", "factory", "equipment", "line", "warning", "advice",
        "ingredients", "ingredient", "allergens", "allergen", "information", "product",
        "batch", "lot", "best", "before", "date", "see", "pack", "clean", "cool",
        "dry", "place", "store", "serving", "size", "per", "daily", "value", "energy",
        "calories", "fat", "saturates", "carbohydrate", "sugars", "protein", "salt",
        "sodium", "net", "weight", "volume", "grams", "kilocalories", "free", "claim",
        "organic", "natural", "certified", "pure", "origin", "directions", "use",
    }
    harmless_terms.update(FUNCTION_WORDS)

    # -------------------------------------------------------------------------
    # 4. Frequencies & Bigram Counts from Real Texts
    # -------------------------------------------------------------------------
    unigram_counts: Counter = Counter()
    bigram_counts: Counter = Counter()

    # Collect sample ingredient texts
    sample_texts: List[str] = []

    # From scan_data ground truths
    scan_txt_files = glob.glob(str(BASE_DIR / "tools" / "scan_data" / "*.txt"))
    for tf in scan_txt_files:
        try:
            with open(tf, "r", encoding="utf-8") as f:
                sample_texts.append(f.read())
        except Exception:
            pass

    # From culinary dishes in food_fetcher.py
    try:
        from food_fetcher import CULINARY_DISHES
        for dish, variants in CULINARY_DISHES.items():
            if isinstance(variants, dict):
                for v_text in variants.values():
                    sample_texts.append(v_text)
            elif isinstance(variants, str):
                sample_texts.append(variants)
    except Exception:
        pass

    # From common label templates
    LABEL_CORPUS_TEMPLATES = [
        "Sugar, wheat flour, skimmed milk powder, vegetable oil (palm), cocoa butter, soy lecithin, hazelnuts, almonds, vanilla.",
        "Water, tomato paste, onions, garlic, sea salt, olive oil, oregano, basil, black pepper, citric acid.",
        "Rolled oats, cane sugar, honey, peanut butter, roasted peanuts, cashews, sunflower seeds, sea salt.",
        "Durum wheat semolina, water, chicken egg yolk, parmesan cheese (cow milk), black pepper, salt.",
        "Rice noodles, tofu, shrimp, egg, bean sprouts, peanuts, tamarind, fish sauce, palm sugar, garlic.",
        "Dark chocolate (cocoa mass, sugar, cocoa butter, emulsifier (soy lecithin), natural vanilla flavouring).",
        "Potatoes, sunflower oil, dextrose, sea salt, maltodextrin, yeast extract, onion powder.",
        "May contain traces of peanuts, tree nuts, sesame seeds, milk, egg and soy.",
        "Manufactured on equipment that processes wheat, crustacean shellfish, fish and mustard.",
    ]
    sample_texts.extend(LABEL_CORPUS_TEMPLATES * 20)

    # Process sample texts for counts
    for stext in sample_texts:
        tokens = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", stext.lower())
        for tok in tokens:
            unigram_counts[tok] += 1
        for i in range(len(tokens) - 1):
            bg = f"{tokens[i]} {tokens[i+1]}"
            bigram_counts[bg] += 1

    # Add counts from harmless words and allergen phrases
    for w in harmless_terms:
        unigram_counts[w] += 10
    for ph in list(allergen_terms.keys()) + list(additive_terms.keys()):
        parts = ph.split()
        if len(parts) == 1:
            unigram_counts[parts[0]] += 20
        elif len(parts) == 2:
            bigram_counts[ph] += 30
            for p in parts:
                unigram_counts[p] += 15

    # Enforce FLOOR FREQUENCY for all allergen words so they are never outranked by rare harmless words
    ALLERGEN_FLOOR = 250
    for aw in allergen_single_words:
        if unigram_counts[aw] < ALLERGEN_FLOOR:
            unigram_counts[aw] = ALLERGEN_FLOOR

    # Also key allergen words and primary staple food words get higher prior
    CORE_ALLERGEN_WORDS = [
        "milk", "peanut", "peanuts", "wheat", "egg", "eggs", "soy", "soya", "soybean", "soybeans",
        "sesame", "shrimp", "shrimps", "almond", "almonds", "hazelnut", "hazelnuts", "cashew", "cashews",
        "walnut", "walnuts", "pistachio", "pistachios", "gluten", "casein", "lactose", "whey",
        "mustard", "celery", "fish", "salmon", "tuna", "cod", "barley", "oats", "rye"
    ]
    for cw in CORE_ALLERGEN_WORDS:
        unigram_counts[cw] = max(unigram_counts[cw], 1000)

    STAPLE_FOOD_WORDS = {
        "salt": 5000, "sugar": 5000, "water": 5000, "oil": 4000, "flour": 4000,
        "powder": 3000, "butter": 3000, "cream": 2000, "extract": 2000, "starch": 2000,
        "syrup": 2000, "fat": 2000, "cocoa": 2000, "yeast": 2000, "whey": 2000,
        "natural": 2000, "seeds": 1500, "lecithin": 1500, "acid": 1500, "sea": 1500,
    }
    for sw, sc in STAPLE_FOOD_WORDS.items():
        unigram_counts[sw] = max(unigram_counts[sw], sc)

    # -------------------------------------------------------------------------
    # 5. Build and Save Output Files
    # -------------------------------------------------------------------------
    vocab_data = {
        "metadata": {
            "version": "1.0.0",
            "built_at": datetime.now().isoformat(),
            "sources": {
                "allergen_kb": "AllerScan name_mapping.json & nlp_parser.py (148 canonical foods + synonyms)",
                "additive_warnings": "AllerScan data/additive_warnings.json (34 additive classes)",
                "open_food_facts": ODBL_CITATION,
                "label_lexicons": "photo_scanner.py (COMMON_FOODS, ENGLISH_COMMON, LABEL_SAFE_WORDS)",
            },
            "counts": {
                "allergen_terms": len(allergen_terms),
                "allergen_single_words": len(allergen_single_words),
                "additive_terms": len(additive_terms),
                "harmless_words": len(harmless_terms),
                "total_unigrams": len(unigram_counts),
                "total_bigrams": len(bigram_counts),
            }
        },
        "allergens": allergen_terms,
        "allergen_single_words": sorted(allergen_single_words),
        "additives": additive_terms,
        "protected_codes": sorted(protected_codes),
        "harmless_words": sorted(harmless_terms),
        "unigram_counts": dict(unigram_counts),
        "bigram_counts": dict(bigram_counts),
    }

    # Save JSON
    vocab_json_path = SPELLING_DATA_DIR / "spelling_vocab.json"
    with open(vocab_json_path, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, indent=2)
    print(f"Saved vocabulary JSON to {vocab_json_path}")

    # Save SymSpell frequency dictionary (word \t count)
    freq_dict_path = SPELLING_DATA_DIR / "frequency_dictionary.txt"
    with open(freq_dict_path, "w", encoding="utf-8") as f:
        for word, count in unigram_counts.most_common():
            if len(word) >= 2 and word.isalnum():
                f.write(f"{word}\t{count}\n")
    print(f"Saved SymSpell frequency dictionary to {freq_dict_path}")

    # Save bigram dictionary
    bigram_dict_path = SPELLING_DATA_DIR / "bigram_dictionary.txt"
    with open(bigram_dict_path, "w", encoding="utf-8") as f:
        for bg, count in bigram_counts.most_common():
            f.write(f"{bg}\t{count}\n")
    print(f"Saved bigram dictionary to {bigram_dict_path}")

    print("\n--- Vocabulary Build Summary ---")
    print(f"Allergen terms: {len(allergen_terms)}")
    print(f"Allergen single words: {len(allergen_single_words)}")
    print(f"Additive terms: {len(additive_terms)}")
    print(f"Harmless words: {len(harmless_terms)}")
    print(f"Unigram frequency entries: {len(unigram_counts)}")
    print(f"Bigram context entries: {len(bigram_counts)}")
    print(f"Licence: {ODBL_CITATION}")

    return vocab_data


if __name__ == "__main__":
    build_spelling_vocabulary()
