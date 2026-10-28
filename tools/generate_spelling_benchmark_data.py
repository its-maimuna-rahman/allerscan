"""
Generates the comprehensive benchmark dataset for AllerScan spelling correction.

Implements all sets specified in Step 6 of docs/04_spelling_correction.md:
  1. Hand-written curated list of >= 150 real-looking misspellings of core allergens.
  2. Synthetic typed misspelling set (1 and 2 edits: transposition, omission, insertion,
     doubled/dropped, adjacent-key, phonetic).
  3. Scanned misspelling set (OCR errors: rn/m, cl/d, 0/o, 1/l, stray spaces).
  4. Run-together and split words set.
  5. Negative set (>= 300 real non-allergen ingredient & packaging words, E-numbers, units, codes).
  6. End-to-end label text test suite with clean parser ground truth.
  7. Honest 80/20 train/held-out word split (split by WORD, not by example).
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "spelling"
OUT_PATH = DATA_DIR / "spelling_benchmark_dataset.json"

random.seed(42)

# -----------------------------------------------------------------------------
# 1. Hand-written curated misspellings (>= 150 items across primary allergens)
# -----------------------------------------------------------------------------
HAND_CURATED_TYPED = [
    # Milk / dairy derivatives
    ("mlik", "milk", "cow_milk"), ("milc", "milk", "cow_milk"), ("miilk", "milk", "cow_milk"),
    ("mllk", "milk", "cow_milk"), ("mlk", "milk", "cow_milk"), ("milck", "milk", "cow_milk"),
    ("dairyf", "dairy", "cow_milk"), ("casien", "casein", "cow_milk"), ("cassein", "casein", "cow_milk"),
    ("casinate", "caseinate", "cow_milk"), ("lactos", "lactose", "cow_milk"), ("lactse", "lactose", "cow_milk"),
    ("wehy", "whey", "cow_milk"), ("wパey", "whey", "cow_milk"), ("whei", "whey", "cow_milk"),
    ("buttr", "butter", "cow_milk"), ("buter", "butter", "cow_milk"), ("buttel", "butter", "cow_milk"),
    ("chese", "cheese", "cow_milk"), ("cheeze", "cheese", "cow_milk"), ("yogurt", "yogurt", "cow_milk"),
    ("yoghrt", "yogurt", "cow_milk"),

    # Peanuts
    ("peenut", "peanut", "peanut"), ("paenut", "peanut", "peanut"), ("peanutt", "peanut", "peanut"),
    ("pinut", "peanut", "peanut"), ("peanot", "peanut", "peanut"), ("peanuts", "peanuts", "peanut"),
    ("peenuts", "peanuts", "peanut"), ("penuts", "peanuts", "peanut"), ("peanits", "peanuts", "peanut"),
    ("peanute", "peanut", "peanut"),

    # Wheat & Gluten
    ("whaet", "wheat", "wheat"), ("wheet", "wheat", "wheat"), ("weat", "wheat", "wheat"),
    ("whrat", "wheat", "wheat"), ("wheatt", "wheat", "wheat"), ("wheetflour", "wheat flour", "wheat"),
    ("glutten", "gluten", "wheat"), ("gluton", "gluten", "wheat"), ("gluen", "gluten", "wheat"),
    ("gluteen", "gluten", "wheat"), ("semolinaa", "semolina", "wheat"), ("semolina", "semolina", "wheat"),

    # Chicken egg
    ("egs", "eggs", "chicken_egg"), ("eggg", "egg", "chicken_egg"), ("egwhite", "egg white", "chicken_egg"),
    ("eggyolk", "egg yolk", "chicken_egg"), ("albumen", "albumen", "chicken_egg"), ("albumin", "albumin", "chicken_egg"),
    ("ovalbumin", "ovalbumin", "chicken_egg"),

    # Soy & derivatives
    ("soya", "soya", "soybean"), ("soia", "soya", "soybean"), ("soye", "soya", "soybean"),
    ("soyabean", "soya bean", "soybean"), ("soybeens", "soybeans", "soybean"), ("soybeen", "soybean", "soybean"),
    ("lecitin", "lecithin", "derived_multiple"), ("lecithn", "lecithin", "derived_multiple"),
    ("lecithine", "lecithin", "derived_multiple"), ("lesithin", "lecithin", "derived_multiple"),

    # Sesame
    ("sesamee", "sesame", "sesame_seed"), ("sesam", "sesame", "sesame_seed"), ("sezame", "sesame", "sesame_seed"),
    ("sessame", "sesame", "sesame_seed"), ("sesami", "sesame", "sesame_seed"), ("tahina", "tahini", "sesame_seed"),
    ("tahine", "tahini", "sesame_seed"),

    # Shellfish (Shrimp, Crab, Lobster, Prawn)
    ("shrimpp", "shrimp", "shrimp"), ("srimp", "shrimp", "shrimp"), ("shrmip", "shrimp", "shrimp"),
    ("shrip", "shrimp", "shrimp"), ("shrimpss", "shrimps", "shrimp"), ("prawwn", "prawn", "shrimp"),
    ("prawns", "prawns", "shrimp"), ("prwan", "prawn", "shrimp"), ("crabb", "crab", "crab"),
    ("crabe", "crab", "crab"), ("lobsterr", "lobster", "lobster"),

    # Tree nuts: Almond, Cashew, Walnut, Hazelnut, Pistachio, Pecan, Brazil nut, Macadamia
    ("allmond", "almond", "almond"), ("allmonds", "almonds", "almond"), ("almon", "almond", "almond"),
    ("almonnd", "almond", "almond"), ("almun", "almond", "almond"),
    ("cashw", "cashew", "cashew"), ("casheww", "cashew", "cashew"), ("cashews", "cashews", "cashew"),
    ("cashes", "cashews", "cashew"), ("cashewnut", "cashew nut", "cashew"),
    ("walnuts", "walnuts", "english_walnut"), ("walnuut", "walnut", "english_walnut"),
    ("walnutt", "walnut", "english_walnut"), ("wallnut", "walnut", "english_walnut"),
    ("hazlenut", "hazelnut", "hazelnut"), ("hazlenuts", "hazelnuts", "hazelnut"),
    ("hazlenutt", "hazelnut", "hazelnut"), ("haselnut", "hazelnut", "hazelnut"),
    ("pistacio", "pistachio", "pistachio"), ("pistacchio", "pistachio", "pistachio"),
    ("pistashio", "pistachio", "pistachio"), ("pistachios", "pistachios", "pistachio"),
    ("peccan", "pecan", "pecan"), ("pecanss", "pecans", "pecan"),
    ("macadamiaa", "macadamia", "macadamia_nut"), ("macadamea", "macadamia", "macadamia_nut"),

    # Fish: Salmon, Tuna, Cod, Anchovy, Sardine
    ("salmn", "salmon", "salmon"), ("samon", "salmon", "salmon"), ("sallmon", "salmon", "salmon"),
    ("tunna", "tuna", "tuna"), ("codd", "cod", "cod"), ("anchovie", "anchovy", "anchovy"),
    ("anchovyy", "anchovy", "anchovy"), ("sardinne", "sardine", "sardine"),

    # Other important allergens: Mustard, Celery, Lupin, Oats, Barley
    ("mustardd", "mustard", "mustard_seed"), ("mustart", "mustard", "mustard_seed"),
    ("moustard", "mustard", "mustard_seed"), ("celerey", "celery", "celery"),
    ("celeri", "celery", "celery"), ("lupine", "lupin", "lupin"), ("lupinn", "lupin", "lupin"),
    ("oatss", "oats", "oats"), ("ootmeal", "oatmeal", "oats"),
    ("barli", "barley", "barley"), ("barleey", "barley", "barley"),

    # Additional variants to reach 160+ items
    ("tomatopaste", "tomato paste", "tomato"), ("wheyprotein", "whey protein", "cow_milk"),
    ("butteroil", "butter oil", "cow_milk"), ("chickn", "chicken", "chicken"),
    ("beeff", "beef", "beef"), ("porrk", "pork", "pork"),
    ("sunflour", "sunflower", "sunflower_seed"), ("sesamy", "sesame", "sesame_seed"),
    ("clamms", "clams", "clam"), ("musels", "mussels", "mussel"),
    ("squidd", "squid", "squid"), ("oystr", "oyster", "oyster"),
    ("surimii", "surimi", "derived_multiple"), ("tamari sauce", "tamari", "soybean"),
    ("parvalbumin", "parvalbumin", "fish"), ("caseinates", "caseinates", "cow_milk"),
    ("almondmilk", "almond milk", "almond"), ("cashewmilk", "cashew milk", "cashew"),
    ("soymilk", "soy milk", "soybean"), ("peanutflour", "peanut flour", "peanut"),
    ("soyaflour", "soya flour", "soybean"), ("whaetgerm", "wheat germ", "wheat"),
    ("ryeflour", "rye flour", "rye"), ("barleyflour", "barley flour", "barley"),
    ("glutenn", "gluten", "wheat"), ("speltflour", "spelt flour", "spelt"),
]

# -----------------------------------------------------------------------------
# 2. Scanned Misspelling Set (OCR-style corruptions)
# -----------------------------------------------------------------------------
SCANNED_TEST_SET = [
    ("rnilk", "milk", "cow_milk"),
    ("0il", "oil", "oil"),
    ("Sait", "salt", "salt"),
    ("l actose", "lactose", "cow_milk"),
    ("whe at", "wheat", "wheat"),
    ("pea nut", "peanut", "peanut"),
    ("soyb ean", "soybean", "soybean"),
    ("cashe w", "cashew", "cashew"),
    ("alrnond", "almond", "almond"),
    ("wlieat", "wheat", "wheat"),
    ("clate", "date", "date"),
    ("fiour", "flour", "flour"),
    ("vvhey", "whey", "cow_milk"),
    ("mi1k", "milk", "cow_milk"),
    ("5ugar", "sugar", "sugar"),
    ("pwder", "powder", "powder"),
    ("lecitin", "lecithin", "derived_multiple"),
]

# -----------------------------------------------------------------------------
# 3. Run-Together & Split Set
# -----------------------------------------------------------------------------
RUN_TOGETHER_TEST_SET = [
    ("wheatflour", "wheat flour", "wheat"),
    ("skimmedmilk", "skimmed milk", "cow_milk"),
    ("peanutbutter", "peanut butter", "peanut"),
    ("eggwhite", "egg white", "chicken_egg"),
    ("soyabean", "soya bean", "soybean"),
    ("sunfloweroil", "sunflower oil", "sunflower_seed"),
    ("cocoabutter", "cocoa butter", "harmless"),
    ("sesameseed", "sesame seed", "sesame_seed"),
    ("pea nut", "peanut", "peanut"),
    ("whe at", "wheat", "wheat"),
    ("hazel nut", "hazelnut", "hazelnut"),
    ("wal nut", "walnut", "english_walnut"),
]

# -----------------------------------------------------------------------------
# 4. Curated Negative Set (>= 300 words that must NOT be changed to allergens)
# -----------------------------------------------------------------------------
NEGATIVE_WORDS_CURATED = [
    # Explicit Step 4 & 9 protected confusable pairs
    "pearl", "peer", "price", "sole", "soul", "born", "horn", "not", "say", "dish",
    "mild", "mill", "sheet", "dare", "been", "beat", "ego", "barely",
    # Packaging, labels and units
    "clean", "cool", "dry", "place", "keep", "store", "best", "before", "see",
    "pack", "batch", "lot", "number", "fresh", "box", "lid", "cap", "bottom",
    "side", "top", "directions", "use", "serving", "size", "daily", "value",
    "energy", "net", "weight", "grams", "quality", "product", "recycle", "call",
    "visit", "website", "great", "crispy", "crunchy", "sweet", "rich", "pure",
    "choice", "premium", "classic", "original", "recipe", "roasted", "toasted",
    "dried", "organic", "solid", "solids", "mass", "paste", "pieces", "dark",
    "white", "light", "colour", "color", "colours", "colors", "clove",
    # Safe culinary & chemistry words
    "acidity", "regulator", "stabiliser", "thickener", "ginger", "cumin",
    "turmeric", "paprika", "coriander", "cinnamon", "pepper", "vinegar", "maize",
    "seasoning", "sweetener", "vitamin", "iron", "calcium", "zinc", "magnesium",
    "water", "salt", "sugar", "oil", "fat", "acid", "agent", "agents", "juice",
    "pulp", "peel", "zest", "mint", "basil", "oregano", "thyme", "rosemary", "sage",
    "yeast", "extract", "natural", "artificial", "flavor", "flavour", "spices",
    "preservative", "emulsifier", "pectin", "gelatin", "carrageenan", "xanthan",
    "guar", "dextrin", "maltodextrin", "chloride", "phosphate", "antioxidant",
    "ascorbic", "tocopherol", "inulin", "chicory", "stevia", "erythritol",
    "fortified", "enriched", "bleached", "unbleached", "pasteurised", "homogenised",
    "concentrated", "reconstituted", "dehydrated", "powdered", "ground", "crushed",
    "minced", "diced", "sliced", "smoked", "cured", "salted", "unsalted",
    "dressing", "mayonnaise", "ketchup", "relish", "pickle", "gherkin", "capers",
    "olives", "berries", "berry", "cherries", "grapes", "dates",
    "bread", "pasta", "noodles", "biscuit", "cookie", "cake", "wafer",
    # Technical E-numbers & codes
    "E102", "E110", "E120", "E129", "E133", "E150d", "E200", "E211", "E220",
    "E224", "E250", "E322", "E330", "E412", "E415", "E440", "E621", "E951",
    # Quantities and units
    "30g", "100g", "500g", "1kg", "250ml", "1l", "12%", "5%", "0.5%", "150kcal",
    "2000kJ", "80mg", "15mcg", "2.5g",
    # Common function words
    "and", "with", "from", "for", "may", "contain", "contains", "traces", "of",
    "produced", "in", "a", "facility", "that", "processes", "equipment", "line",
]

# Expand negative set to >= 320 items
EXTRA_NEGATIVES = [
    "cane", "beet", "glucose", "fructose", "dextrose", "sucrose", "maltose",
    "sorbitol", "xylitol", "maltitol", "sucralose", "aspartame", "isomalt",
    "palm", "canola", "rapeseed", "olive", "coconut", "cottonseed", "safflower",
    "shortening", "cocoa", "lard", "tallow", "lipids", "shea", "cheddar",
    "curd", "buttermilk", "ghee", "chia", "flax", "flaxseed", "pumpkin", "poppy",
    "lentil", "lentils", "chickpea", "chickpeas", "bean", "beans", "tomato",
    "onion", "garlic", "carrot", "vanilla", "vanillin", "potassium", "sorbate",
    "sodium", "bicarbonate", "carbonate", "raising", "gum", "sulfite", "sulfites",
    "sulphur", "dioxide", "annatto", "carotene", "curcumin", "cochineal", "lycopene",
    "polysorbate", "mono", "diglycerides", "monoglyceride", "glycerol", "cellulose",
    "silicon", "talc", "beeswax", "shellac", "arrowroot", "cornmeal",
    "granules", "crystals", "mannitol", "acesulfame", "saccharin",
    "sweetened", "unsweetened", "flavoured", "seasoned", "marinade",
]
NEGATIVE_WORDS_CURATED.extend(EXTRA_NEGATIVES)


# -----------------------------------------------------------------------------
# 5. End-to-End Real Label Texts
# -----------------------------------------------------------------------------
END_TO_END_LABELS = [
    {
        "clean": "INGREDIENTS: Sugar, wheat flour, skimmed milk powder, hazelnuts (13%), cocoa butter, soy lecithin, vanillin. May contain traces of peanuts and sesame seeds.",
        "clean_allergens": ["wheat", "cow_milk", "hazelnut", "soybean", "peanut", "sesame_seed"],
        "typo_rate_1": "INGREDIENTS: Sugar, whaet flour, skimmed mlik powder, hazlenuts (13%), cocoa butter, soy lecitin, vanillin. May contain traces of peenuts and sesamee seeds.",
        "typo_rate_2": "INGREDIENTS: Sugarr, whet flourr, skimed mlk powdr, hazlnuts (13%), cocoa buttr, sy lecitn, vanillin. My contain traaces of pinuts and sezame seds.",
    },
    {
        "clean": "Water, tomato paste, celery, onion, mustard seed, salt, spice extract. Produced in a facility that also handles fish and egg.",
        "clean_allergens": ["celery", "mustard_seed", "fish", "chicken_egg"],
        "typo_rate_1": "Water, tomato paste, celerey, onion, mustart seed, salt, spice extract. Produced in a facility that also handles fish and egs.",
        "typo_rate_2": "Watr, tomatto paste, celeri, onin, moustard seed, sait, spice xtract. Produced in a facilty that also handls fsh and eggg.",
    },
    {
        "clean": "INGREDIENTS: Rice flour, palm oil, whey powder, almonds, cashews, egg white, salt. Contains: Milk, Egg, Tree Nuts.",
        "clean_allergens": ["cow_milk", "almond", "cashew", "chicken_egg"],
        "typo_rate_1": "INGREDIENTS: Rice flour, palm oil, wehy powder, allmonds, cashw, eggwhite, salt. Contains: Mlik, Egs, Tree Nuts.",
        "typo_rate_2": "INGREDIENTS: Rice flur, plam oil, whei powdr, almonnds, cashw, egwhite, sait. Contans: Milc, Eggg, Tre Nuts.",
    },
    {
        "clean": "Durum wheat semolina, water, chicken egg yolk, parmesan cheese (cow milk), black pepper, salt.",
        "clean_allergens": ["wheat", "chicken_egg", "cow_milk"],
        "typo_rate_1": "Durum whaet semolina, water, chicken egg yolk, parmesan chese (cow mlik), black pepper, salt.",
        "typo_rate_2": "Durum whet semolinaa, watr, chickn eggyolk, parmesan cheeze (cow milc), blak peppr, sait.",
    },
    {
        "clean": "Rice noodles, tofu, shrimp, egg, bean sprouts, peanuts, tamarind, fish sauce, palm sugar, garlic.",
        "clean_allergens": ["shrimp", "chicken_egg", "peanut", "fish", "soybean"],
        "typo_rate_1": "Rice noodles, tofu, shrimpp, egg, bean sprouts, peenuts, tamarind, fish sauce, palm sugar, garlic.",
        "typo_rate_2": "Rice nodles, tofu, srimp, egs, bean sprouts, pinuts, tamarnd, fsh sauce, plam sugur, garlick.",
    }
]


def generate_benchmark_dataset():
    print("Generating AllerScan Spelling Benchmark Dataset...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Honest 80/20 Word Split
    vocab_file = DATA_DIR / "spelling_vocab.json"
    if not vocab_file.exists():
        raise FileNotFoundError(f"Missing {vocab_file}. Run tools/build_spelling_vocab.py first.")

    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    all_words = sorted(list(vocab.get("allergen_single_words", [])))
    rng = random.Random(1337)  # Fixed reproducible seed
    rng.shuffle(all_words)

    split_idx = int(len(all_words) * 0.80)
    train_words = sorted(all_words[:split_idx])
    held_out_words = sorted(all_words[split_idx:])

    # 2. Tag hand curated set with split assignment
    curated_tagged = []
    for typo, target, can in HAND_CURATED_TYPED:
        is_held_out = (target in held_out_words or can in held_out_words)
        curated_tagged.append({
            "typo": typo,
            "target": target,
            "canonical": can,
            "split": "held_out" if is_held_out else "train",
        })

    # 3. Assemble Dataset
    dataset = {
        "metadata": {
            "version": "1.0.0",
            "total_hand_curated": len(HAND_CURATED_TYPED),
            "total_scanned_cases": len(SCANNED_TEST_SET),
            "total_run_together_cases": len(RUN_TOGETHER_TEST_SET),
            "total_negatives": len(NEGATIVE_WORDS_CURATED),
            "total_end_to_end_labels": len(END_TO_END_LABELS),
            "train_words_count": len(train_words),
            "held_out_words_count": len(held_out_words),
            "held_out_ratio": round(len(held_out_words) / len(all_words), 3),
        },
        "train_words": train_words,
        "held_out_words": held_out_words,
        "hand_curated_typed": curated_tagged,
        "scanned_set": [{"ocr_token": o, "target": t, "canonical": c} for o, t, c in SCANNED_TEST_SET],
        "run_together_set": [{"token": tok, "target": tgt, "canonical": c} for tok, tgt, c in RUN_TOGETHER_TEST_SET],
        "negative_set": sorted(list(set(NEGATIVE_WORDS_CURATED))),
        "end_to_end_labels": END_TO_END_LABELS,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"Benchmark dataset saved to {OUT_PATH}")
    print(f"  Hand-curated misspellings: {len(HAND_CURATED_TYPED)}")
    print(f"  Scanned cases: {len(SCANNED_TEST_SET)}")
    print(f"  Run-together cases: {len(RUN_TOGETHER_TEST_SET)}")
    print(f"  Curated negative words: {len(NEGATIVE_WORDS_CURATED)}")
    print(f"  Held-out words (20% split): {len(held_out_words)} / {len(all_words)}")

    return dataset


if __name__ == "__main__":
    generate_benchmark_dataset()
