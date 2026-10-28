"""
Shared allergen knowledge base for AllerScan.

One place for everything that must stay identical between data generation,
cleaning, training, inference and the NLP parser:

  * name normalisation (synonym -> canonical allergen key)
  * allergen group (peanut, tree_nut, fish, ...)
  * protein-family flags per allergen (deterministic, literature-based mapping)

The protein mapping is a simplified educational mapping of well-known allergen
families (e.g. Ara h 1/3 = vicilin/legumin, tropomyosin in shellfish,
parvalbumin in fish, Bet v 1 homologues in pollen-related fruit). It is not an
exhaustive allergen database.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).parent

# --------------------------------------------------------------------------
# Name normalisation
# --------------------------------------------------------------------------
# Keys in name_mapping.json that are duplicates / umbrella terms of another key.
# They are merged so the model sees one level per real allergen.
MERGE = {
    "dairy": "cow_milk",
    "prawn": "shrimp",
    "walnut": "english_walnut",
    "nectarine": "peach",
    "crustacean": "shrimp",   # umbrella term; tropomyosin proxy
}


def term_key(text: str) -> str:
    """lower-case, drop apostrophes, spaces/hyphens -> underscore."""
    s = str(text).strip().lower().replace("'", "")
    s = re.sub(r"[\s\-]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


@lru_cache(maxsize=1)
def load_name_mapping() -> dict:
    path = BASE_DIR / "name_mapping.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


@lru_cache(maxsize=1)
def reverse_map() -> dict:
    """term_key(any synonym) -> canonical key. Canonical keys always win a collision."""
    mapping = load_name_mapping()
    rev = {}
    for canonical, synonyms in mapping.items():
        for s in synonyms:
            rev.setdefault(term_key(s), canonical)       # first synonym owner wins
    for canonical in mapping:
        rev[term_key(canonical)] = canonical             # canonical beats synonyms
    return {k: MERGE.get(v, v) for k, v in rev.items()}


def canonicalize(term: str):
    """Return the canonical allergen key, or None if unknown."""
    return reverse_map().get(term_key(term))


def all_allergens() -> list:
    """Sorted canonical allergen keys (after merging). This is the model's category list."""
    return sorted({MERGE.get(k, k) for k in load_name_mapping()})


def pretty(key: str) -> str:
    return str(key).replace("_", " ").title()


# --------------------------------------------------------------------------
# Groups
# --------------------------------------------------------------------------
GROUPS = {
    "peanut": ["peanut"],
    "tree_nut": ["brazil_nut", "english_walnut", "black_walnut", "cashew", "hazelnut",
                 "pistachio", "macadamia_nut", "pecan", "almond", "chestnut"],
    "seed": ["sesame_seed", "mustard_seed", "sunflower_seed", "pumpkin_seed", "hemp_seed",
             "perilla_seed", "buckwheat"],
    "legume": ["soybean", "mung_bean", "chickpea", "lupin", "lentil", "pea", "green_bean"],
    "cereal": ["wheat", "barley", "rye", "spelt", "kamut", "triticale", "corn", "sorghum",
               "millet", "oats", "rice", "quinoa"],
    "meat": ["beef", "pork", "chicken", "lamb", "horse_meat", "turkey", "frog_legs"],
    "milk": ["cow_milk", "sheep_milk", "goat_milk", "buffalo_milk", "yak_milk", "camel_milk",
             "mare_milk", "donkey_milk"],
    "egg": ["chicken_egg"],
    "fish": ["ray", "shark", "cod", "salmon", "tuna", "carp", "tilapia", "trout", "mackerel",
             "sardine", "anchovy", "herring", "halibut", "flounder", "sole", "haddock", "pollock",
             "catfish", "perch", "bass", "swordfish", "snapper", "grouper", "mahi_mahi", "eel"],
    "crustacean": ["shrimp", "crab", "lobster", "crayfish", "krill"],
    "mollusc": ["squid", "octopus", "cuttlefish", "clam", "oyster", "mussel", "scallop", "snail",
                "abalone", "whelk"],
    "insect_other": ["cockroach", "dust_mite", "cricket", "mealworm", "silk_moth_larvae",
                     "anisakis_simplex"],
    "fruit": ["peach", "apple", "pear", "plum", "cherry", "apricot", "strawberry", "raspberry",
              "pomegranate", "grape", "orange", "lemon", "mandarin", "kiwi", "banana", "melon",
              "pineapple", "date_palm", "fig", "avocado", "mango", "papaya", "passion_fruit",
              "cherimoya", "olive"],
    "vegetable": ["tomato", "eggplant", "celery", "cabbage", "lettuce", "onion", "garlic",
                  "asparagus", "carrot", "potato", "parsnip", "parsley", "fennel", "spinach",
                  "bell_pepper"],
    "other_plant": ["coconut", "palm_oil", "ragweed", "mugwort"],
}
GROUP_NAMES = sorted(GROUPS)
_KEY_TO_GROUP = {k: g for g, keys in GROUPS.items() for k in keys}


def group_of(key: str) -> str:
    return _KEY_TO_GROUP.get(key, "other_plant")


# --------------------------------------------------------------------------
# Protein families (columns of the training schema)
# --------------------------------------------------------------------------
PROTEIN_COLS = [
    "albumins", "caseins", "cupins", "legumins", "parvalbumins", "prolamins",
    "seed_storage_proteins", "tropomyosins", "vicilins", "wheys",
    "pr10_bet_v1_homologue", "class_i_chitinase", "thaumatin_like_protein_pr5",
    "actinidin_cysteine_protease", "profilin",
]

_STORAGE = ["cupins", "legumins", "vicilins", "seed_storage_proteins"]
_GROUP_PROTEINS = {
    "peanut": ["albumins", "prolamins", "pr10_bet_v1_homologue", "profilin"] + _STORAGE,
    "tree_nut": ["albumins", "prolamins"] + _STORAGE,
    "seed": ["albumins", "prolamins"] + _STORAGE,
    "legume": list(_STORAGE),
    "cereal": ["prolamins", "albumins"],
    "meat": ["albumins"],
    "milk": ["caseins", "wheys"],
    "egg": ["albumins"],
    "fish": ["parvalbumins"],
    "crustacean": ["tropomyosins"],
    "mollusc": ["tropomyosins"],
    "insect_other": ["tropomyosins"],
    "fruit": ["pr10_bet_v1_homologue", "profilin"],
    "vegetable": ["pr10_bet_v1_homologue", "profilin"],
    "other_plant": ["profilin"],
}
_PROTEIN_OVERRIDE = {
    "hazelnut": ["pr10_bet_v1_homologue", "profilin", "albumins", "prolamins"] + _STORAGE,
    "almond": ["pr10_bet_v1_homologue", "prolamins", "legumins", "seed_storage_proteins"],
    "chestnut": ["class_i_chitinase", "profilin"],
    "macadamia_nut": ["vicilins", "legumins", "cupins", "seed_storage_proteins"],
    "soybean": ["albumins", "pr10_bet_v1_homologue", "profilin"] + _STORAGE,
    "mung_bean": ["pr10_bet_v1_homologue"],
    "lupin": ["cupins", "vicilins", "legumins"],
    "pea": ["albumins"] + _STORAGE,
    "lentil": ["albumins"] + _STORAGE,
    "buckwheat": ["albumins", "vicilins", "cupins", "seed_storage_proteins"],
    "frog_legs": ["parvalbumins"],
    "anisakis_simplex": ["tropomyosins"],
    "kiwi": ["actinidin_cysteine_protease", "thaumatin_like_protein_pr5",
             "pr10_bet_v1_homologue", "profilin"],
    "apple": ["pr10_bet_v1_homologue", "profilin", "thaumatin_like_protein_pr5", "prolamins"],
    "peach": ["pr10_bet_v1_homologue", "profilin", "thaumatin_like_protein_pr5", "prolamins"],
    "cherry": ["pr10_bet_v1_homologue", "profilin", "thaumatin_like_protein_pr5"],
    "banana": ["class_i_chitinase", "profilin", "thaumatin_like_protein_pr5"],
    "avocado": ["class_i_chitinase", "profilin"],
    "papaya": ["actinidin_cysteine_protease", "class_i_chitinase"],
    "pineapple": ["actinidin_cysteine_protease", "profilin"],
    "grape": ["thaumatin_like_protein_pr5", "prolamins", "class_i_chitinase"],
    "melon": ["profilin", "thaumatin_like_protein_pr5"],
    "orange": ["profilin", "prolamins"],
    "tomato": ["profilin", "pr10_bet_v1_homologue", "prolamins"],
    "carrot": ["pr10_bet_v1_homologue", "profilin", "thaumatin_like_protein_pr5"],
    "lettuce": ["prolamins"],
    "cabbage": ["prolamins"],
    "mugwort": ["profilin", "prolamins"],
    "coconut": ["legumins", "vicilins", "cupins"],
    "palm_oil": [],
}


def proteins_of(key: str) -> list:
    return _PROTEIN_OVERRIDE.get(key, _GROUP_PROTEINS[group_of(key)])


def protein_flags(key: str) -> dict:
    p = set(proteins_of(key))
    return {c: int(c in p) for c in PROTEIN_COLS}


# --------------------------------------------------------------------------
# Epidemiology assumptions used ONLY by generate_dataset.py
# (documented in that file; kept here so allergen lists live in one place)
# --------------------------------------------------------------------------
# share of reaction records (relative weights, normalised later)
RECORD_WEIGHT = {
    "peanut": 14.0, "cow_milk": 13.0, "sheep_milk": .25, "goat_milk": .6, "buffalo_milk": .05,
    "camel_milk": .03, "yak_milk": .01, "mare_milk": .01, "donkey_milk": .01,
    "chicken_egg": 6.0,
    "shrimp": 8.5, "crab": 3.2, "lobster": 2.6, "crayfish": .5, "krill": .1,
    "squid": .6, "octopus": .25, "cuttlefish": .05, "clam": .6, "oyster": .7, "mussel": .6,
    "scallop": .5, "snail": .1, "abalone": .05, "whelk": .03,
    "almond": 2.0, "english_walnut": 2.2, "cashew": 1.8, "hazelnut": 1.2, "pistachio": .7,
    "pecan": .6, "brazil_nut": .2, "macadamia_nut": .2, "black_walnut": .1, "chestnut": .3,
    "salmon": 1.5, "tuna": 1.2, "cod": 1.2, "tilapia": .4, "trout": .3, "mackerel": .3,
    "sardine": .3, "anchovy": .3, "herring": .2, "halibut": .2, "haddock": .15, "pollock": .15,
    "catfish": .2, "flounder": .1, "sole": .1, "bass": .1, "snapper": .1, "grouper": .05,
    "swordfish": .05, "mahi_mahi": .05, "perch": .05, "carp": .08, "eel": .08, "shark": .02,
    "ray": .02,
    "wheat": 4.5, "barley": .15, "rye": .1, "spelt": .05, "kamut": .02, "triticale": .01,
    "corn": .5, "sorghum": .03, "millet": .05, "oats": .4, "rice": .3, "quinoa": .1,
    "soybean": 3.5, "chickpea": .5, "lentil": .5, "pea": .5, "lupin": .3, "mung_bean": .2,
    "green_bean": .2,
    "sesame_seed": 2.0, "mustard_seed": .5, "sunflower_seed": .4, "buckwheat": .2,
    "pumpkin_seed": .1, "hemp_seed": .05, "perilla_seed": .05,
    "beef": 1.0, "pork": .7, "chicken": .6, "lamb": .3, "turkey": .2, "horse_meat": .05,
    "frog_legs": .02,
    "apple": 1.0, "peach": .95, "kiwi": .9, "banana": .6, "strawberry": .5, "melon": .4,
    "cherry": .3, "pear": .3, "orange": .3, "avocado": .3, "mango": .25, "apricot": .2,
    "plum": .15, "pineapple": .2, "papaya": .1, "grape": .1, "lemon": .1, "mandarin": .1,
    "raspberry": .1, "fig": .1, "pomegranate": .05, "date_palm": .05, "passion_fruit": .05,
    "cherimoya": .02, "olive": .05,
    "celery": 1.0, "carrot": .8, "tomato": .8, "potato": .5, "onion": .3, "garlic": .3,
    "spinach": .2, "lettuce": .2, "cabbage": .2, "bell_pepper": .2, "eggplant": .1,
    "asparagus": .1, "parsley": .1, "parsnip": .05, "fennel": .1,
    "dust_mite": .3, "cockroach": .1, "cricket": .15, "mealworm": .1, "silk_moth_larvae": .05,
    "anisakis_simplex": .1,
    "coconut": .2, "palm_oil": .05, "ragweed": .1, "mugwort": .1,
}

# latent severity shift per allergen (higher = more severe reactions)
_GROUP_PROP = {
    "peanut": 1.1, "tree_nut": 0.8, "seed": 0.4, "legume": 0.0, "cereal": -0.4, "meat": -0.3,
    "milk": 0.15, "egg": -0.2, "fish": 0.7, "crustacean": 0.9, "mollusc": 0.6,
    "insect_other": 0.3, "fruit": -1.9, "vegetable": -1.7, "other_plant": -1.0,
}
_PROP_OVERRIDE = {
    "cashew": 1.0, "english_walnut": 0.9, "pistachio": 1.0, "hazelnut": 0.2, "almond": 0.3,
    "chestnut": -0.3, "sesame_seed": 1.0, "mustard_seed": 0.8, "buckwheat": 0.8,
    "soybean": -0.2, "chickpea": 0.3, "lentil": 0.3, "lupin": 0.8, "green_bean": -0.5,
    "wheat": 0.2, "corn": 0.1, "oats": -0.5, "rice": -0.3, "beef": 0.0, "pork": -0.1,
    "chicken": -0.6, "camel_milk": -0.2, "cricket": 0.5, "mealworm": 0.5, "dust_mite": 0.0,
    "kiwi": -1.0, "banana": -1.3, "avocado": -1.3, "peach": -1.2, "strawberry": -1.4,
    "mango": -1.2, "papaya": -1.2, "pineapple": -1.6, "apple": -1.7, "celery": -1.0,
    "carrot": -1.5, "tomato": -1.6, "potato": -1.6, "coconut": -0.6, "palm_oil": -0.8,
}
# probability a reaction was GI-predominant (severity 3 in the app scale)
_GROUP_GI = {"milk": 0.15, "egg": 0.10, "cereal": 0.10, "legume": 0.08, "meat": 0.30,
             "fish": 0.06, "crustacean": 0.06, "mollusc": 0.10, "fruit": 0.04, "vegetable": 0.04}
_GI_OVERRIDE = {"soybean": 0.12, "beef": 0.35, "pork": 0.35, "lamb": 0.35}
# probability the exposure came through a hidden / unexpected source
_GROUP_HIDDEN = {"milk": 0.20, "egg": 0.14, "cereal": 0.12, "legume": 0.08, "seed": 0.14,
                 "peanut": 0.06, "tree_nut": 0.07, "fish": 0.07, "crustacean": 0.05,
                 "mollusc": 0.05, "meat": 0.08, "fruit": 0.03, "vegetable": 0.04}
_HIDDEN_OVERRIDE = {"soybean": 0.18, "wheat": 0.14, "mustard_seed": 0.20, "celery": 0.14,
                    "sesame_seed": 0.15}


def severity_shift(key: str) -> float:
    return _PROP_OVERRIDE.get(key, _GROUP_PROP[group_of(key)])


def gi_prob(key: str) -> float:
    return _GI_OVERRIDE.get(key, _GROUP_GI.get(group_of(key), 0.03))


def hidden_prob(key: str) -> float:
    return _HIDDEN_OVERRIDE.get(key, _GROUP_HIDDEN.get(group_of(key), 0.06))


# co-sensitisation clusters (people allergic to one often react to the others)
CLUSTERS = [
    {"peanut", "almond", "english_walnut", "cashew", "hazelnut", "pistachio", "pecan",
     "brazil_nut", "macadamia_nut", "black_walnut", "lupin", "soybean", "sesame_seed"},
    {"cow_milk", "goat_milk", "sheep_milk", "chicken_egg", "wheat"},
    {"shrimp", "crab", "lobster", "crayfish", "krill", "squid", "oyster", "clam", "mussel",
     "scallop", "octopus", "dust_mite", "cockroach", "cricket", "mealworm"},
    {"salmon", "tuna", "cod", "tilapia", "trout", "mackerel", "sardine", "anchovy", "herring",
     "halibut", "haddock", "pollock", "catfish"},
    {"apple", "peach", "kiwi", "cherry", "pear", "apricot", "plum", "celery", "carrot",
     "tomato", "banana", "avocado", "melon", "strawberry", "mango", "hazelnut", "almond"},
    {"beef", "pork", "lamb", "cow_milk"},
]

# age profile per group: P(age bin | allergen); bins below
AGE_BINS = [(0, 5), (6, 12), (13, 17), (18, 29), (30, 59), (60, 89)]
_AGE_PROFILES = {
    "early":   [0.34, 0.25, 0.07, 0.08, 0.20, 0.06],   # milk / egg / wheat / soy
    "peanut":  [0.27, 0.25, 0.13, 0.14, 0.17, 0.04],
    "nut":     [0.12, 0.18, 0.12, 0.20, 0.30, 0.08],
    "adult":   [0.02, 0.04, 0.05, 0.18, 0.52, 0.19],   # shellfish, fish, meat
    "pollen":  [0.02, 0.04, 0.12, 0.30, 0.44, 0.08],
    "general": [0.08, 0.10, 0.08, 0.20, 0.42, 0.12],
}
_GROUP_AGE = {"milk": "early", "egg": "early", "cereal": "early", "legume": "early",
              "peanut": "peanut", "tree_nut": "nut", "seed": "peanut",
              "crustacean": "adult", "mollusc": "adult", "fish": "adult", "meat": "adult",
              "fruit": "pollen", "vegetable": "pollen"}


def age_profile(key: str) -> list:
    return _AGE_PROFILES[_GROUP_AGE.get(group_of(key), "general")]


if __name__ == "__main__":
    ks = all_allergens()
    print(len(ks), "canonical allergens;", len(RECORD_WEIGHT), "with record weights")
    missing = [k for k in ks if k not in RECORD_WEIGHT]
    print("no weight (will get a tiny default):", missing)
    print("not in a group:", [k for k in ks if k not in _KEY_TO_GROUP])
    print(canonicalize("Peanuts"), canonicalize("Groundnut"), canonicalize("sheep's milk"),
          canonicalize("DAIRY"), canonicalize("prawns"))
