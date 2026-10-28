"""
Cross-reactivity prior used by the data generator and by the feature builder.

P[a, b]  = probability that a person known to be allergic to `a` also reacts to `b`.
W[a, b]  = how strongly the SEVERITY of the two reactions moves together in one person
           (0 = independent, 1 = same severity). Derived from P: closely related foods
           share proteins, so a mild almond reaction implies a mild cashew reaction.

Where the numbers come from (clinical cross-reactivity ranges, several independent sources):
  * Review table of clinical cross-reactivity rates (PMC11250430, Table 2): milk -> goat/sheep/
    buffalo >90%, mare milk ~4%, beef ~10%; peanut -> tree nuts 30-40%, other legumes ~5%, lupin
    4-29%; pistachio-allergic -> cashew >97%, cashew -> pistachio 34-83%; pecan -> walnut >97%,
    walnut -> pecan 75-91%; fish -> other fish ~50%; wheat -> rye/barley ~20%, rice/oat/corn <5%;
    Rosaceae fruits ~50%; latex-fruit 30-50%.
  * Tree-nut grouping by IgE inhibition (Ann Allergy 2005): walnut/pecan/hazelnut and
    hazelnut/cashew/Brazil nut/pistachio/almond; strongest links follow botanical families.
  * Cashew-allergic Singapore children: other tree nuts 35-45%.
  * Tropomyosin family: shrimp/crab/lobster/mollusc/dust mite/cockroach/insects.
  * Birch pollen -> oral allergy syndrome in 50-70% (CMAJ), Rosaceae/Apiaceae/hazelnut cluster.
Numbers not covered by those sources are modelling assumptions and are marked ASSUMED.
Replace this file with measured co-allergy rates when real labelled data exists.
"""
from functools import lru_cache

import numpy as np

import allergen_kb as kb

KEYS = kb.all_allergens()
IDX = {k: i for i, k in enumerate(KEYS)}
K = len(KEYS)
OVERALL_PREVALENCE = 0.10          # overall food-allergy prevalence used for baseline reactivity

SAME_GROUP = {                     # default between two members of one food group
    "tree_nut": 0.35, "seed": 0.12, "legume": 0.07, "cereal": 0.06, "meat": 0.15,
    "milk": 0.60, "fish": 0.50, "crustacean": 0.75, "mollusc": 0.40, "insect_other": 0.20,
    "fruit": 0.08, "vegetable": 0.08, "other_plant": 0.05,
}

PR10 = {"apple", "peach", "cherry", "pear", "apricot", "plum", "kiwi", "celery", "carrot",
        "hazelnut", "almond", "soybean", "peanut", "strawberry"}
ROSACEAE = {"apple", "peach", "pear", "plum", "cherry", "apricot", "strawberry", "raspberry",
            "almond"}
APIACEAE = {"celery", "carrot", "parsley", "parsnip", "fennel"}
LATEX_FRUIT = {"banana", "avocado", "kiwi", "papaya", "chestnut", "mango"}
RED_MEAT = {"beef", "pork", "lamb", "horse_meat"}

# (from, to, p, symmetric). Group names or allergen keys. Later rules override earlier ones.
RULES = [
    (PR10, PR10, 0.15, True),                                              # birch/Bet v 1 cluster (weak)
    (ROSACEAE, ROSACEAE, 0.45, True),                                      # ~50% (review table)
    (APIACEAE, APIACEAE, 0.40, True),                                      # ASSUMED, celery-carrot
    ({"hazelnut"}, ROSACEAE | APIACEAE, 0.25, True),                       # pollen-food, ASSUMED
    (LATEX_FRUIT, LATEX_FRUIT, 0.35, True),                                # latex-fruit 30-50%
    ({"melon"}, {"banana"}, 0.30, True),
    ({"mugwort"}, APIACEAE | {"mustard_seed", "sunflower_seed"}, 0.35, False),   # ASSUMED
    ({"ragweed"}, {"melon", "banana"}, 0.35, False),                       # ASSUMED
    ({"tomato", "potato", "eggplant", "bell_pepper"}, {"tomato", "potato", "eggplant", "bell_pepper"}, 0.25, True),
    ({"onion", "garlic"}, {"onion", "garlic"}, 0.45, True),                # ASSUMED
    # legumes / peanut
    ({"peanut"}, {"lentil", "pea", "chickpea", "soybean", "green_bean", "mung_bean"}, 0.05, False),
    ({"soybean"}, {"peanut"}, 0.05, False),
    ({"lentil"}, {"chickpea", "pea"}, 0.80, True),                         # 81%
    ({"peanut"}, {"lupin"}, 0.15, False),                                  # 4-29%
    ({"lupin"}, {"peanut"}, 0.30, False),
    # tree nuts
    ({"peanut"}, "tree_nut", 0.12, False),                                 # 30-40% for any nut, ~12% per nut
    ("tree_nut", {"peanut"}, 0.25, False),                                 # 20-60%
    ({"english_walnut", "black_walnut", "pecan", "hazelnut"}, {"english_walnut", "black_walnut", "pecan", "hazelnut"}, 0.45, True),
    ({"cashew", "pistachio", "brazil_nut", "almond", "hazelnut"}, {"cashew", "pistachio", "brazil_nut", "almond", "hazelnut"}, 0.35, True),
    ({"cashew"}, {"pistachio"}, 0.60, False),                              # 34-83%
    ({"pistachio"}, {"cashew"}, 0.97, False),                              # >97%
    ({"english_walnut", "black_walnut"}, {"pecan"}, 0.83, False),          # 75-91%
    ({"pecan"}, {"english_walnut", "black_walnut"}, 0.97, False),          # >97%
    ({"english_walnut"}, {"black_walnut"}, 0.90, True),
    (("peanut", "tree_nut"), {"sesame_seed"}, 0.15, False),                # ASSUMED
    ({"sesame_seed"}, ("peanut", "tree_nut"), 0.20, False),                # ASSUMED
    # milk / egg / meat
    ({"cow_milk"}, {"goat_milk", "sheep_milk", "buffalo_milk"}, 0.92, False),   # >90%
    ({"goat_milk", "sheep_milk", "buffalo_milk"}, {"cow_milk", "goat_milk", "sheep_milk", "buffalo_milk"}, 0.95, True),
    ({"cow_milk"}, {"mare_milk", "donkey_milk", "camel_milk"}, 0.06, False),    # ~4%
    ({"cow_milk"}, {"yak_milk"}, 0.85, False),                             # bovine, ASSUMED
    ({"cow_milk"}, {"beef"}, 0.10, False),                                 # ~10%
    ({"beef"}, {"cow_milk"}, 0.45, False),                                 # ASSUMED (reverse is higher)
    (RED_MEAT, RED_MEAT, 0.50, True),                                      # alpha-gal pattern, ASSUMED
    ({"chicken", "turkey"}, {"chicken", "turkey"}, 0.50, True),            # ASSUMED
    ({"chicken_egg"}, {"chicken", "turkey"}, 0.04, False),                 # bird-egg syndrome, rare
    ({"chicken"}, {"chicken_egg"}, 0.12, False),
    # cereals
    ({"wheat"}, {"barley", "rye"}, 0.20, False),                           # ~20%
    ({"barley", "rye"}, {"wheat"}, 0.50, False),                           # ASSUMED
    ({"barley"}, {"rye"}, 0.50, True),
    ({"wheat"}, {"spelt", "kamut", "triticale"}, 0.80, True),              # close relatives, ASSUMED
    ({"wheat"}, {"oats", "rice", "corn", "millet", "sorghum", "quinoa"}, 0.04, False),   # <5%
    # tropomyosin family and fish
    ("crustacean", "mollusc", 0.20, False),
    ("mollusc", "crustacean", 0.30, False),
    ("crustacean", {"dust_mite", "cockroach"}, 0.12, False),
    ({"dust_mite", "cockroach"}, "crustacean", 0.15, False),
    ("crustacean", {"cricket", "mealworm", "silk_moth_larvae"}, 0.30, False),
    ({"cricket", "mealworm", "silk_moth_larvae"}, "crustacean", 0.40, False),
    ("fish", {"frog_legs"}, 0.15, False),
]


def expand(spec) -> set:
    if isinstance(spec, str):
        spec = (spec,)
    out = set()
    for s in spec:
        if s in kb.GROUPS:
            out |= {k for k in kb.GROUPS[s] if k in IDX}
        elif s in IDX:
            out.add(s)
        else:
            raise KeyError(f"cross_reactivity: unknown allergen or group {s!r}")
    return out


def base_prevalence():
    """Population prevalence per allergen, from the record weights."""
    w = np.array([kb.RECORD_WEIGHT.get(k, 0.02) for k in KEYS], dtype=float)
    return OVERALL_PREVALENCE * w / w.sum()


@lru_cache(maxsize=1)
def matrices():
    base = np.clip(1.5 * base_prevalence(), 0.0005, 0.03)
    P = np.tile(base, (K, 1))                                    # P[a, b] baseline = chance of b
    for g, members in kb.GROUPS.items():
        dflt = SAME_GROUP.get(g)
        mem = [m for m in members if m in IDX]
        if dflt is None:
            continue
        for a in mem:
            for b in mem:
                if a != b:
                    P[IDX[a], IDX[b]] = max(P[IDX[a], IDX[b]], dflt)
    for a_spec, b_spec, p, sym in RULES:
        for a in expand(a_spec):
            for b in expand(b_spec):
                if a == b:
                    continue
                P[IDX[a], IDX[b]] = p
                if sym:
                    P[IDX[b], IDX[a]] = p
    np.fill_diagonal(P, 0.0)
    W = 0.85 * np.sqrt(np.maximum(P, P.T))
    return P, W


def P_matrix():
    return matrices()[0]


def W_matrix():
    return matrices()[1]


def link(known: str, target: str) -> float:
    if known not in IDX or target not in IDX:
        return 0.0
    return float(P_matrix()[IDX[known], IDX[target]])


if __name__ == "__main__":
    P = P_matrix()
    for a, b in [("almond", "cashew"), ("cashew", "pistachio"), ("pistachio", "cashew"),
                 ("peanut", "almond"), ("shrimp", "crab"), ("cow_milk", "goat_milk"),
                 ("cow_milk", "camel_milk"), ("almond", "shrimp"), ("wheat", "barley"),
                 ("apple", "peach"), ("beef", "cow_milk")]:
        print(f"{a:>10} -> {b:<10} {P[IDX[a], IDX[b]]:.3f}")
