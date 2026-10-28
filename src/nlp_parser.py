"""
Ingredient-label parser.

parse_label(text) -> {canonical_allergen: {sources, confidence, kind, hidden}}

  kind = "direct"   allergen is named ("wheat flour", "skim milk powder", "tahini")
         "derived"  hidden-allergen knowledge base ("natural flavour", "lecithin", "surimi")
         "trace"    precautionary label ("may contain traces of peanuts")

What changed compared with the first version:
  * matches the whole synonym map (1,600 terms, incl. derivatives such as casein, whey,
    semolina, tamari) with longest-phrase matching instead of exact whole-string equality,
    so "wheat flour (gluten)" or "skim milk powder" are found
  * output keys are already canonical (peanut, cow_milk, shrimp ...), same as the model
  * brackets / sub-ingredients are split, percentages and "x-free" claims are dropped
  * "may contain" / "traces of" / shared-facility statements are detected as trace warnings
  * plant milks (oat milk, almond milk) no longer match cow's milk; cocoa butter is not dairy
  * no spaCy model download needed (the old matcher patterns are covered by the map)
"""
import re

from allergen_kb import reverse_map, term_key

# term -> (canonical allergens, confidence). Values use keys from name_mapping.json.
HIDDEN_KB = {
    "natural_flavour": (["wheat", "cow_milk", "shrimp"], "medium"),
    "natural_flavor": (["wheat", "cow_milk", "shrimp"], "medium"),
    "natural_flavours": (["wheat", "cow_milk", "shrimp"], "medium"),
    "natural_flavors": (["wheat", "cow_milk", "shrimp"], "medium"),
    "spice_extract": (["mustard_seed", "celery"], "low"),
    "spice_extracts": (["mustard_seed", "celery"], "low"),
    "surimi": (["pollock", "crab"], "high"),
    "lecithin": (["soybean", "chicken_egg"], "medium"),
    "vegetable_protein": (["soybean", "wheat"], "medium"),
    "hydrolyzed_vegetable_protein": (["soybean", "wheat"], "medium"),
    "hydrolysed_vegetable_protein": (["soybean", "wheat"], "medium"),
    "malt": (["barley"], "medium"),
    "malt_extract": (["barley"], "medium"),
    "malt_flavouring": (["barley"], "medium"),
    "worcestershire_sauce": (["anchovy", "soybean"], "medium"),
    "caesar_dressing": (["anchovy", "chicken_egg"], "medium"),
    "fish_sauce": (["anchovy"], "high"),
    "oyster_sauce": (["oyster", "soybean", "wheat"], "high"),
    "nougat": (["hazelnut", "almond", "chicken_egg"], "medium"),
    "praline": (["hazelnut"], "medium"),
    "marzipan": (["almond"], "high"),
    "tahini": (["sesame_seed"], "high"),
    "halva": (["sesame_seed"], "high"),
    "hummus": (["chickpea", "sesame_seed"], "high"),
    "pesto": (["cashew", "cow_milk"], "low"),
    "modified_starch": (["wheat", "corn"], "low"),
    "modified_food_starch": (["wheat", "corn"], "low"),
    "bouillon": (["celery", "cow_milk"], "low"),
    "tree_nut": (["almond", "english_walnut", "cashew", "hazelnut", "pistachio", "pecan",
                  "brazil_nut", "macadamia_nut"], "medium"),
    "tree_nuts": (["almond", "english_walnut", "cashew", "hazelnut", "pistachio", "pecan",
                   "brazil_nut", "macadamia_nut"], "medium"),
    "nuts": (["peanut", "almond", "english_walnut", "cashew", "hazelnut", "pistachio", "pecan"],
             "medium"),
}

# plant "milks" and similar phrases that must not fall back to cow's milk / dairy
PLANT_PHRASES = {
    "oat_milk": "oats", "oat_drink": "oats", "rice_milk": "rice", "almond_milk": "almond",
    "cashew_milk": "cashew", "hazelnut_milk": "hazelnut", "pea_milk": "pea",
    "pea_protein": "pea", "hemp_milk": "hemp_seed", "macadamia_milk": "macadamia_nut",
}
IGNORE_PHRASES = {"cocoa_butter", "shea_butter", "cream_of_tartar", "nutmeg", "butternut",
                  "butternut_squash", "coconut_aminos_free", "egg_free", "milk_free"}
# one-word terms in the synonym map that are too ambiguous on a food label
AMBIGUOUS_SINGLE = {"ray", "rays", "sole", "date", "perch", "bass", "hen", "hog", "pig", "cattle",
                    "swine", "horse", "broiler", "groats", "tsp", "tvp", "half", "mung", "sepia",
                    "helix", "ambrosia", "koi", "dorado", "kasha"}

TRIGGER = (r"(?:may|might|could|can)\s+contain|traces?\s+of|"
           r"(?:produced|processed|made|manufactured|packed|packaged|handled)\s+(?:in|on)\s+[^.\n]*?"
           r"(?:facility|factory|equipment|line)")
TRACE_RE = re.compile(rf"(?:{TRIGGER})[^.\n]*", re.I)
FREE_RE = re.compile(r"\b[\w']+[\s-]free\b", re.I)
PERCENT_RE = re.compile(r"\d+(?:[.,]\d+)?\s*%")

_LEX = dict(reverse_map())
_LEX.update({k: v for k, v in PLANT_PHRASES.items()})
_MAX_N = min(6, max(k.count("_") + 1 for k in _LEX))
_RANK = {"direct": 3, "derived": 2, "trace": 1}
_CONF_RANK = {"high": 3, "medium": 2, "low": 1}


def _segments(text: str):
    t = PERCENT_RE.sub(" ", text.lower())
    t = FREE_RE.sub(" ", t)
    t = re.sub(r"\bcontains?\b|\bingredients?\b|\ballergens?\b|\ballergy advice\b", " ", t)
    t = re.sub(r"[()\[\]{}:;/&|•\n]|\band\b|\bor\b|\bwith\b", ",", t)
    return [s.strip() for s in re.split(r"[,.]", t) if s.strip()]


def _scan(segment: str):
    """Yield (kind, allergens, confidence, phrase) using longest-phrase-first matching."""
    toks = re.findall(r"[a-z]+(?:'[a-z]+)?", segment)
    i = 0
    while i < len(toks):
        step, hit = 1, None
        for n in range(min(_MAX_N, len(toks) - i), 0, -1):
            phrase = " ".join(toks[i:i + n])
            key = term_key(phrase)
            if key in IGNORE_PHRASES:
                step, hit = n, "ignore"
                break
            if key in HIDDEN_KB:
                alls, conf = HIDDEN_KB[key]
                step, hit = n, ("derived", alls, conf, phrase)
                break
            cand = key if key in _LEX else (key[:-1] if key.endswith("s") and key[:-1] in _LEX else None)
            if cand and not (n == 1 and cand in AMBIGUOUS_SINGLE):
                step, hit = n, ("direct", [_LEX[cand]], "high", phrase)
                break
        if hit and hit != "ignore":
            yield hit
        i += step


def parse_label(raw_text: str) -> dict:
    text = (raw_text or "").replace("_", " ")      # Open Food Facts marks allergens as _milk_
    trace_chunks = []

    def grab(m):
        trace_chunks.append(m.group(0))
        return " . "

    main_text = TRACE_RE.sub(grab, text)
    found = {}

    def add(allergen, kind, conf, source):
        cur = found.get(allergen)
        if cur is None:
            found[allergen] = {"sources": [source], "confidence": conf, "kind": kind}
            return
        if source not in cur["sources"]:
            cur["sources"].append(source)
        if _RANK[kind] > _RANK[cur["kind"]]:
            cur["kind"] = kind
            cur["sources"] = [source]          # a named ingredient replaces the weaker evidence
        if _CONF_RANK[conf] > _CONF_RANK[cur["confidence"]]:
            cur["confidence"] = conf

    for seg in _segments(main_text):
        for kind, allergens, conf, phrase in _scan(seg):
            source = seg if (kind == "direct" and len(seg) < 40) else phrase
            for a in allergens:
                add(a, kind, conf, source)
    for chunk in trace_chunks:
        for seg in _segments(chunk):
            for kind, allergens, _, phrase in _scan(seg):
                for a in allergens:
                    add(a, "trace", "low", phrase)

    for info in found.values():
        info["hidden"] = info["kind"] != "direct"        # feeds is_hidden_allergen_flag
    return found


if __name__ == "__main__":
    import pprint
    tests = [
        "Contains: surimi, natural flavour, shrimp extract, salt",
        "Wheat flour (gluten), skim milk powder, soy lecithin, cocoa butter, oat milk, salt. "
        "May contain traces of peanuts and tree nuts. Dairy-free claim: none.",
        "Hazelnuts 13%, sugar, palm oil, whey protein, lecithin, vanillin. Best before date on lid.",
        "Water, tomato, celery, spice extract, sesame seeds, egg white",
    ]
    for t in tests:
        print("\n" + t)
        pprint.pprint(parse_label(t))
