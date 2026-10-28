"""
Builds the Protected Words list and generates the Confusable Pairs report.

As specified in Step 4 of docs/04_spelling_correction.md:
  1. For every allergen term, lists real words within edit distance 2 using
     a general English word list (/usr/share/dict/words) and harmless vocabulary.
  2. Generates confusable pairs report.
  3. Curates the protected words list: words on this list are NEVER corrected
     into an allergen in any mode.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from rapidfuzz.distance import Levenshtein

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "spelling"
REPORT_DIR = BASE_DIR / "reports"


# Well-known confusable pairs explicitly cited in Step 4 & Step 9 specifications:
EXPLICIT_SPEC_PROTECTED = {
    # Specified examples:
    "pearl", "peer", "price", "sole", "soul", "born", "horn", "not", "say", "dish",
    "mild", "mill", "sheet", "dare", "been", "beat", "ego", "barely",
    # Common packaging and label words that must never turn into allergens:
    "clean", "cool", "dry", "place", "keep", "store", "best", "before", "see",
    "pack", "batch", "lot", "number", "fresh", "box", "lid", "cap", "bottom",
    "side", "top", "directions", "use", "serving", "size", "daily", "value",
    "energy", "net", "weight", "grams", "quality", "product", "recycle", "call",
    "visit", "website", "great", "crispy", "crunchy", "sweet", "rich", "pure",
    "choice", "premium", "classic", "original", "recipe", "roasted", "toasted",
    "dried", "organic", "solid", "solids", "mass", "paste", "pieces", "dark",
    "white", "light", "colour", "color", "colours", "colors", "clove",
    "acidity", "regulator", "stabiliser", "thickener", "ginger", "cumin",
    "turmeric", "paprika", "coriander", "cinnamon", "pepper", "vinegar", "maize",
    "seasoning", "sweetener", "vitamin", "iron", "calcium", "zinc", "magnesium",
    "water", "salt", "sugar", "oil", "fat", "acid", "agent", "agents", "juice",
    "pulp", "peel", "zest", "mint", "basil", "oregano", "thyme", "rosemary", "sage",
}


def load_english_words() -> Set[str]:
    """Load system dictionary words (3+ letters, ascii alpha)."""
    dict_path = Path("/usr/share/dict/words")
    if not dict_path.exists():
        return set()
    words = set()
    with open(dict_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip().lower()
            if len(w) >= 3 and w.isalpha() and w.isascii():
                words.add(w)
    return words


def build_protected_words_and_report():
    print("Generating Confusable Pairs and Protected Words list...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load vocabulary
    vocab_path = DATA_DIR / "spelling_vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing {vocab_path}. Run tools/build_spelling_vocab.py first.")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    allergen_single_words = set(vocab.get("allergen_single_words", []))
    harmless_words = set(vocab.get("harmless_words", []))
    english_words = load_english_words() | harmless_words

    # 2. Find confusable pairs for key and short allergen words
    # Words <= 6 letters are particularly vulnerable to distance-2 collisions
    confusable_report_pairs: List[Tuple[str, str, int, str]] = []
    protected_words: Set[str] = set(EXPLICIT_SPEC_PROTECTED)

    # Select representative allergen target words for the confusable report
    REPORT_TARGETS = [
        "pear", "rice", "sole", "corn", "nut", "soy", "fish", "milk", "wheat",
        "date", "bean", "egg", "barley", "pea", "oat", "meat", "pork", "crab",
        "tuna", "cod", "clam", "eel", "carp", "bass", "beef", "flour", "whey",
        "squid", "rye", "beer"
    ]

    for allergen in sorted(allergen_single_words):
        # We check collisions with english words
        if len(allergen) < 3:
            continue
        max_dist = 1 if len(allergen) <= 4 else 2
        for eng in english_words:
            if eng == allergen:
                continue
            # Check length filter before distance
            if abs(len(eng) - len(allergen)) > max_dist:
                continue
            dist = Levenshtein.distance(allergen, eng)
            if dist <= max_dist:
                # Is eng an allergen?
                is_all = eng in allergen_single_words
                kind = "Allergen-Allergen" if is_all else "Allergen-English"
                if allergen in REPORT_TARGETS and len(confusable_report_pairs) < 500:
                    confusable_report_pairs.append((allergen, eng, dist, kind))
                if not is_all:
                    # Real English non-allergen word -> goes into protected list!
                    protected_words.add(eng)

    # Make sure explicit spec items are in protected list
    for spec_word in EXPLICIT_SPEC_PROTECTED:
        protected_words.add(spec_word)

    # Never put an active allergen into protected_words
    for aw in allergen_single_words:
        if aw in protected_words and aw not in {"sole", "date", "perch", "bass", "ray", "rays"}:
            # Keep ambiguous parser words if already ambiguous, otherwise allergen takes priority
            protected_words.discard(aw)

    # Ensure "barely" is strictly protected
    protected_words.add("barely")

    # 3. Save Protected Words JSON
    protected_path = DATA_DIR / "protected_words.json"
    with open(protected_path, "w", encoding="utf-8") as f:
        json.dump(sorted(protected_words), f, indent=2)
    print(f"Saved {len(protected_words)} protected words to {protected_path}")

    # 4. Generate Confusable Pairs Report
    report_path = REPORT_DIR / "spelling_confusable_pairs.md"
    lines = []
    lines.append("# AllerScan Confusable Pairs and Protected Words Report")
    lines.append("")
    lines.append("Generated as required by Step 4 of `04_spelling_correction.md`.")
    lines.append("")
    lines.append("## 1. Safety Rule")
    lines.append("> **Rule 1**: A real, correctly spelled non-allergen word must **never** be rewritten into an allergen.")
    lines.append("> Tokens on the **Protected List** are immune to fuzzy allergen substitution across all modes.")
    lines.append("")
    lines.append(f"**Total Protected Words in Knowledge Base**: {len(protected_words)}")
    lines.append("")
    lines.append("## 2. Key Confusable Pairs within Edit Distance 1-2")
    lines.append("")
    lines.append("| Allergen Word | Real English Word | Distance | Protection Status | Note / Spec Reference |")
    lines.append("|---|---|---|---|---|")

    spec_highlights = [
        ("pear", "pearl", 1, "PROTECTED", "Pear / Pearl (jewel / barley)"),
        ("pear", "peer", 1, "PROTECTED", "Pear / Peer"),
        ("rice", "price", 1, "PROTECTED", "Rice / Price (label meta)"),
        ("sole", "soul", 1, "PROTECTED", "Sole fish / Soul"),
        ("corn", "born", 1, "PROTECTED", "Corn / Born"),
        ("corn", "horn", 1, "PROTECTED", "Corn / Horn"),
        ("nut", "not", 1, "PROTECTED", "Nut / Not (negation)"),
        ("soy", "say", 1, "PROTECTED", "Soy / Say (common verb)"),
        ("fish", "dish", 1, "PROTECTED", "Fish / Dish (tableware/culinary)"),
        ("milk", "mild", 1, "PROTECTED", "Milk / Mild (flavor descriptor)"),
        ("milk", "mill", 1, "PROTECTED", "Milk / Mill (grinding equipment)"),
        ("wheat", "sheet", 1, "PROTECTED", "Wheat / Sheet (packaging)"),
        ("date", "dare", 1, "PROTECTED", "Date / Dare"),
        ("bean", "been", 1, "PROTECTED", "Bean / Been (verb form)"),
        ("bean", "beat", 1, "PROTECTED", "Bean / Beat (action/beet)"),
        ("egg", "ego", 1, "PROTECTED", "Egg / Ego"),
        ("barley", "barely", 2, "PROTECTED + POSSIBLE", "Misspelling 'barely' stays barely; flagged as possible barley in label"),
    ]

    for all_w, eng_w, dist, status, note in spec_highlights:
        lines.append(f"| `{all_w}` | `{eng_w}` | {dist} | **{status}** | {note} |")

    # Add sample pairs from automated scan
    lines.append("")
    lines.append("### Automated Confusable Sample (Representative Target Allergens)")
    lines.append("")
    lines.append("| Allergen Target | Confusable English Token | Distance | Category |")
    lines.append("|---|---|---|---|")
    for all_w, eng_w, dist, kind in confusable_report_pairs[:60]:
        lines.append(f"| `{all_w}` | `{eng_w}` | {dist} | {kind} |")

    lines.append("")
    lines.append("## 3. Ambiguous and Exact-Inflection Rules")
    lines.append("- `oat` vs `oats`: Handled by exact inflection, never fuzzy mapping.")
    lines.append("- `pea` vs `peas`: Handled by exact inflection.")
    lines.append("- Words treated as ambiguous by parser (`sole`, `date`, `perch`, `bass`, `ray`, `rays`) are preserved without single-token allergen forcing.")
    lines.append("- `barely`: Never rewritten to `barley`. If found in ingredient context, flagged as possible barley.")

    report_content = "\n".join(lines) + "\n"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Confusable pairs report saved to {report_path}")


if __name__ == "__main__":
    build_protected_words_and_report()
