"""
Pure-Python term resolver for typed allergen input in AllerScan.

Implements input cleaning, splitting, sanitization, and 4-tier matching:
  Tier 1: Exact or normalized match (aliases, singular/plural, possessives, separators, KB lookup)
  Tier 2: Group words (tree nuts, shellfish, seafood, fish, legumes, cereals, gluten, dairy) & clarifications
  Tier 3: Conservative fuzzy suggestions (conservative thresholds, no auto-accept)
  Tier 4: Custom allergen for unmodeled foods (slugified key, display name, ingredient scanner support)

Rules:
- Never imports streamlit or any web app framework.
- Always escapes / sanitizes user text.
- Never mutates knowledge base or model data.
"""
from __future__ import annotations

import difflib
import html
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Try local or package import of allergen_kb
try:
    from . import allergen_kb as kb
except ImportError:
    import allergen_kb as kb

BASE_DIR = Path(__file__).parent
TYPED_ALIASES_PATH = BASE_DIR / "typed_aliases.json"

# First-letter confusables for conservative fuzzy matching
CONFUSABLE_FIRST_PAIRS = {
    ("c", "k"), ("k", "c"),
    ("s", "c"), ("c", "s"),
    ("f", "p"), ("p", "f"),
}


@dataclass
class ResolvedItem:
    original: str
    status: str  # 'matched', 'group', 'suggestions', 'custom', 'rejected', 'already_added', 'ask_which', 'over_broad'
    clean_term: str = ""
    matched_keys: List[str] = field(default_factory=list)
    display_name: str = ""
    group_name: str = ""
    message: str = ""
    suggestions: List[Dict[str, Any]] = field(default_factory=list)
    options: List[Dict[str, Any]] = field(default_factory=list)
    custom_key: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)


def load_typed_aliases() -> Dict[str, Any]:
    """Load the typed aliases, groups, and clarification prompts from JSON."""
    if not TYPED_ALIASES_PATH.exists():
        return {"typed_aliases": {}, "groups": {}, "ask_which": {}, "over_broad": {}}
    with open(TYPED_ALIASES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def escape_html(text: str) -> str:
    """Safely escape text for raw HTML output."""
    return html.escape(str(text), quote=True)


def slugify_custom_key(term: str) -> str:
    """Generate a collision-free internal key for custom allergens."""
    s = str(term).strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("utf-8")
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return f"custom_{s}" if s else "custom_allergen"


def split_raw_input(raw: str) -> List[str]:
    """
    Split on commas, semicolons, slashes, new lines, bullets, words 'and', 'or', '&', and plus signs.
    """
    if not raw or not raw.strip():
        return []
    # Split on delimiters: commas, semicolons, slashes, new lines, tabs, plus, ampersand, bullet chars
    # Also split on standalone words 'and' or 'or'
    pattern = r"[,;/\n\r\t+&•\u2022\u2023\u25e6\u2043\u2219]+|\b(?:and|or)\b"
    parts = re.split(pattern, raw, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def clean_term_piece(piece: str) -> Tuple[str, Optional[str]]:
    """
    Clean and validate an individual input piece.
    Returns (cleaned_text, rejection_error_message).
    If rejection_error_message is not None, the term is rejected.
    """
    s = piece.strip()
    if not s:
        return "", "Please enter an allergen."

    # Check maximum length (40 chars)
    if len(s) > 40:
        return s, "Term is too long (maximum 40 characters)."

    # Remove text in brackets: (), [], {}
    s = re.sub(r"\(.*?\)|\[.*?\]|\{.*?\}", "", s).strip()
    if not s:
        return piece.strip(), "Please use plain words (no letters found)."

    # Check allowed characters: letters, spaces, hyphens, apostrophes, periods (allow digits for E-numbers)
    is_e_num = bool(re.match(r"^(?:E|INS)[\s\-]*\d+[a-z]?(\([a-z0-9]+\))?$", s, re.IGNORECASE))
    if not is_e_num and re.search(r"[^a-zA-Z\s\-\'\.]", s):
        return s, "Please use plain words (letters, spaces, and hyphens only)."

    # Remove accents for matching
    norm = unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("utf-8")

    # Collapse repeated spaces and hyphens
    norm = re.sub(r"[\s\-]+", " ", norm).strip()

    # Remove leading phrases: "allergic to", "allergy to", "allergy:", "i am allergic to"
    norm = re.sub(r"^(?:i am allergic to|allergic to|allergy to|allergy:)\s*", "", norm, flags=re.IGNORECASE).strip()

    # Remove trailing words: "allergy", "allergies", "intolerance", "free"
    norm = re.sub(r"\s+(?:allergies|allergy|intolerance|free)$", "", norm, flags=re.IGNORECASE).strip()

    clean = norm.lower()

    # Must contain at least one letter
    if not any(c.isalpha() for c in clean):
        return s, "Please use plain words (no letters found)."

    return clean, None


def _is_first_letter_compatible(term_a: str, term_b: str) -> bool:
    """Check if two words share the same first letter or a known confusable pair."""
    if not term_a or not term_b:
        return False
    a0, b0 = term_a[0].lower(), term_b[0].lower()
    if a0 == b0:
        return True
    if (a0, b0) in CONFUSABLE_FIRST_PAIRS:
        return True
    if (term_a.startswith("ph") and term_b.startswith("f")) or (term_a.startswith("f") and term_b.startswith("ph")):
        return True
    return False


def get_candidate_pool() -> Dict[str, str]:
    """Build candidate pool for fuzzy matching mapping candidate phrases to canonical keys."""
    mapping = kb.load_name_mapping()
    pool: Dict[str, str] = {}
    for canonical, synonyms in mapping.items():
        can = kb.MERGE.get(canonical, canonical)
        pool[canonical] = can
        pool[canonical.replace("_", " ")] = can
        pool[kb.pretty(can).lower()] = can
        for syn in synonyms:
            syn_clean = syn.strip().lower()
            pool[syn_clean] = can
    return pool


def get_fuzzy_suggestions(
    term: str,
    candidate_pool: Optional[Dict[str, str]] = None,
    max_suggestions: int = 3,
) -> List[Dict[str, Any]]:
    """
    Conservative fuzzy suggestions for Tier 3 using unified spelling corrector in Name Mode.
    Never auto-accepts; never suggests for words under 4 letters.
    Requires same first letter or known confusable pair.
    """
    t = term.strip().lower()
    if len(t) < 4:
        return []

    try:
        from . import spelling_corrector as sc
    except ImportError:
        import spelling_corrector as sc

    suggs = sc.suggest_name(term, max_suggestions=max_suggestions)
    results: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for s in suggs:
        k = s["canonical_key"]
        if k not in seen:
            # Use SequenceMatcher ratio for backwards-compatible scoring range with Task 2
            sim_ratio = round(difflib.SequenceMatcher(None, t, s["suggestion"]).ratio(), 3)
            results.append({"key": k, "name": s["display_name"], "score": sim_ratio})
            seen.add(k)
    return results


def _singular_plural_variants(word: str) -> List[str]:
    """Generate safe, simple singular and plural forms."""
    variants = [word]
    if word.endswith("ies") and len(word) > 4:
        variants.append(word[:-3] + "y")
    elif word.endswith("es") and len(word) > 3:
        variants.append(word[:-2])
        variants.append(word[:-1])
    elif word.endswith("s") and len(word) > 2:
        variants.append(word[:-1])
    else:
        # Generate plurals
        if word.endswith(("s", "x", "z", "ch", "sh")):
            variants.append(word + "es")
        elif word.endswith("y") and len(word) > 2 and word[-2] not in "aeiou":
            variants.append(word[:-1] + "ies")
        else:
            variants.append(word + "s")
    return list(dict.fromkeys(variants))


def resolve_single_term(
    clean: str,
    original: str,
    aliases_data: Dict[str, Any],
    candidate_pool: Dict[str, str],
    existing_model_keys: Set[str],
    existing_custom_keys: Set[str],
    current_custom_count: int,
    max_custom: int = 15,
    existing_additive_keys: Optional[Set[str]] = None,
) -> ResolvedItem:
    """Resolve a single cleaned term through Tiers 1-4."""
    groups_dict = aliases_data.get("groups", {})
    typed_aliases = aliases_data.get("typed_aliases", {})
    ask_which_dict = aliases_data.get("ask_which", {})
    over_broad_dict = aliases_data.get("over_broad", {})

    # ------------------------------------------------------------------
    # Tier 2: Check Clarifications & Group words first
    # (Shellfish/seafood must not become shrimp via canonicalize)
    # ------------------------------------------------------------------
    if clean in ask_which_dict:
        info = ask_which_dict[clean]
        return ResolvedItem(
            original=original,
            clean_term=clean,
            status="ask_which",
            display_name=clean.title(),
            message=info.get("prompt", "Please choose which allergen you mean:"),
            options=info.get("options", []),
        )

    if clean in over_broad_dict:
        msg = over_broad_dict[clean]
        custom_k = slugify_custom_key(clean)
        return ResolvedItem(
            original=original,
            clean_term=clean,
            status="over_broad",
            display_name=clean.title(),
            message=msg,
            custom_key=custom_k,
            options=[
                {"action": "keep_custom", "label": f"Keep as custom '{clean.title()}'", "key": custom_k},
                {"action": "dismiss", "label": "Dismiss"},
            ],
        )

    if clean in groups_dict:
        g = groups_dict[clean]
        return ResolvedItem(
            original=original,
            clean_term=clean,
            status="group",
            display_name=g.get("display_name", clean.title()),
            group_name=g.get("display_name", clean.title()),
            matched_keys=list(g.get("members", [])),
            message=g.get("message", f"{g.get('display_name')}: {len(g.get('members', []))} foods added."),
        )

    # ------------------------------------------------------------------
    # Tier 1: Exact or normalised match
    # ------------------------------------------------------------------
    matched_key: Optional[str] = None

    # Direct alias lookup
    if clean in typed_aliases:
        matched_key = typed_aliases[clean]

    # Existing KB canonicalize lookup
    if not matched_key:
        matched_key = kb.canonicalize(clean)

    # Singular and plural variants
    if not matched_key:
        for var in _singular_plural_variants(clean):
            if var in typed_aliases:
                matched_key = typed_aliases[var]
                break
            k = kb.canonicalize(var)
            if k:
                matched_key = k
                break

    # Possessive removal
    if not matched_key and "'" in clean:
        no_poss = clean.replace("'s", "").replace("'", "")
        if no_poss in typed_aliases:
            matched_key = typed_aliases[no_poss]
        if not matched_key:
            matched_key = kb.canonicalize(no_poss)

    # Separator variants (spaces <-> hyphens <-> underscores)
    if not matched_key:
        alt = clean.replace("-", " ")
        if alt in typed_aliases:
            matched_key = typed_aliases[alt]
        if not matched_key:
            matched_key = kb.canonicalize(alt)

    if matched_key:
        # Check if already added
        if matched_key in existing_model_keys:
            return ResolvedItem(
                original=original,
                clean_term=clean,
                status="already_added",
                matched_keys=[matched_key],
                display_name=kb.pretty(matched_key),
                message=f"'{kb.pretty(matched_key)}' is already in your allergen list.",
            )
        return ResolvedItem(
            original=original,
            clean_term=clean,
            status="matched",
            matched_keys=[matched_key],
            display_name=kb.pretty(matched_key),
            message=f"Matched {kb.pretty(matched_key)}.",
        )

    # ------------------------------------------------------------------
    # Additive sensitivity check (Task 3 coordination)
    # ------------------------------------------------------------------
    try:
        from . import additive_detector as ad
    except ImportError:
        import additive_detector as ad

    try:
        detector = ad.get_detector()
        cand_keys = [clean, clean.replace("-", " ")] + _singular_plural_variants(clean)
        additive_hit = None
        for cand in cand_keys:
            if cand in detector.exact_phrase_map:
                additive_hit = detector.exact_phrase_map[cand]
                break
            e_cand = cand.replace(" ", "").replace("-", "")
            if e_cand in detector.e_map:
                additive_hit = detector.e_map[e_cand]
                break

        if additive_hit:
            a_data = detector.additives[additive_hit]
            if existing_additive_keys and additive_hit in existing_additive_keys:
                return ResolvedItem(
                    original=original,
                    clean_term=clean,
                    status="already_added",
                    matched_keys=[additive_hit],
                    display_name=a_data["display_name"],
                    message=f"'{a_data['display_name']}' is already in your additive sensitivities.",
                )
            return ResolvedItem(
                original=original,
                clean_term=clean,
                status="additive",
                matched_keys=[additive_hit],
                display_name=a_data["display_name"],
                message=f"Recognised as additive sensitivity: {a_data['display_name']}.",
            )
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Tier 3: Conservative "did you mean" suggestions
    # ------------------------------------------------------------------
    suggestions = get_fuzzy_suggestions(clean, candidate_pool=candidate_pool, max_suggestions=3)
    if suggestions:
        custom_k = slugify_custom_key(clean)
        return ResolvedItem(
            original=original,
            clean_term=clean,
            status="suggestions",
            display_name=clean.title(),
            custom_key=custom_k,
            suggestions=suggestions,
            message=f"Did you mean {suggestions[0]['name']}?",
        )

    # ------------------------------------------------------------------
    # Tier 4: Custom allergen
    # ------------------------------------------------------------------
    custom_k = slugify_custom_key(clean)
    if custom_k in existing_custom_keys:
        return ResolvedItem(
            original=original,
            clean_term=clean,
            status="already_added",
            custom_key=custom_k,
            display_name=clean.title(),
            message=f"'{clean.title()}' is already in your custom allergens.",
        )

    if current_custom_count >= max_custom:
        return ResolvedItem(
            original=original,
            clean_term=clean,
            status="rejected",
            custom_key=custom_k,
            display_name=clean.title(),
            message=f"Limit reached: You can have at most {max_custom} custom allergens. Please remove some before adding more.",
        )

    return ResolvedItem(
        original=original,
        clean_term=clean,
        status="custom",
        custom_key=custom_k,
        display_name=clean.title(),
        message="Custom allergen (not in the model).",
    )


def resolve_terms(
    raw_text: str,
    existing_model_keys: Optional[Sequence[str]] = None,
    existing_custom_keys: Optional[Sequence[str]] = None,
    current_custom_count: int = 0,
    max_custom: int = 15,
    existing_additive_keys: Optional[Sequence[str]] = None,
) -> List[ResolvedItem]:
    """
    Main entry point for resolving raw typed text into structured ResolvedItems.
    """
    raw_stripped = (raw_text or "").strip()
    if not raw_stripped:
        return [
            ResolvedItem(
                original=raw_text,
                status="rejected",
                message="Please enter an allergen name.",
            )
        ]

    pieces = split_raw_input(raw_stripped)
    if not pieces:
        # Raw input contained only delimiter characters
        return [
            ResolvedItem(
                original=raw_stripped,
                status="rejected",
                message="Please use plain words (no letters found).",
            )
        ]

    aliases_data = load_typed_aliases()
    candidate_pool = get_candidate_pool()

    curr_model_keys: Set[str] = set(existing_model_keys or [])
    curr_custom_keys: Set[str] = set(existing_custom_keys or [])
    curr_additive_keys: Set[str] = set(existing_additive_keys or [])
    custom_count = current_custom_count

    results: List[ResolvedItem] = []

    for piece in pieces:
        clean, err = clean_term_piece(piece)
        if err is not None:
            results.append(
                ResolvedItem(
                    original=piece,
                    clean_term=clean,
                    status="rejected",
                    display_name=piece[:30],
                    message=err,
                )
            )
            continue

        item = resolve_single_term(
            clean=clean,
            original=piece,
            aliases_data=aliases_data,
            candidate_pool=candidate_pool,
            existing_model_keys=curr_model_keys,
            existing_custom_keys=curr_custom_keys,
            current_custom_count=custom_count,
            max_custom=max_custom,
            existing_additive_keys=curr_additive_keys,
        )

        # Track keys added within the same batch to prevent duplicates
        if item.status == "matched":
            for k in item.matched_keys:
                curr_model_keys.add(k)
        elif item.status == "group":
            for k in item.matched_keys:
                curr_model_keys.add(k)
        elif item.status == "additive":
            for k in item.matched_keys:
                curr_additive_keys.add(k)
        elif item.status == "custom":
            curr_custom_keys.add(item.custom_key)
            custom_count += 1

        results.append(item)

    return results


# --------------------------------------------------------------------------
# Step 6: Custom Allergen Ingredient Text Matcher
# --------------------------------------------------------------------------

def _levenshtein_distance_at_most_one(s1: str, s2: str) -> int:
    """Fast check if Levenshtein distance between s1 and s2 is 0 or 1."""
    if abs(len(s1) - len(s2)) > 1:
        return 2
    if s1 == s2:
        return 0
    if len(s1) == len(s2):
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))
    short, long_ = (s1, s2) if len(s1) < len(s2) else (s2, s1)
    for i in range(len(long_)):
        if short == long_[:i] + long_[i + 1:]:
            return 1
    return 2


def match_custom_allergens(
    custom_allergens: Dict[str, Dict[str, Any]] | Sequence[Dict[str, Any]],
    text: str,
) -> List[Dict[str, Any]]:
    """
    Checks the ingredient text for each custom allergen:
      - Whole-word matching of the cleaned term and singular/plural variants
      - Multi-word terms matched as phrases
      - Tolerance of one character difference for terms of 6 letters or more ('possible match, check the label')
      - No matching inside unrelated longer words
    """
    if not custom_allergens or not text or not text.strip():
        return []

    if isinstance(custom_allergens, dict):
        items = list(custom_allergens.values())
    else:
        items = list(custom_allergens)

    # Normalize ingredient text for matching (lowercased, accents removed)
    norm_text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8").lower()
    # Replace punctuation (except hyphens) with spaces for word tokenization
    clean_text = re.sub(r"[^\w\s\-]", " ", norm_text)
    words = [w for w in re.split(r"[\s,;]+", clean_text) if w]

    hits: List[Dict[str, Any]] = []

    for item in items:
        name = item.get("display_name") or item.get("name") or "Custom Allergen"
        raw_term = item.get("clean_term") or item.get("term") or name
        rating = int(item.get("rating", 2))
        key = item.get("key") or slugify_custom_key(raw_term)

        # Normalize target term
        target = unicodedata.normalize("NFKD", raw_term).encode("ASCII", "ignore").decode("utf-8").lower().strip()
        if not target:
            continue

        variants = _singular_plural_variants(target)

        # 1. Exact phrase or whole-word search
        matched_exact = False
        for var in variants:
            # Build regex requiring word boundaries
            pattern = r"\b" + r"\s+".join(re.escape(w) for w in var.split()) + r"\b"
            m = re.search(pattern, norm_text)
            if m:
                hits.append({
                    "key": key,
                    "name": name,
                    "rating": rating,
                    "matched_phrase": m.group(0),
                    "is_possible": False,
                    "sources": [m.group(0)],
                })
                matched_exact = True
                break

        if matched_exact:
            continue

        # 2. 1-character difference tolerance for single-word terms of 6 letters or more
        # Multi-word terms must have total length >= 6
        if len(target) >= 6 and " " not in target:
            target_len = len(target)
            possible_hit = None
            for w in words:
                if abs(len(w) - target_len) <= 1:
                    dist = _levenshtein_distance_at_most_one(w, target)
                    if dist == 1:
                        possible_hit = w
                        break
                    # Also test plural form
                    for var in variants[1:]:
                        if abs(len(w) - len(var)) <= 1 and _levenshtein_distance_at_most_one(w, var) == 1:
                            possible_hit = w
                            break
                    if possible_hit:
                        break

            if possible_hit:
                hits.append({
                    "key": key,
                    "name": name,
                    "rating": rating,
                    "matched_phrase": possible_hit,
                    "is_possible": True,
                    "sources": [f"{possible_hit} (possible match for {name})"],
                })

    return hits
