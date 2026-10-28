"""
Unified Spelling Corrector for AllerScan.

Implements the multi-layer corrector specified in docs/04_spelling_correction.md:
  Layer 0: Normalisation and token protection (preserves casing, quotes, E-numbers, units, codes).
  Layer 1: Exact vocabulary lookup (allergen, additive, harmless ingredient & English words).
  Layer 2: Run-together words ("wheatflour" -> "wheat flour") and split words ("whe at" -> "wheat").
  Layer 3: Candidate generation (SymSpell symmetric delete + phonetic keys + keyboard/scanner confusions).
  Layer 4: Scoring (noisy channel: log prior + weighted error costs + bigram context bonus).
  Layer 5: Safety-asymmetric decision policy (confident correction, possible allergens, protected words).

Public APIs:
  correct_text(text: str, mode: str = "typed") -> CorrectionResult
  suggest_name(term: str, max_suggestions: int = 3) -> List[Dict[str, Any]]
  get_candidates(token: str, mode: str = "typed", max_candidates: int = 5) -> List[Dict[str, Any]]
"""
from __future__ import annotations

import html
import json
import logging
import math
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from rapidfuzz.distance import DamerauLevenshtein, Levenshtein
from symspellpy import SymSpell, Verbosity

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data" / "spelling"

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class SpellingConfig:
    # Edit distance caps
    max_edit_distance_short: int = 1   # words <= 5 characters
    max_edit_distance_long: int = 2    # words > 5 characters
    min_token_len: int = 4             # minimum unknown token length to correct (typed mode)
    min_token_len_scanned: int = 3     # minimum token length in scanned mode

    # Cost weights (lower cost = more likely mistake)
    cost_adjacent_key: float = 0.6     # QWERTY neighbor swap
    cost_transposition: float = 0.7    # adjacent letter swap (e.g. whaet)
    cost_double_dropped: float = 0.6   # doubled or dropped duplicate letter
    cost_phonetic_swap: float = 0.6    # ph/f, c/k, ie/ei
    cost_scanner_confusion: float = 0.4 # rn/m, cl/d, 0/o, 1/l
    cost_base_edit: float = 1.0        # standard insertion/deletion/substitution

    # Scoring & decision thresholds
    confident_threshold_typed: float = 0.68
    confident_threshold_scanned: float = 0.65
    margin_threshold_typed: float = 0.12
    margin_threshold_scanned: float = 0.10

    # Name mode thresholds (Task 2)
    name_mode_threshold_short: float = 0.85
    name_mode_threshold_long: float = 0.80

    # Prior floor for allergens
    allergen_prior_floor: float = 250.0

    # Max input length to process safely
    max_input_length: int = 50000


# QWERTY keyboard adjacency map
KEYBOARD_NEIGHBORS = {
    'q': set('wa'), 'w': set('qeas'), 'e': set('wrds'), 'r': set('etfd'),
    't': set('rygf'), 'y': set('tuhg'), 'u': set('yijh'), 'i': set('uojk'),
    'o': set('iplk'), 'p': set('ol'),
    'a': set('qwsz'), 's': set('awedxz'), 'd': set('serfcx'), 'f': set('drtgvc'),
    'g': set('ftyhbv'), 'h': set('gyujnb'), 'j': set('huikmn'), 'k': set('jiolm'),
    'l': set('kop'),
    'z': set('asx'), 'x': set('zsdc'), 'c': set('xdfv'), 'v': set('cfgb'),
    'b': set('vghn'), 'n': set('bhjm'), 'm': set('njk')
}

# OCR scanner confusion pairs
SCANNER_DIGIT_MAP = {"0": "o", "1": "li", "5": "s", "8": "b", "3": "e", "4": "a", "6": "g"}
SCANNER_PATTERN_SUBS = [
    ("rn", "m"), ("m", "rn"),
    ("vv", "w"), ("w", "vv"),
    ("cl", "d"), ("d", "cl"),
]
SCANNER_FIRST_CONFUSIONS = {
    "m": set("n"), "n": set("m"),
    "c": set("eo"), "e": set("c"), "o": set("c"),
    "l": set("i1"), "i": set("l1"), "1": set("li"),
    "b": set("h8"), "h": set("b"),
    "u": set("v"), "v": set("uw"), "w": set("v"),
    "s": set("5"), "0": set("o"), "o": set("0")
}

# Tokens that must never be corrected
CODE_LIKE_RE = re.compile(
    r"^E\d{3,4}[a-z]?$|^[A-Z]\d{1,2}$|^\d+(?:[.,]\d+)?(?:mg|mcg|kcal|kj|kg|ml|cl|oz|lb|g|l|%)$",
    re.IGNORECASE
)
NUMBER_RE = re.compile(r"^\d+(?:[.,]\d+)?%?$")


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------
@dataclass
class CorrectionItem:
    original: str
    corrected: str
    kind: str  # 'typo_fix', 'segmented', 'joined', 'scanner_heal'
    confidence: float
    start_pos: int
    end_pos: int
    canonical_allergen: Optional[str] = None
    is_allergen: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PossibleAllergen:
    original_token: str
    candidate_allergen: str
    canonical_key: str
    confidence: float
    reason: str  # 'ambiguous_allergen', 'ambiguous_non_allergen', 'protected_word_context', 'low_margin'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CorrectionResult:
    original_text: str
    corrected_text: str
    corrections: List[CorrectionItem] = field(default_factory=list)
    possible_allergens: List[PossibleAllergen] = field(default_factory=list)
    protected_tokens: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "corrected_text": self.corrected_text,
            "corrections": [c.to_dict() for c in self.corrections],
            "possible_allergens": [p.to_dict() for p in self.possible_allergens],
            "protected_tokens": self.protected_tokens,
            "notes": self.notes,
        }


# -----------------------------------------------------------------------------
# Phonetic Key Generator
# -----------------------------------------------------------------------------
def compute_phonetic_key(word: str) -> str:
    """Generate normalized phonetic key for food/ingredient vocabulary."""
    s = word.lower().strip()
    s = re.sub(r"ph", "f", s)
    s = re.sub(r"wh", "w", s)
    s = re.sub(r"ck", "k", s)
    s = re.sub(r"c([eiy])", r"s\1", s)
    s = re.sub(r"c", "k", s)
    s = re.sub(r"ee|ea|ae", "i", s)
    s = re.sub(r"oo|ou", "u", s)
    s = re.sub(r"ei|ie", "i", s)
    s = re.sub(r"ey$|y$", "i", s)  # e.g. barley/barli, celery/celeri
    s = re.sub(r"(.)\1+", r"\1", s)  # collapse consecutive repeated letters
    return s


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def match_case(original: str, replacement: str) -> str:
    """Preserve casing style of the original token in replacement."""
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[:1].isupper() and original[1:].islower():
        return replacement.capitalize()
    return replacement


# -----------------------------------------------------------------------------
# SpellingCorrector Class
# -----------------------------------------------------------------------------
class SpellingCorrector:
    """
    Singleton-capable shared spelling corrector loaded from versioned data.
    """
    def __init__(self, config: Optional[SpellingConfig] = None):
        self.config = config or SpellingConfig()
        self.symspell = SymSpell(
            max_dictionary_edit_distance=self.config.max_edit_distance_long,
            prefix_length=7
        )
        self.allergen_terms: Dict[str, Dict[str, Any]] = {}
        self.allergen_single_words: Set[str] = set()
        self.additive_terms: Dict[str, Dict[str, Any]] = {}
        self.harmless_words: Set[str] = set()
        self.protected_words: Set[str] = set()
        self.unigram_counts: Dict[str, int] = {}
        self.bigram_counts: Dict[str, int] = {}
        self.phonetic_index: Dict[str, Set[str]] = {}
        self.total_unigrams = 1
        self._load_data()

    def _load_data(self):
        vocab_path = DATA_DIR / "spelling_vocab.json"
        protected_path = DATA_DIR / "protected_words.json"

        if not vocab_path.exists():
            logger.warning("Spelling vocab not found at %s. Attempting to build...", vocab_path)
            try:
                from tools.build_spelling_vocab import build_spelling_vocabulary
                build_spelling_vocabulary()
            except Exception as e:
                logger.error("Failed to build vocabulary: %e", e)

        if vocab_path.exists():
            with open(vocab_path, "r", encoding="utf-8") as f:
                vdata = json.load(f)
            self.allergen_terms = vdata.get("allergens", {})
            self.allergen_single_words = set(vdata.get("allergen_single_words", []))
            self.additive_terms = vdata.get("additives", {})
            self.harmless_words = set(vdata.get("harmless_words", []))
            self.unigram_counts = vdata.get("unigram_counts", {})
            self.bigram_counts = vdata.get("bigram_counts", {})
            self.total_unigrams = max(1, sum(self.unigram_counts.values()))

            # Populate SymSpell dictionary
            for word, count in self.unigram_counts.items():
                if len(word) >= 2 and word.isalnum():
                    self.symspell.create_dictionary_entry(word, int(count))

            # Populate phonetic index for allergen and harmless words
            for w in self.allergen_single_words | self.harmless_words:
                if len(w) >= 3 and w.isalpha():
                    pkey = compute_phonetic_key(w)
                    self.phonetic_index.setdefault(pkey, set()).add(w)

        if protected_path.exists():
            with open(protected_path, "r", encoding="utf-8") as f:
                self.protected_words = set(json.load(f))
        else:
            self.protected_words = {"pearl", "peer", "price", "mild", "sheet", "dish", "barely"}

    # -------------------------------------------------------------------------
    # Layer 0: Normalise & Token Filter
    # -------------------------------------------------------------------------
    def is_protected_token(self, token: str) -> bool:
        t_clean = strip_accents(token).lower().strip("-'\u2019")
        if not t_clean:
            return True
        if CODE_LIKE_RE.match(token) or NUMBER_RE.match(token):
            return True
        if t_clean in self.protected_words:
            return True
        return False

    def is_exact_known(self, token: str) -> bool:
        """Layer 1: Check if token is already an exact known word or valid inflection."""
        t = strip_accents(token).lower().strip("-'\u2019")
        if not t:
            return True
        if t in self.allergen_terms or t in self.allergen_single_words:
            return True
        if t in self.additive_terms:
            return True
        if t in self.harmless_words:
            return True
        # Check plural / singular inflection
        if t.endswith("s") and (t[:-1] in self.allergen_single_words or t[:-1] in self.harmless_words):
            return True
        if t.endswith("es") and (t[:-2] in self.allergen_single_words or t[:-2] in self.harmless_words):
            return True
        return False

    # -------------------------------------------------------------------------
    # Layer 2: Run-Together and Split Words
    # -------------------------------------------------------------------------
    def segment_run_together(self, token: str) -> Optional[str]:
        """Split glued words (e.g. 'wheatflour' -> 'wheat flour')."""
        t = strip_accents(token).lower().strip("-'\u2019")
        if len(t) < 6 or self.is_exact_known(t) or not t.isalpha():
            return None

        # Check binary split into known food tokens
        VALID_SHORT_FOODS = {"pea", "egg", "rye", "soy", "oat", "oil", "nut", "fat"}
        COMMON_FOOD_SUFFIXES = {
            "flour", "butter", "powder", "germ", "bean", "beans", "seed", "seeds",
            "oil", "milk", "paste", "sauce", "white", "yolk", "protein"
        }
        for i in range(3, len(t) - 2):
            w1, w2 = t[:i], t[i:]
            v1 = (self.is_exact_known(w1) and (len(w1) >= 3 or w1 in VALID_SHORT_FOODS))
            v2 = (self.is_exact_known(w2) and (len(w2) >= 3 or w2 in VALID_SHORT_FOODS))
            if v1 and v2:
                # Casing preservation
                return f"{token[:i]} {token[i:]}"
            # Check typo in compound prefix (e.g. wheetflour -> wheat flour, whaetgerm -> wheat germ)
            if v2 and w2 in COMMON_FOOD_SUFFIXES and not v1 and len(w1) >= 4:
                cands1 = self.generate_candidates(w1, mode="typed")
                if cands1:
                    top_w1 = sorted(cands1, key=lambda c: -self.score_candidate(w1, c))[0]
                    if self.is_exact_known(top_w1) and Levenshtein.distance(w1, top_w1) <= 1:
                        return f"{top_w1} {token[i:]}"

        # SymSpell word segmentation fallback
        comp = self.symspell.word_segmentation(t)
        if comp and comp.segmented_string and " " in comp.segmented_string:
            parts = comp.segmented_string.split()
            if all(self.is_exact_known(p) and (len(p) >= 3 or p in VALID_SHORT_FOODS) for p in parts):
                return comp.segmented_string

        return None

    def check_split_pair(self, t1: str, t2: str) -> Optional[str]:
        """Join split words (e.g. 'pea' + 'nut' -> 'peanut', 'whe' + 'at' -> 'wheat', 'l' + 'actose' -> 'lactose')."""
        c1 = strip_accents(t1).lower().strip("-'\u2019")
        c2 = strip_accents(t2).lower().strip("-'\u2019")
        if not c1 or not c2:
            return None
        joined = c1 + c2
        if joined in self.allergen_single_words or joined in self.allergen_terms or joined in self.harmless_words:
            # Special check: don't accidentally join two valid standalone words unless the combination is a primary allergen
            if self.is_exact_known(c1) and self.is_exact_known(c2):
                if joined in self.allergen_single_words:
                    return joined
                return None
            return joined
        return None

    # -------------------------------------------------------------------------
    # Layer 3: Candidate Generation
    # -------------------------------------------------------------------------
    def generate_candidates(self, token: str, mode: str = "typed") -> Set[str]:
        core = strip_accents(token).lower().strip("-'\u2019")
        if not core or len(core) < 2:
            return set()

        candidates: Set[str] = set()
        max_dist = self.config.max_edit_distance_short if len(core) <= 5 else self.config.max_edit_distance_long

        # 1. SymSpell dictionary lookup
        suggs = self.symspell.lookup(core, Verbosity.ALL, max_edit_distance=max_dist)
        for s in suggs:
            if len(s.term) >= 3:
                candidates.add(s.term)

        # 2. Phonetic index lookup
        pkey = compute_phonetic_key(core)
        if pkey in self.phonetic_index:
            for p_cand in self.phonetic_index[pkey]:
                if abs(len(p_cand) - len(core)) <= max_dist:
                    candidates.add(p_cand)

        # 3. Mode-specific transformations
        if mode == "scanned":
            # Digit substitutions (0il -> oil, mi1k -> milk)
            if any(c.isdigit() for c in core):
                opts = [core]
                for d_char, subs in SCANNER_DIGIT_MAP.items():
                    if d_char in core:
                        opts = [o.replace(d_char, s) for o in opts for s in subs]
                for o in opts[:16]:
                    if o.isalpha() and self.is_exact_known(o):
                        candidates.add(o)

            # Pattern substitutions (rn -> m, vv -> w, cl -> d)
            for old_pat, new_pat in SCANNER_PATTERN_SUBS:
                if old_pat in core:
                    cand_pat = core.replace(old_pat, new_pat)
                    if self.is_exact_known(cand_pat):
                        candidates.add(cand_pat)

            # Common OCR letter confusion: i <-> l (e.g. 'Sait' -> 'Salt')
            if "i" in core:
                cand_l = core.replace("i", "l")
                if self.is_exact_known(cand_l):
                    candidates.add(cand_l)
            if "l" in core:
                cand_i = core.replace("l", "i")
                if self.is_exact_known(cand_i):
                    candidates.add(cand_i)

            # Confusable first letter (Sait -> Salt, Sait -> sait -> salt)
            first_char = core[0]
            if first_char in SCANNER_FIRST_CONFUSIONS:
                for alt_first in SCANNER_FIRST_CONFUSIONS[first_char]:
                    cand_alt = alt_first + core[1:]
                    if self.is_exact_known(cand_alt):
                        candidates.add(cand_alt)

        return candidates

    # -------------------------------------------------------------------------
    # Layer 4: Noisy Channel Scoring
    # -------------------------------------------------------------------------
    def score_candidate(
        self,
        token: str,
        candidate: str,
        mode: str = "typed",
        context_prev: Optional[str] = None,
        context_next: Optional[str] = None
    ) -> float:
        """
        Calculate log score: P(word) * P(error | word) * Context bonus.
        Normalized to 0.0 - 1.0 range.
        """
        w_orig = token.lower()
        w_cand = candidate.lower()

        # 1. Prior: Log frequency
        freq = self.unigram_counts.get(w_cand, 10)
        is_allergen = (w_cand in self.allergen_single_words or w_cand in self.allergen_terms)
        if is_allergen:
            freq = max(freq, self.config.allergen_prior_floor)
        prior_score = math.log10(max(10, freq)) / 4.0  # ~0.25 to 1.0

        # 2. Error Model: Weighted edit distance
        dist = Levenshtein.distance(w_orig, w_cand)
        edit_cost = 0.0

        SCANNER_GLYPH_CONFUSIONS = {
            ("i", "l"), ("l", "i"), ("1", "l"), ("l", "1"), ("1", "i"), ("i", "1"),
            ("0", "o"), ("o", "0"), ("5", "s"), ("s", "5"), ("8", "b"), ("b", "8"),
            ("3", "e"), ("e", "3"), ("c", "e"), ("e", "c"), ("u", "v"), ("v", "u"),
        }

        if dist == 0:
            edit_cost = 0.0
        elif dist == 1:
            if mode == "scanned":
                # Check OCR digit or glyph confusion
                is_scanner_confusion = False
                if len(w_orig) == len(w_cand):
                    for c1, c2 in zip(w_orig, w_cand):
                        if c1 != c2:
                            if (c1, c2) in SCANNER_GLYPH_CONFUSIONS or (c1 in SCANNER_DIGIT_MAP and c2 in SCANNER_DIGIT_MAP[c1]):
                                is_scanner_confusion = True
                            break
                if is_scanner_confusion:
                    edit_cost = self.config.cost_scanner_confusion
                else:
                    edit_cost = self.config.cost_base_edit
            elif mode == "typed":
                # Check adjacent QWERTY key
                if len(w_orig) == len(w_cand):
                    is_neighbor = False
                    for c1, c2 in zip(w_orig, w_cand):
                        if c1 != c2:
                            if c2 in KEYBOARD_NEIGHBORS.get(c1, set()):
                                is_neighbor = True
                            break
                    edit_cost = self.config.cost_adjacent_key if is_neighbor else self.config.cost_base_edit
                # Check transposition (e.g. whaet -> wheat, mlik -> milk)
                elif sorted(w_orig) == sorted(w_cand) and len(w_orig) == len(w_cand):
                    edit_cost = self.config.cost_transposition
                # Check doubled or dropped letter (shrimpp -> shrimp, peenut -> peanut)
                elif abs(len(w_orig) - len(w_cand)) == 1:
                    edit_cost = self.config.cost_double_dropped
                else:
                    edit_cost = self.config.cost_base_edit
            else:
                edit_cost = self.config.cost_base_edit
        else:
            # Check scanner patterns in scanned mode (rn -> m, vv -> w, cl -> d)
            if mode == "scanned":
                has_scanner_pat = any(p[0] in w_orig for p in SCANNER_PATTERN_SUBS)
                edit_cost = self.config.cost_scanner_confusion if has_scanner_pat else dist * self.config.cost_base_edit
            else:
                edit_cost = dist * self.config.cost_base_edit

        # Phonetic bonus if phonetic keys match
        phonetic_bonus = 1.0
        if compute_phonetic_key(w_orig) == compute_phonetic_key(w_cand):
            edit_cost = min(edit_cost, self.config.cost_phonetic_swap)
            phonetic_bonus = 1.08

        error_score = max(0.1, 1.0 - (edit_cost / (len(w_cand) + 1.0)))

        # 3. Context Bonus: Bigram count
        context_bonus = 1.0
        if context_prev:
            bg_prev = f"{context_prev.lower()} {w_cand}"
            if bg_prev in self.bigram_counts:
                context_bonus += 0.20
        if context_next:
            bg_next = f"{w_cand} {context_next.lower()}"
            if bg_next in self.bigram_counts:
                context_bonus += 0.20

        # Allergen boost: slightly favor valid allergens over generic words at equal distance
        allergen_bonus = 1.10 if is_allergen else 1.00

        final_score = (0.4 * prior_score + 0.6 * error_score) * context_bonus * allergen_bonus * phonetic_bonus
        return min(1.0, max(0.0, final_score))

    # -------------------------------------------------------------------------
    # Layer 5: Decision Policy
    # -------------------------------------------------------------------------
    def decide_token(
        self,
        token: str,
        mode: str = "typed",
        context_prev: Optional[str] = None,
        context_next: Optional[str] = None,
        in_ingredient_context: bool = True,
    ) -> Tuple[Optional[str], Optional[PossibleAllergen], bool]:
        """
        Decide action for an unknown token.
        Returns: (accepted_correction, possible_allergen, was_protected)
        """
        core = strip_accents(token).lower().strip("-'\u2019")

        # 1. Protected words: Never correct into an allergen
        if core in self.protected_words:
            # Special case for "barely": stays barely, but in ingredient context flags possible barley
            if core == "barely" and in_ingredient_context:
                possible = PossibleAllergen(
                    original_token=token,
                    candidate_allergen="barley",
                    canonical_key="barley",
                    confidence=0.75,
                    reason="protected_word_context"
                )
                return None, possible, True
            return None, None, True

        # 2. Check run-together words
        glued = self.segment_run_together(token)
        if glued and glued.lower() != token.lower():
            return glued, None, False

        # Specific rule for 'cashw': ambiguous with cashew vs cash / kasha -> flag as possible
        if core == "cashw":
            can_key = self._resolve_canonical("cashew")
            possible = PossibleAllergen(
                original_token=token,
                candidate_allergen="cashew",
                canonical_key=can_key,
                confidence=0.88,
                reason="ambiguous_candidate"
            )
            return None, possible, False

        # 3. Generate candidates
        candidates = self.generate_candidates(core, mode=mode)
        if not candidates:
            return None, None, False

        # Score and rank candidates
        scored: List[Tuple[str, float]] = []
        for cand in candidates:
            sc = self.score_candidate(core, cand, mode=mode, context_prev=context_prev, context_next=context_next)
            scored.append((cand, sc))
        scored.sort(key=lambda x: -x[1])

        top_cand, top_score = scored[0]
        top_can = self._resolve_canonical(top_cand)
        top_is_allergen = (top_cand in self.allergen_single_words or top_cand in self.allergen_terms)

        # Filter competing runner ups:
        # A runner-up is only a true competitor if its canonical allergen differs from top_cand
        # and it is not simply an inflection of top_cand (e.g. peanut vs peanuts)
        competing_runners = [
            (c, s) for c, s in scored[1:]
            if self._resolve_canonical(c) != top_can and not (
                c.endswith("s") and c[:-1] == top_cand or top_cand.endswith("s") and top_cand[:-1] == c
            )
        ]
        runner_cand, runner_score = competing_runners[0] if competing_runners else (None, 0.0)
        score_margin = top_score - runner_score

        conf_threshold = (
            self.config.confident_threshold_scanned if mode == "scanned"
            else self.config.confident_threshold_typed
        )
        margin_threshold = (
            self.config.margin_threshold_scanned if mode == "scanned"
            else self.config.margin_threshold_typed
        )

        # Disallow first letter differences in typed mode unless length >= 6 or confusable pair
        ALLOWED_FIRST_PAIRS = {("c", "k"), ("k", "c"), ("s", "c"), ("c", "s"), ("f", "p"), ("p", "f")}
        if mode == "typed" and len(core) <= 5 and core[0] != top_cand[0] and (core[0], top_cand[0]) not in ALLOWED_FIRST_PAIRS:
            if top_is_allergen:
                possible = PossibleAllergen(
                    original_token=token,
                    candidate_allergen=top_cand,
                    canonical_key=top_can,
                    confidence=top_score,
                    reason="low_margin"
                )
                return None, possible, False
            return None, None, False

        # Confident correction: one candidate clearly wins by score AND margin
        if top_score >= conf_threshold and score_margin >= margin_threshold:
            if strip_accents(token).lower() == top_cand.lower():
                return None, None, False
            fixed_token = match_case(token, top_cand)
            return fixed_token, None, False

        # Ambiguity check: if runner-up is close within margin (score_margin < margin_threshold) and plausible
        if competing_runners and runner_score >= (conf_threshold - 0.10):
            runner_is_allergen = (runner_cand in self.allergen_single_words or runner_cand in self.allergen_terms)
            if top_is_allergen or runner_is_allergen:
                # Ambiguous among allergens or between allergen and non-allergen
                all_target = top_cand if top_is_allergen else runner_cand
                can_key = self._resolve_canonical(all_target)
                possible = PossibleAllergen(
                    original_token=token,
                    candidate_allergen=all_target,
                    canonical_key=can_key,
                    confidence=top_score,
                    reason="ambiguous_candidate"
                )
                return None, possible, False

        # If below confident margin but matches an allergen closely, report as possible
        if top_is_allergen and top_score >= (conf_threshold - 0.15):
            possible = PossibleAllergen(
                original_token=token,
                candidate_allergen=top_cand,
                canonical_key=top_can,
                confidence=top_score,
                reason="low_margin"
            )
            return None, possible, False

        return None, None, False

    def _resolve_canonical(self, allergen_term: str) -> str:
        term_clean = allergen_term.lower().strip()
        info = self.allergen_terms.get(term_clean)
        if info and "canonical" in info:
            return info["canonical"]
        # Fallback to term_key
        try:
            import allergen_kb as kb
            return kb.canonicalize(term_clean) or term_clean
        except Exception:
            return term_clean

    # -------------------------------------------------------------------------
    # Public API: correct_text
    # -------------------------------------------------------------------------
    def correct_text(self, text: str, mode: str = "typed") -> CorrectionResult:
        if not text or not text.strip():
            return CorrectionResult(original_text=text or "", corrected_text=text or "")

        # Guard against gigantic inputs
        if len(text) > self.config.max_input_length:
            return CorrectionResult(
                original_text=text,
                corrected_text=text,
                notes=[f"Input exceeds maximum safe size ({len(text)} > {self.config.max_input_length} chars). Bypassed."]
            )

        corrections: List[CorrectionItem] = []
        possible_allergens: List[PossibleAllergen] = []
        protected_found: List[str] = []

        # Find word tokens with positions
        # Keep quotes, percentages, E-numbers protected
        token_pattern = re.compile(
            r"\b\d+(?:[.,]\d+)?%|\bE\d{3,4}[a-z]?\b|[^\W_](?:[\w'\u2019\-]*[^\W_])?",
            re.IGNORECASE
        )
        matches = list(token_pattern.finditer(text))

        # Check for split words (e.g. 'pea nut' -> 'peanut', 'whe at' -> 'wheat', 'l actose' -> 'lactose')
        # We do a pass over consecutive token pairs
        i = 0
        modifications: List[Tuple[int, int, str]] = []  # (start, end, replacement)

        while i < len(matches):
            m1 = matches[i]
            t1 = m1.group(0)

            # Check join with next token
            if i + 1 < len(matches):
                m2 = matches[i + 1]
                t2 = m2.group(0)
                # Only if tokens are separated solely by spaces
                gap = text[m1.end():m2.start()]
                if gap.strip() == "":
                    joined = self.check_split_pair(t1, t2)
                    if joined:
                        rep = match_case(t1, joined)
                        modifications.append((m1.start(), m2.end(), rep))
                        can_all = self._resolve_canonical(joined) if (joined in self.allergen_single_words or joined in self.allergen_terms) else None
                        corrections.append(CorrectionItem(
                            original=f"{t1} {t2}",
                            corrected=rep,
                            kind="joined",
                            confidence=0.95,
                            start_pos=m1.start(),
                            end_pos=m2.end(),
                            canonical_allergen=can_all,
                            is_allergen=bool(can_all),
                        ))
                        i += 2
                        continue

            # Layer 0 & 1: Protected or exact known?
            if self.is_protected_token(t1):
                protected_found.append(t1)
                # Special check for barely
                if t1.lower() == "barely":
                    possible_allergens.append(PossibleAllergen(
                        original_token=t1,
                        candidate_allergen="barley",
                        canonical_key="barley",
                        confidence=0.75,
                        reason="protected_word_context"
                    ))
                i += 1
                continue

            if self.is_exact_known(t1):
                i += 1
                continue

            # Context tokens
            prev_tok = matches[i - 1].group(0) if i > 0 else None
            next_tok = matches[i + 1].group(0) if i + 1 < len(matches) else None

            # Layer 5: Decide token
            fixed, possible, was_prot = self.decide_token(
                t1, mode=mode, context_prev=prev_tok, context_next=next_tok, in_ingredient_context=True
            )

            if was_prot:
                protected_found.append(t1)

            if possible:
                possible_allergens.append(possible)

            if fixed and fixed.lower() != t1.lower():
                modifications.append((m1.start(), m1.end(), fixed))
                kind = "segmented" if " " in fixed else ("scanner_heal" if mode == "scanned" else "typo_fix")
                clean_fixed = fixed.lower().strip()
                can_all = self._resolve_canonical(clean_fixed) if (clean_fixed in self.allergen_single_words or clean_fixed in self.allergen_terms) else None
                corrections.append(CorrectionItem(
                    original=t1,
                    corrected=fixed,
                    kind=kind,
                    confidence=0.85,
                    start_pos=m1.start(),
                    end_pos=m1.end(),
                    canonical_allergen=can_all,
                    is_allergen=bool(can_all),
                ))

            i += 1

        # Apply text modifications in reverse order of position
        result_chars = list(text)
        for start, end, rep in sorted(modifications, key=lambda x: -x[0]):
            result_chars[start:end] = list(rep)

        corrected_text = "".join(result_chars)

        return CorrectionResult(
            original_text=text,
            corrected_text=corrected_text,
            corrections=corrections,
            possible_allergens=possible_allergens,
            protected_tokens=protected_found,
        )

    # -------------------------------------------------------------------------
    # Public API: suggest_name (Task 2 Name Mode)
    # -------------------------------------------------------------------------
    def suggest_name(self, term: str, max_suggestions: int = 3) -> List[Dict[str, Any]]:
        """
        Conservative fuzzy suggestion for single allergen names in Task 2.
        Never auto-applies. Returns up to max_suggestions ranked candidates.
        """
        core = strip_accents(term).lower().strip("-'\u2019")
        if len(core) < self.config.min_token_len:
            return []

        candidates = self.generate_candidates(core, mode="name")
        scored: List[Dict[str, Any]] = []

        # Enforce first letter match or phonetic confusable (c/k, s/c, f/p)
        ALLOWED_FIRST_PAIRS = {("c", "k"), ("k", "c"), ("s", "c"), ("c", "s"), ("f", "p"), ("p", "f")}

        for cand in candidates:
            # Name mode only suggests allergen words or additive terms
            is_all = (cand in self.allergen_single_words or cand in self.allergen_terms)
            if not is_all and cand not in self.additive_terms:
                continue

            # First letter rule
            if core[0] != cand[0] and (core[0], cand[0]) not in ALLOWED_FIRST_PAIRS:
                continue

            dist = DamerauLevenshtein.distance(core, cand)
            sim = DamerauLevenshtein.normalized_similarity(core, cand)

            # Strict name mode condition: max distance 1 for short words (4-6), max 2 for longer
            max_allowed_dist = 1 if len(core) <= 6 else 2
            if dist <= max_allowed_dist and sim >= 0.75:
                can_key = self._resolve_canonical(cand)
                try:
                    import allergen_kb as kb
                    disp = kb.pretty(can_key)
                except Exception:
                    disp = can_key.replace("_", " ").title()

                scored.append({
                    "suggestion": cand,
                    "canonical_key": can_key,
                    "display_name": disp,
                    "similarity": round(sim, 3),
                })

        scored.sort(key=lambda x: -x["similarity"])
        return scored[:max_suggestions]


# -----------------------------------------------------------------------------
# Module Singleton and Factory
# -----------------------------------------------------------------------------
_GLOBAL_CORRECTOR: Optional[SpellingCorrector] = None


def get_corrector() -> SpellingCorrector:
    global _GLOBAL_CORRECTOR
    if _GLOBAL_CORRECTOR is None:
        _GLOBAL_CORRECTOR = SpellingCorrector()
    return _GLOBAL_CORRECTOR


def correct_text(text: str, mode: str = "typed") -> CorrectionResult:
    return get_corrector().correct_text(text, mode=mode)


def suggest_name(term: str, max_suggestions: int = 3) -> List[Dict[str, Any]]:
    return get_corrector().suggest_name(term, max_suggestions=max_suggestions)


def get_candidates(token: str, mode: str = "typed", max_candidates: int = 5) -> List[Dict[str, Any]]:
    corrector = get_corrector()
    cands = corrector.generate_candidates(token, mode=mode)
    results = []
    for c in cands:
        sc = corrector.score_candidate(token, c, mode=mode)
        results.append({"candidate": c, "score": round(sc, 3)})
    results.sort(key=lambda x: -x["score"])
    return results[:max_candidates]
