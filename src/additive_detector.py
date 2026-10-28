"""
Pure-Python food additive detection and evidence retrieval module for AllerScan.

Implements:
1. Additive database loading and strict schema validation.
2. Robust text matching for E-numbers, INS numbers, US colour names, chemical names, and synonyms.
3. Scanner OCR confusion healing (O->0, l/I->1, stray spaces) restricted strictly to candidate E-numbers.
4. Negation detection (e.g. "free from artificial colours", "no added sulphites", "sulphite free").
5. Precautionary trace label detection ("may contain traces of...").
6. Union integration with structured product tags (e.g. Open Food Facts).
7. Personal sensitivity coordination ("You listed this").
8. Asthma relevance annotation for sulfites.

Strict Boundaries:
- Pure Python; zero dependency on web frameworks (no Streamlit imports).
- Does not modify any machine learning models, datasets, or allergen KB files.
- Medical disclaimer included on all detailed responses.
"""
from __future__ import annotations

import html
import json
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Path to the versioned additive warnings data file
DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "additive_warnings.json"

MEDICAL_DISCLAIMER = (
    "If you think you react to this, ask an allergist; "
    "reactions to additives are hard to prove and are rarely confirmed in tests."
)


class AdditiveValidationError(Exception):
    """Raised when additive data fails strict schema or consistency validation."""
    pass


@dataclass
class AdditiveOccurrence:
    matched_text: str
    match_kind: str  # 'e_number', 'name', 'synonym', 'us_colour_name', 'structured_tag'
    position: Tuple[int, int]
    is_trace: bool
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AdditiveMatch:
    additive_id: str
    display_name: str
    matched_text: str
    match_kind: str
    position: Tuple[int, int]
    is_trace: bool
    confidence: float
    tier: int
    data: Dict[str, Any]
    all_occurrences: List[AdditiveOccurrence] = field(default_factory=list)
    is_user_listed: bool = False
    asthma_relevant: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["all_occurrences"] = [occ.to_dict() for occ in self.all_occurrences]
        return d


# Negation phrases that indicate absence of the additive
# Windowed checking is applied around candidate matches
NEGATION_BEFORE_PATTERNS = [
    r"\bfree\s+from\s+(?:all\s+|any\s+|artificial\s+|added\s+)?",
    r"\bno\s+(?:added\s+|artificial\s+)?",
    r"\bcontains?\s+no\s+(?:artificial\s+|added\s+)?",
    r"\bwithout\s+(?:any\s+|added\s+|artificial\s+)?",
    r"\b0%\s+",
    r"\bzero\s+(?:added\s+|artificial\s+)?",
    r"\bdoes\s+not\s+contain\s+",
    r"\bnever\s+(?:contains?|uses?)\s+",
    r"\bnon-?",
]

NEGATION_AFTER_PATTERNS = [
    r"-?\s*free\b",
]

TRACE_PATTERNS = [
    r"\bmay\s+contain\s+(?:traces\s+of\s+)?",
    r"\btraces?\s+of\s+",
    r"\bmade\s+in\s+a\s+(?:factory|facility)\s+that\s+(?:also\s+)?(?:handles|processes)\s+",
    r"\bpackaged\s+in\s+a\s+(?:factory|facility)\s+that\s+(?:also\s+)?(?:handles|processes)\s+",
    r"\bproduced\s+on\s+lines\s+that\s+(?:also\s+)?(?:handle|process)\s+",
]


def validate_additive_data(data: Dict[str, Any]) -> List[str]:
    """
    Validate the additive knowledge base data structure.
    Returns a list of error strings (empty if valid).
    """
    errors: List[str] = []

    if not isinstance(data, dict):
        return ["Data root must be a JSON object."]

    if not data.get("version"):
        errors.append("Missing 'version' field.")
    if not data.get("last_reviewed"):
        errors.append("Missing 'last_reviewed' field.")

    additives = data.get("additives")
    if not isinstance(additives, list) or len(additives) == 0:
        errors.append("'additives' must be a non-empty list.")
        return errors

    seen_ids: Set[str] = set()
    alias_to_id: Dict[str, str] = {}
    e_regex = re.compile(r"^E\s*\d{3,4}(?:[a-z]{1,4})?(?:\([a-z0-9]+\))?$", re.IGNORECASE)

    for idx, item in enumerate(additives):
        prefix = f"Additive #{idx}"
        if not isinstance(item, dict):
            errors.append(f"{prefix} is not an object.")
            continue

        a_id = item.get("id")
        if not a_id or not isinstance(a_id, str):
            errors.append(f"{prefix} missing or invalid 'id'.")
        elif a_id in seen_ids:
            errors.append(f"{prefix} duplicate id '{a_id}'.")
        else:
            seen_ids.add(a_id)

        name = item.get("display_name")
        if not name or not isinstance(name, str):
            errors.append(f"{prefix} ('{a_id}') missing 'display_name'.")

        tier = item.get("tier")
        if tier not in (1, 2, 3):
            errors.append(f"{prefix} ('{a_id}') invalid tier '{tier}'. Must be 1, 2, or 3.")

        sources = item.get("sources")
        if not sources or not isinstance(sources, list):
            errors.append(f"{prefix} ('{a_id}') must have a non-empty list of 'sources'.")
        else:
            for s_idx, src in enumerate(sources):
                if isinstance(src, dict):
                    if not src.get("citation"):
                        errors.append(f"{prefix} ('{a_id}') source #{s_idx} missing 'citation'.")
                elif isinstance(src, str):
                    if not src.strip():
                        errors.append(f"{prefix} ('{a_id}') source #{s_idx} empty string.")
                else:
                    errors.append(f"{prefix} ('{a_id}') source #{s_idx} invalid format.")

        # Check E-numbers format
        e_nums = item.get("e_numbers", [])
        if not isinstance(e_nums, list):
            errors.append(f"{prefix} ('{a_id}') 'e_numbers' must be a list.")
        else:
            for en in e_nums:
                if not isinstance(en, str) or not e_regex.match(en.replace(" ", "")):
                    errors.append(f"{prefix} ('{a_id}') invalid E-number format '{en}'.")

        # Collect aliases and verify collision across additives
        all_aliases = []
        for en in e_nums:
            all_aliases.append(en.lower().replace(" ", ""))
        for us_n in item.get("us_names", []):
            all_aliases.append(us_n.lower().strip())
        for syn in item.get("synonyms", []):
            all_aliases.append(syn.lower().strip())

        for alias in all_aliases:
            if not alias:
                continue
            if alias in alias_to_id and alias_to_id[alias] != a_id:
                errors.append(
                    f"Alias collision for '{alias}': shared between '{alias_to_id[alias]}' and '{a_id}'."
                )
            alias_to_id[alias] = a_id

    return errors


_CACHED_DATA: Optional[Dict[str, Any]] = None
_DETECTOR_CACHE: Optional[AdditiveDetector] = None


def load_additive_data(file_path: Optional[Path] = None, force_reload: bool = False) -> Dict[str, Any]:
    """Load and validate the additive knowledge base."""
    global _CACHED_DATA
    if _CACHED_DATA is not None and not force_reload and file_path is None:
        return _CACHED_DATA

    p = Path(file_path) if file_path is not None else DATA_FILE
    if not p.exists():
        raise FileNotFoundError(f"Additive data file not found at: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    errors = validate_additive_data(data)
    if errors:
        raise AdditiveValidationError(f"Additive data validation failed ({len(errors)} errors):\n" + "\n".join(errors))

    if file_path is None:
        _CACHED_DATA = data
    return data


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute exact Levenshtein distance between two short strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


class AdditiveDetector:
    """
    Pure-Python detector engine for food additives in ingredient texts.
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        if data is None:
            data = load_additive_data()
        self.data = data
        self.additives = {a["id"]: a for a in data.get("additives", [])}

        self._build_indexes()

    def _build_indexes(self):
        # E-number map: normalized string -> additive_id
        # e.g., 'e102' -> 'tartrazine', 'ins102' -> 'tartrazine'
        self.e_map: Dict[str, str] = {}
        # Known pure numeric codes for scanner validation (e.g. '102', '160b')
        self.known_e_codes: Set[str] = set()

        for a_id, a in self.additives.items():
            for en in a.get("e_numbers", []):
                norm = en.lower().replace(" ", "").replace("-", "")
                self.e_map[norm] = a_id
                # Strip leading 'e' to get number code
                code = norm.lstrip("e")
                self.known_e_codes.add(code)
            for ins in a.get("ins_numbers", []):
                norm_ins = f"ins{ins.lower().replace(' ', '').replace('-', '')}"
                self.e_map[norm_ins] = a_id
                self.known_e_codes.add(ins.lower().replace(" ", ""))

        # US Colour Names map
        # e.g., 'yellow 5' -> 'tartrazine', 'red 40' -> 'allura_red'
        self.us_colour_map: Dict[str, str] = {
            "yellow 5": "tartrazine",
            "yellow no 5": "tartrazine",
            "yellow no. 5": "tartrazine",
            "yellow #5": "tartrazine",
            "fd&c yellow 5": "tartrazine",
            "fd&c yellow no 5": "tartrazine",
            "fd&c yellow no. 5": "tartrazine",
            "yellow 6": "sunset_yellow",
            "yellow no 6": "sunset_yellow",
            "yellow no. 6": "sunset_yellow",
            "yellow #6": "sunset_yellow",
            "fd&c yellow 6": "sunset_yellow",
            "fd&c yellow no 6": "sunset_yellow",
            "fd&c yellow no. 6": "sunset_yellow",
            "red 40": "allura_red",
            "red no 40": "allura_red",
            "red no. 40": "allura_red",
            "red #40": "allura_red",
            "fd&c red 40": "allura_red",
            "fd&c red no 40": "allura_red",
            "fd&c red no. 40": "allura_red",
            "blue 1": "brilliant_blue",
            "blue no 1": "brilliant_blue",
            "blue no. 1": "brilliant_blue",
            "blue #1": "brilliant_blue",
            "fd&c blue 1": "brilliant_blue",
            "fd&c blue no 1": "brilliant_blue",
            "fd&c blue no. 1": "brilliant_blue",
            "blue 2": "indigo_carmine",
            "blue no 2": "indigo_carmine",
            "blue no. 2": "indigo_carmine",
            "blue #2": "indigo_carmine",
            "fd&c blue 2": "indigo_carmine",
            "fd&c blue no 2": "indigo_carmine",
            "fd&c blue no. 2": "indigo_carmine",
        }

        # Names & synonyms index
        self.exact_phrase_map: Dict[str, str] = {}
        self.fuzzy_candidates: List[Tuple[str, str]] = []  # (phrase >= 8 chars, a_id)

        for a_id, a in self.additives.items():
            names = [a.get("display_name", "")] + a.get("synonyms", []) + a.get("us_names", [])
            for n in names:
                if not n:
                    continue
                # Normalize spaces and lowercase
                n_norm = " ".join(n.lower().split())
                self.exact_phrase_map[n_norm] = a_id
                if len(n_norm) >= 8 and " " not in n_norm:
                    self.fuzzy_candidates.append((n_norm, a_id))

        # Compile master regex for exact multi-word / single-word phrases
        # Sort descending by length to prefer longest match first
        all_phrases = sorted(self.exact_phrase_map.keys(), key=lambda x: -len(x))
        escaped_phrases = [re.escape(p) for p in all_phrases]
        # Allow flexible whitespace between words
        regex_phrases = [p.replace(r"\ ", r"\s+") for p in escaped_phrases]
        pattern_str = r"\b(?:" + "|".join(regex_phrases) + r")\b"
        self.phrase_regex = re.compile(pattern_str, re.IGNORECASE)

        # Regex for US colour patterns
        self.us_colour_regex = re.compile(
            r"\b(?:FD&C\s+)?(Yellow|Red|Blue)\s*(?:No\.?|#)?\s*(\d+)(?:\s+Lake)?\b",
            re.IGNORECASE
        )

        # Candidate E-number / INS-number regex with scanner OCR confusions
        # Captures: E or INS followed by digits/letters/parentheses
        self.cand_e_regex = re.compile(
            r"(?<![a-zA-Z0-9])(?:E|INS)[\s\-]*(?:[0-9oOlI\s]{2,6}[a-zA-Z]?)(?:\([a-zA-Z0-9]+\))?",
            re.IGNORECASE
        )

        # Compile negation and trace regexes
        self.neg_before_regex = re.compile(r"(?:" + "|".join(NEGATION_BEFORE_PATTERNS) + r")$", re.IGNORECASE)
        self.neg_after_regex = re.compile(r"^(?:" + "|".join(NEGATION_AFTER_PATTERNS) + r")", re.IGNORECASE)
        self.trace_regex = re.compile(r"(?:" + "|".join(TRACE_PATTERNS) + r")", re.IGNORECASE)

    def _is_negated(self, text: str, start: int, end: int) -> bool:
        """
        Check if the match at text[start:end] is negated (meaning absence).
        Examines up to 60 characters before and 30 characters after within the current clause.
        """
        # Look behind up to the last clause delimiter (. ; \n)
        pre_chunk = text[max(0, start - 60):start]
        # If clause boundary exists before match, start from boundary
        clause_break = max(pre_chunk.rfind("."), pre_chunk.rfind(";"), pre_chunk.rfind("\n"))
        if clause_break != -1:
            pre_chunk = pre_chunk[clause_break + 1:]

        # If immediately preceded by 'contains' without negation, it is affirmative
        if re.search(r"\bcontains?\s*$", pre_chunk, re.IGNORECASE):
            return False

        # Check negation before
        if self.neg_before_regex.search(pre_chunk):
            return True

        # Look ahead up to the next clause delimiter
        post_chunk = text[end:min(len(text), end + 35)]
        clause_break_post = min(
            [idx for idx in (post_chunk.find("."), post_chunk.find(";"), post_chunk.find(","), post_chunk.find("\n")) if idx != -1]
            or [len(post_chunk)]
        )
        post_chunk = post_chunk[:clause_break_post]

        # Check negation after (e.g. "sulphite free", "preservative free")
        if self.neg_after_regex.search(post_chunk):
            return True

        return False

    def _is_trace(self, text: str, start: int, end: int) -> bool:
        """
        Check if match appears in a precautionary trace / 'may contain' statement.
        """
        pre_chunk = text[max(0, start - 90):start]
        clause_break = max(pre_chunk.rfind("."), pre_chunk.rfind(";"), pre_chunk.rfind("\n"))
        if clause_break != -1:
            pre_chunk = pre_chunk[clause_break + 1:]

        return bool(self.trace_regex.search(pre_chunk))

    def _heal_candidate_e_number(self, raw_match: str) -> Tuple[Optional[str], bool]:
        """
        Heal candidate E-number scanner OCR confusions.
        Returns (normalized_e_code, was_healed) or (None, False).
        """
        s = raw_match.strip()
        prefix = "ins" if s.lower().startswith("ins") else "e"
        after_prefix = s[3:] if prefix == "ins" else s[1:]
        after_prefix = after_prefix.lstrip(" -")

        # Check for confusions: letter O->0, l/I->1, spaces removed
        healed_str = ""
        was_healed = False

        for char in after_prefix:
            if char in ("O", "o"):
                healed_str += "0"
                was_healed = True
            elif char in ("l", "I"):
                healed_str += "1"
                was_healed = True
            elif char.isspace():
                was_healed = True
            else:
                healed_str += char

        norm_code = healed_str.lower()
        # Verify valid format: numbers optionally followed by sub-letter or (letter)
        # e.g. 102, 160b, 160(b), 150d
        norm_key = f"{prefix}{norm_code}".replace("(", "").replace(")", "")
        return norm_key, was_healed

    def detect(
        self,
        text: str,
        structured_tags: Optional[List[str]] = None,
        user_sensitivities: Optional[Set[str]] = None,
        user_has_asthma: bool = False,
    ) -> List[AdditiveMatch]:
        """
        Scan ingredient text for additives and return a deduplicated list of AdditiveMatch.
        """
        if not text and not structured_tags:
            return []

        user_sensitivities = {s.lower().strip() for s in (user_sensitivities or set())}
        raw_text = text or ""
        clean_text = unicodedata.normalize("NFKD", raw_text)

        # Temporary collection: additive_id -> list of AdditiveOccurrence
        hits: Dict[str, List[AdditiveOccurrence]] = {}

        # -------------------------------------------------------------
        # 1. Candidate E-numbers & INS numbers
        # -------------------------------------------------------------
        for m in self.cand_e_regex.finditer(clean_text):
            matched_span = m.span()
            raw_str = m.group(0)

            # Guard 1: Must not be preceded by Vitamin / Vit
            pre_start = max(0, matched_span[0] - 15)
            pre_window = clean_text[pre_start:matched_span[0]]
            if re.search(r"\bvit(?:amin|\.)?\s*$", pre_window, re.IGNORECASE):
                continue

            # Guard 2: Must not be immediately followed by measurement units (e.g. "E 120 g")
            post_end = min(len(clean_text), matched_span[1] + 12)
            post_window = clean_text[matched_span[1]:post_end]
            if re.match(r"^\s*(?:g|mg|kg|ml|l|oz|lb|%|cm|mm)\b", post_window, re.IGNORECASE):
                continue

            norm_key, was_healed = self._heal_candidate_e_number(raw_str)
            if not norm_key:
                continue

            # Check if this E-number or INS exists in our database
            a_id = self.e_map.get(norm_key)
            if not a_id:
                # Try with parentheses format e.g. e160(b) -> e160b
                norm_alt = re.sub(r"[\(\)]", "", norm_key)
                a_id = self.e_map.get(norm_alt)

            if not a_id:
                continue

            # Check negation
            if self._is_negated(clean_text, matched_span[0], matched_span[1]):
                continue

            is_tr = self._is_trace(clean_text, matched_span[0], matched_span[1])
            conf = 0.85 if was_healed else 1.0

            occ = AdditiveOccurrence(
                matched_text=raw_str,
                match_kind="e_number",
                position=matched_span,
                is_trace=is_tr,
                confidence=conf,
            )
            hits.setdefault(a_id, []).append(occ)

        # -------------------------------------------------------------
        # 2. US Colour Names
        # -------------------------------------------------------------
        for m in self.us_colour_regex.finditer(clean_text):
            raw_str = m.group(0)
            col_name = m.group(1).lower()
            num_str = m.group(2)
            key = f"{col_name} {num_str}"
            a_id = self.us_colour_map.get(key)
            if not a_id:
                continue

            matched_span = m.span()
            if self._is_negated(clean_text, matched_span[0], matched_span[1]):
                continue

            is_tr = self._is_trace(clean_text, matched_span[0], matched_span[1])
            occ = AdditiveOccurrence(
                matched_text=raw_str,
                match_kind="us_colour_name",
                position=matched_span,
                is_trace=is_tr,
                confidence=1.0,
            )
            hits.setdefault(a_id, []).append(occ)

        # -------------------------------------------------------------
        # 3. Names and Synonyms (Exact Phrase Regex)
        # -------------------------------------------------------------
        for m in self.phrase_regex.finditer(clean_text):
            raw_str = m.group(0)
            phrase_norm = " ".join(raw_str.lower().split())
            a_id = self.exact_phrase_map.get(phrase_norm)
            if not a_id:
                continue

            matched_span = m.span()
            if self._is_negated(clean_text, matched_span[0], matched_span[1]):
                continue

            is_tr = self._is_trace(clean_text, matched_span[0], matched_span[1])
            occ = AdditiveOccurrence(
                matched_text=raw_str,
                match_kind="name" if phrase_norm == self.additives[a_id]["display_name"].lower() else "synonym",
                position=matched_span,
                is_trace=is_tr,
                confidence=1.0,
            )
            hits.setdefault(a_id, []).append(occ)

        # -------------------------------------------------------------
        # 4. Levenshtein 1-char difference for words >= 8 letters
        # -------------------------------------------------------------
        # Extract unique word tokens of length >= 8 to keep it blazing fast (<15ms even on 5000+ words)
        candidate_words = set(re.findall(r"\b[a-zA-Z]{8,}\b", clean_text))
        matched_fuzzy_targets: Dict[str, Tuple[str, str]] = {}  # token.lower() -> (target_word, a_id)

        for raw_token in candidate_words:
            token = raw_token.lower()
            # If already an exact phrase/synonym, skip fuzzy check
            if token in self.exact_phrase_map:
                continue
            for target_word, a_id in self.fuzzy_candidates:
                if abs(len(token) - len(target_word)) > 1:
                    continue
                if _levenshtein_distance(token, target_word) == 1:
                    matched_fuzzy_targets[token] = (target_word, a_id)
                    break

        if matched_fuzzy_targets:
            for w_match in re.finditer(r"\b[a-zA-Z]{8,}\b", clean_text):
                token = w_match.group(0).lower()
                if token in matched_fuzzy_targets:
                    target_word, a_id = matched_fuzzy_targets[token]
                    token_span = w_match.span()
                    # Skip if already inside an existing match
                    already_covered = False
                    for occ_list in hits.values():
                        for occ in occ_list:
                            if occ.position[0] <= token_span[0] and token_span[1] <= occ.position[1]:
                                already_covered = True
                                break
                        if already_covered:
                            break
                    if not already_covered and not self._is_negated(clean_text, token_span[0], token_span[1]):
                        is_tr = self._is_trace(clean_text, token_span[0], token_span[1])
                        hits.setdefault(a_id, []).append(
                            AdditiveOccurrence(
                                matched_text=w_match.group(0),
                                match_kind="synonym",
                                position=token_span,
                                is_trace=is_tr,
                                confidence=0.85,
                            )
                        )

        # -------------------------------------------------------------
        # 5. Merge structured tags (e.g. from Open Food Facts)
        # -------------------------------------------------------------
        if structured_tags:
            for tag in structured_tags:
                clean_tag = tag.lower().replace("en:", "").replace(" ", "").replace("-", "")
                a_id = self.e_map.get(clean_tag)
                if a_id:
                    hits.setdefault(a_id, []).append(
                        AdditiveOccurrence(
                            matched_text=tag,
                            match_kind="structured_tag",
                            position=(-1, -1),
                            is_trace=False,
                            confidence=1.0,
                        )
                    )

        # -------------------------------------------------------------
        # 6. Build Deduplicated AdditiveMatch List
        # -------------------------------------------------------------
        results: List[AdditiveMatch] = []
        for a_id, occurrences in hits.items():
            a_data = self.additives[a_id]
            # Primary occurrence is the highest confidence, earliest occurrence
            occurrences.sort(key=lambda o: (-o.confidence, o.position[0]))
            primary = occurrences[0]

            # Trace flag: True if ALL occurrences are in trace statements
            is_all_trace = all(o.is_trace for o in occurrences if o.match_kind != "structured_tag")

            # Check user sensitivity
            is_listed = (
                a_id in user_sensitivities
                or a_data["display_name"].lower() in user_sensitivities
                or any(syn.lower() in user_sensitivities for syn in a_data.get("synonyms", []))
            )

            # Check asthma relevance (especially for sulfites)
            asthma_relevant = user_has_asthma and a_id in ("sulfites", "sulfite_caramel")

            results.append(
                AdditiveMatch(
                    additive_id=a_id,
                    display_name=a_data["display_name"],
                    matched_text=primary.matched_text,
                    match_kind=primary.match_kind,
                    position=primary.position,
                    is_trace=is_all_trace,
                    confidence=primary.confidence,
                    tier=a_data["tier"],
                    data=a_data,
                    all_occurrences=occurrences,
                    is_user_listed=is_listed,
                    asthma_relevant=asthma_relevant,
                )
            )

        # Sort results: User listed first, then by tier ascending, then display name
        results.sort(key=lambda r: (not r.is_user_listed, r.tier, r.display_name))
        return results


def get_detector() -> AdditiveDetector:
    """Singleton getter for AdditiveDetector."""
    global _DETECTOR_CACHE
    if _DETECTOR_CACHE is None:
        _DETECTOR_CACHE = AdditiveDetector()
    return _DETECTOR_CACHE


def detect_additives(
    text: str,
    structured_tags: Optional[List[str]] = None,
    user_sensitivities: Optional[Set[str]] = None,
    user_has_asthma: bool = False,
) -> List[AdditiveMatch]:
    """Convenience helper to detect additives in an ingredient text."""
    return get_detector().detect(
        text=text,
        structured_tags=structured_tags,
        user_sensitivities=user_sensitivities,
        user_has_asthma=user_has_asthma,
    )


def detect_additives_multi_text(
    texts: List[str],
    structured_tags: Optional[List[str]] = None,
    user_sensitivities: Optional[Set[str]] = None,
    user_has_asthma: bool = False,
) -> List[AdditiveMatch]:
    """
    Run detection across multiple ingredient texts (e.g. original OCR text + translated text),
    unioning occurrences so that translations dropping an E-number or name do not miss an additive.
    """
    detector = get_detector()
    combined_hits: Dict[str, List[AdditiveOccurrence]] = {}

    for txt in texts:
        if not txt:
            continue
        sub_matches = detector.detect(
            text=txt,
            structured_tags=None,
            user_sensitivities=user_sensitivities,
            user_has_asthma=user_has_asthma,
        )
        for m in sub_matches:
            combined_hits.setdefault(m.additive_id, []).extend(m.all_occurrences)

    # Add structured tags if any
    if structured_tags:
        for tag in structured_tags:
            clean_tag = tag.lower().replace("en:", "").replace(" ", "").replace("-", "")
            a_id = detector.e_map.get(clean_tag)
            if a_id:
                combined_hits.setdefault(a_id, []).append(
                    AdditiveOccurrence(
                        matched_text=tag,
                        match_kind="structured_tag",
                        position=(-1, -1),
                        is_trace=False,
                        confidence=1.0,
                    )
                )

    # Deduplicate and build final matches
    user_sens = {s.lower().strip() for s in (user_sensitivities or set())}
    final_matches: List[AdditiveMatch] = []

    for a_id, occurrences in combined_hits.items():
        a_data = detector.additives[a_id]
        occurrences.sort(key=lambda o: (-o.confidence, o.position[0]))
        primary = occurrences[0]
        is_all_trace = all(o.is_trace for o in occurrences if o.match_kind != "structured_tag")
        is_listed = (
            a_id in user_sens
            or a_data["display_name"].lower() in user_sens
            or any(syn.lower() in user_sens for syn in a_data.get("synonyms", []))
        )
        asthma_relevant = user_has_asthma and a_id in ("sulfites", "sulfite_caramel")

        final_matches.append(
            AdditiveMatch(
                additive_id=a_id,
                display_name=a_data["display_name"],
                matched_text=primary.matched_text,
                match_kind=primary.match_kind,
                position=primary.position,
                is_trace=is_all_trace,
                confidence=primary.confidence,
                tier=a_data["tier"],
                data=a_data,
                all_occurrences=occurrences,
                is_user_listed=is_listed,
                asthma_relevant=asthma_relevant,
            )
        )

    final_matches.sort(key=lambda r: (not r.is_user_listed, r.tier, r.display_name))
    return final_matches


def render_additive_card_html(m: AdditiveMatch) -> str:
    """
    Render a single additive detail card as escaped HTML.
    Never uses red, warning banners, or alarming icons.
    """
    tier_labels = {
        1: "Tier 1: Regulated / well documented",
        2: "Tier 2: Case reports of allergic reactions",
        3: "Tier 3: Commonly blamed (not confirmed in blinded tests)",
    }
    tier_label = tier_labels.get(m.tier, f"Tier {m.tier}")

    # Build pills
    pills = [f'<span class="pill">{html.escape(tier_label)}</span>']
    if m.is_user_listed:
        pills.append('<span class="pill pill-user">You listed this</span>')
    if m.is_trace:
        pills.append('<span class="pill">Precautionary trace</span>')

    # Words found
    found_words = sorted(list(set(occ.matched_text for occ in m.all_occurrences)))
    found_str = ", ".join(found_words) if found_words else m.matched_text

    # E-numbers / US names display
    e_nums = m.data.get("e_numbers", [])
    us_names = m.data.get("us_names", [])
    numbers_str = ", ".join(e_nums) if e_nums else ""
    if us_names:
        numbers_str = f"{numbers_str} (US: {', '.join(us_names[:2])})" if numbers_str else f"US: {', '.join(us_names[:2])}"

    cls_str = f"What it is: {html.escape(m.data.get('class', '').capitalize())} ({html.escape(m.data.get('origin', ''))})"
    if numbers_str:
        cls_str += f" &bull; {html.escape(numbers_str)}"

    groups = ", ".join(m.data.get("relevant_groups", []))
    reactions = ", ".join(m.data.get("reactions", []))
    foods = ", ".join(m.data.get("related_foods", []))

    # Sources links
    source_links = []
    for s in m.data.get("sources", []):
        if isinstance(s, dict):
            c = html.escape(s.get("citation", "Source"))
            u = html.escape(s.get("url", "#"))
            source_links.append(f'<a href="{u}" target="_blank" rel="noopener noreferrer" style="color:#4B5B73;text-decoration:underline;">{c}</a>')
        elif isinstance(s, str):
            source_links.append(html.escape(s))
    sources_html = " &bull; ".join(source_links) if source_links else "Official additive registers"

    # Specific advisory lines
    extra_notes = []
    if m.asthma_relevant:
        extra_notes.append(
            '<div style="margin-top:6px;padding:6px 10px;background:#EAF0F6;border-radius:4px;font-size:0.84rem;">'
            '<strong>Asthma note:</strong> Sulfites are the additive most often linked to breathing symptoms in people with asthma.</div>'
        )
    if m.tier == 3:
        weak_note = m.data.get("weak_evidence_note") or "Controlled double-blind placebo challenges have not confirmed a causal link."
        extra_notes.append(
            f'<div style="margin-top:6px;padding:6px 10px;background:#F0F4F8;border-radius:4px;font-size:0.84rem;color:#4B5B73;">'
            f'<strong>Clinical evidence:</strong> {html.escape(weak_note)}</div>'
        )

    extra_html = "".join(extra_notes)

    card = f"""<div class="additive-card">
  <div class="title">{html.escape(m.display_name)}</div>
  <div style="margin-bottom:8px;">{"".join(pills)}</div>
  <div style="color:#4B5B73;font-size:0.84rem;margin-bottom:6px;">{cls_str}</div>
  <div style="margin-bottom:4px;"><strong>Words found:</strong> <code>{html.escape(found_str)}</code></div>
  <div style="margin-bottom:4px;"><strong>Overview:</strong> {html.escape(m.data.get("description", ""))}</div>
  <div style="margin-bottom:4px;"><strong>Reported reactions:</strong> {html.escape(reactions)}</div>
  <div style="margin-bottom:4px;"><strong>Mainly relevant for:</strong> {html.escape(groups)}</div>
  <div style="margin-bottom:4px;"><strong>How rare & evidence:</strong> {html.escape(m.data.get("rarity", ""))}</div>
  {f'<div style="margin-bottom:4px;"><strong>Commonly found in:</strong> {html.escape(foods)}</div>' if foods else ''}
  {extra_html}
  <div style="margin-top:6px;font-size:0.80rem;color:#6C7A8E;"><strong>Sources:</strong> {sources_html}</div>
  <div class="disclaimer">{MEDICAL_DISCLAIMER}</div>
</div>"""
    return card


def format_additive_line(matches: List[AdditiveMatch], show_weak: bool = False) -> str:
    """
    Format the single calm note line to display under the main verdict.
    Example: "Additives that rarely cause reactions in some people: Tartrazine (E102), Sulphites"
    """
    filtered = [m for m in matches if m.tier < 3 or show_weak or m.is_user_listed]
    if not filtered:
        return ""

    names = []
    for m in filtered:
        # Show name and primary E-number if available
        e_nums = m.data.get("e_numbers", [])
        if e_nums:
            names.append(f"{m.display_name} ({e_nums[0]})")
        else:
            names.append(m.display_name)

    return f"Additives that rarely cause reactions in some people: {', '.join(names)}"
