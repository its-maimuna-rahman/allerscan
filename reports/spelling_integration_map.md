# AllerScan Spelling Corrector Integration Map

This document maps every point where text enters AllerScan and specifies how and when the new unified spelling corrector module (`src/spelling_corrector.py`) is integrated, satisfying the Step 1 Gate of `docs/04_spelling_correction.md`.

---

## Text Entry Points and Integration Plan

| # | Entry Point | File / Component | Mode | Timing | Integration Details & Safety Rules |
|---|---|---|---|---|---|
| 1 | **Pasted or Typed Ingredients** | `app.py` (`text_area` "Ingredients") | **Typed mode** | **NOW** | Runs before `parse_label`. Corrected text is passed to parser and additive detector. Confident corrections are displayed with original vs corrected words ("Spelling fixed: mlik → milk"). Ambiguous or low-margin allergen candidates are flagged as "Possible (spelling unclear)" with user confirmation buttons ("Yes, it is X" / "No, ignore") and never auto-applied as confirmed allergens. Protected words are never modified. |
| 2 | **Typed Allergen Names** | `src/term_resolver.py` (`get_fuzzy_suggestions` / Tier 3) | **Name mode** | **NOW** | Replaces custom `difflib` logic in Tier 3 with the unified corrector's `suggest_name()`. Stricter threshold (>=80% or low edit distance, length >= 4), returns up to 3 ranked suggestions, and **never auto-applies**. |
| 3 | **Photo Scanner Output** | `src/photo_scanner.py` (`merge_multiline_text`, `heal_line_tokens`) | **Scanned mode** | **NOW** | Replaces scanner's ad-hoc repair functions with the unified corrector in scanned mode (which applies OCR-specific substitution costs: rn→m, cl→d, 0→o, 1→l/i, etc.). Low-confidence allergen candidates are sent to uncertain allergens list. Scanner unit tests and benchmarks are validated for zero regressions. |
| 4 | **Additive Hypersensitivity Detector** | `src/additive_detector.py` | **Hybrid / Fed by Corrector** | **NOW** | Strict E-number regexes (`E\d{3,4}[a-z]?`) continue to run on raw/normalized tokens so numbers are never altered. Additive chemical names benefit from receiving the corrected text as an auxiliary input, unioning findings safely. |
| 5 | **Product Search Box** | `app.py` (`tab_search`) | **Search mode** | **NOW (Conservative)** | Product brand names (e.g., "Oreo", "Nutella") are **never auto-corrected**. If a search returns zero results, conservative "Did you mean?" suggestions are offered using the harmless food vocabulary only. |
| 6 | **Translated Label Text (Task 5)** | Translation module / hook | **Hook / LATER** | **LATER** | Correction hook provided for English translated text. Non-English source text will be skipped until dedicated target-language lexicons exist. |

---

## Architectural Guiding Principles (from Rules 1-5)

1. **One Source of Truth**: A single module (`src/spelling_corrector.py`) manages all vocabulary, indexes, confusable protections, and scoring models.
2. **Safety Asymmetry**: A silent miss is 10x worse than a "possible" flag, but a harmful false correction (rewriting a harmless food into an allergen) is equally dangerous. Protected words are guarded by hard negative constraints.
3. **Transparency**: Every modification is recorded with original token, replacement, confidence, and position, displayed clearly to the user.
