# Spelling Corrector Modes and Initial Threshold Configuration

As specified in Step 2 of `04_spelling_correction.md`, the corrector supports three distinct operational modes, each with calibrated thresholds and error models.

---

## 1. Mode Definitions

### Mode 1: Typed Mode
- **Target Use**: Text entered or pasted into the ingredient text box by a user.
- **Error Characteristics**: Fast typing slips (adjacent QWERTY keys), transposed letters (`whaet`), double/dropped letters (`peenut`, `shrimpp`, `allmonds`), and common phonetic representations (`casien`, `glutten`, `pinut`).
- **Safety Policy**:
  - Confident single winner with high margin (`>= 0.15` over runner-up) -> Auto-corrected and visually flagged in the UI ("Spelling fixed: mlik → milk").
  - Ambiguous among allergens (e.g., `cashw` could be cashew) -> Never auto-applied; reported as **Possible allergen (spelling unclear)**.
  - Ambiguous between allergen and non-allergen -> Reported as **Possible allergen**, never confirmed.
  - Protected words -> Untouched.

### Mode 2: Scanned Mode
- **Target Use**: Text extracted from OCR on physical food packaging labels.
- **Error Characteristics**: Optical character recognition confusion pairs:
  - Digit-to-letter: `0` ↔ `o`, `1` ↔ `l` / `i`, `5` ↔ `s`, `8` ↔ `b`, `3` ↔ `e`.
  - Ligatures / split glyphs: `rn` ↔ `m`, `cl` ↔ `d`, `vv` ↔ `w`.
  - Curved letter confusions: `e` ↔ `c`, `o` ↔ `c`, `u` ↔ `v`.
  - Stray spaces breaking words (`l actose`, `whe at`) or joined words (`skimmedmilk`).
- **Safety Policy**:
  - Conservative correction for verified scanner confusions.
  - Low-confidence or ambiguous corrections pass to the scanner's `uncertain_allergens` pipeline.

### Mode 3: Name Mode
- **Target Use**: Single typed allergen names in Task 2 profile setup (`term_resolver.py`).
- **Error Characteristics**: User typing individual allergens (e.g. `penut`, `treenuts`, `shrip`).
- **Safety Policy**:
  - Strictly **no auto-accept**.
  - Returns up to 3 ranked suggestions (`did you mean ...?`).
  - Stricter threshold (`similarity >= 0.80` for length >= 7, `>= 0.85` for length < 7).
  - Enforces first-letter matching or verified phonetic confusable pairs (`c/k`, `s/c`, `f/p`).
  - Drops tokens under 4 characters.

---

## 2. Thresholds and Cost Parameters

These values are defined in `SpellingConfig` and treated as tunable parameters:

| Parameter | Typed Mode | Scanned Mode | Name Mode | Description |
|---|---|---|---|---|
| `max_edit_distance` (short words <= 5) | 1 | 1 | 1 | Edit distance cap for words <= 5 letters |
| `max_edit_distance` (long words > 5) | 2 | 2 | 2 | Edit distance cap for words > 5 letters |
| `min_token_len` | 4 | 3 | 4 | Minimum length of unknown token to consider for correction |
| `adjacent_key_cost` | 0.6 | 1.0 | 0.8 | Cost for QWERTY keyboard neighbor substitution |
| `transposition_cost` | 0.7 | 0.9 | 0.8 | Cost for adjacent letter swap (e.g., `whaet`) |
| `double_dropped_cost` | 0.6 | 0.8 | 0.7 | Cost for dropped or doubled repeated letters |
| `scanner_confusion_cost` | 1.0 | 0.4 | 1.0 | Cost for OCR glyph confusion (`rn`↔`m`, `0`↔`o`, `1`↔`l`) |
| `standard_edit_cost` | 1.0 | 1.0 | 1.0 | Base Levenshtein insertion/deletion/substitution cost |
| `confident_threshold` | 0.72 | 0.70 | 0.80 (short: 0.85) | Minimum normalized score required |
| `margin_threshold` | 0.15 | 0.12 | N/A | Score margin over runner-up required for confident correction |
| `allergen_prior_floor` | 100.0 | 100.0 | 100.0 | Minimum smoothed frequency for any allergen vocabulary entry |
| `auto_apply` | True | True | False | Whether confident corrections are applied to text |
