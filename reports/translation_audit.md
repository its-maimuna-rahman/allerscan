# Task 5 Step 1: English-Assumption Audit Report

**Date**: 2026-10-27  
**Branch**: `ingredient-translation`  
**Purpose**: Comprehensive audit of all places in AllerScan that assume English text, in accordance with Step 1 of `docs/05_ingredient_translation.md`.

---

## 1. Audit Table of English Assumptions

| # | System Area / Component | File & Lines | Current English Assumption | Consequence on Non-English Labels | Required Upgrade |
|---|---|---|---|---|---|
| 1 | **OCR Reader Initialization** | `src/photo_scanner.py:270` | `easyocr.Reader(["en"], gpu=True)` hardcoded to English only. | Reads non-Latin scripts as empty strings or scrambled Latin noise. | Multi-script Reader Manager loading on-demand reader per script group paired with English. |
| 2 | **Orientation & Readability Score** | `src/photo_scanner.py:1040-1080` (`_score_reading`) | Counts English words from `FULL_TOKEN_VOCAB`, `LABEL_SAFE_WORDS`, and `ENGLISH_COMMON`. | Non-English labels receive score near 0.0. Orientation rotation fails or selects random angle. | Script-aware readability score using recognition confidence, script character frequency, and multilingual glossary matches. |
| 3 | **Ingredient Section Extraction** | `src/photo_scanner.py:843-850` (`ING_HEADER_RE`) | Regex `ING_HEADER_RE` only matches `ingredients`, `contains`, `allergens` in English. | Bengali (`উপকরণ:`), Arabic (`المكونات:`), French (`Ingrédients:`), Hindi (`सामग्री:`), Chinese (`配料:`) are missed. | Multilingual header regex table covering all Phase 1 languages. |
| 4 | **Trace & Negation Detection** | `src/photo_scanner.py:845-850` (`WARN_RE`), `src/nlp_parser.py` | `WARN_RE` matches English `may contain`, `traces of`, `facility`, `free from`. | Non-English warning statements (e.g. `sans gluten`, `ohne Milch`, `হতে পারে`) are missed or misinterpreted as direct ingredients. | Language-specific negation ("free from") and trace ("may contain") statement regexes and metadata tags. |
| 5 | **Front vs. Back Classification** | `src/photo_scanner.py:1085-1110` (`_classify_lines`) | Discriminates between ingredient lists and marketing text by counting English generic food words. | Foreign ingredient lists classified incorrectly as front-of-pack or rejected. | Multilingual keyword scoring and layout density classification independent of language. |
| 6 | **Product Database Filter** | `app.py:212-214` (`parse_off`) | Filter `sum(c.isascii() for c in letters) / len(letters) < 0.85` explicitly drops non-Latin products. | Any product in Open Food Facts with Bengali, Arabic, Russian, Chinese, or Hindi ingredients is silently discarded. | Remove ASCII filter; preserve original-language ingredients, send through translation/glossary pipeline, preferring English text when available. |
| 7 | **Text Normalization & Hyphenation** | `src/photo_scanner.py:780-835`, `src/spelling_corrector.py` | English hyphenation rules and accent-stripping logic assume Latin alphabetic text with spaces. | Fails on scripts without spaces (Thai, Chinese, Japanese) and corrupts complex Brahmic/Indic conjuncts (Bengali, Hindi). | Script-specific Unicode NFC normalisation, digit mapping (Bengali/Devanagari/Arabic-Indic to Western), punctuation unification, and no-space segmenters. |
| 8 | **Allergen Knowledge Base** | `src/name_mapping.json`, `src/nlp_parser.py` | Knowledge base entries are ~1,600 English food and ingredient names. | Direct parsing of foreign ingredient words returns 0 allergens. | Multilingual glossary (Route A) providing direct mapping from foreign terms to canonical allergen keys. |
