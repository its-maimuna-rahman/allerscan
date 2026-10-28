# AllerScan Spelling Vocabulary Build and Reference

This document describes the unified vocabulary dataset built for AllerScan's spelling correction module (`src/spelling_corrector.py`), its data sources, licensing, and how to rebuild it reproducibly.

---

## 1. Sources and Licensing

| Dataset / Component | Description | License / Citation |
|---|---|---|
| **Open Food Facts Ingredients Taxonomy** | 6,695 ingredient concepts, English names, and taxonomy relationships (`data/openfoodfacts_ingredients.json`). | **Open Database License (ODbL) v1.0** (https://opendatacommons.org/licenses/odbl/1.0/). Attribution: Open Food Facts contributors (https://world.openfoodfacts.org). Cached on disk; never fetched at runtime. |
| **AllerScan Allergen Knowledge Base** | 148 canonical allergen keys and ~1,600 synonyms (`src/name_mapping.json`), plus hidden allergen derivatives (`src/nlp_parser.py: HIDDEN_KB`) and plant beverage exceptions (`PLANT_PHRASES`). | Internal AllerScan Knowledge Base. |
| **Food Additive Hypersensitivity Database** | 34 additive classes, E-numbers (E220-E228, E621, etc.), US names, and synonyms (`data/additive_warnings.json`). | Internal AllerScan Additives Database. |
| **Food Label & Packaging Lexicons** | Structural words (`ingredients`, `contains`, `may contain`, `traces of`, `facility`), common food words, label safety terms, and common packaging words (`photo_scanner.py`). | Internal AllerScan Lexicons. |
| **Ingredient Ground Truth & Recipes** | Sample ingredient label texts and recipe templates (`tools/scan_data/*.txt`, `src/food_fetcher.py`). | Internal AllerScan Data. |

---

## 2. Rebuilding the Vocabulary

The vocabulary is version-controlled and loaded from disk at application startup. No network calls occur at runtime.

To rebuild the vocabulary from scratch:
```bash
python tools/build_spelling_vocab.py
```

This updates three versioned files in `data/spelling/`:
1. `data/spelling/spelling_vocab.json`: Full structured dictionary containing allergen mappings, additive IDs, harmless food words, unigrams, and bigrams.
2. `data/spelling/frequency_dictionary.txt`: Formatted unigram frequency dictionary for SymSpell symmetric delete lookup.
3. `data/spelling/bigram_dictionary.txt`: Context bigram frequencies for noisy channel scoring.

---

## 3. Dataset Sizes and Statistics

Summary from `build_spelling_vocab.py`:
- **Allergen Terms**: 2,095
- **Allergen Single Words**: 1,039
- **Additive Terms**: 383
- **Harmless Words**: 9,342 (includes food label lexicons, Open Food Facts taxonomy, and `FULL_TOKEN_VOCAB`)
- **Unigram Frequency Entries**: 9,523
- **Bigram Context Entries**: 1,541
- **Protected Words**: 10,901 (`data/spelling/protected_words.json`)
- **Allergen Prior Frequency Floor**: 250 (core allergens boosted to 1,000) so rare harmless words can never outrank allergens.
