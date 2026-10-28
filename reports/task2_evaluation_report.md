# Task 2 Final Report: "Other Allergens" Input & Custom Allergen Engine

**Date:** 2026-10-27  
**Branch:** `fix-other-allergens`  
**Test Suite Status:** 51 passed / 0 failed  
**Environment:** Python 3.12.3, Streamlit 1.65.0, Headless Testing via `streamlit.testing.v1.AppTest`

---

## 1. Before vs. After Behavior Matrix

| Input Term | Before Behavior (Defective) | After Behavior (Upgraded) | Clinical & User Safety Impact |
|:---|:---|:---|:---|
| `"peanuts, sesame"` | Instruction caption; 2 separate sliders | Matches **Peanut** and **Sesame Seed** as chips with ratings | Canonical match working as expected. |
| `"tree nuts"` | Caption `"Not recognised: tree nuts"`; term discarded | Expands to **Tree nuts** (10 foods) with **1 shared severity slider** and editable member checklist | Prevents dropping patient's most common allergy group; uncluttered UI. |
| `"shellfish"` | Silently reduced whole class to **shrimp alone** | Expands to **Shellfish** (15 crustaceans & molluscs) with shared slider and editable checklist | **Eliminates critical clinical under-protection bug**; protects against crabs, lobsters, clams, oysters, etc. |
| `"fish"` | Caption `"Not recognised: fish"`; term discarded | Expands to **Fish** (25 finfish species) with shared slider | All 25 fish species covered under one slider. |
| `"cow's milk"` | Caption `"Not recognised: cow's milk"`; term discarded | Matches **Cow Milk** via possessive normalization & typed alias | Punctuation / apostrophe variations resolve cleanly. |
| `"penut"` | Caption `"Not recognised: penut"`; term discarded | Shows Tier 3 suggestion: **"Did you mean Peanut?"** with "Accept" and "Keep as custom" buttons | Conservative typo recovery; never auto-accepts without confirmation. |
| `"jackfruit"` | Caption `"Not recognised: jackfruit"`; term discarded | Kept as **Custom Allergen (not in model)** with user rating (1-3); scanned in ingredient text | Rare / non-model foods are preserved, detected in products, and warned with user's severity. |
| `"peanut and egg"` | Caption `"Not recognised: peanut and egg"`; term discarded | Delimiter engine splits on `"and"`; adds both **Peanut** and **Chicken Egg** | Natural multi-item phrases parsed effortlessly. |
| `""` (empty) | Handled silently | Handled safely without exception or phantom items | No crash or phantom UI artifacts. |
| `"a" * 200` | Caption `"Not recognised: aaaaa..."` | Visible warning: `"Term is too long (maximum 40 characters)"` | Enforces sanitization and length bounds; escaped against HTML injection. |

---

## 2. Files Created or Modified

1. **`allerscan_4/src/typed_aliases.json`** *(Created)*
   - 45 typed aliases (e.g. `"cow's milk"`, `"pea nut"`, `"chicken egg"`, `"soya"`, `"rockmelon"`).
   - 25 group word mappings (`tree nuts`, `shellfish`, `seafood`, `fish`, `legumes`, `cereals`, `gluten`, `dairy`).
   - Ambiguity clarifications: `"nuts"` / `"nut"` (peanut, tree nuts, or both).
   - Over-broad classifications: `"fruit"`, `"vegetables"`, `"meat"`.

2. **`allerscan_4/src/term_resolver.py`** *(Created)*
   - Pure-Python module (zero Streamlit dependencies).
   - Input splitting across `,`, `;`, `/`, `\n`, `\r`, `•`, `+`, `&`, and words `and`, `or`.
   - Input cleaning: trimming, accent normalization, bracket removal, repeated hyphen/space collapse, leading (`"allergic to"`, `"i am allergic to"`) and trailing (`"allergy"`, `"allergies"`, `"intolerance"`, `"free"`) phrase stripping.
   - 4-Tier resolution logic:
     - Tier 1: Exact/normalized match (aliases, singular/plural, possessives, separators, KB canonicalization).
     - Tier 2: Group words & clarifications (shellfish never becomes shrimp alone).
     - Tier 3: Conservative fuzzy suggestions (`get_fuzzy_suggestions` function ready for Task 4 swapping).
     - Tier 4: Custom allergen generator with `custom_` slug prefix and 15-item cap.
   - Ingredient text custom allergen matcher (`match_custom_allergens`): whole-word, plural, multi-word phrases, 1-char difference tolerance (terms >= 6 letters), with protection against matching inside unrelated words.
   - HTML escaping on all user-supplied text.

3. **`allerscan_4/app.py`** *(Modified)*
   - Replaced dead text input with reactive `st.form` (submits on both Enter key and "Add" button, clears input after submit).
   - Session state tracking for `other_matched`, `other_groups`, `other_custom`, `other_pending`, `other_messages`.
   - Group UI: shared rating slider + editable member checklist (`st.multiselect`).
   - Matched & Custom chips: individual rating sliders + "✕" removal button that clears slider session state.
   - Results engine: runs custom matcher on ingredient text, includes custom hits in verdict banner as "your allergens", displays custom cards with user rating and explicit disclaimer (`"Not covered by the AllerScan model, so there is no estimate for related foods."`).
   - Clinical safety note triggers if any model or custom reaction is severe (level 3) for moderate/severe asthmatics.
   - Negative verdict banner clearly notes whether custom allergens were set and checked.

4. **`allerscan_4/tests/test_other_allergen_input.py`** *(Created)*
   - 20 unit and AppTest integration tests verifying every requirement from Step 8.

5. **`allerscan_4/README.md`** *(Modified)*
   - Documented the 4-tier term resolver, group expansions, fuzzy suggestions, and custom allergen engine.
   - Updated directory structure table.

6. **`allerscan_4/reports/task2_before_table.md`** & **`task2_evaluation_report.md`** *(Created)*
   - Evaluation baseline and final transition audit reports.

---

## 3. Test Suite Counts

```
======================== 51 passed, 1 warning in 5.41s =========================
allerscan_4/tests/test_cleaning_pipeline.py: 7 passed
allerscan_4/tests/test_other_allergen_input.py: 20 passed
allerscan_4/tests/test_scanner_units.py: 24 passed
Total: 51 passed (100% pass rate)
```

---

## 4. Verification Notes & Human Review Items

1. **Streamlit Headless Testing:**
   All interactions (form submission, group card rendering, remove buttons, slider updates, suggestion accept buttons, and label ingredient checking) were verified using `streamlit.testing.v1.AppTest`.
2. **Clinical Disclosures & Copy Review:**
   - Custom Allergen Card sentence: *"Not covered by the AllerScan model, so there is no estimate for related foods."*
   - Negative verdict note: *"Also checked N custom allergen(s); none were found in this text."* vs *"No custom allergens were set in your profile."*
   - Over-broad guidance: For words like `"fruit"` or `"vegetables"`, the app prompts the user to specify individual foods or keep the word as a custom allergen.
