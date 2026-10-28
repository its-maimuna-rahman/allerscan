# Task 2 Baseline: "Other Allergens" Input Evaluation

**Date:** 2026-10-27  
**Branch:** `fix-other-allergens`  
**Environment:** Streamlit 1.65.0 (Python 3.12.3)  
**Testing Mechanism:** Streamlit Headless App Testing (`streamlit.testing.v1.AppTest`)

---

## 1. Before Table (Observed Behaviour in Existing Code)

| Input Term | Observed Sidebar Feedback | Resulting Profile Sliders | Defect / Failure Mode |
|---|---|---|---|
| `"peanuts, sesame"` | Instruction caption | `['Peanut', 'Sesame Seed']` | Only comma-separated canonical items succeed. |
| `"tree nuts"` | Caption: `"Not recognised: tree nuts"` | None (`[]`) | Group term dropped; user left unprotected. |
| `"shellfish"` | Instruction caption | `['Shrimp']` | **Severe clinical defect:** Whole class reduced to shrimp alone. |
| `"fish"` | Caption: `"Not recognised: fish"` | None (`[]`) | Common group category rejected. |
| `"cow's milk"` | Caption: `"Not recognised: cow's milk"` | None (`[]`) | Punctuation/possessive causes failure. |
| `"penut"` | Caption: `"Not recognised: penut"` | None (`[]`) | Minor typo fails; no suggestion offered. |
| `"jackfruit"` | Caption: `"Not recognised: jackfruit"` | None (`[]`) | Non-model food rejected and ignored. |
| `"peanut and egg"` | Caption: `"Not recognised: peanut and egg"` | None (`[]`) | Non-comma separator ("and") fails entirely. |
| `""` (empty) | No message | None (`[]`) | Handled silently. |
| `"a" * 200` (long string) | Caption: `"Not recognised: aaaaa..."` | None (`[]`) | No length validation or sanitisation. |

---

## 2. Confirmation of Section 2 Findings

- **Group terms fail:** "tree nuts", "nuts", "nut", "fish", "gluten", "legumes", "cereals", "beans" all fail.
- **Silent narrowing:** "shellfish" and "seafood" both resolve to `shrimp` only via `kb.canonicalize`.
- **Typo/variant failures:** "cows milk", "cow's milk", "shell fish", "penut", "peanutt", "allmond" fail.
- **Discarded custom items:** Non-model allergens (e.g. jackfruit, mushroom, quinoa) are dropped with a faint caption and never matched in label text.
- **Submission friction:** Streamlit text box requires Enter or defocus; no explicit button or clear instruction.
