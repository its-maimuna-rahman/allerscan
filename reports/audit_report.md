# Audit Report: Asthma 4-Level Implementation

**Project:** AllerScan  
**Date:** 2026-10-27  
**Branch:** `asthma-four-levels`  
**Reference Document:** `docs/01_asthma_levels.md` (Sections 3 & 4)

---

## 1. Audit Checklist & Implementation Status

| Component | File / Location | Status | Details & Findings |
|---|---|---|---|
| **Generator: Prevalence Logic** | `training/generate_dataset.py` (`_comorbidity`) | **PARTIAL** | Asthma prevalence is conditioned on age band, eczema, and cohort multiplier, but currently ignores main allergen group (egg/peanut/tree nut OR ~2.0) and patient sex (boys > girls in childhood, adult women > men). |
| **Generator: Severity Distribution** | `training/generate_dataset.py` (`_comorbidity`) | **PARTIAL** | Fixed split `p=[0.55, 0.32, 0.13]` used indiscriminately across all cohorts and ages. Does not reflect clinical differences between general population / child surveys (65-75% mild) and referral / registry cohorts (shifted to moderate / severe). |
| **Generator: Effect on Severity** | `training/generate_dataset.py` (`simulate`) | **PARTIAL** | Latent shifts of `[0.0, 0.22, 0.48, 0.82]` and nut effects `[0.0, 0.10, 0.20, 0.35]` were arbitrarily chosen and uncalibrated against published odds ratios / relative risks (OR ~1.28 - 1.89 - 3.41). |
| **Generator: Cross-Reactivity Effect** | `training/generate_dataset.py` (`simulate`) | **PARTIAL** | `asthma_react_mult = [0.0, 0.08, 0.16, 0.25]` is monotonic but uncalibrated/unspecified as ASSUMED. |
| **Generator: Raw Messy Export** | `training/generate_dataset.py` (`make_raw`) | **PARTIAL** | Generates string variants for 0..3 and missing values (`0.015 * miss`), but lacks legacy `"yes"`, `"true"`, `"Y"` strings required to exercise legacy cleaning logic. |
| **Data Cleaning: Asthma Mapping** | `src/pipeline.py` (`ASTHMA_MAP`, `_to_asthma`) | **PARTIAL** | Maps strings to 0.0, 1.0, 2.0, 3.0, but silently maps legacy `"yes"`, `"y"`, `"true"` to 1.0 (Mild). Must be mapped to `NaN` (missing level) to prevent biased estimates. |
| **Data Cleaning: Reporting** | `src/pipeline.py` (`clean_raw`) | **PARTIAL** | Tracks `missing_after_cleaning["has_asthma"]`, but does not report per-level counts or counts of converted legacy "yes" rows. |
| **Feature Engineering: Atopic Burden** | `src/pipeline.py` (`make_features`) | **PARTIAL** | Sums raw `has_asthma` (0-3) directly with binary flags `has_eczema` (0/1) and `family_history_allergy` (0/1), causing severe asthma to artificially triple atopic burden. Must use binary indicator `has_asthma > 0`. |
| **Inference Layer: Query Handling & Labels** | `src/model.py` (`predict_severity`, `_label`) | **PARTIAL** | Explanation labels exist for levels 1..3 (`Mild asthma`, `Moderate asthma`, `Severe asthma`). Missing explicit input validation to clamp or convert non-0..3 values to `NaN`. |
| **User Interface: Asthma Input** | `allerscan_4/app.py` (sidebar) | **PARTIAL** | Selectbox has `["No", "Mild", "Moderate", "Severe"]`. Missing 5th option `"Not sure"` (passes `NaN`), lacks detailed patient-friendly help text clarifying each grade, and lacks a disclaimer that this is not a diagnostic tool. |
| **User Interface: Safety Messaging** | `allerscan_4/app.py` (results) | **MISSING** | Missing calm safety notice recommending discussion of an emergency action plan / epinephrine with a doctor when predicted severity is severe and user has moderate or severe asthma. |
| **Training Probes: Asthma Behaviour** | `training/train_model.py` (`run_probes`) | **PARTIAL** | Only 1 probe tests asthma (`peanut=2 + asthma -> cashew` vs `no asthma`). Missing monotonic tests (0->1->2->3), severe probability monotonicity, cross-reaction probability tests, effect size upper bound, and missing-value resilience. |
| **Validation Suite** | `tools/` or `tests/` | **MISSING** | No automated validation suite currently verifies post-cleaning asthma distributions, odds ratios, monotonic severity risks, or legacy conversion counts against literature targets. |
| **Documentation & Data Card** | `README.md`, `data/DATA_CARD.md` | **PARTIAL** | Mentions 4 levels in Data Card, but `README.md` references an "asthma flag", reports stale metrics and probe counts, and lacks calibration citations. |

---

## 2. Key Actions Planned

1. **Step 3:** Draft full specification document `docs/asthma_specification.md` with literature targets, odds ratios, and explicit ASSUMED tags.
2. **Step 4:** Calibrate generator `_comorbidity`, cohort-specific severity split tables, sex/allergen associations, calibrated severity shifts, and raw export legacy strings.
3. **Step 5:** Create automated validation suite script (`tools/validate_asthma_data.py`).
4. **Step 6:** Update cleaning logic in `src/pipeline.py` (legacy `"yes"` -> `NaN`, per-level reporting, binary `has_asthma > 0` in `atopic_burden`).
5. **Step 7:** Update `src/model.py` and `app.py` (add `"Not sure"`, help text, safety message).
6. **Step 8:** Replace weak probe with 5 comprehensive probes in `training/train_model.py`.
7. **Retraining:** Retraining will be handed off to the user per explicit instruction ("DO NOT train the model by yourself. i myself will train the ML model").
