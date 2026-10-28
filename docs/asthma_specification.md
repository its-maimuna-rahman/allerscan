# Clinical Specification: 4-Level Asthma Calibration for AllerScan

**Document Version:** 1.0.0  
**Status:** Approved Specification / Implementation Contract  
**Location:** `allerscan_4/docs/asthma_specification.md`  
**Target Audience:** AllerScan ML engineers, software developers, and clinical reviewers  

---

## 1. Patient-Friendly Definitions of Asthma Levels

The app categorises asthma into four discrete clinical severity tiers plus a "Not sure" missing option:

| Level | Code | Clinical Tier | Patient-Facing Definition | Clinical Analogue / Medication Profile |
|---|---|---|---|---|
| **0** | `0.0` | **No Asthma** | No asthma diagnosis or breathing symptoms. | No asthma diagnosis; no inhalers. |
| **1** | `1.0` | **Mild** | Occasional symptoms or using a reliever inhaler (e.g. albuterol / salbutamol) only when needed. | GINA Step 1–2: Intermittent or mild persistent asthma; low-dose inhaled corticosteroids (ICS) as needed or daily. |
| **2** | `2.0` | **Moderate** | Regular daily controller medication or frequent symptoms (wheezing, coughing, chest tightness). | GINA Step 3–4: Daily low-to-medium dose ICS-LABA; well or partly controlled. |
| **3** | `3.0` | **Severe** | High-dose daily controller medicine, frequent attacks, or past emergency room / hospital visits. | GINA Step 5: High-dose ICS-LABA + add-on (biologic or oral steroids) or persistent frequent exacerbations. |
| **Not sure**| `NaN`| **Unspecified** | Patient is uncertain or unassessed. | Treated natively as missing value by tree ensemble. |

*Disclaimer:* This self-reported grading is intended for risk stratification co-factor adjustment and does **not** replace a formal clinical or spirometric diagnosis by an allergist or pulmonologist.

---

## 2. Prevalence & Comorbidity Targets

All probabilities in the data generator must be calibrated according to published epidemiologic evidence:

### 2.1 Baseline Prevalence by Age and Population Setting

| Population / Cohort | Baseline Asthma Rate | Source / Citation |
|---|---|---|
| **Children (< 13 years)** | 28% – 45% (Child Survey ~30%) | Boston study summary: 29% in national survey; Pediatrics cohort (PMC3070157): 45.6% |
| **Adolescents (13 – 17 years)** | 30% – 48% (Baseline ~33%) | Boston study allergy referral centres: 48% (mdedge.com) |
| **Adults (>= 18 years)** | 20% – 26% (Adult Survey ~22%) | Gupta et al., *JAMA Netw Open* 2019; CDC NHIS adult food allergy estimates `[ASSUMED / Gupta 2019]` |
| **EU Anaphylaxis Registry** | Cohort multiplier: 1.30x (~38% – 50%) | Registry cohorts exhibit higher atopic comorbidity (Allergy-Vigilance Network / CEA 2025) `[ASSUMED 1.30x]` |
| **Pollen-Food Clinic** | Cohort multiplier: 1.20x (~26% – 40%) | Pollen allergy / allergic rhinitis clinic (CMAJ / allergy reviews) `[ASSUMED 1.20x]` |
| **South Asia Clinic** | Cohort multiplier: 1.10x (~24% – 36%) | Urban referral centre reports `[ASSUMED 1.10x]` |

### 2.2 Allergen-Specific Asthma Odds Ratios

Specific food allergies correlate strongly with asthma diagnosis:

| Primary Food Allergen | Multiplier / Odds Ratio on Asthma Prevalence | Published Source |
|---|---|---|
| **Egg Allergy** | Odds Ratio ~ 2.0 (Prevalence multiplier: +12% abs or 1.35x rel) | Pediatrics cohort, PMC3070157 |
| **Peanut Allergy** | Odds Ratio ~ 1.8 – 1.9 (Prevalence multiplier: +10% abs or 1.30x rel) | Boston study summary (mdedge.com) & PMC3070157 |
| **Tree Nut Allergy** | Odds Ratio ~ 1.9 – 2.0 (Prevalence multiplier: +11% abs or 1.32x rel) | Boston study summary & PMC3070157 |
| **Other Allergens (Milk, Fish, etc.)** | Baseline (no allergen boost) | Reference group |

### 2.3 Sex and Atopic March Interactions

| Factor | Relationship | Epidemiological Source |
|---|---|---|
| **Sex (Childhood < 18)** | Boys have higher asthma odds (multiplier 1.12x vs girls) | GINA Guidelines; Almqvist et al. 2008 / Postma 2007 (childhood male preponderance) |
| **Sex (Adulthood >= 18)** | Women have higher asthma odds (multiplier 1.15x vs men) | Postma 2007; GINA adult female preponderance |
| **Eczema Comorbidity** | Eczema increases asthma probability by +8% to +10% absolute | Atopic march paradigm (PMC3070157, Gupta 2018) |
| **Family History** | Presence of asthma increases family history odds by ~1.3x | Standard genetic atopic clustering `[ASSUMED 1.3x]` |

---

## 3. Severity Distribution Among Asthmatics

The breakdown of mild, moderate, and severe asthma depends on clinical care setting and cohort type:

| Cohort Type | Mild (Level 1) | Moderate (Level 2) | Severe (Level 3) | Target Reference / Published Basis |
|---|---|---|---|---|
| **General Population / Child Survey** (`us_child_survey`) | 68% (65%–75%) | 24% (20%–28%) | 8% (5%–10%) | Salvador GINA cohort (76/13/11); Spanish national data (10% severe); target general pop range |
| **General Adult Survey** (`us_adult_survey`) | 55% (50%–60%) | 35% (30%–40%) | 10% (8%–12%) | Adult GINA clinic data (43/47/10.5, PMC7395116); general population skew |
| **EU Anaphylaxis Registry** (`eu_anaphylaxis_registry`) | 45% (40%–50%) | 35% (30%–40%) | 20% (15%–25%) | Chinese referral cohort (16/23/62) & specialist registries skewing severe; target range |
| **Pollen-Food Clinic** (`pollen_food_clinic`) | 70% (65%–75%) | 23% (18%–26%) | 7% (5%–10%) | Outpatient pollen clinic, primarily local oral symptoms (CMAJ) |
| **South Asia Clinic** (`south_asia_clinic`) | 56% (50%–62%) | 31% (26%–35%) | 13% (10%–16%) | Intermediate referral mix `[ASSUMED intermediate]` |

---

## 4. Effect Sizes on Clinical Food Reaction Severity

The effect of asthma on food allergy reaction severity is modest, clinically meaningful, and monotonic:

### 4.1 Target Relative Risk & Odds Ratio Targets for Severe Food Reactions (Level 3)

| Asthma Level | Target Relative Risk (vs Level 0) | Calibrated Latent Shift | Published Source / Evidence Base |
|---|---|---|---|
| **Level 0 (None)** | 1.00 (Reference) | `0.00` | Baseline |
| **Level 1 (Mild)** | **1.10 – 1.40** | `+0.18` `[ASSUMED calibrated]` | Pooled OR 1.28 (95% CI 1.03–1.59), Eur Ann Allergy Clin Immunol 2024 |
| **Level 2 (Moderate)** | **1.35 – 2.00** | `+0.42` `[ASSUMED calibrated]` | Tejedor-Alonso meta-analysis OR 1.89 (95% CI 1.26–2.83), cited in PMC12322906 |
| **Level 3 (Severe)** | **2.00 – 3.50** | `+0.75` `[ASSUMED calibrated]` | French Allergy-Vigilance Network OR 3.41 (95% CI 1.56–7.44) for near-fatal anaphylaxis (CEA 2025) |

*Nut Interaction Shift:* An additional modest increment is applied for peanut/tree-nut/seed allergens: `[0.0, 0.08, 0.16, 0.28]` for levels 0, 1, 2, 3 `[ASSUMED]`.

### 4.2 Cross-Reactivity Probability Effect

Asthma slightly elevates systemic reactivity threshold across non-profile homologous allergens:
- Level 0: `+0.00`
- Level 1: `+0.06` (+6% relative boost) `[ASSUMED]`
- Level 2: `+0.12` (+12% relative boost) `[ASSUMED]`
- Level 3: `+0.20` (+20% relative boost) `[ASSUMED]`

---

## 5. Global Severity Conservation

The dataset target shares among reacting events must remain:
- **Level 1 (Mild):** ~35% (33% – 37%)
- **Level 2 (Moderate):** ~35% (33% – 37%)
- **Level 3 (Severe):** ~30% (28% – 32%)

The quantile cut-point derivation in `build_tables` guarantees that overall dataset tiers match `[0.35, 0.35, 0.30]`.

---

## 6. Implementation Verification Rules (The Contract)

1. **Deterministic Execution:** Fixed seed `SEED = 20261004` maintained.
2. **Explicit Severity Weights:** Defined at the top of `generate_dataset.py` in `ASTHMA_SEVERITY_SPLIT`.
3. **Legacy "Yes" Conversion:** String variants `"yes"`, `"true"`, `"y"` become `NaN` in `pipeline.py`.
4. **Atopic Burden Calculation:** Formed as `(has_asthma > 0).astype(float) + has_eczema + family_history_allergy`.
5. **No Schema Changes:** Model input feature list remains identical (`FEATURES`).
6. **Inference Guarding:** App and model accept `0`, `1`, `2`, `3`, or `NaN`. Any out-of-range value is mapped to `NaN`.
