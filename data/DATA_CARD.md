# AllerScan synthetic dataset

Synthetic, literature-shaped records. NOT patient data. Not for medical decisions.

Rows: 50,000 raw (200 exact duplicates inside). Target: `severity` 1-3 (1=Mild, 2=Moderate, 3=Severe) when `reacted` = 1.
Severity mix among reacting rows: {1.0: 0.345, 2.0: 0.355, 3.0: 0.3}
Asthma levels (0=No, 1=Mild, 2=Moderate, 3=Severe): {0: 0.696, 1: 0.178, 2: 0.089, 3: 0.036}
Row types: index (no known allergens), in_profile (target is one of the person's own allergens,
with their own ratings), cross (target is not in the profile; may or may not react).
Group key: `person_id`. Source cohorts (`source`) are for validation only, not model inputs.

Asthma Calibration & Literature Sources:
- Prevalence by age & setting: Pediatrics cohort (PMC3070157, 45.6% of food-allergic children have asthma),
  Boston referral cohort (48%), national survey (~29%), Gupta et al. JAMA Netw Open 2019 (adults ~22%).
- Allergen association: Egg allergy and tree nut allergy double the odds of asthma (OR ~2.0, PMC3070157);
  peanut OR ~1.8-1.9.
- Sex-by-age pattern: Male preponderance in childhood asthma (<18), shifting to female preponderance in adulthood (Postma 2007; GINA).
- Severity distribution among asthmatics: General population / child surveys ~65-75% mild, 20-28% moderate, 5-10% severe
  (Salvador Brazil cohort; Spanish national data); referral/registry cohorts shifted to moderate/severe (40-50% mild, 30-35% mod, 15-22% severe).
- Severity reaction effect: Monotonic latent shifts calibrated against published meta-analyses:
  Eur Ann Allergy Clin Immunol (pooled OR 1.28 [95% CI 1.03-1.59] for severe reactions with asthma);
  Tejedor-Alonso meta-analysis (OR 1.89 [95% CI 1.26-2.83] for anaphylaxis severity);
  French Allergy-Vigilance Network (OR 3.41 [95% CI 1.56-7.44] for near-fatal anaphylaxis, CEA 2025).
- Cross-reactivity: Modest, monotonic increase in cross-reaction probability with asthma level [ASSUMED].
- Raw messy export: Level strings for 0-3 with cohort missingness; <1% legacy binary 'yes/true' answers for cleaning verification.
- Global severity: Calibrated cut-points preserve 35% Mild / 35% Moderate / 30% Severe overall reacting tiers.
