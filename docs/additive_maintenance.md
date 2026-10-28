# Additive Warnings Maintenance & Extension Guide

**Module**: AllerScan Food Additive Hypersensitivity Warnings  
**Data File**: `data/additive_warnings.json`  
**Detector & Validator**: `src/additive_detector.py`  
**Test Suite**: `tests/test_additive_warnings.py`  
**Last Reviewed Date**: 2026-10-27  
**Review Frequency**: Mandatory at least once every 12 months  

---

## 1. Principles & Evidence Standard

Additive hypersensitivity is clinically rare (estimated at ~0.01%–0.23% in the general population; 2%–7% in atopic cohorts). AllerScan presents additive notes under strict guidelines:

1. **Small, calm, and honest**:
   - Never use red alerts, warning banners, emergency icons, or alarmist styling.
   - Never affect the primary allergy verdict, risk score, severity triage, or ML confidence cards.
   - Clearly state that reactions to additives are rare.
2. **Evidence-based only**:
   - Every additive must cite peer-reviewed clinical immunology or allergy literature.
   - Blog posts, lifestyle detox claims, unverified petitions, and "toxic chemical" lists are strictly banned.
3. **Medical disclaimer**:
   - Every detailed card must display:
     *"If you think you react to this, ask an allergist; reactions to additives are hard to prove and are rarely confirmed in tests."*

---

## 2. Evidence Tiers

| Tier | Category | Default Display | Clinical Criteria |
|---|---|---|---|
| **Tier 1** | Regulated / Well Documented | **Visible** | Legally mandated warning/declaration (e.g. sulfites > 10 ppm) or established IgE-mediated anaphylaxis with high sensitization rates (e.g. carmine/cochineal extract). |
| **Tier 2** | Case Reports of Allergic Reactions | **Visible** | Published, peer-reviewed clinical case reports of IgE-mediated anaphylaxis, urticaria, or angioedema with positive skin prick / challenge tests (e.g. annatto, carrageenan, guar gum, erythritol). |
| **Tier 3** | Commonly Blamed (Unconfirmed) | **Hidden by Default** | Substances widely suspected or blamed by consumers (e.g. MSG, aspartame, BHA/BHT, nitrites, benzoates, azo dyes), but double-blind placebo-controlled food challenges (DBPCFC) have failed to demonstrate reproducible causal reactions. Only displayed if the user toggles *"Also show additives with weak evidence"*. |

---

## 3. How to Add a New Additive

### Step A: Verify Against Official Registers
Before adding an entry to `data/additive_warnings.json`, verify every identifier against at least one official register:
1. **EU Register of Authorized Food Additives** (DG SANTE)
2. **Codex Alimentarius International Numbering System (INS)** (FAO/WHO CXG 36-1989)
3. **US FDA Food Additive Status & Color Certification Lists** (21 CFR Parts 70–82, 170–189)

### Step B: Check Exclusions & Hard Boundaries
- **Foods & Primary Allergens**: Lupin, yeast, soy lecithin, wheat gluten, and milk casein belong in the core allergen knowledge base (`allergen_kb.py`), NOT in the additive file.
- **Gelatin**: Linked to beef/pork in the allergen KB; do not add as a synthetic chemical.
- **Vitamins**: Nutrient fortifiers (e.g. Vitamin E / tocopherol, Vitamin C / ascorbic acid) are not hypersensitivity additives.

### Step C: Add the Entry to `data/additive_warnings.json`
Append an object to the `"additives"` list adhering to this exact schema:

```json
{
  "id": "unique_snake_case_id",
  "display_name": "Official Common Name (Short Identifier)",
  "class": "colour | preservative | antioxidant | sweetener | thickener | flavour enhancer | glazing agent",
  "origin": "natural | synthetic | semi-synthetic",
  "tier": 1,
  "e_numbers": ["E100"],
  "ins_numbers": ["100"],
  "us_names": ["FD&C Name if applicable"],
  "synonyms": ["common chemical names", "spelling variants"],
  "reactions": ["urticaria", "anaphylaxis", "bronchospasm"],
  "relevant_groups": ["specific atopic or cross-reactive groups"],
  "related_foods": ["typical food matrices"],
  "rarity": "Concise summary of prevalence from literature",
  "description": "Original plain-language summary written by reviewer (never copied verbatim)",
  "weak_evidence_note": "Mandatory explanation for Tier 3 entries describing DBPCFC findings",
  "sources": [
    {
      "citation": "Author et al., Journal Name Year; Volume:Pages",
      "url": "https://doi.org/... or https://pmc.ncbi.nlm.nih.gov/..."
    }
  ],
  "reviewed_date": "YYYY-MM-DD"
}
```

### Step D: Check for Alias Collisions
The validator forbids any alias (E-number, US name, synonym) from appearing under more than one additive ID. If an alias is shared between multiple additives, disambiguate it or group related salts under a single parent entry (e.g. grouping all metabisulfites under `sulfites`).

---

## 4. Annual Review Protocol

The additive database must be audited at least once every calendar year.

1. **Verify Literature Updates**:
   - Check PubMed for newly published double-blind placebo-controlled challenge trials on Tier 3 additives.
   - Review anaphylaxis registries for new emerging excipients or additives.
2. **Review Regulatory Status**:
   - Check EU / FDA / Codex updates for changes in authorized food additive numbers or labelling thresholds.
3. **Update Date Stamp**:
   - Update `"last_reviewed": "YYYY-MM-DD"` at the root of `data/additive_warnings.json`.
   - Update the individual `"reviewed_date"` field for any updated records.

---

## 5. How to Run the Validator & Test Suite

Run the built-in validator directly from the command line:

```bash
# Direct validation check
python3 -c "
import sys
sys.path.insert(0, 'src')
import additive_detector as ad
data = ad.load_additive_data(force_reload=True)
print(f'Validation SUCCESS: {len(data[\"additives\"])} additives loaded without errors.')
"
```

Run the complete automated test suite:

```bash
pytest tests/test_additive_warnings.py -v
```

This ensures:
- Unique IDs and zero alias collisions.
- Valid E-number syntax (`E` followed by 3–4 digits and optional suffix).
- Valid tier assignments (`1`, `2`, or `3`).
- Valid source citations with non-empty citation text.
- Negation handling and OCR misread healing performance.
- Zero impact on the existing ML allergen models.
