#!/usr/bin/env python3
"""
Automated Generator Validation Suite for 4-Level Asthma Implementation.

Reads the cleaned dataset (and raw export for legacy verification) and evaluates
the 9 validation criteria specified in docs/01_asthma_levels.md (Step 5).
Prints PASS/FAIL per check and outputs a detailed summary report.

Usage:
    python tools/validate_asthma_data.py
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import allergen_kb as kb

CLEAN_DATA = ROOT / "data" / "allergy_clean.csv"
RAW_DATA = ROOT / "data" / "allergy_raw_50k.csv"
REPORT_JSON = ROOT / "reports" / "asthma_validation_report.json"
REPORT_TXT = ROOT / "reports" / "asthma_validation_report.txt"


def odds(p):
    return p / (1.0 - p) if p < 1.0 else float("inf")


def validate_dataset():
    if not CLEAN_DATA.exists():
        print(f"Error: {CLEAN_DATA} does not exist. Run cleaning first.")
        sys.exit(1)

    clean = pd.read_csv(CLEAN_DATA)
    raw = pd.read_csv(RAW_DATA) if RAW_DATA.exists() else None

    checks = []
    details = {}

    # Check 1: Overall asthma share and share per cohort and per age band
    k_asth = clean[clean["has_asthma"].notna()]
    overall_asthma = float((k_asth["has_asthma"] > 0).mean())
    c_shares = {c: float((sub[sub["has_asthma"].notna()]["has_asthma"] > 0).mean())
                for c, sub in clean.groupby("source")}
    k_child = clean[clean["has_asthma"].notna() & (clean["age"] < 18)]
    k_adult = clean[clean["has_asthma"].notna() & (clean["age"] >= 18)]
    child_share = float((k_child["has_asthma"] > 0).mean())
    adult_share = float((k_adult["has_asthma"] > 0).mean())

    c1_ok = (
        0.25 <= overall_asthma <= 0.45 and
        0.30 <= c_shares.get("us_child_survey", 0) <= 0.50 and
        0.20 <= c_shares.get("us_adult_survey", 0) <= 0.35 and
        0.30 <= c_shares.get("eu_anaphylaxis_registry", 0) <= 0.55 and
        0.28 <= child_share <= 0.50 and
        0.20 <= adult_share <= 0.38
    )
    checks.append(("1. Prevalence targets (overall, cohort, age bands) inside specification ranges", c1_ok))
    details["check_1_prevalence"] = {
        "overall": round(overall_asthma, 3),
        "cohorts": {k: round(v, 3) for k, v in c_shares.items()},
        "child_under_18": round(child_share, 3),
        "adult_18_plus": round(adult_share, 3),
        "status": "PASS" if c1_ok else "FAIL"
    }

    # Check 2: Severity mix among asthmatics per cohort
    asthmatics = clean[clean["has_asthma"].isin([1.0, 2.0, 3.0])]
    cohort_mix = {}
    c2_ok = True
    for c, sub in asthmatics.groupby("source"):
        vc = sub["has_asthma"].value_counts(normalize=True).sort_index().to_dict()
        m1, m2, m3 = float(vc.get(1.0, 0)), float(vc.get(2.0, 0)), float(vc.get(3.0, 0))
        cohort_mix[c] = {"mild": round(m1, 3), "moderate": round(m2, 3), "severe": round(m3, 3)}
        if c == "us_child_survey" and not (0.55 <= m1 <= 0.78 and 0.04 <= m3 <= 0.12):
            c2_ok = False
        elif c == "eu_anaphylaxis_registry" and not (0.38 <= m1 <= 0.55 and 0.14 <= m3 <= 0.28):
            c2_ok = False
        elif c == "us_adult_survey" and not (0.48 <= m1 <= 0.65 and 0.06 <= m3 <= 0.15):
            c2_ok = False
    checks.append(("2. Severity mix among asthmatics per cohort inside target ranges", c2_ok))
    details["check_2_severity_mix"] = {"mix_by_cohort": cohort_mix, "status": "PASS" if c2_ok else "FAIL"}

    # Check 3: Eczema is more common among asthmatics (OR > 1.0)
    k_ecz = clean[clean["has_asthma"].notna() & clean["has_eczema"].notna()]
    p_ecz_asth = float((k_ecz[k_ecz["has_asthma"] > 0]["has_eczema"] == 1).mean())
    p_ecz_noasth = float((k_ecz[k_ecz["has_asthma"] == 0]["has_eczema"] == 1).mean())
    or_ecz = float(odds(p_ecz_asth) / odds(p_ecz_noasth))
    c3_ok = or_ecz > 1.2
    checks.append((f"3. Eczema comorbidity (OR = {or_ecz:.3f} > 1.2)", c3_ok))
    details["check_3_eczema"] = {
        "rate_in_asthmatics": round(p_ecz_asth, 3),
        "rate_in_non_asthmatics": round(p_ecz_noasth, 3),
        "odds_ratio": round(or_ecz, 3),
        "status": "PASS" if c3_ok else "FAIL"
    }

    # Check 4: Family history is more common among asthmatics (OR > 1.0)
    k_fam = clean[clean["has_asthma"].notna() & clean["family_history_allergy"].notna()]
    p_fam_asth = float((k_fam[k_fam["has_asthma"] > 0]["family_history_allergy"] == 1).mean())
    p_fam_noasth = float((k_fam[k_fam["has_asthma"] == 0]["family_history_allergy"] == 1).mean())
    or_fam = float(odds(p_fam_asth) / odds(p_fam_noasth))
    c4_ok = or_fam > 1.2
    checks.append((f"4. Family history comorbidity (OR = {or_fam:.3f} > 1.2)", c4_ok))
    details["check_4_family_history"] = {
        "rate_in_asthmatics": round(p_fam_asth, 3),
        "rate_in_non_asthmatics": round(p_fam_noasth, 3),
        "odds_ratio": round(or_fam, 3),
        "status": "PASS" if c4_ok else "FAIL"
    }

    # Check 5: Egg, peanut and tree nut main allergens have higher asthma share than average
    clean["group"] = clean["detected_allergen"].map(kb.group_of)
    k_nut = clean[clean["has_asthma"].notna() & clean["group"].isin(["egg", "peanut", "tree_nut"])]
    k_other = clean[clean["has_asthma"].notna() & ~clean["group"].isin(["egg", "peanut", "tree_nut"])]
    sh_nut = float((k_nut["has_asthma"] > 0).mean())
    sh_other = float((k_other["has_asthma"] > 0).mean())
    c5_ok = sh_nut > sh_other
    checks.append((f"5. Egg/peanut/tree-nut asthma prevalence ({sh_nut:.1%}) > other allergens ({sh_other:.1%})", c5_ok))
    details["check_5_target_allergens"] = {
        "asthma_share_egg_peanut_treenut": round(sh_nut, 3),
        "asthma_share_other": round(sh_other, 3),
        "status": "PASS" if c5_ok else "FAIL"
    }

    # Check 6: Sex direction by age (boys > girls in childhood; women > men in adulthood)
    kc_m = clean[clean["has_asthma"].notna() & (clean["age"] < 18) & (clean["sex"] == 1.0)]
    kc_f = clean[clean["has_asthma"].notna() & (clean["age"] < 18) & (clean["sex"] == 0.0)]
    ka_m = clean[clean["has_asthma"].notna() & (clean["age"] >= 18) & (clean["sex"] == 1.0)]
    ka_f = clean[clean["has_asthma"].notna() & (clean["age"] >= 18) & (clean["sex"] == 0.0)]
    sh_cm = float((kc_m["has_asthma"] > 0).mean())
    sh_cf = float((kc_f["has_asthma"] > 0).mean())
    sh_am = float((ka_m["has_asthma"] > 0).mean())
    sh_af = float((ka_f["has_asthma"] > 0).mean())
    c6_ok = (sh_cm > sh_cf) and (sh_af > sh_am)
    checks.append((f"6. Sex direction by age (child boys {sh_cm:.1%} > girls {sh_cf:.1%}; adult women {sh_af:.1%} > men {sh_am:.1%})", c6_ok))
    details["check_6_sex_by_age"] = {
        "child_boys": round(sh_cm, 3), "child_girls": round(sh_cf, 3),
        "adult_women": round(sh_af, 3), "adult_men": round(sh_am, 3),
        "status": "PASS" if c6_ok else "FAIL"
    }

    # Check 7: Severity distribution strictly monotonic & relative risks inside calibrated ranges
    react = clean[clean["reacted"] == 1]
    p0 = float((react[react["has_asthma"] == 0]["severity"] == 3).mean())
    p1 = float((react[react["has_asthma"] == 1]["severity"] == 3).mean())
    p2 = float((react[react["has_asthma"] == 2]["severity"] == 3).mean())
    p3 = float((react[react["has_asthma"] == 3]["severity"] == 3).mean())
    rr1, rr2, rr3 = p1 / p0, p2 / p0, p3 / p0
    or1, or2, or3 = odds(p1) / odds(p0), odds(p2) / odds(p0), odds(p3) / odds(p0)
    is_monotonic = p0 < p1 < p2 < p3
    rr_in_range = (1.05 <= rr1 <= 1.45) and (1.30 <= rr2 <= 2.00) and (1.70 <= rr3 <= 3.50)
    c7_ok = is_monotonic and rr_in_range
    checks.append((f"7. Severe reaction risk monotonic and inside calibrated ranges (RR: {rr1:.2f}, {rr2:.2f}, {rr3:.2f}; OR: {or1:.2f}, {or2:.2f}, {or3:.2f})", c7_ok))
    details["check_7_severity_effects"] = {
        "P_severe_by_asthma": {"level_0": round(p0, 4), "level_1": round(p1, 4), "level_2": round(p2, 4), "level_3": round(p3, 4)},
        "relative_risks": {"RR_1": round(rr1, 3), "RR_2": round(rr2, 3), "RR_3": round(rr3, 3)},
        "odds_ratios": {"OR_1": round(or1, 3), "OR_2": round(or2, 3), "OR_3": round(or3, 3)},
        "status": "PASS" if c7_ok else "FAIL"
    }

    # Check 8: Missing-value rate and legacy 'yes' handling
    miss_rate = float(clean["has_asthma"].isna().mean())
    # Check cleaning report for legacy yes answers
    clean_rep_path = ROOT / "data" / "cleaning_report.json"
    legacy_count = 0
    if clean_rep_path.exists():
        crep = json.loads(clean_rep_path.read_text())
        legacy_count = crep.get("legacy_asthma_answers_set_missing", 0)
    c8_ok = (0.10 <= miss_rate <= 0.22) and (legacy_count > 0)
    checks.append((f"8. Missing rate (~{miss_rate:.1%}) & legacy 'yes' properly converted to NaN ({legacy_count} rows)", c8_ok))
    details["check_8_missing_and_legacy"] = {
        "missing_share": round(miss_rate, 3),
        "legacy_yes_converted": legacy_count,
        "status": "PASS" if c8_ok else "FAIL"
    }

    # Check 9: Overall severity shares within 2 percentage points of 35 / 35 / 30
    sm = react["severity"].value_counts(normalize=True).sort_index().to_dict()
    s1, s2, s3 = float(sm.get(1.0, 0)), float(sm.get(2.0, 0)), float(sm.get(3.0, 0))
    c9_ok = (0.33 <= s1 <= 0.37) and (0.33 <= s2 <= 0.37) and (0.28 <= s3 <= 0.32)
    checks.append((f"9. Overall severity shares within 2% of 35/35/30 ({s1:.1%} / {s2:.1%} / {s3:.1%})", c9_ok))
    details["check_9_overall_severity"] = {
        "level_1": round(s1, 3), "level_2": round(s2, 3), "level_3": round(s3, 3),
        "status": "PASS" if c9_ok else "FAIL"
    }

    # Print summary
    print("\n================================================================================")
    print("ALLERSCAN ASTHMA DATA VALIDATION SUITE (STEP 5)")
    print("================================================================================")
    all_pass = True
    lines = []
    for label, ok in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        msg = f"[{status}] {label}"
        print(msg)
        lines.append(msg)
    print("================================================================================")
    print(f"OVERALL RESULT: {'ALL 9 CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print("================================================================================\n")

    # Save reports
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(details, indent=2))
    txt_out = "ALLERSCAN ASTHMA DATA VALIDATION REPORT\n\n" + "\n".join(lines) + f"\n\nOverall: {'ALL PASS' if all_pass else 'FAIL'}\n"
    REPORT_TXT.write_text(txt_out)
    return all_pass


if __name__ == "__main__":
    passed = validate_dataset()
    sys.exit(0 if passed else 1)
