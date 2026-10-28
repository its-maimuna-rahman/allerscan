"""
Synthetic food-allergy dataset, 50,000 raw rows.

One row = one question the app asks:
  "This person (profile + the allergens they already know, with their own 1-3 ratings: 1=Mild, 2=Moderate, 3=Severe)
   meets allergen X. Do they react, and how severely?"

Row types
  index      no allergens known yet; first recorded reaction            (always reacts)
  in_profile X is in the person's profile, they rated it themselves      (always reacts)
  cross      X is NOT in the profile; reacts or not (cross-reactivity)   (reacts with probability)

Why this fixes the old behaviour: severity now depends on the person's own ratings and on how
closely related the foods are. A person who rates almond 1 and meets cashew gets a low severity
estimate (and a modest chance of reacting at all), not the population-level "cashew is severe".

Five independent source cohorts are mixed so the model cannot memorise one source's habits.
Each has its own allergen mix, age range, severity level, sex ratio and missingness:
  us_child_survey          Gupta et al., Pediatrics 2018 (US children)
  us_adult_survey          Gupta et al., JAMA Netw Open 2019 (US adults)
  eu_anaphylaxis_registry  registry-style, severe-skewed (anaphylaxis registries, UK/EU reviews)
  pollen_food_clinic       pollen-food syndrome clinic, mostly mild (CMAJ / allergy reviews)
  south_asia_clinic        Bangladesh / India / Asia reports (shellfish, egg, milk, beef, brinjal,
                           wheat, fish; fewer peanut/tree nut)
Cross-reactivity rates: cross_reactivity.py. Severity unpredictability: published work shows the
previous reaction's severity is a poor predictor (~57% repeat the same grade), so ratings here
agree with the next reaction only about that often.

Run:  python generate_dataset.py
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import allergen_kb as kb
import cross_reactivity as xr

SEED = 20261004
N_RAW = 50_000
N_DUP = 200
N_UNIQUE = N_RAW - N_DUP
OUT_DIR = Path(__file__).resolve().parent.parent / "data"

SIGMA_U = 0.05          # person-level severity tendency (all allergens)
SIGMA_V = 0.08          # allergen-level deviation, partly shared between related foods
SIGMA_EVENT = 0.035     # event-level variation (co-factors)
HIDDEN_EFFECT = 0.22
LABEL_FLIP = 0.002      # grading disagreement +/-1
RATING_ERR = 0.005      # extra self-report error on top of event noise
RATING_MISSING = 0.03
SEV_TARGET = [0.35, 0.35, 0.30]  # 3 clinical tiers: 1=Mild, 2=Moderate, 3=Severe
AGE_EFFECT = np.array([-0.25, -0.05, 0.30, 0.30, 0.10, 0.20])
N_KNOWN_P = [0.50, 0.21, 0.12, 0.08, 0.05, 0.04]
P_INDEX, P_A1, P_A2, P_B1, P_B2, P_B_RELATED = 0.28, 0.80, 0.15, 0.95, 0.15, 0.62

COHORTS = {
    "us_child_survey": dict(
        w=0.24, age_w=[0.42, 0.33, 0.25, 0, 0, 0], male_child=0.56, male_adult=0.5, sev_shift=0.0,
        grp_mult={}, key_mult={}, asthma_mult=1.0, eczema_mult=1.1, miss=1.0),
    "us_adult_survey": dict(
        w=0.28, age_w=[0, 0, 0, 0.28, 0.50, 0.22], male_child=0.5, male_adult=0.42, sev_shift=0.10,
        grp_mult={"crustacean": 1.5, "mollusc": 1.4, "fish": 1.3, "tree_nut": 1.2, "milk": 0.6,
                  "egg": 0.3, "cereal": 0.5, "legume": 0.5, "fruit": 1.3, "vegetable": 1.2},
        key_mult={}, asthma_mult=0.9, eczema_mult=0.8, miss=1.3),
    "eu_anaphylaxis_registry": dict(
        w=0.14, age_w=[0.18, 0.14, 0.10, 0.18, 0.30, 0.10], male_child=0.58, male_adult=0.48,
        sev_shift=0.90,
        grp_mult={"peanut": 1.2, "tree_nut": 1.3, "milk": 0.9, "egg": 0.7, "cereal": 1.5,
                  "vegetable": 1.8, "seed": 1.5, "meat": 1.4, "fruit": 0.5, "crustacean": 0.9},
        key_mult={"celery": 2.0, "wheat": 1.5}, asthma_mult=1.3, eczema_mult=1.0, miss=1.6),
    "pollen_food_clinic": dict(
        w=0.11, age_w=[0, 0.04, 0.14, 0.30, 0.45, 0.07], male_child=0.5, male_adult=0.42,
        sev_shift=-0.60,
        grp_mult={"fruit": 5.0, "vegetable": 4.0, "tree_nut": 1.0, "peanut": 0.4, "milk": 0.1,
                  "egg": 0.05, "crustacean": 0.1, "fish": 0.1, "cereal": 0.2, "meat": 0.1,
                  "legume": 1.5},
        key_mult={"hazelnut": 3.0, "apple": 2.0}, asthma_mult=1.2, eczema_mult=0.7, miss=0.8),
    "south_asia_clinic": dict(
        w=0.23, age_w=[0.22, 0.20, 0.12, 0.18, 0.22, 0.06], male_child=0.55, male_adult=0.47,
        sev_shift=-0.10,
        grp_mult={"crustacean": 2.2, "fish": 2.0, "egg": 1.5, "milk": 1.2, "meat": 2.0,
                  "peanut": 0.35, "tree_nut": 0.3, "seed": 0.8, "vegetable": 2.5, "legume": 0.9,
                  "cereal": 1.6, "mollusc": 0.7},
        key_mult={"eggplant": 14.0, "beef": 1.8, "shrimp": 1.4, "wheat": 1.3},
        asthma_mult=1.1, eczema_mult=0.9, miss=1.8),
}

# ---------------------------------------------------------------- asthma severity distribution
# Severity level given asthma: [Mild (1), Moderate (2), Severe (3)]
# Calibrated to published clinical cohort data (see docs/asthma_specification.md):
# - General population / child surveys: 65-75% mild, 20-28% mod, 5-10% severe
# - Adults / GINA clinic: ~55% mild, 35% mod, 10% severe
# - Referral & registry cohorts: shifted to moderate & severe (40-50% mild, 30-35% mod, 15-22% severe)
ASTHMA_SEVERITY_SPLIT = {
    ("us_child_survey", "child"): [0.68, 0.24, 0.08],
    ("us_child_survey", "adult"): [0.58, 0.32, 0.10],
    ("us_adult_survey", "child"): [0.68, 0.24, 0.08],
    ("us_adult_survey", "adult"): [0.55, 0.35, 0.10],
    ("eu_anaphylaxis_registry", "child"): [0.48, 0.34, 0.18],
    ("eu_anaphylaxis_registry", "adult"): [0.43, 0.35, 0.22],
    ("pollen_food_clinic", "child"): [0.74, 0.20, 0.06],
    ("pollen_food_clinic", "adult"): [0.68, 0.25, 0.07],
    ("south_asia_clinic", "child"): [0.60, 0.29, 0.11],
    ("south_asia_clinic", "adult"): [0.53, 0.33, 0.14],
}


def _comorbidity(age, male, g0, c, cfg, rng):
    if age <= 5:
        p_ecz = 0.52 if g0 in ("milk", "egg") else 0.42
    elif age <= 12:
        p_ecz = 0.32
    elif age <= 17:
        p_ecz = 0.24
    else:
        p_ecz = 0.16 if g0 in ("fruit", "vegetable") else 0.12
    eczema = int(rng.random() < min(p_ecz * cfg["eczema_mult"], 0.9))

    # Base asthma probability by age band
    p_asth = 0.30 if age < 13 else (0.33 if age < 18 else 0.22)
    # Sex direction: boys higher in childhood (<18), women higher in adulthood (>=18)
    sex_mult = (1.08 if male else 0.92) if age < 18 else (0.92 if male else 1.08)
    # Allergen association: egg, peanut, tree nut double asthma odds (OR ~2.0, PMC3070157)
    allergen_add = 0.07 if g0 in ("egg", "peanut", "tree_nut") else 0.0

    p_total = (p_asth * sex_mult + 0.08 * eczema + allergen_add) * cfg["asthma_mult"]
    has_asth = rng.random() < min(p_total, 0.85)

    if has_asth:
        age_cat = "child" if age < 18 else "adult"
        split_p = ASTHMA_SEVERITY_SPLIT.get((c, age_cat), [0.60, 0.28, 0.12])
        asthma = int(rng.choice([1, 2, 3], p=split_p))
    else:
        asthma = 0
    fam = int(rng.random() < 0.34 + 0.12 * eczema + 0.10 * float(asthma > 0))
    return asthma, eczema, fam


def simulate(rng):
    keys, K = xr.KEYS, xr.K
    P, W = xr.P_matrix(), xr.W_matrix()
    prev0 = np.array([kb.RECORD_WEIGHT.get(k, 0.02) for k in keys])
    prof = np.array([kb.age_profile(k) for k in keys])
    groups = np.array([kb.group_of(k) for k in keys])
    shift = np.array([kb.severity_shift(k) for k in keys])
    hid_p = np.array([kb.hidden_prob(k) for k in keys])
    nut_like = np.isin(groups, ["peanut", "tree_nut", "seed"]).astype(float)
    pt = np.isin(groups, ["peanut", "tree_nut"]).astype(float)
    early_atopic = np.isin(groups, ["milk", "egg"]).astype(float)

    names = list(COHORTS)
    cw = np.array([COHORTS[c]["w"] for c in names])
    cw = cw / cw.sum()
    cprev = {c: prev0 * np.array([cfg["grp_mult"].get(g, 1.0) for g in groups])
             * np.array([cfg["key_mult"].get(k, 1.0) for k in keys]) for c, cfg in COHORTS.items()}

    persons, rows = [], []
    pid = 0
    while len(rows) < N_UNIQUE:
        pid += 1
        c = names[rng.choice(len(names), p=cw)]
        cfg = COHORTS[c]
        pc = cprev[c]
        j0 = int(rng.choice(K, p=pc / pc.sum()))
        agew = np.array(cfg["age_w"], dtype=float) * prof[j0]
        agew /= agew.sum()
        b = int(rng.choice(6, p=agew))
        age = int(rng.integers(kb.AGE_BINS[b][0], kb.AGE_BINS[b][1] + 1))
        male = int(rng.random() < (cfg["male_child"] if age < 18 else cfg["male_adult"]))
        asthma, eczema, fam = _comorbidity(age, male, groups[j0], c, cfg, rng)

        n = int(rng.choice(np.arange(1, 7), p=N_KNOWN_P))
        chosen = [j0]
        if n > 1:
            w = pc * prof[:, b] * (1.0 + 20.0 * P[j0])
            w[j0] = 0.0
            chosen += [int(x) for x in rng.choice(K, size=min(n - 1, K - 1), replace=False,
                                                  p=w / w.sum())]
        adult_female = float(age >= 18 and male == 0)
        teen_young = float(13 <= age <= 29)
        # Latent severity shift calibrated by simulation (docs/asthma_specification.md)
        asthma_effect = [0.0, 0.07, 0.16, 0.32][asthma]
        common = (AGE_EFFECT[b] + asthma_effect + 0.12 * eczema + 0.10 * fam + cfg["sev_shift"]
                  + 0.10 * min(len(chosen) - 1, 3) + 0.06 * adult_female)

        def base(k):
            nut_effect = [0.0, 0.035, 0.07, 0.12][asthma]
            return (shift[k] + common + nut_effect * nut_like[k] + 0.15 * teen_young * pt[k]
                    + 0.10 * eczema * early_atopic[k] * float(age <= 5))

        u = rng.normal(0, SIGMA_U)
        v, z = [], []
        for i, k in enumerate(chosen):
            if i == 0:
                vk = rng.normal(0, SIGMA_V)
            else:
                ws = [W[chosen[m], k] for m in range(i)]
                m = int(np.argmax(ws))
                vk = ws[m] * v[m] + np.sqrt(1 - ws[m] ** 2) * rng.normal(0, SIGMA_V)
            v.append(vk)
            z.append(base(k) + u + vk)
        zr = [zk + rng.normal(0, SIGMA_EVENT) for zk in z]            # what the person remembers
        pi = len(persons)
        persons.append(dict(pid=pid, cohort=c, age=age, male=male, asthma=asthma, eczema=eczema,
                            fam=fam, chosen=chosen, z_rating=zr))

        def add_row(rtype, t, reacted, z_t, prof_ids):
            hidden = int(rng.random() < hid_p[t])
            z_lab = None if z_t is None else z_t + HIDDEN_EFFECT * hidden + rng.normal(0, SIGMA_EVENT)
            rows.append(dict(p=pi, rtype=rtype, t=t, hidden=hidden, reacted=int(reacted),
                             z=z_lab, prof=prof_ids))

        if rng.random() < P_INDEX:
            add_row("index", chosen[0], 1, z[0], [])
        n_a = int(rng.random() < P_A1) + int(rng.random() < P_A2 and len(chosen) > 1)
        for i in rng.choice(len(chosen), size=min(n_a, len(chosen)), replace=False):
            add_row("in_profile", chosen[int(i)], 1, z[int(i)], chosen)
        n_b = int(rng.random() < P_B1) + int(rng.random() < P_B2)
        for _ in range(n_b):
            cand = np.ones(K, bool)
            cand[chosen] = False
            linkv = P[chosen].max(axis=0)
            wB = linkv * np.sqrt(pc) * prof[:, b] if rng.random() < P_B_RELATED else pc * prof[:, b]
            wB = np.where(cand, wB, 0.0)
            if wB.sum() <= 0:
                wB = np.where(cand, pc, 0.0)
            t = int(rng.choice(K, p=wB / wB.sum()))
            p_react = 1.0 - np.prod(1.0 - P[chosen, t])
            # Modest, monotonic cross-reactivity effect [ASSUMED]
            asthma_react_mult = [0.0, 0.05, 0.10, 0.18][asthma]
            p_react = min(0.97, p_react * (1 + asthma_react_mult + 0.10 * eczema))
            reacted = rng.random() < p_react
            z_t = None
            if reacted:
                ws = W[chosen, t]
                m = int(np.argmax(ws))
                z_t = base(t) + u + ws[m] * v[m] + np.sqrt(1 - ws[m] ** 2) * rng.normal(0, SIGMA_V)
            add_row("cross", t, reacted, z_t, chosen)
    return persons, rows[:N_UNIQUE], keys


def to_sev(z, cuts, gi, rng):
    s = np.digitize(z, cuts) + 1
    tilt = (s == 1) & (rng.random(len(z)) < gi * 0.15)
    return np.where(tilt, 2, s)


def build_tables(rng):
    persons, rows, keys = simulate(rng)
    gi_p = np.array([kb.gi_prob(k) for k in keys])
    z_arr = np.array([np.nan if r["z"] is None else r["z"] for r in rows])
    cuts = np.quantile(z_arr[~np.isnan(z_arr)], np.cumsum(SEV_TARGET)[:-1])

    ratings = []                                  # the person's own rating of each known allergen
    for p in persons:
        r = to_sev(np.array(p["z_rating"]), cuts, gi_p[p["chosen"]], rng)
        err = rng.random(len(r)) < RATING_ERR
        r = np.clip(np.where(err, r + rng.choice([-1, 1], len(r)), r), 1, 3)
        ratings.append({keys[k]: int(x) for k, x in zip(p["chosen"], r)})

    t_arr = np.array([r["t"] for r in rows])
    sev = np.full(len(rows), np.nan)
    ok = ~np.isnan(z_arr)
    s = to_sev(z_arr[ok], cuts, gi_p[t_arr[ok]], rng)
    flip = rng.random(len(s)) < LABEL_FLIP
    sev[ok] = np.clip(np.where(flip, s + rng.choice([-1, 1], len(s)), s), 1, 3)

    recs = []
    for i, r in enumerate(rows):
        p = persons[r["p"]]
        recs.append(dict(
            person_id=p["pid"], source=p["cohort"], age=p["age"], sex=p["male"],
            has_asthma=p["asthma"], has_eczema=p["eczema"], family_history_allergy=p["fam"],
            profile={keys[k]: ratings[r["p"]][keys[k]] for k in r["prof"]},
            detected_allergen=keys[r["t"]], is_hidden_allergen_flag=r["hidden"],
            reacted=r["reacted"], severity=sev[i], row_type=r["rtype"]))
    df = pd.DataFrame(recs)
    flags = pd.DataFrame([kb.protein_flags(k) for k in df.detected_allergen])
    return pd.concat([df, flags], axis=1)


def _typo(s, rng):
    if len(s) < 4:
        return s
    i = int(rng.integers(1, len(s) - 1))
    return s[:i] + s[i + 1:] if rng.random() < 0.5 else s[:i] + s[i + 1] + s[i] + s[i + 2:]


def make_raw(df, rng):
    """Turn the clean table into a messy, export-like file."""
    n = len(df)
    pool = {}
    for canon, syn in kb.load_name_mapping().items():
        pool.setdefault(kb.MERGE.get(canon, canon), []).extend(syn)

    def messy_name(name):
        if rng.random() < 0.22 and pool.get(name):
            name = pool[name][int(rng.integers(len(pool[name])))]
            name = name.replace("_", " " if rng.random() < 0.6 else "_")
        if rng.random() < 0.12:
            name = name.title() if rng.random() < 0.5 else name.upper()
        if rng.random() < 0.08:
            name = f"  {name} " if rng.random() < 0.5 else f"{name}  "
        if rng.random() < 0.004:
            name = _typo(name.strip(), rng)
        return name

    raw = pd.DataFrame({"record_id": [f"R{i:06d}" for i in range(1, n + 1)],
                        "person_id": df.person_id.map(lambda p: f"P{p:06d}"),
                        "source": df.source})
    raw["age"] = df.age.astype(float).to_numpy().copy()
    raw["sex"] = df.sex
    for c in ["has_asthma", "has_eczema", "family_history_allergy"]:
        raw[c] = df[c]

    seps = [";", "|", ","]
    profs = []
    for d in df.profile:
        sep = seps[int(rng.choice(3, p=[0.7, 0.15, 0.15]))]
        parts = []
        for k, r in d.items():
            rr = "" if rng.random() < RATING_MISSING else (f"{r}.0" if rng.random() < 0.05 else str(r))
            parts.append(f"{messy_name(k)}:{rr}")
        profs.append(sep.join(parts))
    raw["profile"] = profs
    raw["detected_allergen"] = [messy_name(k) for k in df.detected_allergen]
    raw["is_hidden_allergen_flag"] = df.is_hidden_allergen_flag
    for c in kb.PROTEIN_COLS:
        raw[c] = df[c].to_numpy().copy()
    raw["reacted"] = df.reacted
    raw["severity"] = df.severity.astype(float).to_numpy().copy()

    miss = df.source.map(lambda s: COHORTS[s]["miss"]).to_numpy()
    male, female = ["M", "Male", "male", "1", "m"], ["F", "Female", "female", "0", "f"]
    sex = np.array([male[int(rng.integers(5))] if s == 1 else female[int(rng.integers(5))]
                    for s in raw.sex], dtype=object)
    sex[rng.random(n) < 0.012 * miss] = None
    raw["sex"] = sex
    asthma_strings = {
        0: ["0", "No", "no", "None", "N"],
        1: ["1", "Mild", "mild", "1.0", "mild"],
        2: ["2", "Moderate", "moderate", "mod", "2.0"],
        3: ["3", "Severe", "severe", "sev", "3.0"],
    }
    raw_asthma_list = []
    for x in raw["has_asthma"]:
        xv = int(x)
        # Small share (< 1%) of legacy binary 'yes/true/Y' answers among asthmatics to exercise cleaning
        if xv > 0 and rng.random() < 0.007:
            raw_asthma_list.append(rng.choice(["yes", "Yes", "true", "True", "Y", "y"]))
        else:
            strs = asthma_strings[xv]
            raw_asthma_list.append(strs[int(rng.integers(len(strs)))])
    raw_asthma = np.array(raw_asthma_list, dtype=object)
    raw_asthma[rng.random(n) < 0.015 * miss] = None
    raw["has_asthma"] = raw_asthma

    yes, no = ["1", "Y", "Yes", "yes", "True"], ["0", "N", "No", "no", "False"]
    for c, m0 in [("has_eczema", 0.015), ("family_history_allergy", 0.04),
                  ("is_hidden_allergen_flag", 0.0), ("reacted", 0.0)]:
        v = np.array([yes[int(rng.integers(5))] if x == 1 else no[int(rng.integers(5))]
                      for x in raw[c]], dtype=object)
        if m0:
            v[rng.random(n) < m0 * miss] = None
        raw[c] = v
    age = raw.age.to_numpy().copy()
    infants = age == 0
    age[infants] = np.round(rng.uniform(0.3, 1.0, infants.sum()), 1)
    age[rng.random(n) < 0.02 * miss] = np.nan
    bad = rng.random(n) < 0.004
    age[bad] = rng.choice([-1, 150, 999], bad.sum())
    raw["age"] = age
    for c in kb.PROTEIN_COLS:
        f = rng.random(n) < 0.008
        raw.loc[f, c] = 1 - raw.loc[f, c]
    sev = raw.severity.to_numpy().copy()
    badlab = (df.reacted.to_numpy() == 1) & (rng.random(n) < 0.0025)
    sev[badlab] = rng.choice([0, 6, np.nan], badlab.sum())
    raw["severity"] = sev

    raw = pd.concat([raw, raw.sample(N_DUP, random_state=SEED)], ignore_index=True)
    return raw.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def write_data_card(path, df):
    mix = df[df.reacted == 1].severity.value_counts(normalize=True).sort_index().round(3).to_dict()
    asthma_dist = df.has_asthma.value_counts(normalize=True).sort_index().round(3).to_dict()
    path.write_text(f"""# AllerScan synthetic dataset

Synthetic, literature-shaped records. NOT patient data. Not for medical decisions.

Rows: {N_RAW:,} raw ({N_DUP} exact duplicates inside). Target: `severity` 1-3 (1=Mild, 2=Moderate, 3=Severe) when `reacted` = 1.
Severity mix among reacting rows: {mix}
Asthma levels (0=No, 1=Mild, 2=Moderate, 3=Severe): {asthma_dist}
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
""", encoding="utf-8")


def main():
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(exist_ok=True)
    df = build_tables(rng)
    raw = make_raw(df, rng)
    raw.to_csv(OUT_DIR / "allergy_raw_50k.csv", index=False)
    write_data_card(OUT_DIR / "DATA_CARD.md", df)

    print(f"raw rows: {len(raw):,}   persons: {df.person_id.nunique():,}")
    print("row types:", df.row_type.value_counts(normalize=True).round(3).to_dict())
    print("rows by source:", df.source.value_counts(normalize=True).round(3).to_dict())
    r = df[df.reacted == 1]
    print("severity mix (reacting rows):", r.severity.value_counts(normalize=True).sort_index().round(3).to_dict())
    print("severe (Level 3) share by source:", r.groupby("source").severity.apply(lambda s: round((s == 3).mean(), 3)).to_dict())
    print("cross rows that react:", round(df[df.row_type == "cross"].reacted.mean(), 3))
    a = df[df.row_type == "in_profile"].copy()
    a["rating"] = [d.get(k) for d, k in zip(a.profile, a.detected_allergen)]
    print("own rating vs graded severity: exact", round((a.rating == a.severity).mean(), 3),
          " within one", round(((a.rating - a.severity).abs() <= 1).mean(), 3))
    print("asthma levels (0=No, 1=Mild, 2=Mod, 3=Sev):", df.has_asthma.value_counts(normalize=True).sort_index().round(3).to_dict(),
          "eczema", round(df.has_eczema.mean(), 3), "family hx", round(df.family_history_allergy.mean(), 3), "male", round(df.sex.mean(), 3))
    print("top allergens:", df.detected_allergen.value_counts(normalize=True).head(8).round(3).to_dict())


if __name__ == "__main__":
    main()
