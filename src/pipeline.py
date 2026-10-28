"""
Cleaning + feature pipeline.

  clean_raw(df)       raw export -> validated canonical table (+ report of what changed)
  make_features(df)   clean table or single query -> model matrix with a fixed schema
  build_query_df(...) one app question -> the same table format

The same make_features() runs in training and in the app, so they cannot drift apart.

The profile column holds the person's own allergens with their own 1-5 ratings,
canonical form  "almond:1;beef:2"  (rating left empty when unknown).

Missing values (age, sex, comorbidity flags, ratings) stay NaN on purpose: LightGBM routes
missing values natively, so no imputer is fitted on data that later sits in a validation fold.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import allergen_kb as kb
import cross_reactivity as xr

BINARY_YES = {"1", "y", "yes", "true", "t", "1.0"}
BINARY_NO = {"0", "n", "no", "false", "f", "0.0"}
SEX_MALE = {"1", "m", "male", "man", "boy", "1.0"}
SEX_FEMALE = {"0", "f", "female", "woman", "girl", "0.0"}
AGE_MIN, AGE_MAX = 0.0, 110.0

PROFILE_FEATURES = ["target_in_profile", "self_rating", "n_other_known", "max_other_sev",
                    "mean_other_sev", "related_link", "related_sev", "related_wmean_sev",
                    "n_related_known", "same_group_max_sev", "n_same_group_known"]
BASE_NUMERIC = ["age", "sex", "has_asthma", "has_eczema", "family_history_allergy",
                "is_hidden_allergen_flag", "atopic_burden", "n_protein_families"]
CAT_FEATURES = ["detected_allergen", "allergen_group"]
FEATURES = BASE_NUMERIC + PROFILE_FEATURES + kb.PROTEIN_COLS + CAT_FEATURES
# the reactivity model only sees "cross" questions, where these two are constant
REACT_FEATURES = [f for f in FEATURES if f not in ("target_in_profile", "self_rating")]


# ---------------------------------------------------------------- profile helpers
def parse_profile(s) -> dict:
    """'almond:1;beef:2' (canonical) -> {'almond': 1.0, 'beef': 2.0}; unknown rating -> NaN."""
    out = {}
    if s is None or (isinstance(s, float) and np.isnan(s)) or str(s).strip() == "":
        return out
    for part in re.split(r"[;|]", str(s)):
        if not part.strip():
            continue
        name, _, rating = part.partition(":")
        key = name.strip()
        r = np.nan
        m = re.search(r"\d+(?:\.\d+)?", rating)
        if m:
            val = float(m.group(0))
            if 1 <= val <= 3:
                r = float(round(val))
        out[key] = r
    return out


def profile_to_string(d: dict) -> str:
    return ";".join(f"{k}:{'' if v is None or pd.isna(v) else int(v)}" for k, v in sorted(d.items()))


def _clean_profile(raw_s):
    """Messy raw profile text -> canonical string (names mapped, bad names/ratings dropped)."""
    if raw_s is None or (isinstance(raw_s, float) and np.isnan(raw_s)):
        return ""
    text = str(raw_s)
    parts = re.split(r"[;|]", text) if re.search(r"[;|]", text) else text.split(",")
    out = {}
    for part in parts:
        name, _, rating = part.partition(":")
        key = kb.canonicalize(name) if name.strip() else None
        if key is None or key in out:
            continue
        m = re.search(r"\d+(?:\.\d+)?", rating)
        val = float(m.group(0)) if m else np.nan
        out[key] = float(round(val)) if (m and 1 <= val <= 3) else np.nan
    return profile_to_string(out)


# ---------------------------------------------------------------- cleaning
def _to_binary(s):
    v = s.astype("string").str.strip().str.lower()
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    out[v.isin(BINARY_YES)] = 1.0
    out[v.isin(BINARY_NO)] = 0.0
    return out


def _to_sex(s):
    v = s.astype("string").str.strip().str.lower()
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    out[v.isin(SEX_MALE)] = 1.0
    out[v.isin(SEX_FEMALE)] = 0.0
    return out


ASTHMA_MAP = {
    "0": 0.0, "0.0": 0.0, "no": 0.0, "n": 0.0, "none": 0.0, "false": 0.0,
    "1": 1.0, "1.0": 1.0, "mild": 1.0, "mild asthma": 1.0,
    "2": 2.0, "2.0": 2.0, "moderate": 2.0, "mod": 2.0, "moderate asthma": 2.0,
    "3": 3.0, "3.0": 3.0, "severe": 3.0, "sev": 3.0, "severe asthma": 3.0,
}
LEGACY_ASTHMA_YES = {"yes", "y", "true", "t"}


def _to_asthma(s):
    """
    Map strings and numbers to ordinal asthma levels: 0.0, 1.0, 2.0, 3.0.
    Legacy binary 'yes'/'true'/'y' answers are converted to NaN (missing level)
    so as not to bias the ordinal scale, and are counted in the cleaning report.
    Out-of-range or unrecognized values become NaN.
    Returns (cleaned_series, n_legacy_converted).
    """
    if s is None:
        return pd.Series(np.nan, dtype="float64"), 0
    if pd.api.types.is_numeric_dtype(s):
        out = pd.to_numeric(s, errors="coerce").astype("float64")
        out[~out.isin([0.0, 1.0, 2.0, 3.0])] = np.nan
        return out, 0
    v = s.astype("string").str.strip().str.lower()
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    legacy_mask = v.isin(LEGACY_ASTHMA_YES)
    n_legacy = int(legacy_mask.sum())
    for key, val in ASTHMA_MAP.items():
        out[v == key] = val
    # Legacy answers become missing level (NaN)
    out[legacy_mask] = np.nan
    out[~out.isin([0.0, 1.0, 2.0, 3.0])] = np.nan
    return out, n_legacy


def clean_raw(raw: pd.DataFrame):
    """Return (clean_df, report). Each step records how many rows or cells it touched."""
    rep = {"rows_in": int(len(raw))}
    df = raw.copy()

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    rep["exact_duplicates_dropped"] = int(before - len(df))

    names = df["detected_allergen"].astype("string").fillna("")
    canon = names.map(lambda x: kb.canonicalize(x) if x.strip() else None)
    rep["allergen_names_changed"] = int((canon.notna() & (canon != names)).sum())
    unknown = canon.isna()
    rep["unknown_allergen_dropped"] = int(unknown.sum())
    df["detected_allergen"] = canon
    df = df[~unknown].reset_index(drop=True)

    df["reacted"] = _to_binary(df["reacted"])
    bad_r = df["reacted"].isna()
    rep["unreadable_reacted_dropped"] = int(bad_r.sum())
    df = df[~bad_r].reset_index(drop=True)
    sev = pd.to_numeric(df["severity"], errors="coerce")
    invalid = (df["reacted"] == 1) & ~sev.isin([1, 2, 3])
    rep["invalid_severity_dropped"] = int(invalid.sum())
    df = df[~invalid].reset_index(drop=True)
    sev = pd.to_numeric(df["severity"], errors="coerce")
    df["severity"] = sev.where(df["reacted"] == 1)               # no label when no reaction

    df["sex"] = _to_sex(df["sex"])
    df["has_asthma"], n_legacy_asthma = _to_asthma(df["has_asthma"])
    rep["legacy_asthma_answers_set_missing"] = n_legacy_asthma
    rep["asthma_level_counts"] = {
        "0": int((df["has_asthma"] == 0.0).sum()),
        "1": int((df["has_asthma"] == 1.0).sum()),
        "2": int((df["has_asthma"] == 2.0).sum()),
        "3": int((df["has_asthma"] == 3.0).sum()),
        "missing": int(df["has_asthma"].isna().sum()),
    }
    for c in ["has_eczema", "family_history_allergy", "is_hidden_allergen_flag"]:
        df[c] = _to_binary(df[c])
    df["is_hidden_allergen_flag"] = df["is_hidden_allergen_flag"].fillna(0)

    age = pd.to_numeric(df["age"], errors="coerce")
    impossible = (age < AGE_MIN) | (age > AGE_MAX)
    rep["impossible_age_set_missing"] = int(impossible.sum())
    df["age"] = age.mask(impossible)

    df["profile"] = df["profile"].map(_clean_profile)
    in_profile = [t in parse_profile(p) for t, p in zip(df["detected_allergen"], df["profile"])]
    contradict = np.array(in_profile) & (df["reacted"] == 0).to_numpy()
    rep["profile_target_contradictions_dropped"] = int(contradict.sum())
    df = df[~contradict].reset_index(drop=True)

    before = len(df)
    df = df.drop_duplicates(subset=["person_id", "detected_allergen", "profile"]).reset_index(drop=True)
    rep["same_person_question_dropped"] = int(before - len(df))

    flags = pd.DataFrame([kb.protein_flags(k) for k in df["detected_allergen"]])
    changed = 0
    for c in kb.PROTEIN_COLS:
        if c in df.columns:
            changed += int((pd.to_numeric(df[c], errors="coerce") != flags[c]).sum())
        else:
            changed += len(df)
        df[c] = flags[c].to_numpy()
    rep["protein_flag_cells_repaired"] = changed

    df["reacted"] = df["reacted"].astype(int)
    rep["missing_after_cleaning"] = {c: int(df[c].isna().sum()) for c in
                                     ["age", "sex", "has_asthma", "has_eczema", "family_history_allergy"]}
    rep["rows_out"] = int(len(df))
    rep["persons"] = int(df["person_id"].nunique())
    keep = ["record_id", "person_id", "source", "age", "sex", "has_asthma", "has_eczema",
            "family_history_allergy", "profile", "detected_allergen", "is_hidden_allergen_flag"] \
        + kb.PROTEIN_COLS + ["reacted", "severity"]
    return df[[c for c in keep if c in df.columns]], rep


def row_type(df: pd.DataFrame) -> np.ndarray:
    """'index' (no profile), 'in_profile' (target is one of the person's allergens) or 'cross'."""
    out = []
    for t, p in zip(df["detected_allergen"], df["profile"]):
        pf = parse_profile(p)
        out.append("index" if not pf else ("in_profile" if t in pf else "cross"))
    return np.array(out)


# ---------------------------------------------------------------- features
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    P, IDX = xr.P_matrix(), xr.IDX
    X = pd.DataFrame(index=df.index)
    for c in ["age", "sex", "has_asthma", "has_eczema", "family_history_allergy",
              "is_hidden_allergen_flag"]:
        X[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    allergen = df["detected_allergen"].astype(str)
    flags = pd.DataFrame([kb.protein_flags(k) for k in allergen], index=df.index)
    for c in kb.PROTEIN_COLS:
        X[c] = flags[c].astype("float64")
    # atopic_burden uses a binary indicator for any asthma (level > 0),
    # preserving NaN if asthma is missing, rather than mixing the 0-3 ordinal scale.
    asthma_any = pd.Series(
        np.where(X["has_asthma"].isna(), np.nan, (X["has_asthma"] > 0.0).astype(float)),
        index=df.index
    )
    atopic_parts = pd.DataFrame({
        "asthma_any": asthma_any,
        "has_eczema": X["has_eczema"],
        "family_history_allergy": X["family_history_allergy"]
    }, index=df.index)
    X["atopic_burden"] = atopic_parts.sum(axis=1, min_count=1)
    X["n_protein_families"] = flags.sum(axis=1).astype("float64")

    f = {k: np.full(n, np.nan) for k in PROFILE_FEATURES}
    for k in ("target_in_profile", "n_other_known", "related_link", "n_related_known",
              "n_same_group_known"):
        f[k][:] = 0.0
    for i, (t, ps) in enumerate(zip(allergen.to_numpy(), df["profile"].fillna("").to_numpy())):
        pf = parse_profile(ps)
        ti = IDX.get(t)
        if t in pf:
            f["target_in_profile"][i] = 1.0
            f["self_rating"][i] = pf[t]
        others = [(k, r) for k, r in pf.items() if k != t and k in IDX]
        f["n_other_known"][i] = len(others)
        if not others or ti is None:
            continue
        links = np.array([P[IDX[k], ti] for k, _ in others])
        f["related_link"][i] = links.max()
        f["n_related_known"][i] = int((links >= 0.25).sum())
        gt = kb.group_of(t)
        same = [r for k, r in others if kb.group_of(k) == gt]
        f["n_same_group_known"][i] = len(same)
        same_r = [r for r in same if not np.isnan(r)]
        if same_r:
            f["same_group_max_sev"][i] = max(same_r)
        rated = [(k, r, l) for (k, r), l in zip(others, links) if not np.isnan(r)]
        if rated:
            rs = np.array([r for _, r, _ in rated])
            ls = np.array([l for _, _, l in rated])
            f["max_other_sev"][i] = rs.max()
            f["mean_other_sev"][i] = rs.mean()
            j = int(ls.argmax())
            if ls[j] >= 0.05:
                f["related_sev"][i] = rs[j]
            m = ls >= 0.05
            if m.any():
                f["related_wmean_sev"][i] = (ls[m] * rs[m]).sum() / ls[m].sum()
    for k in PROFILE_FEATURES:
        X[k] = f[k]

    X["detected_allergen"] = pd.Categorical(allergen, categories=kb.all_allergens())
    X["allergen_group"] = pd.Categorical(allergen.map(kb.group_of), categories=kb.GROUP_NAMES)
    return X[FEATURES]


def build_query_df(profile: dict, target: str, age=None, sex=None, asthma=None, eczema=None,
                   family=None, hidden=0) -> pd.DataFrame:
    """One app question in the same format as the cleaned table."""
    prof = {}
    for k, v in (profile or {}).items():
        key = kb.canonicalize(k)
        if key:
            prof[key] = np.nan if v is None else float(v)
    asthma_val = np.nan
    if asthma is not None and not pd.isna(asthma):
        try:
            av = float(asthma)
            if av in (0.0, 1.0, 2.0, 3.0):
                asthma_val = av
        except (ValueError, TypeError):
            pass
    return pd.DataFrame([{
        "age": np.nan if age is None else age, "sex": np.nan if sex is None else sex,
        "has_asthma": asthma_val,
        "has_eczema": np.nan if eczema is None else eczema,
        "family_history_allergy": np.nan if family is None else family,
        "is_hidden_allergen_flag": int(bool(hidden)),
        "profile": profile_to_string(prof), "detected_allergen": target}])


if __name__ == "__main__":
    import sys
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data" / "allergy_raw_50k.csv"
    clean, report = clean_raw(pd.read_csv(src))
    print(json.dumps(report, indent=2))
    X = make_features(clean)
    print(X.shape)
    print(X[PROFILE_FEATURES].describe().T[["count", "mean", "min", "max"]].round(2))
