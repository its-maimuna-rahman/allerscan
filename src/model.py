"""
Inference for the app using the Tri-Model GBDT Ensemble (LightGBM + XGBoost + CatBoost)
with the 3-grade clinical severity triage scale:
  Level 1: Mild / Localized (oral itch, localized hives)
  Level 2: Moderate / Systemic (spreading hives, angioedema, GI distress)
  Level 3: Severe / Anaphylaxis (respiratory difficulty, tight throat, epinephrine needed)

predict_severity(query) returns, for one detected allergen:
  mode              "profile"   the allergen is one the user listed (their own rating is used)
                    "cross"     not listed, but closely related to something they listed
                    "unrelated" not listed and no meaningful link to their profile
                    "general"   the user listed nothing, population estimate
  severity_score    1-3 (safety rule: leans to 3 when P(3) passes the tuned safety threshold)
  probabilities     P(level 1..3)
  p_react           chance the user reacts at all (cross / unrelated modes only)
  related           the listed allergen it is most closely linked to, with the link strength
  drivers           what pushed the estimate up or down (TreeSHAP from LightGBM)
"""
from functools import lru_cache
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

import allergen_kb as kb
import cross_reactivity as xr
import pipeline as pl

ARTIFACT = Path(__file__).parent / "allergy_model.joblib"
CROSS_MIN = 0.20          # link strength or reaction chance that makes a card a "possible cross-reaction"
CAT_COLS = ["detected_allergen", "allergen_group"]


@lru_cache(maxsize=1)
def _load():
    art = joblib.load(ARTIFACT)
    if isinstance(art.get("sev_models"), dict):
        sev_lgb = [lgb.Booster(model_str=s) for s in art["sev_models"]["lgb"]]
        sev_xgb = art["sev_models"]["xgb"]
        sev_cat = art["sev_models"]["cat"]
        react_lgb = [lgb.Booster(model_str=s) for s in art["react_models"]["lgb"]]
        react_xgb = art["react_models"]["xgb"]
        react_cat = art["react_models"]["cat"]
    else:
        sev_lgb = [lgb.Booster(model_str=s) for s in art["sev_models"]]
        sev_xgb, sev_cat = [], []
        react_lgb = [lgb.Booster(model_str=s) for s in art["react_models"]]
        react_xgb, react_cat = [], []
    return art, sev_lgb, sev_xgb, sev_cat, react_lgb, react_xgb, react_cat


def model_info() -> dict:
    try:
        art, _, _, _, _, _, _ = _load()
    except Exception as exc:                                    # noqa: BLE001
        return {"error": f"Model not available: {exc}"}
    return {k: art.get(k) for k in ("trained_at", "n_rows", "severe_threshold", "test_severity",
                                    "test_reactivity", "seeds", "num_classes", "algorithms")}


def _decide(proba, threshold):
    """Level 3 (index 2) is severe. If P(Level 3) >= threshold, prioritize safety."""
    if proba[2] >= threshold:
        return 2
    return int(np.argmax(proba))


def _predict_ensemble(X, lgb_models, xgb_models, cat_models, is_binary=False):
    """Calibrated blend: 35% LightGBM + 35% CatBoost + 30% XGBoost."""
    preds = []
    if lgb_models:
        p_lgb = np.mean([b.predict(X) for b in lgb_models], axis=0)
        preds.append((0.35, p_lgb))
    if xgb_models:
        if is_binary:
            p_xgb = np.mean([m.predict_proba(X)[:, 1] for m in xgb_models], axis=0)
        else:
            p_xgb = np.mean([m.predict_proba(X) for m in xgb_models], axis=0)
        preds.append((0.30, p_xgb))
    if cat_models:
        Xc = X.copy()
        for c in CAT_COLS:
            if c in Xc.columns:
                Xc[c] = Xc[c].astype(str)
        if is_binary:
            p_cat = np.mean([m.predict_proba(Xc)[:, 1] for m in cat_models], axis=0)
        else:
            p_cat = np.mean([m.predict_proba(Xc) for m in cat_models], axis=0)
        preds.append((0.35, p_cat))

    total_w = sum(w for w, _ in preds)
    return sum(w * p for w, p in preds) / total_w


def _label(name, x, ctx):
    v = x[name]
    if name == "self_rating":
        return None if pd.isna(v) else f"Your own rating ({int(v)}/3)"
    if name == "related_sev":
        return None if pd.isna(v) or not ctx["related"] else f"Your {kb.pretty(ctx['related']['name'])} rating ({int(v)}/3)"
    if name == "related_wmean_sev":
        return None if pd.isna(v) else "Your ratings of related allergens"
    if name == "related_link":
        return None if v < 0.1 or not ctx["related"] else f"Close to your {kb.pretty(ctx['related']['name'])} allergy"
    if name in ("max_other_sev", "mean_other_sev"):
        return None if pd.isna(v) else "Your ratings of your other allergens"
    if name == "target_in_profile":
        return "Listed in your profile" if v == 1 else "Not in your profile"
    if name == "n_other_known":
        return f"{int(v)} other known allergen(s)"
    if name in ("n_related_known", "n_same_group_known", "same_group_max_sev"):
        return None if (pd.isna(v) or v == 0) else "Related allergens in your profile"
    if name == "detected_allergen":
        return f"Allergen: {kb.pretty(v)}"
    if name == "allergen_group":
        return f"Food group: {kb.pretty(v)}"
    if name == "age":
        return None if pd.isna(v) else f"Age {v:.0f}"
    if name in kb.PROTEIN_COLS:
        return f"Protein family: {kb.pretty(name)}" if v == 1 else None
    if name == "has_asthma":
        asthma_labels = {1: "Mild asthma", 2: "Moderate asthma", 3: "Severe asthma"}
        return asthma_labels.get(int(v)) if (pd.notna(v) and v > 0) else None
    nice = {"has_eczema": "Eczema", "family_history_allergy": "Family history",
            "is_hidden_allergen_flag": "Hidden or trace source", "atopic_burden": "Overall atopic burden",
            "n_protein_families": "Protein families", "sex": "Sex"}
    if name in ("has_eczema", "family_history_allergy", "is_hidden_allergen_flag"):
        return nice[name] if v == 1 else None
    return nice.get(name)



def predict_severity(q: dict) -> dict:
    try:
        art, sev_lgb, sev_xgb, sev_cat, react_lgb, react_xgb, react_cat = _load()
    except Exception as exc:                                    # noqa: BLE001
        return {"error": f"Model artifacts not loaded: {exc}"}
    target = kb.canonicalize(q.get("detected_allergen", ""))
    if target is None:
        return {"error": f"Unknown allergen: {q.get('detected_allergen')!r}"}

    profile = {}
    for k, v in (q.get("known_allergens") or {}).items():
        key = kb.canonicalize(k)
        if key:
            profile[key] = v
    in_profile = target in profile
    asthma_input = q.get("asthma_flag")
    if asthma_input is not None and not pd.isna(asthma_input):
        try:
            av = float(asthma_input)
            asthma_val = av if av in (0.0, 1.0, 2.0, 3.0) else np.nan
        except (ValueError, TypeError):
            asthma_val = np.nan
    else:
        asthma_val = np.nan

    df = pl.build_query_df(profile, target, age=q.get("age"), sex=q.get("sex"),
                           asthma=asthma_val, eczema=q.get("eczema_flag"),
                           family=q.get("family_history_flag"), hidden=q.get("is_hidden_allergen"))
    X = pl.make_features(df)

    proba = _predict_ensemble(X, sev_lgb, sev_xgb, sev_cat, is_binary=False)[0]
    idx = _decide(proba, art["severe_threshold"])

    others = {k: v for k, v in profile.items() if k != target}
    related = None
    if others:
        k_best = max(others, key=lambda k: xr.link(k, target))
        related = {"name": k_best, "link": round(xr.link(k_best, target), 3), "rating": others[k_best]}

    p_react = None
    if profile and not in_profile:
        p_react = float(_predict_ensemble(X[art["react_features"]], react_lgb, react_xgb, react_cat, is_binary=True)[0])

    if in_profile:
        mode = "profile"
    elif not profile:
        mode = "general"
    elif (related and related["link"] >= CROSS_MIN) or (p_react is not None and p_react >= CROSS_MIN):
        mode = "cross"
    else:
        mode = "unrelated"

    contrib = np.mean([b.predict(X, pred_contrib=True)[0].reshape(3, -1) for b in sev_lgb], axis=0)
    ctx = {"related": related}
    xr_row = X.iloc[0]
    drivers = []
    for name, val in sorted(zip(X.columns, contrib[idx][:-1]), key=lambda t: -abs(t[1])):
        label = _label(name, xr_row, ctx)
        if label and abs(val) >= 0.03:
            drivers.append({"label": label, "impact": round(float(val), 3)})
    drivers = drivers[:6]

    return {
        "allergen": target, "mode": mode, "in_profile": in_profile,
        "severity_score": idx + 1,
        "probabilities": [round(float(p), 3) for p in proba],
        "expected_level": round(float(np.dot([1, 2, 3], proba)), 2),
        "severe_risk": round(float(proba[2]), 3),
        "severe_flag": bool(proba[2] >= art["severe_threshold"]),
        "p_react": None if p_react is None else round(p_react, 3),
        "related": related, "rating": profile.get(target) if in_profile else None,
        "drivers": drivers,
    }
