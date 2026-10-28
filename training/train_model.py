"""
Train AllerScan with a Tri-Model GBDT Ensemble (LightGBM + XGBoost + CatBoost)
using a 3-tier clinical severity scale (1: Mild, 2: Moderate, 3: Severe).

  Severity head   multiclass 1-3, trained on rows where the person reacts.
                  Ensemble: LightGBM + XGBoost + CatBoost blend.
  Reactivity head binary, trained on "cross" rows (allergen not in profile):
                  Ensemble: LightGBM + XGBoost + CatBoost blend.

Steps
  1. clean the raw export, print cleaning + dataset checks
  2. hold out 20% of PEOPLE as a final test set (never used for tuning)
  3. 5-fold Stratified Group K-Fold cross-validation (grouped by person)
  4. Train LightGBM + XGBoost + CatBoost across 5 folds with out-of-fold blending
  5. Calibrate safety threshold for "severe" (level 3) on out-of-fold probabilities
  6. Held-out evaluation: overall, by row type, by source cohort, by food group, calibration
  7. Leave-one-source-out check (does it hold up on an unseen source cohort?)
  8. Sanity probes (does almond=1 give a low cashew estimate? etc.)
  9. Final models fit across seeds and saved to allergy_model.joblib

Run:  python train_model.py
Quick check: python train_model.py --skip-loso
"""
import argparse
import json
import os
import platform
import time
import warnings
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             classification_report, cohen_kappa_score, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

import allergen_kb as kb
import pipeline as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

BASE = ROOT
DATA = ROOT / "data"
SRC = ROOT / "src"
REPORTS = ROOT / "reports"
SPLIT_SEED = 42
MODEL_SEEDS = [42, 7, 123, 2024, 99]
LEARNING_RATE = 0.05
MAX_ROUNDS = 300
SEVERE_RECALL_TARGET = 0.88
CAT_COLS = ["detected_allergen", "allergen_group"]


def detect_hardware(requested_device="auto", requested_threads=None):
    threads = requested_threads or max(1, os.cpu_count() or 1)
    cpu_name = platform.processor() or "CPU"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    cpu_name = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    try:
        import torch
        has_cuda = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if has_cuda else "None"
        cuda_ver = torch.version.cuda or "N/A"
    except Exception:
        has_cuda = False
        gpu_name = "None"
        cuda_ver = "N/A"

    use_gpu = False
    if requested_device in ("auto", "gpu", "cuda"):
        use_gpu = has_cuda
    elif requested_device == "cpu":
        use_gpu = False

    xgb_device = "cuda" if use_gpu else "cpu"
    cat_task = "GPU" if use_gpu else "CPU"

    return {
        "use_gpu": use_gpu,
        "gpu_name": gpu_name if use_gpu else "Disabled / CPU Mode",
        "cuda_ver": cuda_ver,
        "cpu_name": cpu_name,
        "threads": threads,
        "xgb_device": xgb_device,
        "cat_task": cat_task,
    }


HW_INFO = detect_hardware()
N_THREADS = HW_INFO["threads"]
XGB_DEVICE = HW_INFO["xgb_device"]
CAT_TASK = HW_INFO["cat_task"]

LGB_PARAMS_SEV = dict(objective="multiclass", num_class=3, learning_rate=LEARNING_RATE,
                      num_leaves=31, max_depth=6, min_data_in_leaf=30, bagging_fraction=0.85,
                      bagging_freq=1, feature_fraction=0.85, lambda_l2=5.0, verbose=-1,
                      num_threads=N_THREADS, deterministic=True, force_row_wise=True)

LGB_PARAMS_BIN = dict(objective="binary", learning_rate=LEARNING_RATE, num_leaves=20,
                      max_depth=5, min_data_in_leaf=30, bagging_fraction=0.85, bagging_freq=1,
                      feature_fraction=0.85, lambda_l2=5.0, verbose=-1, num_threads=N_THREADS,
                      deterministic=True, force_row_wise=True)


def log(*a):
    print(*a, flush=True)


def section(title):
    log("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


def show(df_or_text):
    log(df_or_text.to_string() if hasattr(df_or_text, "to_string") else df_or_text)


# ------------------------------------------------------------------ Tri-Model Training Helpers
def fit_lgb(X, y, params, seed=SPLIT_SEED, rounds=100):
    p = {**params, "seed": seed}
    return lgb.train(p, lgb.Dataset(X, y), rounds)


def fit_xgb_sev(X, y, seed=SPLIT_SEED, rounds=100):
    m = xgb.XGBClassifier(n_estimators=rounds, max_depth=5, learning_rate=LEARNING_RATE,
                          eval_metric="mlogloss", enable_categorical=True, tree_method="hist",
                          device=XGB_DEVICE, subsample=0.85, colsample_bytree=0.85, reg_lambda=5.0,
                          random_state=seed, n_jobs=N_THREADS)
    m.fit(X, y)
    return m


def fit_xgb_bin(X, y, seed=SPLIT_SEED, rounds=100):
    m = xgb.XGBClassifier(n_estimators=rounds, max_depth=5, learning_rate=LEARNING_RATE,
                          eval_metric="logloss", enable_categorical=True, tree_method="hist",
                          device=XGB_DEVICE, subsample=0.85, colsample_bytree=0.85, reg_lambda=5.0,
                          random_state=seed, n_jobs=N_THREADS)
    m.fit(X, y)
    return m


def fit_cat_sev(X, y, seed=SPLIT_SEED, rounds=100):
    Xc = X.copy()
    for c in CAT_COLS:
        if c in Xc.columns:
            Xc[c] = Xc[c].astype(str)
    m = CatBoostClassifier(iterations=rounds, depth=5, learning_rate=LEARNING_RATE,
                           loss_function="MultiClass", task_type=CAT_TASK,
                           cat_features=[c for c in CAT_COLS if c in Xc.columns],
                           verbose=0, random_state=seed, thread_count=N_THREADS)
    m.fit(Xc, y)
    return m


def fit_cat_bin(X, y, seed=SPLIT_SEED, rounds=100):
    Xc = X.copy()
    for c in CAT_COLS:
        if c in Xc.columns:
            Xc[c] = Xc[c].astype(str)
    m = CatBoostClassifier(iterations=rounds, depth=5, learning_rate=LEARNING_RATE,
                           loss_function="Logloss", task_type=CAT_TASK,
                           cat_features=[c for c in CAT_COLS if c in Xc.columns],
                           verbose=0, random_state=seed, thread_count=N_THREADS)
    m.fit(Xc, y)
    return m


def predict_tri_blend(X, lgb_m, xgb_m, cat_m, is_binary=False):
    """Calibrated blend: 35% LightGBM + 35% CatBoost + 30% XGBoost."""
    p_lgb = lgb_m.predict(X)
    if is_binary:
        p_xgb = xgb_m.predict_proba(X)[:, 1]
    else:
        p_xgb = xgb_m.predict_proba(X)
    
    Xc = X.copy()
    for c in CAT_COLS:
        if c in Xc.columns:
            Xc[c] = Xc[c].astype(str)
    if is_binary:
        p_cat = cat_m.predict_proba(Xc)[:, 1]
    else:
        p_cat = cat_m.predict_proba(Xc)
    
    return 0.35 * p_lgb + 0.35 * p_cat + 0.30 * p_xgb


# ------------------------------------------------------------------ severity helpers (3 classes)
def severe(y):
    """Level 3 (index 2) is severe anaphylaxis."""
    return (np.asarray(y) == 2).astype(int)


def pick_threshold(p_sev, y_sev, target):
    best = 0.05
    for t in np.arange(0.05, 0.96, 0.005):
        if recall_score(y_sev, (p_sev >= t).astype(int)) >= target:
            best = float(t)
    return round(best, 3)


def decide(proba, t):
    """Safety first: flag severe when P(Level 3) >= t, otherwise best of 1 or 2."""
    return np.where(proba[:, 2] >= t, 2, proba[:, :2].argmax(1))


def sev_metrics(y, proba, pred):
    y = np.asarray(y)
    p_sev = proba[:, 2]
    out = dict(n=int(len(y)), accuracy=accuracy_score(y, pred),
               within_one=float(np.mean(np.abs(y - pred) <= 1)),
               macro_f1=f1_score(y, pred, average="macro"),
               qwk=cohen_kappa_score(y, pred, weights="quadratic"),
               logloss=log_loss(y, proba, labels=[0, 1, 2]),
               mean_abs_err=float(np.mean(np.abs(y - pred))))
    if 0 < severe(y).sum() < len(y):
        out.update(severe_auc=roc_auc_score(severe(y), p_sev),
                   severe_recall=recall_score(severe(y), severe(pred)),
                   severe_precision=precision_score(severe(y), severe(pred), zero_division=0))
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def calibration_table(y_true, p, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -1e-9, 1 + 1e-9
    idx = np.clip(np.searchsorted(np.unique(edges), p, side="right") - 1, 0, None)
    rows = []
    for b in np.unique(idx):
        m = idx == b
        rows.append(dict(bin=int(b), n=int(m.sum()), mean_predicted=round(float(p[m].mean()), 3),
                         observed=round(float(np.mean(y_true[m])), 3)))
    tab = pd.DataFrame(rows)
    ece = float(np.sum(tab.n * (tab.mean_predicted - tab.observed).abs()) / tab.n.sum())
    return tab, round(ece, 4)


def text_confusion(cm, names):
    head = "actual \\ pred  " + "".join(f"{n:>8}" for n in names) + "    recall"
    lines = [head]
    for i, n in enumerate(names):
        tot = cm[i].sum()
        lines.append(f"{n:>13}  " + "".join(f"{v:>8}" for v in cm[i]) + f"   {cm[i, i] / tot:>6.1%}" if tot else n)
    prec = [cm[j, j] / cm[:, j].sum() if cm[:, j].sum() else float("nan") for j in range(len(names))]
    lines.append("   precision   " + "".join(f"{p:>8.1%}" for p in prec))
    return "\n".join(lines)


def expected_level(p):
    return float(np.dot([1, 2, 3], p))


# ------------------------------------------------------------------ probes (adapted to 3 grades)
def probe(sev_models, react_models, profile, target, **demo):
    q = pl.build_query_df(profile, target, **{"age": 30, "sex": 0, "asthma": 0, "eczema": 0,
                                              "family": 0, **demo})
    X = pl.make_features(q)
    # Average across ensemble seeds
    ps = np.mean([predict_tri_blend(X, lgb_m, xgb_m, cat_m, is_binary=False)
                  for lgb_m, xgb_m, cat_m in sev_models], axis=0)[0]
    pr = None
    if profile and target not in profile:
        pr = float(np.mean([predict_tri_blend(X[pl.REACT_FEATURES], lgb_m, xgb_m, cat_m, is_binary=True)
                            for lgb_m, xgb_m, cat_m in react_models]))
    return ps, pr


def run_probes(sev_models, react_models, t_sev):
    def case(profile, target, **demo):
        ps, pr = probe(sev_models, react_models, profile, target, **demo)
        return dict(profile=profile, target=target, E=expected_level(ps), p_sev=float(ps[2]),
                    level=int(decide(ps[None], t_sev)[0]) + 1, p_react=pr, probs=ps)

    checks, rows = [], []

    def rec(label, c):
        rows.append(dict(case=label, profile=str(c["profile"]), target=c["target"],
                         expected_level=round(c["E"], 2), shown_level=c["level"],
                         P_level_3=round(c["p_sev"], 3),
                         P_react="-" if c["p_react"] is None else round(c["p_react"], 3)))

    a1, a3 = case({"almond": 1}, "cashew"), case({"almond": 3}, "cashew")
    rec("almond=1 -> cashew", a1); rec("almond=3 -> cashew", a3)
    checks += [("almond=1 -> cashew: low severity (expected level <= 1.85)", a1["E"] <= 1.85),
               ("almond=1 -> cashew: P(level 3) <= 0.25", a1["p_sev"] <= 0.25),
               ("cashew estimate rises with the almond rating (3 vs 1 by >= 0.4 level)", a3["E"] - a1["E"] >= 0.4)]

    s1, s3 = case({"shrimp": 1}, "crab"), case({"shrimp": 3}, "crab")
    rec("shrimp=1 -> crab", s1); rec("shrimp=3 -> crab", s3)
    checks += [("shrimp -> crab: high chance of reacting (>= 0.5)", s1["p_react"] >= 0.5),
               ("crab estimate rises with the shrimp rating (by >= 0.4 level)", s3["E"] - s1["E"] >= 0.4)]

    m1 = case({"cow_milk": 1}, "goat_milk")
    rec("cow_milk=1 -> goat_milk", m1)
    checks += [("cow milk -> goat milk: high chance (>= 0.6) and mild if rated 1", m1["p_react"] >= 0.6 and m1["E"] <= 1.85)]

    ap, sh = case({"almond": 1}, "apple"), case({"almond": 1}, "shrimp")
    ca, cs = case({"almond": 2}, "cashew"), case({"almond": 2}, "shrimp")
    rec("almond=1 -> apple", ap); rec("almond=1 -> shrimp (unrelated)", sh)
    checks += [("unrelated shrimp has a far lower reaction chance than related cashew (cashew >= 3x)",
                ca["p_react"] >= 3 * cs["p_react"]),
               ("almond=1 -> shrimp: reaction chance < 0.10", sh["p_react"] < 0.10)]

    p3 = case({"peanut": 3}, "almond")
    rec("peanut=3 -> almond", p3)
    checks += [("peanut -> almond: moderate chance (0.05 - 0.40)", 0.05 <= p3["p_react"] <= 0.40)]

    g_c, g_p = case({}, "cashew"), case({}, "apple")
    rec("no profile -> cashew (general)", g_c); rec("no profile -> apple (general)", g_p)
    checks += [("no profile: cashew estimate above apple estimate", g_c["E"] > g_p["E"] + 0.25)]

    # Strengthened asthma probes across nut and non-nut profiles (Task 1 - Step 8)
    probe_pairs = [
        ({"peanut": 2}, "cashew", "nut target (peanut -> cashew)"),
        ({"cow_milk": 2}, "goat_milk", "dairy target (cow milk -> goat milk)"),
        ({"shrimp": 2}, "crab", "crustacean target (shrimp -> crab)"),
    ]
    asthma_res = []
    for prof, tgt, label in probe_pairs:
        p_cases = {a: case(prof, tgt, asthma=a) for a in [0, 1, 2, 3, None]}
        for a in [0, 1, 2, 3]:
            rec(f"{label} [asthma={a}]", p_cases[a])
        rec(f"{label} [asthma=missing]", p_cases[None])
        asthma_res.append(p_cases)

    # 1. Expected severity must not decrease from asthma 0 to 1 to 2 to 3 (tolerance 0.01)
    e_mono_all = all(
        p[1]["E"] >= p[0]["E"] - 0.01 and
        p[2]["E"] >= p[1]["E"] - 0.01 and
        p[3]["E"] >= p[2]["E"] - 0.01
        for p in asthma_res
    )
    checks.append(("asthma expected severity monotonic (0 -> 1 -> 2 -> 3) across nut & non-nut targets", e_mono_all))

    # 2. Probability of severe level must not decrease with level (tolerance 0.01)
    p_sev_mono_all = all(
        p[1]["p_sev"] >= p[0]["p_sev"] - 0.01 and
        p[2]["p_sev"] >= p[1]["p_sev"] - 0.01 and
        p[3]["p_sev"] >= p[2]["p_sev"] - 0.01
        for p in asthma_res
    )
    checks.append(("asthma severe probability P(level 3) monotonic (0 -> 1 -> 2 -> 3)", p_sev_mono_all))

    # 3. Cross-reactive target reaction probability must not decrease with level (tolerance 0.01)
    pr_mono_all = all(
        (p[1]["p_react"] >= p[0]["p_react"] - 0.01 and
         p[2]["p_react"] >= p[1]["p_react"] - 0.01 and
         p[3]["p_react"] >= p[2]["p_react"] - 0.01)
        for p in asthma_res if p[0]["p_react"] is not None
    )
    checks.append(("asthma cross-reactivity P(react) monotonic with asthma level", pr_mono_all))

    # 4. Modest effect size: E(level 3) - E(level 0) stays below limit (<= 0.85)
    modest_effect = all(
        0.0 <= p[3]["E"] - p[0]["E"] <= 0.85
        for p in asthma_res
    )
    checks.append(("asthma effect size is modest (E(asthma=3) - E(asthma=0) <= 0.85 across profiles)", modest_effect))

    # 5. Missing asthma gives result between levels 0 and 3 / population average, does not crash
    missing_valid = all(
        (1.0 <= p[None]["E"] <= 3.0 and
         p[None]["E"] >= p[0]["E"] - 0.05 and
         p[None]["E"] <= p[3]["E"] + 0.05)
        for p in asthma_res
    )
    checks.append(("missing asthma yields valid bounded prediction (no crash, bounded between 0 and 3)", missing_valid))

    return pd.DataFrame(rows), checks


# ------------------------------------------------------------------ plots
def make_plots(cm, calib, sev_lgb_models, react_lgb_models):
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    cmn = cm / cm.sum(1, keepdims=True)
    im = ax.imshow(cmn, cmap="YlGnBu", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cmn[i, j]:.1%}\n({cm[i, j]})", ha="center", va="center", fontsize=9,
                    color="white" if cmn[i, j] > 0.55 else "#1b2733")
    ax.set_xticks(range(3), labels=["L1 (Mild)", "L2 (Mod)", "L3 (Severe)"])
    ax.set_yticks(range(3), labels=["L1 (Mild)", "L2 (Mod)", "L3 (Severe)"])
    ax.set_xlabel("Predicted severity"); ax.set_ylabel("Actual severity")
    ax.set_title("Held-out people: 3-tier severity confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046); fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0, 1], [0, 1], "--", color="#999")
    ax.plot(calib.mean_predicted, calib.observed, "o-", color="#1B2A41")
    ax.set_xlabel("Predicted chance of reacting"); ax.set_ylabel("Observed")
    ax.set_title("Reactivity head: calibration (held-out)"); fig.tight_layout()
    fig.savefig(REPORTS / "reactivity_calibration.png", dpi=140); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, models, title in ((axs[0], sev_lgb_models, "Severity head"), (axs[1], react_lgb_models, "Reactivity head")):
        imp = pd.Series(np.mean([m.feature_importance("gain") for m in models], axis=0),
                        index=models[0].feature_name()).sort_values().tail(15)
        ax.barh(imp.index, imp.values, color="#2f6f6a"); ax.set_title(title + ": top features (gain)")
    fig.tight_layout(); fig.savefig(REPORTS / "feature_importance.png", dpi=140); plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="AllerScan 2.0 Tri-Model Training (RTX 4050 GPU + Intel Core i7)")
    ap.add_argument("--raw", default=str(DATA / "allergy_raw_50k.csv"), help="Path to raw CSV dataset")
    ap.add_argument("--clean-data", default=None,
                    help="Path to pre-cleaned CSV dataset (e.g. data/allergy_clean.csv). If given, skips raw cleaning.")
    ap.add_argument("--device", choices=["auto", "gpu", "cuda", "cpu"], default="auto",
                    help="Hardware compute device (default: auto -> NVIDIA GPU if available)")
    ap.add_argument("--threads", type=int, default=HW_INFO["threads"],
                    help=f"CPU thread count for LightGBM and data operations (default: {HW_INFO['threads']})")
    ap.add_argument("--skip-loso", action="store_true", help="Skip Leave-One-Source-Out validation")
    ap.add_argument("--quick", action="store_true", help="Quick smoke test (2 folds, 1 seed, 30 rounds) for fast testing")
    args = ap.parse_args()
    t0 = time.time()

    global N_THREADS, XGB_DEVICE, CAT_TASK, LGB_PARAMS_SEV, LGB_PARAMS_BIN
    hw = detect_hardware(requested_device=args.device, requested_threads=args.threads)
    N_THREADS = hw["threads"]
    XGB_DEVICE = hw["xgb_device"]
    CAT_TASK = hw["cat_task"]

    LGB_PARAMS_SEV["num_threads"] = N_THREADS
    LGB_PARAMS_BIN["num_threads"] = N_THREADS

    k = 5 if not args.quick else 2
    model_seeds = MODEL_SEEDS if not args.quick else [42]
    rounds_sev = 120 if not args.quick else 30
    rounds_react = 100 if not args.quick else 25

    section("0. ENVIRONMENT & HARDWARE ACCELERATION")
    log(f"Python:       {platform.python_version()} | Platform: {platform.platform()}")
    log(f"Packages:     lightgbm {lgb.__version__} | xgboost {xgb.__version__} | catboost {CatBoostClassifier().__module__} | sklearn {sklearn.__version__} | pandas {pd.__version__}")
    log(f"CPU:          {hw['cpu_name']} ({hw['threads']} threads active)")
    log(f"GPU:          {hw['gpu_name']} (CUDA version: {hw['cuda_ver']})")
    log(f"Backends:     XGBoost [device='{XGB_DEVICE}', tree_method='hist'] | CatBoost [task_type='{CAT_TASK}'] | LightGBM [threads={N_THREADS}]")
    log(f"Architecture: Tri-Model GBDT Ensemble (35% LightGBM + 35% CatBoost + 30% XGBoost)")
    log(f"Severity:     3-Tier Clinical Triage (1=Mild / Localized, 2=Moderate / Systemic, 3=Severe / Anaphylaxis)")
    log(f"Folds/Seeds:  k={k} folds | seeds={model_seeds} | sev_rounds={rounds_sev} | react_rounds={rounds_react}")
    log(f"Started:      {datetime.now().isoformat(timespec='seconds')}")

    # ---------------------------------------------------------------- 1
    section("1. CLEANING AND DATA CHECKS")
    if args.clean_data and Path(args.clean_data).exists():
        log(f"loading pre-cleaned dataset: {args.clean_data}")
        df = pd.read_csv(args.clean_data)
        report = {"loaded_precleaned": str(args.clean_data), "rows_out": len(df), "persons": df.person_id.nunique()}
    else:
        raw = pd.read_csv(args.raw)
        log(f"raw file: {args.raw}  shape {raw.shape}")
        df, report = pl.clean_raw(raw)
        df.to_csv(DATA / "allergy_clean.csv", index=False)
        (DATA / "cleaning_report.json").write_text(json.dumps(report, indent=2))
        show(pd.Series(report).to_string())
    df["row_type"] = pl.row_type(df)
    X_all = pl.make_features(df)
    log(f"\nfeature matrix: {X_all.shape[1]} columns -> {list(X_all.columns)}")
    log(f"\nrows {len(df):,} | people {df.person_id.nunique():,} | rows per person "
        f"{len(df) / df.person_id.nunique():.2f} | distinct allergens {df.detected_allergen.nunique()}")
    show(df.groupby("source").size().rename("rows").to_frame().assign(share=lambda t: (t.rows / t.rows.sum()).round(3)))
    show(df.groupby("row_type").agg(rows=("reacted", "size"), reacts=("reacted", "mean")).round(3))
    r = df[df.reacted == 1]
    log("\nseverity distribution (reacting rows - 3 tiers)")
    show(pd.concat([r.severity.astype(int).value_counts(normalize=True).sort_index().rename("all"),
                    pd.crosstab(r.severity.astype(int), r.row_type, normalize="columns").add_prefix("type_"),
                    pd.crosstab(r.severity.astype(int), r.source, normalize="columns")], axis=1).round(3))

    # ---------------------------------------------------------------- 2
    section("2. SPLIT (by person - 80% Train, 20% Held-Out Test)")
    persons = df.person_id.unique()
    test_p = set(np.random.RandomState(SPLIT_SEED).choice(persons, size=int(0.2 * len(persons)), replace=False))
    is_test = df.person_id.isin(test_p).to_numpy()
    sev_mask = (df.reacted == 1).to_numpy()
    react_mask = (df.row_type == "cross").to_numpy()

    def part(mask, test):
        sel = mask & (is_test if test else ~is_test)
        return X_all[sel].reset_index(drop=True), df[sel].reset_index(drop=True)

    Xs_tr, ds_tr = part(sev_mask, False); Xs_te, ds_te = part(sev_mask, True)
    Xr_tr, dr_tr = part(react_mask, False); Xr_te, dr_te = part(react_mask, True)
    ys_tr, ys_te = ds_tr.severity.astype(int).to_numpy() - 1, ds_te.severity.astype(int).to_numpy() - 1
    yr_tr, yr_te = dr_tr.reacted.to_numpy(), dr_te.reacted.to_numpy()
    gs_tr, gr_tr = ds_tr.person_id.to_numpy(), dr_tr.person_id.to_numpy()
    log(f"people: train {len(persons) - len(test_p):,} / test {len(test_p):,}")
    log(f"severity head rows: train {len(ys_tr):,} / test {len(ys_te):,}")
    log(f"reactivity head rows: train {len(yr_tr):,} / test {len(yr_te):,}")

    # ---------------------------------------------------------------- 3 & 4
    total_sev_iters = len(model_seeds) * k
    section(f"3. {len(model_seeds)}x{k} REPEATED CV: SEVERITY HEAD ({len(model_seeds)} Seeds x {k} Folds = {total_sev_iters} Iterations)")
    oof_s = np.zeros((len(ys_tr), 3))
    iter_sev = 0

    for s_idx, s in enumerate(model_seeds):
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=s)
        oof_s_seed = np.zeros((len(ys_tr), 3))
        for fold, (tr, va) in enumerate(sgkf.split(Xs_tr, ys_tr, gs_tr)):
            iter_sev += 1
            X_f_tr, y_f_tr = Xs_tr.iloc[tr], ys_tr[tr]
            X_f_va, y_f_va = Xs_tr.iloc[va], ys_tr[va]
            
            m_lgb = fit_lgb(X_f_tr, y_f_tr, LGB_PARAMS_SEV, seed=s + fold * 10, rounds=rounds_sev)
            m_xgb = fit_xgb_sev(X_f_tr, y_f_tr, seed=s + fold * 10, rounds=rounds_sev)
            m_cat = fit_cat_sev(X_f_tr, y_f_tr, seed=s + fold * 10, rounds=rounds_sev)
            
            p_fold = predict_tri_blend(X_f_va, m_lgb, m_xgb, m_cat, is_binary=False)
            oof_s_seed[va] = p_fold
            f_acc = accuracy_score(y_f_va, p_fold.argmax(axis=1))
            f_qwk = cohen_kappa_score(y_f_va, p_fold.argmax(axis=1), weights="quadratic")
            log(f"  [Iter {iter_sev:02d}/{total_sev_iters:02d}] Seed {s:>4} | Fold {fold+1}/{k} -> Accuracy: {f_acc:.4f} | QWK: {f_qwk:.4f}")
        
        oof_s += oof_s_seed / len(model_seeds)

    oof_acc = accuracy_score(ys_tr, oof_s.argmax(axis=1))
    oof_qwk = cohen_kappa_score(ys_tr, oof_s.argmax(axis=1), weights="quadratic")
    log(f"\nOOF Severity (averaged across {total_sev_iters} iterations):")
    log(f"  Accuracy: {oof_acc:.4f} | QWK: {oof_qwk:.4f} | Logloss: {log_loss(ys_tr, oof_s):.4f}")

    t_sev = pick_threshold(oof_s[:, 2], severe(ys_tr), SEVERE_RECALL_TARGET)
    oof_rule = sev_metrics(ys_tr, oof_s, decide(oof_s, t_sev))
    log(f"severe threshold (target recall {SEVERE_RECALL_TARGET}): P(Level 3) >= {t_sev}")
    log("OOF metrics with safety rule:", oof_rule)

    total_react_iters = len(model_seeds) * k
    section(f"4. {len(model_seeds)}x{k} REPEATED CV: REACTIVITY HEAD ({len(model_seeds)} Seeds x {k} Folds = {total_react_iters} Iterations)")
    oof_r = np.zeros(len(yr_tr))
    iter_react = 0

    for s_idx, s in enumerate(model_seeds):
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=s)
        oof_r_seed = np.zeros(len(yr_tr))
        for fold, (tr, va) in enumerate(sgkf.split(Xr_tr[pl.REACT_FEATURES], yr_tr, gr_tr)):
            iter_react += 1
            X_rf_tr, y_rf_tr = Xr_tr[pl.REACT_FEATURES].iloc[tr], yr_tr[tr]
            X_rf_va, y_rf_va = Xr_tr[pl.REACT_FEATURES].iloc[va], yr_tr[va]
            
            m_lgb_r = fit_lgb(X_rf_tr, y_rf_tr, LGB_PARAMS_BIN, seed=s + fold * 10, rounds=rounds_react)
            m_xgb_r = fit_xgb_bin(X_rf_tr, y_rf_tr, seed=s + fold * 10, rounds=rounds_react)
            m_cat_r = fit_cat_bin(X_rf_tr, y_rf_tr, seed=s + fold * 10, rounds=rounds_react)
            
            p_fold_r = predict_tri_blend(X_rf_va, m_lgb_r, m_xgb_r, m_cat_r, is_binary=True)
            oof_r_seed[va] = p_fold_r
            f_auc = roc_auc_score(y_rf_va, p_fold_r)
            log(f"  [Iter {iter_react:02d}/{total_react_iters:02d}] Seed {s:>4} | Fold {fold+1}/{k} -> AUC: {f_auc:.4f}")
        
        oof_r += oof_r_seed / len(model_seeds)

    log(f"\nOOF Reactivity (averaged across {total_react_iters} iterations): AUC: {roc_auc_score(yr_tr, oof_r):.4f} | Logloss: {log_loss(yr_tr, oof_r):.4f}")

    # ---------------------------------------------------------------- 5
    section("5. HELD-OUT EVALUATION (people never seen in training)")
    # Train ensemble seeds on full training set
    sev_models_seeds = []
    react_models_seeds = []
    for s in model_seeds:
        lgb_s = fit_lgb(Xs_tr, ys_tr, LGB_PARAMS_SEV, seed=s, rounds=rounds_sev)
        xgb_s = fit_xgb_sev(Xs_tr, ys_tr, seed=s, rounds=rounds_sev)
        cat_s = fit_cat_sev(Xs_tr, ys_tr, seed=s, rounds=rounds_sev)
        sev_models_seeds.append((lgb_s, xgb_s, cat_s))
        
        lgb_r = fit_lgb(Xr_tr[pl.REACT_FEATURES], yr_tr, LGB_PARAMS_BIN, seed=s, rounds=rounds_react)
        xgb_r = fit_xgb_bin(Xr_tr[pl.REACT_FEATURES], yr_tr, seed=s, rounds=rounds_react)
        cat_r = fit_cat_bin(Xr_tr[pl.REACT_FEATURES], yr_tr, seed=s, rounds=rounds_react)
        react_models_seeds.append((lgb_r, xgb_r, cat_r))

    pte = np.mean([predict_tri_blend(Xs_te, l, x, c, is_binary=False) for l, x, c in sev_models_seeds], axis=0)
    pred_plain, pred_rule = pte.argmax(1), decide(pte, t_sev)
    m_plain, m_rule = sev_metrics(ys_te, pte, pred_plain), sev_metrics(ys_te, pte, pred_rule)
    log("SEVERITY HEAD, plain argmax   :", m_plain)
    log("SEVERITY HEAD, safety rule    :", m_rule)
    log("\nclassification report (plain argmax):")
    log(classification_report(ys_te + 1, pred_plain + 1, digits=3))
    cm = confusion_matrix(ys_te + 1, pred_rule + 1, labels=[1, 2, 3])
    log(text_confusion(cm, ["L1 (Mild)", "L2 (Mod)", "L3 (Severe)"]))

    def subset_table(col):
        rows = {}
        for name, idx in ds_te.groupby(col).groups.items():
            ii = np.asarray(idx)
            if len(ii) >= 50:
                rows[name] = sev_metrics(ys_te[ii], pte[ii], pred_plain[ii])
        return pd.DataFrame(rows).T[["n", "accuracy", "within_one", "qwk", "logloss", "mean_abs_err"]]

    ds_te["allergen_group"] = ds_te.detected_allergen.map(kb.group_of)
    log("\nby row type"); show(subset_table("row_type"))
    log("\nby source cohort"); show(subset_table("source"))
    log("\nby food group"); show(subset_table("allergen_group"))

    cross_pred = np.mean([predict_tri_blend(Xr_te[pl.REACT_FEATURES], l, x, c, is_binary=True)
                          for l, x, c in react_models_seeds], axis=0)
    rm = dict(n=int(len(yr_te)), auc=round(roc_auc_score(yr_te, cross_pred), 4),
              avg_precision=round(average_precision_score(yr_te, cross_pred), 4),
              logloss=round(log_loss(yr_te, cross_pred), 4), brier=round(brier_score_loss(yr_te, cross_pred), 4),
              base_rate=round(float(yr_te.mean()), 4),
              baseline_logloss=round(log_loss(yr_te, np.full(len(yr_te), yr_te.mean())), 4))
    cal_r, ece_r = calibration_table(yr_te, cross_pred)
    rm["ece"] = ece_r
    log("\nREACTIVITY HEAD (cross rows):", rm)

    # ---------------------------------------------------------------- 6
    loso = None
    if not args.skip_loso and not args.quick:
        section("6. LEAVE-ONE-SOURCE-OUT (train on four cohorts, test on the fifth)")
        Xs_all, ds_all = X_all[sev_mask].reset_index(drop=True), df[sev_mask].reset_index(drop=True)
        ys_all = ds_all.severity.astype(int).to_numpy() - 1
        rows = {}
        for s in sorted(ds_all.source.unique()):
            te = (ds_all.source == s).to_numpy()
            m_l = fit_lgb(Xs_all[~te], ys_all[~te], LGB_PARAMS_SEV, seed=SPLIT_SEED, rounds=rounds_sev)
            m_x = fit_xgb_sev(Xs_all[~te], ys_all[~te], seed=SPLIT_SEED, rounds=rounds_sev)
            m_c = fit_cat_sev(Xs_all[~te], ys_all[~te], seed=SPLIT_SEED, rounds=rounds_sev)
            p = predict_tri_blend(Xs_all[te], m_l, m_x, m_c, is_binary=False)
            rows[s] = sev_metrics(ys_all[te], p, p.argmax(1))
            log(f"  held out {s}: accuracy {rows[s]['accuracy']}  qwk {rows[s]['qwk']}  logloss {rows[s]['logloss']}")
        loso = pd.DataFrame(rows).T[["n", "accuracy", "within_one", "qwk", "logloss", "severe_auc", "severe_recall"]]
        show(loso)
        log(f"mean across sources: accuracy {loso.accuracy.mean():.3f}  qwk {loso.qwk.mean():.3f}")

    # ---------------------------------------------------------------- 7
    section("7. SANITY PROBES (does the model respect the person's own ratings?)")
    probes, checks = run_probes(sev_models_seeds, react_models_seeds, t_sev)
    show(probes)
    for label, ok in checks:
        log(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    n_pass = sum(bool(ok) for _, ok in checks)
    log(f"probe checks passed: {n_pass}/{len(checks)}")

    # ---------------------------------------------------------------- 8
    section("8. FINAL FIT ON ALL DATA AND SAVE ARTIFACT")
    Xs_full, ys_full = X_all[sev_mask].reset_index(drop=True), df[sev_mask].severity.astype(int).to_numpy() - 1
    Xr_full, yr_full = X_all[react_mask][pl.REACT_FEATURES].reset_index(drop=True), df[react_mask].reacted.to_numpy()

    final_sev_lgb, final_sev_xgb, final_sev_cat = [], [], []
    final_react_lgb, final_react_xgb, final_react_cat = [], [], []

    for s in model_seeds:
        final_sev_lgb.append(fit_lgb(Xs_full, ys_full, LGB_PARAMS_SEV, seed=s, rounds=rounds_sev))
        final_sev_xgb.append(fit_xgb_sev(Xs_full, ys_full, seed=s, rounds=rounds_sev))
        final_sev_cat.append(fit_cat_sev(Xs_full, ys_full, seed=s, rounds=rounds_sev))

        final_react_lgb.append(fit_lgb(Xr_full, yr_full, LGB_PARAMS_BIN, seed=s, rounds=rounds_react))
        final_react_xgb.append(fit_xgb_bin(Xr_full, yr_full, seed=s, rounds=rounds_react))
        final_react_cat.append(fit_cat_bin(Xr_full, yr_full, seed=s, rounds=rounds_react))

    artifact = dict(
        sev_models={
            "lgb": [m.model_to_string() for m in final_sev_lgb],
            "xgb": final_sev_xgb,
            "cat": final_sev_cat,
        },
        react_models={
            "lgb": [m.model_to_string() for m in final_react_lgb],
            "xgb": final_react_xgb,
            "cat": final_react_cat,
        },
        features=pl.FEATURES, react_features=pl.REACT_FEATURES,
        severe_threshold=t_sev, severe_recall_target=SEVERE_RECALL_TARGET,
        seeds=model_seeds, folds=k, num_classes=3,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        n_rows=int(len(df)), n_sev_rows=int(len(ys_full)), n_react_rows=int(len(yr_full)),
        test_severity=m_plain, test_reactivity=rm,
        algorithms=["LightGBM", "XGBoost", "CatBoost"],
        blend_weights={"lgb": 0.35, "cat": 0.35, "xgb": 0.30})

    joblib.dump(artifact, SRC / "allergy_model.joblib")
    make_plots(cm, cal_r, final_sev_lgb, final_react_lgb)

    minutes = round((time.time() - t0) / 60, 2)
    summary = dict(cleaning=report, folds_chosen=k,
                   ensemble_blend={"lgb": 0.35, "cat": 0.35, "xgb": 0.30},
                   severe_threshold=t_sev,
                   severity_oof=oof_rule, severity_test_plain=m_plain, severity_test_rule=m_rule,
                   reactivity_test=rm, probes_passed=f"{n_pass}/{len(checks)}",
                   leave_one_source_out=None if loso is None else loso.reset_index().to_dict("records"),
                   minutes=minutes)
    (REPORTS / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))

    section("FINAL SUMMARY")
    log(f"data           {len(df):,} clean rows, {df.person_id.nunique():,} people, 5 source cohorts")
    log(f"folds          {k} folds x {len(model_seeds)} seeds = {len(model_seeds)*k} iterations (grouped by person)")
    log(f"severity head  accuracy {m_plain['accuracy']:.4f}  within-one {m_plain['within_one']:.4f}  qwk {m_plain['qwk']:.4f}  "
        f"logloss {m_plain['logloss']:.4f}  severe AUC {m_plain.get('severe_auc'):.4f}  "
        f"severe recall {m_plain.get('severe_recall'):.4f}  severe precision {m_plain.get('severe_precision'):.4f}")
    log(f"reactivity     AUC {rm['auc']:.4f}  logloss {rm['logloss']:.4f} (baseline {rm['baseline_logloss']:.4f})  "
        f"brier {rm['brier']:.4f}  ECE {rm['ece']:.4f}")
    if loso is not None:
        log(f"unseen source  mean accuracy {loso.accuracy.mean():.3f}  mean qwk {loso.qwk.mean():.3f}")
    log(f"probes         {n_pass}/{len(checks)} passed")
    log(f"saved          allergy_model.joblib, metrics.json, confusion_matrix.png, "
        f"reactivity_calibration.png, feature_importance.png, data/allergy_clean.csv")
    log(f"total time     {minutes} min")


if __name__ == "__main__":
    main()
