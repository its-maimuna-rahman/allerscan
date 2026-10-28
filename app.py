import html as _html
import json
import sys
import urllib.request
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

# Ensure src/ is on sys.path so core modules can be imported seamlessly
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st

import allergen_kb as kb
import additive_detector as ad
from food_fetcher import fetch_general_food_ingredients, parse_off
from model import model_info, predict_severity
from nlp_parser import parse_label
from photo_scanner import merge_scan_results, scan_photo
import spelling_corrector as sc
import term_resolver as tr
from multilingual_parser import detect_language, detect_script, normalize_multilingual_text
from multilingual_fusion import (
    analyze_multilingual_label,
    VerificationState,
    MultilingualAnalysisResult,
    FusedAllergenHit,
)

LANGUAGE_OPTIONS = {
    "Auto (Detect)": "auto",
    "English": "en",
    "Bengali (বাংলা)": "bn",
    "Hindi (हिन्दी)": "hi",
    "Urdu (اردو)": "ur",
    "Arabic (العربية)": "ar",
    "French (Français)": "fr",
    "German (Deutsch)": "de",
    "Spanish (Español)": "es",
    "Italian (Italiano)": "it",
    "Portuguese (Português)": "pt",
    "Turkish (Türkçe)": "tr",
    "Russian (Русский)": "ru",
    "Chinese - Simplified (简体中文)": "zh-CN",
    "Chinese - Traditional (繁體中文)": "zh-TW",
    "Japanese (日本語)": "ja",
    "Korean (한국어)": "ko",
    "Thai (ไทย)": "th",
    "Indonesian (Bahasa Indonesia)": "id",
    "Malay (Bahasa Melayu)": "ms",
}

ICON = Path(__file__).parent / "assets" / "favicon.png"
st.set_page_config(page_title="AllerScan", page_icon=str(ICON) if ICON.exists() else None,
                   layout="wide", initial_sidebar_state="expanded")

# ------------------------------------------------------------------
# Design tokens
# ------------------------------------------------------------------
INK, INK_SOFT, CLEAR, FOIL = "#1B2A41", "#4B5B73", "#EDF1F5", "#D5DBE2"
RAMP = {1: "#8FD0B1", 2: "#F2A648", 3: "#A8183A"}
ON_RAMP = {1: INK, 2: INK, 3: "#FFFFFF"}
NEUTRAL = "#B7C1CD"
SCALE = {
    1: ("Mild / Localized", "Oral itch, tingling, localized hives or redness"),
    2: ("Moderate / Systemic", "Spreading hives, angioedema, cramps, nausea"),
    3: ("Severe / Anaphylaxis", "Breathing difficulty, wheeze, tight throat, epinephrine needed"),
}
SCALE_HELP = "  \n".join(f"{k}: {v[0]}. {v[1]}." for k, v in SCALE.items())

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&display=swap');
:root {{ --ink:{INK}; --soft:{INK_SOFT}; --clear:{CLEAR}; --foil:{FOIL}; }}
html, body, .stApp, [data-testid="stAppViewContainer"] {{
  font-family:'Atkinson Hyperlegible', system-ui, sans-serif; color:var(--ink); }}
.stApp {{ background:var(--clear); }}
header[data-testid="stHeader"] {{ background:transparent; }}
#MainMenu, footer {{ visibility:hidden; }}
.block-container {{ padding-top:1.6rem; max-width:1240px; }}
[data-testid="stSidebar"] {{ background:var(--foil); }}
[data-testid="stSidebar"] .block-container {{ padding-top:1.4rem; }}
h1, h2, h3 {{ font-family:'Bricolage Grotesque', sans-serif; letter-spacing:-0.02em; color:var(--ink); }}
.side-title, .section-title {{ font-family:'Bricolage Grotesque',sans-serif; font-weight:800;
  font-size:1.45rem; letter-spacing:-0.02em; margin:0 0 .6rem 0; }}
.section-title {{ margin-top:1.2rem; }}

/* wordmark: lowercase type inside scanner corner marks, no icon */
.logo {{ position:relative; display:inline-block; padding:12px 26px 14px; }}
.logo .word {{ font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:2.7rem;
  letter-spacing:-0.045em; line-height:1; color:var(--ink); }}
.logo .word b {{ color:#C94A08; font-weight:800; }}
.logo .c {{ position:absolute; width:15px; height:15px; border:0 solid #FF7A1A; }}
.logo .tl {{ top:0; left:0; border-top-width:4px; border-left-width:4px; }}
.logo .tr {{ top:0; right:0; border-top-width:4px; border-right-width:4px; }}
.logo .bl {{ bottom:0; left:0; border-bottom-width:4px; border-left-width:4px; }}
.logo .br {{ bottom:0; right:0; border-bottom-width:4px; border-right-width:4px; }}
.lede {{ color:var(--soft); margin:.7rem 0 0 .2rem; max-width:34rem; font-size:1.05rem; line-height:1.5; }}

/* severity legend */
.scale {{ display:grid; grid-template-columns:repeat(3,1fr); background:#C4CCD6; border-radius:22px;
  padding:12px 10px 10px; box-shadow:inset 0 2px 4px rgba(27,42,65,.22); }}
.scale .cell {{ text-align:center; padding:0 4px; }}
.scale .cav {{ width:46px; height:46px; margin:0 auto 6px; }}
.scale .t {{ font-weight:700; font-size:.86rem; line-height:1.15; }}
.scale .d {{ font-size:.74rem; color:var(--soft); line-height:1.25; margin-top:2px; }}

/* blister pieces */
.cav {{ border-radius:50%; background:#E3E8EE; display:grid; place-items:center;
  box-shadow:inset 0 2px 3px rgba(27,42,65,.35); }}
.tab {{ width:76%; height:76%; border-radius:50%; background:var(--c); position:relative; display:grid;
  place-items:center; font-weight:700; font-size:.95rem;
  box-shadow:inset 0 1px 1px rgba(255,255,255,.65), 0 2px 3px rgba(27,42,65,.35); }}
.tab::after {{ content:''; position:absolute; left:14%; right:14%; top:50%; height:1.5px; background:rgba(27,42,65,.18); }}
.tab span {{ position:relative; z-index:1; }}
.strip {{ display:inline-flex; gap:6px; padding:7px 9px; border-radius:999px; background:#C4CCD6;
  box-shadow:inset 0 2px 3px rgba(27,42,65,.25); }}
.strip .cav {{ width:30px; height:30px; }}

/* verdict + cards */
.verdict {{ border-left:10px solid var(--c); background:#fff; padding:18px 22px; margin:6px 0 18px; }}
.verdict .big {{ font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:1.7rem;
  letter-spacing:-0.025em; line-height:1.15; }}
.verdict .sub {{ color:var(--soft); margin-top:6px; font-size:1.02rem; }}
.card {{ border-left:8px solid var(--c); background:rgba(255,255,255,.75); padding:16px 18px 14px; min-height:218px; }}
.card .name {{ font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:1.35rem; letter-spacing:-0.02em; }}
.card .src {{ color:var(--soft); font-size:.92rem; margin:2px 0 8px; }}
.card .est {{ margin:10px 0 2px; display:flex; align-items:center; gap:14px; flex-wrap:wrap; }}
.card .lbl {{ font-size:.86rem; color:var(--soft); }}
.card .facts {{ display:flex; gap:22px; margin-top:10px; }}
.card .fact .v {{ font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:1.6rem; line-height:1; }}
.card .fact .k {{ font-size:.8rem; color:var(--soft); margin-top:3px; }}
.card .plain {{ color:var(--soft); font-size:.95rem; line-height:1.4; margin-top:8px; }}
.tag {{ display:inline-block; font-size:.78rem; padding:2px 10px; border-radius:999px; margin:0 6px 4px 0;
  border:1.5px solid var(--ink); }}
.tag.solid {{ background:var(--ink); color:#fff; }}
.tag.danger {{ background:{RAMP[3]}; border-color:{RAMP[3]}; color:#fff; }}
.bars {{ display:flex; gap:10px; align-items:flex-end; height:84px; margin:4px 0 6px; }}
.bars .b {{ width:34px; text-align:center; font-size:.76rem; color:var(--soft); }}
.bars .b i {{ display:block; width:100%; border-radius:8px 8px 3px 3px; background:var(--c); margin-bottom:3px; }}
.drv {{ display:grid; grid-template-columns:12.5rem 1fr 3.4rem; gap:10px; align-items:center; font-size:.92rem; margin:5px 0; }}
.drv .track {{ background:#D9E0E7; height:10px; border-radius:999px; position:relative; overflow:hidden; }}
.drv .fill {{ position:absolute; top:0; bottom:0; border-radius:999px; }}
.note {{ color:var(--soft); font-size:.86rem; margin-top:26px; max-width:50rem; line-height:1.45; }}

/* additive calm notes and detail cards */
.additive-note {{
  background:#F0F4F8; border-left:4px solid #7B8CA3; padding:9px 14px; border-radius:6px;
  margin:8px 0 14px 0; font-size:0.90rem; color:#1B2A41; display:flex; align-items:center; gap:8px;
}}
.additive-note .icon {{ color:#4B5B73; font-weight:700; font-size:1.05rem; flex-shrink:0; }}
.additive-card {{
  background:#FFFFFF; border:1px solid #D5DBE2; border-radius:8px; padding:12px 16px;
  margin-bottom:10px; font-size:0.88rem; line-height:1.45;
}}
.additive-card .title {{
  font-family:'Bricolage Grotesque',sans-serif; font-weight:700; font-size:1.05rem;
  color:#1B2A41; margin-bottom:4px;
}}
.additive-card .pill {{
  display:inline-block; font-size:0.72rem; padding:1px 7px; border-radius:999px;
  background:#E6ECF2; color:#4B5B73; margin-right:6px;
}}
.additive-card .pill-user {{ background:#1B2A41; color:#FFFFFF; }}
.additive-card .disclaimer {{
  font-size:0.80rem; color:#6C7A8E; margin-top:8px; border-top:1px solid #EDF1F5;
  padding-top:6px; font-style:italic;
}}

/* streamlit parts */
[data-testid="stExpander"] {{ border:0; background:transparent; margin-bottom:10px; }}
[data-testid="stExpander"] summary {{ font-weight:700; }}
button[data-testid="stBaseButton-primary"], button[kind="primary"] {{
  background:var(--ink) !important; color:#fff !important; border:0 !important;
  border-radius:999px !important; padding:.55rem 1.6rem !important; font-weight:700 !important; }}
button[data-testid="stBaseButton-primary"]:hover {{ background:#FF7A1A !important; color:var(--ink) !important; }}
button[data-testid="stBaseButton-secondary"] {{ border-radius:999px !important; border:1.5px solid var(--ink) !important; font-weight:700 !important; }}
button:focus-visible, textarea:focus-visible, input:focus-visible {{ outline:3px solid #FF7A1A !important; outline-offset:2px; }}
textarea {{ font-family:'Atkinson Hyperlegible',sans-serif !important; font-size:1.02rem !important; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition:none !important; animation:none !important; }} }}
@media (max-width:760px) {{
  .scale .d {{ display:none; }} .scale .cav {{ width:34px; height:34px; }}
  .drv {{ grid-template-columns:8rem 1fr 3rem; }} .logo .word {{ font-size:2.1rem; }}
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def h(s: str) -> str:
    return "".join(line.strip() for line in s.splitlines())


def esc(x) -> str:
    return _html.escape(str(x))


def strip_html(level: int) -> str:
    cells = []
    for i in range(1, 4):
        if i <= level:
            cells.append(f'<div class="cav"><div class="tab" style="--c:{RAMP[i]};color:{ON_RAMP[i]}"><span>{i}</span></div></div>')
        else:
            cells.append('<div class="cav"></div>')
    return f'<div class="strip" role="img" aria-label="Severity {level} of 3">{"".join(cells)}</div>'


def scale_html() -> str:
    cells = "".join(
        f'<div class="cell"><div class="cav"><div class="tab" style="--c:{RAMP[i]};color:{ON_RAMP[i]}"><span>{i}</span></div></div>'
        f'<div class="t">{SCALE[i][0]}</div><div class="d">{SCALE[i][1]}</div></div>' for i in range(1, 4))
    return h(f'<div class="scale">{cells}</div>')


# ---- product search (Open Food Facts, free, no key) ---------------------------------
OFF_SEARCH_ENDPOINTS = [
    "https://world.openfoodfacts.net/cgi/search.pl",
    "https://world.openfoodfacts.org/cgi/search.pl",
    "https://us.openfoodfacts.net/cgi/search.pl",
]
OFF_PRODUCT_ENDPOINTS = [
    "https://world.openfoodfacts.net/api/v0/product/{code}.json",
    "https://world.openfoodfacts.org/api/v0/product/{code}.json",
]


# parse_off is provided by food_fetcher (Step 9.5)


@st.cache_data(ttl=3600, show_spinner=False)
def search_products(query: str) -> list:
    q = query.strip()
    headers = {"User-Agent": "AllerScan/1.0 (Student Project; Contact: contact@allerscan.local)"}

    # Direct barcode lookup if query is numeric
    if q.isdigit() and len(q) in (8, 12, 13, 14):
        for ep in OFF_PRODUCT_ENDPOINTS:
            try:
                req = urllib.request.Request(ep.format(code=q), headers=headers)
                with urllib.request.urlopen(req, timeout=6) as resp:
                    d = json.load(resp)
                    if d.get("status") == 1 and d.get("product"):
                        return parse_off({"products": [d["product"]]})
            except Exception:
                continue

    params = {"action": "process", "search_terms": q, "search_simple": 1, "json": 1,
              "sort_by": "unique_scans_n", "page_size": 30,
              "fields": "code,product_name,product_name_en,brands,ingredients_text,ingredients_text_en,traces,unique_scans_n,additives_tags"}
    last_err = None
    for ep in OFF_SEARCH_ENDPOINTS:
        try:
            req = urllib.request.Request(f"{ep}?{urlencode(params)}", headers=headers)
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.load(resp)
                if data and "products" in data:
                    parsed = parse_off(data)
                    return parsed
        except Exception as exc:
            last_err = exc
            continue

    if last_err is not None:
        raise last_err
    return []


# ------------------------------------------------------------------
# header
# ------------------------------------------------------------------
head_l, head_r = st.columns([1, 1.45], gap="large")
with head_l:
    st.markdown(h("""
        <div class="logo"><span class="c tl"></span><span class="c tr"></span><span class="c bl"></span>
        <span class="c br"></span><div class="word">aller<b>scan</b></div></div>
        <p class="lede">Search a product or paste its ingredients. AllerScan finds allergens, including
        hidden ones, and estimates how severe a reaction could be for you.</p>"""), unsafe_allow_html=True)
with head_r:
    st.markdown(scale_html(), unsafe_allow_html=True)

# ------------------------------------------------------------------
# sidebar: profile
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="side-title">Your profile</div>', unsafe_allow_html=True)
    age = st.number_input("Age", min_value=0, max_value=110, value=25, step=1)
    sex_choice = st.radio("Sex", ["Female", "Male", "Not specified"], index=2, horizontal=True)
    asthma_choice = st.selectbox(
        "Asthma",
        ["No", "Mild", "Moderate", "Severe", "Not sure"],
        index=0,
        help=(
            "Asthma severity tiers:\n"
            "• No: No asthma history or breathing symptoms.\n"
            "• Mild: Occasional symptoms or reliever inhaler (e.g. albuterol) only as needed.\n"
            "• Moderate: Regular daily controller medication or frequent symptoms.\n"
            "• Severe: High-dose daily controller medicine or past hospital/ER visits.\n"
            "• Not sure: Unspecified or uncertain.\n\n"
            "Note: Co-factor estimation only; this is not a clinical diagnosis."
        )
    )
    ASTHMA_LEVELS = {"No": 0.0, "Mild": 1.0, "Moderate": 2.0, "Severe": 3.0, "Not sure": None}
    asthma_val = ASTHMA_LEVELS[asthma_choice]
    eczema = st.toggle("Eczema")
    family = st.toggle("Allergies in the family")
    st.divider()
    st.session_state.setdefault("show_additive_notes", True)
    st.session_state.setdefault("show_weak_additives", False)
    show_additives = st.toggle("Show additive notes", value=st.session_state["show_additive_notes"], key="toggle_show_additives")
    st.session_state["show_additive_notes"] = show_additives
    if show_additives:
        show_weak = st.toggle(
            "Also show additives with weak evidence",
            value=st.session_state["show_weak_additives"],
            key="toggle_show_weak",
            help="Includes additives like MSG, aspartame, BHA/BHT, and azo dyes whose reported reactions have not been confirmed in blinded clinical trials."
        )
        st.session_state["show_weak_additives"] = show_weak
    else:
        st.session_state["show_weak_additives"] = False
    st.divider()
    st.session_state.setdefault("check_spelling", True)
    check_spelling = st.toggle(
        "Check spelling",
        value=st.session_state["check_spelling"],
        key="toggle_check_spelling",
        help="Detects and fixes spelling mistakes (e.g. 'mlik' -> 'milk', 'peenut' -> 'peanut') and flags unclear spellings as possible allergens."
    )
    st.session_state["check_spelling"] = check_spelling
    st.divider()
    chosen = st.multiselect("Known allergens", kb.all_allergens(), format_func=kb.pretty)

    # Session state for typed allergen input
    st.session_state.setdefault("other_matched", {})      # key -> {"name": str, "rating": int}
    st.session_state.setdefault("other_groups", {})       # group_id -> {"display_name": str, "all_members": list, "members": list, "rating": int, "message": str}
    st.session_state.setdefault("other_custom", {})       # custom_key -> {"key": str, "display_name": str, "clean_term": str, "rating": int}
    st.session_state.setdefault("other_additives", {})    # additive_id -> {"id": str, "name": str, "clean_term": str}
    st.session_state.setdefault("other_pending", [])      # list of dicts for suggestions / clarifications
    st.session_state.setdefault("other_messages", [])     # list of {"type": "info"|"warning", "text": str}

    with st.form("other_allergens_form", clear_on_submit=True):
        other_input = st.text_input(
            "Other allergens, separated by commas",
            placeholder="e.g. tree nuts, mushroom, fava bean",
            help=(
                "What can be typed:\n"
                "• Specific foods: e.g. peanuts, sesame, cow's milk\n"
                "• Group terms: e.g. tree nuts (expands to 10 nuts), shellfish (15 crustaceans & molluscs), seafood, fish, legumes, gluten\n"
                "• Clarifications: typing 'nuts' asks if you mean peanuts, tree nuts, or both\n"
                "• Misspellings: conservative suggestions (e.g. 'penut' -> Did you mean Peanut?)\n"
                "• Custom allergens: foods outside the model (e.g. jackfruit, mushroom) are kept as custom allergens with your own rating slider\n"
                "• Limitation: Custom allergens are matched on labels, but have no model estimate for related foods.\n"
                "Press Enter or click Add to commit."
            )
        )
        st.caption("Press Enter or Add")
        add_clicked = st.form_submit_button("Add")

    if add_clicked and other_input.strip():
        existing_model = set(chosen) | set(st.session_state["other_matched"].keys())
        for g_data in st.session_state["other_groups"].values():
            existing_model.update(g_data["members"])
        existing_custom = set(st.session_state["other_custom"].keys())
        existing_additives = set(st.session_state["other_additives"].keys())

        resolved = tr.resolve_terms(
            other_input,
            existing_model_keys=existing_model,
            existing_custom_keys=existing_custom,
            current_custom_count=len(st.session_state["other_custom"]),
            max_custom=15,
            existing_additive_keys=existing_additives,
        )

        st.session_state["other_messages"] = []
        for item in resolved:
            if item.status == "matched":
                for k in item.matched_keys:
                    st.session_state["other_matched"][k] = {"name": kb.pretty(k), "rating": 2}
            elif item.status == "group":
                g_key = item.group_name or item.display_name
                st.session_state["other_groups"][g_key] = {
                    "display_name": item.display_name,
                    "all_members": list(item.matched_keys),
                    "members": list(item.matched_keys),
                    "rating": 2,
                    "message": item.message,
                }
            elif item.status in ("suggestions", "ask_which", "over_broad"):
                st.session_state["other_pending"].append(item.to_dict())
            elif item.status == "additive":
                a_id = item.matched_keys[0]
                st.session_state["other_additives"][a_id] = {
                    "id": a_id,
                    "name": item.display_name,
                    "clean_term": item.clean_term,
                }
                st.session_state["other_messages"].append({
                    "type": "info",
                    "text": f"Added '{item.display_name}' as an additive sensitivity. (Additives are monitored separately from allergen severity estimates)."
                })
            elif item.status == "custom":
                st.session_state["other_custom"][item.custom_key] = {
                    "key": item.custom_key,
                    "display_name": item.display_name,
                    "clean_term": item.clean_term,
                    "rating": 2,
                }
            elif item.status == "already_added":
                st.session_state["other_messages"].append({"type": "info", "text": item.message})
            elif item.status == "rejected":
                st.session_state["other_messages"].append({"type": "warning", "text": f"'{item.original}': {item.message}"})
        st.rerun()

    # Show messages
    for msg in st.session_state["other_messages"]:
        if msg["type"] == "warning":
            st.warning(msg["text"])
        else:
            st.info(msg["text"])

    # Show pending items (Tier 3 suggestions & Tier 2 clarifications)
    if st.session_state["other_pending"]:
        pending_list = list(st.session_state["other_pending"])
        for idx, p in enumerate(pending_list):
            if p["status"] == "suggestions":
                st.markdown(f"**Suggestion for '{esc(p.get('clean_term') or p.get('original'))}':**")
                for sug in p.get("suggestions", []):
                    if st.button(f"Did you mean {sug['name']}?", key=f"sug_acc_{idx}_{sug['key']}"):
                        st.session_state["other_matched"][sug["key"]] = {"name": sug["name"], "rating": 2}
                        st.session_state["other_pending"].pop(idx)
                        st.rerun()
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Keep as custom", key=f"sug_custom_{idx}"):
                        ck = p["custom_key"]
                        st.session_state["other_custom"][ck] = {
                            "key": ck, "display_name": p["display_name"],
                            "clean_term": p["clean_term"], "rating": 2
                        }
                        st.session_state["other_pending"].pop(idx)
                        st.rerun()
                with col2:
                    if st.button("Dismiss", key=f"sug_dismiss_{idx}"):
                        st.session_state["other_pending"].pop(idx)
                        st.rerun()

            elif p["status"] == "ask_which":
                st.markdown(f"**{esc(p['message'])}**")
                for opt_idx, opt in enumerate(p.get("options", [])):
                    if st.button(opt["label"], key=f"ask_{idx}_{opt_idx}"):
                        if "keys" in opt:
                            for k in opt["keys"]:
                                st.session_state["other_matched"][k] = {"name": kb.pretty(k), "rating": 2}
                        if "group" in opt:
                            g_name = opt["group"]
                            g_members = list(kb.GROUPS.get(g_name, []))
                            disp_name = "Tree nuts" if g_name == "tree_nut" else kb.pretty(g_name)
                            st.session_state["other_groups"][disp_name] = {
                                "display_name": disp_name,
                                "all_members": g_members,
                                "members": g_members,
                                "rating": 2,
                                "message": f"{disp_name}: {len(g_members)} foods added."
                            }
                        st.session_state["other_pending"].pop(idx)
                        st.rerun()
                if st.button("Dismiss", key=f"ask_dismiss_{idx}"):
                    st.session_state["other_pending"].pop(idx)
                    st.rerun()

            elif p["status"] == "over_broad":
                st.info(p["message"])
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Keep as custom '{p['display_name']}'", key=f"ob_custom_{idx}"):
                        ck = p["custom_key"]
                        st.session_state["other_custom"][ck] = {
                            "key": ck, "display_name": p["display_name"],
                            "clean_term": p["clean_term"], "rating": 2
                        }
                        st.session_state["other_pending"].pop(idx)
                        st.rerun()
                with col2:
                    if st.button("Dismiss", key=f"ob_dismiss_{idx}"):
                        st.session_state["other_pending"].pop(idx)
                        st.rerun()

    # Groups display (Step 5.3: shared rating slider + editable member list)
    for g_key in list(st.session_state["other_groups"].keys()):
        g_info = st.session_state["other_groups"][g_key]
        col_g1, col_g2 = st.columns([4, 1])
        with col_g1:
            st.markdown(f"**Group: {esc(g_info['display_name'])}** ({len(g_info['members'])} foods)")
        with col_g2:
            if st.button("✕", key=f"rm_group_{g_key}", help=f"Remove {g_info['display_name']}"):
                del st.session_state["other_groups"][g_key]
                st.session_state.pop(f"sev_group_{g_key}", None)
                st.session_state.pop(f"members_group_{g_key}", None)
                st.rerun()

        # Shared rating slider
        g_rating = st.slider(
            f"{g_info['display_name']} severity",
            1, 3, g_info.get("rating", 2),
            key=f"sev_group_{g_key}",
            help=SCALE_HELP
        )
        g_info["rating"] = g_rating

        # Editable member list
        active_members = st.multiselect(
            f"Active {g_info['display_name']} members",
            options=g_info["all_members"],
            default=g_info["members"],
            format_func=kb.pretty,
            key=f"members_group_{g_key}"
        )
        g_info["members"] = active_members
        if g_info.get("message"):
            st.caption(g_info["message"])

    # Typed matched chips (Step 5.3: chip with food name and slider)
    for m_key in list(st.session_state["other_matched"].keys()):
        m_info = st.session_state["other_matched"][m_key]
        col_m1, col_m2 = st.columns([4, 1])
        with col_m1:
            st.markdown(f"**{esc(m_info['name'])}**")
        with col_m2:
            if st.button("✕", key=f"rm_matched_{m_key}", help=f"Remove {m_info['name']}"):
                del st.session_state["other_matched"][m_key]
                st.session_state.pop(f"sev_{m_key}", None)
                st.rerun()
        m_rating = st.slider(
            m_info["name"],
            1, 3, m_info.get("rating", 2),
            key=f"sev_{m_key}",
            help=SCALE_HELP
        )
        m_info["rating"] = m_rating

    # Custom allergen chips (Step 5.3: chip marked 'custom, not in the model' with a slider)
    for c_key in list(st.session_state["other_custom"].keys()):
        c_info = st.session_state["other_custom"][c_key]
        col_c1, col_c2 = st.columns([4, 1])
        with col_c1:
            st.markdown(f"**{esc(c_info['display_name'])}** *(custom, not in the model)*")
        with col_c2:
            if st.button("✕", key=f"rm_custom_{c_key}", help=f"Remove {c_info['display_name']}"):
                del st.session_state["other_custom"][c_key]
                st.session_state.pop(f"sev_{c_key}", None)
                st.rerun()
        c_rating = st.slider(
            f"{c_info['display_name']} (custom)",
            1, 3, c_info.get("rating", 2),
            key=f"sev_{c_key}",
            help=SCALE_HELP
        )
        c_info["rating"] = c_rating

    # Additive sensitivity chips (monitored calmly, no model rating slider)
    for a_id in list(st.session_state["other_additives"].keys()):
        a_info = st.session_state["other_additives"][a_id]
        col_a1, col_a2 = st.columns([4, 1])
        with col_a1:
            st.markdown(f"**{esc(a_info['name'])}** *(additive sensitivity)*")
        with col_a2:
            if st.button("✕", key=f"rm_additive_{a_id}", help=f"Remove {a_info['name']}"):
                del st.session_state["other_additives"][a_id]
                st.rerun()

    # Multiselect chosen allergens ratings (if not already handled in matched or groups)
    profile = {}
    already_slid = set(st.session_state["other_matched"].keys())
    for g_info in st.session_state["other_groups"].values():
        already_slid.update(g_info["all_members"])

    standalone_chosen = [k for k in chosen if k not in already_slid]
    if standalone_chosen:
        st.markdown("**How strong are your usual reactions?**")
        st.caption("Rate each from 1 to 3 (1=Mild, 2=Moderate, 3=Severe). The scale is at the top of the page. "
                   "Your ratings shape the estimates, including for related foods.")
        for key in sorted(standalone_chosen):
            profile[key] = st.slider(kb.pretty(key), 1, 3, 2, key=f"sev_{key}", help=SCALE_HELP)

    # Populate final model profile (model foods only!)
    for k in chosen:
        if k in standalone_chosen:
            pass
        elif k in st.session_state["other_matched"]:
            profile[k] = st.session_state["other_matched"][k]["rating"]

    for k, m_data in st.session_state["other_matched"].items():
        profile[k] = m_data["rating"]

    for g_key, g_info in st.session_state["other_groups"].items():
        for m in g_info["members"]:
            profile[m] = g_info["rating"]

    custom_allergens = dict(st.session_state["other_custom"])

SEX = {"Female": 0, "Male": 1, "Not specified": None}[sex_choice]

if "error" in model_info():
    st.error("The model has not been trained yet. Run `python train_model.py` in this folder, then reload.")
    st.stop()


# ------------------------------------------------------------------
# input sources: General Food / Product Search / Photo Scan
#
# Every source is its own workspace. It owns its ingredient text, its list of
# matching products, its additive tags, its "checked" flag, the user's decisions
# on "possible allergen" prompts, and its own result. Switching tabs therefore
# never overwrites or erases what another source is holding.
# ------------------------------------------------------------------
SRC_FOOD, SRC_SEARCH, SRC_SCAN = "food", "search", "scan"
SOURCES = (SRC_FOOD, SRC_SEARCH, SRC_SCAN)


def _text_key(src: str) -> str:
    return f"label_text_{src}"


def _state(src: str) -> dict:
    store = st.session_state.setdefault("source_state", {})
    return store.setdefault(src, {
        "hits": [], "tags": [], "checked": False,
        "confirmed": set(), "ignored": set(),
        "last_q": "", "error": None, "note": None,
        "scan_res": None, "multi_memo": None,
    })


for _src in SOURCES:
    _state(_src)
    st.session_state.setdefault(_text_key(_src), "")


def _load_text(src: str, text: str, tags=None):
    """Put freshly fetched ingredients into ONE source and reset only that source's per-text state."""
    S = _state(src)
    st.session_state[_text_key(src)] = text
    S["tags"] = list(tags or [])
    S["checked"] = True
    S["confirmed"], S["ignored"], S["multi_memo"] = set(), set(), None


# ---- small inline SVG icons (Material Design paths) for HTML banners ---------------
_ICON_PATHS = {
    "warning": "M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z",
    "info": "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z",
    "check_circle": "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z",
    "spellcheck": "M12.45 16h2.09L9.43 3H7.57L2.46 16h2.09l1.12-3h5.64l1.14 3zm-6.02-5L8.5 5.48 10.57 11H6.43zm15.16.59l-8.09 8.09L9.83 16l-1.41 1.41 5.09 5.09L23 13l-1.41-1.41z",
}


def svg_icon(name: str, size: int = 18) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="{size}" height="{size}" '
            f'fill="currentColor" aria-hidden="true" style="vertical-align:-3px;margin-right:6px">'
            f'<path d="{_ICON_PATHS[name]}"/></svg>')



# ------------------------------------------------------------------
# results
# ------------------------------------------------------------------
def tags_html(r):
    t, info, mode = [], r["info"], r["pred"]["mode"]
    kind = {"direct": "Named in the ingredients", "derived": "Possible hidden source",
            "trace": "Precautionary label"}.get(info.get("kind", "direct"), "Ingredient")
    prov = r.get("provenance") or info.get("provenance")
    if prov == "both":
        t.append('<span class="tag solid" style="background:#2ECC71;color:#fff;">✓ Dict & Translation</span>')
    elif prov == "glossary":
        t.append('<span class="tag solid" style="background:#27AE60;color:#fff;">Dictionary match</span>')
    elif prov == "translation":
        t.append('<span class="tag warn" style="background:#F39C12;color:#fff;">Translation only</span>')
    elif prov == "possible":
        t.append('<span class="tag warn">Possible match</span>')
    if mode == "profile":
        t.append('<span class="tag solid">In your profile</span>')
    elif mode == "cross":
        t.append('<span class="tag solid">Possible cross-reaction</span>')
    elif mode == "unrelated":
        t.append('<span class="tag">Not linked to your profile</span>')
    else:
        t.append('<span class="tag">General estimate</span>')
    t.append(f'<span class="tag">{kind}</span>')
    if r["pred"]["severe_flag"] and mode != "unrelated":
        t.append('<span class="tag danger">Severe reaction possible</span>' if mode != "cross"
                 else '<span class="tag danger">Could be severe if you react</span>')
    return "".join(t)


def card_html(r):
    p, info, mode = r["pred"], r["info"], r["pred"]["mode"]
    lvl = p["severity_score"]
    src = ", ".join(esc(s) for s in info["sources"][:3])
    mh = r.get("multilingual_hit") or info.get("multilingual_hit")
    if mh:
        if hasattr(mh, "original_matched_term") and mh.original_matched_term:
            if hasattr(mh, "translated_matched_term") and mh.translated_matched_term:
                src = f"{esc(mh.original_matched_term)} (translated: {esc(mh.translated_matched_term)})"
            else:
                src = f"{esc(mh.original_matched_term)} ({getattr(mh, 'source_language', 'local')})"
    color = NEUTRAL if mode == "unrelated" else RAMP.get(lvl, RAMP[3])
    rel = p["related"]
    rel_name = kb.pretty(rel["name"]) if rel else ""
    head = (f'<div class="name">{esc(kb.pretty(r["allergen"]))}</div><div class="src">Found in: {src}</div>'
            f'{tags_html(r)}')
    if mode == "unrelated":
        body = (f'<div class="plain">No meaningful link to the allergies you listed. '
                f'Chance you react: about {round(100 * p["p_react"])}%.</div>')
    else:
        label = {"profile": f"Model estimate, level {lvl}", "cross": f"If you react: level {lvl}",
                 "general": f"If you were allergic: level {lvl}"}[mode]
        facts = ""
        if mode == "profile" and p["rating"] is not None:
            facts = f'<div class="fact"><div class="v">{int(p["rating"])}</div><div class="k">You rated</div></div>'
        elif mode == "cross":
            facts = (f'<div class="fact"><div class="v">{round(100 * p["p_react"])}%</div>'
                     f'<div class="k">chance you also react</div></div>')
            if rel and rel["rating"] is not None:
                facts += (f'<div class="fact"><div class="v">{int(rel["rating"])}</div>'
                          f'<div class="k">your {esc(rel_name)} rating</div></div>')
        body = f'<div class="est">{strip_html(lvl)}<div class="lbl">{label}</div></div><div class="facts">{facts}</div>'
    return h(f'<div class="card" style="--c:{color}">{head}{body}</div>')


def detail_html(r):
    p = r["pred"]
    bars = "".join(f'<div class="b" style="--c:{RAMP.get(i + 1, INK)}"><i style="height:{max(4, round(pr * 70))}px"></i>'
                   f'{round(pr * 100)}%<br>L{i + 1}</div>' for i, pr in enumerate(p["probabilities"]))
    mx = max([abs(d["impact"]) for d in p["drivers"]] + [0.01])
    drv = ""
    for d in p["drivers"]:
        w = round(abs(d["impact"]) / mx * 50)
        color = RAMP.get(3, "#A8183A") if d["impact"] > 0 else RAMP.get(1, "#8FD0B1")
        drv += (f'<div class="drv"><div>{esc(d["label"])}</div><div class="track"><div class="fill" '
                f'style="left:{50 if d["impact"] > 0 else 50 - w}%;width:{w}%;background:{color}"></div></div>'
                f'<div>{"raises" if d["impact"] > 0 else "lowers"}</div></div>')
    return h(f'<div><div style="font-weight:700;margin-bottom:2px">Chance of each level</div><div class="bars">{bars}</div>'
             f'<div style="font-weight:700;margin:14px 0 4px">What moves the estimate for level {p["severity_score"]}</div>{drv}</div>')


def custom_card_html(hit):
    lvl = hit["rating"]
    color = RAMP.get(lvl, RAMP[3])
    src = esc(hit["matched_phrase"])
    possible_tag = '<span class="tag warn">Possible match, check label</span>' if hit.get("is_possible") else ""
    head = (f'<div class="name">{esc(hit["name"])}</div>'
            f'<div class="src">Found in: {src}</div>'
            f'<div class="tags"><span class="tag profile">Your custom allergen</span>{possible_tag}</div>')
    facts = f'<div class="fact"><div class="v">{int(lvl)}</div><div class="k">You rated</div></div>'
    body = (f'<div class="est">{strip_html(lvl)}<div class="lbl">Your rating, level {lvl}</div></div>'
            f'<div class="facts">{facts}</div>'
            f'<div class="plain" style="margin-top:8px;font-size:12px;color:#555;">'
            f'Not covered by the AllerScan model, so there is no estimate for related foods.</div>')
    return h(f'<div class="card" style="--c:{color}">{head}{body}</div>')


def verdict_html(rows, custom_hits=None):
    if custom_hits is None:
        custom_hits = []
    mine = [r for r in rows if r["pred"]["mode"] == "profile"]
    cross = [r for r in rows if r["pred"]["mode"] == "cross"]
    general = [r for r in rows if r["pred"]["mode"] == "general"]
    names = lambda rs: esc(", ".join(kb.pretty(r["allergen"]) for r in rs))

    total_mine = len(mine) + len(custom_hits)
    if total_mine > 0:
        c_lvls = [c["rating"] for c in custom_hits]
        m_lvls = [r["pred"]["severity_score"] for r in mine]
        lvl = max(m_lvls + c_lvls)

        all_mine_labels = [kb.pretty(r["allergen"]) for r in mine] + [c["name"] for c in custom_hits]
        big = f"Contains {total_mine} of your {'allergen' if total_mine == 1 else 'allergens'}"
        sub = f"{esc(', '.join(all_mine_labels))}. Highest estimated reaction: level {lvl}, {SCALE[lvl][0].lower()}."
        if cross:
            sub += f" Possible cross-reaction: {names(cross)}."
        color = RAMP.get(lvl, RAMP[3])
    elif cross:
        top = max(cross, key=lambda r: r["pred"]["p_react"])
        big = "None of your allergens, but a possible cross-reaction"
        sub = (f"{names(cross)} is related to something you listed. Highest chance you react: "
               f"{round(100 * top['pred']['p_react'])}%.")
        color = RAMP.get(max(r["pred"]["severity_score"] for r in cross), RAMP[3])
    elif general:
        big = f"{len(general)} common {'allergen' if len(general) == 1 else 'allergens'} detected"
        sub = "You have not listed any allergens, so these are general estimates. Add yours in the sidebar for a personal result."
        color = NEUTRAL
    else:
        big = "None of your allergens found"
        sub = "The allergens detected have no meaningful link to the ones you listed."
        color = "#8FD0B1"
    return h(f'<div class="verdict" style="--c:{color}"><div class="big">{big}</div><div class="sub">{sub}</div></div>')



def render_results(src: str, text: str, lang_code: str = "auto"):
    """Result block for ONE source. Reads only that source's state."""
    S = _state(src)
    if not (S["checked"] and text.strip()):
        return
    st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)

    selected_lang_code = lang_code
    script = detect_script(text)
    detected_iso, detected_conf = detect_language(text)

    is_foreign = (
        (script != "latin" and script != "unknown")
        or (selected_lang_code not in ("auto", "en"))
        or (detected_iso not in ("en", "unknown") and detected_conf >= 0.70)
    )

    multi_res: Optional[MultilingualAnalysisResult] = None
    parse_input_text = text
    spelling_res = None
    check_spelling = st.session_state.get("check_spelling", True)

    if is_foreign:
        # Step 8: Multi-route multilingual analysis
        user_add_sens = set(st.session_state.get("other_additives", {}).keys())
        user_has_asthma = (asthma_choice in ("Mild", "Moderate", "Severe"))
        # Memoised per source so a hidden tab does not re-run the (slow) analysis on every rerun.
        _memo_key = (text, selected_lang_code, frozenset(user_add_sens), user_has_asthma)
        _memo = S.get("multi_memo")
        if _memo and _memo[0] == _memo_key:
            multi_res = _memo[1]
        else:
            multi_res = analyze_multilingual_label(
                text,
                selected_language=selected_lang_code if selected_lang_code != "auto" else None,
                user_add_sens=user_add_sens,
                user_has_asthma=user_has_asthma,
            )
            S["multi_memo"] = (_memo_key, multi_res)

        lang_display_names = {v: k for k, v in LANGUAGE_OPTIONS.items()}
        det_name = lang_display_names.get(multi_res.detected_language, multi_res.detected_language.upper())

        # Render Verification State Banner (Step 8.5)
        if multi_res.verification_state == VerificationState.COULD_NOT_VERIFY:
            st.markdown(
                f'<div style="background-color: #FDEDEC; border-left: 5px solid #E74C3C; color: #78281F; '
                f'padding: 14px 18px; border-radius: 8px; margin: 12px 0;">'
                f'<h4 style="margin:0 0 6px 0; color:#922B21;">{svg_icon("warning", 20)}Could Not Verify Label Ingredients</h4>'
                f'<p style="margin:0 0 6px 0;"><strong>Reason:</strong> {esc(multi_res.state_reason)}</p>'
                f'<p style="margin:0; font-size:13px;"><strong>AllerScan Safety Guarantee:</strong> Recognition or translation '
                f'confidence is insufficient to verify safety. <em>No safe verdict is permitted.</em> Please retake a clear photo, '
                f'select the label language manually from the dropdown, or enter ingredients manually.</p></div>',
                unsafe_allow_html=True
            )
        elif multi_res.verification_state == VerificationState.PARTLY_VERIFIED:
            st.markdown(
                f'<div style="background-color: #FEF9E7; border-left: 5px solid #F39C12; color: #7D6608; '
                f'padding: 12px 16px; border-radius: 8px; margin: 12px 0;">'
                f'<strong>{svg_icon("info")}Partially Verified:</strong> {esc(multi_res.state_reason)} Some ingredients could not be verified across '
                f'both routes. Please compare with the physical packaging.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="background-color: #EAFAF1; border-left: 5px solid #2ECC71; color: #145A32; '
                f'padding: 10px 14px; border-radius: 6px; margin: 10px 0; font-size: 13px;">'
                f'{svg_icon("check_circle")}<strong>Verified Multilingual Detection:</strong> Identified as <strong>{esc(det_name)}</strong> '
                f'({int(multi_res.language_confidence * 100)}% confidence). Cross-referenced via Route A (Multilingual Glossary) '
                f'and Route B ({esc(multi_res.translation_engine)}).</div>',
                unsafe_allow_html=True
            )

        # Side-by-side Layout (Step 9.2 & 9.3)
        c_orig, c_trans = st.columns(2, gap="medium")
        with c_orig:
            st.markdown(f"**Original Text ({esc(det_name)}):**")
            rtl_style = ' dir="rtl" style="text-align: right;"' if script == "arabic" else ""
            st.markdown(
                f'<div{rtl_style} style="background:#F8F9FA; border:1px solid #D5DBE2; border-radius:6px; '
                f'padding:10px 14px; min-height:80px; font-size:14px; font-family: sans-serif;">{esc(multi_res.original_text)}</div>',
                unsafe_allow_html=True
            )
        with c_trans:
            st.markdown(f"**English Translation ({esc(multi_res.translation_engine)}):**")
            st.markdown(
                f'<div style="background:#F8F9FA; border:1px solid #D5DBE2; border-radius:6px; '
                f'padding:10px 14px; min-height:80px; font-size:14px; font-family: sans-serif;">{esc(multi_res.translated_text or "No translation available.")}</div>',
                unsafe_allow_html=True
            )

        st.caption(":material/warning: **Notice:** Translated labels can contain mistakes. Always compare with the physical pack.")

        parse_input_text = multi_res.translated_text or ""
        found = {}
        for can_k, hit in multi_res.allergens.items():
            src_desc = hit.original_matched_term or hit.display_name
            if hit.translated_matched_term and hit.translated_matched_term != hit.original_matched_term:
                src_desc += f" (translated: {hit.translated_matched_term})"
            found[can_k] = {
                "sources": [src_desc],
                "confidence": hit.confidence,
                "kind": hit.kind,
                "hidden": False,
                "provenance": hit.provenance,
                "multilingual_hit": hit,
            }

        custom_hits = tr.match_custom_allergens(custom_allergens, parse_input_text)
        detected_additives = multi_res.additives

    else:
        # Standard English Pipeline
        if check_spelling:
            spelling_res = sc.correct_text(text, mode="typed")
            parse_input_text = spelling_res.corrected_text

            if spelling_res.corrections:
                fixes_summary = ", ".join(
                    f"**{esc(c.original)}** &rarr; **{esc(c.corrected)}**"
                    for c in spelling_res.corrections[:5]
                )
                if len(spelling_res.corrections) > 5:
                    fixes_summary += f" *(+{len(spelling_res.corrections) - 5} more)*"

                st.markdown(
                    f'<div style="background-color: #E8F4FD; border-left: 4px solid #2B78C5; color: #1B2A41; '
                    f'padding: 10px 14px; border-radius: 6px; margin-bottom: 12px; font-size: 14px;">'
                    f'{svg_icon("spellcheck")}<strong>Spelling fixed:</strong> {fixes_summary}</div>',
                    unsafe_allow_html=True
                )

                with st.expander(":material/spellcheck: View all spelling corrections & diagnostics"):
                    st.markdown("**Applied Corrections:**")
                    for c in spelling_res.corrections:
                        st.write(f"- `{c.original}` &rarr; **`{c.corrected}`** ({c.kind}, confidence {int(c.confidence*100)}%)")
                    if spelling_res.protected_tokens:
                        st.markdown("**Protected Tokens (never altered into allergens):**")
                        st.write(", ".join(f"`{t}`" for t in sorted(set(spelling_res.protected_tokens))))

        found = parse_label(parse_input_text)
        custom_hits = tr.match_custom_allergens(custom_allergens, parse_input_text)

        user_add_sens = set(st.session_state.get("other_additives", {}).keys())
        user_has_asthma = (asthma_choice in ("Mild", "Moderate", "Severe"))
        current_tags = st.session_state.get("current_additive_tags") or []
        detected_additives = ad.detect_additives(
            text,
            structured_tags=current_tags,
            user_sensitivities=user_add_sens,
            user_has_asthma=user_has_asthma,
        )
        if check_spelling and parse_input_text != text:
            additives_from_corrected = ad.detect_additives(
                parse_input_text,
                structured_tags=current_tags,
                user_sensitivities=user_add_sens,
                user_has_asthma=user_has_asthma,
            )
            existing_ids = {m.id for m in detected_additives}
            for m in additives_from_corrected:
                if m.id not in existing_ids:
                    detected_additives.append(m)
                    existing_ids.add(m.id)

    # Session state for user decisions on possible items
    confirmed, ignored = S["confirmed"], S["ignored"]

    # Add any user-confirmed allergens into found
    for can_k in confirmed:
        if can_k not in found:
            found[can_k] = {
                "sources": ["User-confirmed spelling"],
                "confidence": "high",
                "kind": "direct",
                "hidden": False
            }

    # Drop any user-ignored allergens from found
    for can_k in ignored:
        found.pop(can_k, None)

    # Render possible allergens (Step 8.8) with Yes / No buttons
    unhandled_possibles = []
    if spelling_res and spelling_res.possible_allergens:
        for p in spelling_res.possible_allergens:
            if (p.canonical_key not in confirmed
                and p.canonical_key not in ignored
                and p.canonical_key not in found):
                unhandled_possibles.append(p)

    if unhandled_possibles:
        st.markdown(
            '<div style="background-color: #FFF9E6; border-left: 4px solid #E0A800; color: #1B2A41; '
            'padding: 12px 16px; border-radius: 6px; margin: 12px 0; font-size: 14px;">'
            f'{svg_icon("warning")}<strong>Possible allergen (spelling unclear):</strong> The text contains word(s) that may be allergens. '
            'Please verify each below (never confirmed without your explicit choice):</div>',
            unsafe_allow_html=True
        )
        for idx, p in enumerate(unhandled_possibles):
            c_lbl, c_y, c_no = st.columns([4, 1.2, 1])
            with c_lbl:
                st.markdown(f"Did you mean **{kb.pretty(p.canonical_key)}** for `{esc(p.original_token)}`? *(spelling unclear, {int(p.confidence*100)}% match)*")
            with c_y:
                if st.button(f"Yes, it is {kb.pretty(p.canonical_key)}", key=f"btn_yes_poss_{src}_{idx}_{p.canonical_key}"):
                    confirmed.add(p.canonical_key)
                    st.rerun()
            with c_no:
                if st.button("No, ignore", key=f"btn_no_poss_{src}_{idx}_{p.canonical_key}"):
                    ignored.add(p.canonical_key)
                    st.rerun()

    show_notes = st.session_state.get("show_additive_notes", True)
    show_weak = st.session_state.get("show_weak_additives", False)
    displayable_additives = [
        m for m in detected_additives
        if (m.tier < 3 or show_weak or m.is_user_listed)
    ] if show_notes else []

    if not found and not custom_hits:
        if is_foreign and multi_res and not multi_res.safe_verdict_allowed:
            st.markdown(h(f'<div class="verdict" style="--c:#B7C1CD"><div class="big">Could Not Verify Label Ingredients</div>'
                          f'<div class="sub">Image or text recognition was too low or language could not be confirmed. Check physical packaging.</div></div>'), unsafe_allow_html=True)
        elif displayable_additives:
            sub_msg = "No common allergens detected. See the additive note below."
            st.markdown(h(f'<div class="verdict" style="--c:#8FD0B1"><div class="big">No common allergens detected</div>'
                          f'<div class="sub">{sub_msg}</div></div>'), unsafe_allow_html=True)
        else:
            custom_note = (
                f"Also checked {len(custom_allergens)} custom allergen(s); none were found in this text."
                if custom_allergens
                else "No custom allergens were set in your profile."
            )
            sub_msg = f"Nothing in this text matched the allergen list. {esc(custom_note)} Check the printed allergy advice on the pack."
            st.markdown(h(f'<div class="verdict" style="--c:#8FD0B1"><div class="big">No common allergens detected</div>'
                          f'<div class="sub">{sub_msg}</div></div>'), unsafe_allow_html=True)
        if displayable_additives:
            calm_line = ad.format_additive_line(displayable_additives, show_weak=show_weak)
            st.markdown(
                f'<div class="additive-note"><span class="icon">&#9432;</span><span>{esc(calm_line)}</span></div>',
                unsafe_allow_html=True
            )
            with st.expander(":material/info: View additive details & evidence"):
                for m in displayable_additives:
                    st.markdown(ad.render_additive_card_html(m), unsafe_allow_html=True)
    else:
        rows, failed = [], None
        for allergen, info in found.items():
            pred = predict_severity({
                "age": age, "sex": SEX, "asthma_flag": asthma_val, "eczema_flag": int(eczema),
                "family_history_flag": int(family), "known_allergens": profile,
                "detected_allergen": allergen, "is_hidden_allergen": info.get("hidden", False)})
            if "error" in pred:
                failed = pred["error"]
                break
            rows.append({
                "allergen": allergen,
                "info": info,
                "pred": pred,
                "provenance": info.get("provenance"),
                "multilingual_hit": info.get("multilingual_hit"),
            })
        if failed:
            st.error(f"{failed}")
        else:
            order = {"profile": 0, "cross": 1, "general": 2, "unrelated": 3}
            rows.sort(key=lambda r: (order[r["pred"]["mode"]],
                                     -(r["pred"]["p_react"] or 0) if r["pred"]["mode"] == "cross" else 0,
                                     -r["pred"]["severity_score"]))
            st.markdown(verdict_html(rows, custom_hits=custom_hits), unsafe_allow_html=True)

            if displayable_additives:
                calm_line = ad.format_additive_line(displayable_additives, show_weak=show_weak)
                st.markdown(
                    f'<div class="additive-note"><span class="icon">&#9432;</span><span>{esc(calm_line)}</span></div>',
                    unsafe_allow_html=True
                )
                with st.expander(":material/info: View additive details & evidence"):
                    for m in displayable_additives:
                        st.markdown(ad.render_additive_card_html(m), unsafe_allow_html=True)

            all_scores = [r["pred"]["severity_score"] for r in rows] + [c["rating"] for c in custom_hits]
            if asthma_choice in ("Moderate", "Severe") and any(s == 3 for s in all_scores):
                st.markdown(
                    '<div style="background-color: #FDF3E7; border-left: 4px solid #F2A648; color: #1B2A41; '
                    'padding: 12px 16px; border-radius: 6px; margin: 12px 0; font-size: 14px;">'
                    '<strong>Clinical safety note:</strong> Because you indicated moderate or severe asthma and a severe '
                    'allergic reaction is estimated, we recommend discussing an individualized emergency action plan and '
                    'carrying an epinephrine auto-injector with your doctor.</div>',
                    unsafe_allow_html=True
                )

            all_cards = []
            for chit in custom_hits:
                all_cards.append({
                    "html": custom_card_html(chit),
                    "detail": None,
                    "title": None
                })
            for r in rows:
                detail = detail_html(r) if r["pred"]["mode"] != "unrelated" else None
                title = f"Why level {r['pred']['severity_score']} for {kb.pretty(r['allergen'])}" if detail else None
                all_cards.append({
                    "html": card_html(r),
                    "detail": detail,
                    "title": title
                })

            for i in range(0, len(all_cards), 2):
                cols = st.columns(2, gap="medium")
                for col, card in zip(cols, all_cards[i:i + 2]):
                    with col:
                        st.markdown(card["html"], unsafe_allow_html=True)
                        if card["detail"]:
                            with st.expander(card["title"]):
                                st.markdown(card["detail"], unsafe_allow_html=True)



# ------------------------------------------------------------------
# input: general food / product search / photo scan (one workspace per tab)
# ------------------------------------------------------------------
def _pick_hit(src: str):
    S = _state(src)
    i = st.session_state.get(f"hit_pick_{src}")
    if i is not None and 0 <= i < len(S["hits"]):
        _load_text(src, S["hits"][i]["text"], S["hits"][i].get("additives_tags", []))


def _apply_scan_result(res: dict):
    """Show what the scanner found and copy the right text into the Scan tab's ingredient box."""
    S = _state(SRC_SCAN)
    for w in res.get("warnings", []):
        st.warning(w)
    if not res.get("success"):
        st.error(res.get("error", "Could not read text from this image."))
        return
    if res["mode"] == "ingredient_list":
        ing_text = (res.get("ingredients_text") or "").strip()
        if not ing_text:
            st.warning("Text was detected, but no ingredient list could be isolated. Open the scanner details below.")
            return
        if res.get("uncertain_trace_text"):
            ing_text = f"{ing_text} {res['uncertain_trace_text']}"
        S["hits"] = []
        st.session_state.pop(f"hit_pick_{SRC_SCAN}", None)
        _load_text(SRC_SCAN, ing_text, [])
        notes = res.get("pipeline_notes") or []
        st.success(f"Read the ingredient list ({res.get('boxes_count', 0)} text blocks)"
                   + (f" - {'; '.join(notes)}" if notes else "") + ". Please compare it with the pack before relying on it.")
        if res.get("low_confidence_words"):
            st.info("Words the scanner was unsure about: " + ", ".join(res["low_confidence_words"]))
        return
    matches = res.get("matched_products", [])
    S["hits"] = matches
    if matches:
        m = matches[0]
        st.session_state[f"hit_pick_{SRC_SCAN}"] = 0
        _load_text(SRC_SCAN, m["text"], m.get("additives_tags", []))
        if res.get("match_source") == "barcode":
            st.success(f"Barcode {res.get('best_query', '')} matched **{m['name']}** ({m['brand']}) on Open Food Facts.")
        else:
            st.success(f"Front of pack matched **{m['name']}** ({m['brand']}), match score {m.get('match_score', '?')}.")
            st.warning("This match comes from the product name, not a barcode. Check that it is your exact product "
                       "(pick another one in the list below, or scan the ingredient list on the back).")
    else:
        cands = res.get("product_candidates", [])
        cand_str = ", ".join(f"'{c}'" for c in cands) if cands else "nothing readable"
        st.info(f"Front-of-pack text read: {cand_str}. No matching product was found. "
                "Scan the ingredient list on the back of the pack, or use the Search tab.")


def matching_products(src: str):
    hits = _state(src)["hits"]
    if hits:
        st.selectbox("Matching products", list(range(len(hits))), key=f"hit_pick_{src}",
                     on_change=_pick_hit, args=(src,),
                     format_func=lambda i: f"{hits[i]['name']}" + (f" ({hits[i]['brand']})" if hits[i]["brand"] else ""))
        st.caption("Most scanned version first, it is the most common one. Pick another if it is not yours.")


def ingredients_and_results(src: str):
    """Ingredient box, Check button and result, all bound to this source only."""
    S = _state(src)
    text = st.text_area("Ingredients", key=_text_key(src), height=150,
                        placeholder="Wheat flour, skimmed milk powder, soy lecithin, almonds. May contain traces of peanuts.")
    if st.button("Check this label", type="primary", key=f"btn_check_{src}"):
        S["checked"] = True
        S["multi_memo"] = None
    # The label-language picker lives in the Scan tab, so it only applies to scanned text.
    lang = (LANGUAGE_OPTIONS.get(st.session_state.get("label_lang_select", "Auto (Detect)"), "auto")
            if src == SRC_SCAN else "auto")
    render_results(src, text, lang)


st.markdown('<div class="section-title">Scan a photo, search a product, or paste a label</div>', unsafe_allow_html=True)

tab_food, tab_search, tab_scan = st.tabs([
    ":material/restaurant: General Food / Dish (Internet Recipe)",
    ":material/search: Search Product Name (Open Food Facts)",
    ":material/photo_camera: Scan Photo (Label / Pack)",
])

# ---- tab 1: general food ------------------------------------------------------------
with tab_food:
    S_food = _state(SRC_FOOD)
    st.caption("Enter any general food or dish (e.g. **chicken biriyani**, **beef biriyani**, **chicken chow mein**, **granola bar**, **cream soup**, **pad thai**). AllerScan automatically fetches and parses typical ingredients from internet sources (Wikibooks Cookbook, TheMealDB, Wikipedia, Open Food Facts).")

    c_fq, c_fgo = st.columns([5, 1], gap="small")
    f_query = c_fq.text_input("General food or recipe name", key="food_q", label_visibility="collapsed",
                              placeholder="Type a general food, e.g. chicken biriyani, beef biriyani, chow mein, granola bar, cream soup")
    f_go = c_fgo.button("Search Food", use_container_width=True, key="btn_fetch_food")
    fq = f_query.strip()
    if fq and (f_go or fq != S_food["last_q"]):
        S_food["last_q"] = fq
        S_food["error"] = None
        with st.spinner(f"Searching internet for '{fq}' ingredients..."):
            res = fetch_general_food_ingredients(fq)
            if res.get("success"):
                _load_text(SRC_FOOD, res["ingredients_text"], [])
                S_food["note"] = f"✓ Found ingredients for **{res.get('food_name', fq)}** from *{res.get('source')}*."
            else:
                S_food["error"] = res.get("error", f"Could not find ingredients for '{fq}'.")

    if S_food["error"]:
        st.warning(S_food["error"])
    elif S_food["note"]:
        st.info(S_food["note"])

    ingredients_and_results(SRC_FOOD)

# ---- tab 2: product search ----------------------------------------------------------
with tab_search:
    S_search = _state(SRC_SEARCH)
    c_q, c_go = st.columns([5, 1], gap="small")
    query = c_q.text_input("Product name", key="product_q", label_visibility="collapsed",
                           placeholder="Type a product name, e.g. Nutella, Maggi noodles, Oreo")
    go = c_go.button("Search", use_container_width=True)
    q = query.strip()
    if q and (go or q != S_search["last_q"]):
        S_search["last_q"] = q
        S_search["error"] = None
        with st.spinner("Searching Open Food Facts"):
            try:
                hits = search_products(q)
            except Exception as exc:                                    # noqa: BLE001
                hits, S_search["error"] = [], str(exc)
        S_search["hits"] = hits
        if hits:
            st.session_state[f"hit_pick_{SRC_SEARCH}"] = 0
            _load_text(SRC_SEARCH, hits[0]["text"], hits[0].get("additives_tags", []))   # most scanned = most general version

    if S_search["error"]:
        st.warning("Could not reach Open Food Facts. Check your internet connection, or paste the "
                   "ingredients below.")
    elif q and q == S_search["last_q"] and not S_search["hits"]:
        suggs = sc.suggest_name(q, max_suggestions=2)
        if suggs:
            sug_names = ", ".join(f"**{s['display_name']}**" for s in suggs)
            st.info(f"No product with a readable English ingredient list found. Did you mean {sug_names}? "
                    "Try another name, or paste the ingredients below.")
        else:
            st.info("No product with a readable English ingredient list found. Try another name, or paste "
                    "the ingredients below.")

    matching_products(SRC_SEARCH)
    ingredients_and_results(SRC_SEARCH)

# ---- tab 3: photo scan --------------------------------------------------------------
with tab_scan:
    S_scan = _state(SRC_SCAN)
    st.caption("Photograph the **ingredient list** (back of pack), or the **barcode / front** of the product. "
               "You can add several photos of the same pack (for example both sides).")
    with st.expander(":material/photo_camera: How to get a good scan"):
        st.markdown(
            "- Use the **Upload photo(s)** option with your phone's own camera for small print; the browser webcam option gives much lower resolution.\n"
            "- Do not photograph a phone or computer screen.\n"
            "- Fill the frame with the ingredient list, tap to focus, use daylight, and tilt the pack slightly to avoid glare.\n"
            "- For curved cans or bottles, take two or three photos while turning the pack and upload all of them.\n"
            "- Very small image files (longest side under 900 pixels) cannot be read reliably; a photo of a screen or a downloaded thumbnail is not a valid test."
        )
    c_source, c_mode, c_lang = st.columns([1, 1, 1], gap="medium")
    with c_source:
        photo_source = st.radio("Image input", ["Upload photo(s)", "Use webcam / camera"], horizontal=True, key="photo_source")
    with c_mode:
        scan_mode_label = st.radio("Scanner mode", [":material/auto_fix_high: Smart Auto-Detect", ":material/receipt_long: Ingredient Label", ":material/inventory_2: Product Pack Front"], horizontal=True, key="scan_target")
    with c_lang:
        selected_lang_name = st.selectbox("Label language", list(LANGUAGE_OPTIONS.keys()), index=0, key="label_lang_select")
        selected_lang_code = LANGUAGE_OPTIONS[selected_lang_name]

    st.caption(":material/photo_camera: **Guidance:** Photograph the ingredient list in the local language; add a second photo for the English text if the pack has one. Translated labels can contain mistakes; compare with the pack.")

    mode_map = {
        ":material/auto_fix_high: Smart Auto-Detect": "auto",
        ":material/receipt_long: Ingredient Label": "ingredient",
        ":material/inventory_2: Product Pack Front": "product_front"
    }

    photos = []
    if photo_source == "Upload photo(s)":
        files = st.file_uploader("Upload product packaging photo(s)", type=["png", "jpg", "jpeg", "webp"],
                                 key="file_upload", label_visibility="collapsed", accept_multiple_files=True)
        photos = list(files or [])[:4]
    else:
        shot = st.camera_input("Take a photo of product or label", key="cam_upload", label_visibility="collapsed")
        photos = [shot] if shot is not None else []

    if photos:
        c_prev, c_act = st.columns([1, 2], gap="medium")
        with c_prev:
            for ph in photos:
                st.image(ph, use_container_width=True)
        with c_act:
            if st.button(":material/document_scanner: Scan photo(s)", type="primary", key="btn_scan_photo"):
                with st.spinner("Checking photo quality, finding orientation, zooming in and reading text..."):
                    results = []
                    for ph in photos:
                        try:
                            results.append(scan_photo(ph, mode=mode_map[scan_mode_label], search_fn=search_products, label_lang=selected_lang_code))
                        except Exception as exc:                                # noqa: BLE001
                            results.append({"success": False, "error": f"Error during scanning: {exc}"})
                    scan_res = results[0] if len(results) == 1 else merge_scan_results(results)
                    S_scan["scan_res"] = scan_res
                    _apply_scan_result(scan_res)

    matching_products(SRC_SCAN)

    if S_scan["scan_res"] and S_scan["scan_res"].get("success"):
        s_res = S_scan["scan_res"]
        with st.expander(":material/manage_search: View OCR Scanner Details & Diagnostic Text"):
            qual = s_res.get("quality") or {}
            st.caption(f"**Detected mode:** `{s_res.get('mode')}` | **Text blocks:** {s_res.get('boxes_count', 0)} | "
                       f"**Rotation:** `{s_res.get('rotation_applied', 0)}°` | **Straightened:** `{s_res.get('skew_applied', 0)}°` | "
                       f"**Time:** {s_res.get('scan_seconds', '?')} s")
            if qual:
                st.caption(f"Photo: {qual.get('width')}x{qual.get('height')} px | sharpness {qual.get('blur_var')} | "
                           f"brightness {qual.get('brightness')} | contrast {qual.get('contrast')} | glare {qual.get('glare_fraction')}")
            if s_res.get("barcodes"):
                st.caption("**Barcode(s) read:** " + ", ".join(s_res["barcodes"]))
            if s_res.get("engine_candidates"):
                st.caption("**Readings compared:** " + " | ".join(
                    f"{c['engine']}/{c['variant']}: {c['vocab_words']} food words, score {c['score']}" for c in s_res["engine_candidates"]))
            if s_res.get("corrections"):
                st.caption("**Auto-corrected words:** " + ", ".join(f"{c['read']} → {c['healed_to']}" for c in s_res["corrections"]))
            if s_res.get("uncertain_allergens"):
                st.caption("**Seen in only one reading (verify):** " + ", ".join(s_res["uncertain_allergens"]))
            if s_res.get("product_candidates"):
                st.caption(f"**Front Pack Title Candidates:** {', '.join(s_res['product_candidates'])}")

            readings = s_res.get("readings") or []
            if len(readings) >= 2:
                st.markdown("**Other readings:**")
                chosen_idx = next((i for i, r in enumerate(readings) if r.get("is_chosen")), 0)

                def _fmt_reading(i: int) -> str:
                    r = readings[i]
                    tag = " (chosen)" if r.get("is_chosen") else ""
                    return f"{r.get('label', '')} — {r.get('boxes_count', 0)} boxes, score {r.get('score', 0)}{tag}"

                pick_idx = st.radio(
                    "Select a reading to preview:",
                    list(range(len(readings))),
                    index=chosen_idx,
                    format_func=_fmt_reading,
                    key="ocr_reading_choice",
                )
                selected_reading = readings[pick_idx]
                sel_text = selected_reading.get("text", "")
                st.code(sel_text if sel_text else "(No text detected)", language="text")
                if st.button("Use this reading in the ingredient box", key="btn_use_reading"):
                    st.session_state[_text_key(SRC_SCAN)] = sel_text
                    S_scan["checked"] = True
                    S_scan["confirmed"], S_scan["ignored"], S_scan["multi_memo"] = set(), set(), None
            else:
                st.text_area("Extracted / Healed Text", s_res.get("cleaned_text") or s_res.get("raw_text", ""),
                             height=100, disabled=True, key="ocr_raw_disp")

    ingredients_and_results(SRC_SCAN)


info = model_info()
st.markdown(f'<div class="note">Prototype. The model is trained on {info["n_rows"]:,} synthetic records shaped by '
            f'published allergy statistics, not on patient data. Real reaction severity is hard to predict, '
            f'even from past reactions. Estimates are not medical advice. If you have an emergency plan, '
            f'follow it. Product data: Open Food Facts.</div>', unsafe_allow_html=True)
