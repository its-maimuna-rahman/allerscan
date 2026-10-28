"""
Unit tests for data cleaning and feature engineering pipeline focusing on
the 4-level asthma implementation (Task 1).
"""
import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pipeline as pl


def test_to_asthma_levels_mapping():
    # Canonical numbers
    s_num = pd.Series([0, 1, 2, 3, 4, -1, np.nan])
    res_num, legacy_num = pl._to_asthma(s_num)
    assert legacy_num == 0
    assert res_num.iloc[0] == 0.0
    assert res_num.iloc[1] == 1.0
    assert res_num.iloc[2] == 2.0
    assert res_num.iloc[3] == 3.0
    assert pd.isna(res_num.iloc[4])  # invalid 4 -> NaN
    assert pd.isna(res_num.iloc[5])  # invalid -1 -> NaN
    assert pd.isna(res_num.iloc[6])  # NaN -> NaN

    # Text variants
    s_txt = pd.Series(["No", "none", "0", "Mild", "mild asthma", "1", "Moderate", "mod", "2",
                      "Severe", "sev", "severe asthma", "3", "unknown_val"])
    res_txt, legacy_txt = pl._to_asthma(s_txt)
    assert legacy_txt == 0
    assert list(res_txt.iloc[0:3]) == [0.0, 0.0, 0.0]
    assert list(res_txt.iloc[3:6]) == [1.0, 1.0, 1.0]
    assert list(res_txt.iloc[6:9]) == [2.0, 2.0, 2.0]
    assert list(res_txt.iloc[9:13]) == [3.0, 3.0, 3.0, 3.0]
    assert pd.isna(res_txt.iloc[13])


def test_legacy_yes_converted_to_missing():
    # Legacy yes/true/y should be converted to NaN and counted
    s = pd.Series(["yes", "Yes", "Y", "true", "True", "t", "mild", "no"])
    res, n_legacy = pl._to_asthma(s)
    assert n_legacy == 6
    for i in range(6):
        assert pd.isna(res.iloc[i])
    assert res.iloc[6] == 1.0
    assert res.iloc[7] == 0.0


def test_clean_raw_reports_asthma_details():
    df = pd.DataFrame({
        "record_id": ["R001", "R002", "R003", "R004", "R005"],
        "person_id": ["P001", "P002", "P003", "P004", "P005"],
        "source": ["us_child_survey"] * 5,
        "age": [10, 20, 30, 40, 50],
        "sex": ["M", "F", "1", "0", "M"],
        "has_asthma": ["no", "mild", "moderate", "severe", "yes"],
        "has_eczema": ["1", "0", "1", "0", "1"],
        "family_history_allergy": ["0", "1", "0", "1", "0"],
        "profile": ["peanut:2", "", "almond:1", "", ""],
        "detected_allergen": ["peanut", "almond", "cashew", "milk", "egg"],
        "is_hidden_allergen_flag": [0, 0, 0, 0, 0],
        "reacted": [1, 1, 0, 1, 0],
        "severity": [2, 1, np.nan, 3, np.nan]
    })
    clean, rep = pl.clean_raw(df)
    assert rep["legacy_asthma_answers_set_missing"] == 1
    counts = rep["asthma_level_counts"]
    assert counts["0"] == 1
    assert counts["1"] == 1
    assert counts["2"] == 1
    assert counts["3"] == 1
    assert counts["missing"] == 1


def test_atopic_burden_uses_binary_asthma_indicator():
    # Person with severe asthma (3.0), eczema (1.0), family history (0.0)
    # Atopic burden should be 1.0 + 1.0 + 0.0 = 2.0 (NOT 3.0 + 1.0 = 4.0!)
    df = pd.DataFrame([{
        "age": 25, "sex": 1.0, "has_asthma": 3.0, "has_eczema": 1.0, "family_history_allergy": 0.0,
        "is_hidden_allergen_flag": 0, "profile": "", "detected_allergen": "peanut"
    }])
    X = pl.make_features(df)
    assert X["has_asthma"].iloc[0] == 3.0
    assert X["atopic_burden"].iloc[0] == 2.0

    # Person with mild asthma (1.0), eczema (1.0), family history (1.0) -> burden 3.0
    df2 = pd.DataFrame([{
        "age": 25, "sex": 1.0, "has_asthma": 1.0, "has_eczema": 1.0, "family_history_allergy": 1.0,
        "is_hidden_allergen_flag": 0, "profile": "", "detected_allergen": "peanut"
    }])
    X2 = pl.make_features(df2)
    assert X2["atopic_burden"].iloc[0] == 3.0

    # Person with missing asthma, eczema 1.0, family 0.0 -> burden 1.0
    df3 = pd.DataFrame([{
        "age": 25, "sex": 1.0, "has_asthma": np.nan, "has_eczema": 1.0, "family_history_allergy": 0.0,
        "is_hidden_allergen_flag": 0, "profile": "", "detected_allergen": "peanut"
    }])
    X3 = pl.make_features(df3)
    assert X3["atopic_burden"].iloc[0] == 1.0


def test_feature_list_schema_preserved():
    df = pd.DataFrame([{
        "age": 20, "sex": 0.0, "has_asthma": 2.0, "has_eczema": 0.0, "family_history_allergy": 1.0,
        "is_hidden_allergen_flag": 0, "profile": "almond:2", "detected_allergen": "cashew"
    }])
    X = pl.make_features(df)
    assert list(X.columns) == pl.FEATURES


def test_build_query_df_handles_invalid_asthma():
    q_valid = pl.build_query_df({"peanut": 2}, "cashew", asthma=2)
    assert q_valid["has_asthma"].iloc[0] == 2.0

    q_invalid = pl.build_query_df({"peanut": 2}, "cashew", asthma=99)
    assert pd.isna(q_invalid["has_asthma"].iloc[0])

    q_none = pl.build_query_df({"peanut": 2}, "cashew", asthma=None)
    assert pd.isna(q_none["has_asthma"].iloc[0])


def test_app_levels_and_model_integration():
    import model
    levels = {"No": 0.0, "Mild": 1.0, "Moderate": 2.0, "Severe": 3.0, "Not sure": None}
    for choice, val in levels.items():
        df = pl.build_query_df({"peanut": 2}, "cashew", age=30, asthma=val)
        val_in_df = df["has_asthma"].iloc[0]
        if val is None:
            assert pd.isna(val_in_df)
        else:
            assert val_in_df == val
        pred = model.predict_severity({
            "detected_allergen": "cashew", "known_allergens": {"peanut": 2},
            "age": 30, "asthma_flag": val
        })
        assert "error" not in pred
        assert 1.0 <= pred["expected_level"] <= 3.0
        assert 0.0 <= pred["severe_risk"] <= 1.0
