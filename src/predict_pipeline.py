import streamlit as st
import pandas as pd
import numpy as np
import joblib


from src.feature_utils import create_regular_features
from src.cleaning_utils import (drop_irrelevant_columns, drop_duplicates)


from src.feature_utils import (
basic_clean,
apply_missing_values,
apply_geo_imputer,
apply_log_transform,
apply_value_replacement,
apply_rare_categories,
create_regular_features,
create_binary_features,
apply_target_encoder,
apply_frequency_encoding,
apply_ohe


)

# --------------------------------------------------
# LOAD ARTIFACTS (ONLY ONCE)
# --------------------------------------------------

model = joblib.load("models/model.pkl")

missing_stats = joblib.load("models/missing_stats.pkl")
rare_stats = joblib.load("models/rare_stats.pkl")
val_repl_stats = joblib.load("models/val_repl_stats.pkl")
geo_stats = joblib.load("models/geo_stats.pkl")

log_cols = joblib.load("models/log_cols.pkl")

freq_maps = joblib.load("models/freq_maps.pkl")

ohe = joblib.load("models/ohe.pkl")

lga_map = joblib.load("models/lga_map.pkl")
global_rate = joblib.load("models/global_rate.pkl")

low_card_cols = joblib.load("models/low_card_cols.pkl")
high_card_cols = joblib.load("models/high_card_cols.pkl")


selected_features = joblib.load("models/selected_features.pkl")


# --------------------------------------------------
# MAIN PREDICTION FUNCTION (SINGLE)
# --------------------------------------------------

def preprocess_input(df: pd.DataFrame):

    df = df.copy()

    # --------------------------------------------------
    # 1. BASIC CLEANING
    # --------------------------------------------------
    df = basic_clean(
        df,
        cols_to_drop=[
            "id",
            "wpt_name",
            "recorded_by",
            "scheme_name",
            "num_private",
            "extraction_type_group",
            "payment_type",
            "quantity_group",
            "water_quality",
            "source",
            "waterpoint_type_group",
            "subvillage",
            "extraction_type_class",
            "region_code",
            "district_code",
            "ward",
            "region"
        ]
    )

    # --------------------------------------------------
    # 2. MISSING VALUES
    # --------------------------------------------------
    df = apply_missing_values(
        df,
        missing_stats
    )

    # --------------------------------------------------
    # 3. RARE CATEGORIES
    # --------------------------------------------------
    df = apply_rare_categories(
        df,
        rare_stats
    )

    # --------------------------------------------------
    # 4. VALUE REPLACEMENT
    # --------------------------------------------------
    df = apply_value_replacement(
        df,
        val_repl_stats,
        cols=["gps_height", "population"],
        treat_zero_as_nan=["gps_height", "population"],
        treat_negative_as_nan=["gps_height"],
        group_cols=["lga"]
    )

    # --------------------------------------------------
    # 5. LOG TRANSFORMS
    # --------------------------------------------------
    df = apply_log_transform(
        df,
        log_cols
    )

    # --------------------------------------------------
    # 6. GEO IMPUTATION
    # --------------------------------------------------
    df = apply_geo_imputer(
        df,
        geo_stats,
        lat_col="latitude",
        lon_col="longitude"
    )

    # --------------------------------------------------
    # 7. FEATURE ENGINEERING
    # --------------------------------------------------
    df = create_regular_features(df)

    df = create_binary_features(df)

    # --------------------------------------------------
    # 8. TARGET ENCODING
    # --------------------------------------------------
    df = apply_target_encoder(
        df,
        lga_map,
        global_rate,
        cat_col="lga"
    )

    # --------------------------------------------------
    # 9. FREQUENCY ENCODING
    # --------------------------------------------------
    df = apply_frequency_encoding(
        df,
        freq_maps
    )

    # --------------------------------------------------
    # 10. ONE HOT ENCODING
    # --------------------------------------------------
    df = apply_ohe(
        df,
        ohe,
        low_card_cols
    )

   

    return df

def predict_pump_status(df: pd.DataFrame):

    df = preprocess_input(df)

    df = df.reindex(
        columns=selected_features,
        fill_value=0
    )

    prediction = model.predict(df)[0]

    return prediction

def predict_batch(df: pd.DataFrame):

    df = preprocess_input(df)

    df = df.reindex(columns=selected_features, fill_value=0)

    preds = model.predict(df)

    df = df.copy()
    df["prediction"] = preds
    return df
