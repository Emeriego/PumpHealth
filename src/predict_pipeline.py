import pandas as pd
import numpy as np
import joblib

from src.feature_utils import create_regular_features
from src.data_utils import load_data
from src.cleaning_utils import (save_cleaned_data, drop_irrelevant_columns, drop_duplicates)


from src.feature_utils import (
basic_clean,
apply_missing_values,
apply_geo_imputer,
apply_log_transform,
apply_value_replacement,
apply_outlier_caps,
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

def predict_pump_status(df: pd.DataFrame):

    df = df.copy()

    # -------------------------
    # 1. FEATURE ENGINEERING
    # -------------------------
    df = create_regular_features(df)

    # -------------------------
    # 2. TARGET ENCODING (LGA)
    # -------------------------
    if "lga" in df.columns:
        df["lga_te"] = (
            df["lga"]
            .map(lga_map)
            .fillna(global_rate)
        )
        df = df.drop(columns=["lga"])

    # -------------------------
    # 3. FREQUENCY ENCODING
    # -------------------------
    for col in high_card_cols:
        if col in df.columns:
            df[col + "_freq"] = df[col].map(freq_maps[col]).fillna(0)
            df = df.drop(columns=[col])

    # -------------------------
    # 4. ONE HOT ENCODING
    # -------------------------
    ohe_input = df[low_card_cols]

    ohe_df = pd.DataFrame(
        ohe.transform(ohe_input),
        columns=ohe.get_feature_names_out(low_card_cols),
        index=df.index
    )

    df = df.drop(columns=low_card_cols).join(ohe_df)

    # -------------------------
    # 5. ALIGN COLUMNS
    # -------------------------
    df = df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # -------------------------
    # 6. PREDICTION
    # -------------------------
    prediction = model.predict(df)

    return prediction[0]


# --------------------------------------------------
# BATCH PREDICTION
# --------------------------------------------------

def predict_batch(df: pd.DataFrame):

    df = df.copy()

    # feature engineering
    df = create_regular_features(df)

    # target encoding
    if "lga" in df.columns:
        df["lga_te"] = (
            df["lga"]
            .map(lga_map)
            .fillna(global_rate)
        )
        df = df.drop(columns=["lga"])

    # frequency encoding
    for col in high_card_cols:
        if col in df.columns:
            df[col + "_freq"] = df[col].map(freq_maps[col]).fillna(0)
            df = df.drop(columns=[col])

    # OHE
    ohe_df = pd.DataFrame(
        ohe.transform(df[low_card_cols]),
        columns=ohe.get_feature_names_out(low_card_cols),
        index=df.index
    )

    df = df.drop(columns=low_card_cols).join(ohe_df)

    # align
    df = df.reindex(columns=feature_columns, fill_value=0)

    # predictions
    preds = model.predict(df)

    df["prediction"] = preds

    return df