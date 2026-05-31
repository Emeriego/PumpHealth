import pandas as pd
import numpy as np
import joblib

from src.feature_utils import create_regular_features


# --------------------------------------------------
# LOAD ARTIFACTS (ONLY ONCE)
# --------------------------------------------------

model = joblib.load("models/model.pkl")

ohe = joblib.load("models/ohe.pkl")

freq_maps = joblib.load("models/freq_maps.pkl")

target_encoder = joblib.load("models/lga_target_encoder.pkl")

feature_columns = joblib.load("models/feature_columns.pkl")

lga_map = target_encoder["lga_map"]
global_rate = target_encoder["global_rate"]


low_card_cols = ohe.feature_names_in_.tolist()  # safer than manual list
high_card_cols = list(freq_maps.keys())


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