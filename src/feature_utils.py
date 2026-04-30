import pandas as pd
import numpy as np

from src.cleaning_utils import (
    drop_irrelevant_columns,
    standardize_placeholders,
    to_category,
    convert_to_datetime,
    drop_duplicates,
)

def basic_clean(df: pd.DataFrame):
    df = drop_irrelevant_columns(df, [
        "id", "wpt_name", "recorded_by", "scheme_name",
        "num_private", "extraction_type_group",
        "payment_type", "quantity_group", "water_quality", "source",
        "waterpoint_type_group", "subvillage", "extraction_type_class", "region_code",
        "district_code", "ward"
    ])

    df = standardize_placeholders(df)

    df = to_category(df, cols=["district_code", "region_code"])

    df = convert_to_datetime(df, cols=["date_recorded"])

    df = drop_duplicates(df)

    return df



def fit_missing_values(df: pd.DataFrame):
    df = df.copy()

    df.loc[df["construction_year"] == 0, "construction_year"] = np.nan

    return {
        "num_medians": df.select_dtypes(include=["int64", "float64", "number"]).median(),
        "cat_modes": df.select_dtypes(include=["object", "category"]).mode().iloc[0]
    }


def apply_missing_values(df: pd.DataFrame, stats: dict):
    df = df.copy()

    if "construction_year" in df.columns:
        df.loc[df["construction_year"] == 0, "construction_year"] = np.nan

    for col, val in stats["num_medians"].items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    for col, val in stats["cat_modes"].items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    return df


def fit_rare_categories(df: pd.DataFrame, cols: list, top_n=25):
    stats = {}

    for col in df.columns.intersection(cols):
        stats[col] = df[col].value_counts().nlargest(top_n).index.tolist()

    return stats


def apply_rare_categories(df: pd.DataFrame, stats: dict, other_label="others"):
    df = df.copy()

    for col, top_vals in stats.items():
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: x if x in top_vals else other_label
            )

    return df


def fit_outlier_caps(df: pd.DataFrame, cols=None, exclude_cols=None, factor=1.5):
    if exclude_cols is None:
        exclude_cols = []

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    numeric_cols = numeric_cols.difference(df.columns.intersection(exclude_cols))

    if cols is not None:
        numeric_cols = df.columns.intersection(cols)

    stats = {}

    for col in numeric_cols:
        if df[col].dropna().empty:
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        stats[col] = (
            Q1 - factor * IQR,
            Q3 + factor * IQR
        )

    return stats


def apply_outlier_caps(df: pd.DataFrame, stats: dict):
    df = df.copy()

    for col, (low, high) in stats.items():
        if col in df.columns:
            df[col] = df[col].clip(low, high)

    return df



def fit_value_replacement(
    df: pd.DataFrame,
    cols: list,
    treat_zero_as_nan: list = None,
    treat_negative_as_nan: list = None,
    group_cols: list = None
):
    df = df.copy()

    treat_zero_as_nan = treat_zero_as_nan or []
    treat_negative_as_nan = treat_negative_as_nan or []
    group_cols = group_cols or []

    for col in df.columns.intersection(treat_zero_as_nan):
        df.loc[df[col] == 0, col] = np.nan

    for col in df.columns.intersection(treat_negative_as_nan):
        df.loc[df[col] < 0, col] = np.nan

    stats = {"global": df[cols].median()}

    for group in group_cols:
        stats[group] = df.groupby(group)[cols].median()

    return stats


def apply_value_replacement(
    df,
    stats,
    cols,
    treat_zero_as_nan=None,
    treat_negative_as_nan=None,
    group_cols=None
):
    df = df.copy()

    treat_zero_as_nan = treat_zero_as_nan or []
    treat_negative_as_nan = treat_negative_as_nan or []
    group_cols = group_cols or []

    for col in cols:
        if col in df.columns:
            if col in treat_zero_as_nan:
                df.loc[df[col] == 0, col] = np.nan
            if col in treat_negative_as_nan:
                df.loc[df[col] < 0, col] = np.nan

    def fill_value(row, col):
        if pd.notna(row[col]):
            return row[col]

        for group in group_cols:
            if group in stats and row[group] in stats[group].index:
                val = stats[group].loc[row[group], col]
                if pd.notna(val):
                    return val

        return stats["global"][col]

    for col in cols:
        df[col] = df.apply(lambda r: fill_value(r, col), axis=1)

    for col in cols:
        if col in treat_zero_as_nan:
            df.loc[df[col] == 0, col] = np.nan
        if col in treat_negative_as_nan:
            df.loc[df[col] < 0, col] = np.nan

    return df

def fit_geo_imputer(df: pd.DataFrame, lat_col="latitude", lon_col="longitude"):
    df = df.copy()

    # treat 0 as missing
    df.loc[df[lat_col] == 0, lat_col] = np.nan
    df.loc[df[lon_col] == 0, lon_col] = np.nan

    lga_stats = df.groupby("lga")[[lat_col, lon_col]].median()
    global_stats = df[[lat_col, lon_col]].median()

    return {
        "lga": lga_stats,
        "global": global_stats
    }


def apply_geo_imputer(df: pd.DataFrame, stats: dict, lat_col="latitude", lon_col="longitude"):
    df = df.copy()

    df.loc[df[lat_col] == 0, lat_col] = np.nan
    df.loc[df[lon_col] == 0, lon_col] = np.nan

    lga_stats = stats["lga"]
    global_stats = stats["global"]

    def fill(row, col):
        lga = row["lga"]

        # lga-level imputation
        if lga in lga_stats.index:
            val = lga_stats.loc[lga, col]
            if pd.notna(val):
                return val

        # fallback
        return global_stats[col]

    df[lat_col] = df.apply(lambda r: fill(r, lat_col), axis=1)
    df[lon_col] = df.apply(lambda r: fill(r, lon_col), axis=1)

    return df


def fit_log_transform(df: pd.DataFrame, cols=None, exclude_cols=None, skew_threshold=1.0):
    if exclude_cols is None:
        exclude_cols = []

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    numeric_cols = numeric_cols.difference(df.columns.intersection(exclude_cols))

    if cols is not None:
        numeric_cols = numeric_cols.intersection(cols)

    skewed_cols = [
        col for col in numeric_cols
        if df[col].dropna().skew() > skew_threshold
    ]

    return skewed_cols


def apply_log_transform(df: pd.DataFrame, cols: list, suffix="_log"):
    df = df.copy()

    for col in cols:
        if col in df.columns:
            df[f"{col}{suffix}"] = np.log1p(df[col])

    return df


def create_regular_features(df: pd.DataFrame):
    df = df.copy()

    # --- date handling first ---
    if "date_recorded" in df.columns:
        df["date_recorded"] = pd.to_datetime(df["date_recorded"])

        df["recorded_year"] = df["date_recorded"].dt.year
        df["recorded_month"] = df["date_recorded"].dt.month

        df = df.drop(columns=["date_recorded"])

    # --- pump age ---
    if "construction_year" in df.columns:
        df["pump_age"] = df["recorded_year"] - df["construction_year"]
        df["pump_age"] = df["pump_age"].clip(lower=0, upper=100)

        df["pump_is_new"] = (df["pump_age"] <= 3).astype(int)

        # ordinal encoding (IMPORTANT FIX)
        df["pump_age_band"] = pd.cut(
            df["pump_age"],
            bins=[0, 5, 10, 20, 100],
            labels=[0, 1, 2, 3]   # numeric now
        ).astype(float)

    # --- height bands (ordinal encoding) ---
    if "gps_height" in df.columns:
        df["height_band"] = pd.cut(
            df["gps_height"],
            bins=[-100, 0, 500, 1000, 1500, 3000],
            labels=[0, 1, 2, 3, 4]   # numeric now
        ).astype(float)

    # --- population ---
    if "population" in df.columns:
        df["population_log"] = np.log1p(df["population"])

    # --- quantity mapping ---
    if "quantity" in df.columns:
        quantity_map = {
            "dry": 0,
            "insufficient": 1,
            "seasonal": 2,
            "enough": 3,
            "unknown": np.nan
        }
        df["quantity_score"] = df["quantity"].map(quantity_map)

    return df


def create_binary_features(df: pd.DataFrame):
    df = df.copy()

    # payment signal
    if "payment" in df.columns:
        df["is_paid_water"] = (df["payment"] != "never pay").astype(int)

    # water safety
    if "quality_group" in df.columns:
        df["is_water_safe"] = (df["quality_group"] == "good").astype(int)

    return df

def target_encode_lga(X_train, X_test, y_train, target_col="status_group", cat_col="lga"):
    # 1. combine train features + target
    train_df = X_train.copy()
    train_df[target_col] = y_train

    # 2. compute failure rate per LGA
    lga_map = train_df.groupby(cat_col)[target_col].apply(
        lambda x: (x == "non functional").mean()
    )

    # 3. global fallback rate
    global_rate = (y_train == "non functional").mean()

    # 4. apply mapping
    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[f"{cat_col}_te"] = X_train[cat_col].map(lga_map).fillna(global_rate)
    X_test[f"{cat_col}_te"] = X_test[cat_col].map(lga_map).fillna(global_rate)

    # 5. drop original column
    X_train = X_train.drop(columns=[cat_col])
    X_test = X_test.drop(columns=[cat_col])

    return X_train, X_test, lga_map
