import pandas as pd
import numpy as np

def drop_irrelevant_columns(df: pd.DataFrame, cols_to_drop: list):
    """
    Drops irrelevant columns safely (only if they exist).
    """

    df = df.copy()

    cols_present = df.columns.intersection(cols_to_drop)

    df = df.drop(columns=cols_present)

    print(f"Dropped columns: {list(cols_present)}")

    return df



def standardize_placeholders(df: pd.DataFrame, placeholder_values=None):
    """
    Replaces placeholder values in categorical columns with NaN.
    """

    df = df.copy()

    if placeholder_values is None:
        placeholder_values = [
            "0", "unknown", "n/a", "na", "null", "none", "", " "
        ]

    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    replaced_counts = {}

    for col in cat_cols:
        # normalize for comparison
        col_clean = df[col].astype(str).str.strip().str.lower()

        mask = col_clean.isin(placeholder_values)

        count = mask.sum()

        if count > 0:
            df.loc[mask, col] = np.nan
            replaced_counts[col] = int(count)

    print(" Placeholder values replaced:")
    print(replaced_counts)

    return df



def to_category(df: pd.DataFrame, cols: list):
    """
    Converts selected columns to categorical dtype safely.
    """

    df = df.copy()

    existing_cols = df.columns.intersection(cols)

    for col in existing_cols:
        df[col] = df[col].astype("category")

    print(f"Converted to category: {list(existing_cols)}")

    return df

def convert_to_datetime(df: pd.DataFrame, cols: list):
    """
    Converts selected columns to datetime safely.
    """

    df = df.copy()

    existing_cols = df.columns.intersection(cols)

    for col in existing_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    print(f"Converted to datetime: {list(existing_cols)}")

    return df


def drop_duplicates(df: pd.DataFrame):
    """
    Drops duplicate rows safely.
    """

    df = df.copy()

    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]

    print(f"Dropped {before - after} duplicate rows")

    return df


import pandas as pd

def handle_missing_values(
    df: pd.DataFrame,
    strategy="fill",
    fill_value=None,
    exclude_cols=None
):
    """
    Handles missing values safely.

    Parameters:
    - strategy: "drop" or "fill"
    - fill_value: custom value for categorical fill
    - exclude_cols: columns to ignore from processing
    """

    df = df.copy()

    if exclude_cols is None:
        exclude_cols = []

    # keep only valid columns (safe)
    exclude_cols = df.columns.intersection(exclude_cols)

    # columns to process
    cols_to_use = df.columns.difference(exclude_cols)

    if strategy == "drop":
        df = df.dropna(subset=cols_to_use)
        print(f"Dropped rows based on missing values (excluding: {list(exclude_cols)})")

    elif strategy == "fill":

        num_cols = cols_to_use.intersection(
            df.select_dtypes(include=["int64", "float64"]).columns
        )

        cat_cols = cols_to_use.intersection(
            df.select_dtypes(include=["object", "category"]).columns
        )

        # numeric fill
        for col in num_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

        # categorical fill
        for col in cat_cols:
            if df[col].isna().any():

                if fill_value is not None:
                    df[col] = df[col].fillna(fill_value)
                else:
                    mode = df[col].mode()
                    df[col] = df[col].fillna(mode[0] if not mode.empty else "unknown")

        print(f"Filled missing values (excluding: {list(exclude_cols)})")

    else:
        raise ValueError("strategy must be either 'drop' or 'fill'")

    return df


def group_rare_categories(df: pd.DataFrame, cols: list, top_n=10, other_label="others"):
    """
    Keeps top N categories and groups the rest as 'others'.
    """

    df = df.copy()

    existing_cols = df.columns.intersection(cols)

    for col in existing_cols:
        top_categories = df[col].value_counts().nlargest(top_n).index

        df[col] = df[col].apply(
            lambda x: x if x in top_categories else other_label
        )

    print(f"Grouped rare categories in: {list(existing_cols)}")

    return df

def save_cleaned_data(df: pd.DataFrame, path: str, index=False):
    """
    Saves cleaned dataframe to CSV.
    """

    df.to_csv(path, index=index)

    print(f"Saved cleaned data to: {path}")



def log_transform_skewed_columns(
    df: pd.DataFrame,
    cols=None,
    exclude_cols=None,
    skew_threshold=1.0,
    inplace=False
):
    """
    Applies log1p transformation to highly skewed numeric columns.

    - Only transforms columns with skew > threshold
    - Safe for zeros (uses log1p)
    - Can exclude columns
    """

    if not inplace:
        df = df.copy()

    if exclude_cols is None:
        exclude_cols = []

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    numeric_cols = numeric_cols.difference(df.columns.intersection(exclude_cols))

    if cols is not None:
        cols = df.columns.intersection(cols)
    else:
        cols = numeric_cols

    for col in cols:
        if df[col].dropna().empty:
            continue

        skewness = df[col].skew()

        if skewness > skew_threshold:
            df[col] = np.log1p(df[col])
            print(f"Log transformed: {col} (skew={skewness:.2f})")

    return df

import pandas as pd

def cap_outliers_iqr(
    df: pd.DataFrame,
    cols=None,
    exclude_cols=None,
    factor=1.5,
    inplace=False
):
    """
    Caps outliers using IQR method (winsorization).

    - Does NOT remove rows
    - Only caps extreme values
    """

    if not inplace:
        df = df.copy()

    if exclude_cols is None:
        exclude_cols = []

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    numeric_cols = numeric_cols.difference(df.columns.intersection(exclude_cols))

    if cols is not None:
        cols = df.columns.intersection(cols)
    else:
        cols = numeric_cols

    for col in cols:
        if df[col].dropna().empty:
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR

        df[col] = df[col].clip(lower, upper)

        print(f"Outliers capped: {col}")

    return df





