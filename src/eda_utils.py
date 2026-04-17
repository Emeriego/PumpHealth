import pandas as pd
import matplotlib.pyplot as plt


def data_overview(df: pd.DataFrame):
    """
    Prints basic dataset information.
    """
    print("Shape:", df.shape)
    print("\nData Types:\n", df.dtypes)
    print("\nFirst 5 rows:\n", df.head())

def data_structure(df):
    """
    Shows basic structure of the dataset.
    """
    print(" Shape:", df.shape)
    print("\n Columns:")
    print(df.columns.tolist())

def data_types(df):
    """
    Shows column data types.
    """
    print(" Data Types:")
    print(df.dtypes)

def show_full_dataframe(df, rows=5):
    """
    Displays full dataframe without truncation.
    """

    with pd.option_context(
        'display.max_columns', None,
        'display.width', 200
    ):
        print(df.head(rows))

def full_summary(df):
    """
    Shows full describe output (numeric + categorical).
    """

    with pd.option_context('display.max_columns', None):
        print("Numeric Summary:")
        print(df.describe())


def data_quality_report(df):
    """
    Shows missing values and duplicates.
    """
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100

    report = pd.DataFrame({
        "missing_values": missing,
        "percentage": missing_percent
    }).sort_values(by="percentage", ascending=False)

    print(" Missing Values Report:")
    print(report)


def check_numerical_anomalies(df: pd.DataFrame):
    """
    Checks numeric columns for non-numeric values.
    """

    issues = {}

    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'number']).columns
    numeric_cols = numeric_cols.difference(df.columns.intersection(['region_code', 'district_code','id', 'latitude', 'longitude']))
    for col in numeric_cols:
        # force convert everything to numeric
        converted = pd.to_numeric(df[col], errors='coerce')

        # find values that became NaN during conversion
        bad_values = df[col][converted.isna() & df[col].notna()].unique()

        if len(bad_values) > 0:
            issues[col] = bad_values.tolist()

    print(" Numerical Anomalies:")
    print(issues)

    return issues


import pandas as pd

def check_categorical_anomalies(df: pd.DataFrame):
    """
    Checks categorical columns for numeric-only values
    and returns counts of each anomaly.
    """

    issues = {}

    cat_cols = df.select_dtypes(include=['object',  'category']).columns

    for col in cat_cols:

        # find numeric-only strings
        numeric_like = df[col].astype(str).str.fullmatch(r"\d+")

        # filter problematic values
        bad_series = df[col][numeric_like.fillna(False)]

        if not bad_series.empty:
            # count occurrences of each anomaly
            counts = bad_series.value_counts().to_dict()

            issues[col] = counts

    print(" Categorical Anomalies (with counts):")
    print(issues)

    return issues

import pandas as pd

def check_placeholder_values(df: pd.DataFrame):
    """
    Detects common placeholder/missing values in categorical columns
    such as: "0", "unknown", "N/A", "na", etc.
    """

    issues = {}

    placeholder_values = {
        "0", "unknown", "n/a", "na", "null", "none", "", " "
    }

    cat_cols = df.select_dtypes(include=['object',  'category']).columns

    for col in cat_cols:

        col_clean = df[col].astype(str).str.strip().str.lower()

        mask = col_clean.isin(placeholder_values)

        bad_values = df[col][mask]

        if not bad_values.empty:
            issues[col] = bad_values.value_counts().to_dict()

    print(" Placeholder Values Found:")
    print(issues)

    return issues


def plot_numeric_distributions(df, cols=None):
    """
    Plots histograms for numeric columns.
    """

    if cols is None:
        numeric_cols = df.select_dtypes(include=['int64', 'float64', 'number']).columns
        cols = numeric_cols.difference(df.columns.intersection(['region_code', 'district_code','id', 'latitude', 'longitude']))
    for col in cols:
        plt.figure()
        df[col].dropna().hist(bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()



def plot_categorical_counts(df, cols=None, top_n=10):
    """
    Plots bar charts for categorical columns.
    """

    if cols is None:
        cols = df.select_dtypes(include=['object',  'category']).columns

    for col in cols:
        plt.figure()
        df[col].value_counts().head(top_n).plot(kind='bar')
        plt.title(f"Top {top_n} values in {col}")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()


def plot_target_distribution(df, target_col):
    """
    Shows class balance of target variable.
    """

    plt.figure()
    df[target_col].value_counts().plot(kind='bar')
    plt.title("Target Distribution")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()


def plot_feature_vs_target(df, feature, target):
    """
    Shows relationship between categorical feature and target.
    """

    import pandas as pd

    cross = pd.crosstab(df[feature], df[target])

    cross.plot(kind='bar', stacked=True, figsize=(8, 5))

    plt.title(f"{feature} vs {target}")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()



def skewness_report(
    df: pd.DataFrame,
    cols=None,
    exclude_cols=None,
    threshold=1.0
):
    """
    Reports skewness of numeric columns.

    Parameters:
    - cols: specific columns to check (optional)
    - exclude_cols: columns to exclude
    - threshold: threshold to flag high skew
    """

    if exclude_cols is None:
        exclude_cols = []

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    numeric_cols = numeric_cols.difference(df.columns.intersection(exclude_cols))

    if cols is not None:
        cols = df.columns.intersection(cols)
    else:
        cols = numeric_cols

    skew_vals = df[cols].skew()

    report = pd.DataFrame({
        "skewness": skew_vals.round(2)
    })

    report["status"] = report["skewness"].apply(
        lambda x: "High" if abs(x) > threshold else "OK"
    )

    report = report.sort_values(by="skewness", ascending=False)

    print("📊 Skewness Report:")
    print(report)

    return report

def outlier_overview(df: pd.DataFrame, exclude_cols=None):
    """
    Simple outlier report:
    - number of outliers
    - percentage
    - severity label
    """

    if exclude_cols is None:
        exclude_cols = []

    numeric_cols = df.select_dtypes(include=['int64', 'float64']) \
                     .columns.difference(df.columns.intersection(exclude_cols))

    rows = []

    for col in numeric_cols:
        series = df[col].dropna()

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = series[(series < lower) | (series > upper)]
        pct = len(outliers) / len(series) * 100

        # simple severity label
        if pct < 2:
            severity = "Low"
        elif pct < 10:
            severity = "Moderate"
        else:
            severity = "High"

        rows.append({
            "column": col,
            "outliers": len(outliers),
            "percentage (%)": round(pct, 2),
            "severity": severity
        })

    result = pd.DataFrame(rows).sort_values(by="percentage (%)", ascending=False)

    print(" Simple Outlier Overview:")
    print(result)

    return result

def plot_outliers(df, cols=None):
    """
    Boxplots for detecting outliers in numeric columns.
    """

    if cols is None:
        numeric_cols = df.select_dtypes(include=['int64', 'float64', 'number']).columns
        cols = numeric_cols.difference(df.columns.intersection(['region_code', 'district_code','id', 'latitude', 'longitude']))
    for col in cols:
        plt.figure()
        df.boxplot(column=col)
        plt.title(f"Outliers in {col}")
        plt.show()


def count_unique_values(df: pd.DataFrame, cols=None):
    """
    Returns number of unique values per categorical column.
    """

    if cols is None:
        cols = df.select_dtypes(include=["object", "category"]).columns

    cols = df.columns.intersection(cols)

    result = {col: df[col].nunique() for col in cols}

    return pd.Series(result).sort_values(ascending=False)


def get_numeric_columns(df: pd.DataFrame, exclude_cols=None):
    """
    Returns a list of numeric columns.

    Parameters:
    - exclude_cols: columns to exclude (optional)
    """

    if exclude_cols is None:
        exclude_cols = []

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    numeric_cols = numeric_cols.difference(df.columns.intersection(exclude_cols))

    numeric_cols = list(numeric_cols)

    print("🔢 Numeric Columns:")
    print(numeric_cols)

    return numeric_cols



def get_categorical_columns(df: pd.DataFrame, exclude_cols=None):
    """
    Returns a list of categorical columns (object + category).

    Parameters:
    - exclude_cols: columns to exclude (optional)
    """

    if exclude_cols is None:
        exclude_cols = []

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    cat_cols = cat_cols.difference(df.columns.intersection(exclude_cols))

    cat_cols = list(cat_cols)

    print("🔠 Categorical Columns:")
    print(cat_cols)

    return cat_cols