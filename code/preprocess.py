import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_NUMERIC_COLUMNS = [
    "Year",
    "Financial Loss (in Million $)",
    "Number of Affected Users",
    "Incident Resolution Time (in Hours)",
]

DEFAULT_UNKNOWN_TOKENS = {"unknown", "n/a", "na", "none", "null", "?", "nan", ""}


def preprocess_dataset(
    df, numeric_columns=None, unknown_tokens=None, keep_unknown_as_category=True
):
    """
    Clean the full dataset and return a data quality report.

    Returns:
        tuple[pd.DataFrame, dict]: (cleaned_dataframe, quality_report)
    """
    df_clean = df.copy()
    numeric_columns = numeric_columns or [
        col for col in DEFAULT_NUMERIC_COLUMNS if col in df_clean.columns
    ]
    unknown_tokens = {token.strip().lower() for token in (unknown_tokens or DEFAULT_UNKNOWN_TOKENS)}

    initial_rows, initial_cols = df_clean.shape

    duplicate_count = int(df_clean.duplicated().sum())
    if duplicate_count:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    categorical_columns = list(df_clean.select_dtypes(include=["object", "category"]).columns)
    unknown_like_counts = {}

    for col in categorical_columns:
        as_text = df_clean[col].astype(str).str.strip()
        normalized = as_text.str.lower()
        mask_unknown = normalized.isin(unknown_tokens)
        unknown_like_counts[col] = int(mask_unknown.sum())

        if keep_unknown_as_category:
            df_clean.loc[mask_unknown, col] = "Unknown"
        else:
            df_clean.loc[mask_unknown, col] = pd.NA

    for col in numeric_columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    missing_values_per_column = df_clean.isna().sum().to_dict()
    categorical_cardinality = {
        col: int(df_clean[col].nunique(dropna=False)) for col in categorical_columns
    }

    report = {
        "initial_shape": {"rows": int(initial_rows), "columns": int(initial_cols)},
        "final_shape": {"rows": int(df_clean.shape[0]), "columns": int(df_clean.shape[1])},
        "rows_removed_by_duplicates": duplicate_count,
        "numeric_columns_cast": numeric_columns,
        "missing_values_per_column": missing_values_per_column,
        "unknown_like_counts_per_column": unknown_like_counts,
        "categorical_cardinality": categorical_cardinality,
        "unknown_policy": "keep_as_category" if keep_unknown_as_category else "set_as_missing",
    }
    return df_clean, report


def print_data_quality_report(report):
    """Pretty-print the preprocessing quality report."""
    initial = report["initial_shape"]
    final = report["final_shape"]

    print("\n========== DATA QUALITY REPORT ==========")
    print(f"Initial shape: {initial['rows']} rows x {initial['columns']} columns")
    print(f"Final shape:   {final['rows']} rows x {final['columns']} columns")
    print(f"Rows removed due to duplicates: {report['rows_removed_by_duplicates']}")
    print(f"Unknown policy: {report['unknown_policy']}")
    print(f"Numeric columns cast: {report['numeric_columns_cast']}")

    print("\nMissing values per column:")
    for col, count in report["missing_values_per_column"].items():
        print(f"  - {col}: {count}")

    print("\nUnknown-like values per categorical column:")
    for col, count in report["unknown_like_counts_per_column"].items():
        print(f"  - {col}: {count}")

    print("\nCategorical cardinality (unique values):")
    for col, count in report["categorical_cardinality"].items():
        print(f"  - {col}: {count}")
    print("=========================================\n")


def create_high_impact_target(
    df,
    loss_column="Financial Loss (in Million $)",
    quantile=0.75,
    target_column="High Financial Impact",
):
    """
    Build the project target: a binary label for high financial impact.

    Incidents with financial loss greater than or equal to the selected quantile
    are labeled 1; all others are labeled 0.
    """
    if loss_column not in df.columns:
        raise ValueError(f"Loss column '{loss_column}' is missing from the dataframe.")

    df_target = df.copy()
    threshold = float(df_target[loss_column].quantile(quantile))
    df_target[target_column] = (df_target[loss_column] >= threshold).astype(int)

    report = {
        "target_column": target_column,
        "threshold_column": loss_column,
        "threshold_value": threshold,
        "quantile": quantile,
        "class_counts": df_target[target_column].value_counts().to_dict(),
        "class_ratio": df_target[target_column].value_counts(normalize=True).round(4).to_dict(),
    }
    return df_target, report


def print_high_impact_target_report(report):
    """Pretty-print target creation details for high-impact classification."""
    print("\n========== HIGH IMPACT TARGET REPORT ==========")
    print(f"Target column: {report['target_column']}")
    print(
        f"Threshold: {report['threshold_column']} >= {round(report['threshold_value'], 4)} "
        f"(quantile={report['quantile']})"
    )
    print(f"Class counts: {report['class_counts']}")
    print(f"Class ratios: {report['class_ratio']}")
    print("================================================\n")


def add_feature_engineering_columns(df, include_loss_based_features=True):
    """
    Create simple domain-driven features to improve signal for classification.
    """
    df_fe = df.copy()
    eps = 1e-6

    if include_loss_based_features and (
        "Financial Loss (in Million $)" in df_fe.columns
        and "Number of Affected Users" in df_fe.columns
    ):
        df_fe["Loss Per 1k Users"] = (
            df_fe["Financial Loss (in Million $)"] * 1000.0
        ) / (df_fe["Number of Affected Users"].replace(0, np.nan) + eps)

    if include_loss_based_features and (
        "Financial Loss (in Million $)" in df_fe.columns
        and "Incident Resolution Time (in Hours)" in df_fe.columns
    ):
        df_fe["Loss Per Resolution Hour"] = df_fe["Financial Loss (in Million $)"] / (
            df_fe["Incident Resolution Time (in Hours)"].replace(0, np.nan) + eps
        )

    if (
        "Number of Affected Users" in df_fe.columns
        and "Incident Resolution Time (in Hours)" in df_fe.columns
    ):
        df_fe["Users Per Resolution Hour"] = df_fe["Number of Affected Users"] / (
            df_fe["Incident Resolution Time (in Hours)"].replace(0, np.nan) + eps
        )

    if "Attack Source" in df_fe.columns:
        df_fe["Attack Source Is Unknown"] = (
            df_fe["Attack Source"].astype(str).str.lower().eq("unknown").astype(int)
        )

    if "Year" in df_fe.columns:
        df_fe["Year Bucket"] = pd.cut(
            df_fe["Year"],
            bins=[2014, 2018, 2021, 2024],
            labels=["2015-2018", "2019-2021", "2022-2024"],
            include_lowest=True,
        )

    return df_fe


def build_feature_preprocessor(X, scale_numeric=True):
    """Build a preprocessing pipeline for mixed numeric/categorical data."""
    numeric_columns = list(X.select_dtypes(include=["number"]).columns)
    categorical_columns = list(X.select_dtypes(exclude=["number"]).columns)

    if scale_numeric:
        numeric_steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    else:
        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )
