import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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


def print_data(df):
    """Display rows for quick debugging."""
    for _, row in df.iterrows():
        print(row)
        print("", end="\n\n")


def _safe_save_show(filename):
    plt.tight_layout()
    plt.savefig(filename)
    backend = plt.get_backend().lower()
    if "agg" not in backend:
        plt.show()
    plt.close()


def plot_top_categories(df, column, top_n=15, log_scale=False, filename=None, title=None):
    """Plot a top-N bar chart for a categorical feature."""
    if column not in df.columns:
        return

    counts = df[column].value_counts(dropna=False).head(top_n)
    plt.figure(figsize=(12, 6))
    counts.plot(kind="bar")
    plt.title(title or f"Top {top_n} categories - {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    if log_scale:
        plt.yscale("log", base=10)
    _safe_save_show(filename or f"histogram_{column.lower().replace(' ', '_')}.png")


def histogram_attack_per_country(df):
    plot_top_categories(
        df,
        "Country",
        top_n=20,
        log_scale=True,
        filename="histogram_attack_per_country.png",
        title="Number of Cyber Attacks per Country (Top 20)",
    )


def histogram_attack_per_industry(df):
    plot_top_categories(
        df,
        "Target Industry",
        top_n=20,
        log_scale=True,
        filename="histogram_attack_per_industry.png",
        title="Number of Cyber Attacks per Industry (Top 20)",
    )


def histogram_financial_loss(df):
    """Plot total financial loss by country."""
    if "Country" not in df.columns or "Financial Loss (in Million $)" not in df.columns:
        return

    loss_per_country = (
        df.groupby("Country", dropna=False)["Financial Loss (in Million $)"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
    )
    plt.figure(figsize=(12, 6))
    loss_per_country.plot(kind="bar")
    plt.title("Total Financial Loss per Country (Top 20)")
    plt.xlabel("Country")
    plt.ylabel("Financial Loss (in Million $)")
    plt.xticks(rotation=45, ha="right")
    plt.yscale("log", base=10)
    _safe_save_show("histogram_financial_loss.png")


def plot_numeric_distributions(df, numeric_columns=None, bins=30):
    """Create histogram and boxplot for each numeric feature."""
    if numeric_columns is None:
        numeric_columns = list(df.select_dtypes(include=["number"]).columns)
    else:
        numeric_columns = [col for col in numeric_columns if col in df.columns]

    if not numeric_columns:
        return

    n = len(numeric_columns)
    fig, axes = plt.subplots(n, 2, figsize=(12, max(4, 3 * n)))
    if n == 1:
        axes = [axes]

    for i, col in enumerate(numeric_columns):
        sns.histplot(df[col].dropna(), bins=bins, kde=False, ax=axes[i][0])
        axes[i][0].set_title(f"Histogram - {col}")
        axes[i][0].set_xlabel(col)

        sns.boxplot(y=df[col], ax=axes[i][1])
        axes[i][1].set_title(f"Boxplot - {col}")
        axes[i][1].set_ylabel(col)

    fig.tight_layout()
    fig.savefig("numeric_distributions.png")
    backend = plt.get_backend().lower()
    if "agg" not in backend:
        plt.show()
    plt.close(fig)


def _keep_top_categories(series, top_n):
    """Keep top-N categories and group others as 'Other'."""
    if top_n is None or top_n <= 0:
        return series
    top_values = series.value_counts(dropna=False).head(top_n).index
    return series.where(series.isin(top_values), other="Other")


def _plot_correlation_heatmap(corr_df, title, filename, figsize=(12, 10)):
    """Render and save a single correlation heatmap."""
    if corr_df.empty:
        return

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_df,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        square=False,
        linewidths=0.2,
        cbar=True,
    )
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    _safe_save_show(filename)


def plot_correlation_heatmaps(
    df,
    numeric_columns=None,
    categorical_columns=None,
    top_n_country=10,
    top_n_other_categories=8,
):
    """
    Generate correlation heatmaps:
    - Numeric only: Pearson + Spearman
    - Enriched (numeric + encoded categories): Pearson + Spearman
    """
    if numeric_columns is None:
        numeric_columns = [
            col
            for col in DEFAULT_NUMERIC_COLUMNS
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
        ]
    else:
        numeric_columns = [
            col
            for col in numeric_columns
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
        ]

    if len(numeric_columns) < 2:
        return

    # Numeric-only heatmaps
    numeric_df = df[numeric_columns]
    pearson_numeric = numeric_df.corr(method="pearson")
    spearman_numeric = numeric_df.corr(method="spearman")
    _plot_correlation_heatmap(
        pearson_numeric,
        "Correlation Heatmap (Pearson) - Numeric Features",
        "heatmap_pearson.png",
    )
    _plot_correlation_heatmap(
        spearman_numeric,
        "Correlation Heatmap (Spearman) - Numeric Features",
        "heatmap_spearman.png",
    )

    # Enriched heatmaps with selected categorical features
    if categorical_columns is None:
        categorical_columns = [
            "Attack Type",
            "Target Industry",
            "Security Vulnerability Type",
            "Attack Source",
            "Country",
        ]

    categorical_columns = [col for col in categorical_columns if col in df.columns]
    if not categorical_columns:
        return

    cat_df = df[categorical_columns].copy()
    for col in categorical_columns:
        if col == "Country":
            cat_df[col] = _keep_top_categories(cat_df[col], top_n_country)
        else:
            cat_df[col] = _keep_top_categories(cat_df[col], top_n_other_categories)

    encoded_cat = pd.get_dummies(
        cat_df,
        columns=categorical_columns,
        drop_first=False,
        dummy_na=False,
    )
    enriched_df = pd.concat([numeric_df, encoded_cat], axis=1)

    pearson_enriched = enriched_df.corr(method="pearson")
    spearman_enriched = enriched_df.corr(method="spearman")
    _plot_correlation_heatmap(
        pearson_enriched,
        "Correlation Heatmap (Pearson) - Numeric + Encoded Categories",
        "heatmap_pearson_enriched.png",
        figsize=(16, 14),
    )
    _plot_correlation_heatmap(
        spearman_enriched,
        "Correlation Heatmap (Spearman) - Numeric + Encoded Categories",
        "heatmap_spearman_enriched.png",
        figsize=(16, 14),
    )


def boxplots_analysis(df):
    """Backward-compatible wrapper for numeric boxplots."""
    plot_numeric_distributions(
        df,
        numeric_columns=[
            "Number of Affected Users",
            "Incident Resolution Time (in Hours)",
        ],
    )

