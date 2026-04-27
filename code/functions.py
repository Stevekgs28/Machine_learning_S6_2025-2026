import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from preprocess import DEFAULT_NUMERIC_COLUMNS


def print_data(df):
    """Display rows for quick debugging."""
    for _, row in df.iterrows():
        print(row)
        print("", end="\n\n")


def _safe_save_show(filename):
    plt.tight_layout()
    plt.savefig(filename)
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
        axes[i][0].set_yscale("log", base=10)

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


def plot_high_impact_class_balance(df, target_column="High Financial Impact"):
    """Plot class distribution for the high-impact target."""
    if target_column not in df.columns:
        return

    counts = df[target_column].value_counts().sort_index()
    labels = ["Low/Medium Impact", "High Impact"]
    mapped_labels = [labels[int(idx)] if int(idx) in (0, 1) else str(idx) for idx in counts.index]

    plt.figure(figsize=(7, 5))
    plt.bar(mapped_labels, counts.values, color=["#4C78A8", "#F58518"])
    plt.title("Class Balance — High Financial Impact")
    plt.ylabel("Count")
    plt.xlabel("Target class")
    _safe_save_show("high_impact_class_balance.png")


def plot_high_impact_rate_by_category(
    df,
    category_column,
    target_column="High Financial Impact",
    top_n=12,
    filename=None,
):
    """Plot high-impact rate per category for the top frequent groups."""
    if category_column not in df.columns or target_column not in df.columns:
        return

    top_categories = df[category_column].value_counts().head(top_n).index
    subset = df[df[category_column].isin(top_categories)]

    rate = (
        subset.groupby(category_column, dropna=False)[target_column]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(12, 6))
    rate.plot(kind="bar")
    plt.title(f"High-Impact Rate by {category_column} (Top {top_n})")
    plt.ylabel("Rate")
    plt.xlabel(category_column)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    _safe_save_show(filename or f"high_impact_rate_by_{category_column.lower().replace(' ', '_')}.png")
