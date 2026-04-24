import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


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


def _build_feature_preprocessor(X, scale_numeric=True):
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


def _summarize_fit_quality(train_f1, test_f1, cv_f1):
    """Heuristic summary to flag underfitting/overfitting risk."""
    train_test_gap = train_f1 - test_f1

    if train_f1 < 0.45 and test_f1 < 0.45:
        risk = "possible_underfitting"
    elif train_test_gap > 0.10:
        risk = "possible_overfitting"
    else:
        risk = "good_bias_variance_tradeoff"

    return {
        "risk_flag": risk,
        "train_minus_test_f1": round(train_test_gap, 4),
        "cv_minus_test_f1": round(cv_f1 - test_f1, 4),
    }


def _plot_confusion_matrix_for_model(best_model, X_test, y_test, title, filepath):
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_estimator(
        best_model,
        X_test,
        y_test,
        ax=ax,
        xticks_rotation=45,
        colorbar=True,
    )
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def _plot_model_metrics_barplot(results, filepath):
    labels = [name.replace("_", " ").title() for name in results.keys()]
    f1_vals = [results[k]["test_f1_macro"] for k in results]
    acc_vals = [results[k]["test_accuracy"] for k in results]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, f1_vals, width, label="F1-macro (test)")
    ax.bar(x + width / 2, acc_vals, width, label="Accuracy (test)")
    ax.set_ylabel("Score")
    ax.set_title("Model comparison on held-out test set")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def _plot_decision_tree_feature_importance(best_model, filepath, top_n=20):
    tree_model = best_model.named_steps["model"]
    preprocess = best_model.named_steps["preprocess"]
    feature_names = preprocess.get_feature_names_out()
    imps = tree_model.feature_importances_
    order = np.argsort(imps)[::-1][:top_n]
    top_names = feature_names[order]
    top_imps = imps[order]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_names[::-1], top_imps[::-1], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(f"Decision Tree — top {top_n} feature importances (test pipeline)")
    plt.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def _plot_decision_tree_structure_preview(best_model, filepath, max_depth=3):
    tree_model = best_model.named_steps["model"]
    preprocess = best_model.named_steps["preprocess"]
    feature_names = preprocess.get_feature_names_out()

    fig, ax = plt.subplots(figsize=(22, 12))
    plot_tree(
        tree_model,
        feature_names=feature_names,
        class_names=tree_model.classes_.astype(str),
        filled=True,
        max_depth=max_depth,
        fontsize=8,
        ax=ax,
    )
    ax.set_title(f"Decision Tree structure (depth ≤ {max_depth})")
    plt.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)


def train_and_compare_models(
    df,
    target_column="Attack Type",
    drop_columns=None,
    test_size=0.2,
    random_state=42,
    cv_splits=5,
    n_jobs=1,
    save_plots=True,
    output_dir=".",
):
    """
    Train and compare Logistic Regression vs Decision Tree with CV tuning.

    Returns:
        dict: per-model metrics, best params, and over/underfitting risk hints.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing from the dataframe.")

    drop_columns = drop_columns or []
    blocked_columns = {target_column, *drop_columns}

    work_df = df.dropna(subset=[target_column]).copy()
    X = work_df.drop(columns=[col for col in blocked_columns if col in work_df.columns])
    y = work_df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    logistic_pipeline = Pipeline(
        steps=[
            ("preprocess", _build_feature_preprocessor(X_train, scale_numeric=True)),
            (
                "model",
                LogisticRegression(
                    max_iter=4000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    logistic_grid = {
        "model__C": [0.01, 0.1, 1, 5, 10, 20],
        "model__solver": ["lbfgs"],
    }

    tree_pipeline = Pipeline(
        steps=[
            ("preprocess", _build_feature_preprocessor(X_train, scale_numeric=False)),
            ("model", DecisionTreeClassifier(class_weight="balanced", random_state=random_state)),
        ]
    )
    tree_grid = {
        "model__max_depth": [3, 5, 8, 12, 16],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 3, 5, 10],
        "model__ccp_alpha": [0.0, 0.0005, 0.001, 0.005],
    }

    model_specs = {
        "logistic_regression": (logistic_pipeline, logistic_grid),
        "decision_tree": (tree_pipeline, tree_grid),
    }

    if save_plots:
        os.makedirs(output_dir, exist_ok=True)

    results = {}
    for model_name, (pipeline, param_grid) in model_specs.items():
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring="f1_macro",
            n_jobs=n_jobs,
            refit=True,
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        train_f1 = f1_score(y_train, y_train_pred, average="macro")
        test_f1 = f1_score(y_test, y_test_pred, average="macro")
        cv_f1 = search.best_score_

        plot_paths = []
        if save_plots:
            cm_path = os.path.join(output_dir, f"confusion_matrix_{model_name}.png")
            display_name = model_name.replace("_", " ").title()
            _plot_confusion_matrix_for_model(
                best_model,
                X_test,
                y_test,
                title=f"Confusion matrix — {display_name}",
                filepath=cm_path,
            )
            plot_paths.append(cm_path)

            if model_name == "decision_tree":
                imp_path = os.path.join(output_dir, "decision_tree_feature_importance_top20.png")
                tree_path = os.path.join(output_dir, "decision_tree_structure_preview.png")
                _plot_decision_tree_feature_importance(best_model, imp_path, top_n=20)
                _plot_decision_tree_structure_preview(best_model, tree_path, max_depth=3)
                plot_paths.extend([imp_path, tree_path])

        results[model_name] = {
            "best_params": search.best_params_,
            "best_cv_f1_macro": round(cv_f1, 4),
            "train_f1_macro": round(train_f1, 4),
            "test_f1_macro": round(test_f1, 4),
            "test_accuracy": round(accuracy_score(y_test, y_test_pred), 4),
            "fit_quality": _summarize_fit_quality(train_f1, test_f1, cv_f1),
            "classification_report": classification_report(y_test, y_test_pred, zero_division=0),
            "plot_paths": plot_paths,
        }

    if save_plots and results:
        comparison_path = os.path.join(output_dir, "model_comparison_test_metrics.png")
        _plot_model_metrics_barplot(results, comparison_path)
        results["_comparison_plot"] = comparison_path

    return results


def print_model_comparison(results):
    """Pretty-print model comparison output from train_and_compare_models."""
    print("\n========== MODEL COMPARISON ==========")
    for model_name, payload in results.items():
        if model_name.startswith("_") or not isinstance(payload, dict):
            continue
        print(f"\nModel: {model_name}")
        print(f"  Best params: {payload['best_params']}")
        print(f"  Best CV F1-macro: {payload['best_cv_f1_macro']}")
        print(f"  Train F1-macro:   {payload['train_f1_macro']}")
        print(f"  Test F1-macro:    {payload['test_f1_macro']}")
        print(f"  Test accuracy:    {payload['test_accuracy']}")

        fit_quality = payload["fit_quality"]
        print(f"  Fit quality flag: {fit_quality['risk_flag']}")
        print(f"  Train-Test gap:   {fit_quality['train_minus_test_f1']}")
        print(f"  CV-Test gap:      {fit_quality['cv_minus_test_f1']}")

        print("\n  Classification report:")
        print(payload["classification_report"])

        plot_paths = payload.get("plot_paths") or []
        if plot_paths:
            print("  Saved plots:")
            for path in plot_paths:
                print(f"    - {path}")

    if "_comparison_plot" in results:
        print(f"\nComparison bar chart: {results['_comparison_plot']}")

    print("======================================\n")

