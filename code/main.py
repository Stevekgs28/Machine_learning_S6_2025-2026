import pandas as pd

from diagrams import (
    plot_correlation_heatmaps,
    plot_high_impact_class_balance,
    plot_high_impact_rate_by_category,
    plot_numeric_distributions,
)
from preprocess import (
    create_high_impact_target,
    preprocess_dataset,
    print_data_quality_report,
    print_high_impact_target_report,
)
from training_comparison import print_model_comparison, train_and_compare_models


PROJECT_TARGET_COLUMN = "High Financial Impact"
LOSS_COLUMN = "Financial Loss (in Million $)"


def main():
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)

    # To read the CSV with pandas
    df = pd.read_csv("../dataset.csv")

    # Global preprocessing + quality report
    df_clean, quality_report = preprocess_dataset(df, keep_unknown_as_category=True)
    print_data_quality_report(quality_report)

    # Official project target: predict whether an incident has high financial impact.
    df_impact, impact_report = create_high_impact_target(
        df_clean,
        loss_column=LOSS_COLUMN,
        quantile=0.75,
        target_column=PROJECT_TARGET_COLUMN,
    )
    print_high_impact_target_report(impact_report)

    # EDA focused on the high-financial-impact prediction objective.
    plot_high_impact_class_balance(df_impact, target_column=PROJECT_TARGET_COLUMN)
    plot_high_impact_rate_by_category(
        df_impact,
        category_column="Target Industry",
        target_column=PROJECT_TARGET_COLUMN,
        top_n=12,
        filename="high_impact_rate_by_target_industry.png",
    )
    plot_high_impact_rate_by_category(
        df_impact,
        category_column="Country",
        target_column=PROJECT_TARGET_COLUMN,
        top_n=10,
        filename="high_impact_rate_by_country.png",
    )
    plot_numeric_distributions(
        df_impact,
        numeric_columns=[
            "Year",
            "Number of Affected Users",
            "Incident Resolution Time (in Hours)",
        ],
    )
    plot_correlation_heatmaps(
        df_impact,
        numeric_columns=[
            "Year",
            "Number of Affected Users",
            "Incident Resolution Time (in Hours)",
            PROJECT_TARGET_COLUMN,
        ],
        categorical_columns=[
            "Attack Type",
            "Target Industry",
            "Security Vulnerability Type",
            "Attack Source",
            "Country",
        ],
    )

    # Train and compare candidate classifiers on the official project target.
    model_results = train_and_compare_models(
        df_impact,
        target_column=PROJECT_TARGET_COLUMN,
        drop_columns=[LOSS_COLUMN],
        test_size=0.3,
        random_state=42,
        cv_splits=5,
        n_jobs=10,
        output_dir=".",
        use_feature_engineering=True,
        include_loss_based_features=False,
        include_baseline=True,
        include_ensembles=False,
        scoring="f1",
    )
    print_model_comparison(model_results)


if __name__ == "__main__":
    main()
