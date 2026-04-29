"""
Pipeline complet : preprocessing -> EDA -> training (avec ensembles + MLflow) -> SHAP.

Structure du run produit :
- figures EDA et confusion matrix    : ./<output_dir> (par defaut le dossier courant, soit code/)
- figures SHAP                       : ../graphs/06_shap_*.png et 07_*.png
- runs MLflow (params, metrics, ...) : ../mlruns/

Le script accepte d'etre lance depuis la racine du projet OU depuis code/.
Les chemins sont resolus a partir de l'emplacement du script.
"""

from pathlib import Path

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

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "dataset.csv"
GRAPHS_DIR = ROOT / "graphs"
MLRUNS_DIR = ROOT / "mlruns"
SAVED_MODELS_DIR = ROOT / "saved_models"


def main():
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)

    df = pd.read_csv(DATASET_PATH)

    df_clean, quality_report = preprocess_dataset(df, keep_unknown_as_category=True)
    print_data_quality_report(quality_report)

    df_impact, impact_report = create_high_impact_target(
        df_clean,
        loss_column=LOSS_COLUMN,
        quantile=0.75,
        target_column=PROJECT_TARGET_COLUMN,
    )
    print_high_impact_target_report(impact_report)

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

    model_results, tuned_estimators = train_and_compare_models(
        df_impact,
        target_column=PROJECT_TARGET_COLUMN,
        drop_columns=[LOSS_COLUMN],
        test_size=0.3,
        random_state=42,
        cv_splits=5,
        n_jobs=8,
        output_dir=".",
        use_feature_engineering=True,
        include_loss_based_features=False,
        include_baseline=True,
        include_ensembles=True,
        scoring="f1",
        mlflow_experiment="cybersecurity-high-impact",
        mlflow_tracking_uri=MLRUNS_DIR.as_uri(),
        return_estimators=True,
    )
    print_model_comparison(model_results)

    rf_pipeline = tuned_estimators.get("random_forest")
    if rf_pipeline is not None:
        import joblib
        from sklearn.model_selection import train_test_split

        from interpretability import run_shap_on_tree_pipeline
        from preprocess import add_feature_engineering_columns

        SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        saved_model_path = SAVED_MODELS_DIR / "best_model.joblib"
        joblib.dump(
            {
                "model": rf_pipeline,
                "feature_columns": list(rf_pipeline.feature_names_in_)
                                   if hasattr(rf_pipeline, "feature_names_in_") else None,
                "target_column": PROJECT_TARGET_COLUMN,
                "loss_column": LOSS_COLUMN,
                "model_name": "random_forest",
            },
            saved_model_path,
        )
        print(f"\nChampion model serialise : {saved_model_path}\n")

        df_for_shap = add_feature_engineering_columns(df_impact, include_loss_based_features=False)
        X_shap = df_for_shap.drop(columns=[PROJECT_TARGET_COLUMN, LOSS_COLUMN])
        y_shap = df_for_shap[PROJECT_TARGET_COLUMN]
        X_train_s, X_test_s, _, _ = train_test_split(
            X_shap, y_shap, test_size=0.3, random_state=42, stratify=y_shap
        )
        shap_report = run_shap_on_tree_pipeline(
            rf_pipeline,
            X_train_s,
            X_test_s,
            output_dir=str(GRAPHS_DIR),
        )
        print("\n========== SHAP REPORT (Random Forest) ==========")
        print(f"Features totales : {shap_report['n_features_total']}")
        print(f"Mean |SHAP| max  : {shap_report['max_mean_abs_shap']:.4f}")
        print(f"Top features (mean |SHAP|, classe 'High Financial Impact') :")
        for name, val in shap_report["top_features"][:10]:
            print(f"  - {name:<60}  {val:.4f}")
        print(f"Figures : {shap_report['bar_path']}")
        print(f"          {shap_report['summary_path']}")
        print("=================================================\n")


if __name__ == "__main__":
    main()
