import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from preprocess import add_feature_engineering_columns, build_feature_preprocessor
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.utils.multiclass import type_of_target


def _summarize_fit_quality(train_f1, test_f1, cv_f1=None):
    """Heuristic summary to flag underfitting/overfitting risk."""
    train_test_gap = train_f1 - test_f1
    cv_for_gap = test_f1 if cv_f1 is None else cv_f1

    if train_test_gap > 0.10:
        risk = "possible_overfitting"
    elif train_f1 < 0.45 and test_f1 < 0.45:
        risk = "possible_underfitting"
    else:
        risk = "good_bias_variance_tradeoff"

    return {
        "risk_flag": risk,
        "train_minus_test_f1": round(train_test_gap, 4),
        "cv_minus_test_f1": round(cv_for_gap - test_f1, 4),
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
    keys = [
        k
        for k, v in results.items()
        if isinstance(v, dict) and "test_f1_macro" in v and not k.startswith("_")
    ]
    labels = [name.replace("_", " ").title() for name in keys]
    f1_vals = [results[k]["test_f1_macro"] for k in keys]
    acc_vals = [results[k]["test_accuracy"] for k in keys]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(labels)), 5))
    ax.bar(x - width / 2, f1_vals, width, label="F1-macro (test)")
    ax.bar(x + width / 2, acc_vals, width, label="Accuracy (test)")
    ax.set_ylabel("Score")
    ax.set_title("Model comparison on held-out test set")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
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


def compute_classification_metrics(model, X_test, y_test):
    """
    Compute a standard set of classification metrics.

    Returns:
        dict: metrics including macro precision/recall/F1, accuracy, ROC-AUC (OvR macro if possible).
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
    }

    # ROC-AUC: only if probabilities are available and the task is suitable.
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            classes = getattr(model, "classes_", None)
            if classes is None and hasattr(model, "named_steps") and "model" in model.named_steps:
                classes = getattr(model.named_steps["model"], "classes_", None)

            # Binary or multiclass classification
            if classes is not None:
                y_type = type_of_target(y_test)
                if y_type == "binary":
                    pos_scores = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
                    metrics["roc_auc_ovr_macro"] = float(roc_auc_score(y_test, pos_scores))
                elif y_type == "multiclass":
                    y_bin = label_binarize(y_test, classes=classes)
                    metrics["roc_auc_ovr_macro"] = float(
                        roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")
                    )
    except Exception:
        metrics["roc_auc_ovr_macro"] = None

    return metrics


def plot_multiclass_roc_auc(model, X_test, y_test, filepath, title):
    """
    Save a multiclass ROC curve plot (OvR). For binary, saves a single ROC curve.
    """
    if not hasattr(model, "predict_proba"):
        return False

    proba = model.predict_proba(X_test)
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps") and "model" in model.named_steps:
        classes = getattr(model.named_steps["model"], "classes_", None)
    if classes is None:
        return False

    y_type = type_of_target(y_test)
    if y_type not in {"binary", "multiclass"}:
        return False

    y_bin = label_binarize(y_test, classes=classes)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)

    # Binary case: label_binarize returns shape (n_samples, 1)
    if y_type == "binary" or y_bin.shape[1] == 1:
        # For binary, sklearn expects scores for the positive class
        pos_scores = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        from sklearn.metrics import RocCurveDisplay

        RocCurveDisplay.from_predictions(y_test, pos_scores, ax=ax, name="ROC")
    else:
        from sklearn.metrics import RocCurveDisplay

        for i, cls in enumerate(classes):
            RocCurveDisplay.from_predictions(
                y_bin[:, i],
                proba[:, i],
                ax=ax,
                name=str(cls),
            )

    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)
    return True


def compute_regression_metrics(y_true, y_pred):
    """Compute regression metrics requested in the project: R², MAE, MSE (+ RMSE)."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(1.0 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)))
        if np.sum((y_true - np.mean(y_true)) ** 2) != 0
        else None,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
    }


def train_and_compare_models(
    df,
    target_column="High Financial Impact",
    drop_columns=None,
    test_size=0.3,
    random_state=42,
    cv_splits=5,
    n_jobs=1,
    save_plots=True,
    output_dir=".",
    use_feature_engineering=True,
    include_loss_based_features=True,
    include_baseline=True,
    include_ensembles=False,
    scoring="f1_macro",
):
    """
    Train and compare candidate classifiers with cross-validation tuning.

    The current project uses this for the "High Financial Impact" target created
    from financial loss. The function remains generic enough to support other
    classification targets when passed explicitly.

    Returns:
        dict: per-model metrics, best params, over/underfitting hints, and plot paths.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing from the dataframe.")

    drop_columns = drop_columns or []
    blocked_columns = {target_column, *drop_columns}

    work_df = df.dropna(subset=[target_column]).copy()
    if use_feature_engineering:
        work_df = add_feature_engineering_columns(
            work_df, include_loss_based_features=include_loss_based_features
        )
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
            ("preprocess", build_feature_preprocessor(X_train, scale_numeric=True)),
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
            ("preprocess", build_feature_preprocessor(X_train, scale_numeric=False)),
            ("model", DecisionTreeClassifier(class_weight="balanced", random_state=random_state)),
        ]
    )
    tree_grid = {
        "model__max_depth": [3, 5, 8, 12, 16],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 3, 5, 10],
        "model__ccp_alpha": [0.0, 0.0005, 0.001, 0.005],
    }

    rf_pipeline = Pipeline(
        steps=[
            ("preprocess", build_feature_preprocessor(X_train, scale_numeric=False)),
            ("model", RandomForestClassifier(class_weight="balanced", random_state=random_state)),
        ]
    )
    rf_grid = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }

    model_specs = {
        "logistic_regression": (logistic_pipeline, logistic_grid),
        "decision_tree": (tree_pipeline, tree_grid),
        "random_forest": (rf_pipeline, rf_grid),
    }

    if save_plots:
        os.makedirs(output_dir, exist_ok=True)

    results = {}
    tuned_estimators = {}

    if include_baseline:
        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(X_train, y_train)
        y_train_pred_b = baseline.predict(X_train)
        y_test_pred_b = baseline.predict(X_test)
        baseline_train_f1 = f1_score(y_train, y_train_pred_b, average="macro")
        baseline_test_f1 = f1_score(y_test, y_test_pred_b, average="macro")

        plot_paths_b = []
        if save_plots:
            cm_b = os.path.join(output_dir, "confusion_matrix_baseline_most_frequent.png")
            _plot_confusion_matrix_for_model(
                baseline,
                X_test,
                y_test,
                title="Confusion matrix — Baseline (most frequent)",
                filepath=cm_b,
            )
            plot_paths_b.append(cm_b)

        results["baseline_most_frequent"] = {
            "best_params": {"strategy": "most_frequent"},
            "best_cv_f1_macro": None,
            "train_f1_macro": round(baseline_train_f1, 4),
            "test_f1_macro": round(baseline_test_f1, 4),
            "test_accuracy": round(accuracy_score(y_test, y_test_pred_b), 4),
            "fit_quality": _summarize_fit_quality(baseline_train_f1, baseline_test_f1, None),
            "classification_report": classification_report(y_test, y_test_pred_b, zero_division=0),
            "evaluation_metrics": compute_classification_metrics(baseline, X_test, y_test),
            "plot_paths": plot_paths_b,
        }

    for model_name, (pipeline, param_grid) in model_specs.items():
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=True,
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        tuned_estimators[model_name] = best_model

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

            if model_name in ["decision_tree", "random_forest"]:
                imp_path = os.path.join(output_dir, f"{model_name}_feature_importance_top20.png")
                _plot_decision_tree_feature_importance(best_model, imp_path, top_n=20)
                plot_paths.append(imp_path)

                if model_name == "decision_tree":
                    tree_path = os.path.join(output_dir, "decision_tree_structure_preview.png")
                    _plot_decision_tree_structure_preview(best_model, tree_path, max_depth=3)
                    plot_paths.append(tree_path)

            roc_path = os.path.join(output_dir, f"roc_{model_name}.png")
            if plot_multiclass_roc_auc(
                best_model,
                X_test,
                y_test,
                filepath=roc_path,
                title=f"ROC (OvR) — {display_name}",
            ):
                plot_paths.append(roc_path)

        eval_metrics = compute_classification_metrics(best_model, X_test, y_test)

        results[model_name] = {
            "best_params": search.best_params_,
            "best_cv_f1_macro": round(cv_f1, 4),
            "train_f1_macro": round(train_f1, 4),
            "test_f1_macro": round(test_f1, 4),
            "test_accuracy": round(accuracy_score(y_test, y_test_pred), 4),
            "fit_quality": _summarize_fit_quality(train_f1, test_f1, cv_f1),
            "classification_report": classification_report(y_test, y_test_pred, zero_division=0),
            "evaluation_metrics": eval_metrics,
            "plot_paths": plot_paths,
        }

    if include_ensembles:
        best_lr = tuned_estimators["logistic_regression"]
        best_dt = tuned_estimators["decision_tree"]

        voting = VotingClassifier(
            estimators=[
                ("logistic_regression", clone(best_lr)),
                ("decision_tree", clone(best_dt)),
            ],
            voting="soft",
            n_jobs=n_jobs,
        )
        voting.fit(X_train, y_train)
        y_train_pred_v = voting.predict(X_train)
        y_test_pred_v = voting.predict(X_test)
        train_f1_v = f1_score(y_train, y_train_pred_v, average="macro")
        test_f1_v = f1_score(y_test, y_test_pred_v, average="macro")

        plot_paths_v = []
        if save_plots:
            cm_v = os.path.join(output_dir, "confusion_matrix_voting_soft.png")
            _plot_confusion_matrix_for_model(
                voting,
                X_test,
                y_test,
                title="Confusion matrix — Voting (soft, LR + DT)",
                filepath=cm_v,
            )
            plot_paths_v.append(cm_v)

            roc_v = os.path.join(output_dir, "roc_voting_soft.png")
            if plot_multiclass_roc_auc(
                voting,
                X_test,
                y_test,
                filepath=roc_v,
                title="ROC (OvR) — Voting (soft, LR + DT)",
            ):
                plot_paths_v.append(roc_v)

        eval_metrics_v = compute_classification_metrics(voting, X_test, y_test)

        results["voting_soft"] = {
            "best_params": {
                "ensemble": "VotingClassifier",
                "voting": "soft",
                "estimators": [
                    "tuned logistic_regression",
                    "tuned decision_tree",
                ],
            },
            "best_cv_f1_macro": None,
            "train_f1_macro": round(train_f1_v, 4),
            "test_f1_macro": round(test_f1_v, 4),
            "test_accuracy": round(accuracy_score(y_test, y_test_pred_v), 4),
            "fit_quality": _summarize_fit_quality(train_f1_v, test_f1_v, None),
            "classification_report": classification_report(y_test, y_test_pred_v, zero_division=0),
            "evaluation_metrics": eval_metrics_v,
            "plot_paths": plot_paths_v,
        }

        stacking = StackingClassifier(
            estimators=[
                ("logistic_regression", clone(best_lr)),
                ("decision_tree", clone(best_dt)),
            ],
            final_estimator=LogisticRegression(
                max_iter=4000,
                class_weight="balanced",
                random_state=random_state,
            ),
            cv=cv,
            stack_method="predict_proba",
            passthrough=False,
            n_jobs=n_jobs,
        )
        stacking.fit(X_train, y_train)
        y_train_pred_s = stacking.predict(X_train)
        y_test_pred_s = stacking.predict(X_test)
        train_f1_s = f1_score(y_train, y_train_pred_s, average="macro")
        test_f1_s = f1_score(y_test, y_test_pred_s, average="macro")

        plot_paths_s = []
        if save_plots:
            cm_s = os.path.join(output_dir, "confusion_matrix_stacking.png")
            _plot_confusion_matrix_for_model(
                stacking,
                X_test,
                y_test,
                title="Confusion matrix — Stacking (LR + DT → meta LR)",
                filepath=cm_s,
            )
            plot_paths_s.append(cm_s)

            roc_s = os.path.join(output_dir, "roc_stacking.png")
            if plot_multiclass_roc_auc(
                stacking,
                X_test,
                y_test,
                filepath=roc_s,
                title="ROC (OvR) — Stacking (LR + DT → meta LR)",
            ):
                plot_paths_s.append(roc_s)

        eval_metrics_s = compute_classification_metrics(stacking, X_test, y_test)

        results["stacking"] = {
            "best_params": {
                "ensemble": "StackingClassifier",
                "base_estimators": [
                    "tuned logistic_regression",
                    "tuned decision_tree",
                ],
                "final_estimator": "LogisticRegression(class_weight='balanced')",
                "cv": f"StratifiedKFold(n_splits={cv_splits})",
                "stack_method": "predict_proba",
            },
            "best_cv_f1_macro": None,
            "train_f1_macro": round(train_f1_s, 4),
            "test_f1_macro": round(test_f1_s, 4),
            "test_accuracy": round(accuracy_score(y_test, y_test_pred_s), 4),
            "fit_quality": _summarize_fit_quality(train_f1_s, test_f1_s, None),
            "classification_report": classification_report(y_test, y_test_pred_s, zero_division=0),
            "evaluation_metrics": eval_metrics_s,
            "plot_paths": plot_paths_s,
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
        cv_macro = payload.get("best_cv_f1_macro")
        cv_display = cv_macro if cv_macro is not None else "N/A (ensemble; no GridSearchCV on ensemble)"
        print(f"  Best CV F1-macro: {cv_display}")
        print(f"  Train F1-macro:   {payload['train_f1_macro']}")
        print(f"  Test F1-macro:    {payload['test_f1_macro']}")
        print(f"  Test accuracy:    {payload['test_accuracy']}")

        fit_quality = payload["fit_quality"]
        print(f"  Fit quality flag: {fit_quality['risk_flag']}")
        print(f"  Train-Test gap:   {fit_quality['train_minus_test_f1']}")
        print(f"  CV-Test gap:      {fit_quality['cv_minus_test_f1']}")

        metrics = payload.get("evaluation_metrics") or {}
        if metrics:
            print("  Evaluation metrics:")
            print(f"    - Accuracy:        {round(metrics.get('accuracy', 0.0), 4)}")
            print(f"    - Precision macro: {round(metrics.get('precision_macro', 0.0), 4)}")
            print(f"    - Recall macro:    {round(metrics.get('recall_macro', 0.0), 4)}")
            print(f"    - F1 macro:        {round(metrics.get('f1_macro', 0.0), 4)}")
            roc_auc = metrics.get("roc_auc_ovr_macro")
            if roc_auc is not None:
                print(f"    - ROC-AUC OvR:     {round(roc_auc, 4)}")

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
