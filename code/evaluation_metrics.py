import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import type_of_target


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
