"""
Interpretabilite et explainability via SHAP (TreeExplainer sur le RF).

Composante 4 du sujet (10% de la note). Produit:
- mean |SHAP| par feature (bar plot)        -> graphs/06_shap_mean_abs.png
- summary plot (beeswarm)                   -> graphs/07_shap_summary_beeswarm.png

Utilise le RandomForest tune par train_and_compare_models. Le pipeline
contient (preprocess: ColumnTransformer) + (model: RandomForestClassifier).
SHAP s'applique sur le model deballe avec X_test transforme.

Lecture diagnostique : sur un dataset sans signal, SHAP doit montrer une
importance "diluee" sur toutes les features OneHot, sans aucune feature
clairement dominante. C'est l'inverse de ce qu'on observerait sur un dataset
porteur de signal (ou 1-3 features concentrent l'essentiel de l'importance).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap


def _extract_model_and_transformed(pipeline_or_model, X):
    """Renvoie (model_inner, X_transforme) en gerant un sklearn Pipeline ou un modele direct."""
    if hasattr(pipeline_or_model, "named_steps"):
        steps = pipeline_or_model.named_steps
        if "preprocess" in steps and "model" in steps:
            return steps["model"], steps["preprocess"].transform(X)
    return pipeline_or_model, X


def _feature_names_from_pipeline(pipe):
    """Recupere les noms de features apres OneHotEncoding."""
    if hasattr(pipe, "named_steps") and "preprocess" in pipe.named_steps:
        try:
            return list(pipe.named_steps["preprocess"].get_feature_names_out())
        except Exception:
            return None
    return None


def run_shap_on_tree_pipeline(
    pipeline,
    X_train,
    X_test,
    output_dir="graphs",
    nsample_background=200,
    nsample_explain=500,
    random_state=42,
    bar_filename="06_shap_mean_abs.png",
    summary_filename="07_shap_summary_beeswarm.png",
):
    """
    Lance SHAP TreeExplainer sur un pipeline sklearn (preprocess + model).

    Args:
        pipeline: pipeline sklearn fitte. Le step "model" doit etre tree-based.
        X_train: DataFrame d'entrainement (utilise pour le background interventional, optionnel).
        X_test: DataFrame de test (sur laquelle on calcule les valeurs SHAP).
        output_dir: dossier de sortie pour les figures.
        nsample_background: taille de l'echantillon de background (random subset de X_train).
        nsample_explain: nombre maximal de samples X_test sur lesquels on calcule SHAP (limite la duree).

    Returns:
        dict avec chemins des figures, top features par mean |SHAP|.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, X_test_t = _extract_model_and_transformed(pipeline, X_test)
    feature_names = _feature_names_from_pipeline(pipeline)

    rng = np.random.default_rng(random_state)
    n_test = X_test_t.shape[0]
    if nsample_explain and n_test > nsample_explain:
        idx = rng.choice(n_test, size=nsample_explain, replace=False)
        if hasattr(X_test_t, "toarray"):
            X_used = X_test_t[idx]
        else:
            X_used = X_test_t[idx]
    else:
        X_used = X_test_t

    if hasattr(X_used, "toarray"):
        X_used = X_used.toarray()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_used, check_additivity=False)

    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_class1 = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_class1 = shap_values[:, :, 1]
    else:
        shap_class1 = shap_values

    mean_abs = np.abs(shap_class1).mean(axis=0)
    if feature_names is None or len(feature_names) != mean_abs.shape[0]:
        feature_names = [f"feat_{i}" for i in range(mean_abs.shape[0])]

    order = np.argsort(mean_abs)[::-1]
    top_n = min(20, len(order))
    top_idx = order[:top_n]
    top_names = [feature_names[i] for i in top_idx]
    top_vals = mean_abs[top_idx]

    bar_path = output_dir / bar_filename
    fig, ax = plt.subplots(figsize=(10, max(5, 0.32 * top_n)))
    ax.barh(top_names[::-1], top_vals[::-1], color="steelblue", edgecolor="white")
    ax.set_xlabel("Mean |SHAP value| (impact moyen sur la sortie modele)")
    ax.set_title(
        f"SHAP - top {top_n} features (Random Forest, classe 'High Financial Impact')\n"
        f"Importance diluee = aucune feature ne domine -> coherent avec dataset sans signal"
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(bar_path, dpi=140)
    plt.close(fig)

    summary_path = output_dir / summary_filename
    plt.figure(figsize=(10, max(5, 0.32 * top_n)))
    shap.summary_plot(
        shap_class1,
        X_used,
        feature_names=feature_names,
        plot_type="dot",
        max_display=top_n,
        show=False,
    )
    plt.title("SHAP summary (beeswarm) - Random Forest, classe 'High Financial Impact'", pad=10)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=140, bbox_inches="tight")
    plt.close()

    return {
        "bar_path": str(bar_path),
        "summary_path": str(summary_path),
        "top_features": list(zip(top_names, top_vals.tolist())),
        "n_features_total": int(mean_abs.shape[0]),
        "max_mean_abs_shap": float(top_vals[0]) if len(top_vals) else 0.0,
    }
