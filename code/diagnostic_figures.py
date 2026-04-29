"""
Figures de support du diagnostic de signal pour le rapport.

Produit dans ../graphs/ :
- 01_loss_distribution.png       : histogramme + KDE de Financial Loss avec PDF uniforme overlay
- 02_mutual_info_numeric.png     : barplot des MI feature/target sur les numeriques
- 03_cramers_v_categorical.png   : barplot de Cramer's V avec seuil 0.10
- 04_pos_rate_top_categories.png : pos_rate par modalite (top-3 cat. features) + CI 95% + baseline
- 05_train_vs_test_overfit.png   : F1 macro train vs test pour chaque modele

Toutes les valeurs numeriques sont les memes que celles imprimees par
diagnostic_signal.py. Les figures servent uniquement a visualiser ces resultats
pour le rapport LaTeX.
"""

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif


HERE = Path(__file__).resolve().parent
DATASET_PATH = HERE.parent / "dataset.csv"
GRAPHS_DIR = HERE.parent / "graphs"
LOSS_COLUMN = "Financial Loss (in Million $)"
TARGET_COLUMN = "High Financial Impact"
QUANTILE = 0.75

# Resultats observes par main.py / resultat_shell.txt (test set 900 echantillons)
TRAIN_TEST_F1_MACRO = {
    "Baseline\n(most_frequent)": (0.4284, 0.4286),
    "Logistic\nRegression":      (0.5248, 0.4775),
    "Decision\nTree":            (0.6509, 0.4828),
    "Random\nForest":            (0.9095, 0.4914),
}


def cramers_v(contingency):
    chi2 = stats.chi2_contingency(contingency, correction=False).statistic
    n = contingency.values.sum()
    r, k = contingency.shape
    denom = n * (min(r, k) - 1)
    return math.sqrt(chi2 / denom) if denom > 0 else float("nan")


def wald_ci(p, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    se = math.sqrt(max(p * (1 - p), 0.0) / n)
    return max(0.0, p - z * se), min(1.0, p + z * se)


def fig_loss_distribution(df, out_path):
    loss = df[LOSS_COLUMN].astype(float).values
    lo, hi = loss.min(), loss.max()
    ks = stats.kstest(loss, "uniform", args=(lo, hi - lo))

    fig, ax = plt.subplots(figsize=(9, 5))
    counts, bins, _ = ax.hist(
        loss, bins=40, density=True, alpha=0.55,
        color="steelblue", edgecolor="white", label="Financial Loss observe",
    )
    # PDF uniforme theorique sur [lo, hi]
    uniform_pdf = 1.0 / (hi - lo)
    ax.hlines(uniform_pdf, lo, hi, colors="crimson", linewidth=2.2,
              label=f"Uniforme [{lo:.2f}, {hi:.2f}] (PDF = 1/{hi - lo:.2f})")
    ax.set_xlabel("Financial Loss (in Million $)")
    ax.set_ylabel("Densite")
    ax.set_title(
        f"Distribution de Financial Loss\n"
        f"KS-test vs uniforme : D = {ks.statistic:.4f}, p = {ks.pvalue:.3f} "
        f"(impossible de rejeter l'hypothese uniforme)"
    )
    ax.legend(loc="lower center")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved {out_path.name}")


def fig_mi_numeric(df, out_path):
    numeric_cols = [c for c in df.select_dtypes(include="number").columns
                    if c not in {TARGET_COLUMN, LOSS_COLUMN}]
    mi = mutual_info_classif(df[numeric_cols].values, df[TARGET_COLUMN].values, random_state=42)
    order = np.argsort(mi)
    names = [numeric_cols[i] for i in order]
    values = mi[order]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars = ax.barh(names, values, color="steelblue", edgecolor="white")
    ax.axvline(0.02, color="orange", linestyle="--", linewidth=1.4, label="seuil 'faible' (0.02)")
    ax.axvline(0.05, color="crimson", linestyle="--", linewidth=1.4, label="seuil 'modere' (0.05)")
    for bar, v in zip(bars, values):
        ax.text(max(v, 0.0005) + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=9)
    ax.set_xlabel("Mutual information avec la cible")
    ax.set_title("Mutual information feature/target sur les colonnes numeriques\n"
                 "Toutes les valeurs sont quasiment nulles")
    ax.set_xlim(0, max(0.05, values.max() * 1.5))
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved {out_path.name}")


def fig_cramers_v(df, out_path):
    cat_cols = list(df.select_dtypes(include="object").columns)
    rows = []
    for col in cat_cols:
        contingency = pd.crosstab(df[col], df[TARGET_COLUMN])
        rows.append({
            "feature": col,
            "cramers_v": cramers_v(contingency),
            "chi2_pvalue": stats.chi2_contingency(contingency, correction=False).pvalue,
        })
    cv = pd.DataFrame(rows).sort_values("cramers_v")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(cv["feature"], cv["cramers_v"], color="steelblue", edgecolor="white")
    ax.axvline(0.10, color="orange", linestyle="--", linewidth=1.4, label="seuil 'faible' (0.10)")
    ax.axvline(0.30, color="crimson", linestyle="--", linewidth=1.4, label="seuil 'modere' (0.30)")
    for bar, v, p in zip(bars, cv["cramers_v"], cv["chi2_pvalue"]):
        ax.text(v + 0.003, bar.get_y() + bar.get_height() / 2,
                f"V = {v:.3f} | chi2 p = {p:.2f}", va="center", fontsize=9)
    ax.set_xlabel("Cramer's V")
    ax.set_title("Association categorielle/cible (Cramer's V + chi2 p-value)\n"
                 "Aucune feature ne franchit le seuil 'faible' de 0.10 et toutes les p-values sont > 0.30")
    ax.set_xlim(0, max(0.15, cv["cramers_v"].max() * 1.4))
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved {out_path.name}")


def fig_pos_rate_top_categories(df, out_path):
    cat_cols = list(df.select_dtypes(include="object").columns)
    cv_rows = []
    for col in cat_cols:
        contingency = pd.crosstab(df[col], df[TARGET_COLUMN])
        cv_rows.append((col, cramers_v(contingency)))
    cv_rows.sort(key=lambda r: r[1], reverse=True)
    top3 = [c for c, _ in cv_rows[:3]]
    base = df[TARGET_COLUMN].mean()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for ax, col in zip(axes, top3):
        grouped = (
            df.groupby(col)[TARGET_COLUMN]
            .agg(["count", "mean"])
            .rename(columns={"count": "n", "mean": "pos_rate"})
            .sort_values("pos_rate", ascending=True)
        )
        cis = grouped.apply(
            lambda r: pd.Series(wald_ci(r["pos_rate"], int(r["n"])), index=["lo", "hi"]),
            axis=1,
        )
        out = pd.concat([grouped, cis], axis=1)
        y = np.arange(len(out))
        rates = out["pos_rate"].values
        err_lo = rates - out["lo"].values
        err_hi = out["hi"].values - rates
        ax.errorbar(rates, y, xerr=[err_lo, err_hi], fmt="o", color="steelblue",
                    ecolor="lightsteelblue", elinewidth=2.5, capsize=4, markersize=7)
        ax.axvline(base, color="crimson", linestyle="--", linewidth=1.6,
                   label=f"baseline {base:.3f}")
        ax.set_yticks(y)
        ax.set_yticklabels(out.index)
        ax.set_xlim(0.10, 0.40)
        ax.set_xlabel("pos_rate (avec IC 95%)")
        ax.set_title(col)
        ax.grid(axis="x", alpha=0.25)
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle(
        "Taux de classe 1 par modalite, top-3 features categorielles\n"
        "Tous les IC 95% chevauchent la baseline 0.25 -> aucune modalite n'apporte de discrimination significative",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved {out_path.name}")


def fig_train_vs_test_overfit(out_path):
    models = list(TRAIN_TEST_F1_MACRO.keys())
    train_vals = [TRAIN_TEST_F1_MACRO[m][0] for m in models]
    test_vals = [TRAIN_TEST_F1_MACRO[m][1] for m in models]
    x = np.arange(len(models))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.2))
    b1 = ax.bar(x - width / 2, train_vals, width, label="Train F1 macro", color="steelblue")
    b2 = ax.bar(x + width / 2, test_vals,  width, label="Test F1 macro",  color="crimson")
    for bar, v in list(zip(b1, train_vals)) + list(zip(b2, test_vals)):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1 macro")
    ax.set_title(
        "Capacite (train) vs generalisation (test) sur F1 macro\n"
        "RF memorise (0.91) mais generalise au niveau des autres (0.49)"
    )
    ax.axhline(0.4286, color="gray", linestyle=":", linewidth=1.4,
               label="Baseline majoritaire (test 0.429)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved {out_path.name}")


def main():
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET_PATH)
    df[TARGET_COLUMN] = (df[LOSS_COLUMN] >= df[LOSS_COLUMN].quantile(QUANTILE)).astype(int)
    print(f"Loaded {DATASET_PATH.name}: {df.shape[0]} rows; baseline pos_rate = {df[TARGET_COLUMN].mean():.4f}")
    print(f"Output dir: {GRAPHS_DIR}\n")

    fig_loss_distribution(df,             GRAPHS_DIR / "01_loss_distribution.png")
    fig_mi_numeric(df,                    GRAPHS_DIR / "02_mutual_info_numeric.png")
    fig_cramers_v(df,                     GRAPHS_DIR / "03_cramers_v_categorical.png")
    fig_pos_rate_top_categories(df,       GRAPHS_DIR / "04_pos_rate_top_categories.png")
    fig_train_vs_test_overfit(            GRAPHS_DIR / "05_train_vs_test_overfit.png")


if __name__ == "__main__":
    main()
