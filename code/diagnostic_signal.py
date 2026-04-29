"""
Diagnostic du signal entre les features et la target High Financial Impact.

Objectif : prouver/infirmer que le dataset porte effectivement un signal
exploitable avant d'investir du temps en feature engineering ou en tuning.

Mesures :
- Distribution de Financial Loss (test d'uniformite via Kolmogorov-Smirnov)
- Mutual information classif feature/target sur les colonnes numeriques
- Cramer's V (chi2 de Pearson normalise) pour chaque colonne categorielle
- Taux de classe 1 par modalite categorielle (intervalle de Wald 95%)
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif


HERE = Path(__file__).resolve().parent
DATASET_PATH = HERE.parent / "dataset.csv"
LOSS_COLUMN = "Financial Loss (in Million $)"
TARGET_COLUMN = "High Financial Impact"
QUANTILE = 0.75


def cramers_v(contingency):
    """Cramer's V from a contingency table (>=2x2)."""
    chi2 = stats.chi2_contingency(contingency, correction=False).statistic
    n = contingency.values.sum()
    r, k = contingency.shape
    denom = n * (min(r, k) - 1)
    return math.sqrt(chi2 / denom) if denom > 0 else float("nan")


def wald_ci(p, n, z=1.96):
    """Wald 95% confidence interval for a proportion (clipped to [0, 1])."""
    if n == 0:
        return float("nan"), float("nan")
    se = math.sqrt(max(p * (1 - p), 0.0) / n)
    return max(0.0, p - z * se), min(1.0, p + z * se)


def main():
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded {DATASET_PATH.name}: {df.shape[0]} rows x {df.shape[1]} cols")

    # 1. Test d'uniformite sur Financial Loss
    print("\n========== FINANCIAL LOSS DISTRIBUTION ==========")
    loss = df[LOSS_COLUMN].astype(float)
    print(f"min={loss.min():.4f}  max={loss.max():.4f}")
    print(f"mean={loss.mean():.4f}  std={loss.std():.4f}  median={loss.median():.4f}")
    expected_uniform_std = (loss.max() - loss.min()) / math.sqrt(12)
    print(f"std attendu si uniforme [{loss.min():.2f}, {loss.max():.2f}]: {expected_uniform_std:.4f}")
    ks_stat, ks_pvalue = stats.kstest(
        loss,
        "uniform",
        args=(loss.min(), loss.max() - loss.min()),
    )
    print(f"Kolmogorov-Smirnov vs uniforme: D={ks_stat:.4f}  p={ks_pvalue:.4g}")
    if ks_pvalue > 0.05:
        print("--> impossible de rejeter l'hypothese 'Financial Loss est uniforme'")
    else:
        print("--> distribution non-uniforme (rejet)")

    # 2. Construction de la target
    threshold = float(loss.quantile(QUANTILE))
    df[TARGET_COLUMN] = (loss >= threshold).astype(int)
    pos_rate = df[TARGET_COLUMN].mean()
    print(f"\nTarget: {TARGET_COLUMN} = (loss >= {threshold:.2f})")
    print(f"  positive rate: {pos_rate:.4f} (attendu ~0.25)")
    print(f"  class counts: {df[TARGET_COLUMN].value_counts().to_dict()}")

    # 3. Mutual information sur les numeriques
    print("\n========== MUTUAL INFORMATION (numeric features) ==========")
    numeric_cols = [c for c in df.select_dtypes(include="number").columns
                    if c not in {TARGET_COLUMN, LOSS_COLUMN}]
    X_num = df[numeric_cols].values
    y = df[TARGET_COLUMN].values
    mi_scores = mutual_info_classif(X_num, y, random_state=42)
    mi_table = (
        pd.DataFrame({"feature": numeric_cols, "mutual_info": mi_scores})
        .sort_values("mutual_info", ascending=False)
        .reset_index(drop=True)
    )
    print(mi_table.to_string(index=False))
    print(f"\nMI max = {mi_table.mutual_info.max():.6f}")
    print("Rappel: MI >= 0. Tres faible (~0.01) = quasi pas de signal.")

    # 4. Cramer's V sur les categorielles
    print("\n========== CRAMER'S V (categorical features vs target) ==========")
    cat_cols = list(df.select_dtypes(include="object").columns)
    rows = []
    for col in cat_cols:
        contingency = pd.crosstab(df[col], df[TARGET_COLUMN])
        v = cramers_v(contingency)
        chi2_result = stats.chi2_contingency(contingency, correction=False)
        rows.append({
            "feature": col,
            "n_modalities": int(contingency.shape[0]),
            "cramers_v": v,
            "chi2_pvalue": float(chi2_result.pvalue),
        })
    cv_table = pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)
    print(cv_table.to_string(index=False))
    print("\nRappel:")
    print("  Cramer's V < 0.10 = association negligeable")
    print("  V dans [0.10, 0.30] = faible")
    print("  chi2 p > 0.05 = pas de difference significative entre les modalites")

    # 5. Taux de classe 1 par modalite, focus sur les colonnes les plus prometteuses
    print("\n========== POSITIVE RATE PER CATEGORY (top-3 cat. features par Cramer's V) ==========")
    base = pos_rate
    top_cats = cv_table.head(3)["feature"].tolist()
    for col in top_cats:
        print(f"\n--- {col} ---")
        grouped = (
            df.groupby(col)[TARGET_COLUMN]
            .agg(["count", "mean"])
            .rename(columns={"count": "n", "mean": "pos_rate"})
            .sort_values("pos_rate", ascending=False)
        )
        cis = grouped.apply(
            lambda r: pd.Series(wald_ci(r["pos_rate"], int(r["n"])), index=["ci_lo", "ci_hi"]),
            axis=1,
        )
        out = pd.concat([grouped, cis], axis=1)
        out["delta_vs_base"] = out["pos_rate"] - base
        print(out.round(4).to_string())
        out_min, out_max = out["pos_rate"].min(), out["pos_rate"].max()
        print(f"  -> spread modalites: {out_min:.4f} -> {out_max:.4f} "
              f"(amplitude {out_max - out_min:.4f}, baseline {base:.4f})")

    # 6. Verdict
    print("\n========== VERDICT ==========")
    mi_max = mi_table.mutual_info.max()
    cv_max = cv_table.cramers_v.max()
    print(f"MI numerique max:   {mi_max:.6f}")
    print(f"Cramer's V max:     {cv_max:.6f}")
    if mi_max < 0.005 and cv_max < 0.05:
        print("\n>>> PAS DE SIGNAL EXPLOITABLE. Aucun modele ne pourra battre la baseline.")
        print(">>> Decision recommandee: changer de target/dataset OU rapport honnete sur l'absence de signal.")
    elif mi_max < 0.02 and cv_max < 0.10:
        print("\n>>> SIGNAL TRES FAIBLE. Plafond de performance probablement < 0.55 F1 macro.")
        print(">>> Feature engineering peu utile, focus sur explication du plafond dans le rapport.")
    else:
        print("\n>>> SIGNAL DETECTABLE. Concentrer le FE sur les top features ci-dessus.")


if __name__ == "__main__":
    main()
