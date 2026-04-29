"""
Diagnostic complementaire: regression directe sur Financial Loss.

Objectif: prouver que ce n'est pas la binarisation au quantile 0.75 qui
detruit le signal. Si on regresse directement la valeur continue de
Financial Loss et que R^2 reste a 0, alors les features ne contiennent
aucune information sur la target sous quelque forme que ce soit.

Modeles testes:
- DummyRegressor (predit la moyenne)
- LinearRegression
- RandomForestRegressor (capacite non-lineaire)
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


HERE = Path(__file__).resolve().parent
DATASET_PATH = HERE.parent / "dataset.csv"
TARGET_COLUMN = "Financial Loss (in Million $)"


def build_preprocessor(X):
    numeric_cols = list(X.select_dtypes(include="number").columns)
    cat_cols = list(X.select_dtypes(exclude="number").columns)
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, numeric_cols),
        ("cat", categorical, cat_cols),
    ])


def evaluate(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    return {
        "model": name,
        "train_r2":   r2_score(y_train, y_pred_train),
        "test_r2":    r2_score(y_test, y_pred_test),
        "train_mae":  mean_absolute_error(y_train, y_pred_train),
        "test_mae":   mean_absolute_error(y_test, y_pred_test),
        "test_rmse":  math.sqrt(mean_squared_error(y_test, y_pred_test)),
    }


def main():
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded {DATASET_PATH.name}: {df.shape}\n")

    y = df[TARGET_COLUMN].astype(float)
    X = df.drop(columns=[TARGET_COLUMN])

    print(f"Target = {TARGET_COLUMN}")
    print(f"  range: [{y.min():.2f}, {y.max():.2f}]  mean: {y.mean():.2f}  std: {y.std():.2f}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42,
    )

    preprocessor = build_preprocessor(X_train)
    pipelines = {
        "Dummy (mean)":             Pipeline([("pre", "passthrough"),     ("model", DummyRegressor(strategy="mean"))]),
        "Linear Regression":        Pipeline([("pre", preprocessor),       ("model", LinearRegression())]),
        "Random Forest (n=200)":    Pipeline([("pre", preprocessor),       ("model", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1))]),
    }

    rows = []
    for name, pipe in pipelines.items():
        rows.append(evaluate(name, pipe, X_train, X_test, y_train, y_test))

    results = pd.DataFrame(rows)
    print("========== REGRESSION RESULTS (target = Financial Loss in Million $) ==========")
    print(results.round(4).to_string(index=False))

    # Reference: si on predit la moyenne (baseline), MAE est ~ |y - mean|.mean()
    baseline_mae = float(np.mean(np.abs(y_test - y_train.mean())))
    print(f"\nReference baseline MAE (predire la moyenne du train): {baseline_mae:.4f}")
    print(f"Range de la target: {y.max() - y.min():.2f} M$  --> echelle d'erreur attendue ~28-29 M$ "
          f"si pas de signal (std d'une uniforme 0.5-100)")

    print("\n========== INTERPRETATION ==========")
    rf_test_r2 = results.loc[results["model"].str.contains("Random Forest"), "test_r2"].iloc[0]
    rf_train_r2 = results.loc[results["model"].str.contains("Random Forest"), "train_r2"].iloc[0]
    if rf_test_r2 < 0.02:
        print(">>> R^2 test du RF tres proche de 0 : aucune information predictive dans les features")
        print(">>> sur Financial Loss directement, AVANT toute binarisation.")
        print(">>> Conclusion: ce n'est pas la binarisation au quantile 0.75 qui detruit le signal,")
        print(">>> le signal n'existe simplement pas dans le dataset.")
    if rf_train_r2 - rf_test_r2 > 0.10:
        print(">>> Le gros ecart train-test du RF confirme: capacite suffisante pour memoriser,")
        print(">>> rien a generaliser.")


if __name__ == "__main__":
    main()
