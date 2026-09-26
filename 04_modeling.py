"""
04_modeling.py
--------------
Fase 4 CRISP-DM: Entrenamiento y validación cruzada de modelos.

Modelos evaluados:
  1. Regresión Logística  (baseline interpretable)
  2. Random Forest         (ensamble robusto, captura no-linealidades)

Estrategia de validación: StratifiedKFold (k=5) sobre el conjunto de entrenamiento
(post-SMOTE), con reporte de AUC-ROC y Recall promedio.

Los modelos entrenados se serializan en: data/model_lr.pkl y data/model_rf.pkl
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, roc_auc_score, recall_score

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RANDOM_STATE = 42
CV_FOLDS     = 5


def load_train():
    """Carga el conjunto de entrenamiento procesado (post-SMOTE)."""
    train_path = os.path.join(DATA_DIR, "processed_train.csv")
    df = pd.read_csv(train_path)
    X = df.drop(columns=["stroke"])
    y = df["stroke"]
    return X, y


def build_models() -> dict:
    """Retorna un diccionario con los modelos a evaluar."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def cross_validate_models(X: pd.DataFrame, y: pd.Series, models: dict) -> pd.DataFrame:
    """
    Ejecuta validación cruzada estratificada para cada modelo.
    Retorna un DataFrame con AUC-ROC y Recall medios ± desviación estándar.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "auc_roc": make_scorer(roc_auc_score, needs_proba=True),
        "recall":  make_scorer(recall_score),
    }

    results = []
    for name, model in models.items():
        print(f"[MODEL] Validando: {name} ...")
        cv_results = cross_validate(
            model, X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1,
        )
        row = {
            "Modelo":       name,
            "AUC-ROC (mean)": np.mean(cv_results["test_auc_roc"]).round(4),
            "AUC-ROC (std)":  np.std(cv_results["test_auc_roc"]).round(4),
            "Recall (mean)":  np.mean(cv_results["test_recall"]).round(4),
            "Recall (std)":   np.std(cv_results["test_recall"]).round(4),
        }
        results.append(row)
        print(
            f"       AUC-ROC: {row['AUC-ROC (mean)']} ± {row['AUC-ROC (std)']}  |  "
            f"Recall: {row['Recall (mean)']} ± {row['Recall (std)']}"
        )

    return pd.DataFrame(results)


def train_and_save(X: pd.DataFrame, y: pd.Series, models: dict) -> None:
    """Entrena cada modelo sobre todo el conjunto de entrenamiento y lo serializa."""
    for name, model in models.items():
        model.fit(X, y)
        fname = name.lower().replace(" ", "_").replace("-", "") 
        path  = os.path.join(DATA_DIR, f"model_{fname}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"[MODEL] Modelo guardado: {path}")


if __name__ == "__main__":
    X_train, y_train = load_train()
    models = build_models()

    print(f"\n=== Validación cruzada estratificada ({CV_FOLDS}-fold) ===")
    cv_df = cross_validate_models(X_train, y_train, models)
    print("\n=== Resultados CV ===")
    print(cv_df.to_string(index=False))

    print("\n=== Entrenamiento final sobre todo el conjunto de entrenamiento ===")
    train_and_save(X_train, y_train, models)

    print("\n[OK] Fase de modelado completada.")
