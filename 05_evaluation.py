"""
05_evaluation.py
----------------
Fase 5 CRISP-DM: Evaluación final, métricas y visualizaciones.

Carga los modelos entrenados y los evalúa sobre el conjunto de prueba
(no visto durante el entrenamiento, sin SMOTE).

Genera:
  - Tabla comparativa: AUC-ROC, Recall, F1-score, Accuracy
  - Curvas ROC de ambos modelos (outputs/06_roc_curves.png)
  - Matrices de confusión (outputs/07_confusion_matrices.png)
  - Importancia de características del Random Forest (outputs/08_feature_importance.png)
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_FILES = {
    "Logistic Regression": "model_logistic_regression.pkl",
    "Random Forest":       "model_random_forest.pkl",
}


# ------------------------------------------------------------------
# Carga
# ------------------------------------------------------------------
def load_test():
    test_path = os.path.join(DATA_DIR, "processed_test.csv")
    df = pd.read_csv(test_path)
    X = df.drop(columns=["stroke"])
    y = df["stroke"]
    return X, y


def load_models() -> dict:
    models = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
    return models


# ------------------------------------------------------------------
# Métricas
# ------------------------------------------------------------------
def evaluate_models(X_test: pd.DataFrame, y_test: pd.Series, models: dict) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        rows.append({
            "Modelo":    name,
            "AUC-ROC":  round(roc_auc_score(y_test, y_proba), 4),
            "Recall":   round(recall_score(y_test, y_pred), 4),
            "F1-score": round(f1_score(y_test, y_pred), 4),
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        })
    df = pd.DataFrame(rows)
    print("\n=== Métricas en conjunto de prueba ===")
    print(df.to_string(index=False))
    return df


# ------------------------------------------------------------------
# Curvas ROC
# ------------------------------------------------------------------
def plot_roc_curves(X_test: pd.DataFrame, y_test: pd.Series, models: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"Logistic Regression": "#3b82d4", "Random Forest": "#7c5cd8"}

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=colors.get(name, "gray"))

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random (AUC=0.50)")
    ax.set_xlabel("Tasa de falsos positivos (FPR)")
    ax.set_ylabel("Tasa de verdaderos positivos (TPR / Recall)")
    ax.set_title("Curvas ROC — Conjunto de prueba")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "06_roc_curves.png"), dpi=150)
    plt.close()
    print("[EVAL] Guardado: 06_roc_curves.png")


# ------------------------------------------------------------------
# Matrices de confusión
# ------------------------------------------------------------------
def plot_confusion_matrices(X_test: pd.DataFrame, y_test: pd.Series, models: dict) -> None:
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Pred: No stroke", "Pred: Stroke"],
            yticklabels=["Real: No stroke", "Real: Stroke"],
        )
        ax.set_title(f"Matriz de confusión\n{name}")
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "07_confusion_matrices.png"), dpi=150)
    plt.close()
    print("[EVAL] Guardado: 07_confusion_matrices.png")


# ------------------------------------------------------------------
# Importancia de características (Random Forest)
# ------------------------------------------------------------------
def plot_feature_importance(X_test: pd.DataFrame, models: dict) -> None:
    rf_model = models.get("Random Forest")
    if rf_model is None:
        return

    importances = rf_model.feature_importances_
    feat_df = pd.DataFrame({
        "feature":    X_test.columns,
        "importance": importances,
    }).sort_values("importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(feat_df["feature"][::-1], feat_df["importance"][::-1], color="#3b82d4")
    ax.set_xlabel("Importancia (Gini)")
    ax.set_title("Top 15 variables más importantes\n(Random Forest)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "08_feature_importance.png"), dpi=150)
    plt.close()
    print("[EVAL] Guardado: 08_feature_importance.png")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    X_test, y_test = load_test()
    models = load_models()

    metrics_df = evaluate_models(X_test, y_test, models)

    # Verificación de umbrales del anteproyecto
    print("\n=== Verificación de umbrales objetivo ===")
    for _, row in metrics_df.iterrows():
        auc_ok    = "✓" if row["AUC-ROC"]  > 0.85 else "✗"
        recall_ok = "✓" if row["Recall"]   > 0.75 else "✗"
        f1_ok     = "✓" if row["F1-score"] > 0.60 else "✗"
        print(
            f"  {row['Modelo']:25s} | "
            f"AUC>0.85 {auc_ok}({row['AUC-ROC']})  "
            f"Recall>0.75 {recall_ok}({row['Recall']})  "
            f"F1>0.60 {f1_ok}({row['F1-score']})"
        )

    plot_roc_curves(X_test, y_test, models)
    plot_confusion_matrices(X_test, y_test, models)
    plot_feature_importance(X_test, models)

    print("\n[OK] Fase de evaluación completada. Resultados en outputs/")
