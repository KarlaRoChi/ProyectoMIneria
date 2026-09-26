"""
02_eda.py
---------
Fase 3 CRISP-DM: Análisis Exploratorio de Datos (EDA).

Genera visualizaciones de distribuciones, correlaciones y análisis
bivariado respecto a la variable objetivo `stroke`.

Figuras guardadas en: outputs/
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "outputs")
CSV_PATH  = os.path.join(DATA_DIR, "healthcare-dataset-stroke-data.csv")

os.makedirs(OUT_DIR, exist_ok=True)

# Variables del dataset
NUM_VARS = ["age", "avg_glucose_level", "bmi"]
CAT_VARS = [
    "gender", "hypertension", "heart_disease", "ever_married",
    "work_type", "Residence_type", "smoking_status",
]


def load() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


# ------------------------------------------------------------------
# 1. Distribución de la variable objetivo
# ------------------------------------------------------------------
def plot_target_distribution(df: pd.DataFrame) -> None:
    counts = df["stroke"].value_counts()
    labels = ["Sin stroke (0)", "Con stroke (1)"]
    colors = ["#3b82d4", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, counts.values, color=colors)
    ax.set_title("Distribución de la variable objetivo")
    ax.set_ylabel("Número de pacientes")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 30, f"{v}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "01_target_distribution.png"), dpi=150)
    plt.close()
    print("[EDA] Guardado: 01_target_distribution.png")


# ------------------------------------------------------------------
# 2. Distribuciones de variables numéricas
# ------------------------------------------------------------------
def plot_numeric_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(NUM_VARS), figsize=(14, 4))
    for ax, col in zip(axes, NUM_VARS):
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#3b82d4")
        ax.set_title(f"Distribución: {col}")
        ax.set_xlabel(col)
    plt.suptitle("Variables numéricas", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "02_numeric_distributions.png"), dpi=150)
    plt.close()
    print("[EDA] Guardado: 02_numeric_distributions.png")


# ------------------------------------------------------------------
# 3. Análisis bivariado: variables numéricas vs stroke
# ------------------------------------------------------------------
def plot_numeric_vs_target(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(NUM_VARS), figsize=(14, 4))
    for ax, col in zip(axes, NUM_VARS):
        sns.boxplot(x="stroke", y=col, data=df, ax=ax,
                    palette={0: "#3b82d4", 1: "#e74c3c"})
        ax.set_title(f"{col} vs stroke")
        ax.set_xlabel("stroke (0=No, 1=Sí)")
    plt.suptitle("Variables numéricas vs variable objetivo", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_numeric_vs_target.png"), dpi=150)
    plt.close()
    print("[EDA] Guardado: 03_numeric_vs_target.png")


# ------------------------------------------------------------------
# 4. Análisis bivariado: variables categóricas vs stroke
# ------------------------------------------------------------------
def plot_categorical_vs_target(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    for idx, col in enumerate(CAT_VARS):
        ct = df.groupby([col, "stroke"]).size().unstack(fill_value=0)
        ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
        ct_pct.plot(kind="bar", ax=axes[idx], color=["#3b82d4", "#e74c3c"], legend=False)
        axes[idx].set_title(col, fontsize=9)
        axes[idx].set_xlabel("")
        axes[idx].tick_params(axis="x", labelrotation=30)
    # Ocultar subplot sobrante
    axes[-1].set_visible(False)
    fig.legend(["Sin stroke (0)", "Con stroke (1)"], loc="lower right", fontsize=9)
    plt.suptitle("Tasa de stroke (%) por categoría", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "04_categorical_vs_target.png"), dpi=150)
    plt.close()
    print("[EDA] Guardado: 04_categorical_vs_target.png")


# ------------------------------------------------------------------
# 5. Mapa de correlación (variables numéricas + objetivo)
# ------------------------------------------------------------------
def plot_correlation_matrix(df: pd.DataFrame) -> None:
    subset = df[NUM_VARS + ["hypertension", "heart_disease", "stroke"]].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(subset, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                linewidths=0.5, square=True)
    ax.set_title("Matriz de correlación (variables numéricas y binarias)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "05_correlation_matrix.png"), dpi=150)
    plt.close()
    print("[EDA] Guardado: 05_correlation_matrix.png")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    df = load()
    print(f"[EDA] Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

    plot_target_distribution(df)
    plot_numeric_distributions(df)
    plot_numeric_vs_target(df)
    plot_categorical_vs_target(df)
    plot_correlation_matrix(df)

    print("\n[OK] Fase EDA completada. Figuras guardadas en outputs/")
