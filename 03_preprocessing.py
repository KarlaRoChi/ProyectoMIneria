"""
03_preprocessing.py
-------------------
Fase 2 CRISP-DM: Preprocesamiento y limpieza de datos.

Operaciones aplicadas:
  1. Eliminación de la columna `id` (identificador sin valor predictivo).
  2. Eliminación del registro con gender='Other' (única observación, no aporta información).
  3. Imputación de valores faltantes en `bmi` con la mediana.
  4. Tratamiento de `smoking_status='Unknown'` como categoría propia.
  5. One-hot encoding de variables categóricas.
  6. Estandarización de variables numéricas (StandardScaler).
  7. Manejo del desbalance de clases con SMOTE sobre el conjunto de entrenamiento.

Salida:
  - X_train_res, y_train_res  (con SMOTE aplicado)
  - X_test,      y_test        (sin SMOTE — evaluación real)
  - Guardado como: data/processed_train.csv y data/processed_test.csv
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_PATH = os.path.join(DATA_DIR, "healthcare-dataset-stroke-data.csv")

RANDOM_STATE = 42
TEST_SIZE    = 0.20


def load_raw() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza básica del dataset."""
    df = df.copy()

    # 1. Eliminar columna id
    df = df.drop(columns=["id"])

    # 2. Eliminar registros con gender='Other' (n=1)
    df = df[df["gender"] != "Other"].reset_index(drop=True)

    # 3. Imputar bmi con la mediana
    bmi_median = df["bmi"].median()
    df["bmi"] = df["bmi"].fillna(bmi_median)
    print(f"[PREP] bmi imputado con mediana = {bmi_median:.2f}")

    # 4. smoking_status='Unknown' se mantiene como categoría (ya es string)
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encoding de variables categóricas."""
    cat_cols = [
        "gender", "ever_married", "work_type",
        "Residence_type", "smoking_status",
    ]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    return df


def split_and_scale(df: pd.DataFrame):
    """
    Divide en train/test, estandariza variables numéricas y aplica SMOTE al train.
    Retorna: X_train_res, X_test, y_train_res, y_test, scaler
    """
    X = df.drop(columns=["stroke"])
    y = df["stroke"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Estandarización solo sobre columnas numéricas
    num_cols = ["age", "avg_glucose_level", "bmi"]
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])

    print(f"[PREP] Train: {X_train.shape[0]} muestras | Test: {X_test.shape[0]} muestras")
    print(f"[PREP] Distribución y_train antes de SMOTE: {y_train.value_counts().to_dict()}")

    # SMOTE solo sobre entrenamiento
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"[PREP] Distribución y_train después de SMOTE: {pd.Series(y_train_res).value_counts().to_dict()}")

    return X_train_res, X_test, y_train_res, y_test, scaler


def save_processed(X_train_res, X_test, y_train_res, y_test) -> None:
    """Guarda los conjuntos procesados en CSV para reproducibilidad."""
    train_df = X_train_res.copy()
    train_df["stroke"] = y_train_res.values
    test_df = X_test.copy()
    test_df["stroke"] = y_test.values

    train_path = os.path.join(DATA_DIR, "processed_train.csv")
    test_path  = os.path.join(DATA_DIR, "processed_test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,  index=False)
    print(f"[PREP] Guardado: {train_path}")
    print(f"[PREP] Guardado: {test_path}")


if __name__ == "__main__":
    df_raw  = load_raw()
    df_clean = clean(df_raw)
    df_enc  = encode(df_clean)

    X_train_res, X_test, y_train_res, y_test, scaler = split_and_scale(df_enc)
    save_processed(X_train_res, X_test, y_train_res, y_test)

    print("\n[OK] Fase de preprocesamiento completada.")
