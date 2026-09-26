"""
01_data_loading.py
------------------
Fase 1 CRISP-DM: Recolección y comprensión de datos.

Descarga el Stroke Prediction Dataset desde Kaggle usando kagglehub
y realiza una revisión inicial de atributos, tipos y calidad de datos.

Requisito: tener configurado ~/.kaggle/kaggle.json
           O colocar el CSV manualmente en data/healthcare-dataset-stroke-data.csv
"""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CSV_NAME = "healthcare-dataset-stroke-data.csv"
CSV_PATH = os.path.join(DATA_DIR, CSV_NAME)


def download_dataset() -> str:
    """Descarga el dataset desde Kaggle; retorna la ruta al CSV."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(CSV_PATH):
        print(f"[INFO] Dataset ya presente en: {CSV_PATH}")
        return CSV_PATH

    try:
        import kagglehub  # noqa: PLC0415
        path = kagglehub.dataset_download("fedesoriano/stroke-prediction-dataset")
        # kagglehub descarga en una carpeta temporal; movemos el CSV a data/
        for root, _, files in os.walk(path):
            for fname in files:
                if fname.endswith(".csv"):
                    src = os.path.join(root, fname)
                    import shutil
                    shutil.copy(src, CSV_PATH)
                    print(f"[INFO] Dataset copiado a: {CSV_PATH}")
                    return CSV_PATH
        raise FileNotFoundError("No se encontró CSV en el paquete descargado.")
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] No se pudo descargar el dataset automáticamente: {exc}")
        print(f"        Coloca manualmente el CSV en: {CSV_PATH}")
        raise


def load_data(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Carga el CSV y retorna un DataFrame."""
    df = pd.read_csv(csv_path)
    return df


def describe_data(df: pd.DataFrame) -> None:
    """Imprime un resumen inicial del dataset."""
    print("\n=== Dimensiones ===")
    print(f"Filas: {df.shape[0]}  |  Columnas: {df.shape[1]}")

    print("\n=== Tipos de datos ===")
    print(df.dtypes)

    print("\n=== Primeras filas ===")
    print(df.head())

    print("\n=== Valores faltantes por columna ===")
    missing = df.isnull().sum()
    print(missing[missing > 0])

    print("\n=== Distribución de la variable objetivo ===")
    counts = df["stroke"].value_counts()
    pct = df["stroke"].value_counts(normalize=True) * 100
    print(pd.DataFrame({"count": counts, "pct": pct.round(2)}))

    print("\n=== Estadísticos descriptivos (variables numéricas) ===")
    print(df.describe())


if __name__ == "__main__":
    csv_path = download_dataset()
    df = load_data(csv_path)
    describe_data(df)
    print("\n[OK] Fase 1 completada. Dataset listo en:", csv_path)
