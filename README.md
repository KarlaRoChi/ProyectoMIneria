# Proyecto Minería de Datos — Predicción de Stroke

**UNAM · Facultad de Ingeniería · Minería de Datos · Grupo 1 · Semestre 2027-1**

Equipo:
- Rojas Chimal Karla Beatriz — 320048025
- García Gallegos Alejandro — 423093137

Profesor: Mtro. Gerardo Gabriel Carrasco Zúñiga

---

## Objetivo

Desarrollar un modelo de **clasificación binaria supervisada** para predecir la probabilidad de
que un paciente sufra un evento cerebrovascular (*stroke*), utilizando el
[Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
(5 110 registros, 12 variables).

Metodología de referencia: **CRISP-DM**.

---

## Estructura del repositorio

```
CODIGO/
├── data/                        # Dataset descargado (git-ignorado)
├── outputs/                     # Figuras y reportes generados
├── 01_data_loading.ipynb        # Fase 1 – Descarga y comprensión de datos
├── 02_eda.ipynb                 # Fase 3 – Análisis exploratorio de datos (EDA)
├── 03_preprocessing.ipynb       # Fase 2 – Preprocesamiento y limpieza
├── 04_modeling.ipynb            # Fase 4 – Entrenamiento y validación de modelos
├── 05_evaluation.ipynb          # Fase 5 – Evaluación, métricas y curvas ROC
├── requirements.txt
└── README.md
```

---

## Instalación

```bash
pip install -r requirements.txt
```

> Se requiere una cuenta de Kaggle con `kaggle.json` configurado, **o** colocar el archivo
> `healthcare-dataset-stroke-data.csv` manualmente en la carpeta `data/`.

---

## Ejecución

Abre y ejecuta los notebooks en orden desde Jupyter:

```bash
jupyter notebook
```

| Notebook | Fase CRISP-DM | Descripción |
|---|---|---|
| `01_data_loading.ipynb` | Fase 1 | Descarga del dataset, revisión de tipos, nulos y distribución de `stroke` |
| `02_eda.ipynb` | Fase 3 | Histogramas, boxplots, tasa por categoría, mapa de correlación |
| `03_preprocessing.ipynb` | Fase 2 | Imputación `bmi`, one-hot encoding, StandardScaler, SMOTE |
| `04_modeling.ipynb` | Fase 4 | Regresión Logística + Random Forest, StratifiedKFold (k=5) |
| `05_evaluation.ipynb` | Fase 5 | AUC-ROC, Recall, F1, curvas ROC, matrices de confusión, importancia de variables |

---

## Métricas objetivo

| Métrica | Umbral esperado |
|---------|----------------|
| AUC-ROC | > 0.85 |
| Recall (clase positiva) | > 0.75 |
| F1-score | > 0.60 |

---

## Modelos evaluados

1. **Regresión Logística** — modelo base interpretable.
2. **Random Forest** — modelo de ensamble robusto a no-linealidades.

Desbalance de clases tratado con **SMOTE** y parámetro `class_weight='balanced'`.
