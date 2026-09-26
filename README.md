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
├── data/                      # Dataset descargado (git-ignorado)
├── outputs/                   # Figuras y reportes generados
├── 01_data_loading.py         # Fase 1 – Descarga y comprensión de datos
├── 02_eda.py                  # Fase 3 – Análisis exploratorio de datos (EDA)
├── 03_preprocessing.py        # Fase 2 – Preprocesamiento y limpieza
├── 04_modeling.py             # Fase 4 – Entrenamiento y validación de modelos
├── 05_evaluation.py           # Fase 5 – Evaluación, métricas y curvas ROC
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

Ejecutar los scripts en orden:

```bash
python 01_data_loading.py
python 02_eda.py
python 03_preprocessing.py
python 04_modeling.py
python 05_evaluation.py
```

Las figuras se guardan en `outputs/` y las métricas se imprimen en consola.

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
