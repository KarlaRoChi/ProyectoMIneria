"""Export the trained Logistic Regression model to docs/model.json for the GitHub Pages demo.

Run after 03_preprocessing.ipynb and 04_modeling.ipynb:
    python export_web_model.py

The scaler is refitted exactly as in 03_preprocessing (same cleaning, split and random_state),
so the web page applies the same transformation as the notebooks.
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
RANDOM_STATE = 42
TEST_SIZE = 0.20
NUM_COLS = ["age", "avg_glucose_level", "bmi"]
CAT_COLS = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]

# Same cleaning and split as 03_preprocessing.ipynb
df = pd.read_csv(os.path.join(DATA_DIR, "healthcare-dataset-stroke-data.csv"))
df = df.drop(columns=["id"])
df = df[df["gender"] != "Other"].reset_index(drop=True)
bmi_median = df["bmi"].median()
df["bmi"] = df["bmi"].fillna(bmi_median)
df_enc = pd.get_dummies(df, columns=CAT_COLS, drop_first=False)
X = df_enc.drop(columns=["stroke"])
y = df_enc["stroke"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
scaler = StandardScaler().fit(X_train[NUM_COLS])

with open(os.path.join(DATA_DIR, "model_logistic_regression.pkl"), "rb") as f:
    model = pickle.load(f)

features = list(model.feature_names_in_)
assert features == list(X.columns), "Feature order differs from preprocessing"

# Sanity check: the processed test set must match our own scaling
test_df = pd.read_csv(os.path.join(DATA_DIR, "processed_test.csv"))
X_test_s = X_test.copy()
X_test_s[NUM_COLS] = scaler.transform(X_test[NUM_COLS])
assert np.allclose(X_test_s[NUM_COLS].values, test_df[NUM_COLS].values), "Scaler mismatch"

# Reference cases so the web page can be checked against Python
ref = X_test.iloc[:20]
ref_s = X_test_s.iloc[:20].astype(float)
proba = model.predict_proba(ref_s)[:, 1]
raw_df = df.loc[ref.index]
examples = []
for (_, row), p in zip(raw_df.iterrows(), proba):
    examples.append({
        "input": {k: (row[k].item() if hasattr(row[k], "item") else row[k])
                  for k in NUM_COLS + CAT_COLS + ["hypertension", "heart_disease"]},
        "proba": float(p),
    })

out = {
    "model": "LogisticRegression (class_weight='balanced', trained on SMOTE-balanced train set)",
    "features": features,
    "coef": [float(c) for c in model.coef_[0]],
    "intercept": float(model.intercept_[0]),
    "scaler": {c: {"mean": float(m), "scale": float(s)}
               for c, m, s in zip(NUM_COLS, scaler.mean_, scaler.scale_)},
    "bmi_median": float(bmi_median),
    "thresholds": {"default": 0.50, "screening": 0.30},
    "test_metrics": {"auc": 0.8270, "recall_050": 0.66, "recall_030": 0.72},
    "examples": examples,
}
os.makedirs(DOCS_DIR, exist_ok=True)
with open(os.path.join(DOCS_DIR, "model.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("Saved docs/model.json with", len(features), "features")
