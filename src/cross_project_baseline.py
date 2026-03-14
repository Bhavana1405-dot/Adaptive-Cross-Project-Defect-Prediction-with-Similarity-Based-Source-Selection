import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef
from xgboost import XGBClassifier

# 🔒 Strict metric list
NUMERIC_METRICS = [
    "wmc", "dit", "noc", "cbo", "rfc", "lcom", "ca", "ce", "npm", 
    "lcom3", "loc", "dam", "moa", "mfa", "cam", "ic", "cbm", 
    "amc", "max_cc", "avg_cc"
]

def prepare_dataset(df):
    df = df.copy()

    # Remove metadata columns if they exist
    for col in ["name", "version", "classname"]:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # Binary label
    y = (df["bug"] > 0).astype(int)

    # Strict feature selection
    X = df[NUMERIC_METRICS]

    return X, y

# -------------------------------
# Load source projects
# -------------------------------
source1 = pd.read_csv("data/xalan-2.4.csv")
source2 = pd.read_csv("data/xalan-2.5.csv")
source3 = pd.read_csv("data/xalan-2.6.csv")

source = pd.concat([source1, source2, source3], ignore_index=True)
target = pd.read_csv("data/ant.csv")

# Prepare datasets
X_source, y_source = prepare_dataset(source)
X_target, y_target = prepare_dataset(target)

# Scale (fit only on source)
scaler = StandardScaler()
X_source = scaler.fit_transform(X_source)
X_target = scaler.transform(X_target)

# Train model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_source, y_source)

# Predict
y_pred = model.predict(X_target)
y_prob = model.predict_proba(X_target)[:, 1]

# Evaluate
print("Cross-Project Results")
print("F1:", f1_score(y_target, y_pred))
print("AUC:", roc_auc_score(y_target, y_prob))
print("MCC:", matthews_corrcoef(y_target, y_pred))