import os
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef
from xgboost import XGBClassifier

DATA_DIR = "data"
TOP_K = 3
REPEATS = 5  # run multiple times for stability

NUMERIC_METRICS = [
    "wmc", "dit", "noc", "cbo", "rfc", "lcom", "ca", "ce", "npm", 
    "lcom3", "loc", "dam", "moa", "mfa", "cam", "ic", "cbm", 
    "amc", "max_cc", "avg_cc"
]

def prepare_dataset(df):
    df = df.copy()

    # Remove metadata
    for col in ["name", "version", "classname"]:
        if col in df.columns:
            df = df.drop(col, axis=1)

    y = (df["bug"] > 0).astype(int)
    X = df[NUMERIC_METRICS]

    return X, y

# Collect all project files
project_files = [
    os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR)
    if f.endswith(".csv")
]

all_repeat_results = []

for repeat in range(REPEATS):

    results = []

    for i, target_path in enumerate(project_files):

        # Randomly choose source projects
        indices = list(range(len(project_files)))
        indices.remove(i)

        source_indices = random.sample(indices, TOP_K)

        source_data = []
        source_labels = []

        for idx in source_indices:
            df_src = pd.read_csv(project_files[idx])
            X_src, y_src = prepare_dataset(df_src)
            source_data.append(X_src)
            source_labels.append(y_src)

        X_source = pd.concat(source_data, ignore_index=True)
        y_source = pd.concat(source_labels, ignore_index=True)

        df_target = pd.read_csv(target_path)
        X_target, y_target = prepare_dataset(df_target)

        # Scale
        scaler = StandardScaler()
        X_source = scaler.fit_transform(X_source)
        X_target = scaler.transform(X_target)

        # Train
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_source, y_source)

        y_pred = model.predict(X_target)
        y_prob = model.predict_proba(X_target)[:, 1]

        f1 = f1_score(y_target, y_pred)
        auc = roc_auc_score(y_target, y_prob)
        mcc = matthews_corrcoef(y_target, y_pred)

        results.append((f1, auc, mcc))

    avg_f1 = np.mean([r[0] for r in results])
    avg_auc = np.mean([r[1] for r in results])
    avg_mcc = np.mean([r[2] for r in results])

    print(f"Repeat {repeat+1}: F1={round(avg_f1,3)}, AUC={round(avg_auc,3)}, MCC={round(avg_mcc,3)}")

    all_repeat_results.append((avg_f1, avg_auc, avg_mcc))

# Final average across repeats
final_f1 = np.mean([r[0] for r in all_repeat_results])
final_auc = np.mean([r[1] for r in all_repeat_results])
final_mcc = np.mean([r[2] for r in all_repeat_results])

print("\n===== Random Baseline Average Across Repeats =====")
print("Avg F1:", round(final_f1,3))
print("Avg AUC:", round(final_auc,3))
print("Avg MCC:", round(final_mcc,3))