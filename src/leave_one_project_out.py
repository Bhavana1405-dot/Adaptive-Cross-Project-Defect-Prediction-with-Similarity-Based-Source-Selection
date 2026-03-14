import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBClassifier

DATA_DIR = "data"
TOP_K = 3

# Same numeric metrics list you defined earlier
NUMERIC_METRICS = [
    "wmc", "dit", "noc", "cbo", "rfc", "lcom", "ca", "ce", "npm", 
    "lcom3", "loc", "dam", "moa", "mfa", "cam", "ic", "cbm", 
    "amc", "max_cc", "avg_cc"
]

def coral(source, target):
    cov_source = np.cov(source, rowvar=False) + np.eye(source.shape[1])
    cov_target = np.cov(target, rowvar=False) + np.eye(target.shape[1])

    U_s, S_s, _ = np.linalg.svd(cov_source)
    U_t, S_t, _ = np.linalg.svd(cov_target)

    source_whiten = source @ U_s @ np.diag(1.0 / np.sqrt(S_s)) @ U_s.T
    source_aligned = source_whiten @ U_t @ np.diag(np.sqrt(S_t)) @ U_t.T

    return source_aligned

def get_project_vector(file_path):
    data = pd.read_csv(file_path)
    available = [m for m in NUMERIC_METRICS if m in data.columns]
    X = data[available]

    # Fill missing metrics with 0
    for m in set(NUMERIC_METRICS) - set(available):
        X[m] = 0

    X = X[NUMERIC_METRICS]
    return np.mean(X.values, axis=0)

def load_project(file_path):
    data = pd.read_csv(file_path)

    available = [m for m in NUMERIC_METRICS if m in data.columns]
    X = data[available]

    for m in set(NUMERIC_METRICS) - set(available):
        X[m] = 0

    X = X[NUMERIC_METRICS]

    y = (data["bug"] > 0).astype(int)

    return X, y

# Collect all project files
project_files = [
    os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR)
    if f.endswith(".csv")
]

project_names = [os.path.basename(p) for p in project_files]

# Compute project vectors
project_vectors = []
for path in project_files:
    project_vectors.append(get_project_vector(path))

vectors = np.array(project_vectors)
scaler = StandardScaler()
vectors = scaler.fit_transform(vectors)

results = []

# Leave-one-project-out loop
for i, target_path in enumerate(project_files):

    target_name = os.path.basename(target_path)
    target_vector = vectors[i].reshape(1, -1)

    similarities = cosine_similarity(target_vector, vectors)[0]
    sorted_indices = np.argsort(similarities)[::-1]

    # Select Top-K excluding itself
    source_indices = [idx for idx in sorted_indices if idx != i][:TOP_K]

    # Combine source projects
    source_data = []
    source_labels = []

    for idx in source_indices:
        X_src, y_src = load_project(project_files[idx])
        source_data.append(X_src)
        source_labels.append(y_src)

    X_source = pd.concat(source_data, ignore_index=True)
    y_source = pd.concat(source_labels, ignore_index=True)

    # Load target
    X_target, y_target = load_project(target_path)

    # Scale
    scaler = StandardScaler()
    X_source = scaler.fit_transform(X_source)
    X_target = scaler.transform(X_target)

    X_source = coral(X_source, X_target)

    # Train model
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_source, y_source)

    y_pred = model.predict(X_target)
    y_prob = model.predict_proba(X_target)[:,1]

    f1 = f1_score(y_target, y_pred)
    auc = roc_auc_score(y_target, y_prob)
    mcc = matthews_corrcoef(y_target, y_pred)

    print(f"{target_name}: F1={round(f1,3)}, AUC={round(auc,3)}, MCC={round(mcc,3)}")

    results.append((f1, auc, mcc))

# Compute average
avg_f1 = np.mean([r[0] for r in results])
avg_auc = np.mean([r[1] for r in results])
avg_mcc = np.mean([r[2] for r in results])

print("\n===== Overall Average Performance =====")
print("Avg F1:", round(avg_f1, 3))
print("Avg AUC:", round(avg_auc, 3))
print("Avg MCC:", round(avg_mcc, 3))