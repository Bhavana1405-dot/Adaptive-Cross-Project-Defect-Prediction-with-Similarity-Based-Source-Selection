import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef
from xgboost import XGBClassifier

# Load source project
source = pd.read_csv("data/ant.csv")

# Load target project
target = pd.read_csv("data/camel-1.0.csv")

# Prepare source
y_source = source["bug"]
X_source = source.drop("bug", axis=1)
X_source = X_source.select_dtypes(include=["number"])

# Prepare target
y_target = target["bug"]
X_target = target.drop("bug", axis=1)
X_target = X_target.select_dtypes(include=["number"])

# IMPORTANT: Fit scaler only on source
scaler = StandardScaler()
X_source = scaler.fit_transform(X_source)
X_target = scaler.transform(X_target)

# Train on source
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_source, y_source)

# Test on target
y_pred = model.predict(X_target)
y_prob = model.predict_proba(X_target)[:, 1]

# Evaluate
print("Cross-Project Results")
print("F1:", f1_score(y_target, y_pred))
print("AUC:", roc_auc_score(y_target, y_prob))
print("MCC:", matthews_corrcoef(y_target, y_pred))