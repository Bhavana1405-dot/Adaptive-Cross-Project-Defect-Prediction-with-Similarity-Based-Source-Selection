"""
Smoke test — verifies all pipeline stages work end-to-end
using synthetic data shaped like PROMISE (20 features, binary labels).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from source_selector import SimilaritySourceSelector
from feature_selection import LASSOREFSelector
from coral import CORAL
from smote import SMOTE
from ensemble import StackedEnsemble

np.random.seed(42)

# ── Synthetic PROMISE-like data ──
def make_project(n=300, n_features=20, defect_rate=0.3, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    y = (rng.rand(n) < defect_rate).astype(int)
    return X, y

projects = {
    'ant':     make_project(300, seed=1),
    'camel':   make_project(250, seed=2),
    'jedit':   make_project(200, seed=3),
    'lucene':  make_project(180, seed=4),
    'synapse': make_project(150, seed=5),
}
target_name = 'ant'
target_X, target_y = projects[target_name]

print("=" * 50)
print("SMOKE TEST — All Stages")
print("=" * 50)

# Stage 0
candidate_sources = {k: v[0] for k, v in projects.items() if k != target_name}
selector = SimilaritySourceSelector(top_k=2)
selected = selector.select(target_X[:30], candidate_sources)
print(f"Stage 0 ✓ | Selected sources: {selected}")

# Stage 2
X_src = np.vstack([projects[n][0] for n in selected])
y_src = np.concatenate([projects[n][1] for n in selected])
fs = LASSOREFSelector(min_features=5, random_state=42)
X_src_sel = fs.fit_transform(X_src, y_src)
X_tgt_sel = fs.transform(target_X)
print(f"Stage 2 ✓ | Features: 20 → {fs.selected_mask_.sum()}")

# Stage 3
coral = CORAL()
X_src_adapted = coral.fit_transform(X_src_sel, X_tgt_sel[:30])
print(f"Stage 3 ✓ | CORAL adapted source shape: {X_src_adapted.shape}")

# Stage 4
smote = SMOTE(k_neighbors=5)
X_bal, y_bal = smote.fit_resample(X_src_adapted, y_src)
before = np.bincount(y_src)
after  = np.bincount(y_bal)
print(f"Stage 4 ✓ | Before SMOTE: {before}  After: {after}")

# Stage 5
model = StackedEnsemble(n_folds=3, random_state=42)
model.fit(X_bal, y_bal)
y_pred = model.predict(X_tgt_sel[30:])
y_prob = model.predict_proba(X_tgt_sel[30:])[:, 1]

from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef
y_test = target_y[30:]
print(f"Stage 5 ✓ | F1={f1_score(y_test,y_pred,zero_division=0):.3f}  "
      f"AUC={roc_auc_score(y_test,y_prob):.3f}  "
      f"MCC={matthews_corrcoef(y_test,y_pred):.3f}")

print("\n✅ All stages passed.") 