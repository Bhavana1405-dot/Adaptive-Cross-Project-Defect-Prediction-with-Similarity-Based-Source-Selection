"""
Stage 5: Stacked Ensemble Classifier
======================================
On Windows with NVIDIA GPU, uses a PyTorch MLP as one base learner
(torch runs natively on Windows + CUDA unlike cuML which is Linux-only).

Base learners:
  - RandomForest          (sklearn, all CPU cores)
  - HistGradientBoosting  (sklearn fast histogram impl, x2 versions)
  - XGBoost               (gradient boosting with scale_pos_weight)
  - MLP                   (PyTorch — uses GPU if available, else CPU)

Meta-learner:
  - LogisticRegression (OOF stacking, class_weight='balanced')
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


# ─────────────────────────────────────────────────────────────
# PyTorch MLP (GPU-accelerated on Windows + NVIDIA)
# ─────────────────────────────────────────────────────────────
class TorchMLP:
    """
    2-layer MLP classifier using PyTorch.
    Automatically uses CUDA GPU if available, otherwise CPU.
    Has the same fit/predict_proba interface as sklearn classifiers.

    Install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    """

    def __init__(self, hidden_dim=256, epochs=80, lr=1e-3,
                 batch_size=64, use_focal_loss=False, random_state=42):
        self.hidden_dim     = hidden_dim
        self.epochs         = epochs
        self.lr             = lr
        self.batch_size     = batch_size
        self.use_focal_loss = use_focal_loss
        self.random_state   = random_state
        self.model_         = None
        self.device_        = None

    def _build_model(self, in_dim):
        try:
            import torch
            import torch.nn as nn
            torch.manual_seed(self.random_state)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            h2 = self.hidden_dim // 2
            model  = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.hidden_dim, h2),
                nn.LayerNorm(h2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(h2, 1)
            ).to(device)
            return model, device, torch
        except ImportError:
            return None, None, None

    def fit(self, X, y):
        model, device, torch = self._build_model(X.shape[1])
        if model is None:
            # Fallback: store data for a simple logistic regression
            self._fallback = LogisticRegression(C=1.0, max_iter=500)
            self._fallback.fit(X, y)
            return self

        import torch.nn as nn

        self.device_ = device
        X_t = torch.FloatTensor(X).to(device)
        y_t = torch.FloatTensor(y.astype(float)).unsqueeze(1).to(device)

        pos_weight = torch.tensor(
            [(y == 0).sum() / max((y == 1).sum(), 1)],
            dtype=torch.float32).to(device)

        if self.use_focal_loss:
            bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
            focal_gamma = 2.0
            def criterion(logits, targets):
                bce_loss = bce(logits, targets)
                probs = torch.sigmoid(logits)
                p_t = targets * probs + (1 - targets) * (1 - probs)
                return (((1 - p_t) ** focal_gamma) * bce_loss).mean()
        else:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer  = torch.optim.Adam(model.parameters(), lr=self.lr,
                                      weight_decay=1e-4)
        scheduler  = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=20, gamma=0.5)

        model.train()
        n = len(X_t)
        for epoch in range(self.epochs):
            # Mini-batch SGD — skip batches of size 1 to avoid norm issues
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                if len(idx) < 2:
                    continue
                optimizer.zero_grad()
                out       = model(X_t[idx])
                loss      = criterion(out, y_t[idx])
                loss.backward()
                optimizer.step()
            scheduler.step()

        self.model_ = model
        self._torch  = torch
        return self

    def predict_proba(self, X):
        if hasattr(self, '_fallback'):
            return self._fallback.predict_proba(X)

        import torch
        self.model_.eval()
        with torch.no_grad():
            X_t   = torch.FloatTensor(X).to(self.device_)
            logits = self.model_(X_t)
            probs  = torch.sigmoid(logits).cpu().numpy().flatten()
        return np.column_stack([1 - probs, probs])

    def get_params(self, deep=True):
        return dict(hidden_dim=self.hidden_dim, epochs=self.epochs,
                    lr=self.lr, batch_size=self.batch_size,
                    use_focal_loss=self.use_focal_loss,
                    random_state=self.random_state)

    def __class_getitem__(cls, item):
        return cls


# ─────────────────────────────────────────────────────────────
# Stacked Ensemble
# ─────────────────────────────────────────────────────────────
class StackedEnsemble:
    """
    Parameters
    ----------
    n_folds     : int  – OOF folds (3 = fast, 5 = more accurate)
    random_state: int
    """

    def __init__(self, n_folds=3, random_state=42):
        self.n_folds = n_folds
        self.rs      = random_state

        # Check if torch+CUDA is available once at init
        self._gpu_info = self._check_gpu()

        self.base_learners = [
            ('rf',   RandomForestClassifier(
                         n_estimators=150, max_depth=None,
                         class_weight='balanced_subsample',
                         n_jobs=-1, random_state=random_state)),
            ('hgb1', HistGradientBoostingClassifier(
                         max_iter=100, learning_rate=0.05,
                         max_depth=4, random_state=random_state)),
            ('hgb2', HistGradientBoostingClassifier(
                         max_iter=100, learning_rate=0.1,
                         max_depth=3, random_state=random_state)),
            ('xgb',  XGBClassifier(
                         n_estimators=150, max_depth=5,
                         learning_rate=0.1, subsample=0.8,
                         colsample_bytree=0.8, use_label_encoder=False,
                         eval_metric='logloss', verbosity=0,
                         n_jobs=-1, random_state=random_state)),
            ('mlp',  TorchMLP(hidden_dim=256, epochs=80,
                               random_state=random_state)),
        ]

        self.meta_learner = LogisticRegression(
            C=1.0, max_iter=300, solver='lbfgs',
            class_weight='balanced',
            random_state=random_state)

        self.scaler_      = StandardScaler()
        self._fitted_base = []

    @staticmethod
    def _check_gpu():
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                print(f"  [Ensemble] GPU detected: {name} — MLP will use CUDA ✓")
                return {'available': True, 'name': name}
            else:
                print("  [Ensemble] torch found but no CUDA GPU — MLP on CPU")
                return {'available': False}
        except ImportError:
            print("  [Ensemble] torch not installed — MLP falls back to sklearn LR")
            print("  [Ensemble] For GPU: pip install torch --index-url "
                  "https://download.pytorch.org/whl/cu121")
            return {'available': False}

    # ─────────────────────────────
    # Fit
    # ─────────────────────────────
    def fit(self, X, y):
        X = self.scaler_.fit_transform(X)
        n_learners = len(self.base_learners)
        meta_X     = np.zeros((len(X), n_learners))

        skf = StratifiedKFold(n_splits=self.n_folds,
                              shuffle=True, random_state=self.rs)

        for _, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr        = y[train_idx]
            sw_tr       = compute_sample_weight('balanced', y_tr)
            spw         = max((y_tr == 0).sum(), 1) / max((y_tr == 1).sum(), 1)
            for i, (name, clf) in enumerate(self.base_learners):
                c = self._clone(clf)
                if isinstance(c, HistGradientBoostingClassifier):
                    c.fit(X_tr, y_tr, sample_weight=sw_tr)
                elif isinstance(c, XGBClassifier):
                    c.set_params(scale_pos_weight=spw)
                    c.fit(X_tr, y_tr)
                else:
                    c.fit(X_tr, y_tr)
                meta_X[val_idx, i] = c.predict_proba(X_val)[:, 1]

        self.meta_learner.fit(meta_X, y)

        # Find optimal threshold on OOF predictions instead of using 0.5
        meta_probs = self.meta_learner.predict_proba(meta_X)[:, 1]
        self.threshold_ = self._find_best_threshold(meta_probs, y)

        self._fitted_base = []
        sw_full = compute_sample_weight('balanced', y)
        spw_full = max((y == 0).sum(), 1) / max((y == 1).sum(), 1)
        for _, clf in self.base_learners:
            c = self._clone(clf)
            if isinstance(c, HistGradientBoostingClassifier):
                c.fit(X, y, sample_weight=sw_full)
            elif isinstance(c, XGBClassifier):
                c.set_params(scale_pos_weight=spw_full)
                c.fit(X, y)
            else:
                c.fit(X, y)
            self._fitted_base.append(c)

        return self

    # ─────────────────────────────
    # Predict
    # ─────────────────────────────
    def predict_proba(self, X):
        X      = self.scaler_.transform(X)
        meta_X = np.column_stack([
            clf.predict_proba(X)[:, 1]
            for clf in self._fitted_base
        ])
        return self.meta_learner.predict_proba(meta_X)

    def predict(self, X):
        threshold = getattr(self, 'threshold_', 0.5)
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    @staticmethod
    def _find_best_threshold(y_prob, y_true, beta=1.0):
        from sklearn.metrics import fbeta_score
        best_t, best_f = 0.5, 0.0

        # Coarse pass: scan full range with step=0.05
        for t in np.arange(0.05, 0.96, 0.05):
            y_pred = (y_prob >= t).astype(int)
            if y_pred.sum() == 0:
                continue
            f = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
            if f > best_f:
                best_f = f
                best_t = t

        # Fine pass: refine around best coarse threshold with step=0.005
        lo = max(0.01, best_t - 0.06)
        hi = min(0.99, best_t + 0.06)
        for t in np.arange(lo, hi, 0.005):
            y_pred = (y_prob >= t).astype(int)
            if y_pred.sum() == 0:
                continue
            f = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
            if f > best_f:
                best_f = f
                best_t = t

        return float(best_t)

    # ─────────────────────────────
    # Helper
    # ─────────────────────────────
    @staticmethod
    def _clone(clf):
        if isinstance(clf, TorchMLP):
            return TorchMLP(**clf.get_params())
        if isinstance(clf, XGBClassifier):
            return XGBClassifier(**clf.get_params())
        return clf.__class__(**clf.get_params())