"""
Stage 2: Feature Selection – LASSO + RFE
==========================================
Mirrors the base paper's approach but adds BIC-guided automatic
regularization selection (no manual tuning needed).

Pipeline:
  1. LASSO with BIC-selected lambda → removes globally irrelevant features
  2. RFE with RandomForest estimator → fine-grained recursive elimination
"""

import numpy as np
from sklearn.linear_model import LassoCV, Lasso
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class LASSOREFSelector:
    """
    Parameters
    ----------
    min_features : int   – minimum features to keep after RFE
    max_features : int   – maximum features to keep (None = keep all LASSO survivors)
    random_state : int
    """

    def __init__(self, min_features: int = 10,
                 max_features: int = None,
                 random_state: int = 42):
        self.min_features = min_features
        self.max_features = max_features
        self.rs = random_state
        self.selected_mask_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit LASSO (BIC-based alpha) then RFE on survivors.
        Stores boolean mask of selected features.
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ── Step 1: LASSO with cross-validated alpha selection ──
        lasso_cv = LassoCV(cv=5, max_iter=2000, random_state=self.rs)
        # Convert binary labels to float for LASSO regression
        lasso_cv.fit(X_scaled, y.astype(float))

        # BIC-guided final LASSO: penalize complexity
        # Use slightly larger alpha than CV optimum to encourage sparsity
        alpha_bic = lasso_cv.alpha_ * self._bic_correction(X_scaled, y, lasso_cv)
        lasso_final = Lasso(alpha=alpha_bic, max_iter=2000)
        lasso_final.fit(X_scaled, y.astype(float))

        lasso_mask = np.abs(lasso_final.coef_) > 1e-6
        n_lasso_survivors = lasso_mask.sum()

        # Ensure at least min_features survive LASSO
        if n_lasso_survivors < self.min_features:
            top_idx = np.argsort(np.abs(lasso_final.coef_))[::-1][:self.min_features]
            lasso_mask = np.zeros(X.shape[1], dtype=bool)
            lasso_mask[top_idx] = True

        X_lasso = X_scaled[:, lasso_mask]

        # ── Step 2: RFE on LASSO survivors ──
        n_rfe_target = self.max_features if self.max_features else max(
            self.min_features, int(X_lasso.shape[1] * 0.7))
        n_rfe_target = min(n_rfe_target, X_lasso.shape[1])

        rf = RandomForestClassifier(n_estimators=100, random_state=self.rs,
                                    class_weight='balanced', n_jobs=-1)
        rfe = RFE(estimator=rf, n_features_to_select=n_rfe_target, step=1)
        rfe.fit(X_lasso, y)

        # Map RFE mask back to original feature space
        lasso_indices = np.where(lasso_mask)[0]
        final_indices = lasso_indices[rfe.support_]

        self.selected_mask_ = np.zeros(X.shape[1], dtype=bool)
        self.selected_mask_[final_indices] = True

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selected_mask_ is None:
            raise RuntimeError("Call fit() first.")
        return X[:, self.selected_mask_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    @staticmethod
    def _bic_correction(X, y, lasso_cv):
        """
        Compute a BIC-based correction multiplier.
        Returns a factor >= 1.0 (encourages more sparsity than pure CV).
        log(n) / (2 * n_features) acts as complexity penalty scaling.
        """
        n, p = X.shape
        bic_factor = np.log(n) / (2 * p)
        return max(1.0, 1.0 + bic_factor)