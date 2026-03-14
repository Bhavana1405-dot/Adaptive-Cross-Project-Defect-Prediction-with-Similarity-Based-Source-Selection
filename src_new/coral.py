"""
Stage 3: CORAL Domain Adaptation
=================================
CORrelation ALignment (CORAL) aligns the second-order statistics
(covariance) of source features to match the target domain.

Key advantage over LPP (used in the base paper):
  - Requires ZERO labeled target samples
  - Closed-form solution (fast, deterministic, no training instability)
  - Directly interpretable: whitens source, re-colors with target covariance

Reference: Sun et al., "Return of Frustratingly Easy Domain Adaptation", AAAI 2016
"""

import numpy as np


class CORAL:
    """
    Parameters
    ----------
    reg : float – regularization added to diagonal of covariance matrices
                  (prevents singular matrices for small/skewed datasets)
    """

    def __init__(self, reg: float = 1e-3):
        self.reg = reg
        self.A_ = None          # transformation matrix learned from fit

    def fit(self, X_source: np.ndarray, X_target: np.ndarray):
        """
        Learn the CORAL transformation from source → target statistics.

        Parameters
        ----------
        X_source : (n_s, d)
        X_target : (n_t, d)
        """
        d = X_source.shape[1]

        # Source covariance
        C_s = np.cov(X_source, rowvar=False) + self.reg * np.eye(d)

        # Target covariance
        C_t = np.cov(X_target, rowvar=False) + self.reg * np.eye(d)

        # Whitening matrix for source: C_s^{-1/2}
        eigvals_s, eigvecs_s = np.linalg.eigh(C_s)
        eigvals_s = np.maximum(eigvals_s, 1e-10)   # clip negatives
        C_s_inv_sqrt = eigvecs_s @ np.diag(eigvals_s ** -0.5) @ eigvecs_s.T

        # Re-coloring matrix from target: C_t^{+1/2}
        eigvals_t, eigvecs_t = np.linalg.eigh(C_t)
        eigvals_t = np.maximum(eigvals_t, 1e-10)
        C_t_sqrt = eigvecs_t @ np.diag(eigvals_t ** 0.5) @ eigvecs_t.T

        # Full transformation: A = C_s^{-1/2} C_t^{1/2}
        self.A_ = C_s_inv_sqrt @ C_t_sqrt
        return self

    def transform(self, X_source: np.ndarray) -> np.ndarray:
        """Apply learned CORAL transformation to source features."""
        if self.A_ is None:
            raise RuntimeError("Call fit() before transform().")
        return X_source @ self.A_

    def fit_transform(self, X_source: np.ndarray,
                      X_target: np.ndarray) -> np.ndarray:
        """Fit on source/target pair, then transform source."""
        self.fit(X_source, X_target)
        return self.transform(X_source)