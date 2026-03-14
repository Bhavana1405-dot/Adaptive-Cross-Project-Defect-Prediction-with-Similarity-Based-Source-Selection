"""
Stage 0: Similarity-Based Source Project Selection
===================================================
Computes a combined similarity score between a target project and each candidate
source project using:
  1. Cosine Similarity  – captures directional alignment of feature centroids
  2. MMD (Maximum Mean Discrepancy) – captures distributional overlap
  3. A-Distance          – measures classifier-level domain separability

Final score = w1 * cosine_sim + w2 * (1 - MMD_norm) + w3 * (1 - A_dist_norm)

Returns Top-K most similar source projects (adaptive threshold optional).
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# ─────────────────────────────────────────────
# MMD (linear kernel)
# ─────────────────────────────────────────────
def compute_mmd(X_s: np.ndarray, X_t: np.ndarray) -> float:
    """
    Linear-kernel Maximum Mean Discrepancy between source and target.
    Lower = more similar distributions.
    """
    mu_s = X_s.mean(axis=0)
    mu_t = X_t.mean(axis=0)
    diff = mu_s - mu_t
    return float(np.dot(diff, diff))


# ─────────────────────────────────────────────
# A-Distance (proxy via logistic regression)
# ─────────────────────────────────────────────
def compute_a_distance(X_s: np.ndarray, X_t: np.ndarray) -> float:
    """
    Proxy A-Distance: train a domain discriminator (source=0, target=1).
    A-dist = 2 * (1 - 2*error).  Lower = more similar.
    Capped at [0, 2].
    """
    n_s, n_t = len(X_s), len(X_t)
    n_min = min(n_s, n_t, 300)          # cap for speed

    idx_s = np.random.choice(n_s, n_min, replace=False)
    idx_t = np.random.choice(n_t, n_min, replace=False)

    X_disc = np.vstack([X_s[idx_s], X_t[idx_t]])
    y_disc = np.array([0] * n_min + [1] * n_min)

    clf = LogisticRegression(max_iter=300, solver='lbfgs', C=1.0)
    try:
        scores = cross_val_score(clf, X_disc, y_disc, cv=3, scoring='accuracy')
        error = 1.0 - scores.mean()
    except Exception:
        error = 0.5                     # fallback: assume random

    a_dist = max(0.0, 2.0 * (1.0 - 2.0 * error))
    return a_dist


# ─────────────────────────────────────────────
# Main Selector
# ─────────────────────────────────────────────
class SimilaritySourceSelector:
    """
    Parameters
    ----------
    top_k         : int   – number of source projects to select (None = adaptive)
    threshold     : float – min score to include a source (used when top_k=None)
    w_cos, w_mmd, w_adist : float – weights for each similarity component
    """

    def __init__(self, top_k=3, threshold=0.4,
                 w_cos=0.4, w_mmd=0.4, w_adist=0.2):
        self.top_k = top_k
        self.threshold = threshold
        self.w_cos = w_cos
        self.w_mmd = w_mmd
        self.w_adist = w_adist

    def _project_vector(self, X: np.ndarray) -> np.ndarray:
        """Project-level centroid representation."""
        return X.mean(axis=0, keepdims=True)   # shape (1, d)

    def rank(self, target_X: np.ndarray,
             sources: dict,
             target_y: np.ndarray = None,
             sources_y: dict = None) -> list:
        """
        Parameters
        ----------
        target_X  : np.ndarray  shape (n_target, d)
        sources   : dict  {name: X_array}
        target_y  : optional label array — enables defect-rate similarity
        sources_y : optional dict {name: y_array}

        Returns
        -------
        List of (name, score) sorted descending by score.
        """
        scaler = StandardScaler()
        all_X = np.vstack([target_X] + list(sources.values()))
        scaler.fit(all_X)

        X_t_scaled = scaler.transform(target_X)
        cent_t = self._project_vector(X_t_scaled)

        # Defect rate of target (if labels available)
        dr_t = float(target_y.mean()) if target_y is not None else None

        # Compute raw scores
        raw = {}
        for name, X_s in sources.items():
            X_s_scaled = scaler.transform(X_s)
            cent_s = self._project_vector(X_s_scaled)

            cos  = float(cosine_similarity(cent_s, cent_t)[0, 0])
            mmd  = compute_mmd(X_s_scaled, X_t_scaled)
            adst = compute_a_distance(X_s_scaled, X_t_scaled)

            # Defect rate similarity: penalise large differences
            if dr_t is not None and sources_y is not None and name in sources_y:
                dr_s = float(sources_y[name].mean())
                dr_sim = 1.0 - abs(dr_t - dr_s)   # 1=identical, 0=opposite
            else:
                dr_sim = 0.5   # neutral when unknown

            raw[name] = {'cos': cos, 'mmd': mmd, 'adist': adst, 'dr': dr_sim}

        # Normalize MMD and A-Distance to [0,1]
        mmd_vals   = np.array([v['mmd']   for v in raw.values()])
        adist_vals = np.array([v['adist'] for v in raw.values()])

        mmd_max   = mmd_vals.max()   if mmd_vals.max()   > 0 else 1.0
        adist_max = adist_vals.max() if adist_vals.max() > 0 else 1.0

        # Weights: cos=0.35, mmd=0.35, adist=0.15, defect_rate=0.15
        scored = {}
        for name, v in raw.items():
            cos_sim   = v['cos']
            mmd_sim   = 1.0 - v['mmd']   / mmd_max
            adist_sim = 1.0 - v['adist'] / adist_max
            dr_sim    = v['dr']

            score = (0.35 * cos_sim  +
                     0.35 * mmd_sim  +
                     0.15 * adist_sim +
                     0.15 * dr_sim)
            scored[name] = score

        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def select(self, target_X: np.ndarray, sources: dict,
               target_y: np.ndarray = None,
               sources_y: dict = None) -> list:
        """
        Returns list of selected source project names.
        Uses top_k if set, else adaptive threshold.
        Pass target_y and sources_y to enable defect-rate similarity.
        """
        ranked = self.rank(target_X, sources, target_y, sources_y)
        if self.top_k is not None:
            selected = [name for name, _ in ranked[:self.top_k]]
        else:
            selected = [name for name, score in ranked
                        if score >= self.threshold]
            if not selected:             # always keep at least 1
                selected = [ranked[0][0]]
        return selected