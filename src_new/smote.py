"""
Stage 4: Class Imbalance Handling – SMOTE
==========================================
Synthetic Minority Over-Sampling Technique.

Implemented from scratch using sklearn NearestNeighbors so no
imbalanced-learn dependency is needed.

Reference: Chawla et al., "SMOTE: Synthetic Minority Over-sampling
Technique", JAIR 2002.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


class SMOTE:
    """
    Parameters
    ----------
    k_neighbors : int   – number of nearest neighbors for interpolation
    random_state: int   – seed for reproducibility
    sampling_strategy: 'auto' balances to 50/50; float sets minority/majority ratio
    """

    def __init__(self, k_neighbors: int = 5,
                 random_state: int = 42,
                 sampling_strategy='auto'):
        self.k = k_neighbors
        self.rng = np.random.RandomState(random_state)
        self.sampling_strategy = sampling_strategy

    def fit_resample(self, X: np.ndarray,
                     y: np.ndarray):
        """
        Returns
        -------
        X_res, y_res : resampled arrays (minority oversampled)
        """
        classes, counts = np.unique(y, return_counts=True)
        majority_class = classes[np.argmax(counts)]
        minority_class = classes[np.argmin(counts)]
        n_maj = counts.max()
        n_min = counts.min()

        if self.sampling_strategy == 'auto':
            n_synthetic = n_maj - n_min
        else:
            target_minority = int(n_maj * self.sampling_strategy)
            n_synthetic = max(0, target_minority - n_min)

        if n_synthetic == 0:
            return X, y

        X_min = X[y == minority_class]

        # Fit kNN on minority class
        k = min(self.k, len(X_min) - 1)
        if k < 1:
            return X, y

        nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
        nn.fit(X_min)
        _, indices = nn.kneighbors(X_min)
        # indices[:, 0] is the sample itself, so use 1:
        neighbor_indices = indices[:, 1:]

        # Generate synthetic samples
        synthetic_X = []
        for _ in range(n_synthetic):
            i = self.rng.randint(0, len(X_min))
            nn_idx = neighbor_indices[i][self.rng.randint(0, k)]
            diff = X_min[nn_idx] - X_min[i]
            gap = self.rng.uniform(0, 1)
            synthetic_X.append(X_min[i] + gap * diff)

        synthetic_X = np.array(synthetic_X)
        synthetic_y = np.full(n_synthetic, minority_class)

        X_res = np.vstack([X, synthetic_X])
        y_res = np.concatenate([y, synthetic_y])

        # Shuffle
        perm = self.rng.permutation(len(X_res))
        return X_res[perm], y_res[perm] 