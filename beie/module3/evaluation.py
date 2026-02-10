from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

from .strategies import KMeansClustering

def choose_k_by_silhouette(
    embeddings: np.ndarray,
    k_min: int = 3,
    k_max: int = 15,
    random_state: int = 42,
) -> Tuple[int, Dict[int, float]]:
    """
    Choose k for KMeans by maximizing silhouette score.

    Returns:
        (best_k, scores_by_k)

    Notes:
        - silhouette requires at least 2 clusters
        - valid k range is [2, n_samples - 1]
        - if no valid scores can be computed, returns (k_min_clamped, {})
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    X = np.asarray(embeddings)
    n = X.shape[0]
    if n < 2:
        return 1, {}

    # silhouette needs 2..n-1 clusters
    k_min = max(2, int(k_min))
    k_max = min(int(k_max), n - 1)

    if k_max < k_min:
        return k_min, {}

    scores: Dict[int, float] = {}
    best_k = k_min
    best_score = -1.0

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = km.fit_predict(X)

        # guard: silhouette requires at least 2 clusters present
        if len(set(labels.tolist())) < 2:
            continue

        s = float(silhouette_score(X, labels))
        scores[k] = s

        if s > best_score:
            best_score = s
            best_k = k

    return best_k, scores
