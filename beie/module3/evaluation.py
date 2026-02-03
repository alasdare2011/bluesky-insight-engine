from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

from .strategies import KMeansClustering

def choose_k_by_silhouette(
        embeddings: np.ndarray,
        k_min: int = 3,
        k_max: int = 15,
        random_state: int = 42.
) -> Tuple[int, Dict[int, float]]:
    """
    Pick k that maximizes silhouette score over [k_min, k_max].

    Returns:
      (best_k, scores_by_k)

    Notes:
      - silhouette is only defined for 2 <= k <= n_samples-1
      - if no usable scores, falls back to clamped k_min
    """
    # Import here so Module 3 doesn't hard-require sklearn unless you use auto-k
    from sklearn.metrics import silhouette_score

    X = np.asarray(embeddings)
    n = X.shape[0]

    scores: Dict[int, float] = {}

    # not enough points for silhouette
    if n < 3:
        best = max(2, min(k_min, n-1))
        return best, scores
    
    #clamp to valid silhouette range
    k_min = max(2, k_min)
    k_max = min(k_max, n-1)

    if k_min > k_max:
        return k_max, scores
    
    best_k = k_min
    best_score = float("-inf")

    for k in range(k_min, k_max + 1):
        strat = KMeansClustering(n_clusters=k, random_state=random_state)
        strat.fit(X)
        labels = strat.predict(X)

        # silhouette needs at least 2 clusters
        if len(set(labels.tolist())) < 2:
            continue

        s = float(silhouette_score(X, labels))
        scores[k] = s
        if s > best_score:
            best_score = s
            best_k = k

    if not scores:
        best_k = k_min

    return best_k, scores    
