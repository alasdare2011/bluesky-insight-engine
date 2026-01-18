from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

class ClusteringStrategy(ABC):
    @abstractmethod
    def fit(self, embeddings: np.ndarray) -> None:
        """ Fit the strategy to the embeddings."""
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Return and int label for each embedding."""
        raise NotImplementedError
    
class KMeansClustering(ClusteringStrategy):
    """
    This wrapper around sklearn KMeans to match our Strategy interface
    """

    def __init__(
        self,
        n_clusters: int,
        random_state: int = 42,
        n_init: str | int = "auto",
        max_iter: int = 300):
        
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive integer.")
        
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter

        self._model = None

    def fit(self, embeddings: np.ndarray) -> None:
        from sklearn.cluster import KMeans

        x = self._validate_embeddings(embeddings)

        self._model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
            max_iter=self.max_iter,
        )

        self._model.fit(x)
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("KMeansClustering must be fit() before predict()")
        
        x = self._validate_embeddings(embeddings)
        labels = self._model.predict(x)
        return labels.astype(int)

    @staticmethod
    def _validate_embeddings(embeddings: np.ndarray) -> np.ndarray:
        x = np.asarray(embeddings)

        if x.ndim != 2:
            raise ValueError(f"embeddings must be 2D (n_samples, n_features). Got {x.ndim}D.")
        if x.shape[0] == 0:
            raise ValueError("embeddings must contain at least one row")
        if not np.isfinite(x).all():
            raise ValueError("embeddings contains NaN or infinite values.")
        
        return x