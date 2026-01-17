from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

class ClusteringStrategy(ABC):
    @abstractmethod
    def fit(self, embeddings: np.ndarray) -> None:
        """ Fit the strategy to the embeddings."""
        raise NotImplemented
    
    @abstractmethod
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Return and int label for each embedding."""
        raise NotImplementedError