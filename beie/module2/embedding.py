from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List
import hashlib

import numpy as np

class SentenceTransformerEmbedder:
    """
    Semantic sentence embedder using sentence-transformers.

    Produces dense vector embeddings suitable for clustering and similarity.
    """
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        normalize: bool = True,
     ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.normalize = normalize
        self._model = SentenceTransformer(model_name)
    
    def embed(self, texts: Iterable[str]) -> List[np.ndarray]:
        # Convert iterable to list to allow multiple passes
        texts = list(texts)

        if not texts:
            return []
        
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )

        return [embeddings[i] for i in range(len(embeddings))]
        



@dataclass(frozen=True)
class SimpleHashEmbedder:
    """
    Deterministic, dependency-free embedding generator.

    This embedder exists to support development, testing, and pipeline wiring
    without relying on external embedding services or large ML models.

    It intentionally does NOT produce semantic embeddings. Instead, it produces
    stable, pseudo-random vectors that:
      - are deterministic for the same input text
      - have a fixed dimensionality
      - can be used to test downstream components such as:
          * batching
          * storage
          * clustering
          * similarity calculations

    Embedding strategy:
      1) Compute SHA-256 hash of the input text
      2) Use part of the hash as a random seed
      3) Generate `dim` pseudo-random floats from a normal distribution
      4) Optionally L2-normalize the resulting vector

    
    Because the seed is derived from the text, identical inputs always produce
    identical embeddings across runs and machines.

    Attributes:
        dim:
            Dimensionality of the output embedding vectors.
        normalize:
            Whether to L2-normalize each vector to unit length.
    """
    dim: int = 384
    normalize: bool = True

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of input texts.

        This method processes texts independently and returns a list of NumPy
        arrays. The order of the output vectors matches the order of the input
        texts.

        Args:
            texts:
                List of input strings to embed. Each string is embedded
                independently.

        Returns:
            A list of NumPy arrays of shape (dim,), one per input text.
            If `texts` is empty, an empty list is returned.
        """
        if not texts:
            return []
        
        vectors: List[np.ndarray] = []
        for t in texts:
            vec = self._embed_one(t)
            vectors.append(vec)
        return vectors
    
    def _embed_one(self, text: str) -> np.ndarray:
        """
        Generate a deterministic pseudo-random embedding for a single text.

        Behavior:
          - If `text` is None, it is treated as an empty string.
          - A SHA-256 hash of the text is computed.
          - The first 8 bytes of the hash are converted into an integer seed.
          - A NumPy random generator is initialized with this seed.
          - `dim` samples are drawn from a standard normal distribution.
          - If normalization is enabled, the vector is L2-normalized.

        Args:
            text:
                Input string to embed. None is treated as "".

        Returns:
            A NumPy array of shape (dim,) with dtype float32.
        """
        if text is None:
            text = ""

        # Seed derived from SHA-256(text)
        seed_bytes = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(seed_bytes[:8], "big", signed=False)

        rng = np.random.default_rng(seed)

        # Generate stable pseudo-random floats
        vec = rng.standard_normal(self.dim).astype(np.float32)

        if self.normalize:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

        return vec