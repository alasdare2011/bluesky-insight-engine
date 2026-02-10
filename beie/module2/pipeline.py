from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Protocol

import numpy as np

from beie.ingestion.models import RawPost
from beie.module2.models import PreprocessedPost, EmbeddedPost
from beie.module2.preprocessing import PostPreprocessor

class Embedder(Protocol):
    """
    Interface (protocol) for embedding implementations used by Module 2.

    Any embedder used in the pipeline must implement:
        embed(texts: List[str]) -> List[np.ndarray]

    Contract / expectations:
      - The returned list must have the same length as the input list.
      - Output vectors should have consistent dimensionality across calls.
      - Implementations should be deterministic where possible for testing,
        but production embedders may not be strictly deterministic.
    """
    def embed(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed a batch of input texts into vectors.

        Args:
            texts:
                List of strings to embed (batch).

        Returns:
            A list of NumPy arrays, one per input text, in the same order.
        """
        ...


@dataclass
class EmbeddingPipeline:
    """
    Module 2 orchestration pipeline: RawPost -> PreprocessedPost -> EmbeddedPost.

    This class coordinates two main responsibilities:
      1) Preprocessing raw posts into cleaned text (via `PostPreprocessor`)
      2) Embedding cleaned text into numeric vectors (via `Embedder`)

    The pipeline is intentionally dependency-injected:
      - `preprocessor` can be swapped (simple cleaner vs advanced NLP)
      - `embedder` can be swapped (SimpleHashEmbedder vs real embedding model)

    Attributes:
        preprocessor:
            Component responsible for converting RawPost objects into
            PreprocessedPost objects (cleaned + normalized text).
        embedder:
            Component responsible for converting cleaned text into
            NumPy embedding vectors.
    """
    preprocessor: PostPreprocessor
    embedder: Embedder

    def run(self, posts: Iterable[RawPost]) -> List[EmbeddedPost]:
        """
        Run the Module 2 pipeline over an iterable of RawPosts.

        Steps:
          1) Preprocess raw posts into `PreprocessedPost` objects.
          2) Extract cleaned text and generate embeddings in batch.
          3) Validate embedder output count matches the preprocessed input count.
          4) Produce `EmbeddedPost` objects, converting embeddings to List[float]
             for JSON-serializability and portability.

        Args:
            posts:
                Iterable of `RawPost` records (from Module 1 ingestion).

        Returns:
            List of `EmbeddedPost` objects ready for downstream modules (e.g.,
            clustering, indexing, insight generation). Returns an empty list if:
              - there are no posts
              - all posts are filtered out by preprocessing
        
        Raises:
            ValueError:
                If the embedder returns a different number of vectors than the
                number of preprocessed posts. This is treated as a serious
                integrity violation because it misaligns embeddings to posts.
        """

        preprocessed: List[PreprocessedPost] = self.preprocessor.preprocess_many(posts)
        if not preprocessed:
            return []
        
        texts = [p.clean_text for p in preprocessed]
        vectors = self.embedder.embed(texts)

        if len(vectors) != len(preprocessed):
            raise ValueError(
                f"Embedder returned {len(vectors)} vectors for {len(preprocessed)} texts."
            )
        
        embedded: List[EmbeddedPost] = []
        for p, vec in zip(preprocessed, vectors):
            embedded.append(
                EmbeddedPost(
                    post_id=p.post_id,
                    author=p.author,
                    timestamp=p.timestamp,
                    clean_text=p.clean_text,
                    clean_tokens=list(p.clean_tokens),
                    embedding=vec.tolist(),     # np.ndarray -> List[float]
                    content_hash=p.content_hash,
                    metadata=dict(p.metadata),
                )
            )
        
        return embedded
