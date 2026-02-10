from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Iterable

import numpy as np
import pytest

from beie.module2.pipeline import EmbeddingPipeline
from beie.module2.models import PreprocessedPost, EmbeddedPost


# Keep RawPost as a lightweight stand-in (only needs fields your preprocessor/pipeline uses)
@dataclass(frozen=True)
class RawPost:
    post_id: str
    author: str
    timestamp: datetime
    content: str
    content_hash: str
    metadata: Dict[str, Any]


class FakePreprocessor:
    """Returns None when content contains 'DROP'; else lower/strip and emits tokens."""

    def preprocess_one(self, post: RawPost) -> Optional[PreprocessedPost]:
        if "DROP" in post.content:
            return None

        clean_text = post.content.lower().strip()
        clean_tokens = clean_text.split()  # simple tokenization for test

        return PreprocessedPost(
            post_id=post.post_id,
            author=post.author,
            timestamp=post.timestamp,
            original_content=post.content,
            clean_text=clean_text,
            clean_tokens=clean_tokens,
            content_hash=post.content_hash,
            metadata=dict(post.metadata),
        )

    def preprocess_many(self, posts: Iterable[RawPost]) -> List[PreprocessedPost]:
        out: List[PreprocessedPost] = []
        for p in posts:
            pp = self.preprocess_one(p)
            if pp is not None:
                out.append(pp)
        return out


class FakeEmbedder:
    """Deterministic vectors + captures input texts."""
    def __init__(self, dim: int = 8):
        self.dim = dim
        self.last_texts: List[str] = []

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        self.last_texts = list(texts)
        out: List[np.ndarray] = []
        for i in range(len(texts)):
            v = np.zeros(self.dim, dtype=float)
            v[i % self.dim] = 1.0
            out.append(v)
        return out


def test_pipeline_preserves_order_and_maps_fields():
    pre = FakePreprocessor()
    emb = FakeEmbedder(dim=8)
    pipe = EmbeddingPipeline(preprocessor=pre, embedder=emb)

    t0 = datetime(2026, 1, 6, tzinfo=timezone.utc)
    posts = [
        RawPost("p1", "alice", t0, "Hello World", "h1", {"lang": "en"}),
        RawPost("p2", "bob", t0, "  SECOND post  ", "h2", {"lang": "en"}),
    ]

    result = pipe.run(posts)

    assert len(result) == 2
    assert all(isinstance(r, EmbeddedPost) for r in result)

    # pipeline should embed clean_text in-order
    assert emb.last_texts == ["hello world", "second post"]

    # Check mapping
    r0 = result[0]
    assert r0.post_id == "p1"
    assert r0.author == "alice"
    assert r0.timestamp == t0
    assert r0.clean_text == "hello world"
    assert r0.clean_tokens == ["hello", "world"]
    assert r0.content_hash == "h1"
    assert r0.metadata == {"lang": "en"}

    # embedding is List[float] in your model
    assert isinstance(r0.embedding, list)
    assert len(r0.embedding) == 8

    r1 = result[1]
    assert r1.post_id == "p2"
    assert r1.clean_text == "second post"
    assert r1.clean_tokens == ["second", "post"]
    assert r1.content_hash == "h2"


def test_pipeline_filters_out_none_from_preprocessor():
    pre = FakePreprocessor()
    emb = FakeEmbedder(dim=8)
    pipe = EmbeddingPipeline(preprocessor=pre, embedder=emb)

    t0 = datetime(2026, 1, 6, tzinfo=timezone.utc)
    posts = [
        RawPost("p1", "alice", t0, "KEEP ME", "h1", {}),
        RawPost("p2", "alice", t0, "DROP THIS POST", "h2", {}),
        RawPost("p3", "alice", t0, "ALSO KEEP", "h3", {}),
    ]

    result = pipe.run(posts)

    assert [r.post_id for r in result] == ["p1", "p3"]
    assert emb.last_texts == ["keep me", "also keep"]
    assert [r.clean_tokens for r in result] == [["keep", "me"], ["also", "keep"]]


def test_pipeline_empty_input_returns_empty_list():
    pre = FakePreprocessor()
    emb = FakeEmbedder(dim=8)
    pipe = EmbeddingPipeline(preprocessor=pre, embedder=emb)

    assert pipe.run([]) == []


def test_pipeline_raises_if_embedder_returns_wrong_count():
    class BadEmbedder:
        def embed(self, texts: List[str]) -> List[np.ndarray]:
            return [np.zeros(8, dtype=float)]  # always wrong length

    pre = FakePreprocessor()
    pipe = EmbeddingPipeline(preprocessor=pre, embedder=BadEmbedder())

    t0 = datetime(2026, 1, 6, tzinfo=timezone.utc)
    posts = [
        RawPost("p1", "alice", t0, "one", "h1", {}),
        RawPost("p2", "bob", t0, "two", "h2", {}),
    ]

    with pytest.raises(ValueError):
        pipe.run(posts)
