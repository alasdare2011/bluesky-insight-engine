import numpy as np
import pytest
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict

# These imports should FAIL right now (good).
from beie.module3.models import ClusteredPost, Cluster
from beie.module3 import ClusteringPipeline
from beie.module3.strategies import ClusteringStrategy


# Minimal stand-in for EmbeddedPost (keeps tests isolated from Module 2)
@dataclass(frozen=True)
class EmbeddedPost:
    post_id: str
    author: str
    timestamp: datetime
    clean_text: str
    embedding: np.ndarray
    metadata: Dict[str, Any]


class FakeClusteringStrategy(ClusteringStrategy):
    """
    A deterministic fake strategy:
    - fit() does nothing but records it was called
    - predict() returns a fixed label sequence (provided at init)
    """

    def __init__(self, labels: list[int]):
        self._labels = np.array(labels, dtype=int)
        self.fit_called = False

    def fit(self, embeddings: np.ndarray) -> None:
        self.fit_called = True

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        # Defensive: make sure we don't silently mismatch length
        if len(embeddings) != len(self._labels):
            raise ValueError("Fake labels length must match number of embeddings.")
        return self._labels


@pytest.fixture
def sample_posts() -> list[EmbeddedPost]:
    # 6 posts, 3 clusters: labels could be [0,0,1,1,2,2]
    dim = 4
    now = datetime.now(timezone.utc)

    # Construct embeddings that are easy to reason about
    embs = [
        np.array([1, 0, 0, 0], dtype=float),
        np.array([1, 0, 0, 0], dtype=float),
        np.array([0, 1, 0, 0], dtype=float),
        np.array([0, 1, 0, 0], dtype=float),
        np.array([0, 0, 1, 0], dtype=float),
        np.array([0, 0, 1, 0], dtype=float),
    ]

    posts = []
    for i, e in enumerate(embs):
        posts.append(
            EmbeddedPost(
                post_id=f"p{i}",
                author="did:example:123",
                timestamp=now,
                clean_text=f"post {i}",
                embedding=e,
                metadata={"i": i},
            )
        )
    return posts


def test_pipeline_calls_strategy_fit_and_predict(sample_posts):
    labels = [0, 0, 1, 1, 2, 2]
    strat = FakeClusteringStrategy(labels=labels)
    pipeline = ClusteringPipeline(strategy=strat)

    clustered_posts, clusters = pipeline.run(sample_posts)

    assert strat.fit_called is True
    assert len(clustered_posts) == len(sample_posts)
    assert len(clusters) == 3


def test_clustered_posts_have_expected_fields(sample_posts):
    labels = [0, 0, 1, 1, 2, 2]
    pipeline = ClusteringPipeline(strategy=FakeClusteringStrategy(labels=labels))

    clustered_posts, _ = pipeline.run(sample_posts)

    # Type + structure checks
    assert all(isinstance(cp, ClusteredPost) for cp in clustered_posts)

    # Spot-check a couple items
    cp0 = clustered_posts[0]
    assert cp0.post_id == "p0"
    assert cp0.cluster_id == 0
    assert cp0.clean_text == "post 0"
    assert isinstance(cp0.embedding, np.ndarray)
    assert cp0.metadata["i"] == 0


def test_clusters_sizes_sum_to_number_of_posts(sample_posts):
    labels = [0, 0, 1, 1, 2, 2]
    pipeline = ClusteringPipeline(strategy=FakeClusteringStrategy(labels=labels))

    _, clusters = pipeline.run(sample_posts)

    assert all(isinstance(c, Cluster) for c in clusters)
    assert sum(c.size for c in clusters) == len(sample_posts)


def test_clusters_have_correct_member_ids(sample_posts):
    labels = [0, 0, 1, 1, 2, 2]
    pipeline = ClusteringPipeline(strategy=FakeClusteringStrategy(labels=labels))

    _, clusters = pipeline.run(sample_posts)

    by_id = {c.cluster_id: c for c in clusters}

    assert set(by_id[0].member_post_ids) == {"p0", "p1"}
    assert set(by_id[1].member_post_ids) == {"p2", "p3"}
    assert set(by_id[2].member_post_ids) == {"p4", "p5"}


def test_centroid_shape_matches_embedding_dimension(sample_posts):
    labels = [0, 0, 1, 1, 2, 2]
    pipeline = ClusteringPipeline(strategy=FakeClusteringStrategy(labels=labels))

    _, clusters = pipeline.run(sample_posts)

    dim = sample_posts[0].embedding.shape[0]
    for c in clusters:
        assert isinstance(c.centroid, np.ndarray)
        assert c.centroid.shape == (dim,)


def test_centroid_values_are_mean_of_member_embeddings(sample_posts):
    labels = [0, 0, 1, 1, 2, 2]
    pipeline = ClusteringPipeline(strategy=FakeClusteringStrategy(labels=labels))

    _, clusters = pipeline.run(sample_posts)
    by_id = {c.cluster_id: c for c in clusters}

    # For cluster 0: both embeddings are [1,0,0,0], centroid should equal that.
    assert np.allclose(by_id[0].centroid, np.array([1, 0, 0, 0], dtype=float))
    assert np.allclose(by_id[1].centroid, np.array([0, 1, 0, 0], dtype=float))
    assert np.allclose(by_id[2].centroid, np.array([0, 0, 1, 0], dtype=float))


def test_pipeline_raises_if_labels_length_mismatch(sample_posts):
    labels = [0, 1]  # wrong length
    pipeline = ClusteringPipeline(strategy=FakeClusteringStrategy(labels=labels))

    with pytest.raises(ValueError):
        pipeline.run(sample_posts)
