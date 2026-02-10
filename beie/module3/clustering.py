from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np

from .models import Cluster, ClusteredPost
from .strategies import ClusteringStrategy


class ClusteringPipeline:
    def __init__(self, strategy: ClusteringStrategy):
        self.strategy = strategy

    def run(self, posts: List[Any]) -> Tuple[List[ClusteredPost], List[Cluster]]:
        """
        posts: expects objects with attributes:
            - post_id: str
            - clean_text: str (preferred) OR cleaned_content/content/text (fallback)
            - clean_tokens: List[str] (preferred) OR tokens (fallback)
            - embedding: np.ndarray
            - metadata: dict

        (We keep this duck-typed so tests can pass a local
        EmbeddedPost stand-in)
        """
        if not posts:
            return [], []

        embeddings = np.vstack([p.embedding for p in posts])

        self.strategy.fit(embeddings)
        labels = self.strategy.predict(embeddings)

        if len(labels) != len(posts):
            raise ValueError(
                "Strategy returned a label count that does not match posts length."
            )

        clustered_posts: List[ClusteredPost] = []
        for post, label in zip(posts, labels):
            clustered_posts.append(
                ClusteredPost(
                    post_id=post.post_id,
                    cluster_id=int(label),
                    embedding=post.embedding,
                    clean_text=self._get_clean_text(post),
                    clean_tokens=self._get_clean_tokens(post),
                    metadata=post.metadata,
                )
            )

        clusters = self._build_clusters(clustered_posts)
        return clustered_posts, clusters

    def _build_clusters(self, clustered_posts: List[ClusteredPost]) -> List[Cluster]:
        by_cluster: Dict[int, List[ClusteredPost]] = {}
        for cp in clustered_posts:
            by_cluster.setdefault(cp.cluster_id, []).append(cp)

        clusters: List[Cluster] = []
        for cluster_id, members in by_cluster.items():
            centroid = np.mean(np.vstack([m.embedding for m in members]), axis=0)

            clusters.append(
                Cluster(
                    cluster_id=cluster_id,
                    size=len(members),
                    centroid=centroid,
                    member_post_ids=[m.post_id for m in members],
                )
            )

        return clusters

    @staticmethod
    def _get_clean_text(post: Any) -> str:
        # Prefer Module 2's real field name first
        for attr in ("clean_text", "cleaned_content", "content", "text"):
            val = getattr(post, attr, None)
            if isinstance(val, str):
                return val
        return ""

    @staticmethod
    def _get_clean_tokens(post: Any) -> List[str]:
        # Prefer Module 2 field name(s) first
        for attr in ("clean_tokens", "tokens"):
            val = getattr(post, attr, None)
            if isinstance(val, list) and all(isinstance(t, str) for t in val):
                return val

        # Backward-compatible fallback: derive from clean_text
        txt = ClusteringPipeline._get_clean_text(post)
        if not isinstance(txt, str) or not txt:
            return []

        # Minimal fallback tokenization (no stopword logic here)
        return [t for t in txt.split() if t]
