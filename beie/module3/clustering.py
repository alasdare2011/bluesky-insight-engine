
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple
import numpy as np

from .models import Cluster, ClusteredPost
from .strategies import ClusteringStrategy

class ClusteringPipeline:
    def __init__(self, strategy: ClusteringStrategy):
        self.strategy = strategy

    def run(self, posts: List[Any]) -> Tuple[List[ClusteredPost]
                                             , List[Cluster]]:
        """
        posts: expects objects with attributes:
            - post_id: str
            - clean_text: str
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
            raise ValueError("Strategy returned a label " \
            "count that does not match posts length.")
        
        clustered_posts: List[ClusteredPost] = []
        for post, label in zip(posts, labels):
            clustered_posts.append(
                ClusteredPost(
                    post_id=post.post_id,
                    cluster_id=int(label),
                    embedding=post.embedding,
                    clean_text=self._get_clean_text(post),
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
        for attr in ("cleaned_content", "clean_text", "content", "text"):
            val = getattr(post, attr, None)
            if isinstance(val, str):
                return val
        return ""
