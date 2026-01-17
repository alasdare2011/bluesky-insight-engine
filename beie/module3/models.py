from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np

@dataclass(frozen=True)
class ClusteredPost:
    """
    Output of clustering for a single post.
    Kept intentionally small and immutable so it can be 
    safely passed downstream
    """
    post_id: str
    cluster_id: int
    embedding: np.ndarray
    clean_text: str
    metadata: Dict[str, Any]

@dataclass
class Cluster:
    """
    Aggregate information about a cluster.
    Mutable is fine here (we may enrich clusters later),
    but keep fields stable
    """

    cluster_id: int
    size: int
    centroid: np.ndarray
    member_post_ids: List[str]
