from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

@dataclass(frozen=True)
class PreprocessedPost:
    """
    Immutable representation of a post after text preprocessing.

    This is the intermediate data model between Module 1 ingestion (`RawPost`)
    and Module 2 embedding (`EmbeddedPost`).

    Typical pipeline role:
        RawPost -> PreprocessedPost -> EmbeddedPost

    Fields:
        post_id:
            Stable identifier for the post (URI / unique ID).
        author:
            Author identifier (e.g., DID or handle).
        timestamp:
            Timestamp associated with the post (typically creation time).
        original_content:
            The original raw text content (before cleaning/normalization).
        cleaned_content:
            The cleaned/normalized text content used for embedding.
        content_hash:
            Stable hash representing the post content identity. This is usually
            derived from the raw content and key identifiers upstream (Module 1),
            and is carried forward for traceability/dedupe.
        metadata:
            Free-form dict for extra information passed through the pipeline
            (language, tags, source fields, etc.).
    """
    post_id: str
    author: str
    timestamp: datetime
    original_content: str
    cleaned_content: str
    content_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class EmbeddedPost:
    """
    Immutable representation of a post after embedding.

    This model is produced by the embedding stage. It holds the cleaned text
    plus a numerical embedding vector used by downstream modules (e.g.,
    clustering, similarity search, insight generation).

    Typical pipeline role:
        PreprocessedPost -> EmbeddedPost -> (Clustering / Indexing)

    Fields:
        post_id:
            Stable identifier for the post.
        author:
            Author identifier (e.g., DID or handle).
        timestamp:
            Timestamp associated with the post.
        cleaned_content:
            The cleaned/normalized content used to generate the embedding.
        embedding:
            Embedding vector representing the post text. In production this will
            usually be a fixed-dimensional float vector (e.g., 384, 768, 1536).
            Note: in many implementations this is stored as a NumPy array, but
            this model uses `List[float]` for JSON-serializability.
        content_hash:
            Stable hash representing the post content identity. Carried forward
            for traceability and deduplication across stages.
        metadata:
            Free-form dict for extra information passed through the pipeline.
    """   
    post_id: str
    author: str
    timestamp: datetime
    cleaned_content: str
    embedding: List[float]
    content_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    