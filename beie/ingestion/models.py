from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from typing import Any, Dict

def _parse_iso_utc(s: str | None) -> datetime:
    """
    Parse an ISO-8601 timestamp string into a timezone-aware UTC datetime.

    This helper function is intentionally permissive:
      - If the input is None or empty, it returns the current UTC time.
      - If the string ends with 'Z', it is normalized to '+00:00'.
      - If parsing fails for any reason, it falls back to the current UTC time.
      - If the parsed datetime is naive (no tzinfo), UTC is assumed.

    This design prevents ingestion failures due to malformed or missing
    timestamps while still ensuring all timestamps are UTC-normalized.

    Args:
        s:
            ISO-8601 timestamp string (e.g. "2024-01-01T12:34:56Z"),
            or None.

    Returns:
        A timezone-aware datetime in UTC.
    """
    if not s:
        return datetime.now(timezone.utc)
    
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)
    
@dataclass(frozen=True)
class RawPost:
    """
    Immutable representation of a raw post ingested from a remote source.

    `RawPost` is the canonical data structure used at the ingestion boundary.
    It represents a post *after* basic validation and parsing, but *before*
    downstream preprocessing, embedding, or clustering.

    Design characteristics:
      - Frozen (immutable) dataclass to prevent accidental mutation
      - Contains both normalized fields and a metadata dictionary for
        source-specific or extensible attributes
      - Includes a deterministic content hash for deduplication or change
        detection
    
    Fields:
        post_id:
            Stable identifier for the post (typically a URI).
        author:
            Author identifier (e.g. DID or handle).
        timestamp:
            UTC timestamp representing post creation time.
        content:
            Raw textual content of the post.
        metadata:
            Dictionary containing auxiliary metadata such as:
              - cid (content identifier)
              - author_handle
              - language(s)
              - tags (hashtags, mentions, links)
        content_hash:
            SHA-256 hash computed from (post_id, author, content).
            Automatically derived in __post_init__.
    """
    
    post_id: str
    author: str
    timestamp: datetime
    content: str
    metadata: Dict[str, Any] 
    content_hash: str = field(init=False)

    def __post_init__(self):
        """
        Compute and set the deterministic content hash.

        The hash is derived from:
            post_id | author | content

        Because the dataclass is frozen, object.__setattr__ is used to
        assign the computed hash.

        This hash can be used for:
          - Change detection
          - Deduplication
          - Integrity checks across pipeline stages
        """
        h = hashlib.sha256(f"{self.post_id}|{self.author}|{self.content}".encode("utf-8")).hexdigest()
        object.__setattr__(self, "content_hash", h)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> RawPost:
        """
        Construct a RawPost from a raw payload dictionary.

        This method encapsulates all schema interpretation logic for incoming
        post payloads. It is intentionally permissive and defensive to support
        ingestion from heterogeneous or evolving APIs.

        Parsing behavior:
          - post_id:
              * Uses payload["uri"] if present
              * Falls back to payload["post_id"]
              * Defaults to empty string if missing
          - author:
              * If payload["author"] is a dict, extracts:
                    - did (preferred)
                    - handle (fallback)
              * If payload["author"] is not a dict, coerces to string
          - timestamp:
              * Parsed from payload["createdAt"] using _parse_iso_utc
              * Falls back to current UTC on failure
        - content:
              * Taken from payload["text"], defaulting to empty string
          - language:
              * Accepts either:
                    - payload["langs"] (list)
                    - payload["lang"] (string)
              * Normalized to a single value or None
          - metadata:
              * cid
              * author_handle (if available)
              * langs
              * tags (hashtags, mentions, links)
            Args:
            payload:
                Raw dictionary representing a post from the remote API.

        Returns:
            A fully constructed RawPost instance.

        Raises:
            Any exception raised here is intended to be caught by the caller
            (e.g., fetcher) and routed to a DLQ.
        """
        uri = payload.get("uri") or payload.get("post_id") or ""
        author_val = payload.get("author") or {}
        if isinstance(author_val, dict):
            author = author_val.get("did") or author_val.get("handle") or ""
            author_handle = author_val.get("handle")
        else:
            author = str(author_val)
            author_handle = None
        ts = _parse_iso_utc(payload.get("createdAt"))
        text = payload.get("text", "")
        langs = payload.get("langs") or payload.get("lang") or []
        lang = langs[0] if isinstance(langs, list) and langs else (langs or None)
        tags = payload.get("tags") or []
        md = {
            "cid": payload.get("cid"),
            "author_handle": author_handle,
            "langs": lang,
            "tags": {"hashtags": tags, "mentions": [], "links": []},
        }
        return cls(
            post_id=uri,
            author=author,
            timestamp=ts,
            content=text,
            metadata=md,
        )   
