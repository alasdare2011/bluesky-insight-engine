import json
import time
from typing import Any, Dict, List, Optional
from dataclasses import asdict

class DevStore:
    """
    Development-only in-memory persistence layer.

    This class acts as a lightweight stand-in for production storage. It is
    designed for local development and unit tests where you want to validate
    pipeline behavior without needing a database or external services.

    Responsibilities:
      - Store posts in-memory (keyed by post_id)
      - Store author metadata in-memory (keyed by author DID)
      - Maintain an append-only operation log for debugging/testing
      - Provide a Dead Letter Queue (DLQ) for failed/malformed payloads:
          * always appended to an in-memory list (test assertions)
          * best-effort appended to a JSONL file (developer inspection)
      - Maintain a simple cursor checkpoint in-memory

    Attributes:
        posts:
            Dict keyed by post_id, storing a dict representation of the post.
            Values are typically produced via dataclasses.asdict(post).
        authors:
            Dict keyed by author DID. Values are dicts containing handle,
            display_name, and last_seen timestamp.
        ops:
            Append-only list of operations/events recorded by the pipeline.
            Each op is stamped with a float unix timestamp (ts).
        dlq:
            In-memory list of DLQ entries. Useful for tests to assert failures
            were routed correctly.
        dlq_path:
            Path to a JSONL file used for best-effort DLQ persistence.
        _checkpoint:
            Cursor-style checkpoint used by ingestion to resume paging.
    """


    def __init__(self, dlq_path: str = "dlq.jsonl"):
        """
        Initialize an empty DevStore.

        Args:
            dlq_path:
                File path used to append DLQ entries in JSON Lines format.
                This is best-effort only; failures to write will be ignored.
        """
        self.posts: Dict[str, Dict[str, Any]] = {}       # post_id -> RawPost dict
        self.authors: Dict[str, Dict[str, Any]] = {}     # author_did -> metadata
        self.ops: List[Dict[str, Any]] = []              # op log
        self.dlq: List[Dict[str, Any]] = []              # in-memory DLQ entries
        self.dlq_path = dlq_path
        self._checkpoint: Optional[str] = None

    # ----- persistence -----
    def upsert_post(self, post) -> None:
        """
        Insert or update a post record in the in-memory store.

        This method expects `post` to be a dataclass-like object compatible with
        `dataclasses.asdict()`. The post is keyed by its `post_id` attribute.

        Behavior:
            - If the post_id is new: inserts it.
            - If the post_id exists: overwrites the existing record.

        Args:
            post:
                A dataclass-like object with a `post_id` attribute.

        Returns:
            None
        """
        self.posts[getattr(post, "post_id")] = asdict(post)

    def upsert_author(
        self,
        author_did: str,
        author_handle: Optional[str],
        display_name: Optional[str],
        ts: int,
    ) -> None:
        """
        Insert or update author metadata in the in-memory store.

        The author is keyed by their DID (decentralized identifier). The
        `last_seen` field uses the timestamp provided by the caller rather than
        generating a new timestamp internally. This makes test behavior more
        deterministic and lets the ingestion layer control time semantics.

        Args:
            author_did:
                Stable DID for the author (key for the record).
            author_handle:
                Optional human-readable handle (may be None if unavailable).
            display_name:
                Optional display name (may be None if unavailable).
            ts:
                Timestamp (typically seconds since epoch) representing the time
                this author was observed / processed.

        Returns:
            None
        """
        self.authors[author_did] = {
            "handle": author_handle,
            "display_name": display_name,
            "last_seen": ts,  # use the provided timestamp
        }

    def append_op(self, op: Dict[str, Any]) -> None:
        """
        Append an operation/event record to the operation log.

        This is primarily intended for debugging and unit tests. The provided
        operation dict is copied and augmented with a `ts` field containing the
        current unix timestamp.

        Args:
            op:
                Arbitrary dictionary describing an operation. Common fields might
                include op_type, post_id/uri, cid, or stage name.

        Returns:
            None
        """
        op_with_time = {**op, "ts": time.time()}
        self.ops.append(op_with_time)

    # Optional: support soft deletes for tombstone tests
    def soft_delete(self, uri: str, cid: str) -> None:
        """
        Soft-delete a post (tombstone behavior) and record a delete operation.

        Some ingestion sources emit deletion ("tombstone") events that provide
        identifiers but not the full post content. This method supports those
        cases by either:
          - Marking an existing post as soft_deleted, OR
          - Creating a minimal tombstone record if the post is not present.

        A delete operation is always appended to `ops`.

        Args:
            uri:
                Identifier for the post to delete. In this store, `uri` is also
                treated as the post_id key.
            cid:
                Content identifier associated with the delete event.

        Returns:
            None
        """
        existing = self.posts.get(uri)
        if existing is None:
            self.posts[uri] = {
                "uri": uri,
                "post_id": uri,
                "author": "",
                "timestamp": 0,
                "metadata": {"cid": cid, "soft_deleted": True},
                "extras": {},
                "created_at": None,
                "text": "",
            }
        else:
            md = existing.get("metadata") or {}
            md["soft_deleted"] = True
            md["cid"] = cid
            existing["metadata"] = md
        self.append_op({"op_type": "delete", "uri": uri, "cid": cid})

    # ----- DLQ -----
    def send_to_dlq(self, payload: Dict[str, Any], reason: str) -> None:
        """
        Record a failed or malformed payload in the Dead Letter Queue (DLQ).

        The DLQ is intended for:
          - tests: asserting failures were captured (in-memory list)
          - local debugging: inspecting dlq.jsonl output on disk

        The method is intentionally resilient:
          - It always appends to the in-memory `dlq` list.
          - It attempts to append to `dlq_path` as JSONL.
          - Any file I/O errors are swallowed to avoid breaking dev/test runs.

        Args:
            payload:
                The input dictionary that failed processing (kept as-is).
            reason:
                A human-readable reason string describing why it was sent to DLQ.

        Returns:
            None
        """
        entry = {"reason": reason, "payload": payload, "timestamp": time.time()}
        self.dlq.append(entry)
        try:
            with open(self.dlq_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            # File I/O should never fail the pipeline in tests/dev
            pass

    # ----- checkpoints -----
    def load_checkpoint(self) -> Optional[str]:
        """
        Load the current ingestion checkpoint cursor.

        Returns:
            The checkpoint cursor string if one has been saved; otherwise None.
        """
        return self._checkpoint

    def save_checkpoint(self, cursor: Optional[str]) -> None:
        """
        Save a checkpoint cursor for ingestion resume.

        Note:
            This method only saves truthy cursor values. If `cursor` is None or
            an empty string, the checkpoint is left unchanged. This prevents
            accidental checkpoint clearing during transient failures.

        Args:
            cursor:
                Cursor string to persist, or None if no cursor is available.

        Returns:
            None
        """
        if cursor:
            self._checkpoint = cursor
