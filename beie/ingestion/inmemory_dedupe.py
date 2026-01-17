from typing import Optional, Tuple

class InMemoryDedupe:
    """
    Simple in-memory deduplication helper.

    This class tracks seen keys in a Python set and is intended for
    development, testing, or short-lived ingestion runs.

    Typical usage:
      - Construct a dedupe key such as (post_id, cid)
      - Call check_and_mark(key) before processing a record
      - If the key has already been seen, skip processing

    Characteristics:
      - Process-local only (not shared across workers or restarts)
      - Unbounded growth unless manually cleared
      - Fast O(1) average-time lookups

    Intended use cases:
      - Prevent duplicate processing within a single ingestion run
      - Avoid double-emitting ops during retries or pagination overlap
    
    Attributes:
        _seen:
            A set of deduplication keys that have already been observed.
            Keys are typically tuples like (post_id, cid).
    """

    def __init__(self):
        """
        Initialize an empty in-memory deduplication cache.
        """
        self._seen: set[Tuple[str, Optional[str]]] = set()

    def check_and_mark(self, key: Tuple[str, Optional[str]]) -> bool:
        """
        Check whether a key has already been seen, and mark it if not.

        This method is intended to be called atomically at the point where
        a record is about to be processed.

        Behavior:
          - If the key has already been seen:
              * return False
              * do NOT modify the internal state
          - If the key has not been seen:
              * add it to the internal set
              * return True

        Args:
            key:
                A hashable tuple uniquely identifying a record.
                Commonly (post_id, cid), where cid may be None.
        Returns:
            True if this is the first time the key is seen.
            False if the key has already been processed.
        """
        if key in self._seen:
            return False
        self._seen.add(key)
        return True
        
    def clear(self):
        """
        Clear all recorded deduplication keys.

        This resets the deduper to its initial empty state.
        Useful in tests or between ingestion runs.
        """
        self._seen.clear()