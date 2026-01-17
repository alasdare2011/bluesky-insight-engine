import httpx
import requests
import time
from .models import RawPost
from .rate_limiter import RateLimiter
from .inmemory_dedupe import InMemoryDedupe
from typing import Any, Dict, Optional, List

class BlueskyPostFetcher:
    """
    Fetches posts from a Bluesky-compatible HTTP endpoint and converts them into `RawPost` objects.

    This class is the ingestion boundary for remote post data. It is responsible for:
      - Making HTTP GET requests to a feed/endpoint (with optional bearer auth)
      - Retrying transient network failures with exponential backoff
      - Handling rate-limits (HTTP 429) and respecting `Retry-After`
      - Loading/saving a cursor-style checkpoint (via store if supported)
      - Validating/parsing payloads into `RawPost` (via RawPost.from_payload)
      - Sending malformed payloads to a DLQ (via store.send_to_dlq)
      - Optional language filtering
      - Optional deduplication via a dedupe component
      - Writing to the store:
          * upsert_post
          * upsert_author
          * append_op (create/update/delete)

    Expected store interface (duck-typed):
      - upsert_post(post) -> None
      - upsert_author(author_did, author_handle, display_name, ts) -> None
      - append_op(op_dict) -> None
      - send_to_dlq(payload_dict, reason_str) -> None
      - (optional) load_checkpoint() -> Optional[str]
      - (optional) save_checkpoint(cursor: Optional[str]) -> None
      - (optional) soft_delete(uri, cid) -> None
      - (optional, dev) posts: Dict[str, Any] (used to infer create vs update)

    Attributes:
        auth_token:
            Bearer token used for Authorization header. If falsy/empty, no auth header is sent.
        fetch_url:
            The HTTP endpoint to GET from.
        rate_limit:
            Rate limit budget passed into RateLimiter (semantics depend on RateLimiter implementation).
        store:
            Persistence / DLQ / checkpoint backend (e.g., DevStore in dev, DB-backed store in prod).
        dedupe:
            Deduplication strategy. Defaults to InMemoryDedupe.
        filters:
            Optional filters applied to parsed posts (currently supports language filtering).
        rate_limiter:
            RateLimiter instance used to decorate fetch_posts to enforce budget.
    """

    def __init__(self, auth_token: str, fetch_url: str, rate_limit: int, store, dedupe=None, filters=None):
        """
        Initialize a new BlueskyPostFetcher.

        Args:
            auth_token:
                Bearer token string for the remote API. If empty/None-like, auth header is omitted.
            fetch_url:
                URL to fetch posts from.
            rate_limit:
                Rate limiter budget. Passed into RateLimiter(rate_limit).
            store:
                Storage backend used for checkpointing, DLQ, author/post storage, and ops logging.
            dedupe:
                Optional dedupe object. If not provided, uses InMemoryDedupe.
                Must provide `check_and_mark(key) -> bool` if used.
            filters:
                Optional dict of filters. Example:
                    {"languages": ["en", "fr"]}
        """
        self.auth_token = auth_token
        self.fetch_url = fetch_url
        self.rate_limit = rate_limit
        self.store = store
        self.dedupe = dedupe or InMemoryDedupe()
        self.filters = filters or {}
        self.rate_limiter = RateLimiter(rate_limit)
        self.fetch_posts = self.rate_limiter.decorate(self.fetch_posts)

        # Apply rate limiting dynamically
        self.fetch_posts = self.rate_limiter.decorate(self.fetch_posts)

   
    def fetch_posts(self, params: Dict[str, Any]) -> list[RawPost]:
        """
        Fetch a page of posts from the remote API and return parsed `RawPost` objects.

        Implements:
          - Auth header injection (Bearer token)
          - Retry loop for transient network errors and rate limits
          - Cursor checkpoint persistence (save_checkpoint)
          - Per-payload parsing and validation via `_handle_record`

        Retry behavior:
          - Up to MAX_ATTEMPTS (3) for:
              * httpx.TimeoutException
              * httpx.TransportError
              * HTTP 429 responses
          - Timeout/transport errors use exponential backoff: sleep(2**tries)
          - 429 respects Retry-After header (default 1 second)

        Args:
            params:
                Query parameters passed to the remote endpoint.
                Commonly includes paging/cursor parameters.
        
         Returns:
            A list of `RawPost` objects that were successfully parsed and accepted by filters/dedupe.

        Raises:
            RuntimeError:
                If repeated network failures occur beyond MAX_ATTEMPTS.
                Or if a non-2xx HTTP error occurs (other than 429).
        """
        headers: Dict[str, str] = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        MAX_ATTEMPTS = 3
        tries = 0

        while(True):
            tries += 1
            try:
                response = httpx.get(self.fetch_url, headers=headers, params=params, timeout=30)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                if tries < MAX_ATTEMPTS:
                    time.sleep(2 ** tries)
                    continue    
                raise RuntimeError("Failed to fetch posts after multiple attempts") from e
            if response.status_code == 429 and tries < MAX_ATTEMPTS:
                retry_after = int(response.headers.get("Retry-After", "1"))
                time.sleep(max(0, retry_after))
                continue

            self.raise_for_status_or_fail(response)

            data = response.json()
            self.save_checkpoint(data.get("cursor"))
            payloads = self._extract_post_payloads(data)
            out: List[RawPost] = []
            for payload in payloads:
                rp = self._handle_record(payload)
                if rp:
                    out.append(rp)
            return out

            

    def raise_for_status_or_fail(self, resp: httpx.Response) -> None:
        """
        Convert non-success HTTP responses into exceptions with useful context.

        Behavior:
          - 2xx responses are treated as success (no exception).
          - 429 responses are ignored here, because rate limiting is handled in `fetch_posts`.
          - All other non-2xx responses raise a RuntimeError that includes:
              * HTTP status code
              * request URL
              * a short snippet of the response text

        Args:
            resp:
                The httpx.Response returned by the request.

        Returns:
            None

        Raises:
            RuntimeError:
                For non-2xx, non-429 responses.
        """
        # Success → nothing to do
        if 200 <= resp.status_code < 300:
            return

        # 429 is handled elsewhere (e.g. retry loop)
        if resp.status_code == 429:
            return

        # For all other errors, raise
        try:
            resp.raise_for_status()
        except Exception as e:
            # Optionally add more debug info
             snippet = (resp.text or "")[:200]
             raise RuntimeError(
                 f"HTTP {resp.status_code} for {resp.request.url}: {snippet}"
                 ) from e
        
    def load_checkpoint(self) -> str | None:
        """
        Load the current cursor checkpoint.

        This method delegates to the store if it implements `load_checkpoint()`.
        If the store does not support checkpoints, it falls back to an instance
        attribute `_checkpoint` (set by save_checkpoint).

        Returns:
            The checkpoint cursor string, or None if no checkpoint exists.
        """
        if hasattr(self.store, "load_checkpoint"):
            return self.store.load_checkpoint()
        return getattr(self, "_checkpoint", None)

    def save_checkpoint(self, cursor: str | None) -> None:
        """
        Save a cursor checkpoint used for paging/resume.

        This method delegates to the store if it implements `save_checkpoint(cursor)`.
        Otherwise it stores the cursor on the fetcher instance as `_checkpoint`.

        Args:
            cursor:
                Cursor string from the remote API response. If None, nothing is saved.

        Returns:
            None
        """
        if cursor is None:
            return
        if hasattr(self.store, "save_checkpoint"):
            self.store.save_checkpoint(cursor)
        else:
            self._checkpoint = cursor
    
    def _consume_budget(self) -> None:
        """
        Consume one unit of the rate limiter budget.

        This is a low-level helper that forwards to RateLimiter.consume_budget().
        Depending on your RateLimiter design, this may block/sleep or raise if
        budget is exceeded.

        Returns:
            None
        """
        self.rate_limiter.consume_budget()

    def process_delete(self, *, uri: str, cid: str) -> None:
        """
        Process a deletion/tombstone event for a post.

        Behavior:
          - Always logs a delete operation in the store via append_op.
          - If the store supports `soft_delete(uri, cid)`, it delegates to that.
          - Otherwise it falls back to upserting a minimal record marked as soft-deleted.

        Args:
            uri:
                The identifier for the deleted post (also treated as post_id in fallback mode).
            cid:
                Content identifier associated with the delete event.

        Returns:
            None
        """
        # operation log
        self.store.append_op({"op_type": "delete", "uri": uri, "cid": cid})

        # prefer dedicated soft delete if store supports it
        if hasattr(self.store, "soft_delete"):
            self.store.soft_delete(uri, cid)
            return

        # fallback: mark a minimal record as soft-deleted
        class _P: pass
        p = _P()
        p.post_id = uri
        p.author = ""
        p.timestamp = 0
        p.metadata = {"cid": cid, "soft_deleted": True}
        self.store.upsert_post(p)


    
    def _handle_record(self, payload: Dict[str, Any]) -> Optional[RawPost]:
        """
        Validate, parse, filter, dedupe, and persist a single post payload.

        Processing steps:
          1) Validate that payload contains a non-empty string `uri`.
             - If invalid: send to DLQ and return None.
          2) Parse payload into RawPost via RawPost.from_payload(payload).
             - If parsing fails: send to DLQ and return None.
          3) Apply optional language filters (filters["languages"]).
             - If language is present and not allowed: return None.
             - If language is missing: do not reject (permissive).
          4) Deduplicate using `dedupe.check_and_mark((post_id, cid))` if available.
             - If check_and_mark returns False: treat as duplicate and return None.
             - If dedupe errors: ignore and continue (permissive).
          5) Determine op_type ("create" vs "update") by comparing existing CID
             in the store (only works when store exposes a dict-like `posts`).
          6) Persist:
             - store.upsert_post(rp)
             - store.upsert_author(...)
             - store.append_op({...})
        Args:
            payload:
                Raw post payload dict from the remote API ("posts" list element).

        Returns:
            A `RawPost` if accepted and stored; otherwise None (invalid, filtered, or duplicate).
        """
        uri = payload.get("uri")
        if not isinstance(uri, str) or not uri:
            self.store.send_to_dlq(payload, "schema: InvalidURI")
            return None
        
        try:
            rp = RawPost.from_payload(payload)
        except Exception as e:
            # DLQ and skip
            self.store.send_to_dlq(payload, f"schema: {e.__class__.__name__}")
            return None
        # Apply filters
        languages = (self.filters or {}).get("languages")
        if languages:
            allowed = {l.lower() for l in languages}

            # Accept either metadata["lang"] == "en"
            # or metadata["langs"] == ["en", ...]
            lang_val = rp.metadata.get("lang")
            if not lang_val:
                lang_val = rp.metadata.get("langs")

            if isinstance(lang_val, list):
                lang_val = lang_val[0] if lang_val else ""

            lang = (lang_val or "").lower()

            # Only enforce when we actually have a language
            if lang and (lang not in allowed):
                return None

        
        # Deduplicate
        if self.dedupe and hasattr(self.dedupe, "check_and_mark"):
            key = (rp.post_id, rp.metadata.get("cid"))
            try:
                if not self.dedupe.check_and_mark(key):
                    return None
            except Exception:
                # permissive on cache hiccup
                pass

        op_type = "create"
        existing_post = None
        store_posts = getattr(self.store, "posts", None)

        if isinstance(store_posts, dict):
            existing_post = store_posts.get(rp.post_id)
            if existing_post is not None:
                # extract prior cid from RawPost or dict
                if hasattr(existing_post, "metadata"):
                    old_cid = (existing_post.metadata or {}).get("cid")
                elif isinstance(existing_post, dict):
                    old_cid = (existing_post.get("metadata") or {}).get("cid")
                else:
                    old_cid = None

                if old_cid is not None and old_cid != rp.metadata.get("cid"):
                    op_type = "update"   
        # Store the post
        self.store.upsert_post(rp)
        # Store author info
        self.store.upsert_author(
            rp.author,
            rp.metadata.get("author_handle"),
            None,
            rp.timestamp,
        )
        self.store.append_op({
            "op_type": op_type,
            "uri": rp.post_id,
            "cid": rp.metadata.get("cid"),
        })
        return rp
    
    def _extract_post_payloads(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Normalize different Bluesky feed response shapes into a flat list of
        post-like payloads compatible with RawPost.from_payload().

        Supported:
        - {"posts": [...]}  (some endpoints)
        - {"feed": [{"post": {...}}, ...]} (getTimeline and many feed endpoints)
        """
        # If endpoint already returns "posts", use it directly
        posts = data.get("posts")
        if isinstance(posts, list):
            return posts

        # Timeline/feed endpoints return "feed" items, each with a nested "post"
        feed = data.get("feed", [])
        out: List[Dict[str, Any]] = []

        for item in feed:
            post_view = (item or {}).get("post") or {}
            record = (post_view.get("record") or {})

            # Some feed items might not be normal posts; skip if missing essentials
            uri = post_view.get("uri")
            if not uri:
                continue

            out.append(
                {
                    "uri": uri,
                    "cid": post_view.get("cid"),
                    "author": post_view.get("author") or {},
                    "createdAt": record.get("createdAt"),
                    "text": record.get("text", ""),
                    "langs": record.get("langs") or record.get("lang") or [],
                    "tags": record.get("tags") or [],
                }
            )

        return out

    
    



    

    



