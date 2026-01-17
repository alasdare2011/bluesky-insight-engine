import pytest
from datetime import datetime, timezone
from freezegun import freeze_time

class InMemoryStore:
    """Fake persistence for posts/authors/ops_log; captures writes for assertions."""
    def __init__(self):
        self.posts = {}
        self.ops_log = []
        self.authors = {}
        self.checkpoint = None
        self.dlq = []

    def upsert_post(self, post):  # expects RawPost-like dict or object
        self.posts[post.post_id] = post

    def append_op(self, op):
        self.ops_log.append(op)

    def upsert_author(self, did, handle=None, display_name=None, ts=None):
        self.authors[did] = {"did": did, "handle": handle, "display_name": display_name, "ts": ts}

    def save_checkpoint(self, cursor):
        self.checkpoint = cursor

    def load_checkpoint(self):
        return self.checkpoint

    def send_to_dlq(self, payload, reason):
        self.dlq.append({"payload": payload, "reason": reason})

class InMemoryDedupe:
    def __init__(self):
        self.seen = set()

    def check_and_mark(self, key):
        if key in self.seen:
            return False
        self.seen.add(key)
        return True

@pytest.fixture
def store():
    return InMemoryStore()

@pytest.fixture
def dedupe():
    return InMemoryDedupe()

@pytest.fixture
def frozen_now():
    with freeze_time("2025-10-11 12:00:00", tz_offset=0) as ft:
        yield ft

@pytest.fixture
def sample_payload():
    return {
        "uri": "at://did:plc:123/app.bsky.feed.post/abc",
        "cid": "bafy123",
        "author": {"did": "did:plc:123", "handle": "alice.bsky.social"},
        "createdAt": "2025-10-11T11:59:00Z",
        "text": "We hit an onboarding bottleneck in HR tooling. #SaaS #HR",
        "langs": ["en"],
        "tags": ["SaaS", "HR"],
    }
