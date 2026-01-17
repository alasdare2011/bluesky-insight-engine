import pytest
from datetime import datetime, timezone
from beie.ingestion.models import RawPost

def test_raw_post_constructs_and_is_immutable(sample_payload):
    rp = RawPost(
        post_id=sample_payload["uri"],
        author=sample_payload["author"]["did"],
        timestamp=datetime(2025, 10, 11, 11, 59, tzinfo=timezone.utc),
        content=sample_payload["text"],
        metadata={
            "cid": sample_payload["cid"],
            "author_handle": sample_payload["author"]["handle"],
            "lang": sample_payload["langs"][0],
            "tags": {"hashtags": sample_payload["tags"], "mentions": [], "links": []},
        },
    )
    assert rp.post_id == sample_payload["uri"]
    with pytest.raises(AttributeError):
        rp.post_id = "mutate-not-allowed"

def test_raw_post_has_stable_content_hash(sample_payload):
    from beie.ingestion.models import RawPost
    rp1 = RawPost.from_payload(sample_payload)
    rp2 = RawPost.from_payload(sample_payload)
    assert rp1.content_hash == rp2.content_hash
