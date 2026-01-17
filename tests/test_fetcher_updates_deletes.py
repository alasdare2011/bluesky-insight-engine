import respx, httpx
from beie.ingestion.fetcher import BlueskyPostFetcher

@respx.mock
def test_update_same_uri_new_cid_marks_as_update(store, dedupe, sample_payload):
    newer = dict(sample_payload)
    newer["cid"] = "bafy999"  # updated record
    respx.get("https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts").mock(
        return_value=httpx.Response(200, json={"posts":[sample_payload, newer]})
    )
    f = BlueskyPostFetcher(None, "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts", 120, store, dedupe)
    posts = f.fetch_posts(params={"q":"HR"})
    assert len(posts) == 2
    assert store.posts[sample_payload["uri"]].metadata["cid"] == "bafy999"
    assert any(op.get("op_type") == "update" for op in store.ops_log)

def test_delete_sets_soft_delete_flag(store):
    # Simulate you exposing a public method to process tombstones:
    # f.process_delete(uri, cid)
    from beie.ingestion.fetcher import BlueskyPostFetcher
    f = BlueskyPostFetcher(None, "http://example", 120, store, dedupe={})
    # Preload a post:
    class P: pass
    p = P(); p.post_id = "at://did:plc:1/app.bsky.feed.post/xyz"; p.metadata = {"cid":"c1"}
    store.upsert_post(p)
    f.process_delete(uri=p.post_id, cid="c2")
    # Expect soft delete
    assert getattr(store.posts[p.post_id], "deleted", True) is True
