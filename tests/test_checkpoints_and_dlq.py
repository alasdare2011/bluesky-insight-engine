import respx, httpx
from beie.ingestion.fetcher import BlueskyPostFetcher

@respx.mock
def test_checkpoint_is_saved_and_loaded(store, dedupe, sample_payload):
    respx.get("https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts").mock(
        return_value=httpx.Response(200, json={"posts":[sample_payload], "cursor":"NEXT"})
    )
    f = BlueskyPostFetcher(None, "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts", 120, store, dedupe)
    f.load_checkpoint()  # should return None first call (no crash)
    f.fetch_posts(params={"q":"HR"})
    assert store.load_checkpoint() == "NEXT"

def test_malformed_payload_goes_to_dlq(store, dedupe):
    from beie.ingestion.fetcher import BlueskyPostFetcher
    f = BlueskyPostFetcher(None, "http://example", 120, store, dedupe)
    bad = {"uri": 123, "createdAt": "not-a-time"}  # intentionally wrong types
    f._handle_record(bad)  # internal parse → RawPost.from_payload → SchemaValidationError → DLQ
    assert len(store.dlq) == 1
    assert store.dlq[0]["reason"].lower().startswith("schema")
