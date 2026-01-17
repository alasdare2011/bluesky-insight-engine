import time
import respx, httpx
from beie.ingestion.fetcher import BlueskyPostFetcher

@respx.mock
def test_retry_on_429_with_retry_after(store, dedupe, sample_payload):
    route = respx.get("https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts")
    route.side_effect = [
        httpx.Response(429, headers={"Retry-After":"1"}),
        httpx.Response(200, json={"posts":[sample_payload]})
    ]
    f = BlueskyPostFetcher(None, "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts", 60, store, dedupe)
    posts = f.fetch_posts(params={"q":"SaaS", "limit": 1})
    assert len(posts) == 1
    assert route.calls.call_count == 2

def test_token_bucket_enforces_rate_limit(monkeypatch, store, dedupe):
    from beie.ingestion.fetcher import BlueskyPostFetcher
    # Inject a fake clock and counter into fetcher; call should sleep when budget empty.
    slept = {"n": 0}
    def fake_sleep(s): slept["n"] += 1
    monkeypatch.setattr("time.sleep", fake_sleep)
    f = BlueskyPostFetcher(None, "http://example", rate_limit=1, store=store, dedupe=dedupe)
    # Assume fetcher._consume_budget() is called per request and sleeps if needed
    f._consume_budget()  # first allowed
    f._consume_budget()  # should trigger sleep
    assert slept["n"] >= 1
