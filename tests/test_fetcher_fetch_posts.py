import respx
import httpx
import pytest
from beie.ingestion.fetcher import BlueskyPostFetcher

@respx.mock
def test_fetch_posts_happy_path(store, dedupe, sample_payload, frozen_now):
    # Mock a public AppView endpoint
    route = respx.get("https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts").mock(
        return_value=httpx.Response(200, json={"posts":[sample_payload], "cursor":"CURSOR_1"})
    )
    f = BlueskyPostFetcher(
        auth_token=None,
        fetch_url="https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts",
        rate_limit=120,
        store=store,
        dedupe=dedupe,
    )
    posts = f.fetch_posts(params={"q":"HR OR onboarding", "limit": 10})
    assert route.called
    assert len(posts) == 1
    # persisted side effects
    assert sample_payload["uri"] in store.posts
    assert store.checkpoint == "CURSOR_1"
