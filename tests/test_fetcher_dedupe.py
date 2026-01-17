import respx, httpx
from beie.ingestion.fetcher import BlueskyPostFetcher

@respx.mock
def test_dedupe_by_uri_and_cid(store, dedupe, sample_payload):
    # Match any URL, but constrain the path via regex
    respx.get(path__regex=r".*searchPosts.*").mock(
        return_value=httpx.Response(200, json={"posts": [sample_payload, sample_payload]})
    )
    f = BlueskyPostFetcher(
        None,
        "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts",
        120,
        store,
        dedupe,
    )
    posts = f.fetch_posts(params={"q": "HR", "limit": 20})
    assert len(posts) == 1

