import respx, httpx
from beie.ingestion.fetcher import BlueskyPostFetcher

@respx.mock
def test_record_level_filters_drop_non_english(store, dedupe, sample_payload):
    non_en = dict(sample_payload)
    non_en["langs"] = ["fr"]
    respx.get("https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts").mock(
        return_value=httpx.Response(200, json={"posts":[sample_payload, non_en]})
    )
    f = BlueskyPostFetcher(
        None,
        "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts",
        120,
        store,
        dedupe,
        filters={"languages": ["en"]},
    )
    posts = f.fetch_posts(params={"q":"HR"})
    assert len(posts) == 1
