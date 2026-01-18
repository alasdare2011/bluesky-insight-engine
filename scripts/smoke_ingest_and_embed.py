#!/usr/bin/env python3
"""
scripts/smoke_ingest_and_embed.py

Smoke test for Modules 1 + 2 + 3 against a *real* Bluesky HTTP feed.

What it does:
  1) Fetch N pages from Bluesky timeline (public by default, authenticated if env vars set)
  2) Runs Module 1 ingestion:
       - parses RawPost
       - writes to store (posts/authors/ops)
       - saves checkpoint
       - DLQ captures malformed payloads
  3) Runs Module 2 pipeline:
       - preprocess (clean text)
       - embed (SimpleHashEmbedder by default)
       - produces EmbeddedPost objects
  4) Runs Module 3 clustering:
       - clusters embedded posts using KMeans
       - produces ClusteredPost + Cluster objects
  5) Writes EmbeddedPosts + clustered outputs to JSONL files

Usage:
  python scripts/smoke_ingest_and_embed.py --pages 3 --limit 50 --out embedded.jsonl

Optional auth (to use your own PDS + authenticated timeline):
  export BSKY_PDS="https://bsky.social"
  export BSKY_IDENTIFIER="your-handle.bsky.social"
  export BSKY_PASSWORD="your-app-password"
  python scripts/smoke_ingest_and_embed.py --pages 2 --limit 30

Notes:
  - Public mode uses: https://public.api.bsky.app/xrpc/app.bsky.feed.getTimeline
  - Auth mode uses:   {BSKY_PDS}/xrpc/app.bsky.feed.getTimeline
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from dataclasses import asdict
from typing import Dict, Any, Optional, List
import re
from collections import Counter

import httpx
import numpy as np

from beie.ingestion.fetcher import BlueskyPostFetcher
from beie.ingestion.devStore import DevStore

from beie.module2.pipeline import EmbeddingPipeline
from beie.module2.preprocessing import PostPreprocessor, PreprocessConfig
from beie.module2.embedding import SimpleHashEmbedder

from beie.module3.clustering import ClusteringPipeline
from beie.module3.strategies import KMeansClustering

def json_safe(obj):
    """Recursively convert dataclasses/datetimes/numpy into JSON-serializable values."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj

STOPWORDS = {
    "the","a","an","and","or","but","to","of","in","on","for","with","as","at","by","from",
    "is","are","was","were","be","been","being","it","this","that","these","those","i","you",
    "we","they","he","she","them","his","her","their","our","my","your","me","us","not","no",
    "so","if","then","than","too","very","just","do","does","did","have","has","had", "new",
    "year","now","will","one","much","can","yes","really","live"
}

def top_terms(texts: list[str], k: int = 6) -> list[str]:
    tokens = []
    for t in texts:
        t = t.lower()
        t = re.sub(r"http\S+", " ", t)          # strip urls
        t = re.sub(r"[^a-z0-9\s#@]", " ", t)    # keep hashtags/mentions-ish
        words = [w for w in t.split() if len(w) >= 3 and w not in STOPWORDS]
        tokens.extend(words)
    return [w for w, _ in Counter(tokens).most_common(k)]


def pick_k(n_posts: int, override: int | None = None) -> int:
    """
    Choose number of clusters for KMeans.
    - If override is provided, use it (must be > 0)
    - Else use a sqrt heuristic clamped to a sensible range
    """
    if override is not None:
        if override <= 0:
            raise ValueError("--k must be a positive integer.")
        return override
    return int(np.clip(np.sqrt(n_posts), 8, 60))


def get_access_jwt(pds: str, identifier: str, password: str) -> str:
    """Login to Bluesky PDS and return an access JWT for Bearer auth."""
    resp = httpx.post(
        f"{pds}/xrpc/com.atproto.server.createSession",
        json={"identifier": identifier, "password": password},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["accessJwt"]


def build_fetcher(store: DevStore, rate_limit: int, languages: Optional[List[str]]) -> BlueskyPostFetcher:
    """
    Build a BlueskyPostFetcher either in public mode or authenticated mode,
    depending on environment variables.
    """
    pds = os.getenv("BSKY_PDS")
    identifier = os.getenv("BSKY_IDENTIFIER")
    password = os.getenv("BSKY_PASSWORD")

    # Filters passed into Module 1 fetcher (language filtering happens there)
    filters = {}
    if languages:
        filters["languages"] = languages

    if pds and identifier and password:
        token = get_access_jwt(pds, identifier, password)
        fetch_url = f"{pds}/xrpc/app.bsky.feed.getTimeline"
        print(f"[auth] Using PDS timeline endpoint: {fetch_url}")
        return BlueskyPostFetcher(
            auth_token=token,
            fetch_url=fetch_url,
            rate_limit=rate_limit,
            store=store,
            filters=filters,
        )

    # Default: public endpoint, no auth required
    fetch_url = "https://public.api.bsky.app/xrpc/app.bsky.feed.getTimeline"
    print(f"[public] Using public timeline endpoint: {fetch_url}")
    return BlueskyPostFetcher(
        auth_token="",
        fetch_url=fetch_url,
        rate_limit=rate_limit,
        store=store,
        filters=filters,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=3, help="How many pages to fetch (cursor paging).")
    parser.add_argument("--limit", type=int, default=50, help="Posts per page.")
    parser.add_argument("--rate-limit", type=int, default=120, help="Requests per minute (coarse).")
    parser.add_argument("--out", type=str, default="embedded.jsonl", help="Output JSONL file for EmbeddedPost.")
    parser.add_argument("--lang", action="append", default=None,
                        help="Language filter (repeatable). Example: --lang en --lang fr")
    parser.add_argument("--min-chars", type=int, default=5, help="Preprocessing min chars threshold.")
    parser.add_argument("--no-strip-urls", action="store_true", help="Disable URL stripping in preprocessing.")
    parser.add_argument("--no-lowercase", action="store_true", help="Disable lowercasing in preprocessing.")
    parser.add_argument("--dim", type=int, default=384, help="Embedding dimension for SimpleHashEmbedder.")
    parser.add_argument("--no-normalize", action="store_true", help="Disable L2 normalization in embeddings.")

    # ---- Module 3 args ----
    parser.add_argument("--k", type=int, default=None, help="Number of clusters for KMeans (overrides heuristic).")
    parser.add_argument("--clusters-out", type=str, default="clusters.jsonl", help="Output JSONL file for Cluster records.")
    parser.add_argument(
        "--clustered-posts-out",
        type=str,
        default="clustered_posts.jsonl",
        help="Output JSONL file for ClusteredPost records.",
    )

    args = parser.parse_args()

    store = DevStore()

    # ---- Module 1: Fetch real posts ----
    fetcher = build_fetcher(store=store, rate_limit=args.rate_limit, languages=args.lang)

    all_raw_posts = []
    cursor = store.load_checkpoint()

    for i in range(args.pages):
        params: Dict[str, Any] = {"limit": args.limit}
        if cursor:
            params["cursor"] = cursor

        print(f"\n[fetch] page={i+1}/{args.pages} params={params}")
        raw_posts = fetcher.fetch_posts(params)
        all_raw_posts.extend(raw_posts)

        cursor = store.load_checkpoint()
        print(f"[fetch] got={len(raw_posts)} total={len(all_raw_posts)} checkpoint={cursor!r}")
        print(f"[store] posts={len(store.posts)} authors={len(store.authors)} ops={len(store.ops)} dlq={len(store.dlq)}")

        # If the endpoint stops returning posts, break early
        if not raw_posts:
            print("[fetch] no posts returned; stopping early.")
            break

    if not all_raw_posts:
        print("\nNo RawPosts fetched. Check network, endpoint availability, or auth env vars.")
        return 2

    # ---- Module 2: Preprocess + Embed ----
    pre_cfg = PreprocessConfig(
        lowercase=not args.no_lowercase,
        strip_urls=not args.no_strip_urls,
        min_chars=args.min_chars,
    )
    preprocessor = PostPreprocessor(pre_cfg)
    embedder = SimpleHashEmbedder(dim=args.dim, normalize=not args.no_normalize)

    pipeline = EmbeddingPipeline(preprocessor=preprocessor, embedder=embedder)

    print("\n[module2] running preprocessing + embedding...")
    embedded_posts = pipeline.run(all_raw_posts)

    print(f"[module2] embedded={len(embedded_posts)} (raw_fetched={len(all_raw_posts)})")

    # ---- Module 3: Cluster ----
    k = pick_k(len(embedded_posts), args.k)
    print(f"\n[module3] clustering embedded posts with KMeans (k={k})...")

    strategy = KMeansClustering(n_clusters=k, random_state=42)
    cluster_pipeline = ClusteringPipeline(strategy=strategy)

    clustered_posts, clusters = cluster_pipeline.run(embedded_posts)

    print(f"[module3] clustered_posts={len(clustered_posts)} clusters={len(clusters)}")

    # Quick summary: top clusters by size
    clusters_sorted = sorted(clusters, key=lambda c: c.size, reverse=True)
    print("[module3] top clusters:")
    for c in clusters_sorted[:10]:
        print(f"  - cluster_id={c.cluster_id} size={c.size} sample_posts={c.member_post_ids[:3]}")
    
    # Build quick labels per cluster (top terms)
    posts_by_cluster = {}
    for cp in clustered_posts:
        posts_by_cluster.setdefault(cp.cluster_id, []).append(cp.clean_text)
    
    print("\n[module3] representative posts per cluster:")
    for c in clusters_sorted[:8]:
        texts = posts_by_cluster.get(c.cluster_id, [])
        print(f"\ncluster_id={c.cluster_id} size={c.size}")
        for t in texts[:3]:
            print(f"  - {t[:240]}")

    print("\n[module3] cluster labels (top terms):")
    for c in clusters_sorted[:10]:
        terms = top_terms(posts_by_cluster.get(c.cluster_id, []), k=6)
        print(f"  - cluster_id={c.cluster_id} size={c.size} label={' / '.join(terms)}")

    # ---- Write EmbeddedPosts output ----
    out_path = args.out
    with open(out_path, "w", encoding="utf-8") as f:
        for ep in embedded_posts:
            f.write(json.dumps(json_safe(asdict(ep)), ensure_ascii=False) + "\n")

    print(f"\nWrote {len(embedded_posts)} EmbeddedPost records to: {out_path}")

    # ---- Write clustered outputs ----
    with open(args.clustered_posts_out, "w", encoding="utf-8") as f:
        for cp in clustered_posts:
            f.write(json.dumps(json_safe(asdict(cp)), ensure_ascii=False) + "\n")
    print(f"Wrote ClusteredPost records to: {args.clustered_posts_out}")

    with open(args.clusters_out, "w", encoding="utf-8") as f:
        for c in clusters:
            f.write(json.dumps(json_safe(asdict(c)), ensure_ascii=False) + "\n")
    print(f"Wrote Cluster records to: {args.clusters_out}")

    # ---- Quick diagnostics ----
    if store.dlq:
        print(f"\n[dlq] entries={len(store.dlq)} (showing up to 3)")
        for e in store.dlq[:3]:
            reason = e.get("reason")
            print(f"  - {reason}")
    else:
        print("\n[dlq] empty ✅")

    # Show a small op summary
    if store.ops:
        last_ops = store.ops[-5:]
        print("\n[ops] last 5:")
        for op in last_ops:
            print(f"  - {op.get('op_type')} uri={op.get('uri')} cid={op.get('cid')}")
    else:
        print("\n[ops] none (unexpected if posts were stored)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
