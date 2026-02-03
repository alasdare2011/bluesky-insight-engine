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
from beie.module2.embedding import SimpleHashEmbedder, SentenceTransformerEmbedder

from beie.module3.clustering import ClusteringPipeline
from beie.module3.strategies import KMeansClustering
from beie.module3.evaluation import choose_k_by_silhouette

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
    "year","now","will","one","much","can","yes","really","live", "happy","thank","thanks",
    "ok","yeah","yes","really","much","appreciated", "bluesky", "bsky", "today", "people",
    "there","more","what","when","who","why","how","also","too",
}

NAME_STOPWORDS = {
    "kitty","jane","helen","misty","eric","cindy","jo","ann","sharon","beccy","skylark",
    "lauren","marshall","michelle","mark", "erin", "roberta", "ceri",
}

GREETINGS = {
    "happy","year","new","hello","hi","hey","thanks","thank","welcome","appreciate",
    "fantastic","great","nice","ok","yeah","yes"
}

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def top_representatives(cluster_id: int, clustered_posts: list, clusters_by_id: dict, n: int = 3, min_chars: int = 80):
    centroid = clusters_by_id[cluster_id].centroid
    members = [cp for cp in clustered_posts if cp.cluster_id == cluster_id]

    # Prefer informative posts
    informative = [cp for cp in members if len(cp.clean_text) >= min_chars]
    if len(informative) >= n:
        members = informative  # only use informative if enough exist

    scored = [(cosine_sim(cp.embedding, centroid), cp) for cp in members]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [cp for _, cp in scored[:n]]


def top_terms(texts: list[str], k: int = 6) -> list[str]:
    tokens = []

    for text in texts:
        # 1) normalize full text
        text = text.lower()
        text = re.sub(r"http\S+", " ", text)        # strip URLs
        text = re.sub(r"[^a-z0-9\s]", " ", text)    # keep alphanum only

        # 2) tokenize
        for t in text.split():
            # 3) token-level filters
            if len(t) < 3:
                continue
            if "@" in t:
                continue
            if t == "com" or t.endswith((".com", ".org", ".net")):
                continue
            if t.isdigit():
                continue
            if t in STOPWORDS or t in NAME_STOPWORDS or t in GREETINGS:
                continue

            tokens.append(t)

    return [w for w, _ in Counter(tokens).most_common(k)]


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
    parser.add_argument("--clusters-out", type=str, default="clusters.jsonl", help="Output JSONL file for Cluster records.")
    parser.add_argument(
        "--clustered-posts-out",
        type=str,
        default="clustered_posts.jsonl",
        help="Output JSONL file for ClusteredPost records.",
    )
    parser.add_argument(
        "--embedder",
        choices=["hash", "st"],
        default="hash",
        help="Embedding backend: 'hash' (fast, toy) or 'st' (sentence-transformers).",
    )
    parser.add_argument(
        "--st-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name (used when --embedder st).",
    )
    parser.add_argument(
    "--k",
    type=str,
    default="auto",
    help="KMeans clusters: integer like 9, or 'auto' to choose by silhouette score.",
    )
    parser.add_argument("--k-min", type=int, default=3, help="Min k when --k auto.")
    parser.add_argument("--k-max", type=int, default=15, help="Max k when --k auto.")


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
    if args.embedder == "st":
        embedder = SentenceTransformerEmbedder(
            model_name=args.st_model,
            normalize=not args.no_normalize,
        )
    else:
        embedder = SimpleHashEmbedder(
            dim=args.dim,
            normalize=not args.no_normalize,
        )

    pipeline = EmbeddingPipeline(preprocessor=preprocessor, embedder=embedder)

    print("\n[module2] running preprocessing + embedding...")
    embedded_posts = pipeline.run(all_raw_posts)

    print(f"[module2] embedded={len(embedded_posts)} (raw_fetched={len(all_raw_posts)})")

    # ---- Module 3: Cluster ----
    embeddings = np.vstack([p.embedding for p in embedded_posts])
    n_posts = embeddings.shape[0]

    if args.k.strip().lower() == "auto":
        best_k, scores = choose_k_by_silhouette(
            embeddings,
            k_min=args.k_min,
            k_max=args.k_max,
            random_state=42,
        )

        if scores:
            print("\n[module3] silhouette scores:")
            for kk in sorted(scores):
                print(f"  k={kk:<2} score={scores[kk]:.4f}")
        else:
            print("\n[module3] silhouette: no usable scores")

        k = best_k
    else:
        try:
            k = int(args.k)
        except ValueError:
            raise SystemExit("--k must be an integer like 9, or 'auto'.")

        if k < 1:
            raise SystemExit("--k must be >= 1.")
        if k > n_posts:
            raise SystemExit("--k cannot exceed number of posts.")

    print(f"\n[module3] clustering embedded posts with KMeans (k={k})...")

    strategy = KMeansClustering(n_clusters=k, random_state=42)
    cluster_pipeline = ClusteringPipeline(strategy=strategy)
    clustered_posts, clusters = cluster_pipeline.run(embedded_posts)


    print(f"[module3] clustered_posts={len(clustered_posts)} clusters={len(clusters)}")

    # Quick summary: top clusters by size
    clusters_sorted = sorted(clusters, key=lambda c: c.size, reverse=True)
    clusters_by_id = {c.cluster_id: c for c in clusters}
    print("[module3] top clusters:")
    for c in clusters_sorted[:10]:
        print(f"  - cluster_id={c.cluster_id} size={c.size} sample_posts={c.member_post_ids[:3]}")
    
    # Build quick labels per cluster (top terms)
    posts_by_cluster = {}
    for cp in clustered_posts:
        posts_by_cluster.setdefault(cp.cluster_id, []).append(cp.clean_text)
    
    print("\n[module3] representative posts per cluster (centroid-nearest):")
    for c in clusters_sorted[:9]:
        # 1) Get more candidates than we print
        rep_candidates = top_representatives(
            c.cluster_id,
            clustered_posts,
            clusters_by_id,
            n=10,              # ← more candidates for labeling
        )

        # 2) Decide which texts are usable for labels
        def label_worthy(text: str) -> bool:
            if len(text) < 40:
                return False
            # crude ASCII filter (fixes Japanese-only reps)
            ascii_ratio = sum(1 for ch in text if ord(ch) < 128) / max(1, len(text))
            if ascii_ratio < 0.85:
                return False
            return True

        label_reps = [cp for cp in rep_candidates if label_worthy(cp.clean_text)]

        # 3) Build label from best available reps
        texts_for_label = [cp.clean_text for cp in label_reps[:3]]
        # If our reps are mostly tiny / greeting-like, don't label this cluster.
        total_chars = sum(len(t) for t in texts_for_label)
        if total_chars < 250:  # tune: 200–400
            terms = ["(greetings)"]
        else:
            terms = top_terms(texts_for_label, k=6)
            if not terms:
                terms = ["(no_terms)"]

        # 5) Print (still only show 3 reps)
        reps_to_print = rep_candidates[:3]

        print(f"\ncluster_id={c.cluster_id} size={c.size}")
        for cp in reps_to_print:
            print(f"  - {cp.clean_text[:240]}")

        print(f"  label={' / '.join(terms)}")

    print("\n[module3] cluster labels (top terms):")
    for c in clusters_sorted[:10]:
        reps = top_representatives(c.cluster_id, clustered_posts, clusters_by_id, n=3)
        texts_for_label = [cp.clean_text for cp in reps]
        total_chars = sum(len(t) for t in texts_for_label)

        if total_chars < 250:
            terms = ["(greetings)"]
        else:
            terms = top_terms(texts_for_label, k=6)
            if not terms:
                terms = ["(no_terms)"]

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
