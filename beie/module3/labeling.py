# beie/module3/labeling.py
from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Sequence

GREETINGS_LABEL = "(greetings)"

# Optional: keep a tiny “domain-noise” denylist if you want extra filtering
# AFTER Module 2. Keep it small.
LABEL_DENYLIST = {
    "bluesky", "bsky", "today", "people",
    "new", "year", "happy", "thank", "thanks", "ok", "yeah", "appreciated",
}

def top_terms_from_tokens(tokens_seq: Sequence[Sequence[str]], k: int = 6) -> List[str]:
    c = Counter()
    for toks in tokens_seq:
        for t in toks:
            if len(t) < 3:
                continue
            if t.isdigit():
                continue
            if t in LABEL_DENYLIST:
                continue
            c[t] += 1
    return [w for w, _ in c.most_common(k)]

def _looks_like_greetings(tokens_seq: Sequence[Sequence[str]]) -> bool:
    """
    Heuristic: if we have basically no content terms after filtering,
    assume this is greetings / short chatter.
    """
    return len(top_terms_from_tokens(tokens_seq, k=1)) == 0

def label_tokens(tokens_seq: Sequence[Sequence[str]], k: int = 6) -> List[str]:
    terms = top_terms_from_tokens(tokens_seq, k=k)
    if terms:
        return terms
    if _looks_like_greetings(tokens_seq):
        return [GREETINGS_LABEL]
    return ["(no_terms)"]

def label_reps(
    reps: Iterable[object],
    k: int = 6,
    tokens_attr: str = "clean_tokens",
) -> List[str]:
    """
    reps: iterable of objects with .clean_tokens (from Module 2 / EmbeddedPost / ClusteredPost)
    """
    tokens_seq: List[List[str]] = []
    for r in reps:
        toks = getattr(r, tokens_attr, None)
        if isinstance(toks, list) and toks:
            # ensure all are strings
            toks2 = [t for t in toks if isinstance(t, str) and t]
            if toks2:
                tokens_seq.append(toks2)

    return label_tokens(tokens_seq, k=k)
