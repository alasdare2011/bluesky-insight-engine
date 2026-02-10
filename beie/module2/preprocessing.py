from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Set

from beie.ingestion.models import RawPost
from .models import PreprocessedPost

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_MENTION_RE = re.compile(r"@[^\s]+")           # removes @user.bsky.social
_NON_WORD_RE = re.compile(r"[^a-z0-9\s]+")     # strips punctuation/emojis

def make_clean_tokens(text: str, cfg: "PreprocessConfig") -> List[str]:

        t = text.strip()

        if cfg.strip_urls:
            t = _URL_RE.sub(" ", t)

        if cfg.strip_mentions:
            t = _MENTION_RE.sub(" ", t)

        if cfg.lowercase:
            t = t.lower()

        if cfg.strip_non_words:
            t = _NON_WORD_RE.sub(" ", t)

        t = _WS_RE.sub(" ", t).strip()

        toks = [tok for tok in t.split(" ") if tok and tok not in cfg.stopwords]
        return toks

def make_clean_text(tokens: List[str]) -> str:
    return " ".join(tokens)


@dataclass
class PreprocessConfig:
    """
    Configuration for text preprocessing behavior.

    This config controls how raw post text is normalized before embedding.
    It is intentionally simple and deterministic to keep preprocessing
    transparent and testable.

    Attributes:
        lowercase:
            Whether to convert text to lowercase.
        strip_urls:
            Whether to remove URLs from the text.
        min_chars:
            Minimum number of characters required after cleaning for a post
            to be kept. Posts shorter than this threshold are dropped.
    """

    lowercase: bool = True
    strip_urls: bool = True
    strip_mentions: bool = True
    strip_non_words: bool = True
    min_tokens: int = 2

    stopwords: Set[str] = field(default_factory=set)

class PostPreprocessor:
    """
    Text preprocessing component for Module 2.

    This class converts `RawPost` objects from Module 1 into
    `PreprocessedPost` objects suitable for embedding.

    Responsibilities:
      - Clean and normalize raw text content
      - Apply simple filtering rules (e.g., minimum text length)
      - Preserve original content and metadata for traceability

    The preprocessor is intentionally conservative:
      - It does not attempt semantic transformations
      - It avoids language-specific logic
      - It prioritizes determinism and reproducibility
    """
     
    def __init__(self, config: PreprocessConfig | None = None) -> None:
        """
        Initialize the preprocessor.

        Args:
            config:
                Optional PreprocessConfig instance. If None, defaults
                to a new PreprocessConfig with standard settings.
        """
        self.config = config or PreprocessConfig()

    def preprocess_one(self, post: RawPost) -> PreprocessedPost | None:
        """
        Preprocess a single RawPost.

        Behavior:
          - Cleans the post content using `clean_content`
          - Drops the post if the cleaned content is shorter than `min_chars`
          - Returns a PreprocessedPost otherwise

        Args:
            post:
                RawPost instance from Module 1 ingestion.

        Returns:
            A PreprocessedPost if the post passes preprocessing filters;
            None if the post should be dropped.
        """
        tokens = make_clean_tokens(post.content, self.config)

        if len(tokens) < self.config.min_tokens:
            return None

        clean_text = make_clean_text(tokens)

        return PreprocessedPost(
            post_id=post.post_id,
            author=post.author,
            timestamp=post.timestamp,
            original_content=post.content,
            clean_text=clean_text,
            clean_tokens=tokens,
            content_hash=post.content_hash,
            metadata=dict(post.metadata),
        )
    
    def preprocess_many(self, posts: Iterable[RawPost]) -> List[PreprocessedPost]:
        """
        Preprocess a collection of RawPosts.

        This method applies `preprocess_one` to each post and collects
        only those that pass preprocessing filters.

        Args:
            posts:
                Iterable of RawPost instances.

        Returns:
            List of PreprocessedPost objects that passed preprocessing.
        """
        out: List[PreprocessedPost] = []
        for p in posts:
            pp = self.preprocess_one(p)
            if pp:
                out.append(pp)
        return out

    