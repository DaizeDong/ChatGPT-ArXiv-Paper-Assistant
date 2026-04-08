from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Iterable
from urllib.parse import urlsplit, urlunsplit

from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotCluster, HotspotItem
from arxiv_assistant.utils.hotspot.hotspot_sources import parse_datetime

STOPWORDS = {
    "a",
    "an",
    "and",
    "also",
    "apr",
    "aug",
    "dec",
    "feb",
    "for",
    "from",
    "here",
    "how",
    "in",
    "into",
    "jan",
    "jul",
    "jun",
    "mar",
    "may",
    "of",
    "on",
    "oct",
    "or",
    "plus",
    "sep",
    "special",
    "the",
    "to",
    "watch",
    "with",
    "2024",
    "2025",
    "2026",
}
GENERIC_OVERLAP_TOKENS = {
    # AI/ML domain generic terms
    "open",
    "source",
    "research",
    "agent",
    "agents",
    "model",
    "models",
    "paper",
    "papers",
    "coding",
    "code",
    "reasoning",
    "benchmark",
    "system",
    "systems",
    "data",
    "dataset",
    "training",
    "inference",
    "language",
    "learning",
    "neural",
    "network",
    "transformer",
    "generation",
    "evaluation",
    "performance",
    "framework",
    "tool",
    "tools",
    "platform",
    "release",
    "update",
    "latest",
    "announces",
    "introducing",
    "available",
    # High-frequency English words that pass len>=4 filter
    "local",
    "using",
    "built",
    "just",
    "works",
    "entirely",
    "about",
    "been",
    "best",
    "better",
    "comes",
    "does",
    "doesnt",
    "done",
    "every",
    "first",
    "gets",
    "good",
    "great",
    "heres",
    "high",
    "like",
    "long",
    "made",
    "make",
    "makes",
    "many",
    "more",
    "most",
    "much",
    "need",
    "never",
    "next",
    "only",
    "over",
    "real",
    "really",
    "right",
    "some",
    "still",
    "than",
    "that",
    "them",
    "then",
    "there",
    "they",
    "this",
    "time",
    "very",
    "want",
    "well",
    "what",
    "when",
    "will",
    "your",
    "isnt",
    "cant",
    "dont",
    "wont",
    "near",
    "even",
    "show",
    "test",
    "runs",
    "full",
    "ever",
    "here",
    "back",
    "free",
    "fast",
    "small",
    "large",
    "speed",
    "quality",
    # Financial/scale words that shouldn't count as meaningful overlap
    "billion",
    "million",
}

SOURCE_ROLE_WEIGHTS = {
    "official_news": 7.0,
    "research_backbone": 6.0,
    "editorial_depth": 4.0,
    "paper_trending": 5.0,
    "github_trend": 4.2,
    "builder_momentum": 3.5,
    "community_heat": 1.5,
    "headline_consensus": 1.2,
    "hn_discussion": 1.5,
}

PAPER_LIKE_SOURCE_TYPES = {"paper"}


DOMAIN_CANONICAL_MAP = {
    "paperswithcode.com": "paperswithcode",
    "huggingface.co": "huggingface",
    "github.com": "github",
    "arxiv.org": "arxiv",
}

# Company/product name patterns for entity matching
ENTITY_PATTERNS = [
    re.compile(r"\b(openai|gpt[-\s]?\d|o[1-3]|chatgpt)\b", re.I),
    re.compile(r"\b(anthropic|claude[\s-]?\d?)\b", re.I),
    re.compile(r"\b(google|deepmind|gemini|bard)\b", re.I),
    re.compile(r"\b(meta|llama[\s-]?\d?)\b", re.I),
    re.compile(r"\b(mistral[\s-]?\w*)\b", re.I),
    re.compile(r"\b(deepseek[\s-]?\w*)\b", re.I),
    re.compile(r"\b(qwen[\s-]?\d?)\b", re.I),
    re.compile(r"\b(nvidia|cuda)\b", re.I),
    re.compile(r"\b(microsoft|copilot|phi[-\s]?\d)\b", re.I),
    re.compile(r"\b(stability[\s-]?ai|stable[\s-]?diffusion)\b", re.I),
]


def _extract_entities(text: str) -> set[str]:
    entities = set()
    for pattern in ENTITY_PATTERNS:
        for match in pattern.finditer(text):
            entities.add(match.group(0).lower().strip())
    return entities


def canonicalize_url(url: str) -> str:
    if "arxiv.org/abs/" in url:
        arxiv_id = url.rsplit("/", 1)[-1]
        return f"https://arxiv.org/abs/{arxiv_id}"
    parts = list(urlsplit(url))
    parts[3] = ""
    parts[4] = ""
    return urlunsplit(parts).rstrip("/")


def normalize_title(title: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", title.lower())
    filtered = [token for token in tokens if token not in STOPWORDS]
    return " ".join(filtered)


def significant_title_tokens(title: str) -> set[str]:
    raw = normalize_title(title).split()
    # Normalize number+scale tokens before length filter: "122b" → "122"
    normalized = [re.sub(r'^(\d+)[bmk]$', r'\1', t) for t in raw]
    # Keep tokens ≥4 chars, or numeric tokens ≥3 chars (e.g. "122")
    return {t for t in normalized if len(t) >= 4 or (len(t) >= 3 and t.isdigit())}


def title_similarity(left: str, right: str) -> float:
    left_tokens = significant_title_tokens(left)
    right_tokens = significant_title_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    return intersection / union if union else 0.0


def title_overlap_boost(left: str, right: str) -> float:
    overlap = {
        token
        for token in (significant_title_tokens(left) & significant_title_tokens(right))
        if token not in GENERIC_OVERLAP_TOKENS
    }
    if len(overlap) >= 3:
        return 0.84
    # Two overlapping tokens: only boost if they include a specific name/version token
    if len(overlap) == 2 and any(any(char.isdigit() for char in token) for token in overlap):
        return 0.78
    # Single token with version number (e.g., "gpt4", "llama3") — weak signal
    if len(overlap) == 1 and any(any(char.isdigit() for char in token) for token in overlap):
        return 0.40
    return 0.0


def _is_paper_like(item: HotspotItem) -> bool:
    return item.source_type in PAPER_LIKE_SOURCE_TYPES or bool(str(item.metadata.get("arxiv_id", "")).strip())


def _cluster_match_score(item: HotspotItem, cluster_seed: HotspotItem) -> float:
    if item.canonical_url and item.canonical_url == cluster_seed.canonical_url:
        return 1.0
    left_arxiv = str(item.metadata.get("arxiv_id", ""))
    right_arxiv = str(cluster_seed.metadata.get("arxiv_id", ""))
    if left_arxiv and left_arxiv == right_arxiv:
        return 1.0
    left_repo = canonicalize_url(str(item.metadata.get("github_url", "") or item.url or ""))
    right_repo = canonicalize_url(str(cluster_seed.metadata.get("github_url", "") or cluster_seed.url or ""))
    if left_repo and right_repo and left_repo == right_repo and "github.com" in left_repo:
        return 1.0

    # Link-following: if a roundup URL points to an official blog URL
    item_urls = {item.url, item.canonical_url}
    seed_urls = {cluster_seed.url, cluster_seed.canonical_url}
    if item_urls & seed_urls:
        return 1.0

    if _is_paper_like(item) and _is_paper_like(cluster_seed):
        return 0.0

    base_score = max(title_similarity(item.title, cluster_seed.title), title_overlap_boost(item.title, cluster_seed.title))

    # Entity matching boost: if both mention the same company/product
    # and are from different source types (e.g., official + roundup)
    # Requires either moderate title similarity OR an official source to avoid false merges
    if base_score >= 0.25 and item.source_type != cluster_seed.source_type:
        item_entities = _extract_entities(item.title)
        seed_entities = _extract_entities(cluster_seed.title)
        shared_entities = item_entities & seed_entities
        if shared_entities:
            has_official = item.source_role == "official_news" or cluster_seed.source_role == "official_news"
            # Need BOTH entity match AND meaningful title overlap (not just entity alone)
            non_generic_overlap = {
                token
                for token in (significant_title_tokens(item.title) & significant_title_tokens(cluster_seed.title))
                if token not in GENERIC_OVERLAP_TOKENS
            } - shared_entities  # Exclude the entity itself from overlap count
            if has_official and base_score >= 0.25:
                # Official + entity match: strong signal, lower bar
                base_score = max(base_score, 0.70)
            elif len(non_generic_overlap) >= 1 and base_score >= 0.30:
                # Entity match + at least 1 additional non-generic token overlap
                base_score = max(base_score, 0.70)
            # Entity match alone (no other overlap): NOT enough — too many false positives

    # Lower threshold for (official + roundup/community) pairs
    is_cross_type = (
        (item.source_role == "official_news" and cluster_seed.source_role != "official_news") or
        (cluster_seed.source_role == "official_news" and item.source_role != "official_news")
    )
    if is_cross_type and base_score >= 0.60:
        base_score = max(base_score, 0.68)

    return base_score


def _cluster_id(items: Iterable[HotspotItem]) -> str:
    digest = hashlib.sha1()
    for item in sorted(items, key=lambda entry: (entry.canonical_url, entry.title)):
        digest.update(item.canonical_url.encode("utf-8", errors="ignore"))
        digest.update(item.title.encode("utf-8", errors="ignore"))
    return digest.hexdigest()[:12]


def compute_deterministic_cluster_score(items: list[HotspotItem]) -> float:
    score = 0.0
    source_roles = Counter(item.source_role for item in items)
    for role, count in source_roles.items():
        score += SOURCE_ROLE_WEIGHTS.get(role, 2.5) * count

    score += len({item.source_id for item in items}) * 1.8
    score += len({item.source_type for item in items}) * 1.1

    for item in items:
        metadata = item.metadata
        if "daily_score" in metadata:
            score += float(metadata["daily_score"]) * 0.35
        if "upvotes" in metadata:
            score += math.log1p(max(int(metadata["upvotes"]), 0)) * 1.35
        if "activity" in metadata:
            score += math.log1p(max(int(metadata["activity"]), 0)) * 1.15
        if "stars" in metadata:
            score += math.log1p(max(int(metadata["stars"]), 0)) * 1.1
        if "github_stars" in metadata:
            score += math.log1p(max(int(metadata["github_stars"]), 0)) * 1.1
        if "hn_score" in metadata:
            score += math.log1p(max(int(metadata["hn_score"]), 0)) * 0.9
        if metadata.get("is_official"):
            score += 2.3
        if metadata.get("github_url"):
            score += 1.5
    return round(score, 3)


def build_hotspot_clusters(items: list[HotspotItem], similarity_threshold: float = 0.68) -> list[HotspotCluster]:
    if not items:
        return []

    sorted_items = sorted(
        items,
        key=lambda item: (
            parse_datetime(item.published_at) or parse_datetime("1970-01-01T00:00:00+00:00"),
            item.source_role,
        ),
        reverse=True,
    )
    grouped: list[list[HotspotItem]] = []

    for item in sorted_items:
        for bucket in grouped:
            seed = bucket[0]
            if _cluster_match_score(item, seed) >= similarity_threshold:
                bucket.append(item)
                break
        else:
            grouped.append([item])

    clusters: list[HotspotCluster] = []
    for bucket in grouped:
        bucket_sorted = sorted(
            bucket,
            key=lambda item: (
                SOURCE_ROLE_WEIGHTS.get(item.source_role, 0.0),
                parse_datetime(item.published_at) or parse_datetime("1970-01-01T00:00:00+00:00"),
            ),
            reverse=True,
        )
        seed = bucket_sorted[0]
        all_tags = sorted({tag for item in bucket_sorted for tag in item.tags})
        published_candidates = [
            item.published_at
            for item in bucket_sorted
            if item.published_at is not None
        ]
        clusters.append(
            HotspotCluster(
                cluster_id=_cluster_id(bucket_sorted),
                title=seed.title,
                canonical_url=canonicalize_url(seed.canonical_url or seed.url),
                summary=seed.summary,
                items=[item.to_dict() for item in bucket_sorted],
                source_ids=sorted({item.source_id for item in bucket_sorted}),
                source_names=sorted({item.source_name for item in bucket_sorted}),
                source_roles=sorted({item.source_role for item in bucket_sorted}),
                source_types=sorted({item.source_type for item in bucket_sorted}),
                tags=all_tags,
                published_at=max(published_candidates) if published_candidates else None,
                deterministic_score=compute_deterministic_cluster_score(bucket_sorted),
            )
        )

    # Merge pass: combine clusters with very similar titles that the greedy pass missed
    merged = True
    while merged:
        merged = False
        i = 0
        while i < len(clusters):
            j = i + 1
            while j < len(clusters):
                if title_similarity(clusters[i].title, clusters[j].title) >= 0.85:
                    # Merge smaller into larger
                    big, small = (i, j) if len(clusters[i].items) >= len(clusters[j].items) else (j, i)
                    big_c, small_c = clusters[big], clusters[small]
                    existing_urls = {item.get("canonical_url") or item.get("url") for item in big_c.items}
                    for item in small_c.items:
                        if (item.get("canonical_url") or item.get("url")) not in existing_urls:
                            big_c.items.append(item)
                    big_c.source_ids = sorted(set(big_c.source_ids) | set(small_c.source_ids))
                    big_c.source_names = sorted(set(big_c.source_names) | set(small_c.source_names))
                    big_c.source_roles = sorted(set(big_c.source_roles) | set(small_c.source_roles))
                    big_c.source_types = sorted(set(big_c.source_types) | set(small_c.source_types))
                    big_c.tags = sorted(set(big_c.tags) | set(small_c.tags))
                    clusters.pop(small)
                    merged = True
                    if big > small:
                        i = small  # re-check from the shifted position
                        break
                else:
                    j += 1
            else:
                i += 1

    return sorted(clusters, key=lambda cluster: cluster.deterministic_score, reverse=True)
