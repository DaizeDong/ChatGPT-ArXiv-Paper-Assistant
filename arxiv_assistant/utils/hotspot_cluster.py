from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Iterable
from urllib.parse import urlsplit, urlunsplit

from arxiv_assistant.utils.hotspot_schema import HotspotCluster, HotspotItem
from arxiv_assistant.utils.hotspot_sources import parse_datetime

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
}

SOURCE_ROLE_WEIGHTS = {
    "research_backbone": 5.4,
    "paper_trending": 4.8,
    "community_heat": 4.5,
    "official_news": 5.0,
    "headline_consensus": 4.0,
    "builder_momentum": 3.8,
    "editorial_depth": 3.4,
    "github_trend": 4.2,
    "hn_discussion": 3.0,
}


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
    return {
        token
        for token in normalize_title(title).split()
        if len(token) >= 4
    }


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
    if len(overlap) >= 2:
        return 0.84
    if overlap and any(any(char.isdigit() for char in token) for token in overlap):
        return 0.78
    return 0.0


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
    return max(title_similarity(item.title, cluster_seed.title), title_overlap_boost(item.title, cluster_seed.title))


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

    return sorted(clusters, key=lambda cluster: cluster.deterministic_score, reverse=True)
