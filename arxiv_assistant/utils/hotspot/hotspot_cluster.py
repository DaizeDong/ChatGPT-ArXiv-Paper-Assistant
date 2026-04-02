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
    "what",
    "when",
    "your",
    "that",
    "this",
    "than",
    "over",
    "under",
    "about",
    "just",
    "more",
    "most",
    "some",
    "been",
    "each",
    "only",
    "very",
    "every",
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
    # Common non-technical words that appear across unrelated items
    "teams",
    "across",
    "action",
    "world",
    "work",
    "real",
    "help",
    "helping",
    "turn",
    "turning",
}

# Extended set used for paper-to-paper matching safety check.
# These tokens appear in many unrelated ML paper titles and should not
# be sufficient on their own to merge two papers.
PAPER_GENERIC_TOKENS = GENERIC_OVERLAP_TOKENS | {
    "multi",
    "large",
    "language",
    "vision",
    "learning",
    "training",
    "framework",
    "based",
    "using",
    "towards",
    "efficient",
    "novel",
    "data",
    "neural",
    "network",
    "approach",
    "method",
    "generation",
    "automated",
    "analysis",
    "scale",
    "task",
    "tasks",
    "long",
    "high",
    "performance",
    "improving",
    "deep",
    "agent",
    "agents",
    "token",
    "tokens",
    "level",
    "time",
    "self",
    "dynamic",
    "adaptive",
    "robust",
    "stable",
    "free",
    "unified",
    "representation",
    "attention",
    "parameter",
    "semantic",
    "geometry",
    "spectral",
    "sparse",
    "dense",
    "fine",
    "grained",
    "prediction",
    "detection",
    "optimization",
    "inference",
    "embedding",
    "diffusion",
    "alignment",
    "distillation",
    "compression",
    "reinforcement",
    "contrastive",
    "generative",
    "transformer",
    "pretraining",
    "transfer",
    "knowledge",
    "graph",
    "image",
    "text",
    "multimodal",
    "cross",
    "modal",
    "networks",
    "term",
    "operators",
    "dynamics",
    "frequency",
    "features",
    "memory",
    "policy",
    "reward",
    "loss",
    "layer",
    "layers",
    "head",
    "heads",
    "block",
    "blocks",
    "input",
    "output",
    "latent",
    "space",
    "selection",
    "retrieval",
    "visual",
    "spatial",
    "temporal",
    "sequence",
    "structured",
    "scalable",
    "parallel",
    "distributed",
    "contextual",
    "grounding",
    "planning",
    "simulation",
    "evaluation",
    "verification",
    "complexity",
    "driven",
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

PAPER_LIKE_SOURCE_TYPES = {"paper"}


def canonicalize_url(url: str) -> str:
    if "arxiv.org/abs/" in url:
        arxiv_id = url.rsplit("/", 1)[-1]
        return f"https://arxiv.org/abs/{arxiv_id}"
    if "arxiv.org/pdf/" in url:
        arxiv_id = url.rsplit("/", 1)[-1].replace(".pdf", "")
        return f"https://arxiv.org/abs/{arxiv_id}"
    if "huggingface.co/papers/" in url:
        paper_id = url.rsplit("/", 1)[-1]
        return f"https://arxiv.org/abs/{paper_id}"
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


def _is_paper_like(item: HotspotItem) -> bool:
    return item.source_type in PAPER_LIKE_SOURCE_TYPES or bool(str(item.metadata.get("arxiv_id", "")).strip())


def _canonicalize_github_url(url: str) -> str:
    """Normalize GitHub URLs to owner/repo form for comparison."""
    if not url or "github.com" not in url:
        return ""
    parts = urlsplit(url)
    path_segments = [s for s in parts.path.split("/") if s]
    if len(path_segments) >= 2:
        return f"github.com/{path_segments[0].lower()}/{path_segments[1].lower()}"
    return ""


def _extract_github_repo(item: HotspotItem) -> str:
    """Extract normalized GitHub repo identifier from an item."""
    for url in (str(item.metadata.get("github_url", "") or ""), item.url, item.canonical_url):
        repo = _canonicalize_github_url(url)
        if repo:
            return repo
    return ""


# Named entities that are likely model/product names (mixed case, versioned, or acronym-like)
_ENTITY_PATTERN = re.compile(
    r"\b("
    r"[A-Z][a-zA-Z]+(?:[-.](?:[A-Z][a-zA-Z]+|\d+(?:\.\d+)*))+|"  # CamelCase-Version
    r"[A-Z]{2,}(?:-\d+(?:\.\d+)*)?|"                                # ACRONYM or ACRONYM-1.5
    r"[A-Z][a-z]+[A-Z][a-zA-Z]*"                                    # CamelCase product name
    r")\b"
)


def _extract_named_entities(title: str) -> set[str]:
    """Extract likely product/model named entities from a title."""
    entities = set()
    for match in _ENTITY_PATTERN.finditer(title):
        entity = match.group(0).lower()
        if len(entity) >= 3 and entity not in GENERIC_OVERLAP_TOKENS and entity not in STOPWORDS:
            entities.add(entity)
    return entities


_GENERIC_ENTITIES = {
    "llm", "llms", "moe", "api", "sdk", "cli", "rag", "rlhf", "sft",
    "gpt", "vit", "mcp", "web", "app", "new", "pro", "max", "lite",
    # Common compound terms that appear across unrelated ML papers
    "multi-agent", "multi-modal", "multi-task", "multi-scale", "multi-view",
    "large-scale", "fine-grained", "long-term", "long-horizon", "real-time",
    "end-to-end", "self-supervised", "pre-trained", "open-source",
    "cross-modal", "cross-lingual", "high-frequency", "in-context",
    "zero-shot", "few-shot", "pre-training", "fine-tuning",
    "training-free", "label-free", "token-level", "price-driven",
    "contact-rich", "in-hand",
}


def _entity_match_score(left_title: str, right_title: str) -> float:
    """Score based on shared named entities between two titles."""
    left_entities = _extract_named_entities(left_title)
    right_entities = _extract_named_entities(right_title)
    if not left_entities or not right_entities:
        return 0.0
    shared = left_entities & right_entities - _GENERIC_ENTITIES
    if not shared:
        return 0.0
    # Strong match if a specific entity (5+ chars, not generic) is shared
    specific = {e for e in shared if len(e) >= 5}
    if specific:
        return 0.82
    # Moderate match for versioned entities (contains digit)
    versioned = {e for e in shared if any(c.isdigit() for c in e)}
    if versioned:
        return 0.78
    return 0.0


def _is_multi_topic_digest(title: str) -> bool:
    """Detect multi-topic digest/roundup titles that cover several unrelated stories."""
    # Titles with multiple comma-separated clauses or semicolons typically
    # pack several headlines into one (e.g. "OpenAI's Deal, Grok Cuts Prices, ...")
    separators = title.count(",") + title.count(";") + title.count("·")
    return separators >= 2


def _cluster_match_score(item: HotspotItem, cluster_seed: HotspotItem) -> float:
    # Exact URL match
    if item.canonical_url and item.canonical_url == cluster_seed.canonical_url:
        return 1.0
    # arXiv ID match
    left_arxiv = str(item.metadata.get("arxiv_id", ""))
    right_arxiv = str(cluster_seed.metadata.get("arxiv_id", ""))
    if left_arxiv and left_arxiv == right_arxiv:
        return 1.0
    # GitHub repo match (normalized)
    left_repo = _extract_github_repo(item)
    right_repo = _extract_github_repo(cluster_seed)
    if left_repo and right_repo and left_repo == right_repo:
        return 1.0

    # Multi-topic digest items should not merge via title matching since they
    # contain tokens from multiple unrelated stories
    if _is_multi_topic_digest(item.title) or _is_multi_topic_digest(cluster_seed.title):
        return 0.0

    # Safety check: when either side is a paper, require meaningful
    # non-generic token overlap. This gate applies unconditionally to
    # prevent false merges from entity matching on generic compound terms
    # like "Multi-Agent" or title overlap on common ML vocabulary.
    if _is_paper_like(item) or _is_paper_like(cluster_seed):
        left_specific = significant_title_tokens(item.title) - PAPER_GENERIC_TOKENS
        right_specific = significant_title_tokens(cluster_seed.title) - PAPER_GENERIC_TOKENS
        shared_specific = left_specific & right_specific
        if not shared_specific:
            return 0.0
        # Require either 2+ shared specific tokens, or 1 highly specific
        # token (10+ chars, likely a proper noun/method name). Shorter tokens
        # like "evolving", "scalable" appear across unrelated papers.
        highly_specific = {t for t in shared_specific if len(t) >= 10}
        if len(shared_specific) < 2 and not highly_specific:
            return 0.0

    # Cross-type matching: paper with repo about same project
    if left_arxiv and right_repo or right_arxiv and left_repo:
        entity_score = _entity_match_score(item.title, cluster_seed.title)
        if entity_score >= 0.70:
            return entity_score
    # Title-based matching (applies across all types including paper-to-paper)
    title_sim = title_similarity(item.title, cluster_seed.title)
    overlap_boost = title_overlap_boost(item.title, cluster_seed.title)
    entity_score = _entity_match_score(item.title, cluster_seed.title)

    return max(title_sim, overlap_boost, entity_score)


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


def build_hotspot_clusters(items: list[HotspotItem], similarity_threshold: float = 0.55) -> list[HotspotCluster]:
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

    # Second pass: try to merge singleton clusters into existing multi-item
    # clusters by checking against ALL members (not just seed). This catches
    # items like "Anthropic's secret model leaked" that match a non-seed
    # member but not the seed title. Only singletons are candidates to
    # prevent chain-merging between established clusters.
    singletons = [b for b in grouped if len(b) == 1]
    multi = [b for b in grouped if len(b) > 1]
    merged_singletons: set[int] = set()
    for idx, single in enumerate(singletons):
        item = single[0]
        for bucket in multi:
            for member in bucket:
                if _cluster_match_score(item, member) >= similarity_threshold:
                    bucket.append(item)
                    merged_singletons.add(idx)
                    break
            if idx in merged_singletons:
                break
    grouped = multi + [s for idx, s in enumerate(singletons) if idx not in merged_singletons]

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
