from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from arxiv_assistant.hotspots.enrich import EnrichedItem, EVENT_TYPE_TO_CATEGORY
from arxiv_assistant.utils.hotspot.hotspot_cluster import (
    GENERIC_OVERLAP_TOKENS,
    SOURCE_ROLE_WEIGHTS,
    canonicalize_url,
    significant_title_tokens,
    title_similarity,
)
from arxiv_assistant.utils.hotspot.hotspot_sources import parse_datetime

EVENT_TYPE_WEIGHTS = {
    "product_release": 2.0,
    "funding": 1.8,
    "acquisition": 1.8,
    "research_paper": 1.5,
    "tooling": 1.3,
    "industry_move": 1.2,
    "tutorial": 0.5,
    "opinion": 0.3,
    "recap": 0.2,
    "other": 0.8,
}


def _freshness_weight(published_at: str | None) -> float:
    if not published_at:
        return 0.6
    dt = parse_datetime(published_at)
    if dt is None:
        return 0.6
    hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
    if hours < 12:
        return 1.0
    if hours < 24:
        return 0.8
    if hours < 36:
        return 0.6
    return 0.4


@dataclass
class Story:
    story_id: str
    canonical_item: EnrichedItem
    items: list[EnrichedItem]
    event_type: str
    entity_names: set[str] = field(default_factory=set)
    category: str = ""
    score: float = 0.0
    headline: str = ""
    summary: str = ""
    why_it_matters: str = ""
    key_takeaways: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.category:
            self.category = EVENT_TYPE_TO_CATEGORY.get(self.event_type, "Industry Update")
        if not self.headline:
            self.headline = self.canonical_item.item.title
        if not self.summary:
            self.summary = self.canonical_item.summary


def _story_id(items: list[EnrichedItem]) -> str:
    digest = hashlib.sha1()
    for ei in sorted(items, key=lambda e: (e.item.canonical_url, e.item.title)):
        digest.update(ei.item.canonical_url.encode("utf-8", errors="ignore"))
        digest.update(ei.item.title.encode("utf-8", errors="ignore"))
    return digest.hexdigest()[:12]


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def group_into_stories(enriched_items: list[EnrichedItem]) -> list[Story]:
    """Group enriched items into Stories using Union-Find with 4 merge passes."""
    if not enriched_items:
        return []

    n = len(enriched_items)
    uf = _UnionFind(n)

    # Pass 1: LLM cross-references (same_event_as)
    for i, ei in enumerate(enriched_items):
        if ei.same_event_as is not None and 0 <= ei.same_event_as < n:
            uf.union(i, ei.same_event_as)

    # Pass 2: Shared URL / arxiv_id / github_url
    url_index: dict[str, int] = {}
    for i, ei in enumerate(enriched_items):
        item = ei.item
        for raw_url in [
            canonicalize_url(item.canonical_url or item.url),
            str(item.metadata.get("arxiv_id", "")).strip(),
        ]:
            if raw_url:
                if raw_url in url_index:
                    uf.union(i, url_index[raw_url])
                else:
                    url_index[raw_url] = i
        github_url = canonicalize_url(str(item.metadata.get("github_url", "") or ""))
        if github_url and "github.com" in github_url:
            if github_url in url_index:
                uf.union(i, url_index[github_url])
            else:
                url_index[github_url] = i

    # Pass 3: Shared entity + non-generic title overlap (entity-based, event_type agnostic)
    # Skip paper-paper merges (those only merge via URL/arxiv_id in Pass 2)
    entity_index: dict[str, list[int]] = defaultdict(list)
    for i, ei in enumerate(enriched_items):
        for entity in ei.entities:
            entity_index[entity["name"].lower()].append(i)
    for entity_name, indices in entity_index.items():
        if len(indices) <= 1:
            continue
        for i_idx in range(len(indices)):
            for j_idx in range(i_idx + 1, len(indices)):
                a, b = indices[i_idx], indices[j_idx]
                if uf.find(a) == uf.find(b):
                    continue
                # Don't merge two research papers via entity alone
                if enriched_items[a].event_type == "research_paper" and enriched_items[b].event_type == "research_paper":
                    continue
                a_tokens = significant_title_tokens(enriched_items[a].item.title) - GENERIC_OVERLAP_TOKENS
                b_tokens = significant_title_tokens(enriched_items[b].item.title) - GENERIC_OVERLAP_TOKENS
                # Subtract shared entity names so that entity names alone don't
                # count as title overlap evidence
                a_ent_names = {e["name"].lower() for e in enriched_items[a].entities}
                b_ent_names = {e["name"].lower() for e in enriched_items[b].entities}
                shared_entity_names = a_ent_names & b_ent_names
                overlap = (a_tokens & b_tokens) - shared_entity_names
                if overlap:
                    uf.union(a, b)

    # Pass 4: Title containment and similarity
    # Pre-compute tokens
    tokens_cache = [significant_title_tokens(ei.item.title) for ei in enriched_items]

    for a in range(n):
        for b in range(a + 1, n):
            if uf.find(a) == uf.find(b):
                continue
            # Skip paper-paper pairs (only merge via URL/arxiv_id in Pass 2)
            if enriched_items[a].event_type == "research_paper" and enriched_items[b].event_type == "research_paper":
                continue

            a_tokens = tokens_cache[a]
            b_tokens = tokens_cache[b]
            if not a_tokens or not b_tokens:
                continue

            # Title containment: smaller ⊂ larger ≥ 80%
            smaller, larger = (a_tokens, b_tokens) if len(a_tokens) <= len(b_tokens) else (b_tokens, a_tokens)
            if len(smaller) >= 2 and len(smaller & larger) / len(smaller) >= 0.80:
                uf.union(a, b)
                continue

            # Title Jaccard similarity ≥ 0.55
            intersection = len(a_tokens & b_tokens)
            union_size = len(a_tokens | b_tokens)
            if union_size > 0 and intersection / union_size >= 0.55:
                uf.union(a, b)

    # Build groups
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)

    stories: list[Story] = []
    for indices in groups.values():
        group_items = [enriched_items[i] for i in indices]

        canonical = max(
            group_items,
            key=lambda ei: (
                SOURCE_ROLE_WEIGHTS.get(ei.item.source_role, 2.5),
                ei.importance,
                len(ei.item.summary),
            ),
        )

        type_counts: dict[str, int] = defaultdict(int)
        for ei in group_items:
            type_counts[ei.event_type] += 1
        event_type = max(type_counts, key=lambda t: (type_counts[t], EVENT_TYPE_WEIGHTS.get(t, 0.8)))

        entity_names: set[str] = set()
        for ei in group_items:
            for e in ei.entities:
                entity_names.add(e["name"].lower())

        stories.append(
            Story(
                story_id=_story_id(group_items),
                canonical_item=canonical,
                items=group_items,
                event_type=event_type,
                entity_names=entity_names,
            )
        )

    return stories


def score_stories(stories: list[Story]) -> list[Story]:
    """Score stories using 5-factor formula, sort descending."""
    for story in stories:
        source_weight_sum = min(
            25.0,
            sum(SOURCE_ROLE_WEIGHTS.get(ei.item.source_role, 2.5) for ei in story.items),
        )
        unique_source_ids = len({ei.item.source_id for ei in story.items})
        unique_source_types = len({ei.item.source_type for ei in story.items})
        evidence_breadth = unique_source_ids * 1.5 + unique_source_types * 0.8

        avg_importance = sum(ei.importance for ei in story.items) / len(story.items)
        event_weight = EVENT_TYPE_WEIGHTS.get(story.event_type, 0.8)

        published_dates = [ei.item.published_at for ei in story.items if ei.item.published_at]
        freshness = _freshness_weight(max(published_dates) if published_dates else None)

        raw = (source_weight_sum + evidence_breadth + avg_importance) * event_weight * freshness
        story.score = round(min(10.0, raw / 5.5), 3)

    stories.sort(key=lambda s: s.score, reverse=True)
    return stories


def select_and_categorize(
    stories: list[Story],
    *,
    target_featured: int = 5,
    target_watchlist: int = 3,
    max_per_category: int = 3,
) -> tuple[list[Story], list[Story], dict[str, list[Story]]]:
    """Select featured stories with diversity constraints, build categories."""
    featured: list[Story] = []
    category_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}

    for story in stories:
        if len(featured) >= target_featured:
            break
        cat = story.category
        sig = "|".join(sorted({ei.item.source_name for ei in story.items})[:2])

        if category_counts.get(cat, 0) >= max_per_category:
            continue
        if source_counts.get(sig, 0) >= 2:
            continue

        featured.append(story)
        category_counts[cat] = category_counts.get(cat, 0) + 1
        source_counts[sig] = source_counts.get(sig, 0) + 1

    # Fill remaining spots relaxing constraints
    if len(featured) < target_featured:
        featured_ids = {s.story_id for s in featured}
        for story in stories:
            if story.story_id in featured_ids:
                continue
            if len(featured) >= target_featured:
                break
            featured.append(story)

    featured_ids = {s.story_id for s in featured}

    watchlist: list[Story] = [
        s for s in stories if s.story_id not in featured_ids
    ][:target_watchlist]

    claimed_ids = featured_ids | {s.story_id for s in watchlist}
    categories: dict[str, list[Story]] = defaultdict(list)
    for story in stories:
        if story.story_id not in claimed_ids:
            categories[story.category].append(story)

    return featured, watchlist, dict(categories)
