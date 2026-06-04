from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from difflib import SequenceMatcher

from arxiv_assistant.hotspots.enrich import EnrichedItem, EVENT_TYPE_TO_CATEGORY
from arxiv_assistant.utils.hotspot.gate_date import gate_date
from arxiv_assistant.utils.hotspot.hotspot_cluster import (
    GENERIC_OVERLAP_TOKENS,
    SOURCE_ROLE_WEIGHTS,
    canonicalize_url,
    significant_title_tokens,
)

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

# Key AI figures whose opinions carry higher weight
KEY_FIGURES = {
    "sam altman", "greg brockman", "mira murati",
    "dario amodei", "daniela amodei",
    "demis hassabis", "jeff dean", "sundar pichai",
    "mark zuckerberg", "yann lecun",
    "geoffrey hinton", "yoshua bengio", "andrew ng", "fei-fei li",
    "jensen huang",
    "satya nadella",
    "arthur mensch",  # Mistral
    "ilya sutskever",
    "noam brown",  # OpenAI reasoning
    "jan leike",
}


def _freshness_weight(gate_day: date | None, *, run_date: date) -> float:
    """HN-style gravity from the day-granular gate_date (spec §B.5.1/§C.3).

    T = hours since gate_day (day boundary); decay = 1/(T+2)^1.8 normalized so a
    same-day story scores 1.0. Sub-day jitter cannot affect this because the input
    is already floored to a UTC day (INV2). Unknown date → neutral 0.6.
    """
    if gate_day is None:
        return 0.6
    age_days = (run_date - gate_day).days
    if age_days < 0:
        age_days = 0
    t_hours = age_days * 24.0
    # Normalized HN gravity: weight(T)/weight(0), weight(T) = 1/(T+2)^1.8.
    return (2.0 ** 1.8) / ((t_hours + 2.0) ** 1.8)


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
    # --- stage 0: persistent identity / centroid / status (contract §2.2) ---
    first_seen: str | None = None              # ISO date, immutable once set
    centroid: list[float] | None = None        # embedding; model_id-bound
    centroid_model_id: str = ""
    status: str = "NEW"                         # "NEW" | "ONGOING"
    arxiv_versions: dict[str, int] = field(default_factory=dict)   # id -> version count (monotonic)
    # --- surface snapshots (recorded by StoryStore.record_surface) ---
    last_surfaced: str | None = None
    surfaced_verified_max: str | None = None   # DAY-granular gate_date
    surfaced_entity_names: set[str] = field(default_factory=set)
    surfaced_max_tier: int = 0
    surfaced_arxiv_versions: dict[str, int] = field(default_factory=dict)
    # --- resurgence (§C.4) ---
    resurged_at: str | None = None             # first-ever resurge run-date (immutable)
    surfaced_resurged_at: str | None = None    # last resurgence-lane surface run-date
    # --- evidence ledger (populated by StoryStore.active_stories; used by NoveltyGate §C.3.1) ---
    evidence_ledger: list[dict] = field(default_factory=list)
    # each row dict has keys: canonical_url, source_id, source_role, provenance,
    # source_tier(int), added_at(str ISO date)

    def __post_init__(self):
        if not self.category:
            self.category = EVENT_TYPE_TO_CATEGORY.get(self.event_type, "Industry Update")
        if not self.headline:
            self.headline = self.canonical_item.item.title
        if not self.summary:
            self.summary = self.canonical_item.summary

    def evidence_added_since(self, snapshot_date: str | None) -> list[dict]:
        """Return ledger rows whose added_at > snapshot_date.

        If snapshot_date is None, return all rows.
        ISO date strings are compared lexicographically (zero-padded ISO == chronological).
        """
        if snapshot_date is None:
            return list(self.evidence_ledger)
        return [row for row in self.evidence_ledger if row["added_at"] > snapshot_date]

    def evidence_before(self, snapshot_date: str | None) -> list[dict]:
        """Return ledger rows whose added_at <= snapshot_date.

        If snapshot_date is None, return [].
        ISO date strings are compared lexicographically (zero-padded ISO == chronological).
        """
        if snapshot_date is None:
            return []
        return [row for row in self.evidence_ledger if row["added_at"] <= snapshot_date]


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

    # Pass 4: Title containment, Jaccard, and sequence similarity
    # Pre-compute tokens and lowered titles
    tokens_cache = [significant_title_tokens(ei.item.title) for ei in enriched_items]
    titles_lower = [ei.item.title.lower().strip() for ei in enriched_items]

    for a in range(n):
        for b in range(a + 1, n):
            if uf.find(a) == uf.find(b):
                continue
            # Skip paper-paper pairs (only merge via URL/arxiv_id in Pass 2)
            if enriched_items[a].event_type == "research_paper" and enriched_items[b].event_type == "research_paper":
                continue

            a_tokens = tokens_cache[a]
            b_tokens = tokens_cache[b]

            # Title containment: smaller ⊂ larger ≥ 80%
            if a_tokens and b_tokens:
                smaller, larger = (a_tokens, b_tokens) if len(a_tokens) <= len(b_tokens) else (b_tokens, a_tokens)
                if len(smaller) >= 2 and len(smaller & larger) / len(smaller) >= 0.80:
                    uf.union(a, b)
                    continue

                # Title Jaccard similarity ≥ 0.55
                intersection = len(a_tokens & b_tokens)
                union_size = len(a_tokens | b_tokens)
                if union_size > 0 and intersection / union_size >= 0.55:
                    uf.union(a, b)
                    continue

            # SequenceMatcher: catches synonym substitutions that Jaccard misses
            # e.g., "OpenAI acquires TBPN" vs "OpenAI buys tech podcast TBPN"
            if SequenceMatcher(None, titles_lower[a], titles_lower[b]).ratio() >= 0.65:
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
    """Score stories using 5-factor formula with dynamic normalization."""
    if not stories:
        return stories

    # First pass: compute raw scores
    raw_scores: list[float] = []
    for story in stories:
        source_weight_sum = min(
            25.0,
            sum(SOURCE_ROLE_WEIGHTS.get(ei.item.source_role, 2.5) for ei in story.items),
        )
        unique_source_ids = len({ei.item.source_id for ei in story.items})
        unique_source_types = len({ei.item.source_type for ei in story.items})
        evidence_breadth = unique_source_ids * 1.5 + unique_source_types * 0.8

        avg_importance = sum(ei.importance for ei in story.items) / len(story.items)

        # Opinion differentiation: boost weight when key figures are involved
        event_weight = EVENT_TYPE_WEIGHTS.get(story.event_type, 0.8)
        if story.event_type == "opinion" and story.entity_names & KEY_FIGURES:
            event_weight = 1.2

        # Gravity from the day-granular gate_date (verified first date), not now().
        run_day = datetime.now(UTC).date()
        gate_days = [gate_date(ei.item) for ei in story.items]
        gate_days = [d for d in gate_days if d is not None]
        # min = earliest credible date = most pessimistic (anti-manipulation); a story
        # cannot look fresh just because one late-dated item joined it.
        earliest_gate = min(gate_days) if gate_days else None
        freshness = _freshness_weight(earliest_gate, run_date=run_day)

        raw = (source_weight_sum + evidence_breadth + avg_importance) * event_weight * freshness
        raw_scores.append(raw)

    # Dynamic normalization: linear mapping with full-range discrimination.
    # P50 → 5.0, P95 → 9.5, max → 10.0. This ensures differentiation at every level.
    sorted_raw = sorted(raw_scores)
    n_raw = len(sorted_raw)
    p50 = sorted_raw[max(0, int(n_raw * 0.5) - 1)]
    p95 = sorted_raw[max(0, int(n_raw * 0.95) - 1)]

    for story, raw in zip(stories, raw_scores):
        if p95 <= p50 or n_raw < 3:
            # Degenerate case: all scores similar or too few stories
            story.score = round(min(10.0, max(1.0, raw / max(p50 / 5.0, 0.5))), 3)
        elif raw <= p50:
            # Bottom half: map [0, p50] → [1.0, 5.0]
            story.score = round(max(1.0, 1.0 + 4.0 * raw / p50), 3)
        elif raw <= p95:
            # Upper half: map [p50, p95] → [5.0, 9.5]
            story.score = round(5.0 + 4.5 * (raw - p50) / (p95 - p50), 3)
        else:
            # Top 5%: map [p95, max] → [9.5, 10.0]
            raw_max = sorted_raw[-1]
            if raw_max > p95:
                story.score = round(9.5 + 0.5 * (raw - p95) / (raw_max - p95), 3)
            else:
                story.score = 10.0

    stories.sort(key=lambda s: s.score, reverse=True)
    return stories


def apply_cross_day_penalty(
    stories: list[Story],
    recent_headlines: list[str],
    penalty_factor: float = 0.3,
) -> list[Story]:
    """Penalize stories whose headlines are semantically similar to recent days."""
    if not recent_headlines or not stories:
        return stories
    from arxiv_assistant.utils.hotspot.hotspot_cluster import title_similarity
    for story in stories:
        max_sim = max(
            (title_similarity(story.headline, rh) for rh in recent_headlines),
            default=0.0,
        )
        if max_sim >= 0.5:
            story.score = round(story.score * (1 - penalty_factor * max_sim), 3)
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
