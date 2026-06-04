from __future__ import annotations

import math

from arxiv_assistant.hotspots.embed import EMBED_MODEL_ID, cosine, embed_text
from arxiv_assistant.hotspots.enrich import EnrichedItem
from arxiv_assistant.hotspots.story import Story, group_into_stories

# §C.1: cosine > L1_SEMANTIC_THRESHOLD auto-merges intra-day near-duplicates
# (incl. an English item and its `_zh` translation in the shared multilingual space).
# Default from configs/config.ini `cross_day_cosine_threshold = 0.72`.
# Empirically calibrated 2026-06: mpnet-base-v2 @ 0.72 cleanly separates same-event
# zh/en pairs (>=0.84) from related-but-different pairs (<=0.54).
L1_SEMANTIC_THRESHOLD = 0.72


def _embed_story_text(item) -> str:
    """title + lede (first sentence of summary) — the §C.1 embedding payload."""
    title = (item.title or "").strip()
    lede = (item.summary or "").strip().split(". ")[0]
    return f"{title}. {lede}".strip()


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return list(vec)
    return [x / norm for x in vec]


def _centroid(vectors: list[list[float]]) -> list[float]:
    """L2-normalized mean of unit vectors (deterministic, order-independent)."""
    if not vectors:
        return []
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            acc[i] += v[i]
    mean = [a / len(vectors) for a in acc]
    return _normalize(mean)


def cluster_intraday(
    items: list[EnrichedItem],
    *,
    threshold: float = L1_SEMANTIC_THRESHOLD,
) -> list[Story]:
    """L0 exact (canonical URL) + L1 semantic (cosine > threshold, cross-language).

    Reuses the existing deterministic 4-pass Union-Find grouper (`group_into_stories`)
    for L0/title evidence, then runs an L1 semantic merge pass on the resulting stories.
    Each returned Story carries a model-id-bound L2-normalized `centroid`.

    The `threshold` defaults to `L1_SEMANTIC_THRESHOLD` (0.72, from
    config key `cross_day_cosine_threshold`).  A caller that has loaded the
    config section should pass
        threshold=hotspot_config.getfloat("cross_day_cosine_threshold", fallback=0.72)
    """
    if not items:
        return []

    # L0 + existing deterministic title/URL/entity grouping.
    stories = group_into_stories(items)

    # Embed each story (unit vectors) for L1 and to set centroids.
    vecs: list[list[float]] = []
    for s in stories:
        member_vecs = [_normalize(embed_text(_embed_story_text(ei.item))) for ei in s.items]
        s.centroid = _centroid(member_vecs)
        s.centroid_model_id = EMBED_MODEL_ID
        vecs.append(s.centroid)

    # L1: greedy union of stories whose centroids are cosine > threshold.
    parent = list(range(len(stories)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(stories)):
        for j in range(i + 1, len(stories)):
            if find(i) == find(j):
                continue
            if cosine(vecs[i], vecs[j]) > threshold:
                parent[find(j)] = find(i)

    # Collapse merged groups, re-using group_into_stories to rebuild canonical/entity fields.
    groups: dict[int, list[EnrichedItem]] = {}
    for idx, s in enumerate(stories):
        groups.setdefault(find(idx), []).extend(s.items)

    merged: list[Story] = []
    for member_items in groups.values():
        rebuilt = group_into_stories(member_items)
        # group_into_stories may re-split on title evidence; re-merge to one story per L1 group
        # by pooling all items, since L1 cosine already declared them the same event.
        if len(rebuilt) == 1:
            story = rebuilt[0]
        else:
            story = rebuilt[0]
            for extra in rebuilt[1:]:
                story.items.extend(extra.items)
                story.entity_names |= extra.entity_names
        member_vecs = [_normalize(embed_text(_embed_story_text(ei.item))) for ei in story.items]
        story.centroid = _centroid(member_vecs)
        story.centroid_model_id = EMBED_MODEL_ID
        merged.append(story)

    return merged
