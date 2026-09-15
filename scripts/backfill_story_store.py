"""One-off offline backfill: seed the Story Store from ~30 days of history.

CRITICAL SAFETY RAIL (overview §5): the historical reports ALREADY contain the
6-day-duplicate bug. A naive backfill that mints one Story per daily occurrence
would create 6 polluted `first_seen` anchors for one real event. So we run the
SAME Stage-2 dedup over the whole history FIRST, collapse cross-day duplicates by
centroid, and seed exactly ONE `first_seen` (the earliest occurrence date) per
real story.

Usage:
    python -m scripts.backfill_story_store --history-root out/hot --db out/hot/state/story_store.sqlite
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from arxiv_assistant.hotspots.dedup import (
    EMBED_MODEL_ID,
    L1_SEMANTIC_THRESHOLD,
    _centroid,
    cluster_intraday,
)
from arxiv_assistant.hotspots.embed import cosine


def dedup_history(daily_enriched_by_date: dict[str, list]) -> list[dict]:
    """Collapse a {iso_date: [EnrichedItem, ...]} history into real stories.

    Returns one record per real story:
        {"first_seen", "centroid", "centroid_model_id", "n_days"}
    `first_seen` is the EARLIEST date the story appears — the immutable anchor.

    Reuses production cluster_intraday (L0+L1) for intra-day dedup, then applies
    the same centroid-primary cross-day matching rule as match_crossday (L2) so
    that the backfill grouping is byte-identical to live grouping.
    """
    # Persistent accumulators: each entry is a "real story" with centroid + earliest date.
    persistent: list[dict] = []

    for iso_day in sorted(daily_enriched_by_date.keys()):
        intraday = cluster_intraday(daily_enriched_by_date[iso_day])
        for s in intraday:
            if not s.centroid:
                continue
            # Match against existing persistent stories (centroid-primary, same rule as L2).
            best = None
            best_sim = L1_SEMANTIC_THRESHOLD
            for p in persistent:
                sim = cosine(s.centroid, p["centroid"])
                if sim >= best_sim:
                    best_sim = sim
                    best = p
            if best is None:
                persistent.append({
                    "first_seen": iso_day,
                    "centroid": list(s.centroid),
                    "centroid_model_id": EMBED_MODEL_ID,
                    "n_days": 1,
                    "_member_vecs": [list(s.centroid)],
                })
            else:
                # Earliest date wins; refine centroid as running mean of occurrences.
                best["n_days"] += 1
                best["_member_vecs"].append(list(s.centroid))
                best["centroid"] = _centroid(best["_member_vecs"])
                if iso_day < best["first_seen"]:
                    best["first_seen"] = iso_day

    for p in persistent:
        p.pop("_member_vecs", None)
    return persistent


def _load_history(history_root: Path) -> dict[str, list]:  # pragma: no cover - IO glue
    """Load enriched items per day from existing out/hot reports.

    Wire to the Stage-0 report reader / enrich loader. Left as IO glue: the pure,
    tested core is `dedup_history`. Each value is a list[EnrichedItem] for that day.
    """
    raise NotImplementedError(
        "Wire to the report/enrich loader during Stage-0 integration; "
        "dedup_history is the tested core."
    )


def _seed_id_for(centroid: list[float], centroid_model_id: str) -> str:
    """Stable synthetic story_id for backfill seeds (content-addressed)."""
    digest = hashlib.sha1()
    digest.update(centroid_model_id.encode())
    for x in centroid[:16]:  # first 16 dims enough for uniqueness
        digest.update(str(round(x, 6)).encode())
    return "bfill-" + digest.hexdigest()[:12]


def main(argv=None) -> int:  # pragma: no cover - CLI glue
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-root", type=Path, default=Path("out/hot"))
    parser.add_argument("--db", type=Path, default=Path("out/hot/state/story_store.sqlite"))
    args = parser.parse_args(argv)

    history = _load_history(args.history_root)
    seeds = dedup_history(history)

    from arxiv_assistant.hotspots.store import StoryStore  # Stage-0
    from arxiv_assistant.hotspots.story import Story

    store = StoryStore(args.db)
    for seed in seeds:
        # Build a minimal Story shell so seed_first_seen can persist it.
        # headline=" " is the non-empty sentinel used by StoryStore._row_to_story
        # to avoid dereferencing canonical_item=None (same pattern as store internals).
        story = Story(
            story_id=_seed_id_for(seed["centroid"], seed["centroid_model_id"]),
            canonical_item=None,  # type: ignore[arg-type]
            items=[],
            event_type="other",
            headline=" ",
            summary=" ",
            centroid=seed["centroid"],
            centroid_model_id=seed["centroid_model_id"],
        )
        # Seed via write-once method — idempotent; earliest first_seen wins.
        store.seed_first_seen(story, seed["first_seen"])

    print(f"Backfill seeded {len(seeds)} real stories (dedup-first; no polluted anchors).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
