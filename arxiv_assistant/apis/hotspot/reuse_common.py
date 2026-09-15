from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

import feedparser

from arxiv_assistant.apis.hotspot.hotspot_common import parse_iso_or_rss_datetime
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem, clean_text
from arxiv_assistant.utils.hotspot.hotspot_sources import clip_text, fetch_text, is_fresh

# source_tiers.json lives at repo configs/hotspot/source_tiers.json
_TIERS_PATH = Path(__file__).resolve().parents[3] / "configs" / "hotspot" / "source_tiers.json"

# Reuse-source name -> a source_id that already has a tier mapping in source_tiers.json.
# These inherit Altmetric-style weights directly (spec §D.1: reuse weights, no self-rolled scale).
REUSE_SOURCE_TIER_ANCHOR: dict[str, str] = {
    "hf_daily": "hf_papers",          # trusted_research
    "ainews": "ainews",               # community_signal
    "agents_radar": "github_trend",   # builder_ecosystem
    "horizon": "the_batch",           # trusted_analysis
    "scholar_inbox": "local_papers",  # trusted_research
    "openalex": "local_papers",       # trusted_research (spec §D.2 first-class reuse source)
}


def _load_tier_map() -> dict[str, str]:
    payload = json.loads(_TIERS_PATH.read_text(encoding="utf-8"))
    return payload.get("source_id_to_tier", {})


def reuse_source_role(reuse_name: str) -> str:
    """Map a reuse source to its source_tiers tier name (used as source_role).

    Falls back to 'community_signal' if the anchor is unknown, so a new reuse
    source never crashes harvest — it just gets a medium weight until tiered.
    """
    anchor = REUSE_SOURCE_TIER_ANCHOR.get(reuse_name, "")
    return _load_tier_map().get(anchor, "community_signal")


def build_reuse_item(
    reuse_name: str,
    *,
    title: str,
    url: str,
    summary: str,
    published_at: str | None,
    canonical_url: str | None = None,
    tags: list[str] | None = None,
    authors: list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> HotspotItem:
    """Construct a HotspotItem stamped with reuse provenance + tiered role."""
    metadata: dict[str, Any] = {"reuse_name": reuse_name, "host": urlsplit(url).netloc.lower()}
    if extra_metadata:
        metadata.update(extra_metadata)
    return HotspotItem(
        source_id=f"reuse_{reuse_name}",
        source_name=f"Reuse:{reuse_name}",
        source_role=reuse_source_role(reuse_name),
        source_type="reuse",
        title=title,
        summary=clip_text(summary, 520),
        url=url,
        canonical_url=canonical_url or url,
        published_at=published_at,
        tags=list(tags or []),
        authors=list(authors or []),
        metadata=metadata,
        provenance=f"reuse:{reuse_name}",
    )


def harvest_rss_reuse(
    reuse_name: str,
    feed_url: str,
    target_date: datetime,
    freshness_hours: int,
    *,
    result_limit: int = 24,
    summary_of: Callable[[Any], str] | None = None,
) -> list[HotspotItem]:
    """Generic RSS reuse harvester: fetch feed, freshness-filter, map to reuse items.

    Adapters that consume a plain RSS/Atom feed of finished competitor output
    reuse this verbatim; only feed_url + reuse_name differ per site.
    """
    try:
        rss_text = fetch_text(feed_url)
    except Exception as ex:  # degrade, never crash harvest (spec §E)
        print(f"Warning: reuse:{reuse_name} feed fetch failed ({feed_url}): {ex}")
        return []
    feed = feedparser.parse(rss_text)
    if feed.bozo and not feed.entries:
        print(f"Warning: reuse:{reuse_name} feed parse error: {feed.bozo_exception}")
        return []
    items: list[HotspotItem] = []
    seen: set[str] = set()
    for entry in feed.entries:
        published_at = entry.get("published") or entry.get("updated")
        if not is_fresh(published_at, target_date, freshness_hours):
            continue
        title = clean_text(entry.get("title", ""))
        url = clean_text(entry.get("link", ""))
        if not title or not url or url in seen:
            continue
        seen.add(url)
        summary = summary_of(entry) if summary_of else clean_text(
            entry.get("summary", "") or entry.get("description", "")
        )
        published_iso = parse_iso_or_rss_datetime(published_at)
        items.append(
            build_reuse_item(
                reuse_name,
                title=title,
                url=url,
                summary=summary,
                published_at=published_iso,
                tags=[clean_text(t.get("term", "")) for t in entry.get("tags", []) if clean_text(t.get("term", ""))],
            )
        )
        if len(items) >= result_limit:
            break
    return items
