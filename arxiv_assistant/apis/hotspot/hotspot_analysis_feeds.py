"""Fetch long-form analysis content from trusted expert blogs via RSS/Atom feeds or Playwright."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import feedparser

from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem, clean_text
from arxiv_assistant.utils.hotspot.hotspot_sources import clip_text, fetch_text, is_fresh

# Matches titles like "package-name 0.2a1", "tool 1.3.0", "lib-v2.0"
_VERSION_ONLY_RE = re.compile(
    r"^[\w\-]+\s+v?\d+\.\d+[\.\w]*$"
    r"|^[\w\-]+\s+\d+\.\d+[\.\w]*\s*[\(\[]",
    re.IGNORECASE,
)

# Minimum summary length for analysis content (release notes tend to be short)
_MIN_ANALYSIS_SUMMARY_LEN = 80


def _load_feed_registry(registry_path: str | Path) -> list[dict]:
    path = Path(registry_path)
    if not path.exists():
        return []
    return [f for f in json.loads(path.read_text(encoding="utf-8")) if f.get("enabled", True)]


def _fetch_rss_feed(feed_config: dict, target_date: datetime, effective_freshness: int) -> list[HotspotItem]:
    """Fetch items from an RSS/Atom feed source."""
    feed_id = feed_config.get("feed_id", "unknown")
    feed_name = feed_config.get("name", feed_id)
    feed_url = feed_config.get("url", "")

    try:
        rss_text = fetch_text(feed_url)
        feed = feedparser.parse(rss_text)
    except Exception as ex:
        print(f"Warning: failed to fetch analysis feed {feed_id}: {ex}")
        return []

    items: list[HotspotItem] = []
    for entry in feed.entries:
        published_at = entry.get("published") or entry.get("updated")
        if not is_fresh(published_at, target_date, effective_freshness):
            continue

        title = clean_text(entry.get("title", ""))
        if not title or len(title) < 15:
            continue

        summary = clean_text(entry.get("summary", "") or entry.get("description", ""))

        # Skip version-only release notes (e.g., "datasette-enrichments-llm 0.2a1")
        if _VERSION_ONLY_RE.match(title):
            continue

        # Analysis content should have substantive summaries
        # Allow per-feed override for sources with short teasers (e.g., Chinese media)
        min_len = feed_config.get("min_summary_len", _MIN_ANALYSIS_SUMMARY_LEN)
        if len(summary) < min_len:
            continue

        url = clean_text(entry.get("link", feed_url))

        items.append(
            HotspotItem(
                source_id=f"analysis_{feed_id}",
                source_name=feed_name,
                source_role="editorial_depth",
                source_type="blog_analysis",
                title=title,
                summary=clip_text(summary, 800),
                url=url,
                canonical_url=url,
                published_at=published_at,
                tags=["analysis", "deep-read"],
                authors=[],
                metadata={
                    "feed_source": feed_url,
                    "feed_id": feed_id,
                    "source_quality": 1.3,
                    "signal_tier": "trusted_analysis",
                },
            )
        )
        if len(items) >= 5:
            break

    return items


def _fetch_playwright_llm_feed(feed_config: dict, target_date: datetime, freshness_hours: int) -> list[HotspotItem]:
    """Fetch items from JS-rendered sites using Playwright + LLM extraction."""
    feed_id = feed_config.get("feed_id", "unknown")
    feed_name = feed_config.get("name", feed_id)

    # Reuse the playwright/LLM infrastructure from official_blogs
    from arxiv_assistant.apis.hotspot.hotspot_official_blogs import (
        _extract_generic_blog_rows,
        _extract_with_llm,
        _render_with_playwright,
    )
    from bs4 import BeautifulSoup

    # Build a source dict compatible with official_blogs functions
    source = {
        "source_id": f"analysis_{feed_id}",
        "source_name": feed_name,
        "url": feed_config.get("url", ""),
        "require_date": feed_config.get("require_date", False),
        "llm_model": feed_config.get("llm_model", "gpt-5.4"),
    }

    try:
        page_html = _render_with_playwright(source["url"])
    except Exception as ex:
        print(f"Warning: playwright render failed for analysis feed {feed_id}: {ex}")
        return []

    # Try standard HTML extraction first
    rows = _extract_generic_blog_rows(page_html, source["url"])
    if len(rows) >= 2:
        items: list[HotspotItem] = []
        for row in rows:
            published_at = row.get("published_at")
            if published_at and not is_fresh(published_at, target_date, freshness_hours):
                continue
            if not published_at:
                if source.get("require_date", True):
                    continue
                # Assign target_date for dateless items from require_date=false sources
                published_at = target_date.isoformat()
            title = clean_text(row["title"])
            if not title or len(title) < 10:
                continue
            # Skip subscription/promo items
            if re.search(r"通讯会员|订阅|subscribe|newsletter|PRO会员", title, re.I):
                continue
            items.append(
                HotspotItem(
                    source_id=f"analysis_{feed_id}",
                    source_name=feed_name,
                    source_role="editorial_depth",
                    source_type="blog_analysis",
                    title=title,
                    summary=clip_text(row.get("summary", ""), 800),
                    url=row["url"],
                    canonical_url=row["url"],
                    published_at=published_at,
                    tags=["analysis", "deep-read"],
                    authors=[],
                    metadata={
                        "feed_source": source["url"],
                        "feed_id": feed_id,
                        "source_quality": 1.3,
                        "signal_tier": "trusted_analysis",
                    },
                )
            )
        if items:
            print(f"Analysis feed {feed_id}: extracted {len(items)} items via HTML")
            return items[:8]

    # Fallback: LLM extraction
    soup = BeautifulSoup(page_html, "html.parser")
    page_text = clean_text(soup.get_text(" ", strip=True))
    llm_items = _extract_with_llm(page_text, source)

    items = []
    for item in llm_items:
        title = clean_text(item.get("title", ""))
        if not title or len(title) < 10:
            continue
        summary = clean_text(item.get("summary", ""))
        url = item.get("url", source["url"])
        items.append(
            HotspotItem(
                source_id=f"analysis_{feed_id}",
                source_name=feed_name,
                source_role="editorial_depth",
                source_type="blog_analysis",
                title=title,
                summary=clip_text(summary, 800),
                url=url,
                canonical_url=url,
                published_at=target_date.isoformat(),
                tags=["analysis", "deep-read"],
                authors=[],
                metadata={
                    "feed_source": source["url"],
                    "feed_id": feed_id,
                    "source_quality": 1.3,
                    "signal_tier": "trusted_analysis",
                },
            )
        )

    if items:
        print(f"Analysis feed {feed_id}: extracted {len(items)} items via LLM")
    return items[:8]


def fetch_hotspot_items(
    target_date: datetime,
    freshness_hours: int,
    registry_path: str | Path,
) -> list[HotspotItem]:
    feeds = _load_feed_registry(registry_path)
    if not feeds:
        return []

    # Analysis content is less time-sensitive; use wider freshness window
    effective_freshness = max(freshness_hours, 48)

    items: list[HotspotItem] = []
    for feed_config in feeds:
        feed_id = feed_config.get("feed_id", "unknown")
        mode = feed_config.get("mode", "rss")

        try:
            if mode == "playwright_llm":
                items.extend(_fetch_playwright_llm_feed(feed_config, target_date, effective_freshness))
            else:
                items.extend(_fetch_rss_feed(feed_config, target_date, effective_freshness))
        except Exception as ex:
            print(f"Warning: failed to fetch analysis feed {feed_id}: {ex}")

    return items
