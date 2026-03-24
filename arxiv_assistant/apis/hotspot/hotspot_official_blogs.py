from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

import feedparser
from bs4 import BeautifulSoup

from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem, clean_text
from arxiv_assistant.utils.hotspot.hotspot_sources import clip_text, fetch_text, is_fresh, normalize_url, parse_datetime

OFFICIAL_SOURCES = [
    {
        "source_id": "openai_news",
        "source_name": "OpenAI News",
        "url": "https://openai.com/news/rss.xml",
        "mode": "rss",
    },
    {
        "source_id": "anthropic_news",
        "source_name": "Anthropic News",
        "url": "https://www.anthropic.com/news",
        "mode": "anthropic_html",
    },
    {
        "source_id": "google_ai_blog",
        "source_name": "Google AI Blog",
        "url": "https://blog.google/innovation-and-ai/technology/ai/rss/",
        "mode": "rss",
    },
    {
        "source_id": "meta_ai_blog",
        "source_name": "Meta AI Blog",
        "url": "https://ai.meta.com/blog/",
        "mode": "meta_html",
    },
]
GENERIC_TITLES = {"news", "newsroom", "ai", "blog", "home", "featured", "learn more"}
DATE_PATTERN = re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}\b", re.I)
ANTHROPIC_PREFIX_PATTERN = re.compile(
    r"^(?:(?:Announcements|Product|Policy|Research|Safety)\s+)?"
    r"(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}\s+)?"
    r"(?:(?:Announcements|Product|Policy|Research|Safety)\s+)?",
    re.I,
)


def _normalize_official_title(title: str) -> str:
    normalized = clean_text(ANTHROPIC_PREFIX_PATTERN.sub("", title or ""))
    normalized = clean_text(DATE_PATTERN.sub("", normalized))
    for marker in (" We’re ", " We're ", " Learn how ", " Read more "):
        if marker in normalized:
            normalized = normalized.split(marker, 1)[0].strip()
    if len(normalized) > 140 and ". " in normalized:
        normalized = normalized.split(". ", 1)[0].strip()
    return clean_text(normalized)


def _looks_generic_title(title: str, url: str, base_url: str) -> bool:
    normalized = clean_text(title).lower()
    if normalized in GENERIC_TITLES:
        return True
    return normalize_url(url) == normalize_url(base_url)


def _extract_date_from_text(text: str) -> str | None:
    match = DATE_PATTERN.search(text)
    if match is None:
        return None
    parsed = parse_datetime(match.group(0))
    return parsed.astimezone(UTC).isoformat() if parsed else None


def _build_item(source: dict[str, str], title: str, summary: str, url: str, published_at: str | None) -> HotspotItem:
    return HotspotItem(
        source_id=source["source_id"],
        source_name=source["source_name"],
        source_role="official_news",
        source_type="official_blog",
        title=title,
        summary=clip_text(summary, 650),
        url=url,
        canonical_url=url,
        published_at=published_at,
        tags=[],
        authors=[],
        metadata={"is_official": True, "publisher": source["source_name"]},
    )


def _fetch_rss_source(source: dict[str, str], target_date: datetime, freshness_hours: int) -> list[HotspotItem]:
    feed = feedparser.parse(fetch_text(source["url"]))
    items: list[HotspotItem] = []
    for entry in feed.entries:
        published_at = entry.get("published") or entry.get("updated")
        if not is_fresh(published_at, target_date, freshness_hours):
            continue
        items.append(
            _build_item(
                source,
                title=entry.get("title", source["source_name"]),
                summary=entry.get("summary", "") or entry.get("description", ""),
                url=entry.get("link", source["url"]),
                published_at=published_at,
            )
        )
    return items[:8]


def _extract_anthropic_rows(page_html: str, base_url: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(page_html, "html.parser")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        href = normalize_url(anchor["href"], base_url)
        if "/news/" not in href or href in seen:
            continue
        title_node = anchor.find(["h1", "h2", "h3", "h4"])
        summary_node = anchor.find("p")
        time_node = anchor.find("time")

        published_at = None
        if time_node:
            published_at = time_node.get("datetime") or clean_text(time_node.get_text(" ", strip=True))

        title = clean_text(title_node.get_text(" ", strip=True)) if title_node else clean_text(anchor.get_text(" ", strip=True))
        summary = clean_text(summary_node.get_text(" ", strip=True)) if summary_node else ""
        combined_text = clean_text(anchor.get_text(" ", strip=True))

        title = _normalize_official_title(title)
        if not title or _looks_generic_title(title, href, base_url):
            continue
        if not published_at:
            published_at = _extract_date_from_text(combined_text)
        if not published_at:
            continue
        if not summary:
            summary = clean_text(ANTHROPIC_PREFIX_PATTERN.sub("", combined_text))
            if summary.startswith(title):
                summary = clean_text(summary[len(title):])

        seen.add(href)
        rows.append({"title": title, "summary": summary, "url": href, "published_at": published_at})
        if len(rows) >= 8:
            break
    return rows


def _extract_meta_rows(page_html: str, base_url: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(page_html, "html.parser")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        href = normalize_url(anchor["href"], base_url)
        title = clean_text(anchor.get_text(" ", strip=True))
        if "/blog/" not in href or href == normalize_url(base_url) or href in seen:
            continue
        title = _normalize_official_title(title)
        if not title or _looks_generic_title(title, href, base_url):
            continue

        container = anchor.parent
        parent_text = clean_text(container.get_text(" ", strip=True)) if container else title
        published_at = _extract_date_from_text(parent_text)
        if not published_at:
            continue

        summary = parent_text.replace(title, "", 1)
        summary = DATE_PATTERN.sub("", summary)
        summary = clean_text(summary)

        seen.add(href)
        rows.append({"title": title, "summary": summary, "url": href, "published_at": published_at})
        if len(rows) >= 8:
            break
    return rows


def _fetch_html_source(source: dict[str, str], target_date: datetime, freshness_hours: int) -> list[HotspotItem]:
    page_html = fetch_text(source["url"])
    if source["mode"] == "anthropic_html":
        rows = _extract_anthropic_rows(page_html, source["url"])
    elif source["mode"] == "meta_html":
        rows = _extract_meta_rows(page_html, source["url"])
    else:
        rows = []

    items: list[HotspotItem] = []
    for row in rows:
        published_at = row["published_at"]
        if not is_fresh(published_at, target_date, freshness_hours):
            continue
        items.append(
            _build_item(
                source,
                title=row["title"],
                summary=row["summary"],
                url=row["url"],
                published_at=published_at,
            )
        )
    return items[:8]


def fetch_hotspot_items(target_date: datetime, freshness_hours: int) -> list[HotspotItem]:
    items: list[HotspotItem] = []
    for source in OFFICIAL_SOURCES:
        try:
            if source["mode"] == "rss":
                items.extend(_fetch_rss_source(source, target_date, freshness_hours))
            else:
                items.extend(_fetch_html_source(source, target_date, freshness_hours))
        except Exception as ex:
            print(f"Warning: failed to fetch official blog source {source['source_id']}: {ex}")
    return items
