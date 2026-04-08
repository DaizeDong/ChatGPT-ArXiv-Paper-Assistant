from __future__ import annotations

from datetime import UTC, datetime, timedelta
from html import unescape
import re
from urllib.parse import urlsplit

from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem, clean_text
from arxiv_assistant.utils.hotspot.hotspot_sources import clip_text, fetch_json, is_fresh

# Algolia-based HN Search API — supports historical date queries
HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search"
HN_ITEM_PAGE_URL = "https://news.ycombinator.com/item?id={item_id}"
AI_HOST_KEYWORDS = (
    "openai.com",
    "anthropic.com",
    "huggingface.co",
    "arxiv.org",
    "ai.meta.com",
    "blog.google",
    "deepmind.google",
)


def _strip_html(text: str | None) -> str:
    return clean_text(re.sub(r"<[^>]+>", " ", unescape(text or "")))


def _story_matches(title: str, url: str, keyword_filter: list[str]) -> bool:
    host = urlsplit(url).netloc.lower()
    combined = f"{title.lower()} {url.lower()}"
    if any(keyword in host for keyword in AI_HOST_KEYWORDS):
        return True
    return any(keyword and keyword in combined for keyword in keyword_filter)


def fetch_hotspot_items(
    target_date: datetime,
    freshness_hours: int,
    keyword_filter: list[str],
    story_limit: int,
    score_cutoff: int,
    comments_cutoff: int,
) -> list[HotspotItem]:
    target_utc = target_date.replace(tzinfo=UTC) if target_date.tzinfo is None else target_date
    window_start = target_utc - timedelta(hours=freshness_hours)
    window_end = target_utc + timedelta(hours=6)
    ts_start = int(window_start.timestamp())
    ts_end = int(window_end.timestamp())

    # Build AI-focused search queries and merge results
    ai_queries = ["AI", "LLM", "GPT", "machine learning", "deep learning", "neural network"]
    seen_ids: set[str] = set()
    all_hits: list[dict] = []

    for query in ai_queries:
        try:
            data = fetch_json(
                f"{HN_SEARCH_URL}?query={query}&tags=story"
                f"&numericFilters=created_at_i>{ts_start},created_at_i<{ts_end},points>{score_cutoff}"
                f"&hitsPerPage={story_limit}",
                timeout=15,
            )
        except Exception as ex:
            print(f"Warning: HN Algolia search failed for '{query}': {ex}")
            continue
        for hit in data.get("hits", []):
            oid = hit.get("objectID", "")
            if oid and oid not in seen_ids:
                seen_ids.add(oid)
                all_hits.append(hit)

    items: list[HotspotItem] = []
    for hit in all_hits:
        title = clean_text(hit.get("title", ""))
        story_url = hit.get("url") or HN_ITEM_PAGE_URL.format(item_id=hit.get("objectID", ""))
        score = int(hit.get("points", 0) or 0)
        comments = int(hit.get("num_comments", 0) or 0)

        if score < score_cutoff:
            continue
        if comments < comments_cutoff:
            continue

        created_at = hit.get("created_at", "")
        published_at = created_at if created_at else None
        if not is_fresh(published_at, target_date, freshness_hours):
            continue
        if not _story_matches(title, story_url, keyword_filter):
            continue

        items.append(
            HotspotItem(
                source_id="hn_discussion",
                source_name="Hacker News",
                source_role="hn_discussion",
                source_type="discussion",
                title=title,
                summary=clip_text(title, 420),
                url=story_url,
                canonical_url=story_url,
                published_at=published_at,
                tags=[],
                authors=[hit.get("author", "")] if hit.get("author") else [],
                metadata={
                    "hn_id": hit.get("objectID", ""),
                    "hn_score": score,
                    "hn_comments": comments,
                    "host": urlsplit(story_url).netloc.lower(),
                },
            )
        )

    # Sort by score descending, limit
    items.sort(key=lambda x: x.metadata.get("hn_score", 0), reverse=True)
    return items[:story_limit]
