from __future__ import annotations

from datetime import UTC, datetime
from html import unescape
import re
from urllib.parse import urlsplit

from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem, clean_text
from arxiv_assistant.utils.hotspot.hotspot_sources import clip_text, fetch_json, is_fresh

HN_TOPSTORIES_URL = "https://hacker-news.firebaseio.com/v0/topstories.json"
HN_ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
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


def _story_matches(story: dict, keyword_filter: list[str]) -> bool:
    title = clean_text(story.get("title", "")).lower()
    text = _strip_html(story.get("text", "")).lower()
    url = clean_text(story.get("url", "")).lower()
    host = urlsplit(url).netloc.lower()
    combined = " ".join([title, text, url])
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
    story_ids = fetch_json(HN_TOPSTORIES_URL)
    items: list[HotspotItem] = []

    for story_id in story_ids[:story_limit]:
        story = fetch_json(HN_ITEM_URL.format(item_id=story_id))
        if not story or story.get("type") != "story" or story.get("deleted") or story.get("dead"):
            continue
        if int(story.get("score", 0) or 0) < score_cutoff:
            continue
        if int(story.get("descendants", 0) or 0) < comments_cutoff:
            continue
        published_at = datetime.fromtimestamp(int(story.get("time", 0) or 0), tz=UTC).isoformat()
        if not is_fresh(published_at, target_date, freshness_hours):
            continue
        if not _story_matches(story, keyword_filter):
            continue
        story_url = story.get("url") or HN_ITEM_PAGE_URL.format(item_id=story_id)
        items.append(
            HotspotItem(
                source_id="hn_discussion",
                source_name="Hacker News",
                source_role="hn_discussion",
                source_type="discussion",
                title=clean_text(story.get("title", f"HN story {story_id}")),
                summary=clip_text(_strip_html(story.get("text")) or clean_text(story.get("title", "")), 420),
                url=story_url,
                canonical_url=story_url,
                published_at=published_at,
                tags=[],
                authors=[story.get("by", "")] if story.get("by") else [],
                metadata={
                    "hn_id": story_id,
                    "hn_score": int(story.get("score", 0) or 0),
                    "hn_comments": int(story.get("descendants", 0) or 0),
                    "host": urlsplit(story_url).netloc.lower(),
                },
            )
        )
    return items
