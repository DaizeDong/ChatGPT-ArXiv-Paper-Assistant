from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from arxiv_assistant.apis.hotspot.hotspot_x_common import get_authority_record, is_authoritative_x_identity, is_newsworthy_x_text
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem, clean_text
from arxiv_assistant.utils.hotspot.hotspot_sources import clip_text, fetch_json, is_fresh

PAPERPULSE_RESEARCHER_FEED_URL = "https://www.paperpulse.ai/api/researcher-feed"
URL_PATTERN = re.compile(r"https?://t\.co/\w+", re.I)


def _clean_tweet_text(text: str | None) -> str:
    return clean_text(URL_PATTERN.sub("", text or ""))


def _extract_urls(text: str | None) -> list[str]:
    return re.findall(r"https?://\S+", text or "")


def _derive_title(text: str) -> str:
    if len(text) <= 180:
        return text
    for marker in (". ", "; ", ": "):
        if marker in text[:180]:
            return clean_text(text.split(marker, 1)[0])
    return clip_text(text, 180)


def _compute_activity(metrics: dict[str, Any] | None) -> int:
    metrics = metrics or {}
    likes = int(metrics.get("like_count", 0) or 0)
    replies = int(metrics.get("reply_count", 0) or 0)
    reposts = int(metrics.get("retweet_count", 0) or 0)
    quotes = int(metrics.get("quote_count", 0) or 0)
    impressions = int(metrics.get("impression_count", 0) or 0)
    bookmarks = int(metrics.get("bookmark_count", 0) or 0)
    return likes + bookmarks + replies * 3 + reposts * 2 + quotes * 4 + impressions // 1000


def _is_reply_or_retweet(tweet: dict[str, Any]) -> bool:
    text = (tweet.get("text") or "").strip().lower()
    if text.startswith("rt @") or text.startswith("@"):
        return True
    for ref in tweet.get("referenced_tweets", []) or []:
        if ref.get("type") in {"retweeted", "replied_to"}:
            return True
    return False


def fetch_hotspot_items(target_date: datetime, freshness_hours: int, *, result_limit: int = 20) -> list[HotspotItem]:
    try:
        payload = fetch_json(PAPERPULSE_RESEARCHER_FEED_URL)
    except Exception as ex:
        print(f"Warning: PaperPulse API fetch failed ({PAPERPULSE_RESEARCHER_FEED_URL}): {ex}")
        return []
    rows = payload.get("tweets", []) if isinstance(payload, dict) else []
    if not rows:
        print("Warning: PaperPulse API returned 0 tweets")
        return []

    # Check if the API is returning stale data (frozen feed)
    from arxiv_assistant.utils.hotspot.hotspot_sources import parse_datetime
    newest_date = None
    for row in rows[:5]:
        dt = parse_datetime(row.get("created_at"))
        if dt and (newest_date is None or dt > newest_date):
            newest_date = dt
    if newest_date:
        from datetime import UTC, timedelta
        staleness_limit = target_date - timedelta(days=7)
        if newest_date.tzinfo is None:
            newest_date = newest_date.replace(tzinfo=UTC)
        if target_date.tzinfo is None:
            target_date_tz = target_date.replace(tzinfo=UTC)
        else:
            target_date_tz = target_date
        if newest_date < staleness_limit:
            age_days = (target_date_tz - newest_date).days
            print(f"Warning: PaperPulse API data is stale (newest tweet is {age_days} days old). "
                  f"API may be frozen. Returning 0 items.")
            return []

    items: list[HotspotItem] = []
    seen_urls: set[str] = set()

    for row in rows:
        created_at = row.get("created_at")
        if created_at and not is_fresh(created_at, target_date, freshness_hours):
            continue
        if _is_reply_or_retweet(row):
            continue

        text = _clean_tweet_text(row.get("text"))
        author_handle = clean_text(row.get("author_handle"))
        tweet_id = clean_text(row.get("tweet_id"))
        if not text or not author_handle or not tweet_id:
            continue
        if not is_authoritative_x_identity(author_handle, allowed_kinds={"researcher"}):
            continue
        authority = get_authority_record(author_handle)
        if authority is None:
            continue
        expanded_urls = _extract_urls(row.get("text"))
        if not is_newsworthy_x_text(
            text,
            authority_kind="researcher",
            expanded_urls=expanded_urls,
            activity=_compute_activity(row.get("public_metrics", {}) or {}),
        ):
            continue
        url = f"https://x.com/{author_handle}/status/{tweet_id}"
        if url in seen_urls:
            continue
        seen_urls.add(url)

        metrics = row.get("public_metrics", {}) or {}
        activity = _compute_activity(metrics)
        items.append(
            HotspotItem(
                source_id="paperpulse_researcher_feed",
                source_name="PaperPulse Researcher Feed",
                source_role="community_heat",
                source_type="tweet_feed",
                title=_derive_title(text),
                summary=clip_text(text, 420),
                url=url,
                canonical_url=url,
                published_at=created_at,
                tags=["researcher-feed", "x-proxy"],
                authors=[author_handle],
                metadata={
                    "tweet_id": tweet_id,
                    "author_handle": author_handle,
                    "author_name": clean_text(row.get("author_name")),
                    "public_metrics": metrics,
                    "activity": activity,
                    "source_quality": max(1.15, 1.0 + int(authority.get("tier") or 1) * 0.12),
                    "signal_tier": "x_researcher_proxy",
                    "host": "x.com",
                    "proxy_source": "paperpulse",
                    "authority_kind": "researcher",
                    "authority_tier": int(authority.get("tier") or 1),
                    "organization": clean_text(authority.get("organization")),
                },
            )
        )
        if len(items) >= result_limit:
            break

    return items
