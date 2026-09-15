from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from arxiv_assistant.apis.hotspot.hotspot_x_common import (
    get_authority_record,
    is_authoritative_x_identity,
    is_newsworthy_x_text,
)
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem, clean_text
from arxiv_assistant.utils.hotspot.hotspot_sources import clip_text, is_fresh, record_api_usage
from arxiv_assistant.utils.hotspot.x_authority_registry import (
    iter_active_authority_accounts,
    load_x_authority_registry,
)

SOURCE_ID = "x_twitterapi"
PROVENANCE = "native:x_twitterapi"
TWITTERAPI_IO_URL = "https://api.twitterapi.io/twitter/user/last_tweets"
TWITTERAPI_IO_ENV_KEYS = ("TWITTERAPI_IO_KEY", "TWITTERAPI_KEY")
X_HOSTS = {"x.com", "twitter.com", "mobile.twitter.com", "www.x.com", "www.twitter.com", "pic.x.com"}
_RATE_LIMIT_MAX_RETRIES = 2
_RATE_LIMIT_WAIT_SECONDS = 16


def _get_twitterapi_key() -> str | None:
    for key in TWITTERAPI_IO_ENV_KEYS:
        value = clean_text(os.getenv(key))
        if value:
            return value
    return None


def _classic_created_at_to_iso(value: Any) -> str:
    """twitterapi.io returns Twitter's classic format ('Tue Dec 10 07:00:00 +0000 2024').
    Normalize to ISO 8601 'Z' so is_fresh()/published_at behave exactly as with the official
    X API v2 (which already returns ISO). Falls back to the raw string if unparseable."""
    raw = clean_text(value)
    if not raw:
        return ""
    try:
        return datetime.strptime(raw, "%a %b %d %H:%M:%S %z %Y").astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, TypeError):
        return raw


def _map_twitterapi_tweet(tw: dict[str, Any], *, handle: str, user_id: str | None) -> dict[str, Any]:
    """Map a twitterapi.io tweet to the canonical X-API-v2 dict shape the rest of this module
    consumes (text, id, created_at, public_metrics, entities, in_reply_to_user_id, author)."""
    author = tw.get("author") or {}
    return {
        "id": clean_text(tw.get("id")),
        "text": tw.get("text"),
        "created_at": _classic_created_at_to_iso(tw.get("createdAt")),
        "lang": tw.get("lang"),
        "in_reply_to_user_id": clean_text(tw.get("inReplyToUserId")) or None,
        "referenced_tweets": ([{"type": "retweeted"}] if tw.get("retweeted_tweet") else []),
        "entities": tw.get("entities", {}) or {},
        "public_metrics": {
            "like_count": int(tw.get("likeCount", 0) or 0),
            "reply_count": int(tw.get("replyCount", 0) or 0),
            "retweet_count": int(tw.get("retweetCount", 0) or 0),
            "quote_count": int(tw.get("quoteCount", 0) or 0),
            "impression_count": int(tw.get("viewCount", 0) or 0),
            "bookmark_count": int(tw.get("bookmarkCount", 0) or 0),
        },
        "author": {
            "username": clean_text(author.get("userName") or handle).lower() or clean_text(handle).lower(),
            "id": user_id or clean_text(author.get("id")),
            "name": clean_text(author.get("name")),
            "verified": bool(author.get("isBlueVerified") or author.get("verified")),
        },
    }


def _twitterapi_get(params: dict[str, Any], *, api_key: str) -> dict[str, Any]:
    headers = {"X-API-Key": api_key}
    for attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
        response = requests.get(TWITTERAPI_IO_URL, headers=headers, params=params, timeout=30)
        if response.status_code == 429:
            if attempt < _RATE_LIMIT_MAX_RETRIES:
                wait = _RATE_LIMIT_WAIT_SECONDS * (attempt + 1)
                print(f"twitterapi.io rate limited (429). Waiting {wait}s before retry {attempt + 1}...")
                time.sleep(wait)
                continue
            print("twitterapi.io rate limit exceeded after retries. Stopping.")
            return {}
        response.raise_for_status()
        record_api_usage(source_id=SOURCE_ID, provider="twitterapi.io")
        return response.json()
    return {}


def _fetch_last_tweets_rest(
    *,
    user_id: str | None,
    handle: str,
    api_key: str,
    since: datetime,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    # last_tweets resolves by userName, so a missing user_id needs no extra lookup call.
    # Prefer userId when the registry cached one (provider notes it is more stable/faster).
    params: dict[str, Any] = {"includeReplies": "false"}
    if user_id:
        params["userId"] = user_id
    else:
        params["userName"] = handle
    try:
        payload = _twitterapi_get(params, api_key=api_key)
    except Exception as ex:  # network/HTTP errors degrade to empty, never crash the pipeline
        print(f"Warning: twitterapi.io timeline fetch failed for @{handle}: {ex}")
        return []
    # tolerate both flat {tweets:[...]} and wrapped {data:{tweets:[...]}} shapes
    container = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    raw_tweets = (container or {}).get("tweets", []) or []
    rows: list[dict[str, Any]] = []
    for tw in raw_tweets:
        mapped = _map_twitterapi_tweet(tw, handle=handle, user_id=user_id)
        created = mapped.get("created_at")
        if created:
            try:
                if datetime.strptime(created, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC) < since:
                    continue  # client-side window filter (last_tweets has no start_time param)
            except ValueError:
                pass
        rows.append(mapped)
        if len(rows) >= max_results:
            break
    return rows


def _derive_title(text: str) -> str:
    text = clean_text(text)
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
    text = clean_text(tweet.get("text")).lower()
    if text.startswith("rt @") or text.startswith("@"):
        return True
    if tweet.get("in_reply_to_user_id"):
        return True
    for ref in tweet.get("referenced_tweets", []) or []:
        if ref.get("type") in {"retweeted", "replied_to"}:
            return True
    return False


def _expanded_urls(tweet: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    entities = tweet.get("entities", {}) or {}
    for row in entities.get("urls", []) or []:
        candidate = clean_text(row.get("expanded_url") or row.get("unwound_url") or row.get("url"))
        if candidate:
            urls.append(candidate)
    return urls


def _authority_source_role(record: dict[str, Any]) -> str:
    if str(record.get("kind")) in {"official", "company"}:
        return "official_news"
    return "community_heat"


def _authority_source_quality(record: dict[str, Any]) -> float:
    tier = int(record.get("tier") or 1)
    kind = str(record.get("kind") or "researcher")
    base = 1.05 if kind == "researcher" else 1.2
    return round(base + tier * 0.12, 2)


def _set_provenance(item: HotspotItem) -> HotspotItem:
    """Set the Stage-0 `provenance` field if the dataclass exposes it, and ALWAYS mirror
    provenance + source_id into metadata so downstream works whether or not Stage 0 landed."""
    if hasattr(item, "provenance"):
        item.provenance = PROVENANCE
    item.metadata.setdefault("provenance", PROVENANCE)
    item.metadata.setdefault("source_id", SOURCE_ID)
    return item


def _tweet_to_item(row: dict[str, Any], *, authority: dict[str, Any]) -> HotspotItem | None:
    if _is_reply_or_retweet(row):
        return None

    author = row.get("author", {}) or {}
    author_handle = clean_text(authority.get("handle") or author.get("username"))
    text = clean_text(row.get("text"))
    tweet_id = clean_text(row.get("id"))
    if not author_handle or not text or not tweet_id:
        return None

    metrics = row.get("public_metrics", {}) or {}
    activity = _compute_activity(metrics)
    expanded_urls = _expanded_urls(row)
    if not is_newsworthy_x_text(
        text,
        authority_kind=str(authority.get("kind") or "official"),
        expanded_urls=expanded_urls,
        activity=activity,
    ):
        return None

    url = f"https://x.com/{author_handle}/status/{tweet_id}"
    non_x_urls = [entry for entry in expanded_urls if urlsplit(entry).netloc.lower() not in X_HOSTS]
    created_at = row.get("created_at")

    item = HotspotItem(
        source_id=SOURCE_ID,
        source_name=clean_text(authority.get("name") or author.get("name") or author_handle),
        source_role=_authority_source_role(authority),
        source_type="tweet",
        title=_derive_title(text),
        summary=clip_text(text, 420),
        url=url,
        canonical_url=url,
        published_at=created_at,
        tags=["x-twitterapi", str(authority.get("kind") or "official")],
        authors=[author_handle],
        metadata={
            "tweet_id": tweet_id,
            "author_handle": author_handle,
            "author_name": clean_text(author.get("name")),
            "verified": bool(author.get("verified")),
            "public_metrics": metrics,
            "activity": activity,
            "source_quality": _authority_source_quality(authority),
            "signal_tier": "x_twitterapi_timeline",
            "authority_kind": str(authority.get("kind") or "official"),
            "authority_tier": int(authority.get("tier") or 1),
            "organization": clean_text(authority.get("organization")),
            "expanded_urls": expanded_urls,
            "non_x_urls": non_x_urls,
            "has_external_link": bool(non_x_urls),
            "host": "x.com",
        },
    )
    return _set_provenance(item)


def _collect_timelines(
    accounts: list[dict[str, Any]],
    *,
    api_key: str,
    since: datetime,
    tweets_per_user: int,
    result_limit: int,
    registry: dict[str, Any],
    seen_urls: set[str],
) -> list[HotspotItem]:
    items: list[HotspotItem] = []
    for account in accounts:
        if len(items) >= result_limit:
            break
        handle = clean_text(account.get("handle"))
        if not handle:
            continue
        user_id = clean_text(account.get("x_user_id")) or None
        rows = _fetch_last_tweets_rest(
            user_id=user_id,
            handle=handle,
            api_key=api_key,
            since=since,
            max_results=tweets_per_user,
        )
        for row in rows:
            author = row.get("author", {}) or {}
            author_handle = clean_text(author.get("username")) or handle
            if not is_authoritative_x_identity(author_handle, registry=registry):
                continue
            authority = get_authority_record(author_handle, registry=registry) or account
            item = _tweet_to_item(row, authority=authority)
            if item is None:
                continue
            if item.url in seen_urls:
                continue
            seen_urls.add(item.url)
            items.append(item)
            if len(items) >= result_limit:
                break
    return items


def fetch_hotspot_items(
    target_date: datetime,
    freshness_hours: int,
    seed_path: str | Path,
    *,
    result_limit: int = 80,
    snapshot_path: str | Path | None = None,
    max_age_hours: int = 24,
    official_account_limit: int = 24,
    researcher_account_limit: int = 18,
    tweets_per_user: int = 10,
) -> list[HotspotItem]:
    api_key = _get_twitterapi_key()
    if not api_key:
        print(
            "Warning: no twitterapi.io key configured. Set one of "
            f"{TWITTERAPI_IO_ENV_KEYS} to enable the x_twitterapi source. Skipping."
        )
        return []

    if target_date.tzinfo is None:
        since = target_date.replace(tzinfo=UTC) - timedelta(hours=freshness_hours)
    else:
        since = target_date - timedelta(hours=freshness_hours)

    registry = load_x_authority_registry(
        seed_path=seed_path, snapshot_path=snapshot_path, max_age_hours=max_age_hours
    )
    official_accounts = iter_active_authority_accounts(registry, kinds={"official", "company"}, min_tier=2)
    researcher_accounts = iter_active_authority_accounts(registry, kinds={"researcher"}, min_tier=2)
    official_accounts = official_accounts[:official_account_limit]
    researcher_accounts = researcher_accounts[:researcher_account_limit]

    seen_urls: set[str] = set()
    items = _collect_timelines(
        official_accounts,
        api_key=api_key,
        since=since,
        tweets_per_user=tweets_per_user,
        result_limit=result_limit,
        registry=registry,
        seen_urls=seen_urls,
    )
    items.extend(
        _collect_timelines(
            researcher_accounts,
            api_key=api_key,
            since=since,
            tweets_per_user=tweets_per_user,
            result_limit=max(0, result_limit // 3),
            registry=registry,
            seen_urls=seen_urls,
        )
    )

    # Server-side window is approximate; enforce the canonical is_fresh() gate too.
    fresh_items = [
        item
        for item in items
        if item.published_at is None or is_fresh(item.published_at, target_date, freshness_hours)
    ]
    return fresh_items[:result_limit]
