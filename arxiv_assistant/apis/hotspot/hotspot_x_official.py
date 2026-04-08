from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from arxiv_assistant.apis.hotspot.hotspot_x_common import get_authority_record, is_authoritative_x_identity, is_newsworthy_x_text
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem, clean_text
from arxiv_assistant.utils.hotspot.hotspot_sources import clip_text, is_fresh, record_api_usage
from arxiv_assistant.utils.hotspot.x_authority_registry import iter_active_authority_accounts, load_x_authority_registry

X_USER_TIMELINE_URL = "https://api.x.com/2/users/{user_id}/tweets"
X_USER_LOOKUP_URL = "https://api.x.com/2/users/by/username/{username}"
BEARER_TOKEN_ENV_KEYS = ("X_BEARER_TOKEN", "X_API_BEARER_TOKEN", "TWITTER_BEARER_TOKEN")
X_HOSTS = {"x.com", "twitter.com", "mobile.twitter.com", "www.x.com", "www.twitter.com", "pic.x.com"}
_RATE_LIMIT_MAX_RETRIES = 2
_RATE_LIMIT_WAIT_SECONDS = 16


def _get_bearer_token() -> str | None:
    for key in BEARER_TOKEN_ENV_KEYS:
        value = clean_text(os.getenv(key))
        if value:
            return value
    return None


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
    urls = []
    entities = tweet.get("entities", {}) or {}
    for row in entities.get("urls", []) or []:
        candidate = clean_text(row.get("expanded_url") or row.get("unwound_url") or row.get("url"))
        if candidate:
            urls.append(candidate)
    return urls


def _x_api_get(url: str, headers: dict[str, str], params: dict[str, Any] | None = None) -> dict[str, Any]:
    for attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
        response = requests.get(url, headers=headers, params=params, timeout=30)
        if response.status_code == 429:
            if attempt < _RATE_LIMIT_MAX_RETRIES:
                wait = _RATE_LIMIT_WAIT_SECONDS * (attempt + 1)
                print(f"X API rate limited (429). Waiting {wait}s before retry {attempt + 1}...")
                time.sleep(wait)
                continue
            print("X API rate limit exceeded after retries. Stopping.")
            return {}
        response.raise_for_status()
        record_api_usage()
        return response.json()
    return {}


def _lookup_user_id(handle: str, *, bearer_token: str) -> str | None:
    url = X_USER_LOOKUP_URL.format(username=handle)
    headers = {"Authorization": f"Bearer {bearer_token}"}
    try:
        payload = _x_api_get(url, headers=headers)
    except Exception as ex:
        print(f"Warning: X user lookup failed for @{handle}: {ex}")
        return None
    data = payload.get("data", {}) or {}
    return clean_text(data.get("id"))


def _iter_user_timeline(
    *,
    user_id: str,
    handle: str,
    bearer_token: str,
    since: datetime,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    url = X_USER_TIMELINE_URL.format(user_id=user_id)
    params = {
        "tweet.fields": "created_at,public_metrics,author_id,lang,entities,referenced_tweets,in_reply_to_user_id",
        "max_results": min(100, max_results),
        "start_time": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "exclude": "retweets,replies",
    }
    headers = {"Authorization": f"Bearer {bearer_token}"}
    try:
        payload = _x_api_get(url, headers=headers, params=params)
    except Exception as ex:
        print(f"Warning: X user timeline fetch failed for @{handle} ({user_id}): {ex}")
        return []
    rows: list[dict[str, Any]] = []
    for tweet in payload.get("data", []) or []:
        enriched = dict(tweet)
        enriched["author"] = {"username": handle, "id": user_id}
        rows.append(enriched)
    return rows


def _authority_source_role(record: dict[str, Any]) -> str:
    if str(record.get("kind")) in {"official", "company"}:
        return "official_news"
    return "community_heat"


def _authority_source_quality(record: dict[str, Any]) -> float:
    tier = int(record.get("tier") or 1)
    kind = str(record.get("kind") or "researcher")
    base = 1.05 if kind == "researcher" else 1.2
    return round(base + tier * 0.12, 2)


def _authority_priority(record: dict[str, Any]) -> tuple[int, int, int]:
    source_refs = set(record.get("source_refs") or [])
    return (
        1 if "manual_seed" in source_refs else 0,
        int(record.get("tier") or 0),
        int(record.get("graph_support") or 0),
        len(source_refs),
    )


def _fetch_timelines(
    accounts: list[dict[str, Any]],
    *,
    bearer_token: str,
    since: datetime,
    tweets_per_user: int = 10,
    result_limit: int = 80,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    user_id_cache: dict[str, str] = {}

    for account in accounts:
        if len(rows) >= result_limit:
            break
        handle = str(account.get("handle") or "")
        if not handle:
            continue

        user_id = clean_text(account.get("x_user_id")) or user_id_cache.get(handle)
        if not user_id:
            user_id = _lookup_user_id(handle, bearer_token=bearer_token)
            if not user_id:
                continue
            user_id_cache[handle] = user_id

        chunk = _iter_user_timeline(
            user_id=user_id,
            handle=handle,
            bearer_token=bearer_token,
            since=since,
            max_results=tweets_per_user,
        )
        rows.extend(chunk)

    return rows[:result_limit]


def fetch_hotspot_items(
    target_date: datetime,
    freshness_hours: int,
    seed_path: str | Path,
    *,
    default_result_limit: int = 80,
    snapshot_path: str | Path | None = None,
    max_age_hours: int = 24,
    official_account_limit: int = 24,
    researcher_account_limit: int = 18,
    tweets_per_user: int = 10,
) -> list[HotspotItem]:
    bearer_token = _get_bearer_token()
    if not bearer_token:
        print("Warning: X API bearer token not configured. Set one of "
              f"{BEARER_TOKEN_ENV_KEYS} to enable X/Twitter source. "
              "Skipping x_official source.")
        return []

    if target_date.tzinfo is None:
        since = (target_date.replace(tzinfo=UTC) - timedelta(hours=freshness_hours))
    else:
        since = target_date - timedelta(hours=freshness_hours)

    registry = load_x_authority_registry(seed_path=seed_path, snapshot_path=snapshot_path, max_age_hours=max_age_hours)
    official_accounts = iter_active_authority_accounts(registry, kinds={"official", "company"}, min_tier=2)
    researcher_accounts = iter_active_authority_accounts(registry, kinds={"researcher"}, min_tier=2)
    official_accounts = sorted(official_accounts, key=_authority_priority, reverse=True)[:official_account_limit]
    researcher_accounts = sorted(researcher_accounts, key=_authority_priority, reverse=True)[:researcher_account_limit]

    rows = _fetch_timelines(
        official_accounts,
        bearer_token=bearer_token,
        since=since,
        tweets_per_user=tweets_per_user,
        result_limit=default_result_limit,
    )
    rows.extend(
        _fetch_timelines(
            researcher_accounts,
            bearer_token=bearer_token,
            since=since,
            tweets_per_user=tweets_per_user,
            result_limit=max(0, default_result_limit // 3),
        )
    )

    items: list[HotspotItem] = []
    seen_urls: set[str] = set()

    for row in rows:
        created_at = row.get("created_at")
        if created_at and not is_fresh(created_at, target_date, freshness_hours):
            continue
        if _is_reply_or_retweet(row):
            continue

        author = row.get("author", {}) or {}
        author_handle = clean_text(author.get("username"))
        if not is_authoritative_x_identity(author_handle, registry=registry):
            continue
        authority = get_authority_record(author_handle, registry=registry)
        if authority is None:
            continue

        text = clean_text(row.get("text"))
        tweet_id = clean_text(row.get("id"))
        if not text or not tweet_id:
            continue

        metrics = row.get("public_metrics", {}) or {}
        activity = _compute_activity(metrics)
        expanded_urls = _expanded_urls(row)
        if not is_newsworthy_x_text(
            text,
            authority_kind=str(authority.get("kind") or "official"),
            expanded_urls=expanded_urls,
            activity=activity,
        ):
            continue

        url = f"https://x.com/{author_handle}/status/{tweet_id}"
        if url in seen_urls:
            continue
        seen_urls.add(url)
        non_x_urls = [entry for entry in expanded_urls if urlsplit(entry).netloc.lower() not in X_HOSTS]

        items.append(
            HotspotItem(
                source_id=f"x_authority:{author_handle}",
                source_name=clean_text(authority.get("name") or author.get("name") or author_handle),
                source_role=_authority_source_role(authority),
                source_type="tweet",
                title=_derive_title(text),
                summary=clip_text(text, 420),
                url=url,
                canonical_url=url,
                published_at=created_at,
                tags=["x-authority", str(authority.get("kind") or "official")],
                authors=[author_handle],
                metadata={
                    "tweet_id": tweet_id,
                    "author_handle": author_handle,
                    "author_name": clean_text(author.get("name")),
                    "verified": bool(author.get("verified")),
                    "public_metrics": metrics,
                    "activity": activity,
                    "source_quality": _authority_source_quality(authority),
                    "signal_tier": "x_authority_search",
                    "authority_kind": str(authority.get("kind") or "official"),
                    "authority_tier": int(authority.get("tier") or 1),
                    "organization": clean_text(authority.get("organization")),
                    "expanded_urls": expanded_urls,
                    "non_x_urls": non_x_urls,
                    "has_external_link": bool(non_x_urls),
                    "host": "x.com",
                },
            )
        )
    return items
