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
    return []
