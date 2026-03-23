from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from arxiv_assistant.apis.hotspot.hotspot_x_common import get_authority_record, is_authoritative_x_identity, is_newsworthy_x_text
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem, clean_text
from arxiv_assistant.utils.hotspot.hotspot_sources import clip_text, fetch_json, is_fresh
from arxiv_assistant.utils.hotspot.x_authority_registry import iter_active_authority_accounts, load_x_authority_registry

X_RECENT_SEARCH_URL = "https://api.x.com/2/tweets/search/recent"
BEARER_TOKEN_ENV_KEYS = ("X_BEARER_TOKEN", "X_API_BEARER_TOKEN", "TWITTER_BEARER_TOKEN")
X_HOSTS = {"x.com", "twitter.com", "mobile.twitter.com", "www.x.com", "www.twitter.com", "pic.x.com"}


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


def _build_query(handles: list[str]) -> str:
    return "(" + " OR ".join(f"from:{handle}" for handle in handles) + ") -is:retweet -is:reply has:links lang:en"


def _build_query_batches(handles: list[str], *, batch_size: int = 6, max_query_length: int = 260) -> list[list[str]]:
    batches: list[list[str]] = []
    current: list[str] = []
    for handle in handles:
        candidate_parts = current + [handle]
        candidate_query = _build_query(candidate_parts)
        if current and (len(candidate_parts) > batch_size or len(candidate_query) > max_query_length):
            batches.append(list(current))
            current = [handle]
        else:
            current = candidate_parts
    if current:
        batches.append(list(current))
    return batches


def _iter_recent_search(
    *,
    query: str,
    bearer_token: str,
    max_results: int,
) -> list[dict[str, Any]]:
    params = {
        "query": query,
        "tweet.fields": "created_at,public_metrics,author_id,lang,entities,referenced_tweets,in_reply_to_user_id",
        "user.fields": "username,name,verified",
        "expansions": "author_id",
        "max_results": min(100, max_results),
    }
    headers = {"Authorization": f"Bearer {bearer_token}"}
    rows: list[dict[str, Any]] = []
    remaining = max_results
    next_token: str | None = None

    while remaining > 0:
        params["max_results"] = min(100, remaining)
        if next_token:
            params["next_token"] = next_token
        elif "next_token" in params:
            del params["next_token"]

        payload = fetch_json(X_RECENT_SEARCH_URL, headers=headers, params=params)
        includes = payload.get("includes", {}) or {}
        users_by_id = {user.get("id"): user for user in includes.get("users", []) or []}

        for tweet in payload.get("data", []) or []:
            author = users_by_id.get(tweet.get("author_id"), {})
            enriched = dict(tweet)
            enriched["author"] = author
            rows.append(enriched)

        remaining = max_results - len(rows)
        next_token = (payload.get("meta") or {}).get("next_token")
        if not next_token:
            break

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


def _query_accounts(
    accounts: list[dict[str, Any]],
    *,
    bearer_token: str,
    result_limit: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    remaining = result_limit
    handles = [str(row.get("handle")) for row in accounts if row.get("handle")]

    def fetch_handle_group(group: list[str]) -> None:
        nonlocal remaining, rows
        if remaining <= 0 or not group:
            return
        query = _build_query(group)
        try:
            chunk = _iter_recent_search(query=query, bearer_token=bearer_token, max_results=min(remaining, 50))
            rows.extend(chunk)
            remaining = result_limit - len(rows)
        except Exception:
            if len(group) == 1:
                return
            midpoint = len(group) // 2
            fetch_handle_group(group[:midpoint])
            fetch_handle_group(group[midpoint:])

    for handle_group in _build_query_batches(handles, batch_size=batch_size):
        if remaining <= 0:
            break
        fetch_handle_group(handle_group)
    return rows


def fetch_hotspot_items(
    target_date: datetime,
    freshness_hours: int,
    seed_path: str | Path,
    *,
    default_result_limit: int = 80,
    snapshot_path: str | Path | None = None,
    max_age_hours: int = 24,
    official_batch_size: int = 10,
    researcher_batch_size: int = 8,
    official_account_limit: int = 60,
    researcher_account_limit: int = 90,
) -> list[HotspotItem]:
    bearer_token = _get_bearer_token()
    if not bearer_token:
        return []

    registry = load_x_authority_registry(seed_path=seed_path, snapshot_path=snapshot_path, max_age_hours=max_age_hours)
    official_accounts = iter_active_authority_accounts(registry, kinds={"official", "company"}, min_tier=2)
    researcher_accounts = iter_active_authority_accounts(registry, kinds={"researcher"}, min_tier=2)
    official_accounts = sorted(official_accounts, key=_authority_priority, reverse=True)[:official_account_limit]
    researcher_accounts = sorted(researcher_accounts, key=_authority_priority, reverse=True)[:researcher_account_limit]

    rows = _query_accounts(
        official_accounts,
        bearer_token=bearer_token,
        result_limit=default_result_limit,
        batch_size=official_batch_size,
    )
    rows.extend(
        _query_accounts(
            researcher_accounts,
            bearer_token=bearer_token,
            result_limit=max(0, default_result_limit // 3),
            batch_size=researcher_batch_size,
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
