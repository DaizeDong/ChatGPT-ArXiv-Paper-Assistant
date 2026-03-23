from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem
from arxiv_assistant.utils.hotspot.hotspot_sources import clip_text, fetch_json

GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"
GITHUB_API_VERSION = "2026-03-10"


def _github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _build_query(raw_query: str, target_date: datetime, stars_cutoff: int, created_within_days: int) -> str:
    since = (target_date - timedelta(days=created_within_days)).date().isoformat()
    return f"{raw_query} stars:>={stars_cutoff} created:>={since} archived:false fork:false"


def fetch_hotspot_items(
    target_date: datetime,
    search_queries: list[str],
    stars_cutoff: int,
    created_within_days: int,
    result_limit: int,
) -> list[HotspotItem]:
    if not search_queries or result_limit <= 0:
        return []

    items: list[HotspotItem] = []
    seen_repos: set[str] = set()
    per_query_limit = max(3, min(10, result_limit))

    for raw_query in search_queries:
        query = _build_query(raw_query, target_date, stars_cutoff, created_within_days)
        payload = fetch_json(
            GITHUB_SEARCH_URL,
            headers=_github_headers(),
            params={
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": per_query_limit,
                "page": 1,
            },
        )
        for row in payload.get("items", []):
            full_name = row.get("full_name")
            html_url = row.get("html_url")
            if not full_name or not html_url or full_name in seen_repos:
                continue
            seen_repos.add(full_name)
            created_at = row.get("created_at") or row.get("updated_at") or datetime.now(UTC).isoformat()
            items.append(
                HotspotItem(
                    source_id="github_trend",
                    source_name="GitHub Trending Repos",
                    source_role="github_trend",
                    source_type="repository",
                    title=full_name,
                    summary=clip_text(row.get("description", ""), 420),
                    url=html_url,
                    canonical_url=html_url,
                    published_at=created_at,
                    tags=list(row.get("topics") or []),
                    authors=[row.get("owner", {}).get("login", "")] if row.get("owner", {}).get("login") else [],
                    metadata={
                        "stars": row.get("stargazers_count", 0),
                        "forks": row.get("forks_count", 0),
                        "language": row.get("language", ""),
                        "github_url": html_url,
                        "github_full_name": full_name,
                        "search_query": raw_query,
                    },
                )
            )
            if len(items) >= result_limit:
                return items

    return items
