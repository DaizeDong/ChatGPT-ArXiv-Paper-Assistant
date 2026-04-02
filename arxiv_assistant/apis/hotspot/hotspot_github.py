from __future__ import annotations

import os
import re
from datetime import UTC, datetime, timedelta

from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem
from arxiv_assistant.utils.hotspot.hotspot_sources import clip_text, fetch_json

GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"
GITHUB_API_VERSION = "2026-03-10"

BLACKLIST_NAME_PATTERNS = [
    re.compile(r"^awesome[-_]", re.I),
    re.compile(r"[-_]awesome$", re.I),
]
BLACKLIST_DESC_KEYWORDS_LOW_STARS = {
    "wrapper", "wechat", "weixin", "\u5fae\u4fe1",  # 微信
}
BLACKLIST_DESC_KEYWORD_STAR_THRESHOLD = 500
MCP_AGENT_SDK_STAR_THRESHOLD = 300


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

            # Blacklist: skip awesome-lists
            repo_name = full_name.split("/", 1)[-1] if "/" in full_name else full_name
            if any(pat.search(repo_name) for pat in BLACKLIST_NAME_PATTERNS):
                continue

            stars = int(row.get("stargazers_count", 0) or 0)
            description = (row.get("description") or "").lower()

            # Blacklist: skip wrapper/wechat repos below star threshold
            if any(kw in description for kw in BLACKLIST_DESC_KEYWORDS_LOW_STARS) and stars < BLACKLIST_DESC_KEYWORD_STAR_THRESHOLD:
                continue

            # Blacklist: skip generic mcp/agent-sdk repos below threshold
            name_lower = repo_name.lower()
            if (name_lower.startswith("mcp-") or name_lower.endswith("-mcp") or
                name_lower.endswith("-agent-sdk")) and stars < MCP_AGENT_SDK_STAR_THRESHOLD:
                if not any(term in description for term in ("model", "training", "benchmark", "transformer")):
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
