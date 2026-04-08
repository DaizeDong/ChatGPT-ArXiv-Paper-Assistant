from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests


USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
_CURRENT_USAGE_SOURCE: str | None = None
_CURRENT_USAGE_PROVIDER: str | None = None
_API_USAGE: dict[str, dict[str, Any]] = {}


@contextmanager
def api_usage_scope(source_id: str, provider: str) -> Any:
    global _CURRENT_USAGE_SOURCE, _CURRENT_USAGE_PROVIDER
    previous_source = _CURRENT_USAGE_SOURCE
    previous_provider = _CURRENT_USAGE_PROVIDER
    _CURRENT_USAGE_SOURCE = source_id
    _CURRENT_USAGE_PROVIDER = provider
    try:
        yield
    finally:
        _CURRENT_USAGE_SOURCE = previous_source
        _CURRENT_USAGE_PROVIDER = previous_provider


def reset_api_usage() -> None:
    _API_USAGE.clear()


def record_api_usage(*, requests: int = 1, estimated_cost: float = 0.0, source_id: str | None = None, provider: str | None = None) -> None:
    source_key = source_id or _CURRENT_USAGE_SOURCE
    provider_name = provider or _CURRENT_USAGE_PROVIDER or "external"
    if not source_key:
        return
    row = _API_USAGE.setdefault(
        source_key,
        {
            "provider": provider_name,
            "requests": 0,
            "estimated_cost": 0.0,
        },
    )
    row["provider"] = provider_name
    row["requests"] += int(requests)
    row["estimated_cost"] = round(float(row.get("estimated_cost", 0.0)) + float(estimated_cost or 0.0), 6)


def snapshot_api_usage() -> dict[str, dict[str, Any]]:
    return {
        source_id: {
            "provider": str(row.get("provider", "")),
            "requests": int(row.get("requests", 0) or 0),
            "estimated_cost": round(float(row.get("estimated_cost", 0.0) or 0.0), 6),
        }
        for source_id, row in _API_USAGE.items()
    }


def fetch_text(url: str, timeout: int = 30) -> str:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.encoding or "utf-8"
    record_api_usage()
    return response.text


def fetch_json(url: str, timeout: int = 30, headers: dict[str, str] | None = None, params: dict[str, Any] | None = None) -> Any:
    request_headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }
    if headers:
        request_headers.update(headers)
    response = requests.get(
        url,
        headers=request_headers,
        params=params,
        timeout=timeout,
    )
    response.raise_for_status()
    record_api_usage()
    return response.json()


def normalize_url(url: str, base_url: str | None = None) -> str:
    if base_url:
        url = urljoin(base_url, url)
    parts = list(urlsplit(url))
    parts[3] = ""
    parts[4] = ""
    normalized = urlunsplit(parts)
    return normalized.rstrip("/")


def load_roundup_registry(config_path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("sites", "sources"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"Unsupported hotspot registry format: {config_path}")


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    candidates = [
        normalized,
        normalized.replace(" GMT", "+00:00"),
    ]
    for candidate in candidates:
        try:
            dt = datetime.fromisoformat(candidate)
            return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
        except ValueError:
            continue
    for fmt in (
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d",
        "%b %d, %Y",
        "%B %d, %Y",
    ):
        try:
            dt = datetime.strptime(value, fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


_FETCHED_AT_VALID_SOURCES = {"github_trend"}


def get_freshness_date(item: "HotspotItem") -> str | None:
    """Return the most appropriate date for freshness evaluation.

    Only github_trend sources may use fetched_at to override published_at,
    since GitHub repos can trend long after creation. All other sources
    use published_at directly.
    """
    if item.source_id in _FETCHED_AT_VALID_SOURCES:
        fetched_at = (item.metadata or {}).get("fetched_at")
        if fetched_at:
            return fetched_at
    return item.published_at


def is_fresh(published_at: str | None, target_date: datetime, freshness_hours: int) -> bool:
    if published_at is None:
        return True
    published_dt = parse_datetime(published_at)
    if published_dt is None:
        return True
    if target_date.tzinfo is None:
        target_date = target_date.replace(tzinfo=UTC)
    window_start = target_date - timedelta(hours=freshness_hours)
    window_end = target_date + timedelta(hours=6)
    return window_start <= published_dt <= window_end


def clip_text(text: str | None, limit: int = 500) -> str:
    if not text:
        return ""
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


# --- URL-title consistency validation ---

import re as _re

_TRUSTED_URL_HOSTS = {
    "arxiv.org", "huggingface.co", "openai.com", "anthropic.com",
    "blog.google", "deepmind.google", "ai.meta.com", "microsoft.com",
    "nvidia.com", "deeplearning.ai", "x.com", "twitter.com",
}

_URL_STOPWORDS = {
    "the", "and", "for", "are", "with", "that", "this", "from", "very",
    "good", "more", "about", "just", "some", "been", "have", "will",
    "new", "can", "all", "not", "but", "was", "its", "has", "how",
    "into", "than", "your", "models", "model", "code", "open", "source",
}


def url_title_consistent(title: str, url: str) -> bool:
    """Check if a URL is plausibly related to the given title.

    Returns True (pass) for trusted domains, short titles, or titles with
    keyword overlap in the URL path.  Returns False when the URL has no
    detectable relation to the title — a strong signal of mis-extraction.
    """
    if not url or not title:
        return True

    parsed = urlsplit(url)
    host = parsed.netloc.lower()

    # Trusted domains always pass
    if any(trusted in host for trusted in _TRUSTED_URL_HOSTS):
        return True

    # Reddit URLs encode the title in the slug — always pass
    if "reddit.com" in host:
        return True

    # Titles with significant non-ASCII content (CJK, etc.) can't be
    # checked against URL slugs — always pass them through.
    non_ascii = sum(1 for c in title if ord(c) > 127)
    if non_ascii >= len(title) * 0.3:
        return True

    # Extract significant title tokens (>= 3 chars, not stopwords)
    title_tokens = {
        t for t in _re.findall(r"[a-z0-9]+", title.lower())
        if len(t) >= 3 and t not in _URL_STOPWORDS
    }
    if len(title_tokens) < 2:
        return True  # Too little info to judge

    # Also check stemmed forms (strip trailing 's' for plural)
    title_stems = title_tokens | {t.rstrip("s") for t in title_tokens if len(t) >= 4}

    # Extract tokens from URL path (repo name, article slug, etc.)
    path_tokens = set(_re.findall(r"[a-z0-9]+", parsed.path.lower()))
    url_full = f"{host}{parsed.path}".lower()

    # Check host for concatenated names (e.g., artificialintelligencemadesimple.com)
    host_raw = host.replace("www.", "")
    if any(t in host_raw for t in title_tokens if len(t) >= 4):
        return True

    # Check if any longer title token appears as substring in full URL
    if any(t in url_full for t in title_stems if len(t) >= 5):
        return True

    # For GitHub: check repo name and owner
    if "github.com" in host:
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 2:
            repo_tokens = set(_re.findall(r"[a-z0-9]+", parts[1].lower()))
            owner_tokens = set(_re.findall(r"[a-z0-9]+", parts[0].lower()))
            combined = repo_tokens | owner_tokens
            if title_tokens & combined:
                return True
        return False  # GitHub URL with no title overlap → mismatch

    # General check: any title token in URL path or host tokens
    host_tokens = set(_re.findall(r"[a-z0-9]+", host))
    return bool(title_tokens & (path_tokens | host_tokens))
