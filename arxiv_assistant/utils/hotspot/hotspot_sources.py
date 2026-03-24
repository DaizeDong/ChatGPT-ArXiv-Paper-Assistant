from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests


USER_AGENT = "ChatGPT-ArXiv-Paper-Assistant/1.0 (+https://github.com/DaizeDong/ChatGPT-ArXiv-Paper-Assistant)"
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


def is_fresh(published_at: str | None, target_date: datetime, freshness_hours: int) -> bool:
    if published_at is None:
        return True
    published_dt = parse_datetime(published_at)
    if published_dt is None:
        return True
    if target_date.tzinfo is None:
        target_date = target_date.replace(tzinfo=UTC)
    window_start = target_date - timedelta(hours=freshness_hours)
    window_end = target_date + timedelta(hours=freshness_hours)
    return window_start <= published_dt <= window_end


def clip_text(text: str | None, limit: int = 500) -> str:
    if not text:
        return ""
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."
