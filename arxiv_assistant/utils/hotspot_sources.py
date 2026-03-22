from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests


USER_AGENT = "ChatGPT-ArXiv-Paper-Assistant/1.0 (+https://github.com/DaizeDong/ChatGPT-ArXiv-Paper-Assistant)"


def fetch_text(url: str, timeout: int = 30) -> str:
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.encoding or "utf-8"
    return response.text


def fetch_json(url: str, timeout: int = 30) -> Any:
    response = requests.get(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        },
        timeout=timeout,
    )
    response.raise_for_status()
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
