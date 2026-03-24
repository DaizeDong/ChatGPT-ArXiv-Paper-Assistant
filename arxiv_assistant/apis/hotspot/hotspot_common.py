from datetime import UTC, date, datetime, timedelta
from html import unescape
import re
from typing import List, Tuple

import feedparser
import requests

from arxiv_assistant.utils.hotspot.hotspot_sources import record_api_usage
from arxiv_assistant.utils.hotspot.hotspot_schema import clean_text, normalize_url

DEFAULT_TIMEOUT = 20
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AIHotspotBot/1.0; +https://github.com/DaizeDong/ChatGPT-ArXiv-Paper-Assistant)",
}


def fetch_text(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    response = requests.get(url, timeout=timeout, headers=DEFAULT_HEADERS)
    response.raise_for_status()
    response.encoding = response.encoding or "utf-8"
    record_api_usage()
    return response.text


def parse_iso_or_rss_datetime(raw_value: str | None) -> str:
    if not raw_value:
        return datetime.now(UTC).isoformat()

    raw_value = raw_value.strip()
    if raw_value.endswith("Z"):
        raw_value = raw_value[:-1] + "+00:00"

    for fmt in (
        None,
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z",
        "%b %d, %Y",
        "%B %d, %Y",
        "%Y-%m-%d",
    ):
        try:
            if fmt is None:
                return datetime.fromisoformat(raw_value).astimezone(UTC).isoformat()
            parsed = datetime.strptime(raw_value, fmt)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC).isoformat()
        except ValueError:
            continue

    return datetime.now(UTC).isoformat()


def iso_within_days(iso_value: str, target_date: date, lookback_days: int) -> bool:
    parsed = datetime.fromisoformat(iso_value.replace("Z", "+00:00")).date()
    return target_date - timedelta(days=lookback_days) <= parsed <= target_date


def strip_html(html_text: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", html_text or "")
    return clean_text(unescape(without_tags))


def extract_anchor_pairs(html_text: str) -> List[Tuple[str, str]]:
    pairs = []
    for match in re.finditer(r"<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", html_text or "", re.I | re.S):
        href = normalize_url(unescape(match.group(1)))
        anchor_text = strip_html(match.group(2))
        if href:
            pairs.append((href, anchor_text))
    return pairs


def split_html_segments(html_text: str) -> List[str]:
    segments = re.split(r"</(?:p|li|h\d|blockquote|div|section)>", html_text or "", flags=re.I)
    return [segment for segment in segments if clean_text(strip_html(segment))]


def is_probably_generic_anchor(text: str) -> bool:
    lowered = clean_text(text).lower()
    return not lowered or lowered in {"here", "link", "read more", "source", "thread", "tweet", "post", "website"} or len(lowered) < 6


def parse_feed(url: str):
    return feedparser.parse(fetch_text(url))
