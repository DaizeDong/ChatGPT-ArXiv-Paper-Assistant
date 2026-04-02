from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem
from arxiv_assistant.utils.hotspot.hotspot_sources import clip_text, fetch_text, is_fresh, load_roundup_registry, normalize_url

LOW_SIGNAL_TITLE_PREFIXES = ("how to ", "watch:", "listen:", "tutorial:")
LOW_SIGNAL_TITLE_SNIPPETS = ("how to ", "also:", "plus:", "special:", "use ", "generate ", "build ")
DATE_PREFIX_PATTERN = re.compile(r"^(?:[A-Z][a-z]{2,8} \d{1,2}, \d{4}\s+)+")
EMOJI_PREFIX_PATTERN = re.compile(r"^[^\w\"']+")

# Paper URL patterns that should come from dedicated paper sources, not roundups
PAPER_URL_PATTERNS = ("arxiv.org", "huggingface.co/papers")

# Signal keywords for low-trust source filtering
SIGNAL_KEYWORDS = {
    "openai", "anthropic", "google", "deepmind", "meta", "nvidia", "microsoft",
    "claude", "gpt", "gemini", "llama", "mistral", "qwen", "deepseek",
    "model", "launch", "release", "benchmark", "training", "inference",
    "agent", "reasoning", "transformer", "multimodal", "api",
}

# Tier-aware per-site limits
LOW_TRUST_ROLES = {"headline_consensus"}
TRUSTED_ROLES = {"editorial_depth", "builder_momentum"}
LOW_TRUST_MAX_ITEMS = 2
TRUSTED_MAX_ITEMS = 5
MIN_TITLE_LENGTH = 30


def _normalize_roundup_title(title: str) -> str:
    normalized = " ".join((title or "").split())
    for marker in (" PLUS:", " ALSO:", " Two new eps", " | "):
        if marker in normalized:
            normalized = normalized.split(marker, 1)[0].strip()
    normalized = DATE_PREFIX_PATTERN.sub("", normalized)
    normalized = EMOJI_PREFIX_PATTERN.sub("", normalized).strip()
    if normalized.lower().startswith("ai "):
        normalized = normalized[3:].strip()
    if normalized.lower().startswith("tech "):
        normalized = normalized[5:].strip()
    return normalized


def _is_low_signal_title(title: str, url: str) -> bool:
    lowered = (title or "").strip().lower()
    if len(lowered) < MIN_TITLE_LENGTH:
        return True
    if any(lowered.startswith(prefix) for prefix in LOW_SIGNAL_TITLE_PREFIXES):
        return True
    if any(snippet in lowered for snippet in LOW_SIGNAL_TITLE_SNIPPETS):
        return True
    return "university." in url


def _is_paper_url(url: str) -> bool:
    return any(pattern in url.lower() for pattern in PAPER_URL_PATTERNS)


def _has_signal_keyword(title: str) -> bool:
    words = set(re.split(r"\W+", title.lower()))
    return bool(words & SIGNAL_KEYWORDS)


def _extract_generic_roundup_items(page_html: str, base_url: str) -> list[dict]:
    soup = BeautifulSoup(page_html, "html.parser")
    rows: list[dict] = []

    for article in soup.find_all("article"):
        anchor = article.find("a", href=True)
        title_node = article.find(["h1", "h2", "h3", "h4"])
        time_node = article.find("time")
        body_node = article.find("p")
        title = title_node.get_text(" ", strip=True) if title_node else anchor.get_text(" ", strip=True) if anchor else ""
        title = _normalize_roundup_title(title)
        href = anchor["href"] if anchor else ""
        if not title or not href:
            continue
        rows.append(
            {
                "title": title,
                "url": normalize_url(href, base_url),
                "summary": body_node.get_text(" ", strip=True) if body_node else "",
                "published_at": time_node.get("datetime") if time_node else None,
            }
        )

    if rows:
        return rows

    for anchor in soup.find_all("a", href=True):
        text = _normalize_roundup_title(anchor.get_text(" ", strip=True))
        href = anchor["href"]
        if len(text) < 28 or href.startswith("#"):
            continue
        rows.append(
            {
                "title": text,
                "url": normalize_url(href, base_url),
                "summary": "",
                "published_at": None,
            }
        )
        if len(rows) >= 10:
            break
    return rows


def fetch_hotspot_items(target_date: datetime, freshness_hours: int, registry_path: str | Path) -> list[HotspotItem]:
    registry = load_roundup_registry(registry_path)
    items: list[HotspotItem] = []
    for site in registry:
        if not site.get("enabled") or site.get("site_id") in {"ainews", "alphasignal"}:
            continue
        if site.get("phase_priority") not in {"phase_1", "phase_2"}:
            continue
        try:
            page_html = fetch_text(site["url"])
            rows = _extract_generic_roundup_items(page_html, site["url"])
        except Exception as ex:
            print(f"Warning: failed to fetch roundup site {site['site_id']}: {ex}")
            continue
        signal_role = site.get("signal_role", "headline_consensus")
        is_low_trust = signal_role in LOW_TRUST_ROLES
        max_per_site = LOW_TRUST_MAX_ITEMS if is_low_trust else TRUSTED_MAX_ITEMS
        seen: set[str] = set()
        kept = 0
        for row in rows:
            url = row["url"]
            if url in seen:
                continue
            seen.add(url)

            # Skip paper URLs in roundups (should come from dedicated sources)
            if _is_paper_url(url):
                continue

            title = _normalize_roundup_title(row["title"])
            if _is_low_signal_title(title, url):
                continue

            # For low-trust sources, require at least one signal keyword
            if is_low_trust and not _has_signal_keyword(title):
                continue

            published_at = row.get("published_at")
            if published_at and not is_fresh(published_at, target_date, freshness_hours):
                continue
            items.append(
                HotspotItem(
                    source_id=site["site_id"],
                    source_name=site["name"],
                    source_role=signal_role,
                    source_type="roundup",
                    title=title,
                    summary=clip_text(row.get("summary", ""), 480),
                    url=url,
                    canonical_url=url,
                    published_at=published_at,
                    tags=[],
                    authors=[],
                    metadata={"roundup_site": site["name"]},
                )
            )
            kept += 1
            if kept >= max_per_site:
                break
    return items
