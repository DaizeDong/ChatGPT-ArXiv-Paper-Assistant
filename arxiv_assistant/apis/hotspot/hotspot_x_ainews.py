from __future__ import annotations

from datetime import datetime
from urllib.parse import urlsplit

import feedparser
from bs4 import BeautifulSoup

from arxiv_assistant.apis.hotspot.hotspot_x_common import get_authority_record, is_authoritative_x_identity, is_newsworthy_x_text
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem, clean_text
from arxiv_assistant.utils.hotspot.hotspot_sources import clip_text, fetch_text, is_fresh, normalize_url
from arxiv_assistant.apis.hotspot.hotspot_common import extract_anchor_pairs, strip_html
from arxiv_assistant.apis.hotspot.hotspot_ainews import AINEWS_RSS_URL


def _extract_twitter_section_items(content_html: str, issue_title: str, issue_url: str, published_at: str | None) -> list[HotspotItem]:
    soup = BeautifulSoup(content_html or "", "html.parser")
    heading = None
    for node in soup.find_all(["h1", "h2", "h3"]):
        if clean_text(node.get_text(" ", strip=True)).lower() == "ai twitter recap":
            heading = node
            break
    if heading is None:
        return []

    items: list[HotspotItem] = []
    seen_urls: set[str] = set()
    current_subtopic = ""

    for sibling in heading.find_next_siblings():
        if sibling.name in {"h1", "h2"}:
            break
        if sibling.name == "p":
            strong = sibling.find("strong")
            current_subtopic = clean_text(strong.get_text(" ", strip=True) if strong else sibling.get_text(" ", strip=True))
            continue
        if sibling.name != "ul":
            continue

        for li in sibling.find_all("li", recursive=False):
            anchors = extract_anchor_pairs(str(li))
            x_urls = [
                normalize_url(url)
                for url, _ in anchors
                if any(host in urlsplit(url).netloc.lower() for host in ("x.com", "twitter.com", "mobile.twitter.com"))
            ]
            if not x_urls:
                continue
            authoritative_urls = []
            primary_record = None
            for x_url in x_urls:
                handle = clean_text(urlsplit(x_url).path.split("/", 2)[1] if len(urlsplit(x_url).path.split("/", 2)) > 1 else "")
                if is_authoritative_x_identity(handle):
                    authoritative_urls.append(x_url)
                    if primary_record is None:
                        primary_record = get_authority_record(handle)
            if not authoritative_urls:
                continue
            primary_url = authoritative_urls[0]
            if primary_url in seen_urls:
                continue
            seen_urls.add(primary_url)

            strong = li.find("strong")
            title = clean_text(strong.get_text(" ", strip=True) if strong else li.get_text(" ", strip=True).split(":", 1)[0])
            if not title and current_subtopic:
                title = current_subtopic
            text = clean_text(strip_html(str(li)))
            if not is_newsworthy_x_text(
                text,
                authority_kind=str((primary_record or {}).get("kind") or "researcher"),
                expanded_urls=authoritative_urls + [issue_url],
                activity=max(60, len(authoritative_urls) * 80),
            ):
                continue
            summary = clip_text(text, 520)
            x_handle_count = len({urlsplit(url).path for url in authoritative_urls})
            activity = max(60, x_handle_count * 80)

            items.append(
                HotspotItem(
                    source_id="ainews_twitter",
                    source_name="AINews AI Twitter Recap",
                    source_role="community_heat",
                    source_type="roundup",
                    title=title,
                    summary=summary,
                    url=primary_url,
                    canonical_url=primary_url,
                    published_at=published_at,
                    tags=["ai-twitter-recap"],
                    authors=[],
                    metadata={
                        "issue_title": issue_title,
                        "issue_url": issue_url,
                        "x_urls": authoritative_urls,
                        "activity": activity,
                        "source_quality": max(1.1, 1.0 + int((primary_record or {}).get("tier") or 1) * 0.08),
                        "signal_tier": "x_editorial_recap",
                        "host": "x.com",
                        "subtopic": current_subtopic,
                        "authority_kind": str((primary_record or {}).get("kind") or ""),
                        "authority_tier": int((primary_record or {}).get("tier") or 0),
                    },
                )
            )
    return items


def fetch_hotspot_items(target_date: datetime, freshness_hours: int) -> list[HotspotItem]:
    feed = feedparser.parse(fetch_text(AINEWS_RSS_URL))
    items: list[HotspotItem] = []
    for entry in feed.entries:
        published_at = entry.get("published") or entry.get("updated")
        if not is_fresh(published_at, target_date, freshness_hours):
            continue
        issue_title = clean_text(entry.get("title", "AINews issue"))
        issue_url = clean_text(entry.get("link", AINEWS_RSS_URL))
        content_parts = entry.get("content", [])
        content_html = content_parts[0].get("value", "") if content_parts else entry.get("description", "")
        items.extend(_extract_twitter_section_items(content_html, issue_title, issue_url, published_at))
        if items:
            break
    return items
