from __future__ import annotations

import re
from datetime import datetime
from pathlib import PurePosixPath
from urllib.parse import urlsplit

import feedparser

from arxiv_assistant.apis.hotspot.hotspot_common import extract_anchor_pairs, split_html_segments, strip_html
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem, clean_text
from arxiv_assistant.utils.hotspot.hotspot_sources import clip_text, fetch_text, is_fresh, normalize_url

AINEWS_RSS_URL = "https://news.smol.ai/rss.xml"
INTERNAL_AINEWS_DOMAINS = {
    "news.smol.ai",
    "latent.space",
    "www.latent.space",
    "support.substack.com",
    "x.com",
    "twitter.com",
    "mobile.twitter.com",
}
MEDIA_DOMAINS = {"i.redd.it", "v.redd.it", "pbs.twimg.com", "video.twimg.com"}
MEDIA_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".mp4", ".mov", ".avi"}
GENERIC_TITLES = {"view image", "github repository", "learn more", "read more", "website", "link", "source"}
LOW_SIGNAL_TITLE_PATTERNS = (
    "ainews is now a section",
    "not much happened today",
)
LOW_SIGNAL_DISCUSSION_PATTERNS = (
    "help me",
    "what plugins",
    "worth it",
    "working dog",
    "drama just dropped",
    "chatgtp is legit helping me cook",
    "for local agentic ai",
    "accurate prompts for any ai tool",
    "life-threatened dog",
    "cancerous tumor",
    "personalized mrna vaccine",
    "security system",
    "home-security-benchmark",
    "powerbook g4",
)
HIGH_SIGNAL_HOST_KEYWORDS = (
    "github.com",
    "arxiv.org",
    "huggingface.co",
    "openai.com",
    "anthropic.com",
    "blog.google",
    "deepmind.google",
    "ai.meta.com",
)
HIGH_SIGNAL_TITLE_KEYWORDS = {
    "openai",
    "anthropic",
    "claude",
    "gpt",
    "gemini",
    "deepseek",
    "qwen",
    "llama",
    "mistral",
    "model",
    "release",
    "launch",
    "paper",
    "reasoning",
    "agent",
    "benchmark",
    "github",
    "open-source",
    "fine-tuning",
}
ACTIVITY_PATTERN = re.compile(r"\(Activity:\s*(?P<count>[\d,]+)\)", re.I)


def _is_external_link(url: str) -> bool:
    domain = urlsplit(url).netloc.lower()
    return bool(domain) and domain not in INTERNAL_AINEWS_DOMAINS


def _is_media_url(url: str) -> bool:
    parsed = urlsplit(url)
    if parsed.netloc.lower() in MEDIA_DOMAINS:
        return True
    suffix = PurePosixPath(parsed.path).suffix.lower()
    return suffix in MEDIA_EXTENSIONS


def _extract_activity(segment_text: str) -> int:
    match = ACTIVITY_PATTERN.search(segment_text)
    if match is None:
        return 0
    return int(match.group("count").replace(",", ""))


def _derive_segment_title(segment_text: str) -> str:
    title = ACTIVITY_PATTERN.split(segment_text, maxsplit=1)[0]
    title = clean_text(title.strip(" :-|"))
    if title.startswith("- "):
        title = clean_text(title[2:])
    if len(title) > 180 and ". " in title:
        title = title.split(". ", 1)[0].strip()
    return title


def _anchor_score(url: str, anchor_text: str) -> tuple[int, int]:
    host = urlsplit(url).netloc.lower()
    text = clean_text(anchor_text).lower()
    score = 0
    if not _is_external_link(url) or _is_media_url(url):
        return (-100, 0)
    if "github.com" in host or "arxiv.org" in host or "huggingface.co" in host:
        score += 6
    elif "reddit.com" in host or "news.ycombinator.com" in host:
        score += 4
    else:
        score += 5
    if text and text not in GENERIC_TITLES:
        score += 2
    if "github repository" in text:
        score += 1
    return (score, -len(text))


def _choose_best_anchor(segment_html: str) -> str | None:
    anchors = extract_anchor_pairs(segment_html)
    if not anchors:
        return None
    best_url: str | None = None
    best_score = (-100, 0)
    for url, anchor_text in anchors:
        score = _anchor_score(url, anchor_text)
        if score > best_score:
            best_score = score
            best_url = normalize_url(url)
    return best_url if best_score[0] > -100 else None


def _is_low_signal_title(title: str) -> bool:
    lowered = clean_text(title).lower()
    if not lowered or len(lowered) < 12:
        return True
    if lowered in GENERIC_TITLES:
        return True
    return any(pattern in lowered for pattern in LOW_SIGNAL_TITLE_PATTERNS)


def _is_high_signal_item(title: str, url: str, activity: int) -> bool:
    host = urlsplit(url).netloc.lower()
    lowered = clean_text(title).lower()
    if any(pattern in lowered for pattern in LOW_SIGNAL_DISCUSSION_PATTERNS):
        return False
    if any(keyword in host for keyword in HIGH_SIGNAL_HOST_KEYWORDS):
        return True
    keyword_match = any(keyword in lowered for keyword in HIGH_SIGNAL_TITLE_KEYWORDS)
    if "reddit.com" in host:
        if activity < 150:
            return False
        if "?" in title and not keyword_match:
            return False
        return keyword_match
    return activity >= 200 and keyword_match


def fetch_hotspot_items(target_date: datetime, freshness_hours: int) -> list[HotspotItem]:
    feed = feedparser.parse(fetch_text(AINEWS_RSS_URL))
    items: list[HotspotItem] = []
    seen: set[tuple[str, str]] = set()

    for entry in feed.entries:
        published_at = entry.get("published") or entry.get("updated")
        if not is_fresh(published_at, target_date, freshness_hours):
            continue

        issue_title = clean_text(entry.get("title", "AINews issue"))
        issue_url = clean_text(entry.get("link", AINEWS_RSS_URL))
        issue_tags = [clean_text(tag.get("term", "")) for tag in entry.get("tags", []) if clean_text(tag.get("term", ""))]
        content_parts = entry.get("content", [])
        content_html = content_parts[0].get("value", "") if content_parts else entry.get("description", "")

        for segment_html in split_html_segments(content_html):
            segment_text = strip_html(segment_html)
            title = _derive_segment_title(segment_text)
            if _is_low_signal_title(title):
                continue
            url = _choose_best_anchor(segment_html)
            if not url:
                continue
            activity = _extract_activity(segment_text)
            if not _is_high_signal_item(title, url, activity):
                continue
            key = (title.lower(), url)
            if key in seen:
                continue
            seen.add(key)
            items.append(
                HotspotItem(
                    source_id="ainews",
                    source_name="AINews",
                    source_role="community_heat",
                    source_type="roundup",
                    title=title,
                    summary=clip_text(segment_text, 520),
                    url=url,
                    canonical_url=url,
                    published_at=published_at,
                    tags=["community-roundup", *issue_tags],
                    authors=[],
                    metadata={
                        "feed_source": AINEWS_RSS_URL,
                        "issue_title": issue_title,
                        "issue_url": issue_url,
                        "activity": activity,
                        "host": urlsplit(url).netloc.lower(),
                    },
                )
            )
            if len(items) >= 24:
                return items
    return items
