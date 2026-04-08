from __future__ import annotations

import html
import json
import re
from datetime import datetime

from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem
from arxiv_assistant.utils.hotspot.hotspot_sources import clip_text, fetch_text, is_fresh

HF_TRENDING_URL = "https://huggingface.co/papers/trending"
HF_DATE_URL = "https://huggingface.co/papers?date={date}"
DAILY_PAPERS_PATTERN = re.compile(
    r'data-target="DailyPapers"[^>]*data-props="(?P<payload>[^"]+)"',
    re.DOTALL,
)


def _parse_daily_papers(page_html: str) -> list[dict]:
    match = DAILY_PAPERS_PATTERN.search(page_html)
    if match is None:
        return []
    payload = json.loads(html.unescape(match.group("payload")))
    return payload.get("dailyPapers", [])


MIN_UPVOTES = 5
MAX_PAPER_AGE_HOURS = 30 * 24


def fetch_hotspot_items(
    target_date: datetime,
    freshness_hours: int,
    result_limit: int = 12,
    *,
    daily_hot_score_cutoff: int = 15,
) -> list[HotspotItem]:
    # Use date-specific URL to fetch papers that were trending on target_date
    date_str = target_date.strftime("%Y-%m-%d")
    url = HF_DATE_URL.format(date=date_str)
    try:
        page_html = fetch_text(url)
    except Exception:
        # Fall back to trending page if date-specific fetch fails
        page_html = fetch_text(HF_TRENDING_URL)
    daily_papers = _parse_daily_papers(page_html)
    items: list[HotspotItem] = []
    skipped_upvotes = 0

    for row in daily_papers:
        paper = row.get("paper", {})
        paper_id = paper.get("id")
        if not paper_id:
            continue
        published_at = paper.get("publishedAt")
        if not published_at:
            continue  # reject papers without a real publication date
        upvotes = int(paper.get("upvotes", 0) or 0)

        # Upvote filter: skip low-engagement papers
        if upvotes < MIN_UPVOTES:
            skipped_upvotes += 1
            continue

        canonical_url = f"https://arxiv.org/abs/{paper_id}"

        # Classify for paper spotlight based on upvotes
        spotlight_kind = ""
        spotlight_label = ""
        spotlight_comment = ""
        if upvotes >= daily_hot_score_cutoff:
            spotlight_kind = "daily_hot"
            spotlight_label = "Daily Hot Papers"
            spotlight_comment = clip_text(paper.get("summary", ""), 300)

        item = HotspotItem(
            source_id="hf_papers",
            source_name="Hugging Face Trending Papers",
            source_role="paper_trending",
            source_type="paper",
            title=paper.get("title", paper_id),
            summary=clip_text(paper.get("summary", ""), 600),
            url=f"https://huggingface.co/papers/{paper_id}",
            canonical_url=canonical_url,
            published_at=published_at,
            tags=list(paper.get("ai_keywords") or []),
            authors=[author.get("name", "") for author in paper.get("authors", []) if author.get("name")],
            metadata={
                "arxiv_id": paper_id,
                "upvotes": upvotes,
                "daily_score": upvotes,
                "relevance": 0,
                "novelty": 0,
                "github_url": paper.get("githubRepo"),
                "hf_url": f"https://huggingface.co/papers/{paper_id}",
                "ai_summary": paper.get("ai_summary", ""),
                "spotlight_primary_kind": spotlight_kind,
                "spotlight_primary_label": spotlight_label,
                "spotlight_comment": spotlight_comment,
            },
        )
        if not is_fresh(item.published_at, target_date, MAX_PAPER_AGE_HOURS):
            continue
        items.append(item)
        if len(items) >= result_limit:
            break

    if skipped_upvotes > 0:
        print(f"HF Papers: kept {len(items)}, skipped {skipped_upvotes} low-upvotes (<{MIN_UPVOTES})")
    return items
