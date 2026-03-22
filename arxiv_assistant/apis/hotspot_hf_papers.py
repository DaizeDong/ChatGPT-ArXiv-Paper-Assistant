from __future__ import annotations

import html
import json
import re
from datetime import datetime

from arxiv_assistant.utils.hotspot_schema import HotspotItem
from arxiv_assistant.utils.hotspot_sources import clip_text, fetch_text

HF_TRENDING_URL = "https://huggingface.co/papers/trending"
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


def fetch_hotspot_items(target_date: datetime, freshness_hours: int, result_limit: int = 24) -> list[HotspotItem]:
    page_html = fetch_text(HF_TRENDING_URL)
    daily_papers = _parse_daily_papers(page_html)
    items: list[HotspotItem] = []

    for row in daily_papers:
        paper = row.get("paper", {})
        paper_id = paper.get("id")
        if not paper_id:
            continue
        published_at = paper.get("publishedAt") or target_date.isoformat()
        canonical_url = f"https://arxiv.org/abs/{paper_id}"
        items.append(
            HotspotItem(
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
                    "upvotes": paper.get("upvotes", 0),
                    "github_url": paper.get("githubRepo"),
                    "hf_url": f"https://huggingface.co/papers/{paper_id}",
                    "ai_summary": paper.get("ai_summary", ""),
                },
            )
        )
        if len(items) >= result_limit:
            break
    return items
