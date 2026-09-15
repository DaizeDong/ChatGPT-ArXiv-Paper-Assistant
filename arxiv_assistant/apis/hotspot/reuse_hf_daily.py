from __future__ import annotations

from datetime import datetime

from arxiv_assistant.apis.hotspot.hotspot_hf_papers import HF_DATE_URL, HF_TRENDING_URL, _parse_daily_papers
from arxiv_assistant.apis.hotspot.reuse_common import build_reuse_item
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem
from arxiv_assistant.utils.hotspot.hotspot_sources import fetch_text

REUSE_NAME = "hf_daily"


def fetch_hotspot_items(
    target_date: datetime,
    freshness_hours: int,
    result_limit: int = 30,
) -> list[HotspotItem]:
    """Reuse HF Daily Papers as a first-class competitor-output source.

    No upvote cutoff here (recall-first); the shared DateVerify + max_age gate
    decides staleness downstream. provenance='reuse:hf_daily'.
    """
    date_str = target_date.strftime("%Y-%m-%d")
    try:
        page_html = fetch_text(HF_DATE_URL.format(date=date_str))
    except Exception:
        try:
            page_html = fetch_text(HF_TRENDING_URL)
        except Exception as ex:
            print(f"Warning: reuse:hf_daily fetch failed: {ex}")
            return []
    items: list[HotspotItem] = []
    for row in _parse_daily_papers(page_html):
        paper = row.get("paper", {})
        paper_id = paper.get("id")
        if not paper_id:
            continue
        published_at = paper.get("publishedAt")  # platform date; NOT trusted as first_date
        items.append(
            build_reuse_item(
                REUSE_NAME,
                title=paper.get("title", paper_id),
                url=f"https://huggingface.co/papers/{paper_id}",
                summary=paper.get("summary", ""),
                published_at=published_at,
                canonical_url=f"https://arxiv.org/abs/{paper_id}",
                tags=list(paper.get("ai_keywords") or []),
                authors=[a.get("name", "") for a in paper.get("authors", []) if a.get("name")],
                extra_metadata={"arxiv_id": paper_id, "upvotes": int(paper.get("upvotes", 0) or 0)},
            )
        )
        if len(items) >= result_limit:
            break
    return items
