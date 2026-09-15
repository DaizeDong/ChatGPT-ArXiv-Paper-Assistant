from __future__ import annotations

from datetime import datetime
from typing import Any

from arxiv_assistant.apis.hotspot.hotspot_ainews import AINEWS_RSS_URL
from arxiv_assistant.apis.hotspot.hotspot_common import strip_html
from arxiv_assistant.apis.hotspot.reuse_common import harvest_rss_reuse
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem

REUSE_NAME = "ainews"


def _summary(entry: Any) -> str:
    parts = entry.get("content", [])
    raw = parts[0].get("value", "") if parts else entry.get("summary", "") or entry.get("description", "")
    return strip_html(raw)


def fetch_hotspot_items(
    target_date: datetime,
    freshness_hours: int,
    result_limit: int = 24,
) -> list[HotspotItem]:
    """Reuse AINews recap issues as competitor output. provenance='reuse:ainews'.

    AINews publishes weekdays only; widen the window like the native adapter.
    """
    effective = max(freshness_hours, 36)
    return harvest_rss_reuse(
        REUSE_NAME,
        AINEWS_RSS_URL,
        target_date,
        effective,
        result_limit=result_limit,
        summary_of=_summary,
    )
