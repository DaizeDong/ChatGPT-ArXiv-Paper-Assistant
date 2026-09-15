from __future__ import annotations

from datetime import datetime

from arxiv_assistant.apis.hotspot.reuse_common import harvest_rss_reuse
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem

REUSE_NAME = "horizon"
# Horizon (The Batch / deeplearning.ai weekly AI roundup) RSS.
# Real feed: https://www.deeplearning.ai/the-batch/rss.xml  (update here if the path changes)
FEED_URL = "https://www.deeplearning.ai/the-batch/rss.xml"


def fetch_hotspot_items(
    target_date: datetime,
    freshness_hours: int,
    result_limit: int = 24,
) -> list[HotspotItem]:
    """Reuse Horizon / The Batch roundup. provenance='reuse:horizon'.

    Weekly cadence — widen window to >=8 days so a single weekly issue is in range.
    """
    effective = max(freshness_hours, 8 * 24)
    return harvest_rss_reuse(REUSE_NAME, FEED_URL, target_date, effective, result_limit=result_limit)
