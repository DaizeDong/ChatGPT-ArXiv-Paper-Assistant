from __future__ import annotations

from datetime import datetime

from arxiv_assistant.apis.hotspot.reuse_common import harvest_rss_reuse
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem

REUSE_NAME = "agents_radar"
# agents-radar publishes a daily AI-agents digest feed.
# Real feed (Atom): https://www.agents-radar.com/feed.xml  (update here if the site moves its feed path)
FEED_URL = "https://www.agents-radar.com/feed.xml"


def fetch_hotspot_items(
    target_date: datetime,
    freshness_hours: int,
    result_limit: int = 24,
) -> list[HotspotItem]:
    """Reuse agents-radar daily digest. provenance='reuse:agents_radar'."""
    return harvest_rss_reuse(REUSE_NAME, FEED_URL, target_date, freshness_hours, result_limit=result_limit)
