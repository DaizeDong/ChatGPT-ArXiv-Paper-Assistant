from __future__ import annotations

from datetime import datetime

from arxiv_assistant.apis.hotspot.reuse_common import harvest_rss_reuse
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem

REUSE_NAME = "scholar_inbox"
# Scholar Inbox exposes per-user paper-recommendation feeds. The public digest feed
# is configured per account; FEED_URL is the single integration point.
# Real feed shape: https://www.scholar-inbox.com/api/feeds/digest.rss?token=<TOKEN>
# (token injected from env at deploy; default uses the public trending digest)
# verify feed URL when running
FEED_URL = "https://www.scholar-inbox.com/digest.rss"


def fetch_hotspot_items(
    target_date: datetime,
    freshness_hours: int,
    result_limit: int = 24,
) -> list[HotspotItem]:
    """Reuse Scholar Inbox digest. provenance='reuse:scholar_inbox'."""
    return harvest_rss_reuse(REUSE_NAME, FEED_URL, target_date, freshness_hours, result_limit=result_limit)
