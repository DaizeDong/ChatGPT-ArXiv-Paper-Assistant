from __future__ import annotations

from datetime import UTC, date

from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem
from arxiv_assistant.utils.hotspot.hotspot_sources import parse_datetime


def floor_to_utc_day(iso_ts: str | None) -> date | None:
    """Truncate any timestamp to its UTC calendar day (drops H:M:S).

    Returns None for None/empty/unparseable input. Naive timestamps are
    assumed UTC; offset-aware timestamps are converted to UTC before flooring.
    This is the day-granular floor that makes sub-day WebSearch jitter unable
    to flip a discrete gate (spec §B.5.1, INV2).
    """
    if not iso_ts:
        return None
    dt = parse_datetime(iso_ts)
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).date()


_ANCHOR_KEYS = ("arxiv_announced_date", "crossref_registered_date")


def gate_date(item: HotspotItem) -> date | None:
    """Day-granular gate date = floor_to_utc_day(min(credible_dates(item))).

    credible_dates = {verified_first_date} ∪ {authoritative whole-day anchors:
    arXiv announced day / Crossref registration day}. Earliest-credible-date-wins
    (spec §B.3): pollution only back-dates forward to look fresh, so min beats it.
    Source-claimed published_at is NEVER credible (INV1). Returns None when no
    credible date exists (gate treats None as cannot-verify → do not drop).

    Metadata anchor keys (spec §2.4 / §B.3.1):
      - "arxiv_announced_date"   — arXiv announced day (whole-day, authoritative)
      - "crossref_registered_date" — Crossref registration day (whole-day, authoritative)
    """
    credible: list[date] = []

    floored = floor_to_utc_day(item.verified_first_date)
    if floored is not None:
        credible.append(floored)

    metadata = item.metadata or {}
    for key in _ANCHOR_KEYS:
        anchor = floor_to_utc_day(metadata.get(key))
        if anchor is not None:
            credible.append(anchor)

    if not credible:
        return None
    return min(credible)
