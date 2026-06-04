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


# Stage 1 anchor keys (whole-day authoritative stamps; spec §2.4 / §B.3.1).
# The "_date" variants were established in Stage 1; the "_day" variants are the
# kernel-stamped keys added in Stage 3 (§B.3.1 addendum).  Both are accepted so
# the gate is forward-compatible without a data migration.
_ANCHOR_KEYS = (
    "arxiv_announced_date",    # Stage 1
    "arxiv_announced_day",     # Stage 3 kernel-stamped
    "crossref_registered_date",  # Stage 1
    "crossref_registered_day",   # Stage 3 kernel-stamped
)


def credible_dates(item: HotspotItem) -> list[str]:
    """All machine-independent credible dates for an item, as ISO strings.

    §B.3.1: authoritative whole-day anchors (arXiv announced day / Crossref
    registration day) join {verified_first_date}. Anchors are kernel-stamped
    into item.metadata during Tier-0 so this function performs no network I/O.
    """
    dates: list[str] = []
    verified = getattr(item, "verified_first_date", None)
    if verified:
        dates.append(verified)
    meta = item.metadata or {}
    for key in _ANCHOR_KEYS:
        val = meta.get(key)
        if val:
            # Whole-day anchor: normalise to start-of-day ISO so floor_to_utc_day is a no-op.
            dates.append(f"{val}T00:00:00Z" if "T" not in str(val) else val)
    return dates


def gate_date(item: HotspotItem) -> date | None:
    """Day-granular gate date = floor_to_utc_day(min(credible_dates(item))).

    credible_dates = {verified_first_date} ∪ {authoritative whole-day anchors:
    arXiv announced day / Crossref registration day}. Earliest-credible-date-wins
    (spec §B.3): pollution only back-dates forward to look fresh, so min beats it.
    Source-claimed published_at is NEVER credible (INV1). Returns None when no
    credible date exists (gate treats None as cannot-verify → do not drop).

    Metadata anchor keys (spec §2.4 / §B.3.1):
      - "arxiv_announced_date" / "arxiv_announced_day"     — arXiv announced day
      - "crossref_registered_date" / "crossref_registered_day" — Crossref registration day
    """
    candidates = [floor_to_utc_day(d) for d in credible_dates(item)]
    candidates = [d for d in candidates if d is not None]
    if not candidates:
        return None
    return min(candidates)
