"""GapFill — eligible/dropped split + directed-fetch diff (Stage 4, spec §D.3/§D.4).

Three public functions with locked signatures:

    eligible_competitor_items(competitor_items, store, *, max_age_days, as_of)
        -> tuple[list[HotspotItem], list[HotspotItem]]

    gapfill(our_coverage: set[str], eligible: list[HotspotItem]) -> list[HotspotItem]

    assert_union_floor(our_coverage: set[str], eligible: list[HotspotItem]) -> None

Design notes
------------
- DateVerify is the SOLE cross-validator (§D.4): multi-competitor consensus on a
  backdated-old paper cannot override the arXiv/Wayback hard anchor.  There is no
  majority-vote path; the eligibility gate IS the cross-validation.
- The ⊇ obligation (assert_union_floor) is scoped ONLY to `eligible` — items that
  pass both DateVerify AND within_max_age.  Items gated into `dropped_stale` carry
  no ⊇ obligation (§D.3 self-contradiction fix).
"""
from __future__ import annotations

import importlib
import statistics
from datetime import date, datetime

from arxiv_assistant.hotspots import date_verify
from arxiv_assistant.utils.hotspot import run_journal
from arxiv_assistant.utils.hotspot.gate_date import gate_date
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem


def _key(item: HotspotItem) -> str:
    """Coverage identity = canonical_url (already normalized by HotspotItem.__post_init__)."""
    return item.canonical_url or item.url


def _within_max_age(gd: date | None, *, max_age_days: int, as_of: date) -> bool:
    """day-granular gate (INV2): item is within max_age iff gate_date >= as_of - max_age_days."""
    if gd is None:
        return False
    return (as_of - gd).days <= max_age_days


def _apply_verdict(item: HotspotItem, store) -> HotspotItem:
    """Run DateVerify and stamp verified_first_date so gate_date(item) is authoritative."""
    verdict = date_verify.verify(item, store)
    item.verified_first_date = verdict.get("verified_first_date")
    return item


def eligible_competitor_items(
    competitor_items: list[HotspotItem],
    store,
    *,
    max_age_days: int,
    as_of: date,
) -> tuple[list, list]:
    """Split competitor items into (eligible, dropped_stale) per spec §D.3.

    eligible  := passes DateVerify AND within_max_age(gate_date)
    dropped   := everything else (legitimately gated; NOT a gate failure)

    DateVerify is the sole cross-validator (§D.4): multi-competitor consensus on a
    backdated-old paper cannot override the arXiv/Wayback hard anchor.
    """
    eligible: list[HotspotItem] = []
    dropped: list[HotspotItem] = []
    for raw in competitor_items:
        item = _apply_verdict(raw, store)
        gd = gate_date(item)
        if gd is not None and _within_max_age(gd, max_age_days=max_age_days, as_of=as_of):
            eligible.append(item)
        else:
            dropped.append(item)
    return eligible, dropped


def assert_union_floor(our_coverage: set, eligible: list) -> None:
    """Acceptance: our_coverage ⊇ eligible_competitor_items (spec §D.3, scoped).

    Only eligible (verified + within max_age) competitor items carry a ⊇ obligation.
    Items dropped by the hard gate are intentionally excluded.
    """
    eligible_keys = {_key(i) for i in eligible}
    missing = eligible_keys - set(our_coverage)
    if missing:
        raise AssertionError(
            "GapFill union-floor violated: our_coverage missing eligible competitor items: "
            + ", ".join(sorted(missing))
        )


def gapfill(our_coverage: set, eligible: list) -> list:
    """Return new items to ingest = eligible \\ our_coverage (the verifiable diff gate).

    These are 'someone has it, it passed OUR verification, we don't' — directed
    fetch + already-verified, ready to merge into our coverage.
    """
    seen: set[str] = set()
    new_items: list[HotspotItem] = []
    for item in eligible:
        k = _key(item)
        if k in our_coverage or k in seen:
            continue
        seen.add(k)
        new_items.append(item)
    return new_items


def second_order_pollution_alerts(
    today_record: dict,
    history: list[dict],
    *,
    multiplier: float = 2.0,
    abs_floor: float = 0.30,
    trailing: int = 14,
) -> list[dict]:
    """Per-source upstream-pollution alert (spec §E, decision 4).

    For each competitor source in today's intentionally_dropped_stale_competitor
    record: fire iff today's drop_ratio >= multiplier * trailing-median baseline
    AND today's drop_ratio >= abs_floor. Distinguishes 'normal steady curation of
    old items' (high but stable -> no alert) from 'this competitor suddenly dumped
    old content' (single-source spike -> alert). Pure read over journal records.
    Observability only — never affects pipeline decisions.
    """
    recent = [
        r for r in history
        if r.get("channel") == "intentionally_dropped_stale_competitor"
    ][-trailing:]
    today_per_source = today_record.get("per_source", {})
    alerts: list[dict] = []
    for src, stats in today_per_source.items():
        today_ratio = float(stats.get("drop_ratio", 0.0))
        baseline_samples = [
            float(r["per_source"][src]["drop_ratio"])
            for r in recent
            if src in r.get("per_source", {})
        ]
        baseline = statistics.median(baseline_samples) if baseline_samples else 0.0
        if today_ratio >= abs_floor and today_ratio >= multiplier * baseline:
            alerts.append({
                "source": src,
                "today_ratio": round(today_ratio, 4),
                "baseline_median": round(baseline, 4),
                "multiplier": multiplier,
                "abs_floor": abs_floor,
                "message": (
                    f"competitor source {src} suspected upstream pollution: "
                    f"drop_ratio {today_ratio:.2f} vs baseline {baseline:.2f}"
                ),
            })
    return alerts


# ---------------------------------------------------------------------------
# Stage-4 integration seams (Task 11) — callable by Stage-6 Kernel, NOT wired
# into pipeline.py / kernel.py here.
# ---------------------------------------------------------------------------

REUSE_ADAPTERS: dict[str, str] = {
    "hf_daily": "arxiv_assistant.apis.hotspot.reuse_hf_daily",
    "ainews": "arxiv_assistant.apis.hotspot.reuse_ainews",
    "agents_radar": "arxiv_assistant.apis.hotspot.reuse_agents_radar",
    "horizon": "arxiv_assistant.apis.hotspot.reuse_horizon",
    "scholar_inbox": "arxiv_assistant.apis.hotspot.reuse_scholar_inbox",
}


def harvest_reuse_layer(
    reuse_sources: list[str],
    target_date: datetime,
    freshness_hours: int,
) -> list[HotspotItem]:
    """Fan out enabled reuse adapters -> competitor_items (one schema, stamped provenance).

    Config-driven dispatch: only sources present in `reuse_sources` are harvested.
    Per-source fault-tolerant: one failing adapter degrades to [] without killing others (spec §E).
    """
    out: list[HotspotItem] = []
    for name in reuse_sources:
        mod_path = REUSE_ADAPTERS.get(name)
        if not mod_path:
            continue
        try:
            mod = importlib.import_module(mod_path)
            out.extend(mod.fetch_hotspot_items(target_date, freshness_hours))
        except Exception as ex:  # degrade per source (spec §E)
            print(f"Warning: reuse adapter {name} failed: {ex}")
    return out


def run_gapfill_floor(
    our_coverage: set,
    competitor_items: list[HotspotItem],
    store,
    *,
    max_age_days: int,
    as_of: date,
    run_date: str,
    journal_path=None,
) -> dict:
    """End-to-end Stage-4 seam: verify -> split -> gapfill -> assert floor -> journal -> alerts.

    competitor_items must be the FULL set (eligible + dropped) so per-source
    seen/dropped ratios in the journal record are correct (binding contract).

    Returns {"new_items", "eligible", "dropped", "alerts"}.
    The floor is asserted on (our_coverage ∪ gapfilled keys) so it is satisfied by
    exactly the directed fetch we are about to ingest.
    """
    eligible, dropped = eligible_competitor_items(
        competitor_items, store, max_age_days=max_age_days, as_of=as_of
    )
    new_items = gapfill(our_coverage, eligible)
    covered = set(our_coverage) | {_key(i) for i in new_items}
    assert_union_floor(covered, eligible)

    record = run_journal.record_dropped_stale_competitor(run_date, eligible, dropped, competitor_items)
    # Read PRIOR runs for the baseline BEFORE appending today's record — otherwise today's
    # drop_ratio enters its own baseline median and self-suppresses the spike alert.
    prior_history = run_journal.read_runs(journal_path=journal_path)
    alerts = second_order_pollution_alerts(record, prior_history)
    run_journal.append(run_date, record, journal_path=journal_path)
    return {"new_items": new_items, "eligible": eligible, "dropped": dropped, "alerts": alerts}
