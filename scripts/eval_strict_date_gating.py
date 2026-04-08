"""Simulate strict date gating on raw cached source data.

Compares OLD filtering rules (36h window, wider source overrides) vs
NEW strict rules (24h window, tighter overrides) across 3 dimensions:
  1. Date overlap between consecutive days (URL overlap %)
  2. Information richness (unique sources, source diversity)
  3. Content volume (items surviving each filter)

Usage:
    python scripts/eval_strict_date_gating.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "out" / "hot" / "raw"

UTC = timezone.utc

# ---------------------------------------------------------------------------
# Date parsing (mirrors hotspot_sources.parse_datetime)
# ---------------------------------------------------------------------------

_COMMON_FORMATS = [
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
]

_RFC2822_FORMATS = [
    "%a, %d %b %Y %H:%M:%S %z",
    "%a, %d %b %Y %H:%M:%S %Z",
    "%d %b %Y %H:%M:%S %z",
]


def parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    s = s.strip()
    for fmt in _COMMON_FORMATS + _RFC2822_FORMATS:
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Filtering rules
# ---------------------------------------------------------------------------

# Source-specific freshness overrides
OLD_OVERRIDES = {
    "ainews": 48,
    "x_ainews_twitter": 48,
    "analysis_feeds": 72,
    "reddit": 36,  # was using freshness_hours directly (36)
}
MOD_OVERRIDES = {
    "ainews": 36,
    "x_ainews_twitter": 36,
    "analysis_feeds": 48,
    "reddit": 30,
}
NEW_OVERRIDES = {
    "ainews": 36,
    "x_ainews_twitter": 36,
    "analysis_feeds": 48,
    "reddit": 24,  # uses freshness_hours directly (now 24)
}

# Sources where fetched_at overrides published_at
FETCHED_AT_SOURCES = {"github_trend"}


def get_freshness_date(item: dict) -> str | None:
    """Mirror the pipeline's get_freshness_date logic."""
    sid = item.get("source_id", "")
    if sid in FETCHED_AT_SOURCES:
        fa = (item.get("metadata") or {}).get("fetched_at")
        if fa:
            return fa
    return item.get("published_at")


def is_fresh_old(published_at: str | None, target: datetime, hours: int) -> bool:
    """OLD rule: symmetric window (target - hours) to (target + hours)."""
    if not published_at:
        return True
    dt = parse_dt(published_at)
    if dt is None:
        return True
    # Before the asymmetric fix, the window was symmetric
    start = target - timedelta(hours=hours)
    end = target + timedelta(hours=hours)
    return start <= dt <= end


def is_fresh_new(published_at: str | None, target: datetime, hours: int) -> bool:
    """NEW rule: asymmetric window (target - hours) to (target + 6h)."""
    if not published_at:
        return True
    dt = parse_dt(published_at)
    if dt is None:
        return True
    start = target - timedelta(hours=hours)
    end = target + timedelta(hours=6)
    return start <= dt <= end


def filter_items(
    items: list[dict],
    target_date: datetime,
    base_freshness: int,
    overrides: dict[str, int],
    is_fresh_fn,
    reject_no_date: bool = False,
    max_age_days: int = 14,
) -> list[dict]:
    """Apply pipeline-level filtering to raw items."""
    target = target_date.replace(tzinfo=UTC) if target_date.tzinfo is None else target_date
    result = []

    for item in items:
        sid = item.get("source_id", "")

        # Layer 1: reject items without published_at (if strict mode)
        if reject_no_date and not item.get("published_at"):
            continue

        # Source-level freshness (what the adapter would do)
        effective_hours = overrides.get(sid, base_freshness)
        pa = item.get("published_at")

        # HF papers: in OLD mode, no is_fresh at source level; in NEW mode, 30-day cap
        if sid == "hf_papers":
            if reject_no_date:  # NEW mode
                if not pa:
                    continue
                dt = parse_dt(pa)
                if dt and (target - dt).total_seconds() / 3600 > 30 * 24:
                    continue
            # OLD mode: no source-level filtering for HF papers
        elif sid == "github_trend":
            pass  # GitHub uses fetched_at, no source-level freshness check
        else:
            if not is_fresh_fn(pa, target, effective_hours):
                continue

        # Pipeline-level freshness gate (using get_freshness_date)
        fd = get_freshness_date(item)
        fd_dt = parse_dt(fd)
        pipeline_cutoff = target - timedelta(hours=base_freshness)
        if fd_dt is not None and fd_dt < pipeline_cutoff:
            continue

        # Hard cap on published_at
        if pa:
            pub_dt = parse_dt(pa)
            if pub_dt is not None:
                age_cutoff = target - timedelta(days=max_age_days)
                if pub_dt < age_cutoff:
                    continue

        result.append(item)
    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def extract_urls(items: list[dict]) -> set[str]:
    urls = set()
    for item in items:
        u = item.get("canonical_url") or item.get("url") or ""
        if u:
            urls.add(u)
    return urls


def source_diversity(items: list[dict]) -> dict[str, int]:
    return Counter(item.get("source_id", "unknown") for item in items)


def url_overlap(urls_a: set[str], urls_b: set[str]) -> float:
    if not urls_a or not urls_b:
        return 0.0
    shared = urls_a & urls_b
    union = urls_a | urls_b
    return len(shared) / len(union) * 100 if union else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_raw_items(date_str: str) -> list[dict]:
    """Load all raw cached items for a given date."""
    day_dir = RAW_DIR / date_str
    if not day_dir.exists():
        return []
    items = []
    for fp in sorted(day_dir.glob("*.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            if isinstance(data, list):
                items.extend(data)
        except Exception:
            pass
    return items


SCENARIOS = {
    "OLD(36h-sym)": {
        "base": 36, "overrides": OLD_OVERRIDES,
        "is_fresh_fn": is_fresh_old, "reject_no_date": False,
    },
    "MOD(30h-asym)": {
        "base": 30, "overrides": MOD_OVERRIDES,
        "is_fresh_fn": is_fresh_new, "reject_no_date": True,
    },
    "NEW(24h-asym)": {
        "base": 24, "overrides": NEW_OVERRIDES,
        "is_fresh_fn": is_fresh_new, "reject_no_date": True,
    },
}


def main():
    dates = sorted(
        d.name for d in RAW_DIR.iterdir()
        if d.is_dir() and d.name.startswith("202") and d.name >= "2026-"
    )
    if not dates:
        print("No raw data found!")
        return

    print(f"Raw data directory: {RAW_DIR}")
    print(f"Dates found: {len(dates)}")
    print()

    # Collect per-day metrics under each scenario
    all_metrics: dict[str, dict] = {name: {} for name in SCENARIOS}

    for date_str in dates:
        target = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC).replace(hour=12)
        raw = load_raw_items(date_str)

        for name, cfg in SCENARIOS.items():
            filtered = filter_items(
                raw, target,
                base_freshness=cfg["base"],
                overrides=cfg["overrides"],
                is_fresh_fn=cfg["is_fresh_fn"],
                reject_no_date=cfg["reject_no_date"],
                max_age_days=14,
            )
            all_metrics[name][date_str] = {
                "raw": len(raw),
                "filtered": len(filtered),
                "urls": extract_urls(filtered),
                "sources": source_diversity(filtered),
                "n_sources": len(source_diversity(filtered)),
            }

    sc_names = list(SCENARIOS.keys())
    W = 120

    # ---- DIMENSION 3: CONTENT VOLUME ----
    print("=" * W)
    print("DIMENSION 3: CONTENT VOLUME (items surviving filtering)")
    print("=" * W)
    hdr = f"{'Date':<14} {'Raw':>6}"
    for n in sc_names:
        hdr += f" {n:>15}"
    print(hdr)
    print("-" * W)
    totals = {n: 0 for n in sc_names}
    total_raw = 0
    for d in dates:
        raw_cnt = all_metrics[sc_names[0]][d]["raw"]
        total_raw += raw_cnt
        line = f"{d:<14} {raw_cnt:>6}"
        for n in sc_names:
            cnt = all_metrics[n][d]["filtered"]
            totals[n] += cnt
            line += f" {cnt:>15}"
        print(line)
    print("-" * W)
    line = f"{'AVERAGE':<14} {total_raw/len(dates):>6.0f}"
    for n in sc_names:
        line += f" {totals[n]/len(dates):>15.1f}"
    print(line)
    print()

    # ---- DIMENSION 2: INFORMATION RICHNESS ----
    print("=" * W)
    print("DIMENSION 2: INFORMATION RICHNESS (unique sources per day)")
    print("=" * W)
    hdr = f"{'Date':<14}"
    for n in sc_names:
        hdr += f" {n:>15}"
    print(hdr)
    print("-" * W)
    src_totals = {n: 0 for n in sc_names}
    for d in dates:
        line = f"{d:<14}"
        for n in sc_names:
            ns = all_metrics[n][d]["n_sources"]
            src_totals[n] += ns
            line += f" {ns:>15}"
        print(line)
    print("-" * W)
    line = f"{'AVERAGE':<14}"
    for n in sc_names:
        line += f" {src_totals[n]/len(dates):>15.1f}"
    print(line)
    print()

    # ---- PER-SOURCE VOLUME ----
    print("=" * W)
    print("PER-SOURCE VOLUME COMPARISON")
    print("=" * W)
    all_sources = set()
    for n in sc_names:
        for d in dates:
            all_sources.update(all_metrics[n][d]["sources"].keys())
    all_sources = sorted(all_sources)
    hdr = f"{'Source':<25}"
    for n in sc_names:
        hdr += f" {n:>15}"
    print(hdr)
    print("-" * W)
    for src in all_sources:
        line = f"{src:<25}"
        for n in sc_names:
            total = sum(all_metrics[n][d]["sources"].get(src, 0) for d in dates)
            line += f" {total:>15}"
        print(line)
    print()

    # ---- DIMENSION 1: DATE OVERLAP ----
    print("=" * W)
    print("DIMENSION 1: DATE OVERLAP (URL overlap % between consecutive days)")
    print("=" * W)
    hdr = f"{'Pair':<32}"
    for n in sc_names:
        hdr += f" {n:>15}"
    print(hdr)
    print("-" * W)
    overlap_sums = {n: [] for n in sc_names}
    for i in range(len(dates) - 1):
        d1, d2 = dates[i], dates[i + 1]
        line = f"{d1} / {d2:<14}"
        for n in sc_names:
            ov = url_overlap(all_metrics[n][d1]["urls"], all_metrics[n][d2]["urls"])
            overlap_sums[n].append(ov)
            line += f" {ov:>14.1f}%"
        print(line)
    print("-" * W)
    line = f"{'AVERAGE':<32}"
    for n in sc_names:
        avg = sum(overlap_sums[n]) / len(overlap_sums[n]) if overlap_sums[n] else 0
        line += f" {avg:>14.1f}%"
    print(line)
    print()

    # ---- GRAND SUMMARY ----
    print("=" * W)
    print("GRAND SUMMARY (3 scenarios compared)")
    print("=" * W)
    print(f"  {'Metric':<30}", end="")
    for n in sc_names:
        print(f" {n:>15}", end="")
    print()
    print(f"  {'-'*30}", end="")
    for _ in sc_names:
        print(f" {'-'*15}", end="")
    print()

    print(f"  {'Avg items/day':<30}", end="")
    for n in sc_names:
        print(f" {totals[n]/len(dates):>15.1f}", end="")
    print()

    print(f"  {'Avg sources/day':<30}", end="")
    for n in sc_names:
        print(f" {src_totals[n]/len(dates):>15.1f}", end="")
    print()

    print(f"  {'Avg URL overlap %':<30}", end="")
    for n in sc_names:
        avg = sum(overlap_sums[n]) / len(overlap_sums[n]) if overlap_sums[n] else 0
        print(f" {avg:>14.1f}%", end="")
    print()

    print(f"  {'Min items on any day':<30}", end="")
    for n in sc_names:
        mn = min(all_metrics[n][d]["filtered"] for d in dates)
        print(f" {mn:>15}", end="")
    print()
    print()


if __name__ == "__main__":
    main()
