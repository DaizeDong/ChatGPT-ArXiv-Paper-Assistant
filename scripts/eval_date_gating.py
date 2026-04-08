#!/usr/bin/env python
"""
Baseline metrics for hotspot web-data daily JSON files.

Collects per-day counts and inter-day overlap metrics, then prints a
summary table suitable for regression comparisons.
"""

import difflib
import glob
import json
import os
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "out" / "web_data" / "hot"
HEADLINE_SIM_THRESHOLD = 0.7


# ── helpers ──────────────────────────────────────────────────────────────

def _evidence_urls(topic: dict) -> list[str]:
    """Return all evidence URLs from a topic dict."""
    return [e["url"] for e in topic.get("evidence", []) if e.get("url")]


def _source_item_urls(section: dict) -> list[str]:
    """Return all item URLs from a source_section."""
    return [it["url"] for it in section.get("items", []) if it.get("url")]


def _all_headlines(topics: list[dict]) -> list[str]:
    """Collect headline strings from a list of topic dicts."""
    return [t["headline"] for t in topics if t.get("headline")]


# ── per-day extraction ───────────────────────────────────────────────────

def extract_day_metrics(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    date = data.get("meta", {}).get("date", os.path.basename(path).replace(".json", ""))

    # --- featured_topics ---
    featured = data.get("featured_topics", [])
    ft_count = len(featured)
    ft_urls = []
    ft_headlines = _all_headlines(featured)
    for t in featured:
        ft_urls.extend(_evidence_urls(t))

    # --- category_sections ---
    cat_sections = data.get("category_sections", [])
    cat_section_count = len(cat_sections)
    cat_topic_count = 0
    cat_urls = []
    cat_headlines = []
    unique_categories = set()
    for cs in cat_sections:
        topics = cs.get("topics", [])
        cat_topic_count += len(topics)
        cat_headlines.extend(_all_headlines(topics))
        unique_categories.add(cs.get("category", "unknown"))
        for t in topics:
            cat_urls.extend(_evidence_urls(t))

    # --- source_sections ---
    src_sections = data.get("source_sections", [])
    src_section_count = len(src_sections)
    src_item_count = 0
    src_urls = []
    unique_sources = set()
    for ss in src_sections:
        items = ss.get("items", [])
        src_item_count += len(items)
        for it in items:
            if it.get("url"):
                src_urls.append(it["url"])
            if it.get("source_name"):
                unique_sources.add(it["source_name"])

    all_urls = set(ft_urls + cat_urls + src_urls)
    all_headlines = ft_headlines + cat_headlines

    return {
        "date": date,
        "ft_count": ft_count,
        "cat_section_count": cat_section_count,
        "cat_topic_count": cat_topic_count,
        "src_section_count": src_section_count,
        "src_item_count": src_item_count,
        "unique_sources": len(unique_sources),
        "unique_categories": len(unique_categories),
        "all_urls": all_urls,
        "all_headlines": all_headlines,
    }


# ── pair-wise overlap ───────────────────────────────────────────────────

def compute_pair_overlap(day_a: dict, day_b: dict) -> dict:
    urls_a = day_a["all_urls"]
    urls_b = day_b["all_urls"]
    union = urls_a | urls_b
    shared = urls_a & urls_b
    url_overlap = len(shared) / len(union) * 100 if union else 0.0

    # headline near-duplicate count
    headline_overlaps = 0
    for ha in day_a["all_headlines"]:
        for hb in day_b["all_headlines"]:
            ratio = difflib.SequenceMatcher(None, ha.lower(), hb.lower()).ratio()
            if ratio >= HEADLINE_SIM_THRESHOLD:
                headline_overlaps += 1

    return {
        "pair": f"{day_a['date']} / {day_b['date']}",
        "url_overlap_pct": url_overlap,
        "headline_overlap_count": headline_overlaps,
        "shared_urls": len(shared),
        "union_urls": len(union),
    }


# ── main ─────────────────────────────────────────────────────────────────

def main():
    pattern = str(DATA_DIR / "202*.json")
    files = sorted(
        p for p in glob.glob(pattern)
        if not os.path.basename(p).startswith("index")
    )

    if not files:
        print(f"ERROR: no daily JSON files found matching {pattern}", file=sys.stderr)
        sys.exit(1)

    print(f"Data directory : {DATA_DIR}")
    print(f"Files found    : {len(files)}")
    print()

    # ── per-day metrics ──
    days = [extract_day_metrics(f) for f in files]

    hdr = (
        f"{'Date':<12}  "
        f"{'FeatTopics':>10}  "
        f"{'CatSects':>8}  "
        f"{'CatTopics':>9}  "
        f"{'SrcSects':>8}  "
        f"{'SrcItems':>8}  "
        f"{'UniqSrcs':>8}  "
        f"{'UniqCats':>8}  "
        f"{'URLs':>6}"
    )
    sep = "-" * len(hdr)

    print("=" * len(hdr))
    print("PER-DAY METRICS")
    print("=" * len(hdr))
    print(hdr)
    print(sep)

    for d in days:
        print(
            f"{d['date']:<12}  "
            f"{d['ft_count']:>10}  "
            f"{d['cat_section_count']:>8}  "
            f"{d['cat_topic_count']:>9}  "
            f"{d['src_section_count']:>8}  "
            f"{d['src_item_count']:>8}  "
            f"{d['unique_sources']:>8}  "
            f"{d['unique_categories']:>8}  "
            f"{len(d['all_urls']):>6}"
        )

    # averages
    n = len(days)
    avg_ft   = sum(d["ft_count"]          for d in days) / n
    avg_cs   = sum(d["cat_section_count"] for d in days) / n
    avg_ct   = sum(d["cat_topic_count"]   for d in days) / n
    avg_ss   = sum(d["src_section_count"] for d in days) / n
    avg_si   = sum(d["src_item_count"]    for d in days) / n
    avg_usrc = sum(d["unique_sources"]    for d in days) / n
    avg_ucat = sum(d["unique_categories"] for d in days) / n
    avg_urls = sum(len(d["all_urls"])     for d in days) / n

    print(sep)
    print(
        f"{'AVERAGE':<12}  "
        f"{avg_ft:>10.1f}  "
        f"{avg_cs:>8.1f}  "
        f"{avg_ct:>9.1f}  "
        f"{avg_ss:>8.1f}  "
        f"{avg_si:>8.1f}  "
        f"{avg_usrc:>8.1f}  "
        f"{avg_ucat:>8.1f}  "
        f"{avg_urls:>6.1f}"
    )
    print()

    # ── pair-wise overlap ──
    pairs = []
    for i in range(len(days) - 1):
        pairs.append(compute_pair_overlap(days[i], days[i + 1]))

    phdr = (
        f"{'Pair':<27}  "
        f"{'URL Overlap%':>12}  "
        f"{'SharedURLs':>10}  "
        f"{'UnionURLs':>9}  "
        f"{'HdlnOverlap':>11}"
    )
    psep = "-" * len(phdr)

    print("=" * len(phdr))
    print("CONSECUTIVE-DAY OVERLAP")
    print("=" * len(phdr))
    print(phdr)
    print(psep)

    for p in pairs:
        print(
            f"{p['pair']:<27}  "
            f"{p['url_overlap_pct']:>11.1f}%  "
            f"{p['shared_urls']:>10}  "
            f"{p['union_urls']:>9}  "
            f"{p['headline_overlap_count']:>11}"
        )

    if pairs:
        avg_url_ov  = sum(p["url_overlap_pct"]        for p in pairs) / len(pairs)
        avg_hdl_ov  = sum(p["headline_overlap_count"]  for p in pairs) / len(pairs)
        print(psep)
        print(
            f"{'AVERAGE':<27}  "
            f"{avg_url_ov:>11.1f}%  "
            f"{'':>10}  "
            f"{'':>9}  "
            f"{avg_hdl_ov:>11.1f}"
        )
    print()

    # ── grand summary ──
    print("=" * 50)
    print("GRAND SUMMARY")
    print("=" * 50)
    print(f"  Total daily files       : {n}")
    print(f"  Avg featured topics/day : {avg_ft:.1f}")
    print(f"  Avg category topics/day : {avg_ct:.1f}")
    print(f"  Avg source items/day    : {avg_si:.1f}")
    print(f"  Avg unique sources/day  : {avg_usrc:.1f}")
    print(f"  Avg unique categories   : {avg_ucat:.1f}")
    print(f"  Avg unique URLs/day     : {avg_urls:.1f}")
    if pairs:
        print(f"  Avg URL overlap (adj.)  : {avg_url_ov:.1f}%")
        print(f"  Avg headline overlap    : {avg_hdl_ov:.1f}")
    print()


if __name__ == "__main__":
    main()
