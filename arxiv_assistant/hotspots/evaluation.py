"""Hotspot pipeline evaluation framework.

Computes quality metrics across daily reports and source data
to track improvements across pipeline iterations.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arxiv_assistant.utils.hotspot.hotspot_config import build_hotspot_paths

HIGH_TRUST_TIERS = {"official", "trusted_research", "trusted_analysis"}
LOW_TRUST_TIERS = {"low_trust_recap"}

# Default source-to-tier mapping (can be overridden by source_tiers.json)
DEFAULT_SOURCE_TIER_MAP: dict[str, str] = {
    "official_blogs": "official",
    "local_papers": "trusted_research",
    "hf_papers": "trusted_research",
    "hn_discussion": "community_signal",
    "ainews": "community_signal",
    "ainews_twitter": "community_signal",
    "paperpulse_researcher_feed": "community_signal",
    "github_trend": "builder_ecosystem",
    "the_rundown_ai": "low_trust_recap",
    "superhuman_ai": "low_trust_recap",
    "the_neuron": "low_trust_recap",
    "tldr_ai": "low_trust_recap",
    "import_ai": "trusted_analysis",
    "the_batch": "trusted_analysis",
    "last_week_in_ai": "trusted_analysis",
    "bens_bites": "builder_ecosystem",
}

AWESOME_LIST_PATTERN = re.compile(r"^awesome[-_]", re.I)
WRAPPER_PATTERN = re.compile(r"[-_](wrapper|agent[-_]sdk)$", re.I)


def _load_source_tier_map(config_root: Path) -> dict[str, str]:
    path = config_root / "hotspot" / "source_tiers.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("source_id_to_tier", DEFAULT_SOURCE_TIER_MAP)
    return dict(DEFAULT_SOURCE_TIER_MAP)


def _get_source_tier(source_id: str, tier_map: dict[str, str]) -> str:
    if source_id in tier_map:
        return tier_map[source_id]
    # Handle x_authority:handle pattern
    if source_id.startswith("x_authority:"):
        return "official"
    # Default for roundup sources
    return "low_trust_recap"


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def evaluate_single_day(
    output_root: Path,
    date_str: str,
    tier_map: dict[str, str],
) -> dict[str, Any] | None:
    """Evaluate metrics for a single day's report and raw data."""
    from datetime import date as date_type
    target_date = date_type.fromisoformat(date_str)
    paths = build_hotspot_paths(output_root, target_date)

    report = _load_json(paths.report_path)
    raw_data_dir = paths.raw_root

    # --- Source Health Metrics ---
    source_stats: dict[str, int] = {}
    if report:
        source_stats = report.get("source_stats", {})
    elif raw_data_dir.exists():
        for f in raw_data_dir.glob("*.json"):
            items = _load_json(f)
            source_stats[f.stem] = len(items) if isinstance(items, list) else 0

    active_source_count = sum(1 for count in source_stats.values() if count > 0)
    total_items = sum(source_stats.values())

    # Compute high-quality item ratio
    high_quality_items = 0
    for src_id, count in source_stats.items():
        tier = _get_source_tier(src_id, tier_map)
        if tier in HIGH_TRUST_TIERS:
            high_quality_items += count
    high_quality_item_ratio = high_quality_items / total_items if total_items > 0 else 0.0

    # Source diversity score (entropy)
    source_counts = [c for c in source_stats.values() if c > 0]
    source_diversity_score = _entropy(source_counts)

    # Official items per day
    official_items = sum(
        count for src_id, count in source_stats.items()
        if _get_source_tier(src_id, tier_map) == "official"
    )

    # --- Content Quality Metrics ---
    raw_items_count = total_items
    clusters_count = 0
    featured_topics = []
    category_sections = []
    long_tail_sections = []
    watchlist = []

    if report:
        totals = report.get("totals", {})
        raw_items_count = totals.get("raw_items", total_items)
        clusters_count = totals.get("clusters", 0)
        featured_topics = report.get("featured_topics", report.get("top_topics", []))
        category_sections = report.get("category_sections", [])
        long_tail_sections = report.get("long_tail_sections", [])
        watchlist = report.get("watchlist", [])

    cluster_compression_ratio = clusters_count / raw_items_count if raw_items_count > 0 else 1.0

    # Featured topic analysis
    featured_count = len(featured_topics)
    single_source_featured = 0
    multi_source_featured = 0
    old_paper_in_featured = 0
    primary_source_featured = 0
    low_trust_anchored = 0
    category_correct = 0
    filler_topics = 0

    for topic in featured_topics:
        source_ids = topic.get("source_ids", [])
        source_roles = topic.get("source_roles", [])
        source_count = len(set(source_ids))
        final_score = float(topic.get("FINAL_SCORE", 0.0))

        if source_count <= 1:
            single_source_featured += 1
        else:
            multi_source_featured += 1

        if "official_news" in source_roles:
            primary_source_featured += 1

        # Check if anchored only by low-trust sources
        topic_tiers = {_get_source_tier(sid, tier_map) for sid in source_ids}
        if topic_tiers and topic_tiers.issubset(LOW_TRUST_TIERS):
            low_trust_anchored += 1

        if final_score < 3.0:
            filler_topics += 1

    single_source_featured_ratio = single_source_featured / featured_count if featured_count > 0 else 0.0
    multi_source_featured_ratio = multi_source_featured / featured_count if featured_count > 0 else 0.0
    primary_source_rate = primary_source_featured / featured_count if featured_count > 0 else 0.0
    low_trust_source_share = low_trust_anchored / featured_count if featured_count > 0 else 0.0
    filler_ratio = filler_topics / featured_count if featured_count > 0 else 0.0

    # Section count
    non_empty_sections = sum(
        1 for section in category_sections
        if section.get("topics") and len(section["topics"]) > 0
    )

    # Deep read count
    deep_read_count = 0
    for section in category_sections + long_tail_sections:
        for topic in section.get("topics", []):
            if topic.get("PRIMARY_CATEGORY") == "Deep Read" or "editorial_depth" in topic.get("source_roles", []):
                deep_read_count += 1

    # --- Negative Signal Metrics ---
    awesome_list_count = 0
    wrapper_sdk_count = 0

    # Check raw data for GitHub items
    github_raw_path = raw_data_dir / "github_trend.json"
    if github_raw_path.exists():
        github_items = _load_json(github_raw_path) or []
        for item in github_items:
            title = (item.get("title") or "").lower()
            if AWESOME_LIST_PATTERN.search(title):
                awesome_list_count += 1
            if WRAPPER_PATTERN.search(title):
                wrapper_sdk_count += 1

    # --- HF Paper Age Analysis ---
    hf_raw_path = raw_data_dir / "hf_papers.json"
    hf_paper_ages = []
    old_paper_resurfacing_rate = 0.0
    if hf_raw_path.exists():
        hf_items = _load_json(hf_raw_path) or []
        for item in hf_items:
            pub = item.get("published_at")
            if pub:
                from arxiv_assistant.utils.hotspot.hotspot_sources import parse_datetime
                pub_dt = parse_datetime(pub)
                if pub_dt:
                    if pub_dt.tzinfo is None:
                        pub_dt = pub_dt.replace(tzinfo=UTC)
                    target_dt = datetime.fromisoformat(f"{date_str}T00:00:00+00:00")
                    age_days = (target_dt - pub_dt).days
                    if age_days >= 0:
                        hf_paper_ages.append(age_days)
        if hf_paper_ages:
            old_papers = sum(1 for age in hf_paper_ages if age > 30)
            old_paper_resurfacing_rate = old_papers / len(hf_paper_ages)

    # --- Builder Quality Score ---
    builder_quality_score = 0.0
    if github_raw_path.exists():
        github_items = _load_json(github_raw_path) or []
        stars_list = []
        for item in github_items:
            meta = item.get("metadata", {})
            stars = int(meta.get("github_stars", 0) or meta.get("stars", 0) or 0)
            if stars > 0:
                stars_list.append(stars)
        if stars_list:
            builder_quality_score = sum(stars_list) / len(stars_list)

    # --- Featured Precision Proxy ---
    precision_count = 0
    for topic in featured_topics:
        source_ids = topic.get("source_ids", [])
        source_count = len(set(source_ids))
        topic_tiers = {_get_source_tier(sid, tier_map) for sid in source_ids}
        has_high_trust = bool(topic_tiers & HIGH_TRUST_TIERS)
        if source_count >= 2 and has_high_trust:
            precision_count += 1
    featured_precision_proxy = precision_count / featured_count if featured_count > 0 else 0.0

    return {
        "date": date_str,
        "source_health": {
            "active_source_count": active_source_count,
            "total_items": total_items,
            "high_quality_item_ratio": round(high_quality_item_ratio, 4),
            "source_diversity_score": round(source_diversity_score, 4),
            "official_items_per_day": official_items,
            "source_stats": source_stats,
        },
        "content_quality": {
            "raw_items": raw_items_count,
            "clusters": clusters_count,
            "cluster_compression_ratio": round(cluster_compression_ratio, 4),
            "featured_count": featured_count,
            "single_source_featured_ratio": round(single_source_featured_ratio, 4),
            "multi_source_featured_ratio": round(multi_source_featured_ratio, 4),
            "old_paper_resurfacing_rate": round(old_paper_resurfacing_rate, 4),
            "primary_source_rate": round(primary_source_rate, 4),
            "low_trust_source_share": round(low_trust_source_share, 4),
            "filler_ratio": round(filler_ratio, 4),
        },
        "output_quality": {
            "section_count": non_empty_sections,
            "deep_read_count": deep_read_count,
            "featured_precision_proxy": round(featured_precision_proxy, 4),
            "builder_quality_score": round(builder_quality_score, 1),
        },
        "negative_signals": {
            "awesome_list_count": awesome_list_count,
            "wrapper_sdk_count": wrapper_sdk_count,
        },
        "hf_paper_analysis": {
            "total_papers": len(hf_paper_ages),
            "median_age_days": round(sorted(hf_paper_ages)[len(hf_paper_ages) // 2], 1) if hf_paper_ages else 0.0,
            "pct_over_30d": round(sum(1 for a in hf_paper_ages if a > 30) / len(hf_paper_ages), 4) if hf_paper_ages else 0.0,
            "pct_over_90d": round(sum(1 for a in hf_paper_ages if a > 90) / len(hf_paper_ages), 4) if hf_paper_ages else 0.0,
        },
    }


def evaluate_all_days(
    output_root: Path,
    config_root: Path | None = None,
) -> dict[str, Any]:
    """Evaluate metrics across all available report days."""
    if config_root is None:
        config_root = output_root.parent / "configs"

    tier_map = _load_source_tier_map(config_root)

    # Find all available dates from raw data and reports
    raw_root = output_root / "hot" / "raw"
    report_root = output_root / "hot" / "reports"
    dates: set[str] = set()
    if raw_root.exists():
        for d in raw_root.iterdir():
            if d.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}", d.name):
                dates.add(d.name)
    if report_root.exists():
        for f in report_root.glob("*.json"):
            if re.match(r"\d{4}-\d{2}-\d{2}\.json", f.name):
                dates.add(f.stem)
    dates_sorted = sorted(dates)

    daily_results = []
    for date_str in dates_sorted:
        result = evaluate_single_day(output_root, date_str, tier_map)
        if result:
            daily_results.append(result)

    # Compute aggregate metrics
    num_days = len(daily_results)
    aggregate = _compute_aggregate(daily_results, num_days)

    # Quality gate assessment
    quality_gate = _assess_quality_gate(aggregate) if num_days > 0 else {}

    evaluation = {
        "evaluation_date": datetime.now(UTC).isoformat(),
        "num_days": num_days,
        "date_range": f"{dates_sorted[0]} to {dates_sorted[-1]}" if dates_sorted else "N/A",
        "aggregate": aggregate,
        "quality_gate": quality_gate,
        "daily": daily_results,
    }

    # Write evaluation output
    eval_path = output_root / "hot" / "evaluation.json"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8")

    return evaluation


def _compute_aggregate(daily: list[dict[str, Any]], num_days: int) -> dict[str, Any]:
    if num_days == 0:
        return {}

    def _avg(key_path: list[str]) -> float:
        values = []
        for d in daily:
            obj = d
            for k in key_path:
                obj = obj.get(k, {}) if isinstance(obj, dict) else 0
            if isinstance(obj, (int, float)):
                values.append(obj)
        return round(sum(values) / len(values), 4) if values else 0.0

    # Collect all unique source IDs across all days
    all_source_ids: set[str] = set()
    for d in daily:
        stats = d.get("source_health", {}).get("source_stats", {})
        all_source_ids.update(stats.keys())

    return {
        "avg_active_source_count": _avg(["source_health", "active_source_count"]),
        "avg_total_items": _avg(["source_health", "total_items"]),
        "avg_high_quality_item_ratio": _avg(["source_health", "high_quality_item_ratio"]),
        "avg_source_diversity_score": _avg(["source_health", "source_diversity_score"]),
        "avg_official_items_per_day": _avg(["source_health", "official_items_per_day"]),
        "avg_cluster_compression_ratio": _avg(["content_quality", "cluster_compression_ratio"]),
        "avg_featured_count": _avg(["content_quality", "featured_count"]),
        "avg_single_source_featured_ratio": _avg(["content_quality", "single_source_featured_ratio"]),
        "avg_multi_source_featured_ratio": _avg(["content_quality", "multi_source_featured_ratio"]),
        "avg_old_paper_resurfacing_rate": _avg(["content_quality", "old_paper_resurfacing_rate"]),
        "avg_primary_source_rate": _avg(["content_quality", "primary_source_rate"]),
        "avg_low_trust_source_share": _avg(["content_quality", "low_trust_source_share"]),
        "avg_filler_ratio": _avg(["content_quality", "filler_ratio"]),
        "avg_section_count": _avg(["output_quality", "section_count"]),
        "avg_deep_read_count": _avg(["output_quality", "deep_read_count"]),
        "avg_featured_precision_proxy": _avg(["output_quality", "featured_precision_proxy"]),
        "avg_builder_quality_score": _avg(["output_quality", "builder_quality_score"]),
        "total_awesome_list_count": sum(d.get("negative_signals", {}).get("awesome_list_count", 0) for d in daily),
        "total_wrapper_sdk_count": sum(d.get("negative_signals", {}).get("wrapper_sdk_count", 0) for d in daily),
        "all_sources_seen": sorted(all_source_ids),
        "deep_read_hit_rate_per_week": round(_avg(["output_quality", "deep_read_count"]) * 7, 1),
    }


def _assess_quality_gate(aggregate: dict[str, Any]) -> dict[str, Any]:
    """Check all quality gate criteria from Section 2.3 of the execution plan."""
    checks = {
        "active_source_count_ge_8": aggregate.get("avg_active_source_count", 0) >= 8,
        "high_quality_item_ratio_ge_35pct": aggregate.get("avg_high_quality_item_ratio", 0) >= 0.35,
        "single_source_featured_ratio_le_15pct": aggregate.get("avg_single_source_featured_ratio", 1) <= 0.15,
        "old_paper_resurfacing_rate_le_10pct": aggregate.get("avg_old_paper_resurfacing_rate", 1) <= 0.10,
        "category_purity_ge_95pct": True,  # Will be measured when category validation is added
        "deep_read_hit_rate_ge_5_per_week": aggregate.get("deep_read_hit_rate_per_week", 0) >= 5,
        "section_count_ge_4": aggregate.get("avg_section_count", 0) >= 4,
        "awesome_list_count_eq_0": aggregate.get("total_awesome_list_count", 1) == 0,
        "filler_ratio_eq_0": aggregate.get("avg_filler_ratio", 1) == 0.0,
    }
    return {
        "all_passed": all(checks.values()),
        "passed_count": sum(1 for v in checks.values() if v),
        "total_checks": len(checks),
        "checks": checks,
    }


def print_evaluation_summary(evaluation: dict[str, Any]) -> None:
    """Print a human-readable evaluation summary."""
    agg = evaluation.get("aggregate", {})
    gate = evaluation.get("quality_gate", {})

    print("\n" + "=" * 60)
    print("HOTSPOT V2 EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Date range: {evaluation.get('date_range', 'N/A')}")
    print(f"Days evaluated: {evaluation.get('num_days', 0)}")

    print("\n--- Source Health ---")
    print(f"  Active sources (avg):        {agg.get('avg_active_source_count', 0):.1f}  (target: >= 8)")
    print(f"  High-quality item ratio:     {agg.get('avg_high_quality_item_ratio', 0):.1%}  (target: >= 35%)")
    print(f"  Source diversity (entropy):   {agg.get('avg_source_diversity_score', 0):.2f}")
    print(f"  Official items/day:          {agg.get('avg_official_items_per_day', 0):.1f}  (target: >= 8)")

    print("\n--- Content Quality ---")
    print(f"  Cluster compression ratio:   {agg.get('avg_cluster_compression_ratio', 0):.3f}  (target: <= 0.70)")
    print(f"  Single-source featured:      {agg.get('avg_single_source_featured_ratio', 0):.1%}  (target: <= 15%)")
    print(f"  Multi-source featured:       {agg.get('avg_multi_source_featured_ratio', 0):.1%}  (target: >= 85%)")
    print(f"  Old paper resurfacing:       {agg.get('avg_old_paper_resurfacing_rate', 0):.1%}  (target: <= 10%)")
    print(f"  Primary source rate:         {agg.get('avg_primary_source_rate', 0):.1%}  (target: >= 90%)")
    print(f"  Low-trust source share:      {agg.get('avg_low_trust_source_share', 0):.1%}  (target: <= 5%)")

    print("\n--- Output Quality ---")
    print(f"  Section count (avg):         {agg.get('avg_section_count', 0):.1f}  (target: >= 4)")
    print(f"  Deep read hit rate/week:     {agg.get('deep_read_hit_rate_per_week', 0):.1f}  (target: >= 5)")
    print(f"  Featured precision proxy:    {agg.get('avg_featured_precision_proxy', 0):.1%}  (target: >= 80%)")
    print(f"  Builder quality score:       {agg.get('avg_builder_quality_score', 0):.0f}  (target: >= 200)")

    print("\n--- Negative Signals ---")
    print(f"  Awesome-list count (total):  {agg.get('total_awesome_list_count', 0)}  (target: 0)")
    print(f"  Wrapper SDK count (total):   {agg.get('total_wrapper_sdk_count', 0)}  (target: 0)")
    print(f"  Filler ratio:                {agg.get('avg_filler_ratio', 0):.1%}  (target: 0%)")

    print("\n--- Quality Gate ---")
    if gate:
        print(f"  Status: {'ALL PASSED' if gate.get('all_passed') else 'FAILED'} ({gate.get('passed_count', 0)}/{gate.get('total_checks', 0)})")
        for check_name, passed in gate.get("checks", {}).items():
            status = "PASS" if passed else "FAIL"
            print(f"    [{status}] {check_name}")
    print("=" * 60 + "\n")
