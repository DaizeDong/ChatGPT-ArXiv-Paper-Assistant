"""Hotspot quality evaluation framework.

Computes quality metrics over daily hotspot reports to track improvement
across pipeline iterations. Metrics are defined in the redesign plan
(docs/HOTSPOT_QUALITY_REDESIGN_PLAN.md §7).
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------

ERROR_TAXONOMY = {
    "resurfaced_old_paper": "Old paper (>30 days) surfaced as same-day research",
    "paper_leak_into_non_research": "Paper artifact placed in non-research section",
    "low_trust_anchor": "Featured item anchored only by community/recap source",
    "missing_primary_source": "Product/launch event missing official primary source",
    "single_source_featured": "Featured topic backed by only one source",
    "category_mismatch": "Item placed in wrong category based on artifact type",
    "filler_topic": "Low-signal topic used to fill section quota",
    "fragment_duplicate": "Multiple final topics referring to the same real-world event",
    "mixed_cluster": "Cluster merging unrelated items due to title overlap",
    "stale_roundup_item": "Newsletter/roundup item older than 48h treated as fresh",
}

# Roles that count as paper-only
PAPER_ONLY_ROLES = {"research_backbone", "paper_trending"}

# Roles that count as low-trust
LOW_TRUST_ROLES = {"community_heat", "headline_consensus", "hn_discussion"}

# Categories that should not contain paper artifacts
NON_RESEARCH_CATEGORIES = {"Product Release", "Tooling", "Industry Update", "Community Signal"}


# ---------------------------------------------------------------------------
# Per-day metric computation
# ---------------------------------------------------------------------------

@dataclass
class DailyMetrics:
    """Quality metrics for a single day's hotspot report."""

    date: str = ""
    raw_items: int = 0
    clusters: int = 0
    featured_count: int = 0

    # Core metrics
    cluster_compression_ratio: float = 1.0
    single_source_featured_ratio: float = 0.0
    paper_like_featured_ratio: float = 0.0
    old_paper_resurfacing_rate: float = 0.0
    primary_source_rate: float = 0.0
    low_trust_source_share: float = 0.0
    paper_leakage_rate: float = 0.0
    category_purity: float = 1.0

    # Iteration 4 metrics
    multi_source_featured_ratio: float = 0.0
    section_count: int = 0
    deep_read_count: int = 0

    # Supplementary
    hf_paper_ages_days: list[float] = field(default_factory=list)
    x_official_count: int = 0
    official_blogs_count: int = 0

    # Error counts
    errors: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "date": self.date,
            "raw_items": self.raw_items,
            "clusters": self.clusters,
            "featured_count": self.featured_count,
            "cluster_compression_ratio": round(self.cluster_compression_ratio, 4),
            "single_source_featured_ratio": round(self.single_source_featured_ratio, 4),
            "paper_like_featured_ratio": round(self.paper_like_featured_ratio, 4),
            "old_paper_resurfacing_rate": round(self.old_paper_resurfacing_rate, 4),
            "primary_source_rate": round(self.primary_source_rate, 4),
            "low_trust_source_share": round(self.low_trust_source_share, 4),
            "paper_leakage_rate": round(self.paper_leakage_rate, 4),
            "category_purity": round(self.category_purity, 4),
            "multi_source_featured_ratio": round(self.multi_source_featured_ratio, 4),
            "section_count": self.section_count,
            "deep_read_count": self.deep_read_count,
            "x_official_count": self.x_official_count,
            "official_blogs_count": self.official_blogs_count,
            "errors": self.errors,
        }
        if self.hf_paper_ages_days:
            result["hf_paper_age_median_days"] = round(statistics.median(self.hf_paper_ages_days), 1)
            result["hf_paper_age_p90_days"] = round(sorted(self.hf_paper_ages_days)[int(len(self.hf_paper_ages_days) * 0.9)], 1)
        return result


def _parse_date(date_str: str | None) -> datetime | None:
    if not date_str:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%d"):
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
    return None


def _paper_age_days(item: dict[str, Any], report_date: datetime) -> float | None:
    """Return the age of a paper in days relative to the report date."""
    published = _parse_date(item.get("published_at"))
    if published is None:
        return None
    delta = report_date - published.replace(tzinfo=None)
    return max(0.0, delta.total_seconds() / 86400)


def _is_paper_item(item: dict[str, Any]) -> bool:
    return (
        item.get("source_type") == "paper"
        or bool((item.get("metadata") or {}).get("arxiv_id"))
    )


def _has_official_source(topic: dict[str, Any]) -> bool:
    return "official_news" in (topic.get("source_roles") or [])


def _is_product_or_launch(topic: dict[str, Any]) -> bool:
    category = topic.get("PRIMARY_CATEGORY", "")
    return category in {"Product Release", "Industry Update"}


def _section_has_paper(section_topics: list[dict[str, Any]]) -> int:
    """Count topics in a non-research section that contain paper artifacts."""
    count = 0
    for topic in section_topics:
        items = topic.get("items", [])
        if any(_is_paper_item(item) for item in items):
            count += 1
    return count


_VENDOR_NAMES = {"openai", "anthropic", "google", "deepmind", "meta", "nvidia", "amazon", "apple", "cursor", "claude", "gpt", "gemini", "qwen", "deepseek", "mistral", "llama", "microsoft", "cohere", "stability"}
_RELEASE_VERBS = {"launch", "launches", "release", "released", "announced", "introducing", "debuts", "acquire", "rollout", "bets", "testing", "ships"}


def _estimate_category_purity(topic: dict[str, Any]) -> bool:
    """Heuristic: does the topic's source types align with its category?"""
    category = topic.get("PRIMARY_CATEGORY", "")
    roles = set(topic.get("source_roles") or [])
    source_types = set(topic.get("source_types") or [])
    artifact_type = topic.get("ARTIFACT_TYPE", "")
    event_type = topic.get("EVENT_TYPE", "")
    title_lower = str(topic.get("title", topic.get("TITLE", ""))).lower()
    title_words = set(title_lower.split())

    # When typed fields are present, use them for a direct check
    if artifact_type or event_type:
        from arxiv_assistant.filters.filter_hotspots import _ARTIFACT_CATEGORY_MAP, _EVENT_CATEGORY_MAP
        expected = _EVENT_CATEGORY_MAP.get(event_type) or _ARTIFACT_CATEGORY_MAP.get(artifact_type)
        if expected:
            return category == expected
        return True

    if category == "Research":
        return bool({"paper"} & source_types) or bool(PAPER_ONLY_ROLES & roles)
    if category == "Product Release":
        has_vendor = any(v in title_lower for v in _VENDOR_NAMES)
        has_action = bool(_RELEASE_VERBS & title_words)
        return "official_news" in roles or (has_vendor and has_action)
    if category == "Tooling":
        return "github_trend" in roles or "builder_momentum" in roles or "repo" in source_types
    # For Industry Update and Community Signal, be lenient
    return True


def compute_daily_metrics(report: dict[str, Any]) -> DailyMetrics:
    """Compute quality metrics from a single daily report."""
    metrics = DailyMetrics()
    metrics.date = report.get("date", "")
    report_date = _parse_date(metrics.date) or datetime(2026, 1, 1)

    totals = report.get("totals", {})
    metrics.raw_items = totals.get("raw_items", 0)
    metrics.clusters = totals.get("clusters", 0)

    if metrics.raw_items > 0:
        metrics.cluster_compression_ratio = metrics.clusters / metrics.raw_items
    else:
        metrics.cluster_compression_ratio = 1.0

    source_stats = report.get("source_stats", {})
    metrics.x_official_count = source_stats.get("x_official", 0)
    metrics.official_blogs_count = source_stats.get("official_blogs", 0)

    # Analyze featured topics
    featured = report.get("featured_topics") or report.get("top_topics") or []
    metrics.featured_count = len(featured)

    if not featured:
        return metrics

    single_source = 0
    paper_like = 0
    old_paper_in_research = 0
    research_featured = 0
    product_launch_count = 0
    product_with_primary = 0
    low_trust_only = 0
    pure_category = 0
    errors: dict[str, int] = {}

    for topic in featured:
        source_ids = topic.get("source_ids") or topic.get("source_names") or []
        source_roles = set(topic.get("source_roles") or [])
        category = topic.get("PRIMARY_CATEGORY", "")

        # Single source
        if len(source_ids) <= 1:
            single_source += 1
            errors["single_source_featured"] = errors.get("single_source_featured", 0) + 1

        # Paper-like (only paper roles)
        if source_roles and source_roles.issubset(PAPER_ONLY_ROLES):
            paper_like += 1

        # Old paper resurfacing in research featured
        if category == "Research":
            research_featured += 1
            items = topic.get("items", [])
            has_old_paper = False
            for item in items:
                if _is_paper_item(item):
                    age = _paper_age_days(item, report_date)
                    if age is not None and age > 30:
                        has_old_paper = True
                        break
            if has_old_paper:
                old_paper_in_research += 1
                errors["resurfaced_old_paper"] = errors.get("resurfaced_old_paper", 0) + 1

        # Primary source rate for product/launch
        if _is_product_or_launch(topic):
            product_launch_count += 1
            if _has_official_source(topic):
                product_with_primary += 1
            else:
                errors["missing_primary_source"] = errors.get("missing_primary_source", 0) + 1

        # Low trust source share
        if source_roles and source_roles.issubset(LOW_TRUST_ROLES):
            low_trust_only += 1
            errors["low_trust_anchor"] = errors.get("low_trust_anchor", 0) + 1

        # Category purity
        if _estimate_category_purity(topic):
            pure_category += 1
        else:
            errors["category_mismatch"] = errors.get("category_mismatch", 0) + 1

    n = len(featured)
    multi_source = sum(1 for t in featured if len(t.get("source_ids", t.get("source_names", []))) >= 2)
    metrics.single_source_featured_ratio = single_source / n
    metrics.multi_source_featured_ratio = multi_source / n
    metrics.paper_like_featured_ratio = paper_like / n
    metrics.old_paper_resurfacing_rate = old_paper_in_research / research_featured if research_featured > 0 else 0.0
    metrics.primary_source_rate = product_with_primary / product_launch_count if product_launch_count > 0 else 1.0
    metrics.low_trust_source_share = low_trust_only / n
    metrics.category_purity = pure_category / n

    # Section metrics (Iteration 4)
    sections = report.get("category_sections", [])
    metrics.section_count = len(sections)
    for section in sections:
        slug = section.get("slug", "")
        if slug == "deep_reads" or section.get("category") == "Deep Reads":
            metrics.deep_read_count = len(section.get("topics", []))

    # Paper leakage rate: check category_sections for non-research sections containing papers
    paper_leak_count = 0
    non_research_topic_count = 0
    for section in report.get("category_sections", []):
        cat = section.get("category", "")
        topics_in_section = section.get("topics", [])
        if cat in NON_RESEARCH_CATEGORIES and topics_in_section:
            non_research_topic_count += len(topics_in_section)
            leaks = _section_has_paper(topics_in_section)
            paper_leak_count += leaks
            if leaks > 0:
                errors["paper_leak_into_non_research"] = errors.get("paper_leak_into_non_research", 0) + leaks

    metrics.paper_leakage_rate = paper_leak_count / non_research_topic_count if non_research_topic_count > 0 else 0.0
    metrics.errors = errors

    # Compute HF paper ages from raw items if available in featured topics
    for topic in featured:
        for item in topic.get("items", []):
            if item.get("source_id") == "hf_papers" and _is_paper_item(item):
                age = _paper_age_days(item, report_date)
                if age is not None:
                    metrics.hf_paper_ages_days.append(age)

    return metrics


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------

@dataclass
class AggregateMetrics:
    """Aggregated quality metrics across multiple days."""

    num_days: int = 0
    date_range: str = ""

    avg_cluster_compression_ratio: float = 0.0
    avg_single_source_featured_ratio: float = 0.0
    avg_paper_like_featured_ratio: float = 0.0
    avg_old_paper_resurfacing_rate: float = 0.0
    avg_primary_source_rate: float = 0.0
    avg_low_trust_source_share: float = 0.0
    avg_paper_leakage_rate: float = 0.0
    avg_category_purity: float = 0.0
    avg_multi_source_featured_ratio: float = 0.0
    avg_section_count: float = 0.0
    deep_read_hit_rate_per_week: float = 0.0

    hf_paper_age_median: float = 0.0
    hf_paper_age_p90: float = 0.0
    hf_paper_age_p95: float = 0.0

    avg_x_official_count: float = 0.0
    avg_official_blogs_count: float = 0.0
    avg_featured_count: float = 0.0

    total_errors: dict[str, int] = field(default_factory=dict)

    # Iteration targets
    iteration_1_pass: dict[str, bool] = field(default_factory=dict)
    iteration_2_pass: dict[str, bool] = field(default_factory=dict)
    iteration_3_pass: dict[str, bool] = field(default_factory=dict)
    iteration_4_pass: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_days": self.num_days,
            "date_range": self.date_range,
            "avg_cluster_compression_ratio": round(self.avg_cluster_compression_ratio, 4),
            "avg_single_source_featured_ratio": round(self.avg_single_source_featured_ratio, 4),
            "avg_paper_like_featured_ratio": round(self.avg_paper_like_featured_ratio, 4),
            "avg_old_paper_resurfacing_rate": round(self.avg_old_paper_resurfacing_rate, 4),
            "avg_primary_source_rate": round(self.avg_primary_source_rate, 4),
            "avg_low_trust_source_share": round(self.avg_low_trust_source_share, 4),
            "avg_paper_leakage_rate": round(self.avg_paper_leakage_rate, 4),
            "avg_category_purity": round(self.avg_category_purity, 4),
            "avg_multi_source_featured_ratio": round(self.avg_multi_source_featured_ratio, 4),
            "avg_section_count": round(self.avg_section_count, 2),
            "deep_read_hit_rate_per_week": round(self.deep_read_hit_rate_per_week, 2),
            "hf_paper_age_median_days": round(self.hf_paper_age_median, 1),
            "hf_paper_age_p90_days": round(self.hf_paper_age_p90, 1),
            "hf_paper_age_p95_days": round(self.hf_paper_age_p95, 1),
            "avg_x_official_count": round(self.avg_x_official_count, 2),
            "avg_official_blogs_count": round(self.avg_official_blogs_count, 2),
            "avg_featured_count": round(self.avg_featured_count, 2),
            "total_errors": self.total_errors,
            "iteration_targets": {
                "iteration_1": self.iteration_1_pass,
                "iteration_2": self.iteration_2_pass,
                "iteration_3": self.iteration_3_pass,
                "iteration_4": self.iteration_4_pass,
            },
        }


def aggregate_metrics(daily_list: list[DailyMetrics]) -> AggregateMetrics:
    """Aggregate daily metrics into summary statistics."""
    agg = AggregateMetrics()
    if not daily_list:
        return agg

    agg.num_days = len(daily_list)
    dates = sorted(m.date for m in daily_list if m.date)
    agg.date_range = f"{dates[0]} to {dates[-1]}" if dates else ""

    def _avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    agg.avg_cluster_compression_ratio = _avg([m.cluster_compression_ratio for m in daily_list])
    agg.avg_single_source_featured_ratio = _avg([m.single_source_featured_ratio for m in daily_list])
    agg.avg_paper_like_featured_ratio = _avg([m.paper_like_featured_ratio for m in daily_list])
    agg.avg_old_paper_resurfacing_rate = _avg([m.old_paper_resurfacing_rate for m in daily_list])
    agg.avg_primary_source_rate = _avg([m.primary_source_rate for m in daily_list])
    agg.avg_low_trust_source_share = _avg([m.low_trust_source_share for m in daily_list])
    agg.avg_paper_leakage_rate = _avg([m.paper_leakage_rate for m in daily_list])
    agg.avg_category_purity = _avg([m.category_purity for m in daily_list])
    agg.avg_multi_source_featured_ratio = _avg([m.multi_source_featured_ratio for m in daily_list])
    agg.avg_section_count = _avg([float(m.section_count) for m in daily_list])
    total_deep_reads = sum(m.deep_read_count for m in daily_list)
    weeks = max(1.0, agg.num_days / 7.0)
    agg.deep_read_hit_rate_per_week = total_deep_reads / weeks
    agg.avg_x_official_count = _avg([float(m.x_official_count) for m in daily_list])
    agg.avg_official_blogs_count = _avg([float(m.official_blogs_count) for m in daily_list])
    agg.avg_featured_count = _avg([float(m.featured_count) for m in daily_list])

    # HF paper ages
    all_ages = []
    for m in daily_list:
        all_ages.extend(m.hf_paper_ages_days)
    if all_ages:
        all_ages_sorted = sorted(all_ages)
        agg.hf_paper_age_median = statistics.median(all_ages_sorted)
        agg.hf_paper_age_p90 = all_ages_sorted[int(len(all_ages_sorted) * 0.9)]
        agg.hf_paper_age_p95 = all_ages_sorted[min(int(len(all_ages_sorted) * 0.95), len(all_ages_sorted) - 1)]

    # Error totals
    for m in daily_list:
        for error_type, count in m.errors.items():
            agg.total_errors[error_type] = agg.total_errors.get(error_type, 0) + count

    # Check iteration targets
    agg.iteration_1_pass = {
        "single_source_featured_ratio_le_20pct": agg.avg_single_source_featured_ratio <= 0.20,
        "low_trust_source_share_le_25pct": agg.avg_low_trust_source_share <= 0.25,
        "primary_source_rate_ge_60pct": agg.avg_primary_source_rate >= 0.60,
        "paper_leakage_rate_le_10pct": agg.avg_paper_leakage_rate <= 0.10,
    }
    agg.iteration_2_pass = {
        "cluster_compression_ratio_le_065": agg.avg_cluster_compression_ratio <= 0.65,
    }
    agg.iteration_3_pass = {
        "category_purity_ge_90pct": agg.avg_category_purity >= 0.90,
        "paper_leakage_rate_le_3pct": agg.avg_paper_leakage_rate <= 0.03,
        "old_paper_resurfacing_rate_le_15pct": agg.avg_old_paper_resurfacing_rate <= 0.15,
    }
    agg.iteration_4_pass = {
        "multi_source_featured_ge_80pct": agg.avg_multi_source_featured_ratio >= 0.80,
        "avg_section_count_ge_3": agg.avg_section_count >= 3.0,
        "deep_read_hit_rate_ge_4_per_week": agg.deep_read_hit_rate_per_week >= 4.0,
    }

    return agg


# ---------------------------------------------------------------------------
# Report loading and batch evaluation
# ---------------------------------------------------------------------------

def load_reports(reports_dir: Path) -> list[dict[str, Any]]:
    """Load all daily reports from a directory, sorted by date."""
    reports = []
    for path in sorted(reports_dir.glob("*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            if "date" in report:
                reports.append(report)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return reports


def _analyze_raw_hf_papers(raw_root: Path, report_dates: list[str]) -> dict[str, Any]:
    """Analyze all ingested HF papers across all days for age distribution."""
    all_ages: list[float] = []
    total_papers = 0
    for date_str in report_dates:
        hf_path = raw_root / date_str / "hf_papers.json"
        if not hf_path.exists():
            continue
        try:
            items = json.loads(hf_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        report_date = _parse_date(date_str)
        if report_date is None:
            continue
        for item in items:
            total_papers += 1
            age = _paper_age_days(item, report_date)
            if age is not None:
                all_ages.append(age)

    if not all_ages:
        return {"total_papers": total_papers, "papers_with_date": 0}

    all_ages_sorted = sorted(all_ages)
    n = len(all_ages_sorted)
    return {
        "total_papers": total_papers,
        "papers_with_date": n,
        "median_age_days": round(statistics.median(all_ages_sorted), 1),
        "p90_age_days": round(all_ages_sorted[int(n * 0.9)], 1),
        "p95_age_days": round(all_ages_sorted[min(int(n * 0.95), n - 1)], 1),
        "max_age_days": round(all_ages_sorted[-1], 1),
        "pct_over_30d": round(sum(1 for a in all_ages if a > 30) / n, 4),
        "pct_over_90d": round(sum(1 for a in all_ages if a > 90) / n, 4),
        "pct_over_365d": round(sum(1 for a in all_ages if a > 365) / n, 4),
    }


def evaluate_reports(reports_dir: Path) -> dict[str, Any]:
    """Run full evaluation over all reports in the directory.

    Returns a dict with per-day metrics, aggregate metrics, and error taxonomy.
    """
    reports = load_reports(reports_dir)
    daily_metrics = [compute_daily_metrics(report) for report in reports]
    agg = aggregate_metrics(daily_metrics)

    report_dates = [r["date"] for r in reports if "date" in r]
    raw_root = reports_dir.parent / "raw"
    raw_hf_analysis = _analyze_raw_hf_papers(raw_root, report_dates) if raw_root.exists() else {}

    return {
        "evaluation_date": datetime.now(UTC).isoformat(),
        "error_taxonomy": ERROR_TAXONOMY,
        "aggregate": agg.to_dict(),
        "raw_hf_paper_analysis": raw_hf_analysis,
        "daily": [m.to_dict() for m in daily_metrics],
    }


def print_evaluation_summary(eval_result: dict[str, Any]) -> str:
    """Format a human-readable evaluation summary."""
    agg = eval_result["aggregate"]
    lines = [
        "=" * 60,
        "HOTSPOT QUALITY EVALUATION SUMMARY",
        "=" * 60,
        f"Period: {agg['date_range']}  ({agg['num_days']} days)",
        "",
        "--- Core Metrics ---",
        f"  Cluster Compression Ratio:     {agg['avg_cluster_compression_ratio']:.3f}  (target ≤0.65)",
        f"  Single-source Featured Ratio:  {agg['avg_single_source_featured_ratio']:.1%}  (target ≤20%)",
        f"  Paper-like Featured Ratio:     {agg['avg_paper_like_featured_ratio']:.1%}",
        f"  Old-Paper Resurfacing Rate:    {agg['avg_old_paper_resurfacing_rate']:.1%}  (target ≤15%)",
        f"  Primary-Source Rate:           {agg['avg_primary_source_rate']:.1%}  (target ≥60%)",
        f"  Low-Trust Source Share:        {agg['avg_low_trust_source_share']:.1%}  (target ≤25%)",
        f"  Paper Leakage Rate:            {agg['avg_paper_leakage_rate']:.1%}  (target ≤10%)",
        f"  Category Purity:               {agg['avg_category_purity']:.1%}  (target ≥90%)",
        "",
        "--- HF Paper Ages ---",
        f"  Median age: {agg['hf_paper_age_median_days']:.0f} days",
        f"  P90 age:    {agg['hf_paper_age_p90_days']:.0f} days",
        f"  P95 age:    {agg['hf_paper_age_p95_days']:.0f} days",
        "",
        "--- Section & Diversity ---",
        f"  Multi-source Featured Ratio: {agg['avg_multi_source_featured_ratio']:.1%}  (target ≥80%)",
        f"  Avg section count/day:       {agg['avg_section_count']:.1f}  (target ≥3)",
        f"  Deep-read hit rate/week:     {agg['deep_read_hit_rate_per_week']:.1f}  (target ≥4)",
        "",
        "--- Source Coverage ---",
        f"  Avg X official items/day:    {agg['avg_x_official_count']:.1f}",
        f"  Avg official blogs/day:      {agg['avg_official_blogs_count']:.1f}",
        f"  Avg featured topics/day:     {agg['avg_featured_count']:.1f}",
    ]

    raw_hf = eval_result.get("raw_hf_paper_analysis", {})
    if raw_hf:
        lines.extend([
            "",
            "--- All Ingested HF Papers ---",
            f"  Total papers: {raw_hf.get('total_papers', 0)} ({raw_hf.get('papers_with_date', 0)} with dates)",
            f"  Median age: {raw_hf.get('median_age_days', 0):.0f} days",
            f"  P90 age:    {raw_hf.get('p90_age_days', 0):.0f} days",
            f"  P95 age:    {raw_hf.get('p95_age_days', 0):.0f} days",
            f"  % over 30d:  {raw_hf.get('pct_over_30d', 0):.1%}",
            f"  % over 90d:  {raw_hf.get('pct_over_90d', 0):.1%}",
            f"  % over 365d: {raw_hf.get('pct_over_365d', 0):.1%}",
        ])

    lines.extend([
        "",
        "--- Iteration Target Status ---",
    ])

    for iteration_key, label in [("iteration_1", "Iteration 1"), ("iteration_2", "Iteration 2"), ("iteration_3", "Iteration 3"), ("iteration_4", "Iteration 4")]:
        targets = agg["iteration_targets"].get(iteration_key, {})
        passed = sum(1 for v in targets.values() if v)
        total = len(targets)
        lines.append(f"  {label}: {passed}/{total} targets met")
        for target_name, met in targets.items():
            status = "PASS" if met else "FAIL"
            lines.append(f"    [{status}] {target_name}")

    errors = agg.get("total_errors", {})
    if errors:
        lines.extend(["", "--- Error Counts ---"])
        for error_type, count in sorted(errors.items(), key=lambda x: -x[1]):
            lines.append(f"  {error_type}: {count}")

    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import sys

    reports_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out/hot/reports")
    if not reports_dir.exists():
        print(f"Reports directory not found: {reports_dir}")
        sys.exit(1)

    result = evaluate_reports(reports_dir)

    output_path = reports_dir.parent / "evaluation.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Evaluation written to {output_path}")
    print()
    print(print_evaluation_summary(result))


if __name__ == "__main__":
    main()
