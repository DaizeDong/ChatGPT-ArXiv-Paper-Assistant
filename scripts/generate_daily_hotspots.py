from __future__ import annotations

import argparse
import configparser
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arxiv_assistant.apis.hotspot_ainews import fetch_hotspot_items as fetch_ainews_items
from arxiv_assistant.apis.hotspot_hf_papers import fetch_hotspot_items as fetch_hf_items
from arxiv_assistant.apis.hotspot_local_papers import fetch_hotspot_items as fetch_local_paper_items
from arxiv_assistant.apis.hotspot_official_blogs import fetch_hotspot_items as fetch_official_blog_items
from arxiv_assistant.apis.hotspot_roundups import fetch_hotspot_items as fetch_roundup_items
from arxiv_assistant.filters.filter_hotspots import heuristic_screen_clusters, screen_clusters_with_openai, synthesize_digest_with_openai
from arxiv_assistant.utils.hotspot_cluster import build_hotspot_clusters
from arxiv_assistant.utils.hotspot_schema import HotspotCluster, HotspotItem

ROLE_PRIORITY = {
    "research_backbone": 0,
    "paper_trending": 1,
    "official_news": 2,
    "community_heat": 3,
    "headline_consensus": 4,
    "builder_momentum": 5,
    "editorial_depth": 6,
    "github_trend": 7,
    "hn_discussion": 8,
}
APP_TIMEZONE = ZoneInfo("America/New_York")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate daily AI hotspots from multiple signals.")
    parser.add_argument("--output-root", default="out", help="Output root directory.")
    parser.add_argument("--target-date", "--date", dest="target_date", default=None, help="Target date in YYYY-MM-DD format.")
    parser.add_argument("--mode", choices=["auto", "openai", "heuristic"], default="auto", help="Override hotspot mode.")
    parser.add_argument("--force", action="store_true", help="Regenerate even when cached raw items exist.")
    return parser.parse_args()


def load_config(config_path: Path) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")
    return config


def read_prompt(prompt_path: Path) -> str:
    return prompt_path.read_text(encoding="utf-8")


def parse_target_datetime(target_date_arg: str | None) -> datetime:
    if target_date_arg:
        return datetime.strptime(target_date_arg, "%Y-%m-%d").replace(tzinfo=APP_TIMEZONE)
    now = datetime.now(APP_TIMEZONE)
    return datetime(now.year, now.month, now.day, tzinfo=APP_TIMEZONE)


def date_string(target_date: datetime) -> str:
    return f"{target_date.year:04d}-{target_date.month:02d}-{target_date.day:02d}"


def month_string(target_date: datetime) -> str:
    return f"{target_date.year:04d}-{target_date.month:02d}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_cached_items(path: Path) -> list[HotspotItem]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [HotspotItem(**row) for row in payload]


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _serialize_items(items: list[HotspotItem]) -> list[dict[str, Any]]:
    return [item.to_dict() for item in items]


def _item_sort_key(item: HotspotItem) -> tuple[int, str, str]:
    return (ROLE_PRIORITY.get(item.source_role, 99), item.published_at or "", item.title.lower())


def _dedupe_items(items: list[HotspotItem]) -> list[HotspotItem]:
    deduped: dict[tuple[str, str, str], HotspotItem] = {}
    for item in items:
        key = (item.source_id, item.canonical_url or item.url, item.title.lower())
        existing = deduped.get(key)
        if existing is None or len(item.summary) > len(existing.summary):
            deduped[key] = item
    return sorted(deduped.values(), key=_item_sort_key)


def _raw_source_cache_path(output_root: Path, target_date: datetime, source_id: str) -> Path:
    return output_root / "hot" / "raw" / date_string(target_date) / f"{source_id}.json"


def render_hot_daily_md(report: dict[str, Any]) -> str:
    lines = [
        f"# Daily AI Hotspots {report['date']}",
        "",
        report.get("summary", "").strip() or "No strong hotspots were selected today.",
        "",
        "## Source Stats",
        "",
    ]
    for source_name, count in sorted((report.get("source_stats") or {}).items()):
        lines.append(f"- `{source_name}`: {count}")

    lines.extend(
        [
            "",
            "## Top Topics",
            "",
        ]
    )

    top_topics = report.get("top_topics") or []
    if not top_topics:
        lines.append("- No topic cleared the main-list threshold.")
    else:
        for index, topic in enumerate(top_topics, start=1):
            lines.extend(
                [
                    f"### {index}. {topic.get('HEADLINE') or topic.get('title', 'Untitled Topic')}",
                    "",
                    f"- Category: `{topic.get('PRIMARY_CATEGORY', 'Uncategorized')}`",
                    f"- Scores: `quality={topic.get('QUALITY', 0)}` `heat={topic.get('HEAT', 0)}` `importance={topic.get('IMPORTANCE', 0)}` `final={topic.get('FINAL_SCORE', 0)}`",
                    f"- Why it matters: {topic.get('WHY_IT_MATTERS', '').strip() or topic.get('summary', '').strip()}",
                    "",
                ]
            )
            takeaway_lines = [clean for clean in topic.get("KEY_TAKEAWAYS", []) if isinstance(clean, str) and clean.strip()]
            if takeaway_lines:
                lines.append("Key takeaways:")
                for takeaway in takeaway_lines:
                    lines.append(f"- {takeaway}")
                lines.append("")
            lines.append("Evidence:")
            for item in topic.get("items", [])[:4]:
                title = item.get("title", "Untitled")
                url = item.get("url", "")
                source = item.get("source_name", item.get("source_id", "source"))
                if url:
                    lines.append(f"- [{title}]({url}) ({source})")
                else:
                    lines.append(f"- {title} ({source})")
            lines.append("")

    watchlist = report.get("watchlist") or []
    if watchlist:
        lines.extend(["## Watchlist", ""])
        for topic in watchlist:
            lines.append(f"- **{topic.get('title', 'Untitled Topic')}**: {topic.get('WHY_IT_MATTERS', '').strip() or topic.get('summary', '').strip()}")
    return "\n".join(lines).rstrip() + "\n"


def fetch_source_payloads(target_date: datetime, output_root: Path, config: configparser.ConfigParser, force: bool) -> tuple[list[HotspotItem], dict[str, int]]:
    hotspot_sources = config["HOTSPOT_SOURCES"]
    hotspot_config = config["HOTSPOTS"]
    freshness_hours = hotspot_config.getint("freshness_hours", fallback=36)
    registry_path = REPO_ROOT / hotspot_config.get("source_registry_path", "configs/hotspot_roundup_sites.json")
    reuse_cached_raw = hotspot_sources.getboolean("reuse_cached_raw", fallback=True)
    local_papers_max_staleness_days = hotspot_sources.getint("local_papers_max_staleness_days", fallback=2)
    hf_result_limit = hotspot_sources.getint("hf_result_limit", fallback=24)

    specs = []
    if hotspot_sources.getboolean("use_local_papers", fallback=True):
        specs.append(("local_papers", lambda: fetch_local_paper_items(target_date, output_root, max_staleness_days=local_papers_max_staleness_days)))
    if hotspot_sources.getboolean("use_hf_papers", fallback=True):
        specs.append(("hf_papers", lambda: fetch_hf_items(target_date, freshness_hours, result_limit=hf_result_limit)))
    if hotspot_sources.getboolean("use_ainews", fallback=True):
        specs.append(("ainews", lambda: fetch_ainews_items(target_date, freshness_hours)))
    if hotspot_sources.getboolean("use_official_blogs", fallback=True):
        specs.append(("official_blogs", lambda: fetch_official_blog_items(target_date, freshness_hours)))
    if hotspot_sources.getboolean("use_roundup_sites", fallback=True):
        specs.append(("roundup_sites", lambda: fetch_roundup_items(target_date, freshness_hours, registry_path)))

    all_items: list[HotspotItem] = []
    source_stats: dict[str, int] = {}

    for source_id, fetch_fn in specs:
        cache_path = _raw_source_cache_path(output_root, target_date, source_id)
        if reuse_cached_raw and cache_path.exists() and not force:
            items = load_cached_items(cache_path)
        else:
            try:
                items = fetch_fn()
            except Exception as ex:
                print(f"Warning: failed to fetch hotspot source {source_id}: {ex}")
                items = []
            write_json(cache_path, _serialize_items(items))

        items = _dedupe_items(items)
        source_stats[source_id] = len(items)
        all_items.extend(items)

    return _dedupe_items(all_items), source_stats


def deterministic_trim(clusters: list[HotspotCluster], max_clusters: int) -> list[HotspotCluster]:
    ranked = sorted(clusters, key=lambda cluster: cluster.deterministic_score, reverse=True)
    role_budgets = {
        "research_backbone": 3,
        "paper_trending": 5,
        "official_news": 3,
        "headline_consensus": 3,
        "editorial_depth": 2,
        "community_heat": 2,
        "builder_momentum": 1,
    }
    selected: list[HotspotCluster] = []
    selected_ids: set[str] = set()

    for role, budget in role_budgets.items():
        kept = 0
        for cluster in ranked:
            if cluster.cluster_id in selected_ids:
                continue
            if role not in cluster.source_roles:
                continue
            selected.append(cluster)
            selected_ids.add(cluster.cluster_id)
            kept += 1
            if kept >= budget or len(selected) >= max_clusters:
                break
        if len(selected) >= max_clusters:
            break

    if len(selected) < max_clusters:
        for cluster in ranked:
            if cluster.cluster_id in selected_ids:
                continue
            selected.append(cluster)
            selected_ids.add(cluster.cluster_id)
            if len(selected) >= max_clusters:
                break

    return selected[:max_clusters]


def _fallback_digest_summary(top_topics: list[dict[str, Any]]) -> str:
    if not top_topics:
        return "No strong AI hotspots cleared the selection threshold today."
    headlines = [topic.get("HEADLINE") or topic.get("title") for topic in top_topics[:3]]
    if len(headlines) == 1:
        return f"Today's strongest AI hotspot was {headlines[0]}."
    if len(headlines) == 2:
        return f"Today's AI discussion centered on {headlines[0]} and {headlines[1]}."
    return f"Today's AI discussion centered on {headlines[0]}, {headlines[1]}, and {headlines[2]}."


def apply_digest_synthesis(top_topics: list[dict[str, Any]], watchlist: list[dict[str, Any]], system_prompt: str, digest_prompt: str, config: configparser.ConfigParser, mode: str) -> tuple[str, float, float]:
    for topic in top_topics:
        topic["HEADLINE"] = topic.get("title", "")
        topic["KEY_TAKEAWAYS"] = [topic.get("WHY_IT_MATTERS", "")]

    if mode != "openai" or not top_topics:
        return _fallback_digest_summary(top_topics), 0.0, 0.0

    try:
        payload, prompt_cost, completion_cost = synthesize_digest_with_openai(
            top_topics=top_topics,
            watchlist=watchlist,
            system_prompt=system_prompt,
            digest_prompt=digest_prompt,
            model=config["HOTSPOTS"].get("model_summarize", config["HOTSPOTS"].get("model_screen")),
            retry_count=config["HOTSPOTS"].getint("retry", fallback=3),
        )
    except Exception as ex:
        print(f"Warning: hotspot digest synthesis failed, falling back to heuristic: {ex}")
        return _fallback_digest_summary(top_topics), 0.0, 0.0

    by_id = {topic["TOPIC_ID"]: topic for topic in top_topics}
    for row in payload.get("top_topics", []):
        topic_id = row.get("TOPIC_ID")
        if topic_id not in by_id:
            continue
        by_id[topic_id]["HEADLINE"] = row.get("HEADLINE") or by_id[topic_id]["HEADLINE"]
        by_id[topic_id]["WHY_IT_MATTERS"] = row.get("WHY_IT_MATTERS") or by_id[topic_id]["WHY_IT_MATTERS"]
        by_id[topic_id]["KEY_TAKEAWAYS"] = [point for point in row.get("KEY_TAKEAWAYS", []) if point]

    return payload.get("summary", "").strip() or _fallback_digest_summary(top_topics), prompt_cost, completion_cost


def _decide_mode(requested_mode: str) -> str:
    has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))
    if requested_mode == "heuristic":
        return "heuristic"
    if requested_mode == "openai":
        return "openai" if has_openai_key else "heuristic"
    return "openai" if has_openai_key else "heuristic"


def _topic_bucket(topic: dict[str, Any]) -> str:
    roles = set(topic.get("source_roles", []))
    category = topic.get("PRIMARY_CATEGORY", "")
    if "official_news" in roles or category in {"Product Release", "Industry Update"}:
        return "official"
    if category == "Tooling" or roles & {"builder_momentum", "github_trend"}:
        return "tooling"
    if roles & {"research_backbone", "paper_trending"}:
        return "research"
    if roles & {"community_heat", "headline_consensus", "hn_discussion", "editorial_depth"}:
        return "community"
    return "research"


def _source_signature(topic: dict[str, Any]) -> str:
    names = sorted(topic.get("source_names", []))
    if names:
        return "|".join(names[:2])
    roles = sorted(topic.get("source_roles", []))
    return "|".join(roles[:2]) if roles else "unknown"


def _is_paper_heavy(topic: dict[str, Any]) -> bool:
    roles = set(topic.get("source_roles", []))
    return bool(roles) and roles.issubset({"research_backbone", "paper_trending"})


def _topic_sort_key(topic: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        float(topic.get("FINAL_SCORE", 0.0)),
        float(topic.get("EVIDENCE_STRENGTH", 0.0)),
        float(topic.get("CROSS_SOURCE_RESONANCE", 0.0)),
        float(topic.get("QUALITY", 0.0)),
        float(topic.get("HEAT", 0.0)),
    )


def _diverse_select(candidates: list[dict[str, Any]], limit: int, max_per_category: int = 3, max_per_source: int = 2) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    category_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    bucket_counts: dict[str, int] = {}
    paper_heavy_count = 0
    bucket_caps = {"research": 3, "official": 2, "community": 2, "tooling": 2}

    for topic in sorted(candidates, key=_topic_sort_key, reverse=True):
        category = topic.get("PRIMARY_CATEGORY", "Unknown")
        source = _source_signature(topic)
        bucket = _topic_bucket(topic)
        if category_counts.get(category, 0) >= max_per_category:
            continue
        if source_counts.get(source, 0) >= max_per_source:
            continue
        if bucket_counts.get(bucket, 0) >= bucket_caps.get(bucket, 2):
            continue
        if _is_paper_heavy(topic) and paper_heavy_count >= 2:
            continue
        selected.append(topic)
        category_counts[category] = category_counts.get(category, 0) + 1
        source_counts[source] = source_counts.get(source, 0) + 1
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        if _is_paper_heavy(topic):
            paper_heavy_count += 1
        if len(selected) >= limit:
            return selected
    return selected


def _trim_topics(top_topics: list[dict[str, Any]], watchlist: list[dict[str, Any]], config: configparser.ConfigParser) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target_topics = config["HOTSPOTS"].getint("target_topics", fallback=5)
    min_topics = config["HOTSPOTS"].getint("min_topics", fallback=3)
    target_watchlist_topics = config["HOTSPOTS"].getint("target_watchlist_topics", fallback=3)
    ranked_top = sorted(top_topics, key=_topic_sort_key, reverse=True)
    ranked_watchlist = sorted(watchlist, key=_topic_sort_key, reverse=True)
    combined = ranked_top + ranked_watchlist

    selected_top: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add_topic(topic: dict[str, Any], promoted: bool = False) -> None:
        if topic["TOPIC_ID"] in selected_ids:
            return
        selected_ids.add(topic["TOPIC_ID"])
        if promoted:
            selected_top.append({**topic, "KEEP_IN_DAILY_HOTSPOTS": True, "WATCHLIST": False, "PROMOTED_FROM_WATCHLIST": True})
        else:
            selected_top.append(topic)

    reserve_predicates = [
        lambda topic: _topic_bucket(topic) == "official",
        lambda topic: _topic_bucket(topic) == "community",
        lambda topic: _topic_bucket(topic) == "research",
    ]
    for predicate in reserve_predicates:
        for topic in combined:
            if predicate(topic):
                candidate = _diverse_select(selected_top + [topic], len(selected_top) + 1)
                if any(row["TOPIC_ID"] == topic["TOPIC_ID"] for row in candidate):
                    add_topic(topic, promoted=topic.get("WATCHLIST", False))
                    break

    fill_pool = ranked_top + ranked_watchlist
    for topic in fill_pool:
        if len(selected_top) >= target_topics:
            break
        if topic["TOPIC_ID"] in selected_ids:
            continue
        candidate = _diverse_select(selected_top + [topic], len(selected_top) + 1)
        if any(row["TOPIC_ID"] == topic["TOPIC_ID"] for row in candidate):
            add_topic(topic, promoted=topic.get("WATCHLIST", False))

    has_heat_slot = any(
        ("community_heat" in topic.get("source_roles", []) or "headline_consensus" in topic.get("source_roles", []) or "hn_discussion" in topic.get("source_roles", []))
        and "paper_trending" not in topic.get("source_roles", [])
        and "research_backbone" not in topic.get("source_roles", [])
        for topic in selected_top
    )
    if not has_heat_slot and len(selected_top) < target_topics:
        for topic in ranked_watchlist:
            if topic["TOPIC_ID"] in selected_ids:
                continue
            roles = set(topic.get("source_roles", []))
            if roles & {"community_heat", "headline_consensus", "hn_discussion"}:
                add_topic(topic, promoted=True)
                break

    if len(selected_top) < min_topics:
        fallback_pool = [topic for topic in combined if topic["TOPIC_ID"] not in selected_ids]
        for topic in fallback_pool:
            add_topic(topic, promoted=topic.get("WATCHLIST", False))
            if len(selected_top) >= min_topics:
                break

    remaining_watchlist = [topic for topic in ranked_watchlist if topic["TOPIC_ID"] not in selected_ids]
    return selected_top[:target_topics], _diverse_select(remaining_watchlist, target_watchlist_topics, max_per_category=1, max_per_source=1)


def generate_daily_hotspot_report(output_root: str | Path, target_date: datetime, config: configparser.ConfigParser, mode_override: str = "auto", force: bool = False) -> dict[str, Any] | None:
    if not config["HOTSPOTS"].getboolean("enabled", fallback=False):
        return None

    output_root = Path(output_root)
    effective_mode = _decide_mode(mode_override or config["HOTSPOTS"].get("mode", "auto"))

    raw_items, source_stats = fetch_source_payloads(target_date, output_root, config, force)
    raw_items = raw_items[: config["HOTSPOTS"].getint("max_raw_items", fallback=120)]

    clusters = build_hotspot_clusters(raw_items)
    candidate_clusters = deterministic_trim(clusters, config["HOTSPOTS"].getint("max_clusters_for_llm", fallback=24))

    system_prompt = read_prompt(REPO_ROOT / "prompts" / "hotspot_system_prompt.txt")
    criteria_prompt = read_prompt(REPO_ROOT / "prompts" / "hotspot_screening_criteria.txt")
    postfix_prompt = read_prompt(REPO_ROOT / "prompts" / "postfix_prompt_hotspot_screening.txt")
    digest_prompt = read_prompt(REPO_ROOT / "prompts" / "hotspot_digest_writer.txt")

    score_cutoff = config["HOTSPOTS"].getfloat("screening_score_cutoff", fallback=6.0)
    watchlist_cutoff = config["HOTSPOTS"].getfloat("watchlist_score_cutoff", fallback=4.9)

    prompt_cost = 0.0
    completion_cost = 0.0
    if effective_mode == "openai" and candidate_clusters:
        kept, watchlist, prompt_cost, completion_cost = screen_clusters_with_openai(
            clusters=candidate_clusters,
            system_prompt=system_prompt,
            criteria_prompt=criteria_prompt,
            postfix_prompt=postfix_prompt,
            model=config["HOTSPOTS"].get("model_screen"),
            batch_size=config["HOTSPOTS"].getint("screen_batch_size", fallback=8),
            retry_count=config["HOTSPOTS"].getint("retry", fallback=3),
            score_cutoff=score_cutoff,
            watchlist_cutoff=watchlist_cutoff,
        )
    else:
        kept, watchlist = heuristic_screen_clusters(candidate_clusters, score_cutoff, watchlist_cutoff)

    kept, watchlist = _trim_topics(kept, watchlist, config)
    summary, digest_prompt_cost, digest_completion_cost = apply_digest_synthesis(kept, watchlist, system_prompt, digest_prompt, config, effective_mode)
    prompt_cost += digest_prompt_cost
    completion_cost += digest_completion_cost

    report = {
        "date": date_string(target_date),
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": effective_mode,
        "summary": summary,
        "source_stats": source_stats,
        "totals": {"raw_items": len(raw_items), "clusters": len(clusters), "candidate_clusters": len(candidate_clusters)},
        "costs": {"prompt": round(prompt_cost, 6), "completion": round(completion_cost, 6), "total": round(prompt_cost + completion_cost, 6)},
        "top_topics": kept,
        "watchlist": watchlist,
    }

    hot_root = output_root / "hot"
    write_json(hot_root / "normalized" / f"{date_string(target_date)}.json", _serialize_items(raw_items))
    write_json(hot_root / "clusters" / f"{date_string(target_date)}.json", [cluster.to_dict() for cluster in clusters])
    write_json(hot_root / "reports" / f"{date_string(target_date)}.json", report)
    md_path = hot_root / "md" / month_string(target_date) / f"{date_string(target_date)}-hotspots.md"
    ensure_dir(md_path.parent)
    md_path.write_text(render_hot_daily_md(report), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    config = load_config(REPO_ROOT / "configs" / "config.ini")
    report = generate_daily_hotspot_report(
        output_root=args.output_root,
        target_date=parse_target_datetime(args.target_date),
        config=config,
        mode_override=args.mode,
        force=args.force,
    )
    if report is None:
        print("Hotspots disabled.")
        return
    print(Path(args.output_root) / "hot" / "reports" / f"{report['date']}.json")


if __name__ == "__main__":
    main()
