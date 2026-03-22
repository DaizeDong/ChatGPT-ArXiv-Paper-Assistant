from __future__ import annotations

import argparse
import configparser
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arxiv_assistant.apis.hotspot_ainews import fetch_hotspot_items as fetch_ainews_items
from arxiv_assistant.apis.hotspot_github import fetch_hotspot_items as fetch_github_items
from arxiv_assistant.apis.hotspot_hf_papers import fetch_hotspot_items as fetch_hf_items
from arxiv_assistant.apis.hotspot_hn import fetch_hotspot_items as fetch_hn_items
from arxiv_assistant.apis.hotspot_local_papers import fetch_hotspot_items as fetch_local_paper_items
from arxiv_assistant.apis.hotspot_official_blogs import fetch_hotspot_items as fetch_official_blog_items
from arxiv_assistant.apis.hotspot_roundups import fetch_hotspot_items as fetch_roundup_items
from arxiv_assistant.filters.filter_hotspots import build_candidate_topics, heuristic_screen_clusters, screen_clusters_with_openai, synthesize_digest_with_openai
from arxiv_assistant.renderers.render_hot_daily import render_hot_daily_md
from arxiv_assistant.utils.hotspot_cluster import build_hotspot_clusters
from arxiv_assistant.utils.hotspot_schema import HotspotCluster, HotspotItem
from arxiv_assistant.utils.hotspot_web_data import write_hotspot_web_data

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
DAY_OUTPUT_PATTERN = re.compile(r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})-output\.json$")
SOCIAL_PRIMARY_ROLES = {"community_heat"}
SOCIAL_BACKFILL_ROLES = {"headline_consensus", "builder_momentum", "hn_discussion"}
SOCIAL_HOST_SNIPPETS = ("x.com", "twitter.com", "reddit.com", "news.ycombinator.com")
CATEGORY_DISPLAY_ORDER = [
    "Product Release",
    "Tooling",
    "Industry Update",
    "Research",
    "Community Signal",
]
GENERIC_DISPLAY_TITLES = {
    "information collection notice",
    "privacy policy",
    "terms of service",
}
AI_RELEVANCE_TERMS = {
    "ai",
    "llm",
    "agent",
    "agents",
    "model",
    "models",
    "reasoning",
    "prompt",
    "retrieval",
    "inference",
    "training",
    "multimodal",
    "transformer",
    "moe",
    "quantization",
    "benchmark",
    "arxiv",
    "openai",
    "anthropic",
    "claude",
    "gpt",
    "gemini",
    "qwen",
    "deepseek",
    "cursor",
    "copilot",
}


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


def parse_target_datetime(target_date_arg: str | None, output_root: Path) -> datetime:
    if target_date_arg:
        return datetime.strptime(target_date_arg, "%Y-%m-%d").replace(tzinfo=APP_TIMEZONE)
    latest_local = detect_latest_local_output_date(output_root)
    if latest_local is not None:
        return latest_local
    now = datetime.now(APP_TIMEZONE)
    return datetime(now.year, now.month, now.day, tzinfo=APP_TIMEZONE)


def detect_latest_local_output_date(output_root: Path) -> datetime | None:
    json_root = output_root / "json"
    latest: tuple[int, int, int] | None = None
    if not json_root.exists():
        return None
    for file_path in json_root.glob("*/*-output.json"):
        match = DAY_OUTPUT_PATTERN.fullmatch(file_path.name)
        if match is None:
            continue
        date_key = (
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
        )
        if latest is None or date_key > latest:
            latest = date_key
    if latest is None:
        return None
    return datetime(latest[0], latest[1], latest[2], tzinfo=APP_TIMEZONE)


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


def _x_buzz_item_score(item: HotspotItem, linked_topic: str) -> float:
    score = 0.0
    if item.source_role in SOCIAL_PRIMARY_ROLES:
        score += 6.0
    elif item.source_role in SOCIAL_BACKFILL_ROLES:
        score += 3.5
    metadata = item.metadata or {}
    score += min(4.0, float(metadata.get("activity", 0) or 0) / 250.0)
    score += min(3.0, float(metadata.get("hn_score", 0) or 0) / 80.0)
    host = str(metadata.get("host", "") or item.url or "").lower()
    if any(snippet in host for snippet in SOCIAL_HOST_SNIPPETS):
        score += 1.2
    if linked_topic:
        score += 2.5
    return score


def _build_x_buzz_items(
    raw_items: list[HotspotItem],
    top_topics: list[dict[str, Any]],
    watchlist: list[dict[str, Any]],
    *,
    target_count: int = 5,
    min_count: int = 3,
) -> list[dict[str, Any]]:
    topic_lookup: dict[str, str] = {}
    for topic in top_topics + watchlist:
        topic_title = topic.get("HEADLINE") or topic.get("title") or ""
        for item in topic.get("items", []):
            url = str(item.get("url", "")).strip()
            title = str(item.get("title", "")).strip().lower()
            if url:
                topic_lookup[url] = topic_title
            if title:
                topic_lookup[title] = topic_title

    primary_candidates: list[tuple[float, dict[str, Any]]] = []
    backfill_candidates: list[tuple[float, dict[str, Any]]] = []
    seen_keys: set[tuple[str, str]] = set()

    for item in raw_items:
        if item.source_role not in SOCIAL_PRIMARY_ROLES | SOCIAL_BACKFILL_ROLES:
            continue
        url_key = item.url.strip()
        title_key = item.title.strip().lower()
        dedupe_key = (url_key, title_key)
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        linked_topic = topic_lookup.get(url_key) or topic_lookup.get(title_key, "")
        payload = {
            "title": item.title,
            "summary": item.summary,
            "url": item.url,
            "source_id": item.source_id,
            "source_name": item.source_name,
            "source_role": item.source_role,
            "linked_topic": linked_topic,
        }
        ranked = (_x_buzz_item_score(item, linked_topic), payload)
        if item.source_role in SOCIAL_PRIMARY_ROLES:
            primary_candidates.append(ranked)
        else:
            backfill_candidates.append(ranked)

    selected: list[dict[str, Any]] = []
    for _, payload in sorted(primary_candidates, key=lambda row: row[0], reverse=True):
        selected.append(payload)
        if len(selected) >= target_count:
            return selected

    if len(selected) < min_count:
        for _, payload in sorted(backfill_candidates, key=lambda row: row[0], reverse=True):
            if any(existing["url"] == payload["url"] and existing["title"] == payload["title"] for existing in selected):
                continue
            selected.append(payload)
            if len(selected) >= target_count:
                break

    return selected[:target_count]


def _topic_occurrence_score(topic: dict[str, Any]) -> float:
    source_ids = len(topic.get("source_ids", []))
    source_types = len(topic.get("source_types", []))
    item_count = len(topic.get("items", []))
    resonance = float(topic.get("CROSS_SOURCE_RESONANCE", 0.0))
    return round(
        min(
            10.0,
            1.2
            + source_ids * 1.3
            + max(source_types - 1, 0) * 0.9
            + min(2.2, item_count * 0.45)
            + 0.35 * resonance,
        ),
        3,
    )


def _llm_status(topic: dict[str, Any]) -> str:
    if topic.get("KEEP_IN_DAILY_HOTSPOTS"):
        return "featured"
    if topic.get("WATCHLIST"):
        return "watchlist"
    return "candidate"


def _display_priority(topic: dict[str, Any]) -> float:
    llm_boost = 0.0
    if topic.get("KEEP_IN_DAILY_HOTSPOTS"):
        llm_boost = 1.0
    elif topic.get("WATCHLIST"):
        llm_boost = 0.45
    paper_penalty = 0.0
    if _is_isolated_research_topic(topic):
        paper_penalty = 0.55
    elif _is_paper_heavy(topic):
        paper_penalty = 0.25
    return round(
        0.42 * float(topic.get("FINAL_SCORE", 0.0))
        + 0.24 * float(topic.get("OCCURRENCE_SCORE", 0.0))
        + 0.16 * float(topic.get("EVIDENCE_STRENGTH", 0.0))
        + 0.12 * float(topic.get("HEAT", 0.0))
        + 0.06 * float(topic.get("QUALITY", 0.0))
        + llm_boost
        - paper_penalty,
        3,
    )


def _merge_display_candidates(
    candidate_topics: list[dict[str, Any]],
    screened_top: list[dict[str, Any]],
    screened_watchlist: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    screened_lookup = {
        topic["TOPIC_ID"]: topic
        for topic in screened_top + screened_watchlist
    }
    merged: list[dict[str, Any]] = []
    for topic in candidate_topics:
        combined = dict(topic)
        screened = screened_lookup.get(topic["TOPIC_ID"])
        if screened is not None:
            combined.update(screened)
        combined["OCCURRENCE_SCORE"] = _topic_occurrence_score(combined)
        combined["LLM_STATUS"] = _llm_status(combined)
        combined["DISPLAY_PRIORITY"] = _display_priority(combined)
        merged.append(combined)
    merged.sort(
        key=lambda row: (
            row["DISPLAY_PRIORITY"],
            row["FINAL_SCORE"],
            row["OCCURRENCE_SCORE"],
            row["EVIDENCE_STRENGTH"],
            row["HEAT"],
        ),
        reverse=True,
    )
    return merged


def _build_category_sections(
    display_candidates: list[dict[str, Any]],
    featured_topics: list[dict[str, Any]],
    *,
    target_total_topics: int,
    max_per_category: int,
    min_display_score: float,
) -> list[dict[str, Any]]:
    featured_ids = {topic["TOPIC_ID"] for topic in featured_topics}
    grouped: dict[str, list[dict[str, Any]]] = {category: [] for category in CATEGORY_DISPLAY_ORDER}

    for topic in display_candidates:
        title = (topic.get("HEADLINE") or topic.get("title") or "").strip().lower()
        text = " ".join(
            [
                title,
                str(topic.get("summary", "") or "").lower(),
                str(topic.get("SHORT_COMMENT", "") or "").lower(),
                " ".join(str(tag).lower() for tag in topic.get("tags", [])),
            ]
        )
        if topic["TOPIC_ID"] in featured_ids:
            continue
        if float(topic.get("DISPLAY_PRIORITY", 0.0)) < min_display_score:
            continue
        if title in GENERIC_DISPLAY_TITLES:
            continue
        category = topic.get("PRIMARY_CATEGORY", "Community Signal")
        if category in {"Community Signal", "Industry Update", "Product Release"} and not any(term in text for term in AI_RELEVANCE_TERMS):
            continue
        grouped.setdefault(category, []).append(topic)

    selected_ids: set[str] = set()
    section_topics: dict[str, list[dict[str, Any]]] = {}
    total_selected = 0

    for category in CATEGORY_DISPLAY_ORDER:
        section_topics[category] = []
        for topic in grouped.get(category, []):
            if total_selected >= target_total_topics:
                break
            if len(section_topics[category]) >= max_per_category:
                break
            if topic["TOPIC_ID"] in selected_ids:
                continue
            section_topics[category].append(topic)
            selected_ids.add(topic["TOPIC_ID"])
            total_selected += 1

    sections: list[dict[str, Any]] = []
    for category in CATEGORY_DISPLAY_ORDER:
        displayed = section_topics.get(category, [])
        if not displayed:
            continue
        total_candidates = max(len(grouped.get(category, [])), len(displayed))
        sections.append(
            {
                "category": category,
                "total_candidates": total_candidates,
                "topics": displayed,
            }
        )
    return sections


def _build_long_tail_sections(
    display_candidates: list[dict[str, Any]],
    featured_topics: list[dict[str, Any]],
    category_sections: list[dict[str, Any]],
    *,
    target_total_topics: int,
    max_per_category: int,
    min_display_score: float,
) -> list[dict[str, Any]]:
    excluded_ids = {topic["TOPIC_ID"] for topic in featured_topics}
    for section in category_sections:
        excluded_ids.update(topic["TOPIC_ID"] for topic in section.get("topics", []))

    grouped: dict[str, list[dict[str, Any]]] = {category: [] for category in CATEGORY_DISPLAY_ORDER}
    for topic in display_candidates:
        title = (topic.get("HEADLINE") or topic.get("title") or "").strip().lower()
        text = " ".join(
            [
                title,
                str(topic.get("summary", "") or "").lower(),
                str(topic.get("SHORT_COMMENT", "") or "").lower(),
                " ".join(str(tag).lower() for tag in topic.get("tags", [])),
            ]
        )
        if topic["TOPIC_ID"] in excluded_ids:
            continue
        if float(topic.get("DISPLAY_PRIORITY", 0.0)) < min_display_score:
            continue
        if title in GENERIC_DISPLAY_TITLES:
            continue
        category = topic.get("PRIMARY_CATEGORY", "Community Signal")
        if category in {"Community Signal", "Industry Update", "Product Release"} and not any(term in text for term in AI_RELEVANCE_TERMS):
            continue
        grouped.setdefault(category, []).append(topic)

    sections: list[dict[str, Any]] = []
    total_selected = 0
    for category in CATEGORY_DISPLAY_ORDER:
        if total_selected >= target_total_topics:
            break
        available = grouped.get(category, [])
        if not available:
            continue
        selected = available[:max_per_category]
        if total_selected + len(selected) > target_total_topics:
            selected = selected[: target_total_topics - total_selected]
        if not selected:
            continue
        sections.append(
            {
                "category": category,
                "total_candidates": len(available),
                "topics": selected,
            }
        )
        total_selected += len(selected)
    return sections


def fetch_source_payloads(target_date: datetime, output_root: Path, config: configparser.ConfigParser, force: bool) -> tuple[list[HotspotItem], dict[str, int]]:
    hotspot_sources = config["HOTSPOT_SOURCES"]
    hotspot_config = config["HOTSPOTS"]
    freshness_hours = hotspot_config.getint("freshness_hours", fallback=36)
    github_config = config["HOTSPOT_GITHUB"]
    hn_config = config["HOTSPOT_HN"]
    registry_path = REPO_ROOT / hotspot_config.get("source_registry_path", "configs/hotspot_roundup_sites.json")
    reuse_cached_raw = hotspot_sources.getboolean("reuse_cached_raw", fallback=True)
    local_papers_max_staleness_days = hotspot_sources.getint("local_papers_max_staleness_days", fallback=2)
    hf_result_limit = hotspot_sources.getint("hf_result_limit", fallback=24)
    github_queries = [query.strip() for query in github_config.get("search_queries", "").split(",") if query.strip()]
    hn_keywords = [keyword.strip().lower() for keyword in hn_config.get("keyword_filter", "").split(",") if keyword.strip()]

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
    if hotspot_sources.getboolean("use_github", fallback=False):
        specs.append(
            (
                "github_trend",
                lambda: fetch_github_items(
                    target_date=target_date,
                    search_queries=github_queries,
                    stars_cutoff=github_config.getint("stars_cutoff", fallback=20),
                    created_within_days=github_config.getint("created_within_days", fallback=10),
                    result_limit=github_config.getint("result_limit", fallback=30),
                ),
            )
        )
    if hotspot_sources.getboolean("use_hn", fallback=False):
        specs.append(
            (
                "hn_discussion",
                lambda: fetch_hn_items(
                    target_date=target_date,
                    freshness_hours=freshness_hours,
                    keyword_filter=hn_keywords,
                    story_limit=hn_config.getint("story_limit", fallback=80),
                    score_cutoff=hn_config.getint("score_cutoff", fallback=30),
                    comments_cutoff=hn_config.getint("comments_cutoff", fallback=8),
                ),
            )
        )

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
        "github_trend": 2,
        "hn_discussion": 2,
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


def _is_isolated_research_topic(topic: dict[str, Any]) -> bool:
    roles = set(topic.get("source_roles", []))
    return (
        bool(roles)
        and roles.issubset({"research_backbone", "paper_trending"})
        and len(topic.get("source_names", [])) <= 1
        and len(topic.get("source_ids", [])) <= 1
    )


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
    isolated_research_count = 0
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
        if _is_isolated_research_topic(topic) and isolated_research_count >= 1:
            continue
        selected.append(topic)
        category_counts[category] = category_counts.get(category, 0) + 1
        source_counts[source] = source_counts.get(source, 0) + 1
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        if _is_paper_heavy(topic):
            paper_heavy_count += 1
        if _is_isolated_research_topic(topic):
            isolated_research_count += 1
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
    radar_clusters = sorted(clusters, key=lambda cluster: cluster.deterministic_score, reverse=True)[
        : config["HOTSPOTS"].getint("max_clusters_for_radar", fallback=48)
    ]
    candidate_topics = build_candidate_topics(radar_clusters)

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

    screened_kept = list(kept)
    screened_watchlist = list(watchlist)
    featured_topics, watchlist = _trim_topics(kept, watchlist, config)
    display_candidates = _merge_display_candidates(candidate_topics, screened_kept, screened_watchlist)
    display_lookup = {topic["TOPIC_ID"]: topic for topic in display_candidates}
    featured_topics = [{**display_lookup.get(topic["TOPIC_ID"], {}), **topic} for topic in featured_topics]
    watchlist = [{**display_lookup.get(topic["TOPIC_ID"], {}), **topic} for topic in watchlist]
    category_sections = _build_category_sections(
        display_candidates,
        featured_topics,
        target_total_topics=config["HOTSPOTS"].getint("target_category_topics", fallback=12),
        max_per_category=config["HOTSPOTS"].getint("max_topics_per_category", fallback=4),
        min_display_score=config["HOTSPOTS"].getfloat("category_display_score_cutoff", fallback=max(2.8, watchlist_cutoff - 0.5)),
    )
    long_tail_sections = _build_long_tail_sections(
        display_candidates,
        featured_topics,
        category_sections,
        target_total_topics=config["HOTSPOTS"].getint("target_long_tail_topics", fallback=18),
        max_per_category=config["HOTSPOTS"].getint("max_long_tail_per_category", fallback=8),
        min_display_score=config["HOTSPOTS"].getfloat("long_tail_display_score_cutoff", fallback=1.6),
    )
    summary, digest_prompt_cost, digest_completion_cost = apply_digest_synthesis(featured_topics, watchlist, system_prompt, digest_prompt, config, effective_mode)
    prompt_cost += digest_prompt_cost
    completion_cost += digest_completion_cost
    x_buzz = _build_x_buzz_items(raw_items, featured_topics, watchlist)

    report = {
        "date": date_string(target_date),
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": effective_mode,
        "summary": summary,
        "source_stats": source_stats,
        "totals": {
            "raw_items": len(raw_items),
            "clusters": len(clusters),
            "candidate_clusters": len(candidate_clusters),
            "radar_clusters": len(radar_clusters),
        },
        "costs": {"prompt": round(prompt_cost, 6), "completion": round(completion_cost, 6), "total": round(prompt_cost + completion_cost, 6)},
        "top_topics": featured_topics,
        "featured_topics": featured_topics,
        "category_sections": category_sections,
        "long_tail_sections": long_tail_sections,
        "x_buzz": x_buzz,
        "watchlist": watchlist,
    }

    hot_root = output_root / "hot"
    write_json(hot_root / "normalized" / f"{date_string(target_date)}.json", _serialize_items(raw_items))
    write_json(hot_root / "clusters" / f"{date_string(target_date)}.json", [cluster.to_dict() for cluster in clusters])
    write_json(hot_root / "reports" / f"{date_string(target_date)}.json", report)
    write_hotspot_web_data(output_root, report, raw_items)
    md_path = hot_root / "md" / month_string(target_date) / f"{date_string(target_date)}-hotspots.md"
    ensure_dir(md_path.parent)
    md_path.write_text(render_hot_daily_md(report), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    config = load_config(REPO_ROOT / "configs" / "config.ini")
    report = generate_daily_hotspot_report(
        output_root=args.output_root,
        target_date=parse_target_datetime(args.target_date, Path(args.output_root)),
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
