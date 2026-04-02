from __future__ import annotations

import configparser
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[2]

from arxiv_assistant.apis.hotspot.hotspot_ainews import fetch_hotspot_items as fetch_ainews_items
from arxiv_assistant.apis.hotspot.hotspot_analysis_feeds import fetch_hotspot_items as fetch_analysis_feed_items
from arxiv_assistant.apis.hotspot.hotspot_github import fetch_hotspot_items as fetch_github_items
from arxiv_assistant.apis.hotspot.hotspot_hf_papers import fetch_hotspot_items as fetch_hf_items
from arxiv_assistant.apis.hotspot.hotspot_hn import fetch_hotspot_items as fetch_hn_items
from arxiv_assistant.apis.hotspot.hotspot_local_papers import fetch_hotspot_items as fetch_local_paper_items
from arxiv_assistant.apis.hotspot.hotspot_official_blogs import fetch_hotspot_items as fetch_official_blog_items
from arxiv_assistant.apis.hotspot.hotspot_reddit import fetch_hotspot_items as fetch_reddit_items
from arxiv_assistant.apis.hotspot.hotspot_roundups import fetch_hotspot_items as fetch_roundup_items
from arxiv_assistant.apis.hotspot.hotspot_x_ainews import fetch_hotspot_items as fetch_x_ainews_items
from arxiv_assistant.apis.hotspot.hotspot_x_official import fetch_hotspot_items as fetch_x_official_items
from arxiv_assistant.apis.hotspot.hotspot_x_paperpulse import fetch_hotspot_items as fetch_x_paperpulse_items
from arxiv_assistant.filters.filter_hotspots import build_candidate_topics, heuristic_screen_clusters, screen_clusters_with_openai, synthesize_digest_with_openai
from arxiv_assistant.renderers.hotspot.render_hot_daily import render_hot_daily_md
from arxiv_assistant.utils.hotspot.hotspot_cluster import build_hotspot_clusters
from arxiv_assistant.utils.hotspot.hotspot_config import build_hotspot_paths, ensure_parent_dirs
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotCluster, HotspotItem
from arxiv_assistant.utils.hotspot.hotspot_sources import api_usage_scope, reset_api_usage, snapshot_api_usage
from arxiv_assistant.utils.prompt_loader import read_prompt
from arxiv_assistant.utils.hotspot.hotspot_web_data import write_hotspot_web_data

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
PAPER_SPOTLIGHT_LABELS = {
    "new_frontier": "New Frontier Papers",
    "daily_hot": "Daily Hot Papers",
}
PAPER_SPOTLIGHT_DESCRIPTIONS = {
    "new_frontier": "Papers that appear to open a genuinely new direction, paradigm, or field.",
    "daily_hot": "Papers that feel broadly important to the day and belong in the hotspot paper feed.",
}
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
SOURCE_USAGE_META = {
    "local_papers": {"provider": "Local cache", "billing_model": "local"},
    "hf_papers": {"provider": "Hugging Face", "billing_model": "free"},
    "ainews": {"provider": "AINews RSS", "billing_model": "free"},
    "official_blogs": {"provider": "Official blogs", "billing_model": "free"},
    "roundup_sites": {"provider": "Roundup sites", "billing_model": "free"},
    "analysis_feeds": {"provider": "Analysis RSS feeds", "billing_model": "free"},
    "reddit": {"provider": "Reddit JSON API", "billing_model": "free"},
    "x_ainews_twitter": {"provider": "AINews Twitter recap", "billing_model": "free"},
    "x_paperpulse": {"provider": "PaperPulse", "billing_model": "free"},
    "x_official": {"provider": "X API", "billing_model": "quota"},
    "github_trend": {"provider": "GitHub API", "billing_model": "free"},
    "hn_discussion": {"provider": "Hacker News API", "billing_model": "free"},
}


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
    score += min(1.6, float(metadata.get("source_quality", 0.0) or 0.0) * 1.1)
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


def _paper_spotlight_item_score(item: HotspotItem) -> float:
    metadata = item.metadata or {}
    score = 0.0
    score += float(metadata.get("daily_score", 0) or 0) * 0.8
    score += float(metadata.get("relevance", 0) or 0) * 0.35
    score += float(metadata.get("novelty", 0) or 0) * 0.35
    if metadata.get("spotlight_primary_kind") == "new_frontier":
        score += 1.0
    return round(score, 3)


def _build_paper_spotlight(
    raw_items: list[HotspotItem],
    *,
    max_daily_hot: int,
    max_new_frontier: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[float, dict[str, Any]]]] = {"new_frontier": [], "daily_hot": []}
    seen_ids: set[tuple[str, str]] = set()

    for item in raw_items:
        metadata = item.metadata or {}
        spotlight_kind = str(metadata.get("spotlight_primary_kind", "") or "").strip()
        if spotlight_kind not in grouped:
            continue
        arxiv_id = str(metadata.get("arxiv_id", "") or "").strip()
        dedupe_key = (spotlight_kind, arxiv_id or item.canonical_url or item.url)
        if dedupe_key in seen_ids:
            continue
        seen_ids.add(dedupe_key)
        payload = {
            "title": item.title,
            "summary": item.summary,
            "url": item.url,
            "canonical_url": item.canonical_url,
            "source_id": item.source_id,
            "source_name": item.source_name,
            "source_role": item.source_role,
            "arxiv_id": arxiv_id,
            "authors": list(item.authors),
            "primary_topic_id": metadata.get("primary_topic_id", ""),
            "primary_topic_label": metadata.get("primary_topic_label", ""),
            "spotlight_primary_kind": spotlight_kind,
            "spotlight_primary_label": metadata.get("spotlight_primary_label", PAPER_SPOTLIGHT_LABELS.get(spotlight_kind, "")),
            "spotlight_comment": metadata.get("spotlight_comment", ""),
            "daily_score": int(metadata.get("daily_score", 0) or 0),
            "relevance": int(metadata.get("relevance", 0) or 0),
            "novelty": int(metadata.get("novelty", 0) or 0),
        }
        grouped[spotlight_kind].append((_paper_spotlight_item_score(item), payload))

    sections: list[dict[str, Any]] = []
    for kind, limit in (("new_frontier", max_new_frontier), ("daily_hot", max_daily_hot)):
        ranked_items = [
            payload
            for _, payload in sorted(
                grouped[kind],
                key=lambda row: (
                    row[0],
                    row[1].get("novelty", 0),
                    row[1].get("daily_score", 0),
                    row[1].get("title", "").lower(),
                ),
                reverse=True,
            )
        ][:limit]
        if not ranked_items:
            continue
        sections.append(
            {
                "kind": kind,
                "label": PAPER_SPOTLIGHT_LABELS[kind],
                "description": PAPER_SPOTLIGHT_DESCRIPTIONS[kind],
                "items": ranked_items,
            }
        )
    return sections


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
    confidence = float(topic.get("CONFIDENCE", 0.0))
    paper_penalty = 0.0
    if _is_isolated_research_topic(topic):
        paper_penalty = 0.55
    elif _is_paper_heavy(topic):
        paper_penalty = 0.25
    low_confidence_penalty = max(0.0, 4.8 - confidence) * 0.35
    return round(
        0.42 * float(topic.get("FINAL_SCORE", 0.0))
        + 0.24 * float(topic.get("OCCURRENCE_SCORE", 0.0))
        + 0.16 * float(topic.get("EVIDENCE_STRENGTH", 0.0))
        + 0.10 * confidence
        + 0.12 * float(topic.get("HEAT", 0.0))
        + 0.04 * float(topic.get("QUALITY", 0.0))
        + llm_boost
        - paper_penalty
        - low_confidence_penalty,
        3,
    )


def _screening_decision(topic: dict[str, Any], score_cutoff: float, watchlist_cutoff: float, config: configparser.ConfigParser) -> str:
    hotspot_config = config["HOTSPOTS"]
    final_score = float(topic.get("FINAL_SCORE", 0.0))
    confidence = float(topic.get("CONFIDENCE", 0.0))
    evidence = float(topic.get("EVIDENCE_STRENGTH", 0.0))
    resonance = float(topic.get("CROSS_SOURCE_RESONANCE", 0.0))
    hype_penalty = float(topic.get("HYPE_PENALTY", 0.0))
    source_count = len(topic.get("source_names", []))
    roles = set(topic.get("source_roles", []))
    source_types = set(topic.get("source_types", []))
    has_official = "official_news" in roles
    has_non_paper_signal = bool(roles - {"research_backbone", "paper_trending"})
    has_repo_signal = "repo" in source_types
    has_discussion_signal = "discussion" in source_types
    has_paper_signal = "paper" in source_types
    weak_single_source_roles = (
        {"research_backbone", "paper_trending"},
        {"github_trend", "builder_momentum"},
        {"headline_consensus"},
        {"hn_discussion"},
    )

    auto_keep_score = hotspot_config.getfloat("screening_auto_keep_score_cutoff", fallback=6.8)
    auto_keep_conf = hotspot_config.getfloat("screening_auto_keep_confidence_cutoff", fallback=6.4)
    auto_keep_evidence = hotspot_config.getfloat("screening_auto_keep_evidence_cutoff", fallback=5.4)
    auto_watch_score = hotspot_config.getfloat("screening_auto_watchlist_score_cutoff", fallback=4.6)
    auto_watch_conf = hotspot_config.getfloat("screening_auto_watchlist_confidence_cutoff", fallback=5.4)
    auto_drop_score = hotspot_config.getfloat("screening_auto_drop_score_cutoff", fallback=2.8)
    auto_drop_conf = hotspot_config.getfloat("screening_auto_drop_confidence_cutoff", fallback=3.6)
    auto_drop_evidence = hotspot_config.getfloat("screening_auto_drop_evidence_cutoff", fallback=3.2)
    heuristic_only_conf = hotspot_config.getfloat("screening_heuristic_only_confidence_cutoff", fallback=5.0)
    heuristic_only_evidence = hotspot_config.getfloat("screening_heuristic_only_evidence_cutoff", fallback=3.4)
    heuristic_only_score = hotspot_config.getfloat("screening_heuristic_only_score_cutoff", fallback=5.8)
    strong_evidence_keep = confidence >= 8.0 and evidence >= 6.5 and resonance >= 5.5 and source_count >= 2

    if (
        source_count <= 1
        and not has_official
        and any(roles.issubset(role_group) for role_group in weak_single_source_roles)
    ):
        if confidence >= heuristic_only_conf and evidence >= heuristic_only_evidence and final_score >= heuristic_only_score:
            return "heuristic_only"
        return "auto_drop"

    strong_multi_source = source_count >= 2 or has_official or (has_non_paper_signal and resonance >= 4.8)
    if (strong_evidence_keep or final_score >= auto_keep_score) and confidence >= auto_keep_conf and evidence >= auto_keep_evidence and strong_multi_source and hype_penalty <= 5.2:
        return "auto_keep"
    auto_watch_upper_bound = max(watchlist_cutoff + 1.2, score_cutoff + 0.9)
    if auto_watch_score <= final_score <= auto_watch_upper_bound and confidence >= auto_watch_conf and evidence >= auto_keep_evidence - 0.6 and (source_count >= 2 or has_official or resonance >= 5.2):
        return "auto_watch"
    if final_score < auto_drop_score or (confidence < auto_drop_conf and evidence < auto_drop_evidence and resonance < 4.0 and source_count <= 1):
        return "auto_drop"
    if (
        source_count <= 1
        and not has_official
        and not has_paper_signal
        and (has_repo_signal or has_discussion_signal)
        and final_score <= heuristic_only_score
        and confidence <= heuristic_only_conf
        and evidence <= heuristic_only_evidence
    ):
        return "heuristic_only"
    return "review"


def _review_priority(topic: dict[str, Any], score_cutoff: float, watchlist_cutoff: float) -> float:
    final_score = float(topic.get("FINAL_SCORE", 0.0))
    confidence = float(topic.get("CONFIDENCE", 0.0))
    evidence = float(topic.get("EVIDENCE_STRENGTH", 0.0))
    resonance = float(topic.get("CROSS_SOURCE_RESONANCE", 0.0))
    uncertainty = min(abs(final_score - score_cutoff), abs(final_score - watchlist_cutoff))
    return round(
        max(0.0, 3.0 - uncertainty) * 1.7
        + evidence * 0.45
        + resonance * 0.35
        + max(0.0, 7.0 - confidence) * 0.4,
        3,
    )


def _screening_queue(
    candidate_topics: list[dict[str, Any]],
    score_cutoff: float,
    watchlist_cutoff: float,
    config: configparser.ConfigParser,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    auto_keep: list[dict[str, Any]] = []
    auto_watch: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    heuristic_only: list[dict[str, Any]] = []
    auto_drop = 0

    for topic in candidate_topics:
        decision = _screening_decision(topic, score_cutoff, watchlist_cutoff, config)
        if decision == "auto_keep":
            auto_keep.append({**topic, "KEEP_IN_DAILY_HOTSPOTS": True, "WATCHLIST": False, "SCREENING_DECISION": decision})
        elif decision == "auto_watch":
            auto_watch.append({**topic, "KEEP_IN_DAILY_HOTSPOTS": False, "WATCHLIST": True, "SCREENING_DECISION": decision})
        elif decision == "auto_drop":
            auto_drop += 1
        elif decision == "heuristic_only":
            heuristic_only.append({**topic, "SCREENING_DECISION": decision, "REVIEW_PRIORITY": _review_priority(topic, score_cutoff, watchlist_cutoff)})
        else:
            review.append({**topic, "SCREENING_DECISION": decision, "REVIEW_PRIORITY": _review_priority(topic, score_cutoff, watchlist_cutoff)})

    review.sort(
        key=lambda row: (
            row["REVIEW_PRIORITY"],
            row["CONFIDENCE"],
            row["EVIDENCE_STRENGTH"],
            row["FINAL_SCORE"],
        ),
        reverse=True,
    )
    heuristic_only.sort(
        key=lambda row: (
            row["DISPLAY_PRIORITY"] if "DISPLAY_PRIORITY" in row else _display_priority(row),
            row["CONFIDENCE"],
            row["EVIDENCE_STRENGTH"],
            row["FINAL_SCORE"],
        ),
        reverse=True,
    )
    return auto_keep, auto_watch, review, heuristic_only, {
        "auto_keep": len(auto_keep),
        "auto_watchlist": len(auto_watch),
        "heuristic_only": len(heuristic_only),
        "auto_drop": auto_drop,
        "review": len(review),
    }


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
            row.get("CONFIDENCE", 0.0),
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


def fetch_source_payloads(
    target_date: datetime,
    output_root: Path,
    config: configparser.ConfigParser,
    force: bool,
) -> tuple[list[HotspotItem], dict[str, int], dict[str, dict[str, Any]]]:
    hotspot_sources = config["HOTSPOT_SOURCES"]
    hotspot_config = config["HOTSPOTS"]
    freshness_hours = hotspot_config.getint("freshness_hours", fallback=36)
    github_config = config["HOTSPOT_GITHUB"]
    hn_config = config["HOTSPOT_HN"]
    x_config = config["HOTSPOT_X"] if config.has_section("HOTSPOT_X") else {}
    registry_path = REPO_ROOT / hotspot_config.get("source_registry_path", "configs/hotspot/roundup_sites.json")
    x_seed_path = REPO_ROOT / x_config.get("authority_seeds_path", "configs/hotspot/x_authority_seeds.json")
    x_registry_snapshot_path = REPO_ROOT / x_config.get("authority_inventory_path", "configs/hotspot/x_authority_inventory.json")
    x_registry_max_age_hours = int(x_config.get("authority_registry_max_age_hours", 24))
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
    official_blogs_registry = REPO_ROOT / "configs" / "hotspot" / "official_blogs.json"
    if hotspot_sources.getboolean("use_official_blogs", fallback=True):
        specs.append(("official_blogs", lambda: fetch_official_blog_items(
            target_date, freshness_hours,
            registry_path=str(official_blogs_registry) if official_blogs_registry.exists() else None,
        )))
    if hotspot_sources.getboolean("use_roundup_sites", fallback=True):
        specs.append(("roundup_sites", lambda: fetch_roundup_items(target_date, freshness_hours, registry_path)))
    analysis_feeds_registry = REPO_ROOT / "configs" / "hotspot" / "analysis_feeds.json"
    if hotspot_sources.getboolean("use_analysis_feeds", fallback=True) and analysis_feeds_registry.exists():
        specs.append(("analysis_feeds", lambda: fetch_analysis_feed_items(target_date, freshness_hours, analysis_feeds_registry)))
    if hotspot_sources.getboolean("use_reddit", fallback=True):
        specs.append(("reddit", lambda: fetch_reddit_items(target_date, freshness_hours)))
    if hotspot_sources.getboolean("use_x_ainews_twitter", fallback=True):
        specs.append(("x_ainews_twitter", lambda: fetch_x_ainews_items(target_date, freshness_hours)))
    if hotspot_sources.getboolean("use_x_paperpulse", fallback=True):
        specs.append(
            (
                "x_paperpulse",
                lambda: fetch_x_paperpulse_items(
                    target_date,
                    freshness_hours,
                    result_limit=int(x_config.get("paperpulse_result_limit", 20)),
                ),
            )
        )
    if hotspot_sources.getboolean("use_x_official", fallback=False):
        specs.append(
            (
                "x_official",
                lambda: fetch_x_official_items(
                    target_date,
                    freshness_hours,
                    x_seed_path,
                    default_result_limit=int(x_config.get("list_result_limit", 80)),
                    snapshot_path=x_registry_snapshot_path,
                    max_age_hours=x_registry_max_age_hours,
                    official_batch_size=int(x_config.get("official_batch_size", 10)),
                    researcher_batch_size=int(x_config.get("researcher_batch_size", 8)),
                ),
            )
        )
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
    api_usage: dict[str, dict[str, Any]] = {}
    reset_api_usage()

    for source_id, fetch_fn in specs:
        cache_path = _raw_source_cache_path(output_root, target_date, source_id)
        usage_meta = SOURCE_USAGE_META.get(source_id, {"provider": source_id, "billing_model": "unknown"})
        if reuse_cached_raw and cache_path.exists() and not force:
            items = load_cached_items(cache_path)
            api_usage[source_id] = {
                "provider": usage_meta["provider"],
                "billing_model": usage_meta["billing_model"],
                "requests": 0,
                "estimated_cost": 0.0,
                "cache_hit": True,
            }
        else:
            try:
                with api_usage_scope(source_id, usage_meta["provider"]):
                    items = fetch_fn()
            except Exception as ex:
                print(f"Warning: failed to fetch hotspot source {source_id}: {ex}")
                items = []
            write_json(cache_path, _serialize_items(items))
            usage_row = snapshot_api_usage().get(source_id, {"provider": usage_meta["provider"], "requests": 0, "estimated_cost": 0.0})
            api_usage[source_id] = {
                "provider": usage_meta["provider"],
                "billing_model": usage_meta["billing_model"],
                "requests": int(usage_row.get("requests", 0) or 0),
                "estimated_cost": round(float(usage_row.get("estimated_cost", 0.0) or 0.0), 6),
                "cache_hit": False,
            }

        items = _dedupe_items(items)
        source_stats[source_id] = len(items)
        api_usage[source_id]["items"] = len(items)
        all_items.extend(items)

    return _dedupe_items(all_items), source_stats, api_usage


def deterministic_trim(clusters: list[HotspotCluster], max_clusters: int) -> list[HotspotCluster]:
    def cluster_priority(cluster: HotspotCluster) -> tuple[float, int, int]:
        source_count = len(cluster.source_ids)
        source_type_count = len(cluster.source_types)
        roles = set(cluster.source_roles)
        boost = 0.0
        if "official_news" in roles:
            boost += 7.0
        if roles & {"community_heat", "headline_consensus", "editorial_depth"}:
            boost += 3.5
        if source_count > 1:
            boost += 2.2 * min(source_count - 1, 3)
        if source_type_count > 1:
            boost += 1.2 * min(source_type_count - 1, 2)
        if roles.issubset({"github_trend"}) and source_count == 1:
            boost -= 4.0
        if roles.issubset({"research_backbone", "paper_trending"}) and source_count == 1:
            boost -= 2.0
        if roles.issubset({"headline_consensus"}) and source_count == 1:
            boost -= 5.5
        if roles.issubset({"hn_discussion"}) and source_count == 1:
            boost -= 5.0
        if roles.issubset({"builder_momentum"}) and source_count == 1:
            boost -= 3.2
        if roles.issubset({"community_heat"}) and source_count == 1:
            boost -= 1.8
        return (cluster.deterministic_score + boost, source_count, source_type_count)

    ranked = sorted(clusters, key=cluster_priority, reverse=True)
    role_budgets = {
        "official_news": 4,
        "community_heat": 4,
        "headline_consensus": 4,
        "research_backbone": 2,
        "paper_trending": 4,
        "editorial_depth": 2,
        "builder_momentum": 1,
        "github_trend": 1,
        "hn_discussion": 1,
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


def _heuristic_takeaways(topic: dict[str, Any], max_takeaways: int = 3) -> list[str]:
    """Extract meaningful takeaways from a topic's items in heuristic mode."""
    title_lower = topic.get("title", "").lower().strip()
    title_tokens = set(title_lower.split())
    title_prefix = title_lower[:40]
    seen: set[str] = set()
    takeaways: list[str] = []
    for item in topic.get("items", []):
        summary = str(item.get("summary", "")).strip()
        if not summary or len(summary) < 20:
            continue
        summary_lower = summary.lower()
        item_title_lower = str(item.get("title", "")).lower().strip()
        # Skip if summary is essentially the item's own title
        if item_title_lower and (summary_lower.startswith(item_title_lower[:40]) or item_title_lower in summary_lower):
            continue
        # Skip if summary starts with the same prefix as the cluster title
        if title_prefix and summary_lower.startswith(title_prefix):
            continue
        # Skip if cluster title is a substring of the summary
        if title_lower and title_lower in summary_lower:
            continue
        # Skip if too similar to the cluster title by token overlap
        s_tokens = set(summary_lower.split())
        if title_tokens and len(s_tokens & title_tokens) / max(len(s_tokens), 1) > 0.7:
            continue
        # Skip near-duplicate takeaways
        s_key = " ".join(sorted(s_tokens)[:8])
        if s_key in seen:
            continue
        seen.add(s_key)
        takeaways.append(summary if len(summary) <= 200 else summary[:197] + "...")
        if len(takeaways) >= max_takeaways:
            break
    return takeaways


def apply_digest_synthesis(
    top_topics: list[dict[str, Any]],
    watchlist: list[dict[str, Any]],
    system_prompt: str,
    digest_prompt: str,
    config: configparser.ConfigParser,
    mode: str,
 ) -> tuple[str, float, float, int, int, int]:
    for topic in top_topics:
        topic["HEADLINE"] = topic.get("title", "")
        if not topic.get("KEY_TAKEAWAYS"):
            topic["KEY_TAKEAWAYS"] = _heuristic_takeaways(topic)

    if mode != "openai" or not top_topics:
        return _fallback_digest_summary(top_topics), 0.0, 0.0, 0, 0, 0

    try:
        payload, prompt_cost, completion_cost, prompt_tokens, completion_tokens, request_count = synthesize_digest_with_openai(
            top_topics=top_topics,
            watchlist=watchlist,
            system_prompt=system_prompt,
            digest_prompt=digest_prompt,
            model=config["HOTSPOTS"].get("model_summarize", config["HOTSPOTS"].get("model_screen")),
            retry_count=config["HOTSPOTS"].getint("retry", fallback=3),
        )
    except Exception as ex:
        print(f"Warning: hotspot digest synthesis failed, falling back to heuristic: {ex}")
        return _fallback_digest_summary(top_topics), 0.0, 0.0, 0, 0, 0

    by_id = {topic["TOPIC_ID"]: topic for topic in top_topics}
    for row in payload.get("top_topics", []):
        topic_id = row.get("TOPIC_ID")
        if topic_id not in by_id:
            continue
        by_id[topic_id]["HEADLINE"] = row.get("HEADLINE") or by_id[topic_id]["HEADLINE"]
        by_id[topic_id]["WHY_IT_MATTERS"] = row.get("WHY_IT_MATTERS") or by_id[topic_id]["WHY_IT_MATTERS"]
        by_id[topic_id]["KEY_TAKEAWAYS"] = [point for point in row.get("KEY_TAKEAWAYS", []) if point]

    return (
        payload.get("summary", "").strip() or _fallback_digest_summary(top_topics),
        prompt_cost,
        completion_cost,
        prompt_tokens,
        completion_tokens,
        request_count,
    )


def _build_usage_payload(
    config: configparser.ConfigParser,
    effective_mode: str,
    prompt_cost: float,
    completion_cost: float,
    llm_prompt_tokens: int,
    llm_completion_tokens: int,
    llm_requests: int,
    api_usage: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    llm_total_cost = round(prompt_cost + completion_cost, 6)
    llm_row = {
        "provider": "OpenAI",
        "billing_model": "quota" if effective_mode == "openai" else "disabled",
        "screen_model": config["HOTSPOTS"].get("model_screen"),
        "summary_model": config["HOTSPOTS"].get("model_summarize", config["HOTSPOTS"].get("model_screen")),
        "requests": int(llm_requests),
        "prompt_tokens": int(llm_prompt_tokens),
        "completion_tokens": int(llm_completion_tokens),
        "total_tokens": int(llm_prompt_tokens + llm_completion_tokens),
        "prompt_cost": round(prompt_cost, 6),
        "completion_cost": round(completion_cost, 6),
        "total_cost": llm_total_cost,
    }

    external_rows: dict[str, dict[str, Any]] = {}
    external_request_total = 0
    x_request_total = 0
    estimated_external_cost = 0.0
    for source_id, row in api_usage.items():
        requests_count = int(row.get("requests", 0) or 0)
        estimated_cost = round(float(row.get("estimated_cost", 0.0) or 0.0), 6)
        normalized = {
            "provider": str(row.get("provider", source_id)),
            "billing_model": str(row.get("billing_model", "unknown")),
            "requests": requests_count,
            "items": int(row.get("items", 0) or 0),
            "estimated_cost": estimated_cost,
            "cache_hit": bool(row.get("cache_hit", False)),
        }
        external_rows[source_id] = normalized
        external_request_total += requests_count
        estimated_external_cost += estimated_cost
        if source_id.startswith("x_"):
            x_request_total += requests_count

    return {
        "llm": llm_row,
        "external": external_rows,
        "summary": {
            "external_requests": external_request_total,
            "x_requests": x_request_total,
            "estimated_external_cost": round(estimated_external_cost, 6),
        },
    }


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
    display_priority = float(topic.get("DISPLAY_PRIORITY", _display_priority(topic)))
    demotion_penalty = 1.0 if topic.get("DEMOTED_LOW_CONFIDENCE") else 0.0
    return (
        display_priority - demotion_penalty,
        float(topic.get("CONFIDENCE", 0.0)),
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

    def should_demote_featured(topic: dict[str, Any]) -> bool:
        roles = set(topic.get("source_roles", []))
        source_count = len(topic.get("source_names", []))
        final_score = float(topic.get("FINAL_SCORE", 0.0))
        evidence = float(topic.get("EVIDENCE_STRENGTH", 0.0))
        confidence = float(topic.get("CONFIDENCE", max(final_score - 1.0, evidence, 0.0)))
        if source_count > 1:
            return False
        if "official_news" in roles:
            return confidence < 5.0 or evidence < 4.4 or final_score < 4.8
        if roles.issubset({"research_backbone", "paper_trending"}):
            return True
        if roles.issubset({"github_trend", "builder_momentum"}):
            return True
        if roles.issubset({"headline_consensus", "hn_discussion", "community_heat"}):
            return True
        return confidence < 4.7 and evidence < 4.0 and bool(roles & {"github_trend", "builder_momentum", "community_heat", "hn_discussion"})

    def is_quality_fill_candidate(topic: dict[str, Any]) -> bool:
        roles = set(topic.get("source_roles", []))
        source_count = len(topic.get("source_names", []))
        final_score = float(topic.get("FINAL_SCORE", 0.0))
        evidence = float(topic.get("EVIDENCE_STRENGTH", max(final_score - 1.2, 0.0)))
        confidence = float(topic.get("CONFIDENCE", max(final_score - 1.0, evidence, 0.0)))
        if source_count >= 2:
            return True
        if "official_news" in roles:
            return confidence >= 5.0 and evidence >= 4.4 and final_score >= 4.8
        return False

    demoted_watchlist: list[dict[str, Any]] = []
    retained_top: list[dict[str, Any]] = []
    for topic in top_topics:
        if should_demote_featured(topic):
            demoted_watchlist.append(
                {
                    **topic,
                    "KEEP_IN_DAILY_HOTSPOTS": False,
                    "WATCHLIST": True,
                    "DEMOTED_LOW_CONFIDENCE": True,
                    "POST_FILTER_DECISION": "demoted_low_confidence",
                }
            )
        else:
            retained_top.append(topic)

    ranked_top = sorted(retained_top, key=_topic_sort_key, reverse=True)
    ranked_watchlist = sorted(watchlist + demoted_watchlist, key=_topic_sort_key, reverse=True)
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

    reserve_min_score = config["HOTSPOTS"].getfloat("reserve_min_score", fallback=4.5)
    reserve_predicates = [
        lambda topic: _topic_bucket(topic) == "official",
        lambda topic: _topic_bucket(topic) == "community",
        lambda topic: _topic_bucket(topic) == "research",
    ]
    for predicate in reserve_predicates:
        for topic in combined:
            if topic.get("DEMOTED_LOW_CONFIDENCE"):
                continue
            if float(topic.get("FINAL_SCORE", 0.0)) < reserve_min_score:
                continue
            if predicate(topic):
                candidate = _diverse_select(selected_top + [topic], len(selected_top) + 1)
                if any(row["TOPIC_ID"] == topic["TOPIC_ID"] for row in candidate):
                    add_topic(topic, promoted=topic.get("WATCHLIST", False))
                    break

    regular_fill = [topic for topic in ranked_top + ranked_watchlist if not topic.get("DEMOTED_LOW_CONFIDENCE") and is_quality_fill_candidate(topic)]
    demoted_fill = [topic for topic in ranked_watchlist if topic.get("DEMOTED_LOW_CONFIDENCE") and is_quality_fill_candidate(topic)]
    fill_pool = sorted(regular_fill, key=_topic_sort_key, reverse=True)
    if len(selected_top) < min_topics:
        fill_pool += sorted(demoted_fill, key=_topic_sort_key, reverse=True)
    for topic in fill_pool:
        if len(selected_top) >= target_topics:
            break
        if topic.get("DEMOTED_LOW_CONFIDENCE") and len(selected_top) >= min_topics:
            continue
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
        fallback_pool = [topic for topic in combined if topic["TOPIC_ID"] not in selected_ids and is_quality_fill_candidate(topic)]
        for topic in fallback_pool:
            add_topic(topic, promoted=topic.get("WATCHLIST", False))
            if len(selected_top) >= min_topics:
                break

    selected_top.sort(
        key=lambda row: (
            float(row.get("DISPLAY_PRIORITY", 0.0)),
            float(row.get("CONFIDENCE", 0.0)),
            float(row.get("FINAL_SCORE", 0.0)),
            float(row.get("EVIDENCE_STRENGTH", 0.0)),
            float(row.get("CROSS_SOURCE_RESONANCE", 0.0)),
        ),
        reverse=True,
    )
    remaining_watchlist = [topic for topic in ranked_watchlist if topic["TOPIC_ID"] not in selected_ids]
    return selected_top[:target_topics], _diverse_select(remaining_watchlist, target_watchlist_topics, max_per_category=1, max_per_source=1)


def generate_daily_hotspot_report(output_root: str | Path, target_date: datetime, config: configparser.ConfigParser, mode_override: str = "auto", force: bool = False) -> dict[str, Any] | None:
    if not config["HOTSPOTS"].getboolean("enabled", fallback=False):
        return None

    output_root = Path(output_root)
    effective_mode = _decide_mode(mode_override or config["HOTSPOTS"].get("mode", "auto"))

    raw_items, source_stats, api_usage = fetch_source_payloads(target_date, output_root, config, force)
    raw_items = raw_items[: config["HOTSPOTS"].getint("max_raw_items", fallback=120)]

    clusters = build_hotspot_clusters(raw_items)
    candidate_clusters = deterministic_trim(clusters, config["HOTSPOTS"].getint("max_clusters_for_llm", fallback=24))
    radar_clusters = sorted(clusters, key=lambda cluster: cluster.deterministic_score, reverse=True)[
        : config["HOTSPOTS"].getint("max_clusters_for_radar", fallback=48)
    ]
    candidate_topics = build_candidate_topics(radar_clusters)

    system_prompt = read_prompt("hotspot.system_prompt")
    criteria_prompt = read_prompt("hotspot.screening_criteria")
    postfix_prompt = read_prompt("hotspot.postfix_screening")
    digest_prompt = read_prompt("hotspot.digest_writer")

    score_cutoff = config["HOTSPOTS"].getfloat("screening_score_cutoff", fallback=6.0)
    watchlist_cutoff = config["HOTSPOTS"].getfloat("watchlist_score_cutoff", fallback=4.9)
    candidate_screen_topics = build_candidate_topics(candidate_clusters)
    auto_keep_topics, auto_watch_topics, review_topics, heuristic_only_topics, screening_stats = _screening_queue(
        candidate_screen_topics,
        score_cutoff,
        watchlist_cutoff,
        config,
    )
    target_topics = config["HOTSPOTS"].getint("target_topics", fallback=5)
    target_watchlist_topics = config["HOTSPOTS"].getint("target_watchlist_topics", fallback=3)
    llm_review_budget = min(
        config["HOTSPOTS"].getint("max_review_clusters", fallback=10),
        max(6, target_topics + target_watchlist_topics + 1 - len(auto_keep_topics) - len(auto_watch_topics)),
    )
    review_topic_ids = {topic["TOPIC_ID"] for topic in review_topics[:llm_review_budget]}
    review_clusters = [cluster for cluster in candidate_clusters if cluster.cluster_id in review_topic_ids]
    heuristic_fallback_topics = heuristic_only_topics + [topic for topic in review_topics if topic["TOPIC_ID"] not in review_topic_ids]

    prompt_cost = 0.0
    completion_cost = 0.0
    llm_prompt_tokens = 0
    llm_completion_tokens = 0
    llm_requests = 0
    llm_kept: list[dict[str, Any]] = []
    llm_watchlist: list[dict[str, Any]] = []
    if effective_mode == "openai" and review_clusters:
        kept, watchlist, prompt_cost, completion_cost, review_prompt_tokens, review_completion_tokens, review_requests = screen_clusters_with_openai(
            clusters=review_clusters,
            system_prompt=system_prompt,
            criteria_prompt=criteria_prompt,
            postfix_prompt=postfix_prompt,
            model=config["HOTSPOTS"].get("model_screen"),
            batch_size=config["HOTSPOTS"].getint("screen_batch_size", fallback=8),
            retry_count=config["HOTSPOTS"].getint("retry", fallback=3),
            score_cutoff=score_cutoff,
            watchlist_cutoff=watchlist_cutoff,
        )
        llm_prompt_tokens += review_prompt_tokens
        llm_completion_tokens += review_completion_tokens
        llm_requests += review_requests
        llm_kept.extend(kept)
        llm_watchlist.extend(watchlist)
    else:
        kept, watchlist = heuristic_screen_clusters(review_clusters, score_cutoff, watchlist_cutoff)
        llm_kept.extend(kept)
        llm_watchlist.extend(watchlist)

    fallback_kept: list[dict[str, Any]] = []
    fallback_watchlist: list[dict[str, Any]] = []
    for topic in heuristic_fallback_topics:
        if topic["FINAL_SCORE"] >= score_cutoff and topic["QUALITY"] >= 5:
            fallback_kept.append({**topic, "KEEP_IN_DAILY_HOTSPOTS": True, "WATCHLIST": False, "SCREENING_DECISION": "heuristic_fallback"})
        elif topic["FINAL_SCORE"] >= watchlist_cutoff:
            fallback_watchlist.append({**topic, "KEEP_IN_DAILY_HOTSPOTS": False, "WATCHLIST": True, "SCREENING_DECISION": "heuristic_fallback"})

    kept = auto_keep_topics + llm_kept + fallback_kept
    watchlist = auto_watch_topics + llm_watchlist + fallback_watchlist

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
    summary, digest_prompt_cost, digest_completion_cost, digest_prompt_tokens, digest_completion_tokens, digest_requests = apply_digest_synthesis(
        featured_topics,
        watchlist,
        system_prompt,
        digest_prompt,
        config,
        effective_mode,
    )
    prompt_cost += digest_prompt_cost
    completion_cost += digest_completion_cost
    llm_prompt_tokens += digest_prompt_tokens
    llm_completion_tokens += digest_completion_tokens
    llm_requests += digest_requests
    x_buzz = _build_x_buzz_items(raw_items, featured_topics, watchlist)
    paper_spotlight = _build_paper_spotlight(
        raw_items,
        max_daily_hot=config["HOTSPOTS"].getint("paper_spotlight_max_daily_hot", fallback=6),
        max_new_frontier=config["HOTSPOTS"].getint("paper_spotlight_max_new_frontier", fallback=4),
    )
    usage = _build_usage_payload(
        config,
        effective_mode,
        prompt_cost,
        completion_cost,
        llm_prompt_tokens,
        llm_completion_tokens,
        llm_requests,
        api_usage,
    )

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
            "screening_auto_keep": screening_stats["auto_keep"],
            "screening_auto_watchlist": screening_stats["auto_watchlist"],
            "screening_heuristic_only": screening_stats["heuristic_only"],
            "screening_auto_drop": screening_stats["auto_drop"],
            "screening_review_candidates": screening_stats["review"],
            "screening_reviewed_by_llm": len(review_clusters),
            "screening_heuristic_fallback": len(heuristic_fallback_topics),
            "paper_spotlight_items": sum(len(section.get("items", [])) for section in paper_spotlight),
        },
        "costs": {"prompt": round(prompt_cost, 6), "completion": round(completion_cost, 6), "total": round(prompt_cost + completion_cost, 6)},
        "usage": usage,
        "top_topics": featured_topics,
        "featured_topics": featured_topics,
        "category_sections": category_sections,
        "long_tail_sections": long_tail_sections,
        "paper_spotlight": paper_spotlight,
        "x_buzz": x_buzz,
        "watchlist": watchlist,
    }

    paths = build_hotspot_paths(output_root, target_date.date())
    ensure_parent_dirs(paths)
    write_json(paths.normalized_path, _serialize_items(raw_items))
    write_json(paths.clusters_path, [cluster.to_dict() for cluster in clusters])
    write_json(paths.report_path, report)
    write_hotspot_web_data(output_root, report, raw_items)
    paths.markdown_path.write_text(render_hot_daily_md(report), encoding="utf-8")
    return report
