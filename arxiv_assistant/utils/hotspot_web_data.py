from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from arxiv_assistant.utils.hotspot_schema import HotspotItem

SOURCE_FAMILIES = [
    {
        "slug": "x-buzz",
        "label": "X / Buzz",
        "description": "Fast-moving social and community signal proxied through roundup and discussion sources.",
    },
    {
        "slug": "blogs",
        "label": "Blogs / Newsletters",
        "description": "Curated roundup and newsletter coverage that helps track same-day narrative consensus.",
    },
    {
        "slug": "official",
        "label": "Official Updates",
        "description": "Product, research, and platform updates from official vendor or lab channels.",
    },
    {
        "slug": "papers",
        "label": "Papers",
        "description": "Research papers and paper-trending signals tied to the current day.",
    },
    {
        "slug": "github",
        "label": "GitHub / Tools",
        "description": "Repositories, tooling launches, and builder momentum around practical AI systems.",
    },
    {
        "slug": "discussions",
        "label": "Discussions",
        "description": "Forum-style discussion threads and broader community reactions.",
    },
]

SOURCE_FAMILY_LOOKUP = {entry["slug"]: entry for entry in SOURCE_FAMILIES}
SOURCE_FAMILY_ORDER = [entry["slug"] for entry in SOURCE_FAMILIES]
SOCIAL_HOST_SNIPPETS = ("x.com", "twitter.com", "reddit.com")
BLOG_HOST_HINTS = ("rundown", "superhuman", "neuron", "smol.ai", "ben", "newsletter", "substack")
GITHUB_HOSTS = ("github.com",)
DISCUSSION_HOSTS = ("news.ycombinator.com", "reddit.com")
YEAR_PATTERN = re.compile(r"^\d{4}$")
MONTH_PATTERN = re.compile(r"^\d{4}-\d{2}$")
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}\.json$")


def _normalize_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()


def _slugify(value: str, *, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or fallback


def _short_text(value: str, limit: int = 180) -> str:
    normalized = _normalize_text(value)
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _url_host(url: str | None) -> str:
    if not url:
        return ""
    return urlparse(url).netloc.lower()


def _topic_slug(topic: dict[str, Any]) -> str:
    topic_id = str(topic.get("TOPIC_ID", "")).strip() or "topic"
    headline = str(topic.get("HEADLINE") or topic.get("title") or topic_id).strip()
    return _slugify(headline, fallback=_slugify(topic_id, fallback="topic"))


def _topic_score(topic: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(topic.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _item_identifier(item: HotspotItem) -> str:
    base = item.canonical_url or item.url or item.title
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
    return f"{item.source_id}-{digest}"


def _topic_paths(date_str: str, topic: dict[str, Any]) -> dict[str, str]:
    slug = _topic_slug(topic)
    return {
        "topic_id": str(topic.get("TOPIC_ID", "")).strip(),
        "slug": slug,
        "daily_route": f"/hot/{date_str}/topic/{slug}/",
    }


def _classify_source_family(item: HotspotItem) -> str:
    host = _url_host(item.url or item.canonical_url)
    metadata = item.metadata or {}
    if item.source_role == "community_heat" or any(snippet in host for snippet in SOCIAL_HOST_SNIPPETS):
        return "x-buzz"
    if item.source_role == "official_news" or bool(metadata.get("is_official")):
        return "official"
    if item.source_type == "paper" or metadata.get("arxiv_id"):
        return "papers"
    if item.source_role == "github_trend" or any(repo_host in host for repo_host in GITHUB_HOSTS) or metadata.get("github_url") or metadata.get("github_stars") or metadata.get("stars"):
        return "github"
    if item.source_role == "hn_discussion" or item.source_type == "discussion" or any(snippet in host for snippet in DISCUSSION_HOSTS):
        return "discussions"
    if item.source_role in {"headline_consensus", "editorial_depth", "builder_momentum"} or any(hint in host for hint in BLOG_HOST_HINTS):
        return "blogs"
    return "blogs"


def _item_signal_score(item: HotspotItem, topic_lookup: dict[str, dict[str, Any]]) -> float:
    metadata = item.metadata or {}
    score = 0.0
    score += min(3.0, float(metadata.get("activity", 0) or 0) / 250.0)
    score += min(2.4, float(metadata.get("github_stars", metadata.get("stars", 0)) or 0) / 500.0)
    score += min(2.0, float(metadata.get("hn_score", 0) or 0) / 75.0)
    score += min(1.6, float(metadata.get("upvotes", 0) or 0) / 100.0)
    score += min(1.8, float(metadata.get("daily_score", metadata.get("score", 0)) or 0) / 10.0)

    linked_scores = []
    canonical = item.canonical_url or item.url
    if canonical and canonical in topic_lookup:
        linked_scores.append(_topic_score(topic_lookup[canonical], "DISPLAY_PRIORITY"))
    title_key = item.title.lower()
    if title_key and title_key in topic_lookup:
        linked_scores.append(_topic_score(topic_lookup[title_key], "DISPLAY_PRIORITY"))
    if linked_scores:
        score += min(3.0, max(linked_scores))
    return round(score, 3)


def _build_topic_lookup(report: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    lookup: dict[str, dict[str, Any]] = {}
    topics: dict[str, dict[str, Any]] = {}
    sections = []
    sections.extend(report.get("featured_topics") or report.get("top_topics") or [])
    for section in report.get("category_sections") or []:
        sections.extend(section.get("topics", []))
    for section in report.get("long_tail_sections") or []:
        sections.extend(section.get("topics", []))
    sections.extend(report.get("watchlist") or [])

    for topic in sections:
        topic_id = str(topic.get("TOPIC_ID", "")).strip()
        if not topic_id:
            continue
        if topic_id not in topics:
            enriched = dict(topic)
            enriched.update(_topic_paths(str(report.get("date", "")), topic))
            topics[topic_id] = enriched
        for item in topic.get("items", []):
            url = str(item.get("url", "")).strip()
            if url:
                lookup[url] = topics[topic_id]
            title = str(item.get("title", "")).strip().lower()
            if title:
                lookup[title] = topics[topic_id]
    return lookup, list(topics.values())


def _build_topic_summary(topics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        topics,
        key=lambda topic: (
            _topic_score(topic, "DISPLAY_PRIORITY"),
            _topic_score(topic, "FINAL_SCORE"),
            _topic_score(topic, "OCCURRENCE_SCORE"),
            _topic_score(topic, "HEAT"),
        ),
        reverse=True,
    )
    summary = []
    for topic in ordered:
        summary.append(
            {
                "topic_id": topic["topic_id"],
                "slug": topic["slug"],
                "headline": topic.get("HEADLINE") or topic.get("title") or "Untitled Topic",
                "category": topic.get("PRIMARY_CATEGORY", "Research"),
                "final_score": _topic_score(topic, "FINAL_SCORE"),
                "heat": _topic_score(topic, "HEAT"),
                "occurrence_score": _topic_score(topic, "OCCURRENCE_SCORE"),
                "display_priority": _topic_score(topic, "DISPLAY_PRIORITY"),
                "llm_status": topic.get("LLM_STATUS") or ("featured" if topic.get("KEEP_IN_DAILY_HOTSPOTS") else "watchlist" if topic.get("WATCHLIST") else "candidate"),
                "source_count": len(topic.get("source_names", [])),
                "item_count": len(topic.get("items", [])),
                "route": topic.get("daily_route", ""),
            }
        )
    return summary


def _build_compact_topic(topic: dict[str, Any]) -> dict[str, Any]:
    evidence = []
    for item in topic.get("items", [])[:2]:
        evidence.append(
            {
                "title": item.get("title", "Untitled"),
                "url": item.get("url", ""),
                "source_name": item.get("source_name", item.get("source_id", "source")),
            }
        )
    return {
        "topic_id": topic["topic_id"],
        "slug": topic["slug"],
        "headline": topic.get("HEADLINE") or topic.get("title") or "Untitled Topic",
        "category": topic.get("PRIMARY_CATEGORY", "Research"),
        "summary_short": _short_text(topic.get("SHORT_COMMENT") or topic.get("summary") or topic.get("WHY_IT_MATTERS") or "", 150),
        "why_it_matters": _short_text(topic.get("WHY_IT_MATTERS", ""), 220),
        "scores": {
            "final": _topic_score(topic, "FINAL_SCORE"),
            "quality": _topic_score(topic, "QUALITY"),
            "heat": _topic_score(topic, "HEAT"),
            "importance": _topic_score(topic, "IMPORTANCE"),
            "occurrence": _topic_score(topic, "OCCURRENCE_SCORE"),
            "display_priority": _topic_score(topic, "DISPLAY_PRIORITY"),
        },
        "source_names": list(topic.get("source_names", [])),
        "source_roles": list(topic.get("source_roles", [])),
        "source_types": list(topic.get("source_types", [])),
        "llm_status": topic.get("LLM_STATUS") or ("featured" if topic.get("KEEP_IN_DAILY_HOTSPOTS") else "watchlist" if topic.get("WATCHLIST") else "candidate"),
        "evidence": evidence,
        "route": topic.get("daily_route", ""),
    }


def _build_source_sections(raw_items: list[HotspotItem], report: dict[str, Any], topic_lookup: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    date_str = str(report.get("date", ""))
    for item in raw_items:
        family = _classify_source_family(item)
        linked_topics = []
        canonical = item.canonical_url or item.url
        if canonical and canonical in topic_lookup:
            linked_topics.append(_topic_paths(date_str, topic_lookup[canonical]))
        title_key = item.title.lower()
        if title_key and title_key in topic_lookup:
            candidate = _topic_paths(date_str, topic_lookup[title_key])
            if not any(existing["topic_id"] == candidate["topic_id"] for existing in linked_topics):
                linked_topics.append(candidate)
        grouped[family].append(
            {
                "id": _item_identifier(item),
                "title": item.title,
                "summary_short": _short_text(item.summary, 180),
                "url": item.url,
                "canonical_url": item.canonical_url,
                "source_id": item.source_id,
                "source_name": item.source_name,
                "source_role": item.source_role,
                "source_type": item.source_type,
                "source_family": family,
                "published_at": item.published_at,
                "tags": list(item.tags),
                "authors": list(item.authors),
                "signals": {
                    "activity": int(item.metadata.get("activity", 0) or 0),
                    "github_stars": int(item.metadata.get("github_stars", item.metadata.get("stars", 0)) or 0),
                    "hn_score": int(item.metadata.get("hn_score", 0) or 0),
                    "upvotes": int(item.metadata.get("upvotes", 0) or 0),
                    "daily_score": int(item.metadata.get("daily_score", item.metadata.get("score", 0)) or 0),
                },
                "signal_score": _item_signal_score(item, topic_lookup),
                "topic_refs": linked_topics,
            }
        )

    sections = []
    for family in SOURCE_FAMILY_ORDER:
        items = sorted(
            grouped.get(family, []),
            key=lambda row: (
                row["signal_score"],
                row["published_at"] or "",
                row["title"].lower(),
            ),
            reverse=True,
        )
        entry = SOURCE_FAMILY_LOOKUP[family]
        sections.append(
            {
                "slug": family,
                "label": entry["label"],
                "description": entry["description"],
                "count": len(items),
                "items": items,
            }
        )
    return sections


def build_daily_hotspot_web_payload(report: dict[str, Any], raw_items: list[HotspotItem], *, previous_date: str | None = None, next_date: str | None = None) -> dict[str, Any]:
    topic_lookup, all_topics = _build_topic_lookup(report)
    source_sections = _build_source_sections(raw_items, report, topic_lookup)
    featured_topics = [
        _build_compact_topic(_build_topic_paths_and_merge(report, topic))
        for topic in report.get("featured_topics") or report.get("top_topics") or []
    ]
    category_sections = [
        {
            "category": section.get("category", "Other"),
            "total_candidates": int(section.get("total_candidates", len(section.get("topics", [])))),
            "topics": [_build_compact_topic(topic_lookup_row) for topic_lookup_row in [_build_topic_paths_and_merge(report, topic) for topic in section.get("topics", [])]],
        }
        for section in report.get("category_sections") or []
    ]
    long_tail_sections = [
        {
            "category": section.get("category", "Other"),
            "total_candidates": int(section.get("total_candidates", len(section.get("topics", [])))),
            "topics": [_build_compact_topic(topic_lookup_row) for topic_lookup_row in [_build_topic_paths_and_merge(report, topic) for topic in section.get("topics", [])]],
        }
        for section in report.get("long_tail_sections") or []
    ]
    watchlist_topics = [_build_compact_topic(_build_topic_paths_and_merge(report, topic)) for topic in report.get("watchlist") or []]
    topic_summary = _build_topic_summary(all_topics)
    section_counts = {section["slug"]: section["count"] for section in source_sections}
    return {
        "schema_version": 1,
        "meta": {
            "date": report.get("date"),
            "month": str(report.get("date", ""))[:7],
            "year": str(report.get("date", ""))[:4],
            "generated_at": report.get("generated_at"),
            "mode": report.get("mode", "heuristic"),
            "summary": report.get("summary", ""),
            "previous_date": previous_date,
            "next_date": next_date,
            "counts": {
                "featured_topics": len(featured_topics),
                "topic_summary": len(topic_summary),
                "category_radar": sum(len(section["topics"]) for section in category_sections),
                "long_tail": sum(len(section["topics"]) for section in long_tail_sections),
                "x_buzz": len(report.get("x_buzz") or []),
                "watchlist": len(watchlist_topics),
                "source_items": sum(section["count"] for section in source_sections),
            },
        },
        "totals": dict(report.get("totals") or {}),
        "costs": dict(report.get("costs") or {}),
        "source_stats": dict(report.get("source_stats") or {}),
        "source_section_counts": section_counts,
        "source_sections": source_sections,
        "topic_summary": topic_summary,
        "featured_topics": featured_topics,
        "category_sections": category_sections,
        "long_tail_sections": long_tail_sections,
        "watchlist": watchlist_topics,
        "x_buzz": list(report.get("x_buzz") or []),
    }


def _build_topic_paths_and_merge(report: dict[str, Any], topic: dict[str, Any]) -> dict[str, Any]:
    merged = dict(topic)
    merged.update(_topic_paths(str(report.get("date", "")), topic))
    return merged


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _daily_payload_path(web_root: Path, date_str: str) -> Path:
    return web_root / "hot" / f"{date_str}.json"


def _load_existing_daily_payloads(web_root: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    hot_root = web_root / "hot"
    if not hot_root.exists():
        return payloads
    for file_path in sorted(hot_root.glob("*.json")):
        if file_path.name == "index.json":
            continue
        if not DATE_PATTERN.fullmatch(file_path.name):
            continue
        payloads.append(json.loads(file_path.read_text(encoding="utf-8")))
    payloads.sort(key=lambda payload: str(payload.get("meta", {}).get("date", "")))
    return payloads


def _build_root_index(daily_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    months: dict[str, dict[str, Any]] = {}
    years: dict[str, dict[str, Any]] = {}
    dates = []
    for payload in daily_payloads:
        meta = payload.get("meta", {})
        date = str(meta.get("date", ""))
        month = str(meta.get("month", ""))
        year = str(meta.get("year", ""))
        counts = meta.get("counts", {})
        dates.append(
            {
                "date": date,
                "month": month,
                "year": year,
                "summary": meta.get("summary", ""),
                "featured_topics": int(counts.get("featured_topics", 0)),
                "source_items": int(counts.get("source_items", 0)),
                "topic_summary": int(counts.get("topic_summary", 0)),
                "source_section_counts": payload.get("source_section_counts", {}),
                "route": f"/hot/{date}/",
            }
        )
        month_entry = months.setdefault(month, {"month": month, "days": 0, "featured_topics": 0, "source_items": 0})
        month_entry["days"] += 1
        month_entry["featured_topics"] += int(counts.get("featured_topics", 0))
        month_entry["source_items"] += int(counts.get("source_items", 0))
        month_entry["route"] = f"/hot/{month}/"
        year_entry = years.setdefault(year, {"year": year, "days": 0, "featured_topics": 0, "source_items": 0, "route": f"/hot/{year}/"})
        year_entry["days"] += 1
        year_entry["featured_topics"] += int(counts.get("featured_topics", 0))
        year_entry["source_items"] += int(counts.get("source_items", 0))
    return {
        "schema_version": 1,
        "latest_date": dates[-1]["date"] if dates else None,
        "dates": dates,
        "months": sorted(months.values(), key=lambda row: row["month"]),
        "years": sorted(years.values(), key=lambda row: row["year"]),
    }


def _build_month_index(month: str, daily_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    entries = []
    for payload in daily_payloads:
        meta = payload.get("meta", {})
        if meta.get("month") != month:
            continue
        counts = meta.get("counts", {})
        entries.append(
            {
                "date": meta.get("date"),
                "summary": meta.get("summary", ""),
                "featured_topics": int(counts.get("featured_topics", 0)),
                "source_items": int(counts.get("source_items", 0)),
                "topic_summary": int(counts.get("topic_summary", 0)),
                "source_section_counts": payload.get("source_section_counts", {}),
                "route": f"/hot/{meta.get('date')}/",
            }
        )
    entries.sort(key=lambda row: row["date"])
    return {
        "schema_version": 1,
        "month": month,
        "year": month[:4],
        "days": entries,
    }


def _build_year_index(year: str, daily_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    month_rollup: dict[str, dict[str, Any]] = {}
    for payload in daily_payloads:
        meta = payload.get("meta", {})
        if meta.get("year") != year:
            continue
        month = str(meta.get("month", ""))
        counts = meta.get("counts", {})
        entry = month_rollup.setdefault(
            month,
            {
                "month": month,
                "days": 0,
                "featured_topics": 0,
                "source_items": 0,
                "topic_summary": 0,
                "route": f"/hot/{month}/",
            },
        )
        entry["days"] += 1
        entry["featured_topics"] += int(counts.get("featured_topics", 0))
        entry["source_items"] += int(counts.get("source_items", 0))
        entry["topic_summary"] += int(counts.get("topic_summary", 0))
    return {
        "schema_version": 1,
        "year": year,
        "months": sorted(month_rollup.values(), key=lambda row: row["month"]),
    }


def write_hotspot_web_data(output_root: str | Path, report: dict[str, Any], raw_items: list[HotspotItem]) -> dict[str, Path]:
    output_root = Path(output_root)
    web_root = output_root / "web_data"
    existing_payloads = _load_existing_daily_payloads(web_root)
    existing_dates = [str(payload.get("meta", {}).get("date", "")) for payload in existing_payloads]
    current_date = str(report.get("date", ""))
    ordered_dates = sorted(set([date for date in existing_dates if date] + [current_date]))
    previous_date = None
    next_date = None
    if current_date in ordered_dates:
        current_index = ordered_dates.index(current_date)
        if current_index > 0:
            previous_date = ordered_dates[current_index - 1]
        if current_index + 1 < len(ordered_dates):
            next_date = ordered_dates[current_index + 1]

    daily_payload = build_daily_hotspot_web_payload(report, raw_items, previous_date=previous_date, next_date=next_date)
    daily_path = _daily_payload_path(web_root, current_date)
    _write_json(daily_path, daily_payload)

    all_payloads = _load_existing_daily_payloads(web_root)
    ordered_payloads = sorted(all_payloads, key=lambda payload: str(payload.get("meta", {}).get("date", "")))
    ordered_dates = [str(payload.get("meta", {}).get("date", "")) for payload in ordered_payloads]
    for index, payload in enumerate(ordered_payloads):
        prev_date = ordered_dates[index - 1] if index > 0 else None
        following_date = ordered_dates[index + 1] if index + 1 < len(ordered_dates) else None
        meta = payload.setdefault("meta", {})
        if meta.get("previous_date") != prev_date or meta.get("next_date") != following_date:
            meta["previous_date"] = prev_date
            meta["next_date"] = following_date
            _write_json(_daily_payload_path(web_root, str(meta.get("date", ""))), payload)

    root_index = _build_root_index(all_payloads)
    _write_json(web_root / "hot" / "index.json", root_index)

    written_paths = {"daily": daily_path, "root_index": web_root / "hot" / "index.json"}
    months = sorted({str(payload.get("meta", {}).get("month", "")) for payload in all_payloads if payload.get("meta", {}).get("month")})
    years = sorted({str(payload.get("meta", {}).get("year", "")) for payload in all_payloads if payload.get("meta", {}).get("year")})

    for month in months:
        month_path = web_root / "hot" / month / "index.json"
        _write_json(month_path, _build_month_index(month, all_payloads))
        written_paths[f"month:{month}"] = month_path

    for year in years:
        year_path = web_root / "hot" / year / "index.json"
        _write_json(year_path, _build_year_index(year, all_payloads))
        written_paths[f"year:{year}"] = year_path

    return written_paths
