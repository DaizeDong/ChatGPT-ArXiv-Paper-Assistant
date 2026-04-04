from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime
from pathlib import Path

from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem
from arxiv_assistant.utils.hotspot.hotspot_sources import clip_text

DAILY_JSON_PATTERN = re.compile(r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})-output\.json$")


def _spotlight_bundle_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name.replace("-output.json", "-hotspot-papers.json"))


def _clip_summary(paper: dict, *, prefer_spotlight_comment: bool = False) -> str:
    if prefer_spotlight_comment and paper.get("HOTSPOT_PAPER_COMMENT"):
        return clip_text(paper.get("HOTSPOT_PAPER_COMMENT") or "", 600)
    return clip_text(paper.get("abstract") or paper.get("COMMENT") or paper.get("HOTSPOT_PAPER_COMMENT") or "", 600)


def _discover_daily_json(json_root: Path) -> dict[tuple[int, int, int], Path]:
    discovered: dict[tuple[int, int, int], Path] = {}
    if not json_root.exists():
        return discovered

    for file_path in json_root.glob("*/*-output.json"):
        match = DAILY_JSON_PATTERN.fullmatch(file_path.name)
        if match is None:
            continue
        date_key = (
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
        )
        discovered[date_key] = file_path
    return discovered


def _resolve_best_source_path(target_date: datetime, json_root: Path) -> tuple[date, Path] | None:
    day_sources = _discover_daily_json(json_root)
    candidates = [
        (date_key, path)
        for date_key, path in day_sources.items()
        if date_key <= (target_date.year, target_date.month, target_date.day)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_date_key, best_path = candidates[0]
    return date(*best_date_key), best_path


def fetch_hotspot_items(target_date: datetime, output_root: str | Path, max_staleness_days: int = 2) -> list[HotspotItem]:
    json_root = Path(output_root) / "json"
    resolved = _resolve_best_source_path(target_date, json_root)
    if resolved is None:
        return []

    source_date, source_path = resolved
    if not source_path.exists():
        return []

    if max_staleness_days >= 0:
        target_day = date(target_date.year, target_date.month, target_date.day)
        staleness_days = (target_day - source_date).days
        if staleness_days > max_staleness_days:
            return []

    items: list[HotspotItem] = []
    published_at = datetime(source_date.year, source_date.month, source_date.day, tzinfo=UTC).isoformat()
    selected_payload = json.loads(source_path.read_text(encoding="utf-8"))
    spotlight_path = _spotlight_bundle_path(source_path)
    spotlight_payload = json.loads(spotlight_path.read_text(encoding="utf-8")) if spotlight_path.exists() else {}
    spotlight_papers = spotlight_payload.get("papers", {}) if isinstance(spotlight_payload, dict) else {}
    spotlight_section_lookup = {}
    for section in spotlight_payload.get("sections", []) if isinstance(spotlight_payload, dict) else []:
        for paper_id in section.get("paper_ids", []):
            spotlight_section_lookup[str(paper_id)] = {
                "kind": section.get("kind"),
                "label": section.get("label"),
            }

    if not selected_payload and not spotlight_papers:
        return []

    seen_arxiv_ids: set[str] = set()

    for arxiv_id, paper in selected_payload.items():
        url = f"https://arxiv.org/abs/{paper.get('arxiv_id', arxiv_id)}"
        seen_arxiv_ids.add(str(paper.get("arxiv_id", arxiv_id)))
        primary_topic_id = paper.get("PRIMARY_TOPIC_ID")
        primary_topic_label = paper.get("PRIMARY_TOPIC_LABEL")
        tags = [tag for tag in [primary_topic_id, primary_topic_label] if tag]
        items.append(
            HotspotItem(
                source_id="local_papers",
                source_name="Selected arXiv Papers",
                source_role="research_backbone",
                source_type="paper",
                title=paper.get("title", arxiv_id),
                summary=_clip_summary(paper),
                url=url,
                canonical_url=url,
                published_at=published_at,
                tags=tags,
                authors=list(paper.get("authors", [])),
                metadata={
                    "arxiv_id": paper.get("arxiv_id", arxiv_id),
                    "daily_score": paper.get("SCORE", 0),
                    "relevance": paper.get("RELEVANCE", 0),
                    "novelty": paper.get("NOVELTY", 0),
                    "comment": paper.get("COMMENT", ""),
                    "primary_topic_id": primary_topic_id,
                    "primary_topic_label": primary_topic_label,
                },
            )
        )

    for arxiv_id, paper in spotlight_papers.items():
        if str(paper.get("arxiv_id", arxiv_id)) in seen_arxiv_ids:
            continue
        url = f"https://arxiv.org/abs/{paper.get('arxiv_id', arxiv_id)}"
        primary_topic_id = paper.get("PRIMARY_TOPIC_ID")
        primary_topic_label = paper.get("PRIMARY_TOPIC_LABEL")
        spotlight_info = spotlight_section_lookup.get(str(arxiv_id), {})
        spotlight_kind = (
            spotlight_info.get("kind")
            or paper.get("HOTSPOT_PAPER_PRIMARY_KIND")
            or (paper.get("HOTSPOT_PAPER_TAGS") or [None])[0]
        )
        spotlight_label = spotlight_info.get("label") or paper.get("HOTSPOT_PAPER_PRIMARY_LABEL")
        tags = [
            tag
            for tag in [
                primary_topic_id,
                primary_topic_label,
                spotlight_kind,
                spotlight_label,
            ]
            if tag
        ]
        items.append(
            HotspotItem(
                source_id="local_hotspot_papers",
                source_name="Daily Hotspot Papers",
                source_role="paper_trending",
                source_type="paper",
                title=paper.get("title", arxiv_id),
                summary=_clip_summary(paper, prefer_spotlight_comment=True),
                url=url,
                canonical_url=url,
                published_at=published_at,
                tags=tags,
                authors=list(paper.get("authors", [])),
                metadata={
                    "arxiv_id": paper.get("arxiv_id", arxiv_id),
                    "daily_score": paper.get("SCORE", 0),
                    "relevance": paper.get("RELEVANCE", 0),
                    "novelty": paper.get("NOVELTY", 0),
                    "comment": paper.get("COMMENT", ""),
                    "primary_topic_id": primary_topic_id,
                    "primary_topic_label": primary_topic_label,
                    "spotlight_kinds": list(paper.get("HOTSPOT_PAPER_TAGS", [])),
                    "spotlight_primary_kind": spotlight_kind,
                    "spotlight_primary_label": spotlight_label,
                    "spotlight_comment": paper.get("HOTSPOT_PAPER_COMMENT", ""),
                },
            )
        )
    return items
