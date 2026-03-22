from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime
from pathlib import Path

from arxiv_assistant.utils.hotspot_schema import HotspotItem
from arxiv_assistant.utils.hotspot_sources import clip_text

DAILY_JSON_PATTERN = re.compile(r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})-output\.json$")


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

    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not payload:
        return []

    items: list[HotspotItem] = []
    published_at = datetime(source_date.year, source_date.month, source_date.day, tzinfo=UTC).isoformat()

    for arxiv_id, paper in payload.items():
        url = f"https://arxiv.org/abs/{paper.get('arxiv_id', arxiv_id)}"
        items.append(
            HotspotItem(
                source_id="local_papers",
                source_name="Selected arXiv Papers",
                source_role="research_backbone",
                source_type="paper",
                title=paper.get("title", arxiv_id),
                summary=clip_text(paper.get("abstract") or paper.get("COMMENT") or "", 600),
                url=url,
                canonical_url=url,
                published_at=published_at,
                tags=[],
                authors=list(paper.get("authors", [])),
                metadata={
                    "arxiv_id": paper.get("arxiv_id", arxiv_id),
                    "daily_score": paper.get("SCORE", 0),
                    "relevance": paper.get("RELEVANCE", 0),
                    "novelty": paper.get("NOVELTY", 0),
                    "comment": paper.get("COMMENT", ""),
                },
            )
        )
    return items
