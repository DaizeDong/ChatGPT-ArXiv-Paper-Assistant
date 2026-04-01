from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Mapping, Tuple

from arxiv_assistant.paper_topics import build_daily_topic_bundle, sort_paper_mapping_for_daily_display

DAY_JSON_PATTERN = re.compile(r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})-(?P<suffix>[^/\\\\]+)\.json$")
JSON_SUFFIX_PRIORITY = {"output": 0, "daily-papers": 1, "latest": 2}


def _candidate_priority(path: Path) -> Tuple[int, str]:
    match = DAY_JSON_PATTERN.match(path.name)
    suffix = match.group("suffix") if match else path.stem
    return JSON_SUFFIX_PRIORITY.get(suffix, 99), path.name


def discover_daily_json(json_root: Path) -> Dict[Tuple[int, int, int], Path]:
    candidates: Dict[Tuple[int, int, int], list[Path]] = {}
    if not json_root.exists():
        return {}

    for file_path in sorted(json_root.glob("*/*.json")):
        match = DAY_JSON_PATTERN.match(file_path.name)
        if match is None:
            continue
        if match.group("suffix") not in JSON_SUFFIX_PRIORITY:
            continue
        date = (
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
        )
        candidates.setdefault(date, []).append(file_path)

    resolved: Dict[Tuple[int, int, int], Path] = {}
    for date, file_paths in candidates.items():
        resolved[date] = sorted(file_paths, key=_candidate_priority)[0]
    return resolved


def extract_paper_mapping(payload: Mapping) -> Dict[str, Dict]:
    papers_payload = payload.get("papers") if isinstance(payload, Mapping) else None
    if isinstance(papers_payload, Mapping):
        return {str(arxiv_id): dict(paper_entry) for arxiv_id, paper_entry in papers_payload.items()}
    if isinstance(payload, Mapping):
        return {
            str(arxiv_id): dict(paper_entry)
            for arxiv_id, paper_entry in payload.items()
            if isinstance(paper_entry, Mapping)
        }
    return {}


def load_daily_paper_mapping(path: Path) -> Dict[str, Dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return extract_paper_mapping(payload)


def write_daily_json_outputs(
    output_root: str | Path,
    date: Tuple[int, int, int],
    paper_mapping: Mapping[str, Mapping[str, object]],
    *,
    usage: Mapping[str, object] | None = None,
) -> Dict:
    output_root = Path(output_root)
    sorted_mapping = sort_paper_mapping_for_daily_display(paper_mapping)
    bundle = build_daily_topic_bundle(date, sorted_mapping, usage=usage)
    date_string = f"{date[0]:04d}-{date[1]:02d}-{date[2]:02d}"
    json_month_dir = output_root / "json" / f"{date[0]:04d}-{date[1]:02d}"
    json_month_dir.mkdir(parents=True, exist_ok=True)
    (json_month_dir / f"{date_string}-output.json").write_text(
        json.dumps(sorted_mapping, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (json_month_dir / f"{date_string}-daily-papers.json").write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return bundle
