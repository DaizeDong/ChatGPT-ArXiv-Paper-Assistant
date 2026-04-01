import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from arxiv_assistant.paper_daily_io import discover_daily_json, extract_paper_mapping
from arxiv_assistant.paper_topics import ensure_topic_fields, get_topic_registry
from arxiv_assistant.renderers.site_paths import site_day_page_path

MONTH_TOPIC_ORDER = list(get_topic_registry().topic_ids)
MONTHLY_SUMMARY_PATTERN = re.compile(r"(?P<year>\d{4})-(?P<month>\d{2})-summary\.json$")


def _paper_sort_key(paper_entry: Dict) -> Tuple[int, int, int, int, Tuple[int, int, int], str]:
    return (
        int(paper_entry.get("MONTHLY_PRIORITY", 0)),
        int(paper_entry.get("SCORE", 0)),
        int(paper_entry.get("RELEVANCE", 0)),
        int(paper_entry.get("NOVELTY", 0)),
        tuple(paper_entry.get("SOURCE_DATE", (0, 0, 0))),
        str(paper_entry.get("arxiv_id", "")),
    )


def empty_month_topic_buckets() -> Dict[str, List[Dict]]:
    return {topic_id: [] for topic_id in MONTH_TOPIC_ORDER}


def classify_monthly_summary_topic(paper_entry: Dict) -> str:
    return ensure_topic_fields(paper_entry).get("PRIMARY_TOPIC_ID", get_topic_registry().default_topic_id)


def load_generated_monthly_summary_data(monthly_root: Path) -> Dict[Tuple[int, int], Dict[str, List[Dict]]]:
    summary_data: Dict[Tuple[int, int], Dict[str, List[Dict]]] = {}
    if not monthly_root.exists():
        return summary_data

    for file_path in sorted(monthly_root.glob("*-summary.json")):
        match = MONTHLY_SUMMARY_PATTERN.match(file_path.name)
        if match is None:
            continue

        month_key = (int(match.group("year")), int(match.group("month")))
        with file_path.open("r", encoding="utf-8") as infile:
            payload = json.load(infile)

        topic_buckets = empty_month_topic_buckets()
        for paper_entry in payload.get("papers", []):
            if not paper_entry.get("KEEP_IN_MONTHLY", False):
                continue
            normalized_entry = ensure_topic_fields(paper_entry, arxiv_id=paper_entry.get("arxiv_id") or paper_entry.get("ARXIVID"))
            topic_id = normalized_entry["PRIMARY_TOPIC_ID"]
            topic_buckets.setdefault(topic_id, []).append(normalized_entry)

        for topic_id in MONTH_TOPIC_ORDER:
            topic_buckets[topic_id] = sorted(topic_buckets[topic_id], key=_paper_sort_key, reverse=True)

        summary_data[month_key] = topic_buckets

    return summary_data


def build_heuristic_monthly_summary_data(
    json_root: Path,
    day_page_mapping: Dict[Tuple[int, int, int], str],
) -> Dict[Tuple[int, int], Dict[str, List[Dict]]]:
    day_sources = discover_daily_json(json_root)
    month_entries: Dict[Tuple[int, int], Dict[str, Dict]] = {}

    for date, source_path in sorted(day_sources.items()):
        month_key = (date[0], date[1])
        with source_path.open("r", encoding="utf-8") as infile:
            daily_payload = json.load(infile)

        month_bucket = month_entries.setdefault(month_key, {})
        for arxiv_id, paper_entry in extract_paper_mapping(daily_payload).items():
            normalized_entry = ensure_topic_fields(
                {
                    **paper_entry,
                    "SOURCE_DATE": date,
                    "SOURCE_DAY_PAGE": day_page_mapping.get(date, site_day_page_path(date)),
                },
                arxiv_id=paper_entry.get("arxiv_id", arxiv_id),
            )
            existing_entry = month_bucket.get(arxiv_id)
            if existing_entry is None or _paper_sort_key(normalized_entry) > _paper_sort_key(existing_entry):
                month_bucket[arxiv_id] = normalized_entry

    monthly_summary_data: Dict[Tuple[int, int], Dict[str, List[Dict]]] = {}
    for month_key, papers_by_id in month_entries.items():
        topic_buckets = empty_month_topic_buckets()
        for paper_entry in papers_by_id.values():
            topic_buckets[classify_monthly_summary_topic(paper_entry)].append(paper_entry)

        for topic_id in MONTH_TOPIC_ORDER:
            topic_buckets[topic_id] = sorted(topic_buckets[topic_id], key=_paper_sort_key, reverse=True)

        monthly_summary_data[month_key] = topic_buckets

    return monthly_summary_data


def build_monthly_summary_data(
    json_root: Path,
    monthly_root: Path,
    day_page_mapping: Dict[Tuple[int, int, int], str],
) -> Dict[Tuple[int, int], Dict[str, List[Dict]]]:
    generated_summary_data = load_generated_monthly_summary_data(monthly_root)
    if generated_summary_data:
        return generated_summary_data
    return build_heuristic_monthly_summary_data(json_root, day_page_mapping)
