import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from arxiv_assistant.renderers.site_paths import site_day_page_path

MONTH_CATEGORY_ORDER = [
    "Model Architecture",
    "Model Compression and Efficiency",
    "High Performance Computing",
    "Representation Learning",
    "Other Foundational Research",
]

MONTH_CATEGORY_KEYWORDS = {
    "Model Architecture": {
        "comment": [
            "mixture-of-experts",
            "mixture of experts",
            "moe",
            "transformer",
            "architecture",
            "architectural",
            "routing",
            "expert",
            "experts",
            "attention",
            "autoencoder",
        ],
        "title": [
            "mixture of experts",
            "moe",
            "transformer",
            "architecture",
            "routing",
            "expert",
            "attention",
            "autoencoder",
        ],
        "abstract": [
            "mixture of experts",
            "moe",
            "architecture",
            "architectural",
            "routing",
            "expert",
            "transformer",
            "attention",
            "autoencoder",
        ],
    },
    "Model Compression and Efficiency": {
        "comment": [
            "quantization",
            "quantisation",
            "quantized",
            "pruning",
            "pruned",
            "compression",
            "compressed",
            "low-rank",
            "low rank",
            "sparsity",
            "sparse",
            "cache",
            "low-bit",
            "efficiency",
            "efficient",
        ],
        "title": [
            "quantization",
            "quantisation",
            "pruning",
            "compression",
            "low-rank",
            "low rank",
            "sparsity",
            "cache",
            "low-bit",
            "efficient",
        ],
        "abstract": [
            "quantization",
            "quantisation",
            "pruning",
            "compression",
            "compressed",
            "low-rank",
            "low rank",
            "sparsity",
            "sparse",
            "cache",
            "low-bit",
            "efficient",
            "efficiency",
        ],
    },
    "High Performance Computing": {
        "comment": [
            "distributed",
            "parallel",
            "systems",
            "system",
            "memory optimization",
            "memory",
            "throughput",
            "communication",
            "large-scale training",
            "large scale training",
        ],
        "title": [
            "distributed",
            "parallel",
            "systems",
            "system",
            "throughput",
            "kernel",
            "communication",
            "serving",
        ],
        "abstract": [
            "distributed",
            "parallel",
            "systems",
            "system",
            "memory optimization",
            "throughput",
            "communication",
            "large-scale training",
            "large scale training",
            "serving",
            "kernel",
            "cluster",
            "pipeline",
        ],
    },
    "Representation Learning": {
        "comment": [
            "representation learning",
            "representation",
            "feature learning",
            "feature",
            "embedding",
            "contrastive",
            "dictionary learning",
            "sparse coding",
            "generalization",
            "generalisation",
            "interpretability",
        ],
        "title": [
            "representation",
            "feature",
            "embedding",
            "contrastive",
            "dictionary learning",
            "generalization",
            "generalisation",
            "interpretability",
        ],
        "abstract": [
            "representation",
            "feature",
            "embedding",
            "contrastive",
            "dictionary learning",
            "sparse coding",
            "generalization",
            "generalisation",
            "interpretability",
            "latent",
        ],
    },
}

DAY_JSON_PATTERN = re.compile(r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})-(?P<suffix>[^/\\\\]+)\.json$")
JSON_SUFFIX_PRIORITY = {"output": 0, "latest": 1}
MONTHLY_SUMMARY_PATTERN = re.compile(r"(?P<year>\d{4})-(?P<month>\d{2})-summary\.json$")


def _candidate_priority(path: Path) -> Tuple[int, str]:
    match = DAY_JSON_PATTERN.match(path.name)
    suffix = match.group("suffix") if match else path.stem
    return JSON_SUFFIX_PRIORITY.get(suffix, 99), path.name


def discover_daily_json(json_root: Path) -> Dict[Tuple[int, int, int], Path]:
    candidates: Dict[Tuple[int, int, int], List[Path]] = {}
    if not json_root.exists():
        return {}

    for file_path in sorted(json_root.glob("*/*.json")):
        match = DAY_JSON_PATTERN.match(file_path.name)
        if match is None:
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


def _match_weight(text: str, keywords: List[str], weight: int) -> int:
    return sum(weight for keyword in keywords if keyword in text)


def classify_monthly_summary_category(paper_entry: Dict) -> str:
    comment_text = str(paper_entry.get("COMMENT", "")).lower()
    title_text = str(paper_entry.get("title", "")).lower()
    abstract_text = str(paper_entry.get("abstract", "")).lower()

    best_category = "Other Foundational Research"
    best_score = 0

    for category in MONTH_CATEGORY_ORDER[:-1]:
        keyword_groups = MONTH_CATEGORY_KEYWORDS[category]
        category_score = (
            _match_weight(comment_text, keyword_groups["comment"], 4)
            + _match_weight(title_text, keyword_groups["title"], 3)
            + _match_weight(abstract_text, keyword_groups["abstract"], 1)
        )
        if category_score > best_score:
            best_category = category
            best_score = category_score

    return best_category


def _paper_sort_key(paper_entry: Dict) -> Tuple[int, int, int, Tuple[int, int, int], str]:
    return (
        int(paper_entry.get("MONTHLY_PRIORITY", 0)),
        int(paper_entry.get("SCORE", 0)),
        int(paper_entry.get("RELEVANCE", 0)),
        int(paper_entry.get("NOVELTY", 0)),
        tuple(paper_entry.get("SOURCE_DATE", (0, 0, 0))),
        str(paper_entry.get("arxiv_id", "")),
    )


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

        category_buckets = {category: [] for category in MONTH_CATEGORY_ORDER}
        for paper_entry in payload.get("papers", []):
            if not paper_entry.get("KEEP_IN_MONTHLY", False):
                continue
            category = paper_entry.get("PRIMARY_CATEGORY", "Other Foundational Research")
            if category not in category_buckets:
                category = "Other Foundational Research"
            category_buckets[category].append(paper_entry)

        for category in MONTH_CATEGORY_ORDER:
            category_buckets[category] = sorted(category_buckets[category], key=_paper_sort_key, reverse=True)

        summary_data[month_key] = category_buckets

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
        for arxiv_id, paper_entry in daily_payload.items():
            normalized_entry = {
                **paper_entry,
                "arxiv_id": paper_entry.get("arxiv_id", arxiv_id),
                "SOURCE_DATE": date,
                "SOURCE_DAY_PAGE": day_page_mapping.get(date, site_day_page_path(date)),
            }
            existing_entry = month_bucket.get(arxiv_id)
            if existing_entry is None or _paper_sort_key(normalized_entry) > _paper_sort_key(existing_entry):
                month_bucket[arxiv_id] = normalized_entry

    monthly_summary_data: Dict[Tuple[int, int], Dict[str, List[Dict]]] = {}
    for month_key, papers_by_id in month_entries.items():
        category_buckets = {category: [] for category in MONTH_CATEGORY_ORDER}
        for paper_entry in papers_by_id.values():
            category_buckets[classify_monthly_summary_category(paper_entry)].append(paper_entry)

        for category in MONTH_CATEGORY_ORDER:
            category_buckets[category] = sorted(category_buckets[category], key=_paper_sort_key, reverse=True)

        monthly_summary_data[month_key] = category_buckets

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
