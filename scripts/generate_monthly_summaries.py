import argparse
import configparser
import hashlib
import json
import re
import sys
from calendar import monthrange
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Tuple

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arxiv_assistant.renderers.paper.monthly_summary import MONTH_CATEGORY_ORDER, discover_daily_json
from arxiv_assistant.renderers.site_paths import site_day_page_path
from arxiv_assistant.utils.local_env import load_local_env
from arxiv_assistant.utils.prompt_loader import read_prompt
from arxiv_assistant.utils.pricing_loader import get_model_pricing

load_local_env()

ALLOWED_CATEGORIES = set(MONTH_CATEGORY_ORDER)
MONTHLY_PRIORITY_POSITIVE_SIGNALS = {
    "theory": 50,
    "theoretical": 50,
    "scaling law": 55,
    "generalization": 35,
    "generalisation": 35,
    "convergence": 35,
    "bound": 30,
    "bounds": 30,
    "mixture-of-experts": 30,
    "mixture of experts": 30,
    "moe": 25,
    "transformer": 20,
    "quantization": 30,
    "quantisation": 30,
    "pruning": 30,
    "low-rank": 25,
    "low rank": 25,
    "distributed": 25,
    "throughput": 20,
    "memory optimization": 20,
    "representation learning": 25,
    "contrastive": 20,
    "interpretability": 18,
}
MONTHLY_PRIORITY_NEGATIVE_SIGNALS = {
    "survey": -120,
    "benchmark": -100,
    "dataset": -90,
    "medical": -80,
    "segmentation": -80,
    "recommendation": -70,
    "summarization": -60,
    "retrieval": -60,
    "rag": -60,
    "video": -50,
    "speech": -50,
    "time series": -45,
    "knowledge graph": -45,
    "multimodal": -40,
    "vision-language": -40,
    "vla": -35,
    "vlm": -35,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate monthly summary classification reports.")
    parser.add_argument("--output-root", default="out", help="Root output directory containing json/ and monthly/.")
    parser.add_argument(
        "--mode",
        choices=["openai", "heuristic"],
        default="openai",
        help="Use OpenAI for classification or generate a local heuristic fallback report.",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate even if the source hash did not change.")
    parser.add_argument(
        "--include-open-months",
        action="store_true",
        help="Generate summaries for the current in-progress month as well.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read(config_path)
    return config


def _paper_sort_key(paper_entry: Dict) -> Tuple[int, int, int, Tuple[int, int, int], str]:
    return (
        int(paper_entry.get("MONTHLY_PRIORITY", 0)),
        int(paper_entry.get("SCORE", 0)),
        int(paper_entry.get("RELEVANCE", 0)),
        int(paper_entry.get("NOVELTY", 0)),
        tuple(paper_entry.get("SOURCE_DATE", (0, 0, 0))),
        str(paper_entry.get("arxiv_id", "")),
    )


def _normalize_paper_entry(date: Tuple[int, int, int], arxiv_id: str, paper_entry: Dict) -> Dict:
    return {
        **paper_entry,
        "arxiv_id": paper_entry.get("arxiv_id", arxiv_id),
        "ARXIVID": paper_entry.get("ARXIVID", arxiv_id),
        "SOURCE_DATE": list(date),
        "SOURCE_DAY_PAGE": site_day_page_path(date),
    }


def collect_monthly_papers(json_root: Path) -> Dict[Tuple[int, int], List[Dict]]:
    day_sources = discover_daily_json(json_root)
    month_entries: Dict[Tuple[int, int], Dict[str, Dict]] = {}

    for date, source_path in sorted(day_sources.items()):
        month_key = (date[0], date[1])
        with source_path.open("r", encoding="utf-8") as infile:
            daily_payload = json.load(infile)

        month_bucket = month_entries.setdefault(month_key, {})
        for arxiv_id, paper_entry in daily_payload.items():
            normalized_entry = _normalize_paper_entry(date, arxiv_id, paper_entry)
            existing_entry = month_bucket.get(arxiv_id)
            if existing_entry is None or _paper_sort_key(normalized_entry) > _paper_sort_key(existing_entry):
                month_bucket[arxiv_id] = normalized_entry

    return {
        month_key: sorted(papers_by_id.values(), key=_paper_sort_key, reverse=True)
        for month_key, papers_by_id in month_entries.items()
    }


def is_month_closed(month_key: Tuple[int, int], now_time: datetime, include_open_months: bool) -> bool:
    if include_open_months:
        return True

    year, month = month_key
    if (year, month) < (now_time.year, now_time.month):
        return True

    return (year, month) == (now_time.year, now_time.month) and now_time.day == monthrange(year, month)[1]


def build_source_hash(papers: List[Dict]) -> str:
    digest = hashlib.sha256()
    for paper in papers:
        digest.update(json.dumps(
            {
                "arxiv_id": paper["arxiv_id"],
                "title": paper.get("title", ""),
                "comment": paper.get("COMMENT", ""),
                "score": paper.get("SCORE", 0),
                "source_date": paper.get("SOURCE_DATE", []),
            },
            sort_keys=True,
        ).encode("utf-8"))
    return digest.hexdigest()


def monthly_summary_report_path(monthly_root: Path, month_key: Tuple[int, int]) -> Path:
    return monthly_root / f"{month_key[0]:04d}-{month_key[1]:02d}-summary.json"


def load_existing_source_hash(report_path: Path) -> str | None:
    if not report_path.exists():
        return None
    try:
        with report_path.open("r", encoding="utf-8") as infile:
            return json.load(infile).get("source_hash")
    except Exception:
        return None


def build_openai_batch_prompt(criteria_prompt: str, postfix_prompt: str, papers: List[Dict]) -> str:
    config = load_config(REPO_ROOT / "configs" / "config.ini")
    selection_guidance = "\n".join(
        [
            "## Digest Size Guidance",
            f"- Aim to keep roughly {config['MONTHLY_SUMMARY']['target_total_papers']} papers total for the month when possible.",
            f"- Avoid keeping more than about {config['MONTHLY_SUMMARY']['max_papers_per_category']} papers in any one category unless the month is unusually strong.",
        ]
    )
    rendered_papers = []
    for paper in papers:
        rendered_papers.append(
            "\n".join(
                [
                    f"ArXiv ID: {paper['arxiv_id']}",
                    f"Date: {paper['SOURCE_DATE'][0]}-{paper['SOURCE_DATE'][1]:02d}-{paper['SOURCE_DATE'][2]:02d}",
                    f"Title: {paper['title']}",
                    f"Authors: {', '.join(paper.get('authors', []))}",
                    f"Daily Score: {paper.get('SCORE', 0)}",
                    f"Daily Relevance: {paper.get('RELEVANCE', 0)}",
                    f"Daily Novelty: {paper.get('NOVELTY', 0)}",
                    f"Daily Comment: {paper.get('COMMENT', '')}",
                    f"Abstract: {paper.get('abstract', '')[:3500]}",
                ]
            )
        )

    return "\n\n".join(
        [
            criteria_prompt,
            selection_guidance,
            "## Papers",
            "\n\n".join(rendered_papers),
            postfix_prompt,
        ]
    )


def parse_jsonl_response(raw_text: str) -> List[Dict]:
    normalized = re.sub(r"```jsonl?\n?", "", raw_text)
    normalized = re.sub(r"```", "", normalized)
    normalized = re.sub(r"\n+", "\n", normalized).strip()
    parsed = []
    for line in normalized.splitlines():
        line = line.strip()
        if not line:
            continue
        parsed.append(json.loads(line))
    return parsed


def calc_price(model: str, usage) -> Tuple[float, float]:
    model_pricing = get_model_pricing()
    if model not in model_pricing:
        return 0.0, 0.0

    cached_tokens = usage.model_extra.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    prompt_tokens = usage.prompt_tokens - cached_tokens
    completion_tokens = usage.completion_tokens

    cache_pricing = model_pricing[model].get("cache", model_pricing[model]["prompt"])
    prompt_pricing = model_pricing[model]["prompt"]
    completion_pricing = model_pricing[model]["completion"]
    return (
        cache_pricing * cached_tokens / 1_000_000 + prompt_pricing * prompt_tokens / 1_000_000,
        completion_pricing * completion_tokens / 1_000_000,
    )


def classify_month_with_openai(
    papers: List[Dict],
    system_prompt: str,
    criteria_prompt: str,
    postfix_prompt: str,
    model: str,
    batch_size: int,
    retry_count: int,
) -> Tuple[List[Dict], float, float]:
    openai_api_key = __import__("os").environ.get("OPENAI_API_KEY")
    openai_base_url = __import__("os").environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for monthly summary generation in openai mode")

    client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
    classified: Dict[str, Dict] = {}
    total_prompt_cost = 0.0
    total_completion_cost = 0.0

    pending_batches = [papers[idx: idx + batch_size] for idx in range(0, len(papers), batch_size)]
    for batch in pending_batches:
        remaining = list(batch)
        last_error: Exception | None = None
        for _ in range(max(retry_count, 1)):
            user_prompt = build_openai_batch_prompt(criteria_prompt, postfix_prompt, remaining)
            try:
                completion = client.chat.completions.create(
                    model=model,
                    seed=0,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                prompt_cost, completion_cost = calc_price(model, completion.usage)
                total_prompt_cost += prompt_cost
                total_completion_cost += completion_cost
                parsed_rows = parse_jsonl_response(completion.choices[0].message.content)
            except Exception as ex:
                last_error = ex
                continue

            parsed_by_id = {
                row["ARXIVID"]: row
                for row in parsed_rows
                if row.get("PRIMARY_CATEGORY") in ALLOWED_CATEGORIES
            }
            missing_ids = {paper["arxiv_id"] for paper in remaining} - set(parsed_by_id.keys())
            for paper in remaining:
                row = parsed_by_id.get(paper["arxiv_id"])
                if row is None:
                    continue
                classified[paper["arxiv_id"]] = {
                    **paper,
                    "PRIMARY_CATEGORY": row["PRIMARY_CATEGORY"],
                    "SECONDARY_CATEGORIES": [
                        category
                        for category in row.get("SECONDARY_CATEGORIES", [])
                        if category in ALLOWED_CATEGORIES and category != row["PRIMARY_CATEGORY"]
                    ],
                    "KEEP_IN_MONTHLY": bool(row.get("KEEP_IN_MONTHLY", False)),
                    "MONTHLY_COMMENT": row.get("MONTHLY_COMMENT", ""),
                }

            if not missing_ids:
                break
            remaining = [paper for paper in remaining if paper["arxiv_id"] in missing_ids]
        else:
            if last_error is not None:
                raise last_error

        for paper in remaining:
            if paper["arxiv_id"] not in classified:
                classified[paper["arxiv_id"]] = {
                    **paper,
                    "PRIMARY_CATEGORY": "Other Foundational Research",
                    "SECONDARY_CATEGORIES": [],
                    "KEEP_IN_MONTHLY": False,
                    "MONTHLY_COMMENT": "Excluded because monthly classification did not return a stable structured result.",
                }

    return list(classified.values()), total_prompt_cost, total_completion_cost


def classify_month_with_heuristic(papers: List[Dict]) -> List[Dict]:
    from arxiv_assistant.renderers.paper.monthly_summary import classify_monthly_summary_category
    config = load_config(REPO_ROOT / "configs" / "config.ini")
    score_cutoff = int(config["MONTHLY_SUMMARY"]["fallback_score_cutoff"])
    relevance_cutoff = int(config["MONTHLY_SUMMARY"]["fallback_relevance_cutoff"])
    novelty_cutoff = int(config["MONTHLY_SUMMARY"]["fallback_novelty_cutoff"])

    classified = []
    for paper in papers:
        score = int(paper.get("SCORE", 0))
        relevance = int(paper.get("RELEVANCE", 0))
        novelty = int(paper.get("NOVELTY", 0))
        classified.append(
            {
                **paper,
                "PRIMARY_CATEGORY": classify_monthly_summary_category(paper),
                "SECONDARY_CATEGORIES": [],
                "KEEP_IN_MONTHLY": score >= score_cutoff and relevance >= relevance_cutoff and novelty >= novelty_cutoff,
                "MONTHLY_COMMENT": paper.get("COMMENT", ""),
            }
        )
    return classified


def compute_monthly_priority(paper: Dict) -> int:
    score = int(paper.get("SCORE", 0))
    relevance = int(paper.get("RELEVANCE", 0))
    novelty = int(paper.get("NOVELTY", 0))
    base_priority = score * 20 + relevance * 12 + novelty * 10

    if relevance == 0 and novelty == 0:
        base_priority -= 300

    if paper.get("PRIMARY_CATEGORY") == "Other Foundational Research":
        base_priority -= 25

    title_text = str(paper.get("title", "")).lower()
    comment_text = str(paper.get("MONTHLY_COMMENT", "") or paper.get("COMMENT", "")).lower()
    text = f"{title_text}\n{comment_text}"

    for signal, bonus in MONTHLY_PRIORITY_POSITIVE_SIGNALS.items():
        if signal in text:
            base_priority += bonus

    for signal, penalty in MONTHLY_PRIORITY_NEGATIVE_SIGNALS.items():
        if signal in text:
            base_priority += penalty

    if "author match" in comment_text and relevance == 0 and novelty == 0:
        base_priority -= 400

    return base_priority


def apply_digest_caps(papers: List[Dict], config: configparser.ConfigParser) -> List[Dict]:
    max_per_category = int(config["MONTHLY_SUMMARY"]["max_papers_per_category"])
    target_total = int(config["MONTHLY_SUMMARY"]["target_total_papers"])
    enriched_papers = [{**paper, "MONTHLY_PRIORITY": compute_monthly_priority(paper)} for paper in papers]

    kept_by_category: Dict[str, List[Dict]] = {category: [] for category in MONTH_CATEGORY_ORDER}
    dropped_ids = set()

    for paper in enriched_papers:
        if not paper.get("KEEP_IN_MONTHLY", False):
            continue
        category = paper.get("PRIMARY_CATEGORY", "Other Foundational Research")
        if category not in kept_by_category:
            category = "Other Foundational Research"
        kept_by_category[category].append(paper)

    for category in MONTH_CATEGORY_ORDER:
        kept_by_category[category] = sorted(kept_by_category[category], key=_paper_sort_key, reverse=True)
        if max_per_category > 0 and len(kept_by_category[category]) > max_per_category:
            dropped_ids.update(paper["arxiv_id"] for paper in kept_by_category[category][max_per_category:])
            kept_by_category[category] = kept_by_category[category][:max_per_category]

    kept_after_category_cap = [paper for category in MONTH_CATEGORY_ORDER for paper in kept_by_category[category]]
    kept_after_category_cap = sorted(kept_after_category_cap, key=_paper_sort_key, reverse=True)

    if target_total > 0 and len(kept_after_category_cap) > target_total:
        preserved_ids = []
        for category in MONTH_CATEGORY_ORDER:
            if kept_by_category[category]:
                preserved_ids.append(kept_by_category[category][0]["arxiv_id"])
        preserved_set = set(preserved_ids)

        final_ids = list(preserved_ids)
        remaining_capacity = max(target_total - len(final_ids), 0)
        remaining_candidates = [
            paper["arxiv_id"]
            for paper in kept_after_category_cap
            if paper["arxiv_id"] not in preserved_set
        ]
        final_ids.extend(remaining_candidates[:remaining_capacity])
        final_set = set(final_ids)
        dropped_ids.update(paper["arxiv_id"] for paper in kept_after_category_cap if paper["arxiv_id"] not in final_set)

    else:
        final_set = {paper["arxiv_id"] for paper in kept_after_category_cap}

    adjusted = []
    for paper in enriched_papers:
        keep_in_monthly = paper.get("KEEP_IN_MONTHLY", False) and paper["arxiv_id"] in final_set
        if paper.get("KEEP_IN_MONTHLY", False) and not keep_in_monthly:
            monthly_comment = (paper.get("MONTHLY_COMMENT", "") + " ").strip()
            monthly_comment = (monthly_comment + "Trimmed during monthly digest curation to keep the report concise.").strip()
        else:
            monthly_comment = paper.get("MONTHLY_COMMENT", "")

        adjusted.append(
            {
                **paper,
                "KEEP_IN_MONTHLY": keep_in_monthly,
                "MONTHLY_COMMENT": monthly_comment,
            }
        )

    return adjusted


def write_monthly_report(
    report_path: Path,
    month_key: Tuple[int, int],
    source_hash: str,
    model_name: str,
    papers: List[Dict],
    prompt_cost: float = 0.0,
    completion_cost: float = 0.0,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "month": f"{month_key[0]:04d}-{month_key[1]:02d}",
        "generated_at": datetime.now(UTC).isoformat(),
        "model": model_name,
        "source_hash": source_hash,
        "prompt_cost": prompt_cost,
        "completion_cost": completion_cost,
        "papers": papers,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = REPO_ROOT
    config = load_config(repo_root / "configs" / "config.ini")
    output_root = repo_root / args.output_root
    json_root = output_root / "json"
    monthly_root = output_root / "monthly"

    monthly_root.mkdir(parents=True, exist_ok=True)
    monthly_papers = collect_monthly_papers(json_root)
    now_time = datetime.now(UTC)

    system_prompt = read_prompt("monthly.system_prompt")
    criteria_prompt = read_prompt("monthly.criteria")
    postfix_prompt = read_prompt("monthly.postfix")

    for month_key, papers in sorted(monthly_papers.items()):
        if not is_month_closed(month_key, now_time, args.include_open_months):
            continue

        source_hash = build_source_hash(papers)
        report_path = monthly_summary_report_path(monthly_root, month_key)
        if not args.force and load_existing_source_hash(report_path) == source_hash:
            print(f"Skip {month_key[0]}-{month_key[1]:02d}: source hash unchanged")
            continue

        if args.mode == "openai":
            classified_papers, prompt_cost, completion_cost = classify_month_with_openai(
                papers=papers,
                system_prompt=system_prompt,
                criteria_prompt=criteria_prompt,
                postfix_prompt=postfix_prompt,
                model=config["MONTHLY_SUMMARY"]["model"],
                batch_size=int(config["MONTHLY_SUMMARY"]["batch_size"]),
                retry_count=int(config["MONTHLY_SUMMARY"]["retry"]),
            )
            model_name = config["MONTHLY_SUMMARY"]["model"]
        else:
            classified_papers = classify_month_with_heuristic(papers)
            prompt_cost, completion_cost = 0.0, 0.0
            model_name = "heuristic"

        classified_papers = apply_digest_caps(classified_papers, config)
        classified_papers = sorted(classified_papers, key=_paper_sort_key, reverse=True)
        write_monthly_report(
            report_path=report_path,
            month_key=month_key,
            source_hash=source_hash,
            model_name=model_name,
            papers=classified_papers,
            prompt_cost=prompt_cost,
            completion_cost=completion_cost,
        )
        print(f"Wrote monthly summary report: {report_path}")


if __name__ == "__main__":
    main()
