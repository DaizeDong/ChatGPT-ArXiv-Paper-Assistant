from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arxiv_assistant.paper_daily_io import discover_daily_json, extract_paper_mapping, write_daily_json_outputs
from arxiv_assistant.paper_topics import sort_paper_mapping_for_daily_display
from arxiv_assistant.renderers.paper.render_daily import render_daily_md, render_summary_table
from arxiv_assistant.utils.prompt_loader import read_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild grouped daily paper markdown from saved JSON payloads.")
    parser.add_argument("--output-root", default="out", help="Root output directory containing json/ and md/.")
    parser.add_argument(
        "--skip-json-backfill",
        action="store_true",
        help="Do not rewrite enriched daily output.json and companion daily-papers.json files.",
    )
    return parser.parse_args()
def rebuild_paper_markdown(output_root: str | Path, *, write_back_json: bool = True) -> list[str]:
    output_root = Path(output_root)
    json_root = output_root / "json"
    md_root = output_root / "md"
    prompts = (
        read_prompt("paper.system_prompt"),
        read_prompt("paper.topics"),
        read_prompt("paper.score_criteria"),
        read_prompt("paper.postfix_abstract"),
    )

    day_sources = discover_daily_json(json_root)
    if not day_sources:
        return []

    rebuilt_dates: list[str] = []
    latest_date = max(day_sources)
    latest_content = ""

    for date, source_path in sorted(day_sources.items()):
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        paper_mapping = sort_paper_mapping_for_daily_display(extract_paper_mapping(payload))
        date_string = f"{date[0]:04d}-{date[1]:02d}-{date[2]:02d}"
        existing_bundle_path = json_root / f"{date[0]:04d}-{date[1]:02d}" / f"{date_string}-daily-papers.json"
        usage_meta: Dict | None = None
        if existing_bundle_path.exists():
            existing_bundle = json.loads(existing_bundle_path.read_text(encoding="utf-8"))
            usage_candidate = existing_bundle.get("meta", {}).get("usage", {})
            if isinstance(usage_candidate, dict) and usage_candidate:
                usage_meta = usage_candidate

        month_dir = md_root / f"{date[0]:04d}-{date[1]:02d}"
        month_dir.mkdir(parents=True, exist_ok=True)
        md_path = month_dir / f"{date_string}-latest.md"
        rendered_md = render_daily_md(
            [],
            {},
            paper_mapping,
            now_date=date,
            prompts=prompts,
            head_table=(
                {
                    "html": render_summary_table(
                        model=usage_meta["model"],
                        prompt_tokens=int(usage_meta["prompt_tokens"]),
                        completion_tokens=int(usage_meta["completion_tokens"]),
                        prompt_cost=float(usage_meta["prompt_cost"]),
                        completion_cost=float(usage_meta["completion_cost"]),
                        total_arxiv_papers=int(usage_meta["total_arxiv_papers"]),
                        total_scanned_papers=int(usage_meta["total_scanned_papers"]),
                        total_relevant_papers=int(usage_meta["total_relevant_papers"]),
                    )
                }
                if usage_meta
                else None
            ),
        )
        md_path.write_text(rendered_md, encoding="utf-8")

        if write_back_json:
            write_daily_json_outputs(output_root, date, paper_mapping, usage=usage_meta)

        if date == latest_date:
            latest_content = rendered_md

        rebuilt_dates.append(date_string)

    if latest_content:
        (output_root / "latest.md").write_text(latest_content, encoding="utf-8")

    return rebuilt_dates


def main() -> None:
    args = parse_args()
    rebuilt_dates = rebuild_paper_markdown(
        args.output_root,
        write_back_json=not args.skip_json_backfill,
    )
    print(f"Rebuilt daily paper markdown for {len(rebuilt_dates)} day(s).")


if __name__ == "__main__":
    main()
