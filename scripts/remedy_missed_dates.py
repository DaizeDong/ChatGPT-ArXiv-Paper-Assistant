import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable


DateTuple = tuple[int, int, int]
RemedyPlan = dict[DateTuple, tuple[DateTuple, DateTuple]]
REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MISSED_DATES: RemedyPlan = {
    # date_to_remedy: [start_date_to_search, end_date_to_search]
    (2025, 5, 16): ((2025, 5, 15), (2025, 5, 15)),
    (2025, 5, 19): ((2025, 5, 16), (2025, 5, 18)),
    (2025, 5, 20): ((2025, 5, 19), (2025, 5, 19)),
    (2025, 5, 21): ((2025, 5, 20), (2025, 5, 20)),
    (2025, 5, 22): ((2025, 5, 21), (2025, 5, 21)),
    (2025, 5, 23): ((2025, 5, 22), (2025, 5, 22)),
}


def parse_date_string(value: str) -> date:
    return datetime.strptime(value.strip(), "%Y-%m-%d").date()


def tuple_to_date(value: DateTuple) -> date:
    return date(*value)


def date_to_tuple(value: date) -> DateTuple:
    return value.year, value.month, value.day


def iter_recorded_dates(output_root: str) -> Iterable[date]:
    root = Path(output_root)
    seen = set()

    for subdir in ("json", "md"):
        subroot = root / subdir
        if not subroot.exists():
            continue

        for path in subroot.glob("*/*-output.*"):
            try:
                recorded = parse_date_string(path.name.split("-output", 1)[0])
            except ValueError:
                continue
            if recorded not in seen:
                seen.add(recorded)
                yield recorded


def build_plan_from_dates(remedy_dates: list[date], output_root: str) -> RemedyPlan:
    existing_dates = sorted(iter_recorded_dates(output_root))
    existing_date_set = set(existing_dates)
    processed_dates: set[date] = set()
    plan: RemedyPlan = {}

    for remedy_date in sorted(set(remedy_dates)):
        candidate_dates = {d for d in existing_date_set if d < remedy_date}
        candidate_dates.update(d for d in processed_dates if d < remedy_date)
        if not candidate_dates:
            raise ValueError(f"Unable to infer the previous recorded date for {remedy_date.isoformat()}")

        begin_date = max(candidate_dates)
        end_date = remedy_date - timedelta(days=1)
        if begin_date > end_date:
            raise ValueError(
                f"Invalid inferred search window for {remedy_date.isoformat()}: "
                f"{begin_date.isoformat()} -> {end_date.isoformat()}"
            )

        plan[date_to_tuple(remedy_date)] = (date_to_tuple(begin_date), date_to_tuple(end_date))
        processed_dates.add(remedy_date)

    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-run paper filtering for missed or incomplete daily outputs.")
    parser.add_argument(
        "--dates",
        help="Comma-separated remedy dates in YYYY-MM-DD format. Search windows are inferred from existing outputs.",
    )
    parser.add_argument(
        "--date",
        action="append",
        default=[],
        help="Repeatable remedy date in YYYY-MM-DD format. Can be combined with --dates.",
    )
    parser.add_argument(
        "--output-root",
        default="out",
        help="Output root containing md/json folders. Defaults to out.",
    )
    parser.add_argument(
        "--begin-date",
        help="Explicit begin date for the search window. Requires exactly one remedy date and --end-date.",
    )
    parser.add_argument(
        "--end-date",
        help="Explicit end date for the search window. Requires exactly one remedy date and --begin-date.",
    )
    parser.add_argument(
        "--build-site",
        action="store_true",
        help="Build the multipage site after all remedial runs. Disabled by default because site publishing happens separately.",
    )
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="Print the inferred remedy plan and exit without running any API calls.",
    )
    return parser.parse_args()


def load_remedy_plan(args: argparse.Namespace) -> RemedyPlan:
    cli_dates = list(args.date)
    if args.dates:
        cli_dates.extend(part for part in args.dates.split(",") if part.strip())

    if args.begin_date or args.end_date:
        if not (args.begin_date and args.end_date):
            raise ValueError("Both --begin-date and --end-date are required when specifying an explicit search window.")
        if len(cli_dates) != 1:
            raise ValueError("Exactly one remedy date must be provided when using --begin-date/--end-date.")

        remedy_date = parse_date_string(cli_dates[0])
        begin_date = parse_date_string(args.begin_date)
        end_date = parse_date_string(args.end_date)
        return {date_to_tuple(remedy_date): (date_to_tuple(begin_date), date_to_tuple(end_date))}

    if not cli_dates:
        return DEFAULT_MISSED_DATES

    remedy_dates = [parse_date_string(value) for value in cli_dates]
    return build_plan_from_dates(remedy_dates, args.output_root)


def print_plan(plan: RemedyPlan) -> None:
    for remedy_date, (begin_date, end_date) in sorted(plan.items()):
        print(
            f"{remedy_date[0]:04d}-{remedy_date[1]:02d}-{remedy_date[2]:02d}: "
            f"{begin_date[0]:04d}-{begin_date[1]:02d}-{begin_date[2]:02d} -> "
            f"{end_date[0]:04d}-{end_date[1]:02d}-{end_date[2]:02d}"
        )


def run_remedy_plan(plan: RemedyPlan, output_root: str, build_site: bool) -> None:
    from arxiv_assistant.apis.arxiv import get_papers_from_arxiv
    from arxiv_assistant.apis.semantic_scholar import get_authors
    from arxiv_assistant.environment import (
        AUTHOR_ID_SET,
        CONFIG,
        NOW_DAY,
        NOW_MONTH,
        NOW_YEAR,
        POSTFIX_PROMPT_ABSTRACT,
        POSTFIX_PROMPT_TITLE,
        S2_API_KEY,
        SCORE_PROMPT,
        SLACK_KEY,
        SYSTEM_PROMPT,
        TOPIC_PROMPT,
    )
    from arxiv_assistant.filters.filter_author import filter_papers_by_hindex, select_by_author
    from arxiv_assistant.filters.filter_gpt import filter_by_gpt
    from arxiv_assistant.push_to_slack import push_to_slack
    from arxiv_assistant.renderers.build_multipage_site import build_multipage_site
    from arxiv_assistant.renderers.render_daily import render_daily_md
    from arxiv_assistant.utils.io import copy_file_or_dir, create_dir, delete_file_or_dir
    from arxiv_assistant.utils.utils import EnhancedJSONEncoder

    CONFIG["OUTPUT"]["output_path"] = output_root

    for remedy_date, (begin_date, end_date) in sorted(plan.items()):
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(f"Start remedying for date: {remedy_date}")
        print(f"Searching date range: {begin_date} - {end_date}")

        remedy_year, remedy_month, remedy_day = remedy_date

        output_debug_dir = os.path.join(
            CONFIG["OUTPUT"]["output_path"],
            "debug",
            f"{remedy_year}-{remedy_month:02d}",
            f"{remedy_year}-{remedy_month:02d}-{remedy_day:02d}",
        )
        output_debug_file_format = os.path.join(output_debug_dir, "{}")
        create_dir(output_debug_dir)

        output_md_dir = os.path.join(CONFIG["OUTPUT"]["output_path"], "md", f"{remedy_year}-{remedy_month:02d}")
        output_md_file_format = os.path.join(
            output_md_dir,
            f"{remedy_year}-{remedy_month:02d}-{remedy_day:02d}-" + "{}",
        )
        create_dir(output_md_dir)

        output_json_dir = os.path.join(CONFIG["OUTPUT"]["output_path"], "json", f"{remedy_year}-{remedy_month:02d}")
        output_json_file_format = os.path.join(
            output_json_dir,
            f"{remedy_year}-{remedy_month:02d}-{remedy_day:02d}-" + "{}",
        )
        create_dir(output_json_dir)

        all_entries, arxiv_paper_dict = get_papers_from_arxiv(
            CONFIG,
            source="api",
            begin_date=begin_date,
            end_date=end_date,
        )

        paper_list = list(set(v for area_papers in arxiv_paper_dict.values() for v in area_papers))
        print("Total number of papers:" + str(len(paper_list)))
        if len(paper_list) == 0:
            raise RuntimeError(f"No papers found for remedy date {remedy_date}")

        if CONFIG["SELECTION"].getboolean("run_author_match"):
            all_authors = set()
            for paper in paper_list:
                all_authors.update(set(paper.authors))
            print("Getting author info for " + str(len(all_authors)) + " authors")
            all_authors = get_authors(list(all_authors), S2_API_KEY, config=CONFIG)
        else:
            print("Skipping author info")
            all_authors = {}

        if CONFIG["OUTPUT"].getboolean("dump_debug_file"):
            with open(output_debug_file_format.format("config.json"), "w", encoding="utf-8") as outfile:
                json.dump({section: dict(CONFIG[section]) for section in CONFIG.sections()}, outfile, cls=EnhancedJSONEncoder, indent=4)
            with open(output_debug_file_format.format("author_id_set.json"), "w", encoding="utf-8") as outfile:
                json.dump(list(AUTHOR_ID_SET), outfile, cls=EnhancedJSONEncoder, indent=4)
            with open(output_debug_file_format.format("all_papers.json"), "w", encoding="utf-8") as outfile:
                json.dump(paper_list, outfile, cls=EnhancedJSONEncoder, indent=4)
            with open(output_debug_file_format.format("all_authors.json"), "w", encoding="utf-8") as outfile:
                json.dump(all_authors, outfile, cls=EnhancedJSONEncoder, indent=4)

        selected_paper_dict = {}
        filtered_paper_dict = {}

        if CONFIG["SELECTION"].getboolean("run_author_match"):
            paper_list, selected_results = select_by_author(
                all_authors,
                paper_list,
                AUTHOR_ID_SET,
                CONFIG,
            )
            selected_paper_dict.update(selected_results)
        else:
            print("Skipping selection by author")

        if CONFIG["SELECTION"].getboolean("run_author_match"):
            paper_list, filtered_results = filter_papers_by_hindex(
                all_authors,
                paper_list,
                CONFIG,
            )
            filtered_paper_dict.update(filtered_results)
        else:
            print("Skipping h-index filtering")

        if CONFIG["SELECTION"].getboolean("run_openai"):
            selected_results, filtered_results, total_prompt_cost, total_completion_cost, total_prompt_tokens, total_completion_tokens = filter_by_gpt(
                paper_list,
                SYSTEM_PROMPT,
                TOPIC_PROMPT,
                SCORE_PROMPT,
                POSTFIX_PROMPT_TITLE,
                POSTFIX_PROMPT_ABSTRACT,
                CONFIG,
            )
            selected_paper_dict.update(selected_results)
            filtered_paper_dict.update(filtered_results)
        else:
            total_prompt_cost, total_completion_cost, total_prompt_tokens, total_completion_tokens = 0.0, 0.0, 0, 0
            print("Skipping GPT filtering")

        selected_paper_dict = {
            key: value
            for key, value in sorted(
                selected_paper_dict.items(),
                key=lambda item: (item[1].get("SCORE", 0), item[1].get("RELEVANCE", 0)),
                reverse=True,
            )
        }

        if CONFIG["OUTPUT"].getboolean("dump_debug_file"):
            with open(output_debug_file_format.format("selected_paper_dict.json"), "w", encoding="utf-8") as outfile:
                json.dump(selected_paper_dict, outfile, cls=EnhancedJSONEncoder, indent=4)
            with open(output_debug_file_format.format("filtered_paper_dict.json"), "w", encoding="utf-8") as outfile:
                json.dump(filtered_paper_dict, outfile, cls=EnhancedJSONEncoder, indent=4)

        if CONFIG["OUTPUT"].getboolean("dump_json"):
            with open(output_json_file_format.format("output.json"), "w", encoding="utf-8") as outfile:
                json.dump(selected_paper_dict, outfile, indent=4)

        if CONFIG["OUTPUT"].getboolean("dump_md"):
            head_table = {
                "headers": [f"*[{CONFIG['SELECTION']['model']}]*", "Prompt", "Completion", "Total"],
                "data": [
                    ["**Token**", total_prompt_tokens, total_completion_tokens, total_prompt_tokens + total_completion_tokens],
                    [
                        "**Cost**",
                        f"${round(total_prompt_cost, 2)}",
                        f"${round(total_completion_cost, 2)}",
                        f"${round(total_prompt_cost + total_completion_cost, 2)}",
                    ],
                ],
            }
            with open(output_md_file_format.format("output.md"), "w", encoding="utf-8") as output_file:
                output_file.write(
                    "\n\n".join(
                        [
                            f"> This is a remedial run for missed papers from {begin_date[1]:02d}/{begin_date[2]:02d}/{begin_date[0]} to {end_date[1]:02d}/{end_date[2]:02d}/{end_date[0]}.\n"
                            f"> \n"
                            f"> Results generated on {NOW_MONTH:02d}/{NOW_DAY:02d}/{NOW_YEAR}.",
                            render_daily_md(
                                all_entries,
                                arxiv_paper_dict,
                                selected_paper_dict,
                                now_date=remedy_date,
                                prompts=(SYSTEM_PROMPT, POSTFIX_PROMPT_ABSTRACT, SCORE_PROMPT, TOPIC_PROMPT),
                                head_table=head_table,
                            ),
                        ]
                    )
                )

        if CONFIG["OUTPUT"].getboolean("push_to_slack"):
            if SLACK_KEY is None:
                print("Warning: push_to_slack is true, but SLACK_KEY is not set - not pushing to slack")
            else:
                push_to_slack(selected_paper_dict)

        copy_file_or_dir(output_md_file_format.format("output.md"), CONFIG["OUTPUT"]["output_path"], print_info=True)
        delete_file_or_dir(os.path.join(CONFIG["OUTPUT"]["output_path"], "output.md"))
        os.rename(
            os.path.join(CONFIG["OUTPUT"]["output_path"], os.path.basename(output_md_file_format.format("output.md"))),
            os.path.join(CONFIG["OUTPUT"]["output_path"], "output.md"),
        )

    if build_site:
        site_root = build_multipage_site(CONFIG["OUTPUT"]["output_path"])
        if site_root is not None:
            print(f"Built multipage site at {site_root}")


if __name__ == "__main__":
    parsed_args = parse_args()
    remedy_plan = load_remedy_plan(parsed_args)
    print_plan(remedy_plan)

    if not parsed_args.print_plan:
        run_remedy_plan(remedy_plan, parsed_args.output_root, parsed_args.build_site)
