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
    parser.add_argument(
        "--skip-done",
        action="store_true",
        help=(
            "Skip dates already rebuilt by a previous remedy run. A day counts as "
            "done only when its bundle carries a filter_health record, which is "
            "what a remedied day writes and what a legacy or empty day does not, "
            "so this resumes a killed run without redoing finished work and "
            "without mistaking an old empty archive for a completed one."
        ),
    )
    parser.add_argument(
        "--skip-latest-copy",
        action="store_true",
        help=(
            "Do not refresh the root out/output.md. Set automatically for the "
            "children of --jobs, where that shared path is a race between "
            "concurrent dates and means nothing for a historical backfill."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "Remedy this many dates concurrently, each in its own process. Dates "
            "are independent, so this is the only lever that makes a hundred-day "
            "backfill finish in hours instead of days. Separate PROCESSES rather "
            "than threads because the pipeline keeps module-level state (the "
            "config singleton, the call ledger, the filter's rate-limit counters) "
            "that is not safe to share. The usable number depends on what else is "
            "running on the machine, not on the machine's size: this is clamped "
            "at startup against actually-free memory (see MB_PER_JOB)."
        ),
    )
    parser.add_argument(
        "--source",
        choices=("corpus", "oai", "api"),
        default="corpus",
        help=(
            "Where to get the window's papers. Default \"corpus\" reads a "
            "locally harvested corpus indexed by SUBMISSION date, which is the "
            "only source that stays complete for an old date: both network "
            "sources lose papers that were revised after the date being rebuilt "
            "(measured), and the search endpoint additionally rate limited this "
            "host for hours mid-backfill. \"oai\" harvests over the network, "
            "\"api\" is the interactive Atom search endpoint; use either only to "
            "reproduce an old run or when no corpus has been built."
        ),
    )
    return parser.parse_args()


#: Peak resident cost of one remedy job, MEASURED against a running backfill by
#: sampling the whole process subtree (worker python + the model subprocess it
#: spawns per batch) every 4s and keeping the maximum: 1124 MB across 3 jobs,
#: so ~375. Set to 400 with margin.
#:
#: Two earlier values here were wrong in opposite directions and both were
#: guesses. 1400 throttled the tool to a third of what the machine could carry;
#: 700 was still nearly double. Sampling the STEADY state alone gives ~113 and
#: would let far too many jobs start, because what kills a run is the moment
#: every worker happens to hold a model subprocess at once.
MB_PER_JOB = 400
#: Never plan to consume the last of the machine.
MB_HEADROOM = 2000


def free_memory_mb() -> int | None:
    """Actually-free physical memory, or None when it cannot be determined.

    Returns None rather than a guess: clamping against a fabricated number would
    be worse than not clamping, because it would look like a considered decision.
    """
    try:
        if sys.platform == "win32":
            import ctypes

            class _Status(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _Status()
            status.dwLength = ctypes.sizeof(_Status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            return int(status.ullAvailPhys // (1024 * 1024))
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        return None
    return None


def clamp_jobs_to_memory(requested: int) -> int:
    """Reduce --jobs to what free memory can actually hold.

    A backfill killed halfway is worse than a slow one: it leaves an archive in
    a state nobody has counted, and the operator finds out from a task
    notification rather than from the tool. So the clamp happens up front and
    says what it did.
    """
    free_mb = free_memory_mb()
    if free_mb is None:
        print("Could not read free memory; leaving --jobs as requested.", flush=True)
        return requested
    affordable = max(1, (free_mb - MB_HEADROOM) // MB_PER_JOB)
    if affordable >= requested:
        print(f"Free memory {free_mb} MB; running {requested} job(s).", flush=True)
        return requested
    print(
        f"Free memory is {free_mb} MB. At ~{MB_PER_JOB} MB per job with "
        f"{MB_HEADROOM} MB headroom that affords {affordable}, not {requested}. "
        f"Clamping to {affordable}. Close memory-heavy apps, or pass --jobs "
        f"anyway on a quieter machine, to go faster.",
        flush=True,
    )
    return affordable


def already_remedied(output_root: str, remedy_date: DateTuple) -> bool:
    """True when this date's bundle already carries a remedy health record.

    Presence of `filter_health` is the signal because it is written only by a run
    that went through the outage gate. File existence alone would be wrong: the
    80-odd days this backfill exists to repair all HAVE a file, and it is two
    bytes of nothing.
    """
    label = f"{remedy_date[0]:04d}-{remedy_date[1]:02d}-{remedy_date[2]:02d}"
    path = (
        Path(output_root) / "json" / f"{remedy_date[0]:04d}-{remedy_date[1]:02d}"
        / f"{label}-daily-papers.json"
    )
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return bool(payload.get("meta", {}).get("usage", {}).get("filter_health"))


def run_plan_in_parallel(plan: RemedyPlan, args: argparse.Namespace) -> int:
    """Fan the plan out over `args.jobs` child processes, one date each.

    Each child is this same script with --jobs 1 and a single explicit window, so
    a child never re-infers a window from an output tree its siblings are writing
    into at the same time.
    """
    import concurrent.futures
    import subprocess

    items = sorted(plan.items())
    jobs = clamp_jobs_to_memory(max(1, args.jobs))
    print(f"Remedying {len(items)} dates with {jobs} concurrent job(s)", flush=True)

    def run_one(item) -> tuple[str, int, str]:
        remedy_date, (begin_date, end_date) = item
        label = f"{remedy_date[0]:04d}-{remedy_date[1]:02d}-{remedy_date[2]:02d}"
        cmd = [
            sys.executable, "-X", "utf8", str(Path(__file__).resolve()),
            "--date", label,
            "--begin-date", f"{begin_date[0]:04d}-{begin_date[1]:02d}-{begin_date[2]:02d}",
            "--end-date", f"{end_date[0]:04d}-{end_date[1]:02d}-{end_date[2]:02d}",
            "--output-root", args.output_root,
            "--jobs", "1",
            "--skip-latest-copy",
            # The child re-parses its own args, so a source chosen on the parent
            # is NOT inherited: without this the fan-out would quietly fall back
            # to the default while the parent reported the source it was told.
            "--source", getattr(args, "source", "corpus"),
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        # On success the last few stdout lines are enough. On FAILURE they are
        # actively misleading: the traceback goes to stderr, so keeping only
        # stdout reports the last thing that WORKED and hides the reason. A
        # twelve-date failure run was undiagnosable for exactly this reason --
        # every line said "Getting papers from ..." and the ReadTimeout that
        # actually killed them was discarded here.
        if proc.returncode == 0:
            tail = (proc.stdout or "").strip().splitlines()[-3:]
        else:
            err = (proc.stderr or "").strip().splitlines()
            tail = err[-4:] if err else (proc.stdout or "").strip().splitlines()[-3:]
        return label, proc.returncode, " | ".join(t.strip() for t in tail)

    failures: list[str] = []
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(run_one, item) for item in items]
        # as_completed, NOT map: map yields in submission order, so a slow first
        # date hides every date that finished behind it and the log reads as if
        # nothing is happening for half an hour.
        for future in concurrent.futures.as_completed(futures):
            label, code, tail = future.result()
            completed += 1
            status = "ok " if code == 0 else "FAIL"
            print(f"  [{status}] {label}  ({completed}/{len(items)})  rc={code}  {tail}", flush=True)
            if code != 0:
                failures.append(label)

    if failures:
        print(
            f"\n{len(failures)} of {len(items)} dates FAILED and must be redone:\n  "
            + ", ".join(failures),
            file=sys.stderr, flush=True,
        )
        return 1
    print(f"\nAll {len(items)} dates remedied.", flush=True)
    return 0


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


def run_remedy_plan(plan: RemedyPlan, output_root: str, build_site: bool, skip_latest_copy: bool = False, source: str = "corpus") -> int:
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
    from arxiv_assistant.paper_topics import build_daily_topic_bundle, build_hotspot_paper_bundle, ensure_topic_fields_for_mapping, sort_paper_mapping_for_daily_display
    from arxiv_assistant.push_to_slack import push_to_slack
    from arxiv_assistant.renderers.build_multipage_site import build_multipage_site
    from arxiv_assistant.renderers.paper.render_daily import render_daily_md
    from arxiv_assistant.utils.io import copy_file_or_dir, create_dir, delete_file_or_dir
    from arxiv_assistant.utils.llm_gateway import (
        BACKEND_OPENAI,
        LEDGER,
        describe_backend,
        resolve_backend,
    )
    from arxiv_assistant.utils.pipeline_health import assess_paper_filter_health, format_banner
    from arxiv_assistant.utils.utils import EnhancedJSONEncoder

    CONFIG["OUTPUT"]["output_path"] = output_root
    print(describe_backend(CONFIG), flush=True)

    # Days whose scoring never actually ran. Collected rather than merely printed:
    # over a hundred-day backfill a per-day banner scrolls past, and the operator
    # needs one list at the end saying which days must be redone.
    outage_dates: list[str] = []

    for remedy_date, (begin_date, end_date) in sorted(plan.items()):
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print(f"Start remedying for date: {remedy_date}")
        print(f"Searching date range: {begin_date} - {end_date}")
        # Per-day ledger: the archive each day writes must describe that day's
        # calls, not the running total since the process started.
        LEDGER.reset()

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
            source=source,
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

        hotspot_paper_bundle = build_hotspot_paper_bundle(remedy_date, {})

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

            scored_results_for_hotspot = ensure_topic_fields_for_mapping(
                {
                    arxiv_id: paper_entry
                    for arxiv_id, paper_entry in {**selected_results, **filtered_results}.items()
                    if "RELEVANCE" in paper_entry or "NOVELTY" in paper_entry
                }
            )
            hotspot_paper_bundle = build_hotspot_paper_bundle(
                remedy_date,
                scored_results_for_hotspot,
                max_daily_hot=CONFIG["HOTSPOTS"].getint("paper_spotlight_max_daily_hot", fallback=6),
                max_new_frontier=CONFIG["HOTSPOTS"].getint("paper_spotlight_max_new_frontier", fallback=4),
                daily_hot_score_cutoff=CONFIG["HOTSPOTS"].getint("paper_spotlight_daily_hot_score_cutoff", fallback=15),
                daily_hot_relevance_cutoff=CONFIG["HOTSPOTS"].getint("paper_spotlight_daily_hot_relevance_cutoff", fallback=7),
                new_frontier_score_cutoff=CONFIG["HOTSPOTS"].getint("paper_spotlight_new_frontier_score_cutoff", fallback=15),
                new_frontier_novelty_cutoff=CONFIG["HOTSPOTS"].getint("paper_spotlight_new_frontier_novelty_cutoff", fallback=8),
            )
            hotspot_paper_ids = set(hotspot_paper_bundle["papers"].keys())
            if hotspot_paper_ids:
                selected_paper_dict = {
                    arxiv_id: paper_entry
                    for arxiv_id, paper_entry in selected_paper_dict.items()
                    if arxiv_id not in hotspot_paper_ids
                }
                for arxiv_id in hotspot_paper_ids:
                    spotlight_entry = dict(scored_results_for_hotspot[arxiv_id])
                    spotlight_entry["DIVERTED_TO_HOTSPOT_PAPERS"] = True
                    spotlight_entry["FILTER_REASON"] = "Diverted to the daily hotspot paper spotlight."
                    filtered_paper_dict[arxiv_id] = {
                        **filtered_paper_dict.get(arxiv_id, {}),
                        **spotlight_entry,
                    }
        else:
            total_prompt_cost, total_completion_cost, total_prompt_tokens, total_completion_tokens = 0.0, 0.0, 0, 0
            print("Skipping GPT filtering")

        selected_paper_dict = sort_paper_mapping_for_daily_display(selected_paper_dict)
        filtered_paper_dict = ensure_topic_fields_for_mapping(filtered_paper_dict)
        # Same outage gate main.py carries. A backfill is exactly where a silent
        # failure is most expensive: it writes an authoritative-looking archive
        # for a day that can no longer be distinguished from a real one, and it
        # does so for a hundred days in a row without anyone watching.
        total_scanned_papers = sum(len(area_papers) for area_papers in arxiv_paper_dict.values())
        filter_health = assess_paper_filter_health(
            scanned_papers=total_scanned_papers,
            selected_papers=len(selected_paper_dict),
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            llm_filtering_enabled=(
                CONFIG["SELECTION"].getboolean("run_openai")
                and (
                    CONFIG["SELECTION"].getboolean("run_title_filter")
                    or CONFIG["SELECTION"].getboolean("run_abstract_filter")
                )
            ),
            llm_calls_attempted=LEDGER.attempted,
            llm_calls_succeeded=LEDGER.succeeded,
        )
        if filter_health.is_outage:
            print(format_banner(filter_health), file=sys.stderr, flush=True)
            outage_dates.append(f"{remedy_year}-{remedy_month:02d}-{remedy_day:02d}")

        daily_topic_bundle = build_daily_topic_bundle(
            remedy_date,
            selected_paper_dict,
            usage={
                "model": (
                    CONFIG["SELECTION"]["model"]
                    if resolve_backend(CONFIG) == BACKEND_OPENAI
                    else f"{resolve_backend(CONFIG)}:{'+'.join(LEDGER.answering_providers()) or 'none'}"
                ),
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "prompt_cost": total_prompt_cost,
                "completion_cost": total_completion_cost,
                "total_arxiv_papers": len(all_entries),
                "total_scanned_papers": total_scanned_papers,
                "total_relevant_papers": len(selected_paper_dict),
                "filter_health": filter_health.to_dict(),
                "llm": {"backend": describe_backend(CONFIG), **LEDGER.to_dict()},
                "remedied": True,
            },
        )

        if CONFIG["OUTPUT"].getboolean("dump_debug_file"):
            with open(output_debug_file_format.format("selected_paper_dict.json"), "w", encoding="utf-8") as outfile:
                json.dump(selected_paper_dict, outfile, cls=EnhancedJSONEncoder, indent=4)
            with open(output_debug_file_format.format("filtered_paper_dict.json"), "w", encoding="utf-8") as outfile:
                json.dump(filtered_paper_dict, outfile, cls=EnhancedJSONEncoder, indent=4)
            with open(output_debug_file_format.format("topic_diagnostics.json"), "w", encoding="utf-8") as outfile:
                json.dump(daily_topic_bundle["diagnostics"], outfile, indent=4)
            with open(output_debug_file_format.format("hotspot_paper_bundle.json"), "w", encoding="utf-8") as outfile:
                json.dump(hotspot_paper_bundle, outfile, indent=4)

        if CONFIG["OUTPUT"].getboolean("dump_json"):
            with open(output_json_file_format.format("output.json"), "w", encoding="utf-8") as outfile:
                json.dump(selected_paper_dict, outfile, indent=4)
            with open(output_json_file_format.format("daily-papers.json"), "w", encoding="utf-8") as outfile:
                json.dump(daily_topic_bundle, outfile, indent=4)
            with open(output_json_file_format.format("hotspot-papers.json"), "w", encoding="utf-8") as outfile:
                json.dump(hotspot_paper_bundle, outfile, indent=4)

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

        # The root out/output.md is a "most recent run" convenience copy. Under
        # --jobs it is a RACE: every child copies, deletes and renames the same
        # path, so one child can delete the file another is about to rename and
        # fail a date that actually scored fine. It is also meaningless for a
        # backfill, where "most recent" would just be whichever historical date
        # happened to finish last. The per-date file under md/ is the real
        # artifact and is written either way.
        if skip_latest_copy:
            print("Skipping the root output.md copy (parallel backfill)")
        else:
            copy_file_or_dir(output_md_file_format.format("output.md"), CONFIG["OUTPUT"]["output_path"], print_info=True)
            delete_file_or_dir(os.path.join(CONFIG["OUTPUT"]["output_path"], "output.md"))
            os.rename(
                os.path.join(CONFIG["OUTPUT"]["output_path"], os.path.basename(output_md_file_format.format("output.md"))),
                os.path.join(CONFIG["OUTPUT"]["output_path"], "output.md"),
            )

    if outage_dates:
        print("", file=sys.stderr)
        print(
            f"REMEDY INCOMPLETE: scoring never ran for {len(outage_dates)} of "
            f"{len(plan)} dates. These archives were written EMPTY and are not "
            f"evidence that nothing was relevant:\n  " + ", ".join(outage_dates),
            file=sys.stderr, flush=True,
        )

    if build_site:
        site_root = build_multipage_site(CONFIG["OUTPUT"]["output_path"])
        if site_root is not None:
            print(f"Built multipage site at {site_root}")

    # Non-zero when any day failed, so a batch driver and CI can both see it.
    return 1 if outage_dates else 0


if __name__ == "__main__":
    parsed_args = parse_args()
    remedy_plan = load_remedy_plan(parsed_args)
    print_plan(remedy_plan)

    if parsed_args.skip_done:
        before = len(remedy_plan)
        remedy_plan = {
            d: w for d, w in remedy_plan.items()
            if not already_remedied(parsed_args.output_root, d)
        }
        print(f"--skip-done: {before - len(remedy_plan)} already remedied, {len(remedy_plan)} to go")
        if not remedy_plan:
            print("Nothing left to remedy.")
            raise SystemExit(0)

    if parsed_args.print_plan:
        raise SystemExit(0)
    if parsed_args.jobs > 1 and len(remedy_plan) > 1:
        raise SystemExit(run_plan_in_parallel(remedy_plan, parsed_args))
    raise SystemExit(
        run_remedy_plan(
            remedy_plan,
            parsed_args.output_root,
            parsed_args.build_site,
            skip_latest_copy=parsed_args.skip_latest_copy,
            source=parsed_args.source,
        ) or 0
    )
