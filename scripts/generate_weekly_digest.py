"""Build the weekly delta digest from N days of archive."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arxiv_assistant.reader.delta import (
    DeltaCandidate,
    DeltaStatus,
    DeltaVerdict,
    candidates_from_hotspot_report,
    candidates_from_paper_mapping,
    rank,
    resolve_reader_settings,
    score_candidates,
)
from arxiv_assistant.reader.questions import ReaderQuestionError, load_questions
from arxiv_assistant.renderers.weekly.render_weekly_digest import render_weekly_digest_md
from arxiv_assistant.utils import llm_gateway
from arxiv_assistant.utils.hotspot.hotspot_config import load_repo_config, repo_root
from arxiv_assistant.utils.console import force_utf8_stdio
from arxiv_assistant.utils.local_env import load_local_env

load_local_env()

REPO_ROOT = repo_root()


# ---------------------------------------------------------------------------
# Archive access
# ---------------------------------------------------------------------------


def resolve_archive_dir(archive_root: Path) -> Path:
    """Return the directory that directly contains ``json/`` and ``hot/``."""
    archive_root = Path(archive_root)
    if (archive_root / "json").is_dir() or (archive_root / "hot").is_dir():
        return archive_root
    nested = archive_root / "out"
    if (nested / "json").is_dir() or (nested / "hot").is_dir():
        return nested
    # Neither shape exists. Return the literal path so the caller reports
    # "0 days found" against the directory the operator actually typed.
    return archive_root


def _read_json(path: Path) -> Any:
    """Read one archive file, or None when it is absent or corrupt.

    A corrupt file is reported to stderr rather than swallowed: it is a
    different failure from a missing one, and the day still counts as missing.
    """
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"WARNING: unreadable archive file {path}: {exc}", file=sys.stderr)
        return None


def enumerate_days(end_date: date, days: int) -> List[date]:
    """The N calendar days ending at (and including) *end_date*, oldest first."""
    if days < 1:
        raise ValueError(f"--days must be >= 1, got {days}")
    return [end_date - timedelta(days=offset) for offset in range(days - 1, -1, -1)]


def _paper_usage(daily_papers: Any) -> Tuple[int, int, int]:
    """Return (llm_tokens, scanned_papers, produced_papers) from -daily-papers.json.

    Only exists from 2026-04-01 on; an older day yields zeros and is not, on its
    own, evidence of an outage -- which is why the outage test in the renderer
    also requires ``scanned > 0``.
    """
    if not isinstance(daily_papers, Mapping):
        return 0, 0, 0
    meta = daily_papers.get("meta")
    meta = meta if isinstance(meta, Mapping) else {}
    usage = meta.get("usage")
    usage = usage if isinstance(usage, Mapping) else {}

    def _num(value: Any) -> int:
        return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0

    tokens = _num(usage.get("prompt_tokens")) + _num(usage.get("completion_tokens"))
    scanned = _num(usage.get("total_scanned_papers"))
    produced = _num(meta.get("total_papers")) or _num(usage.get("total_relevant_papers"))
    return tokens, scanned, produced


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def collect_candidates(
    archive_dir: Path, days: Sequence[date]
) -> Tuple[List[DeltaCandidate], Dict[str, Any]]:
    """Load every day in the window. Returns (candidates, diagnostics-so-far)."""
    candidates: List[DeltaCandidate] = []
    days_found = 0
    days_missing: List[str] = []
    paper_days_empty: List[str] = []
    tokens_total = 0
    scanned_total = 0
    produced_total = 0

    for day in days:
        iso = day.isoformat()
        month_dir = archive_dir / "json" / day.strftime("%Y-%m")
        paper_mapping = _read_json(month_dir / f"{iso}-output.json")
        daily_papers = _read_json(month_dir / f"{iso}-daily-papers.json")
        hotspot_report = _read_json(archive_dir / "hot" / "reports" / f"{iso}.json")

        found_anything = paper_mapping is not None or hotspot_report is not None
        if not found_anything:
            days_missing.append(iso)
            continue
        days_found += 1

        if isinstance(paper_mapping, Mapping):
            if not paper_mapping:
                # The literal two-byte {}. Normal on a weekend, and the exact
                # shape of the three-month outage on a weekday; the token
                # counters below are what tells those two apart.
                paper_days_empty.append(iso)
            else:
                candidates.extend(candidates_from_paper_mapping(paper_mapping, date=iso))

        tokens, scanned, produced = _paper_usage(daily_papers)
        tokens_total += tokens
        scanned_total += scanned
        produced_total += produced

        if isinstance(hotspot_report, Mapping):
            candidates.extend(candidates_from_hotspot_report(hotspot_report))

    diagnostics: Dict[str, Any] = {
        "days_requested": len(days),
        "days_found": days_found,
        "days_missing": days_missing,
        "paper_days_empty": paper_days_empty,
        "paper_llm_tokens_seen": tokens_total,
        "paper_scanned_seen": scanned_total,
        "paper_papers_seen": produced_total,
    }
    return candidates, diagnostics


def _item_row(candidate: DeltaCandidate, verdict: DeltaVerdict) -> Dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "kind": candidate.kind,
        "title": candidate.title,
        "url": candidate.url,
        "date": candidate.date,
        "delta_score": verdict.delta_score,
        "question_id": verdict.question_id,
        "field": verdict.field,
        "one_line_reason": verdict.one_line_reason,
        "tiebreak": candidate.tiebreak,
    }


def build_weekly_digest(
    *,
    archive_dir: Path,
    end_date: date,
    days: int,
    config: Any,
    agent_fn=None,
    questions_dir: Path | None = None,
) -> Dict[str, Any]:
    """Assemble the digest dict the renderer consumes.

    Every "why is it empty" fact is computed HERE and only here; the renderer
    reads the diagnostics block and never re-derives a cause of its own. One
    place to compute, one place to display.
    """
    # The ledger is a per-process counter, and this digest reports on ITS OWN
    # run. Resetting here is what makes "attempted N, succeeded 0" a statement
    # about this window rather than about everything the interpreter has done
    # since it started.
    llm_gateway.LEDGER.reset()

    # agent_fn stays None in production: the gateway resolves the backend from
    # [LLM]. Passing one pins the agent transport, which is what the tests want
    # and what a deploy that has lost llmcall can fall back to.
    settings = resolve_reader_settings(config)
    window = enumerate_days(end_date, days)
    candidates, diagnostics = collect_candidates(archive_dir, window)

    # An explicit override lets you preview a digest against a DRAFT reader model
    # without editing the committed one. The question documents are human-only, so
    # the alternative would be temporarily overwriting them, which is exactly the
    # kind of edit that gets forgotten.
    questions_dir = Path(questions_dir) if questions_dir else Path(settings.questions_dir)
    if not questions_dir.is_absolute():
        questions_dir = REPO_ROOT / questions_dir
    questions = load_questions(questions_dir)
    populated = [question for question in questions if question.is_populated]

    verdicts = score_candidates(
        candidates,
        questions,
        model=settings.model,
        timeout_s=settings.timeout_s,
        config=config,
        agent_fn=agent_fn,
    )

    status_counts = {DeltaStatus.SCORED: 0, DeltaStatus.REJECTED: 0, DeltaStatus.UNAVAILABLE: 0}
    for verdict in verdicts.values():
        if verdict.status in status_counts:
            status_counts[verdict.status] += 1

    kept = [
        candidate
        for candidate in candidates
        if (verdict := verdicts.get(candidate.candidate_id)) is not None
        and verdict.passes
        and verdict.delta_score >= settings.delta_score_cutoff
    ]
    ranked = rank(kept, verdicts)
    deep_read = ranked[: settings.max_deep_read]
    skim = ranked[settings.max_deep_read : settings.max_deep_read + settings.max_skim]

    # Scored, but under the bar. These stay out of the digest and in the archive;
    # the closest ten are what you read when deciding whether the bar is right.
    below_cutoff = [
        candidate
        for candidate in candidates
        if (verdict := verdicts.get(candidate.candidate_id)) is not None
        and verdict.passes
        and verdict.delta_score < settings.delta_score_cutoff
    ]
    near_misses = rank(below_cutoff, verdicts)[:10]

    diagnostics.update(
        {
            "candidates_scored": len(candidates),
            "verdict_status_counts": {
                "scored": status_counts[DeltaStatus.SCORED],
                "rejected": status_counts[DeltaStatus.REJECTED],
                "unavailable": status_counts[DeltaStatus.UNAVAILABLE],
            },
            "reader_questions_populated": len(populated),
            "reader_questions_total": len(questions),
            "delta_score_cutoff": settings.delta_score_cutoff,
            "max_deep_read": settings.max_deep_read,
            "max_skim": settings.max_skim,
            "items_over_cutoff": len(kept),
            "archive_dir": str(archive_dir),
            # Without the distribution you cannot tell "the scorer returned 0 for
            # everything" from "it scored 4-6 and the bar is at 7". Both produce an
            # empty digest; only one of them means the cutoff needs moving. The
            # histogram is what makes delta_score_cutoff tunable with evidence
            # instead of by feel.
            "score_histogram": _score_histogram(verdicts),
            # WHICH model answered, and whether any did. The token counters above
            # only ever described the OpenAI path; on the llmcall chain they are
            # structurally zero, so an outage detector that read them would either
            # fire every week or never. The ledger says the provider-agnostic
            # thing instead: N calls attempted, M succeeded, and here are the
            # errors. "Every provider in the chain failed" is a cause the renderer
            # can name out loud.
            "llm_backend": llm_gateway.describe_backend(config),
            "llm_ledger": llm_gateway.LEDGER.to_dict(),
        }
    )

    return {
        "schema_version": 1,
        "week_start": window[0].isoformat(),
        "week_end": window[-1].isoformat(),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "deep_read": [_item_row(c, verdicts[c.candidate_id]) for c in deep_read],
        "skim": [_item_row(c, verdicts[c.candidate_id]) for c in skim],
        "near_misses": [
            _item_row(candidate, verdicts[candidate.candidate_id])
            for candidate in near_misses
        ],
        "diagnostics": diagnostics,
    }


def _score_histogram(verdicts: Mapping[str, DeltaVerdict]) -> Dict[str, int]:
    """Count of scored verdicts per delta score, plus the non-scored statuses."""
    histogram: Dict[str, int] = {str(score): 0 for score in range(11)}
    for verdict in verdicts.values():
        if verdict.passes:
            histogram[str(verdict.delta_score)] += 1
        else:
            histogram[verdict.status] = histogram.get(verdict.status, 0) + 1
    return histogram


# ---------------------------------------------------------------------------
# Slack
# ---------------------------------------------------------------------------


def push_digest_to_slack(digest: Mapping[str, Any], markdown: str) -> None:
    """Post the digest to Slack, gated on the NEW ``[OUTPUT] push_weekly_to_slack``."""
    from arxiv_assistant.environment import SLACK_CHANNEL_ID, SLACK_KEY
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError

    if not SLACK_KEY or not SLACK_CHANNEL_ID:
        raise RuntimeError(
            "push_weekly_to_slack is true but SLACK_KEY / SLACK_CHANNEL_ID are unset. "
            "Refusing to exit 0 on a send that never happened."
        )

    header = f"Weekly delta digest {digest.get('week_start')} ~ {digest.get('week_end')}"
    # Slack hard-caps a text block at 3000 characters; truncate visibly rather
    # than letting the API reject the whole message.
    body = markdown if len(markdown) <= 2900 else markdown[:2900] + "\n...(truncated)"
    client = WebClient(token=SLACK_KEY)
    try:
        client.chat_postMessage(
            channel=SLACK_CHANNEL_ID,
            text=header,
            blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": body}}],
            unfurl_links=False,
        )
    except SlackApiError as exc:
        # Loud, not swallowed: a digest that was never delivered must not look
        # like a delivered one.
        raise RuntimeError(f"Slack post failed: {exc}") from exc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the weekly delta digest from N days of archive."
    )
    parser.add_argument("--output-root", default="out", help="Where to WRITE the digest.")
    parser.add_argument(
        "--archive-root",
        default=None,
        help="Where to READ out/ from (defaults to --output-root). The archive lives on "
        "the data branch, so reads and writes can point at different trees.",
    )
    parser.add_argument(
        "--end-date", default=None, help="Last day of the window, YYYY-MM-DD (default: today)."
    )
    parser.add_argument("--days", type=int, default=7, help="Window length in days.")
    parser.add_argument(
        "--force", action="store_true", help="Rewrite even when this week's files already exist."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the markdown; write nothing."
    )
    parser.add_argument(
        "--questions-dir",
        default=None,
        help="Override [READER] questions_dir. Use it to preview a digest against a "
        "draft reader model without touching the committed question documents.",
    )
    return parser.parse_args(argv)


def _parse_end_date(raw: str | None) -> date:
    if not raw:
        return date.today()
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"--end-date must be YYYY-MM-DD, got {raw!r}") from exc


def main() -> None:
    force_utf8_stdio()  # the archive is full of Chinese titles; see utils/console.py
    args = parse_args()
    config = load_repo_config(REPO_ROOT / "configs" / "config.ini")

    end_date = _parse_end_date(args.end_date)
    output_root = Path(args.output_root)
    archive_dir = resolve_archive_dir(Path(args.archive_root or args.output_root))
    print(f"Reading archive from: {archive_dir}")

    md_path = output_root / "weekly" / end_date.strftime("%Y-%m") / f"{end_date.isoformat()}-weekly.md"
    json_path = md_path.with_suffix(".json")
    if md_path.is_file() and json_path.is_file() and not args.force and not args.dry_run:
        print(f"Already present (use --force to rewrite): {md_path.resolve()}")
        return

    try:
        digest = build_weekly_digest(
            archive_dir=archive_dir,
            end_date=end_date,
            days=args.days,
            config=config,
            questions_dir=Path(args.questions_dir) if args.questions_dir else None,
        )
    except ReaderQuestionError as exc:
        # A missing questions directory is a deployment fault, not a quiet week.
        # Exit non-zero so a scheduler notices instead of archiving a blank page.
        raise SystemExit(f"ERROR: {exc}") from exc

    markdown = render_weekly_digest_md(digest)

    if args.dry_run:
        print(markdown)
        print(json.dumps(digest["diagnostics"], ensure_ascii=False, indent=2, sort_keys=True))
        return

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(
        json.dumps(digest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if config.has_section("OUTPUT") and config["OUTPUT"].getboolean(
        "push_weekly_to_slack", fallback=False
    ):
        push_digest_to_slack(digest, markdown)
        print("Pushed weekly digest to Slack.")

    print(json_path.resolve())
    print(md_path.resolve())


if __name__ == "__main__":
    main()
