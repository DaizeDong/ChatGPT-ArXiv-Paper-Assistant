"""Rewrite the usage table at the top of already-rendered daily markdown."""

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arxiv_assistant.renderers.paper.render_daily import render_summary_table  # noqa: E402
from arxiv_assistant.utils.llm_gateway import public_model_label  # noqa: E402

#: The remedial writer's hand-rolled pipe table: a header naming the model, a
#: separator row, then Token and Cost rows.
PIPE_TABLE = re.compile(
    r"^\| \*\[[^\]]*\]\*.*\n\|[:\-| ]+\n\| \*\*Token\*\*.*\n\| \*\*Cost\*\*.*\n",
    re.MULTILINE,
)
#: The daily writer's HTML table, which may also name a stale model.
HTML_TABLE = re.compile(r"^<table>\n(?:.*\n)*?</table>\n(?:<sub>.*</sub>\n)?", re.MULTILINE)


def _usage_of(bundle_path: Path) -> dict:
    try:
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return ((payload.get("meta") or {}).get("usage") or {})


def build_table(usage: dict) -> str:
    llm = usage.get("llm") or {}
    return render_summary_table(
        model=public_model_label(usage.get("model")),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        prompt_cost=float(usage.get("prompt_cost") or 0.0),
        completion_cost=float(usage.get("completion_cost") or 0.0),
        total_arxiv_papers=int(usage.get("total_arxiv_papers") or 0),
        total_scanned_papers=int(usage.get("total_scanned_papers") or 0),
        total_relevant_papers=int(usage.get("total_relevant_papers") or 0),
        calls_attempted=int(llm.get("attempted") or 0),
        calls_succeeded=int(llm.get("succeeded") or 0),
        seconds=float(llm.get("seconds") or 0.0),
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_root)
    rewritten = skipped_no_bundle = skipped_no_table = 0

    # Only the remedial render. The `-latest.md` beside it is the ORIGINAL
    # failed run's page -- its body still says no papers were selected, and
    # giving it a rebuilt day's usage table would produce a page that
    # contradicts itself. The site publishes the remedial render anyway.
    for md_path in sorted(root.glob("md/*/*-output.md")):
        date = md_path.name[:10]
        month = date[:7]
        bundle = root / "json" / month / f"{date}-daily-papers.json"
        usage = _usage_of(bundle)
        # Only days whose bundle carries a health record were rebuilt by this
        # pipeline. A legacy day's table is a true record of its own run and
        # must not be overwritten with today's vocabulary.
        if not usage or not usage.get("filter_health"):
            skipped_no_bundle += 1
            continue

        text = md_path.read_text(encoding="utf-8")
        table = build_table(usage)
        new_text, n = PIPE_TABLE.subn(table, text, count=1)
        if not n:
            new_text, n = HTML_TABLE.subn(table, text, count=1)
        if not n:
            skipped_no_table += 1
            continue

        rewritten += 1
        if not args.dry_run:
            md_path.write_text(new_text, encoding="utf-8", newline="\n")

    verb = "would rewrite" if args.dry_run else "rewrote"
    print(f"{verb} {rewritten} usage tables; "
          f"skipped {skipped_no_bundle} without a rebuilt bundle, "
          f"{skipped_no_table} with no table found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
