"""Re-scrub ledger strings already written into archive bundles.

The gateway now reduces every provider message to one path-free line, but
bundles written before that fix still carry the raw CLI banners, and those
banners name the working directory -- which on a personal machine contains the
account name. The archive is published, so the stored copies have to be fixed
too, not just the writer.

Uses llm_gateway._scrub itself rather than a second regex: a cleanup that
disagrees with the thing it is cleaning up after is how a leak survives one of
the two.

    python scripts/scrub_archive_ledgers.py --output-root <archive>/out [--dry-run]
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arxiv_assistant.utils.llm_gateway import _scrub  # noqa: E402

LEDGER_LIST_FIELDS = ("trace", "errors")


def scrub_bundle(payload: dict) -> int:
    """Scrub in place; return how many strings changed."""
    llm = ((payload.get("meta") or {}).get("usage") or {}).get("llm")
    if not isinstance(llm, dict):
        return 0
    changed = 0
    for field in LEDGER_LIST_FIELDS:
        values = llm.get(field)
        if not isinstance(values, list):
            continue
        cleaned = []
        for value in values:
            new = _scrub(value)
            changed += (new != value)
            cleaned.append(new)
        llm[field] = cleaned
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-root", required=True, help="Archive out/ directory.")
    parser.add_argument("--dry-run", action="store_true", help="Report without writing.")
    args = parser.parse_args()

    root = Path(args.output_root)
    files_changed = 0
    strings_changed = 0

    for path in sorted(root.glob("json/*/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        changed = scrub_bundle(payload)
        if not changed:
            continue
        files_changed += 1
        strings_changed += changed
        if not args.dry_run:
            path.write_text(json.dumps(payload, indent=4, ensure_ascii=False), encoding="utf-8", newline="\n")

    verb = "would change" if args.dry_run else "changed"
    print(f"{verb} {strings_changed} ledger strings across {files_changed} bundles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
