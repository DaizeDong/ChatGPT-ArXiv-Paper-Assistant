"""Harvest arXiv metadata into a local corpus that a backfill can read offline.

WHY THIS EXISTS, and why a plain date-ranged harvest is not enough.

OAI-PMH selects records by `datestamp`, which is when the record LAST CHANGED,
not when the paper was submitted. A paper keeps exactly one current record, so
asking for datestamp 2026-07-20 does not return a paper submitted that day if it
was revised in August: its record now lives under the August datestamp and
nowhere else. MEASURED: 2608.21386, created 2026-07-20, is absent from the
2026-07-20..2026-08-03 harvest and present in the 2026-08-25 one. The loss is
not random either -- it grows with the age of the date being rebuilt, which is
precisely backwards for a backfill.

The way out is to stop slicing by datestamp per target date. Harvest the whole
datestamp range from the oldest date being rebuilt up to today ONCE, keep each
record under its `created` date, and every target date becomes a local lookup
that is complete by construction: whatever the current record's datestamp is, it
falls somewhere in that range.

The corpus lives OUTSIDE both the code repo and the archive repo. It is bulk
third-party data, it is large, and it is reproducible from arXiv at any time, so
it does not belong in version control.

Usage:
    python scripts/harvest_arxiv_corpus.py --from 2025-01-01 --until 2026-09-12
    python scripts/harvest_arxiv_corpus.py --from 2025-01-01 --resume
"""

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arxiv_assistant.apis.arxiv_oai import (  # noqa: E402
    _NS_ARXIV,
    _NS_OAI,
    _oai_get,
    _record_to_paper,
)
from arxiv_assistant.apis.corpus import default_corpus_dir  # noqa: E402


def _month_chunks(begin: date, end: date):
    """Split [begin, end] into calendar-month windows.

    Chunking is what makes a multi-hour harvest resumable at a useful grain: a
    kill costs the current month, not the whole run. Resumption tokens are NOT
    durable across restarts -- the server may expire them -- so they are used
    only within a chunk.
    """
    cursor = begin
    while cursor <= end:
        if cursor.month == 12:
            nxt = date(cursor.year + 1, 1, 1)
        else:
            nxt = date(cursor.year, cursor.month + 1, 1)
        yield cursor, min(nxt - timedelta(days=1), end)
        cursor = nxt


def harvest_chunk(oai_set: str, begin: date, end: date, out_handle, areas) -> int:
    params = {
        "verb": "ListRecords",
        "from": begin.isoformat(),
        "until": end.isoformat(),
        "metadataPrefix": "arXiv",
        "set": oai_set,
    }
    written = 0
    while True:
        response = _oai_get(params)
        root = ElementTree.fromstring(response.text)

        error = root.find(f"{_NS_OAI}error")
        if error is not None:
            if error.get("code") == "noRecordsMatch":
                return written
            raise RuntimeError(f"OAI-PMH error {error.get('code')}: {(error.text or '').strip()}")

        list_records = root.find(f"{_NS_OAI}ListRecords")
        if list_records is None:
            return written

        for record in list_records.findall(f"{_NS_OAI}record"):
            paper, created, categories = _record_to_paper(record)
            if paper is None or not created:
                continue
            if areas and not (set(categories) & areas):
                continue
            out_handle.write(json.dumps({
                "arxiv_id": paper.arxiv_id,
                "created": created,
                "title": paper.title,
                "abstract": paper.abstract,
                "authors": paper.authors,
                "categories": categories,
            }, ensure_ascii=False) + "\n")
            written += 1

        token_node = list_records.find(f"{_NS_OAI}resumptionToken")
        token = (token_node.text or "").strip() if token_node is not None else ""
        if not token:
            return written
        params = {"verb": "ListRecords", "resumptionToken": token}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--from", dest="begin", required=True, help="First datestamp to harvest (YYYY-MM-DD).")
    parser.add_argument("--until", dest="end", default=date.today().isoformat(), help="Last datestamp to harvest. Defaults to today.")
    parser.add_argument("--set", dest="oai_set", default="cs", help="OAI set to harvest. Defaults to cs.")
    parser.add_argument("--areas", default="", help="Comma-separated categories to keep. Empty keeps the whole set.")
    parser.add_argument("--corpus-dir", default=str(default_corpus_dir()), help="Where to write the corpus.")
    parser.add_argument("--resume", action="store_true", help="Skip month chunks already recorded as complete.")
    args = parser.parse_args()

    begin = datetime.strptime(args.begin, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    areas = {a.strip() for a in args.areas.split(",") if a.strip()}

    corpus_dir = Path(args.corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    state_path = corpus_dir / "harvest_state.json"

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {}
    done = set(state.get("chunks_done", [])) if args.resume else set()

    chunks = list(_month_chunks(begin, end))
    print(f"Harvesting set {args.oai_set} over {len(chunks)} month chunk(s) into {corpus_dir}", flush=True)

    for chunk_begin, chunk_end in chunks:
        key = f"{args.oai_set}:{chunk_begin.isoformat()}:{chunk_end.isoformat()}"
        if key in done:
            print(f"  [skip] {chunk_begin} -> {chunk_end}", flush=True)
            continue

        # Write to a per-chunk part file and only then record the chunk as done.
        # A chunk killed halfway leaves a partial part file that the next run
        # OVERWRITES, so a resumed harvest can never splice half a month into
        # the corpus and call it complete.
        part_path = corpus_dir / f"{args.oai_set}_{chunk_begin.strftime('%Y%m')}.jsonl"
        started = time.time()
        with open(part_path, "w", encoding="utf-8") as handle:
            written = harvest_chunk(args.oai_set, chunk_begin, chunk_end, handle, areas)

        done.add(key)
        state["chunks_done"] = sorted(done)
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"  [ok  ] {chunk_begin} -> {chunk_end}: {written} records in {time.time() - started:.0f}s", flush=True)

    print("Harvest complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
