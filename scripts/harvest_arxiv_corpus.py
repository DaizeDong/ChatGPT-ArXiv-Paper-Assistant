"""Harvest arXiv metadata into a local corpus that a backfill can read offline.

nowhere else. MEASURED: 2608.21386, created 2026-07-20, is absent from the
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

from arxiv_assistant.apis.arxiv_oai import NS_OAI, oai_get, record_to_paper  # noqa: E402
from arxiv_assistant.apis.corpus import default_corpus_dir  # noqa: E402


def _month_chunks(begin: date, end: date):
    """Split [begin, end] into calendar-month windows."""
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
        response = oai_get(params)
        root = ElementTree.fromstring(response.text)

        error = root.find(f"{NS_OAI}error")
        if error is not None:
            if error.get("code") == "noRecordsMatch":
                return written
            raise RuntimeError(f"OAI-PMH error {error.get('code')}: {(error.text or '').strip()}")

        list_records = root.find(f"{NS_OAI}ListRecords")
        if list_records is None:
            return written

        for record in list_records.findall(f"{NS_OAI}record"):
            paper, created, categories = record_to_paper(record)
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

        token_node = list_records.find(f"{NS_OAI}resumptionToken")
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
