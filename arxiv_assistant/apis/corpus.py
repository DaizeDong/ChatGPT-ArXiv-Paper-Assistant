"""Read papers for a date window out of a locally harvested arXiv corpus.

The corpus is produced by scripts/harvest_arxiv_corpus.py. Its whole reason for
existing is that arXiv's OAI-PMH selects on `datestamp` (last modified) while a
daily digest wants `created` (submitted): a paper revised after the target date
is invisible to any date-ranged harvest of that date, and that loss grows with
how old the date is. Harvesting the full datestamp range once and indexing by
`created` removes the hole rather than narrowing it.

Nothing here touches the network, so a backfill reading the corpus cannot be
rate limited and cannot half-finish a date because a fetch failed.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

from arxiv_assistant.utils.utils import Paper

#: Where the corpus lives when nothing says otherwise.
#:
#: OUTSIDE both the code repo and the archive repo, deliberately: it is bulk
#: third-party metadata, it is hundreds of megabytes, and it can be rebuilt from
#: arXiv at any time. None of those belong in version control.
CORPUS_DIR_ENV = "ARXIV_ASSISTANT_CORPUS"
_DEFAULT_CORPUS_DIR = Path.home() / "CodesSelf" / "arxiv-assistant-corpus"


class CorpusUnavailable(RuntimeError):
    """The corpus cannot answer for the requested window.

    A distinct type because "the corpus has no papers for this day" and "there
    is no corpus" must not read the same to a caller: the first is a real
    answer about a quiet day, the second means the run never looked anywhere.
    """


def default_corpus_dir() -> Path:
    override = os.environ.get(CORPUS_DIR_ENV)
    return Path(override) if override else _DEFAULT_CORPUS_DIR


def _date_string(value: Tuple[int, int, int]) -> str:
    year, month, day = value
    return f"{year:04d}-{month:02d}-{day:02d}"


def _iter_records(corpus_dir: Path):
    parts = sorted(corpus_dir.glob("*.jsonl"))
    if not parts:
        raise CorpusUnavailable(
            f"No corpus found in {corpus_dir}. Build one with "
            f"scripts/harvest_arxiv_corpus.py, or point {CORPUS_DIR_ENV} at an existing one."
        )
    for part in parts:
        with open(part, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except ValueError:
                    # A truncated last line is what a killed harvest leaves.
                    # Skipping it loses one paper; aborting would lose the day.
                    continue


def get_papers_from_corpus(
    area: str,
    begin_date: Tuple[int, int, int],
    end_date: Tuple[int, int, int],
    force_primary: bool = False,
    debug_messages: bool = False,
    corpus_dir: Path = None,
) -> Tuple[List, List[Paper]]:
    """Papers in `area` submitted between `begin_date` and `end_date`, inclusive."""
    corpus_dir = Path(corpus_dir) if corpus_dir else default_corpus_dir()
    begin_string = _date_string(begin_date)
    end_string = _date_string(end_date)

    print(f"Reading papers for {begin_string}..{end_string} from corpus {corpus_dir}")

    entries: List[Dict] = []
    papers: List[Paper] = []
    seen: Set[str] = set()

    for record in _iter_records(corpus_dir):
        created = record.get("created", "")
        if not (begin_string <= created <= end_string):
            continue
        categories = record.get("categories") or []
        if area not in categories:
            continue
        if force_primary and (not categories or categories[0] != area):
            if debug_messages:
                print(f"Ignoring \"{record.get('title', '')}\" by `paper_area` ({categories[:1]})")
            continue

        arxiv_id = record.get("arxiv_id", "")
        # A revised paper can appear in more than one monthly part when a
        # harvest is re-run over an overlapping range; the id is what makes
        # that visible, and the first copy is as good as the second.
        if not arxiv_id or arxiv_id in seen:
            continue
        seen.add(arxiv_id)

        entries.append(record)
        papers.append(
            Paper(
                authors=record.get("authors") or [],
                title=record.get("title", ""),
                abstract=record.get("abstract", ""),
                arxiv_id=arxiv_id,
            )
        )

    if not papers:
        print(f"No entries found for {area}")
        return [], []

    print(f"{len(papers)} papers left for {area}")
    return entries, papers
