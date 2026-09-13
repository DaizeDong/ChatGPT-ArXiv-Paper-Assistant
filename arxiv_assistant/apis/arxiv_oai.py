"""Harvest papers for a date window through arXiv's OAI-PMH interface.

WHY A SECOND SOURCE EXISTS. The Atom API at export.arxiv.org/api/query is rate
limited per host, and a backfill that walks a hundred past dates can earn a
block that outlives the run. MEASURED 2026-09-12: after such a run that endpoint
answered `429 Rate exceeded.` to this host for hours, to a max_results=1 probe
just as much as to a real query, while OAI-PMH on the SAME host answered 200
with 826 records for one day of the cs set. OAI-PMH is the interface arXiv
publishes for bulk and date-ranged harvesting, which is exactly what a backfill
does, so a backfill belongs here rather than on the interactive search API.

The two sources return the same (entries, papers) shape, so callers can swap
between them without knowing which one they got.
"""

import time
from datetime import date, timedelta
from typing import Dict, List, Set, Tuple
from urllib.parse import urlencode
from xml.etree import ElementTree

import requests
import retry

from arxiv_assistant.apis.arxiv import (
    ARXIV_READ_TIMEOUT_S,
    pace_arxiv_request,
)
from arxiv_assistant.utils.utils import Paper, normalize_whitespace

OAI_BASE_URL = "http://export.arxiv.org/oai2"

_NS_OAI = "{http://www.openarchives.org/OAI/2.0/}"
_NS_ARXIV = "{http://arxiv.org/OAI/arXiv/}"

#: OAI-PMH answers 503 with Retry-After as ordinary FLOW CONTROL, not as an
#: outage: it is how the server asks a harvester to slow down. Treating it as an
#: error would abort a perfectly healthy harvest partway through.
_OAI_MAX_WAITS = 6

#: How far PAST the wanted window to harvest, in days.
#:
#: OAI-PMH selects on `datestamp` (when the record last changed), while a daily
#: digest wants `created` (when the paper was submitted). Those differ: a paper
#: created on the 20th is typically datestamped the 21st to the 24th, so
#: harvesting exactly [begin, end] and then filtering on `created` returns
#: almost nothing. MEASURED over the cs set for 2026-07-20..27: of records whose
#: `created` fell in the window, the datestamp lag was 1-4 days for 1254 of
#: them and 5-7 days for 32; the long tail beyond that is old papers picking up
#: an edit, which the `created` filter drops anyway. Fourteen days is twice the
#: observed spread for fresh submissions and costs only extra records to skip.
OAI_DATESTAMP_LAG_DAYS = 14


def tuple_to_date(value: Tuple[int, int, int]) -> date:
    return date(*value)


def date_to_tuple(value: date) -> Tuple[int, int, int]:
    return value.year, value.month, value.day


def _oai_get(params: Dict[str, str]):
    """GET one OAI-PMH page, honouring the protocol's 503/Retry-After handshake."""
    url = f"{OAI_BASE_URL}?{urlencode(params)}"
    for _ in range(_OAI_MAX_WAITS):
        pace_arxiv_request()
        response = requests.get(url, timeout=ARXIV_READ_TIMEOUT_S)
        if response.status_code != 503:
            response.raise_for_status()
            return response

        retry_after = response.headers.get("Retry-After")
        try:
            delay = float(retry_after) if retry_after else 0.0
        except (TypeError, ValueError):
            delay = 0.0
        delay = min(max(delay, 10.0), 300.0)
        print(f"OAI-PMH asked for a {delay:.0f}s pause (flow control)")
        time.sleep(delay)

    response.raise_for_status()  # out of patience: surface it rather than hide it
    return response


def _date_string(date: Tuple[int, int, int]) -> str:
    year, month, day = date
    return f"{year:04d}-{month:02d}-{day:02d}"


def _author_name(author) -> str:
    keyname = author.find(f"{_NS_ARXIV}keyname")
    forenames = author.find(f"{_NS_ARXIV}forenames")
    parts = [
        normalize_whitespace(node.text)
        for node in (forenames, keyname)
        if node is not None and node.text
    ]
    return " ".join(parts)


def _record_to_paper(record) -> Tuple[Paper, str, List[str]]:
    """Return (paper, created_date_string, categories) for one OAI record.

    `created` is read from the metadata rather than the header `datestamp`. They
    are NOT the same thing: the datestamp is when the record last changed, so a
    window around 2026-07-22 hands back papers from 2012 that merely got an
    edit that day (measured: 1205.1277). Filtering on `created` is what keeps a
    backfilled day equal to what the day itself announced.
    """
    metadata = record.find(f"{_NS_OAI}metadata")
    if metadata is None:
        return None, "", []
    meta = metadata.find(f"{_NS_ARXIV}arXiv")
    if meta is None:
        return None, "", []

    def text(tag: str) -> str:
        node = meta.find(f"{_NS_ARXIV}{tag}")
        return normalize_whitespace(node.text) if node is not None and node.text else ""

    arxiv_id = text("id")
    title = text("title")
    abstract = text("abstract")
    created = text("created")
    categories = text("categories").split()
    if not arxiv_id or not title:
        return None, created, categories

    authors_node = meta.find(f"{_NS_ARXIV}authors")
    authors = (
        [_author_name(a) for a in authors_node.findall(f"{_NS_ARXIV}author")]
        if authors_node is not None
        else []
    )
    authors = [a for a in authors if a]

    return Paper(authors=authors, title=title, abstract=abstract, arxiv_id=arxiv_id), created, categories


@retry.retry(tries=3, delay=30.0)
def get_papers_from_arxiv_oai(
    area: str,
    begin_date: Tuple[int, int, int],
    end_date: Tuple[int, int, int],
    force_primary: bool = False,
    debug_messages: bool = False,
    dump_debug_file: bool = False,
) -> Tuple[List, List[Paper]]:
    """Papers in `area` created between `begin_date` and `end_date`, inclusive.

    The harvest is per SET (`cs` for any `cs.*` area), because OAI-PMH sets stop
    at the archive level. The per-area narrowing happens here, on `categories`,
    which is the same predicate the Atom API's `cat:` search applies.
    """
    oai_set = area.split(".")[0]
    begin_string = _date_string(begin_date)
    end_string = _date_string(end_date)
    # Clamped to today: arXiv answers `badArgument: until date too late` for a
    # future datestamp, so the lag window would break the source outright on
    # any recent date -- the exact case a daily run would hit.
    harvest_until = _date_string(
        date_to_tuple(min(
            tuple_to_date(end_date) + timedelta(days=OAI_DATESTAMP_LAG_DAYS),
            date.today(),
        ))
    )

    params = {
        "verb": "ListRecords",
        "from": begin_string,
        "until": harvest_until,
        "metadataPrefix": "arXiv",
        "set": oai_set,
    }
    print(f"Getting papers from OAI-PMH set {oai_set} for {begin_string}..{end_string}")

    entries: List = []
    paper_list: List[Paper] = []
    seen_ids: Set[str] = set()
    page = 0

    while True:
        response = _oai_get(params)
        if dump_debug_file:
            from arxiv_assistant.environment import OUTPUT_DEBUG_FILE_FORMAT

            with open(OUTPUT_DEBUG_FILE_FORMAT.format(f"raw_oai_{area}_{page}.xml"), "w", encoding="utf-8") as outfile:
                outfile.write(response.text)

        root = ElementTree.fromstring(response.text)
        error = root.find(f"{_NS_OAI}error")
        if error is not None:
            code = error.get("code", "")
            # An empty window is a normal answer, not a failure: weekends and
            # holidays really do have no records.
            if code == "noRecordsMatch":
                print(f"No entries found for {area}")
                return [], []
            raise RuntimeError(f"OAI-PMH error {code}: {normalize_whitespace(error.text or '')}")

        list_records = root.find(f"{_NS_OAI}ListRecords")
        if list_records is None:
            break

        for record in list_records.findall(f"{_NS_OAI}record"):
            paper, created, categories = _record_to_paper(record)
            if paper is None:
                continue
            if not (begin_string <= created <= end_string):
                if debug_messages:
                    print(f"Ignoring \"{paper.title}\" by `created` ({created})")
                continue
            if area not in categories:
                continue
            if force_primary and categories[0] != area:
                if debug_messages:
                    print(f"Ignoring \"{paper.title}\" by `paper_area` ({categories[0]})")
                continue
            # A resumption-token harvest can hand back the same record twice if
            # the server re-slices mid-walk; ids are what make that visible.
            if paper.arxiv_id in seen_ids:
                continue
            seen_ids.add(paper.arxiv_id)

            entries.append(record)
            paper_list.append(paper)

        token_node = list_records.find(f"{_NS_OAI}resumptionToken")
        token = normalize_whitespace(token_node.text) if token_node is not None and token_node.text else ""
        if not token:
            break
        # A resumption request carries the token and the verb, and NOTHING else:
        # repeating from/until/set alongside it is a protocol error.
        params = {"verb": "ListRecords", "resumptionToken": token}
        page += 1

    if not paper_list:
        print(f"No entries found for {area}")
        return [], []

    print(f"{len(paper_list)} papers left for {area}")
    return entries, paper_list
