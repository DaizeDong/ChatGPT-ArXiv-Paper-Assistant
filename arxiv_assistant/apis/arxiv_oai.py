import time
from datetime import date, timedelta
from typing import Dict, List, Set, Tuple
from urllib.parse import urlencode
from xml.etree import ElementTree

import requests
import retry

from arxiv_assistant.apis.arxiv import ARXIV_READ_TIMEOUT_S, pace_arxiv_request
from arxiv_assistant.utils.utils import Paper, normalize_whitespace

OAI_BASE_URL = "http://export.arxiv.org/oai2"

NS_OAI = "{http://www.openarchives.org/OAI/2.0/}"
NS_ARXIV = "{http://arxiv.org/OAI/arXiv/}"

# OAI-PMH answers 503 with Retry-After as flow control, not as an outage.
OAI_MAX_WAITS = 6

# OAI-PMH selects on `datestamp` (last changed) while a digest wants `created`
# (submitted). Measured over the cs set for 2026-07-20..27: the lag was 1-4 days
# for 1254 records and 5-7 for 32. Harvest wider, then filter on `created`.
OAI_DATESTAMP_LAG_DAYS = 14


def tuple_to_date(value: Tuple[int, int, int]) -> date:
    return date(*value)


def date_to_tuple(value: date) -> Tuple[int, int, int]:
    return value.year, value.month, value.day


def date_string(value: Tuple[int, int, int]) -> str:
    year, month, day = value
    return f"{year:04d}-{month:02d}-{day:02d}"


def oai_get(params: Dict[str, str]):
    url = f"{OAI_BASE_URL}?{urlencode(params)}"
    for _ in range(OAI_MAX_WAITS):
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


def _author_name(author) -> str:
    keyname = author.find(f"{NS_ARXIV}keyname")
    forenames = author.find(f"{NS_ARXIV}forenames")
    parts = [normalize_whitespace(n.text) for n in (forenames, keyname) if n is not None and n.text]
    return " ".join(parts)


def record_to_paper(record) -> Tuple[Paper, str, List[str]]:
    # `created` comes from the metadata, not the header datestamp: a 2026-07-22
    # window otherwise hands back 2012 papers that merely got an edit that day.
    metadata = record.find(f"{NS_OAI}metadata")
    meta = metadata.find(f"{NS_ARXIV}arXiv") if metadata is not None else None
    if meta is None:
        return None, "", []

    def text(tag: str) -> str:
        node = meta.find(f"{NS_ARXIV}{tag}")
        return normalize_whitespace(node.text) if node is not None and node.text else ""

    arxiv_id, title = text("id"), text("title")
    created, categories = text("created"), text("categories").split()
    if not arxiv_id or not title:
        return None, created, categories

    authors_node = meta.find(f"{NS_ARXIV}authors")
    authors = [_author_name(a) for a in authors_node.findall(f"{NS_ARXIV}author")] if authors_node is not None else []
    paper = Paper(authors=[a for a in authors if a], title=title, abstract=text("abstract"), arxiv_id=arxiv_id)
    return paper, created, categories


@retry.retry(tries=3, delay=30.0)
def get_papers_from_arxiv_oai(
    area: str,
    begin_date: Tuple[int, int, int],
    end_date: Tuple[int, int, int],
    force_primary: bool = False,
    debug_messages: bool = False,
    dump_debug_file: bool = False,
) -> Tuple[List, List[Paper]]:
    """Papers in `area` created between `begin_date` and `end_date`, inclusive."""
    oai_set = area.split(".")[0]  # sets stop at the archive level; narrow below
    begin_string, end_string = date_string(begin_date), date_string(end_date)
    # Clamped to today: arXiv rejects a future `until` with badArgument, which
    # would break this source on exactly the recent dates a daily run asks for.
    harvest_until = date_string(date_to_tuple(min(
        tuple_to_date(end_date) + timedelta(days=OAI_DATESTAMP_LAG_DAYS), date.today())))

    params = {"verb": "ListRecords", "from": begin_string, "until": harvest_until,
              "metadataPrefix": "arXiv", "set": oai_set}
    print(f"Getting papers from OAI-PMH set {oai_set} for {begin_string}..{end_string}")

    entries: List = []
    paper_list: List[Paper] = []
    seen_ids: Set[str] = set()
    page = 0

    while True:
        response = oai_get(params)
        if dump_debug_file:
            from arxiv_assistant.environment import OUTPUT_DEBUG_FILE_FORMAT
            with open(OUTPUT_DEBUG_FILE_FORMAT.format(f"raw_oai_{area}_{page}.xml"), "w", encoding="utf-8") as f:
                f.write(response.text)

        root = ElementTree.fromstring(response.text)
        error = root.find(f"{NS_OAI}error")
        if error is not None:
            code = error.get("code", "")
            if code == "noRecordsMatch":  # weekends really do have no records
                print(f"No entries found for {area}")
                return [], []
            raise RuntimeError(f"OAI-PMH error {code}: {normalize_whitespace(error.text or '')}")

        list_records = root.find(f"{NS_OAI}ListRecords")
        if list_records is None:
            break

        for record in list_records.findall(f"{NS_OAI}record"):
            paper, created, categories = record_to_paper(record)
            if paper is None or paper.arxiv_id in seen_ids:
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
            seen_ids.add(paper.arxiv_id)
            entries.append(record)
            paper_list.append(paper)

        token_node = list_records.find(f"{NS_OAI}resumptionToken")
        token = normalize_whitespace(token_node.text) if token_node is not None and token_node.text else ""
        if not token:
            break
        # A resumption request carries the token and the verb and nothing else.
        params = {"verb": "ListRecords", "resumptionToken": token}
        page += 1

    if not paper_list:
        print(f"No entries found for {area}")
        return [], []

    print(f"{len(paper_list)} papers left for {area}")
    return entries, paper_list
