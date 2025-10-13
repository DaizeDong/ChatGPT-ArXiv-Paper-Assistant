import re
import warnings
from html import unescape
from typing import Dict, List, Set, Tuple
from xml.etree import ElementTree

import feedparser
import requests
import retry

from arxiv_assistant.environment import OUTPUT_DEBUG_FILE_FORMAT
from arxiv_assistant.utils.utils import Paper, normalize_whitespace


@retry.retry(tries=3, delay=30.0)
def get_papers_from_arxiv_api(
    area: str,
    begin_date: Tuple[int, int, int],  # year, month, day
    end_date: Tuple[int, int, int],  # year, month, day
    force_primary: bool = False,
    debug_messages: bool = False,
    dump_debug_file: bool = False,
) -> Tuple[List, List[Paper]]:
    """
    Get papers by calling the arXiv API.
    - Pros:
        Support filtering papers by their uploaded dates.
    - Cons:
        The uploaded dates don't always match the announced dates. Therefore, the filtering is not accurate and may miss some papers.
        Not support filtering by `announce_type`.
    """
    begin_year, begin_month, begin_day = begin_date
    end_year, end_month, end_day = end_date

    base_url = "http://export.arxiv.org/api/query"
    begin_date_string = f"{begin_year}{format(begin_month, '02d')}{format(begin_day, '02d')}"
    end_date_string = f"{end_year}{format(end_month, '02d')}{format(end_day, '02d')}"
    date_query = f"submittedDate:[{begin_date_string}0000+TO+{end_date_string}2359]"
    area_query = f"cat:{area}"

    url = f"{base_url}?search_query={area_query}+AND+{date_query}&start=0&max_results=10000"
    print(f"Getting papers from {url}")
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    if dump_debug_file:
        with open(OUTPUT_DEBUG_FILE_FORMAT.format(f"raw_content_{area}.xml"), "w", encoding="utf-8") as outfile:
            outfile.write(response.text)

    # Parse the XML response
    root = ElementTree.fromstring(response.text)

    entries = root.findall("{http://www.w3.org/2005/Atom}entry")
    if len(entries) == 0:
        print(f"No entries found for {area}")
        return [], []
    print(f"{len(entries)} entries found for {area}")

    # extract papers as a list
    paper_list = []

    for entry in entries:
        title = normalize_whitespace(entry.find("{http://www.w3.org/2005/Atom}title").text)

        # ignore papers not in primary area
        paper_area = entry.find("{http://arxiv.org/schemas/atom}primary_category").get("term")
        if (area != paper_area) and force_primary:
            if debug_messages:
                print(f"Ignoring \"{title}\" by `paper_area` ({paper_area})")
            continue

        # add a new paper
        arxiv_id = normalize_whitespace(entry.find("{http://www.w3.org/2005/Atom}id").text).split("/")[-1].split("v")[0]
        abstract = normalize_whitespace(entry.find("{http://www.w3.org/2005/Atom}summary").text)
        authors = [normalize_whitespace(author.find("{http://www.w3.org/2005/Atom}name").text) for author in entry.findall("{http://www.w3.org/2005/Atom}author")]

        new_paper = Paper(authors=authors, title=title, abstract=abstract, arxiv_id=arxiv_id)
        paper_list.append(new_paper)

    print(f"{len(paper_list)} papers left for {area}")

    return entries, paper_list


@retry.retry(tries=3, delay=30.0)
def get_papers_from_arxiv_rss(
    area: str,
    announce_type: Set[str] = None,
    force_primary: bool = False,
    debug_messages: bool = False,
    dump_debug_file: bool = False,
) -> Tuple[List[Dict], List[Paper]]:
    """
    Get papers from the arXiv RSS feed.
    - Cons:
        Cannot go back to a previous date to get corresponding announced papers.
    """
    if announce_type is None:
        announce_type = {"new"}

    # get the list of entries
    url = f"https://export.arxiv.org/rss/{area}"
    print(f"Getting papers from {url}")
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    feed = feedparser.parse(response.text)
    if dump_debug_file:
        with open(OUTPUT_DEBUG_FILE_FORMAT.format(f"raw_content_{area}.rss"), "w", encoding="utf-8") as outfile:
            outfile.write(response.text)

    # get the list of entries
    entries = feed.entries
    if len(entries) == 0:
        print(f"No entries found for {area}")
        return [], []
    print(f"{len(entries)} entries found for {area}")

    # extract papers as a list
    paper_list = []

    for paper in entries:
        # filter by `announce_type`
        if not paper["arxiv_announce_type"] in announce_type:
            if debug_messages:
                print(f"Ignoring \"{paper.title}\" by `announce_type` ({paper['arxiv_announce_type']})")
            continue

        # ignore papers not in primary area
        paper_area = paper.tags[0]["term"]
        if (area != paper_area) and force_primary:
            if debug_messages:
                print(f"Ignoring \"{paper.title}\" by `paper_area` ({paper_area})")
            continue

        # otherwise make a new paper, for the author field make sure to strip the HTML tags
        authors = [
            unescape(re.sub("<[^<]+?>", "", author)).strip()
            for author in paper.author.replace("\n", ", ").split(",")
        ]
        # strip html tags from summary
        summary = re.sub("<[^<]+?>", "", paper.summary)
        summary = unescape(re.sub("\n", " ", summary))
        # strip the last pair of parentehses containing (arXiv:xxxx.xxxxx [area.XX])
        title = re.sub(r"\(arXiv:[0-9]+\.[0-9]+v[0-9]+ \[.*\]\)$", "", paper.title)
        # strip the abstract
        abstract = summary.split("Abstract: ")[-1]
        # remove the link part of the id
        arxiv_id = paper.link.split("/")[-1].split("v")[0]

        # make a new paper
        new_paper = Paper(authors=authors, title=title, abstract=abstract, arxiv_id=arxiv_id)
        paper_list.append(new_paper)

    print(f"{len(paper_list)} papers left for {area}")

    return entries, paper_list


def get_papers_from_arxiv(
    config,
    source="rss",
    begin_date: Tuple[int, int, int] = None,
    end_date: Tuple[int, int, int] = None,
) -> Tuple[List[Dict], Dict[str, List[Paper]]]:
    all_entries = []
    arxiv_paper_dict = {}

    area_list = [s.strip() for s in config["FILTERING"]["arxiv_category"].split(",")]
    announce_type_list = [s.strip() for s in config["FILTERING"].get("announce_type", "new").split(",")]
    force_primary = config["FILTERING"].getboolean("force_primary")
    debug_messages = config["OUTPUT"].getboolean("debug_messages")
    dump_debug_file = config["OUTPUT"].getboolean("dump_debug_file")

    if source == "rss":
        print(f"Using RSS feed to get papers...")
        if begin_date is not None or end_date is not None:
            warnings.warn(f"Specifying `begin_date` and `end_date` is not supported for \"rss\" source, ignoring them")
        for area in area_list:
            entries, papers = get_papers_from_arxiv_rss(
                area,
                set(announce_type_list),
                force_primary,
                debug_messages,
                dump_debug_file,
            )
            all_entries.extend(entries)
            arxiv_paper_dict[area] = papers

    elif source == "api":
        print(f"Using arXiv API to get papers...")
        warnings.warn(
            "The arXiv API is not reliable for getting daily announced papers. "
            "Consider using the arXiv RSS feed instead if you are filtering papers by their *announced* date. "
            "You can neglect this warning if you are filtering papers *uploaded* on a specified date range."
        )
        if begin_date is None or end_date is None:
            raise ValueError(f"Both `begin_date` and `end_date` arguments are required for \"api\" source")
        if not (len(announce_type_list) == 1 and "new" in announce_type_list):
            warnings.warn(f"Specifying `announce_type` is not supported for \"api\" source, ignoring {announce_type_list}")
        for area in area_list:
            entries, papers = get_papers_from_arxiv_api(
                area,
                begin_date,
                end_date,
                force_primary,
                debug_messages,
            )
            all_entries.extend(entries)
            arxiv_paper_dict[area] = papers

    else:
        raise ValueError(f"Unknown source \"{source}\"")

    return all_entries, arxiv_paper_dict


if __name__ == "__main__":
    from arxiv_assistant.environment import NOW_TIME
    from datetime import timedelta

    area = "cs.LG"
    announce_type = {"new", "cross"}
    force_primary = False
    debug_messages = True
    dump_debug_file = True

    print("Getting papers from arXiv RSS...")
    entries, papers = get_papers_from_arxiv_rss(area, announce_type, force_primary, debug_messages, dump_debug_file)

    yesterday = NOW_TIME - timedelta(days=1)  # use yesterday's time as the API returns papers by their uploaded date instead of the announced date
    yesterday_date = (int(yesterday.strftime("%Y")), int(yesterday.strftime("%m")), int(yesterday.strftime("%d")))

    print("Getting papers from arXiv API...")
    entries, papers = get_papers_from_arxiv_api(area, yesterday_date, yesterday_date, force_primary, debug_messages, dump_debug_file)  # this is inaccurate for getting today's announced paper

    print("Done!")
