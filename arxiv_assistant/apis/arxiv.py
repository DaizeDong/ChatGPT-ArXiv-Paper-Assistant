from html import unescape

import arxiv
import configparser
import feedparser
import re
import requests
from datetime import datetime, timedelta
from requests import Session
from typing import Any, Dict, Generator, List, Optional, Tuple

from arxiv_assistant.environment import OUTPUT_DEBUG_FILE_FORMAT
from arxiv_assistant.utils.utils import Paper, batched, is_earlier


def get_papers_from_arxiv_api(area: str, timestamp, last_id) -> List[Paper]:
    # look for papers that are newer than the newest papers in RSS.
    # we do this by looking at last_id and grabbing everything newer.
    end_date = timestamp
    start_date = timestamp - timedelta(days=4)
    search = arxiv.Search(
        query="("
              + area
              + ") AND submittedDate:["
              + start_date.strftime("%Y%m%d")
              + "* TO "
              + end_date.strftime("%Y%m%d")
              + "*]",
        max_results=None,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    results = list(arxiv.Client().results(search))
    api_papers = []
    for result in results:
        new_id = result.get_short_id()[:10]
        if is_earlier(last_id, new_id):
            authors = [author.name for author in result.authors]
            summary = result.summary
            summary = unescape(re.sub("\n", " ", summary))
            paper = Paper(
                authors=authors,
                title=result.title,
                abstract=summary,
                arxiv_id=result.get_short_id()[:10],
            )
            api_papers.append(paper)
    return api_papers


def get_papers_from_arxiv_rss(area: str, config: Optional[Dict]) -> Tuple[List, None, None] | Tuple[List[Paper], datetime, Any]:
    # get the feed from http://export.arxiv.org/rss/ and use the updated timestamp to avoid duplicates
    updated = datetime.utcnow() - timedelta(days=1)

    # format this into the string format 'Fri, 03 Nov 2023 00:30:00 GMT'
    url = f"https://export.arxiv.org/rss/{area}"
    updated_string = updated.strftime("%a, %d %b %Y %H:%M:%S GMT")
    print(f"Getting papers from {url}")
    feed = feedparser.parse(url, modified=updated_string)
    if feed.status == 304:
        if config is not None:
            print(f"No {config['FILTERING'].get('announce_type', 'new').replace(',', '/')} papers since {updated_string} for {area}")
        return [], None, None  # if there are no new paper return an empty list

    # get the list of entries
    entries = feed.entries
    if len(entries) == 0:
        print(f"No entries found for {area}")
        return [], None, None  # if there are no new paper return an empty list
    print(f"{len(entries)} entries found for {area}")

    if config["OUTPUT"].getboolean("dump_debug_file"):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(OUTPUT_DEBUG_FILE_FORMAT.format(f"raw_content_{area}.rss"), "w", encoding="utf-8") as outfile:
                    outfile.write(response.text)
            else:
                print(f"Warning: Failed to fetch RSS content, status code {response.status_code}")
        except Exception as e:
            print(f"Error fetching RSS content: {e}")

    # parse last-modified date
    paper_list = []
    timestamp = datetime.strptime(feed.feed["updated"], "%a, %d %b %Y %H:%M:%S +0000")
    last_id = entries[0].link.split("/")[-1]
    announce_type = set(config["FILTERING"].get("announce_type", "new").split(","))

    for paper in entries:
        # ignore updated papers
        if not paper["arxiv_announce_type"] in announce_type:
            if config["OUTPUT"].getboolean("debug_messages"):
                print(f"Ignoring \"{paper.title}\" by `announce_type` ({paper['arxiv_announce_type']})")
            continue
        # extract area
        paper_area = paper.tags[0]["term"]
        # ignore papers not in primary area
        if (area != paper_area) and (config["FILTERING"].getboolean("force_primary")):
            if config["OUTPUT"].getboolean("debug_messages"):
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
        id = paper.link.split("/")[-1]
        # make a new paper
        new_paper = Paper(authors=authors, title=title, abstract=abstract, arxiv_id=id)
        paper_list.append(new_paper)

    print(f"{len(paper_list)} papers left for {area}")

    return entries, paper_list, timestamp, last_id


def merge_paper_list(paper_list, api_paper_list):
    # TODO: seems not used in the codebase. remove if not needed
    api_set = set([paper.arxiv_id for paper in api_paper_list])
    merged_paper_list = api_paper_list
    for paper in paper_list:
        if paper.arxiv_id not in api_set:
            merged_paper_list.append(paper)
    return merged_paper_list


def get_papers_from_arxiv_rss_api(area: str, config: Optional[Dict]) -> Tuple[List, List[Paper]]:
    entries, paper_list, timestamp, last_id = get_papers_from_arxiv_rss(area, config)
    # if timestamp is None:
    #    return []
    # api_paper_list = get_papers_from_arxiv_api(area, timestamp, last_id)
    # merged_paper_list = merge_paper_list(paper_list, api_paper_list)
    # return merged_paper_list
    return entries, paper_list


def get_papers_from_arxiv(config) -> Tuple[List, Dict[str, List[Paper]]]:
    area_list = config["FILTERING"]["arxiv_category"].split(",")
    all_entries = []
    arxiv_paper_dict = {}
    for area in area_list:
        entries, papers = get_papers_from_arxiv_rss_api(area.strip(), config)
        all_entries.extend(entries)
        arxiv_paper_dict[area] = papers
    return all_entries, arxiv_paper_dict


def get_paper_batch(
    session: Session,
    ids: List[str],
    S2_API_KEY: str,
    fields: str = "paperId,title",
    **kwargs,
) -> List[Dict]:
    # TODO: seems not used in the codebase. remove if not needed
    # gets a batch of papers. taken from the sem scholar example.
    params = {
        "fields": fields,
        **kwargs,
    }
    if S2_API_KEY is None:
        headers = {}
    else:
        headers = {
            "X-API-KEY": S2_API_KEY,
        }
    body = {
        "ids": ids,
    }

    # https://api.semanticscholar.org/api-docs/graph#tag/Paper-Data/operation/post_graph_get_papers
    with session.post(
        "https://api.semanticscholar.org/graph/v1/paper/batch",
        params=params,
        headers=headers,
        json=body,
    ) as response:
        response.raise_for_status()
        return response.json()


def get_papers(
    ids: List[str], S2_API_KEY: str, batch_size: int = 100, **kwargs
) -> Generator[Dict, None, None]:
    # TODO: seems not used in the codebase. remove if not needed
    # gets all papers, doing batching to avoid hitting the max paper limit.
    # use a session to reuse the same TCP connection
    with Session() as session:
        # take advantage of S2 batch paper endpoint
        for ids_batch in batched(ids, batch_size=batch_size):
            yield from get_paper_batch(session, ids_batch, S2_API_KEY, **kwargs)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("configs/config.ini")
    paper_list, timestamp, last_id = get_papers_from_arxiv_rss("cs.CL", config)
    print(timestamp)
    api_paper_list = get_papers_from_arxiv_api("cs.CL", timestamp, last_id)
    merged_paper_list = merge_paper_list(paper_list, api_paper_list)
    print([paper.arxiv_id for paper in merged_paper_list])
    print([paper.arxiv_id for paper in paper_list])
    print([paper.arxiv_id for paper in api_paper_list])
    print("success")
