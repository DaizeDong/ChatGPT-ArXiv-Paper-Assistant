import json
import os
import time
from typing import Generator, TypeVar

from requests import Session
from retry import retry
from tqdm import tqdm

from arxiv_scraper import EnhancedJSONEncoder, get_papers_from_arxiv_rss_api
from environment import AUTHOR_ID_SET, BASE_PROMPT, CONFIG, OUTPUT_DEBUG_FILE_FORMAT, OUTPUT_JSON_FILE_FORMAT, OUTPUT_MD_FILE_FORMAT, POSTFIX_PROMPT, S2_API_KEY, SCORE_PROMPT, SLACK_KEY, TOPIC_PROMPT
from filter_papers import batched, filter_by_gpt, filter_papers_by_hindex, select_by_author
from parse_json_to_md import render_md_string
from push_to_slack import push_to_slack
from utils import argsort, copy_file_or_dir

T = TypeVar("T")


def get_paper_batch(
    session: Session,
    ids: list[str],
    S2_API_KEY: str,
    fields: str = "paperId,title",
    **kwargs,
) -> list[dict]:
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


def get_author_batch(
    session: Session,
    ids: list[str],
    S2_API_KEY: str,
    fields: str = "name,hIndex,citationCount",
    **kwargs,
) -> list[dict]:
    # gets a batch of authors. analogous to author batch
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

    with session.post(
        "https://api.semanticscholar.org/graph/v1/author/batch",
        params=params,
        headers=headers,
        json=body,
    ) as response:
        response.raise_for_status()
        return response.json()


@retry(tries=3, delay=3.0)
def get_one_author(session, author: str, S2_API_KEY: str) -> str:
    # query the right endpoint https://api.semanticscholar.org/graph/v1/author/search?query=adam+smith
    params = {"query": author, "fields": "authorId,name,hIndex", "limit": "10"}
    if S2_API_KEY is None:
        headers = {}
    else:
        headers = {"X-API-KEY": S2_API_KEY}
    with session.get(
        "https://api.semanticscholar.org/graph/v1/author/search",
        params=params,
        headers=headers,
    ) as response:
        response.raise_for_status()
        response_json = response.json()
        if len(response_json["data"]) >= 1:
            return response_json["data"]
        else:
            return None


def get_papers(
    ids: list[str], S2_API_KEY: str, batch_size: int = 100, **kwargs
) -> Generator[dict, None, None]:
    # gets all papers, doing batching to avoid hitting the max paper limit.
    # use a session to reuse the same TCP connection
    with Session() as session:
        # take advantage of S2 batch paper endpoint
        for ids_batch in batched(ids, batch_size=batch_size):
            yield from get_paper_batch(session, ids_batch, S2_API_KEY, **kwargs)


def get_authors(
    all_authors: list[str], S2_API_KEY: str, batch_size: int = 100, **kwargs
):
    # first get the list of all author ids by querying by author names
    author_metadata_dict = {}
    with Session() as session:
        for author in tqdm(all_authors):
            try:
                auth_map = get_one_author(session, author, S2_API_KEY)
            except Exception as ex:
                print("exception happened" + str(ex))
                auth_map = None
            if auth_map is not None:
                author_metadata_dict[author] = auth_map
            # add a 20ms wait time to avoid rate limiting
            # otherwise, semantic scholar aggressively rate limits, so do 1.0s
            if S2_API_KEY is not None:
                time.sleep(0.02)
            else:
                time.sleep(1.0)
    return author_metadata_dict


def get_papers_from_arxiv(config):
    area_list = config["FILTERING"]["arxiv_category"].split(",")
    paper_set = set()
    for area in area_list:
        papers = get_papers_from_arxiv_rss_api(area.strip(), config)
        paper_set.update(set(papers))
    print("Number of papers:" + str(len(paper_set)))
    return paper_set


if __name__ == "__main__":
    # get the paper list from arxiv
    papers = list(get_papers_from_arxiv(CONFIG))
    if len(papers) == 0:
        print("No papers found")
        exit(0)

    # get the author list from papers
    if CONFIG["SELECTION"].getboolean("run_author_match"):
        all_authors = set()
        for paper in papers:
            all_authors.update(set(paper.authors))
        print("Getting author info for " + str(len(all_authors)) + " authors")
        all_authors = get_authors(list(all_authors), S2_API_KEY)
    else:
        print("Skipping author info")
        all_authors = {}

    # dump all papers for debugging
    if CONFIG["OUTPUT"].getboolean("dump_debug_file"):
        with open(OUTPUT_DEBUG_FILE_FORMAT.format("config.json"), "w") as outfile:
            json.dump({section: dict(CONFIG[section]) for section in CONFIG.sections()}, outfile, cls=EnhancedJSONEncoder, indent=4)
        with open(OUTPUT_DEBUG_FILE_FORMAT.format("AUTHOR_ID_SET.json"), "w") as outfile:
            json.dump(list(AUTHOR_ID_SET), outfile, cls=EnhancedJSONEncoder, indent=4)
        with open(OUTPUT_DEBUG_FILE_FORMAT.format("papers.json"), "w") as outfile:
            json.dump(papers, outfile, cls=EnhancedJSONEncoder, indent=4)
        with open(OUTPUT_DEBUG_FILE_FORMAT.format("all_authors.json"), "w") as outfile:
            json.dump(all_authors, outfile, cls=EnhancedJSONEncoder, indent=4)

    # initialize vars for filtering
    selected_papers = {}
    sort_dict = {}  # dict storing key and score

    # select papers by author
    if CONFIG["SELECTION"].getboolean("run_author_match"):
        selected_papers = select_by_author(
            all_authors,
            papers,
            selected_papers,
            sort_dict,
            AUTHOR_ID_SET,
            CONFIG
        )
    else:
        print("Skipping selection by author")

    # filter papers by h-index
    if CONFIG["SELECTION"].getboolean("run_author_match"):
        papers = filter_papers_by_hindex(all_authors, papers, CONFIG)
    else:
        print("Skipping h-index filtering")

    # filter papers by GPT
    if CONFIG["SELECTION"].getboolean("run_openai"):
        total_prompt_cost, total_completion_cost, total_prompt_tokens, total_completion_tokens = filter_by_gpt(
            papers,
            selected_papers,
            sort_dict,
            BASE_PROMPT,
            TOPIC_PROMPT,
            SCORE_PROMPT,
            POSTFIX_PROMPT,
            CONFIG,
        )
    else:
        print("Skipping GPT filtering")
        total_prompt_cost, total_completion_cost, total_prompt_tokens, total_completion_tokens = 0.0, 0.0, 0, 0

    # sort the papers by relevance and novelty
    keys = list(sort_dict.keys())
    values = list(sort_dict.values())
    sorted_keys = [keys[idx] for idx in argsort(values)[::-1]]
    selected_papers = {key: selected_papers[key] for key in sorted_keys}
    if CONFIG["OUTPUT"].getboolean("debug_messages"):
        print(sort_dict)
        print(selected_papers)

    # pick endpoints and push the summaries
    if CONFIG["OUTPUT"].getboolean("dump_json"):
        with open(OUTPUT_JSON_FILE_FORMAT.format("output.json"), "w") as outfile:
            json.dump(selected_papers, outfile, indent=4)
    if CONFIG["OUTPUT"].getboolean("dump_md"):
        head_table = {
            "headers": ["", "Prompt", "Completion", "Total"],
            "data": [
                ["**Token**", total_prompt_tokens, total_completion_tokens, total_prompt_tokens + total_completion_tokens],
                ["**Cost**", f"${round(total_prompt_cost, 2)}", f"${round(total_completion_cost, 2)}", f"${round(total_prompt_cost + total_completion_cost, 2)}"],
            ]
        }
        with open(OUTPUT_MD_FILE_FORMAT.format("output.md"), "w") as f:
            f.write(render_md_string(selected_papers, head_table=head_table))

    # only push to slack for non-empty dicts
    if CONFIG["OUTPUT"].getboolean("push_to_slack"):
        if SLACK_KEY is None:
            print("Warning: push_to_slack is true, but SLACK_KEY is not set - not pushing to slack")
        else:
            push_to_slack(selected_papers)

    # copy files
    copy_file_or_dir(OUTPUT_MD_FILE_FORMAT.format("output.md"), CONFIG["OUTPUT"]["output_path"])
    os.rename(
        os.path.join(CONFIG["OUTPUT"]["output_path"], os.path.basename(OUTPUT_MD_FILE_FORMAT.format("output.md"))),
        os.path.join(CONFIG["OUTPUT"]["output_path"], "output.md"),
    )
