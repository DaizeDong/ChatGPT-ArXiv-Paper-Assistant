import dataclasses
import json
import re
from typing import List

import retry
from openai import OpenAI
from tqdm import tqdm

from arxiv_scraper import EnhancedJSONEncoder, Paper
from environment import BASE_PROMPT, CONFIG, OPENAI_API_KEY, OPENAI_BASE_URL, OUTPUT_DEBUG_FILE_FORMAT, POSTFIX_PROMPT, SCORE_PROMPT, TOPIC_PROMPT


def select_by_author(all_authors, papers, selected_papers, sort_dict, author_targets, config):
    # author based selection
    for paper in papers:
        for author in paper.authors:
            if author in all_authors:
                for alias in all_authors[author]:
                    if alias["authorId"] in author_targets:
                        selected_papers[paper.arxiv_id] = {
                            **dataclasses.asdict(paper),
                            **{"COMMENT": "Author match"},
                        }
                        sort_dict[paper.arxiv_id] = float(
                            config["SELECTION"]["author_match_score"]
                        )
                        break
    print(f"Selected {len(selected_papers)} papers based on author match")
    return selected_papers


def filter_papers_by_hindex(all_authors, papers, config):
    # filters papers by checking to see if there's at least one author with > h_cutoff hindex
    paper_list = []
    for paper in papers:
        max_h = 0
        for author in paper.authors:
            if author in all_authors:
                max_h = max(
                    max_h, max([alias["hIndex"] for alias in all_authors[author]])
                )
        if max_h >= float(config["FILTERING"]["h_cutoff"]):
            paper_list.append(paper)
    print(str(len(paper_list)) + " papers after hindex filtering")
    return paper_list


def calc_price(model, usage):
    if model in ("gpt-3.5-turbo",):
        return (7.5 * usage.prompt_tokens + 22.5 * usage.completion_tokens) / 1_000_000
    elif model in ("gpt-3.5-turbo-0125",):
        return (2.5 * usage.prompt_tokens + 7.5 * usage.completion_tokens) / 1_000_000
    elif model in ("gpt-4",):
        return (150 * usage.prompt_tokens + 300 * usage.completion_tokens) / 1_000_000
    elif model in ("gpt-4-32k", "gpt-4-dalle", "gpt-4-v"):
        return (300 * usage.prompt_tokens + 600 * usage.completion_tokens) / 1_000_000
    elif model in ("gpt-4-all",):
        return (300 * usage.prompt_tokens + 300 * usage.completion_tokens) / 1_000_000
    elif model in ("gpt-4-turbo",):
        return (300 * usage.prompt_tokens + 900 * usage.completion_tokens) / 1_000_000
    elif model in ("gpt-4-turbo-preview",):
        return (50 * usage.prompt_tokens + 150 * usage.completion_tokens) / 1_000_000
    elif model in ("gpt-4o", "gpt-4o-2024-08-06", "gpt-4o-2024-11-20"):
        return (25 * usage.prompt_tokens + 100 * usage.completion_tokens) / 1_000_000
    elif model in ("gpt-4o-all",):
        return (300 * usage.prompt_tokens + 1200 * usage.completion_tokens) / 1_000_000
    elif model in ("gpt-4o-mini",):
        return (7.5 * usage.prompt_tokens + 30 * usage.completion_tokens) / 1_000_000
    elif model in ("gpt-ask-internet",):
        return (50 * usage.prompt_tokens + 50 * usage.completion_tokens) / 1_000_000
    else:
        return 0


@retry.retry(tries=3, delay=2)
def call_chatgpt(full_prompt, openai_client, model):
    return openai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.0,
        seed=0,
    )


def run_and_parse_chatgpt(full_prompt, openai_client, config):
    # just runs the chatgpt prompt, tries to parse the resulting JSON
    completion = call_chatgpt(full_prompt, openai_client, config["SELECTION"]["model"])
    out_text = completion.choices[0].message.content
    out_text = re.sub("```jsonl\n", "", out_text)
    out_text = re.sub("```", "", out_text)
    out_text = re.sub(r"\n+", "\n", out_text)
    out_text = re.sub("},", "}", out_text).strip()
    # split out_text line by line and parse each as a json.
    json_dicts = []
    for line in out_text.split("\n"):
        # try catch block to attempt to parse json
        try:
            json_dicts.append(json.loads(line))
        except Exception as ex:
            if config["OUTPUT"].getboolean("debug_messages"):
                print("Exception happened " + str(ex))
                print("Failed to parse LM output as json")
                print(out_text)
                print("RAW output")
                print(completion.choices[0].message.content)
            continue
    return json_dicts, calc_price(config["SELECTION"]["model"], completion.usage)


def paper_to_string(paper_entry: Paper) -> str:
    # renders each paper into a string to be processed by GPT
    new_str = (
        "ArXiv ID: "
        + paper_entry.arxiv_id
        + "\n"
        + "Title: "
        + paper_entry.title
        + "\n"
        + "Authors: "
        + " and ".join(paper_entry.authors)
        + "\n"
        + "Abstract: "
        + paper_entry.abstract[:4000]
    )
    return new_str


def batched(items, batch_size):
    # takes a list and returns a list of list with batch_size
    return [items[i: i + batch_size] for i in range(0, len(items), batch_size)]


def filter_papers_by_title(
    papers, openai_client, base_prompt, topic_prompt, config
) -> List[Paper]:
    filter_postfix = 'Identify any papers that are absolutely and completely irrelavent to the criteria, and you are absolutely sure your friend will not enjoy, formatted as a list of arxiv ids like ["ID1", "ID2", "ID3"..]. Be extremely cautious, and if you are unsure at all, do not add a paper in this list. You will check it in detail later.\n Directly respond with the list, do not add ANY extra text before or after the list. Even if every paper seems irrelevant, please keep at least TWO papers'
    batches_of_papers = batched(papers, 20)
    final_list = []
    cost = 0
    for batch in batches_of_papers:
        papers_string = "".join([paper_to_titles(paper) for paper in batch])
        full_prompt = (
            base_prompt + "\n " + topic_prompt + "\n" + papers_string + filter_postfix
        )
        model = config["SELECTION"]["model"]
        completion = call_chatgpt(full_prompt, openai_client, model)
        cost += calc_price(model, completion.usage)
        out_text = completion.choices[0].message.content
        try:
            filtered_set = set(json.loads(out_text))
            for paper in batch:
                if paper.arxiv_id not in filtered_set:
                    final_list.append(paper)
                else:
                    print(f"Filtered out paper {paper.arxiv_id} by title")
        except Exception as ex:
            print("Exception happened " + str(ex))
            print("Failed to parse LM output as list " + out_text)
            print(completion)
            continue
    return final_list, cost


def paper_to_titles(paper_entry: Paper) -> str:
    return "ArXiv ID: " + paper_entry.arxiv_id + " Title: " + paper_entry.title + "\n"


def get_full_prompt(base_prompt, topic_prompt, score_prompt, postfix_prompt, batch_str):
    full_prompt = "\n\n".join(
        [
            base_prompt,
            topic_prompt,
            "## Papers",
            "\n\n".join(batch_str),
            score_prompt,
            postfix_prompt,
        ]
    )
    return full_prompt


def run_on_batch(
    paper_batch, openai_client, base_prompt, topic_prompt, score_prompt, postfix_prompt, config
):
    batch_str = [paper_to_string(paper) for paper in paper_batch]
    full_prompt = get_full_prompt(base_prompt, topic_prompt, score_prompt, postfix_prompt, batch_str)
    json_dicts, cost = run_and_parse_chatgpt(full_prompt, openai_client, config)
    return json_dicts, cost


def filter_by_gpt(papers, selected_papers, sort_dict, base_prompt, topic_prompt, score_prompt, postfix_prompt, config):
    all_papers = {paper.arxiv_id: paper for paper in papers}
    all_cost = 0

    openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    papers, title_cost = filter_papers_by_title(papers, openai_client, base_prompt, topic_prompt, config)
    print(f"{len(papers)} papers after title filtering with cost of ${title_cost}")
    all_cost += title_cost

    # batch the remaining papers and invoke GPT
    preserved_paper_cnt = 0
    abstract_cost = 0
    scored_batches = []
    for batch in tqdm(batched(papers, int(config["SELECTION"]["batch_size"]))):
        scored_in_batch = []
        json_dicts, cost = run_on_batch(
            batch, openai_client, base_prompt, topic_prompt, score_prompt, postfix_prompt, config
        )
        for jdict in json_dicts:
            if (
                int(jdict["RELEVANCE"])
                >= int(config["FILTERING"]["relevance_cutoff"])
                and jdict["NOVELTY"] >= int(config["FILTERING"]["novelty_cutoff"])
                and jdict["ARXIVID"] in all_papers
            ):
                selected_papers[jdict["ARXIVID"]] = {
                    **dataclasses.asdict(all_papers[jdict["ARXIVID"]]),
                    **jdict,
                }
                sort_dict[jdict["ARXIVID"]] = jdict["RELEVANCE"] + jdict["NOVELTY"]
                preserved_paper_cnt += 1
            scored_in_batch.append(
                {
                    **dataclasses.asdict(all_papers[jdict["ARXIVID"]]),
                    **jdict,
                }
            )
        abstract_cost += cost
        scored_batches.append(scored_in_batch)
    print(f"{preserved_paper_cnt} papers after abstract filtering with cost of ${abstract_cost}")
    all_cost += abstract_cost

    if config["OUTPUT"].getboolean("dump_debug_file"):
        with open(OUTPUT_DEBUG_FILE_FORMAT.format("gpt_paper_batches.json"), "w") as outfile:
            json.dump(scored_batches, outfile, cls=EnhancedJSONEncoder, indent=4)
    print("Total cost: $" + str(all_cost))

    return all_cost


if __name__ == "__main__":
    openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    # loads papers from 'in/debug_papers.json' and filters them
    with open("in/debug_papers.json", "r") as f:
        paper_list_in_dict = json.load(f)

    papers = [
        [
            Paper(
                arxiv_id=paper["arxiv_id"],
                authors=paper["authors"],
                title=paper["title"],
                abstract=paper["abstract"],
            )
            for paper in batch
        ]
        for batch in paper_list_in_dict
    ]
    all_papers = {}
    paper_outputs = {}
    sort_dict = {}
    total_cost = 0
    for batch in tqdm(papers):
        json_dicts, cost = run_on_batch(
            batch, openai_client, BASE_PROMPT, TOPIC_PROMPT, SCORE_PROMPT, POSTFIX_PROMPT, CONFIG
        )
        total_cost += cost
        for paper in batch:
            all_papers[paper.arxiv_id] = paper
        for jdict in json_dicts:
            paper_outputs[jdict["ARXIVID"]] = {
                **dataclasses.asdict(all_papers[jdict["ARXIVID"]]),
                **jdict,
            }
            sort_dict[jdict["ARXIVID"]] = jdict["RELEVANCE"] + jdict["NOVELTY"]

    # sort the papers by relevance and novelty
    print("total cost:" + str(total_cost))
    keys = list(sort_dict.keys())
    values = list(sort_dict.values())


    def argsort(seq):
        return sorted(range(len(seq)), key=seq.__getitem__)


    sorted_keys = [keys[idx] for idx in argsort(values)[::-1]]
    selected_papers = {key: paper_outputs[key] for key in sorted_keys}

    with open(OUTPUT_DEBUG_FILE_FORMAT.format("filter_paper_test.json"), "w") as outfile:
        json.dump(selected_papers, outfile, cls=EnhancedJSONEncoder, indent=4)
