import dataclasses
import datetime
import json
import math
import re
import time
from typing import Dict, List, Tuple

import retry
from openai import OpenAI
from tqdm import tqdm

from arxiv_scraper import EnhancedJSONEncoder, Paper
from environment import BASE_PROMPT, CONFIG, OPENAI_API_KEY, OPENAI_BASE_URL, OUTPUT_DEBUG_FILE_FORMAT, POSTFIX_PROMPT, SCORE_PROMPT, TOPIC_PROMPT
from pricing import MODEL_PRICING

"""author filtering"""


def select_by_author(all_authors, paper_list, author_targets, config):
    # author-based selection
    new_paper_list = []
    selected_results = {}

    for paper in paper_list:
        selected = any(
            alias["authorId"] in author_targets
            for author in paper.authors if author in all_authors
            for alias in all_authors[author]
        )
        if selected:
            selected_results[paper.arxiv_id] = {
                "COMMENT": "Author match",
                "SCORE": float(config["SELECTION"]["author_match_score"]),
                **dataclasses.asdict(paper),
            }
        else:
            new_paper_list.append(paper)

    print(f"Selected {len(selected_results)} papers based on author match, remaining {len(new_paper_list)} papers")
    return new_paper_list, selected_results


def filter_papers_by_hindex(all_authors, paper_list, config):
    # filters papers by checking to see if there's at least one author with > h_cutoff hindex
    new_paper_list = []
    filtered_results = {}

    for paper in paper_list:
        max_hindex = max(
            [
                alias["hIndex"]
                for author in paper.authors if author in all_authors
                for alias in all_authors[author]
            ] + [0]
        )
        filtered = (max_hindex < float(config["FILTERING"]["h_cutoff"]))
        if filtered:
            filtered_results[paper.arxiv_id] = {
                "COMMENT": f"H-index filtered (max is {max_hindex}<{config['FILTERING']['h_cutoff']})",
                "SCORE": 0,
                **dataclasses.asdict(paper),
            }
        else:
            new_paper_list.append(paper)

    print(f"Filtered {len(filtered_results)} papers based on h-index, remaining {len(new_paper_list)} papers")
    return new_paper_list, filtered_results


"""gpt filtering"""

ABSTRACT_CUTOFF = 4000


def calc_price(model, usage):
    if model not in MODEL_PRICING:
        print(f"Model \"{model}\" not found in pricing table, skip pricing calculation")
        return 0, 0

    cached_tokens = usage.model_extra["prompt_tokens_details"].get("cached_tokens", 0)
    prompt_tokens = usage.prompt_tokens - cached_tokens
    completion_tokens = usage.completion_tokens

    cache_pricing = MODEL_PRICING[model]["cache"] if "cache" in MODEL_PRICING[model] else MODEL_PRICING[model]["prompt"]
    prompt_pricing = MODEL_PRICING[model]["prompt"]
    completion_pricing = MODEL_PRICING[model]["completion"]

    cache_cost = cache_pricing * cached_tokens / 1_000_000
    prompt_cost = prompt_pricing * prompt_tokens / 1_000_000
    completion_cost = completion_pricing * completion_tokens / 1_000_000

    return cache_cost + prompt_cost, completion_cost


def paper_to_titles(paper_entry: Paper) -> str:
    return (
        "ArXiv ID: "
        + paper_entry.arxiv_id
        + "\n"
        + "Title: "
        + paper_entry.title
    )


def paper_to_string(paper_entry: Paper) -> str:
    # renders each paper into a string to be processed by GPT
    return (
        "ArXiv ID: "
        + paper_entry.arxiv_id
        + "\n"
        + "Title: "
        + paper_entry.title
        + "\n"
        + "Authors: "
        + ", ".join(paper_entry.authors)
        + "\n"
        + "Abstract: "
        + paper_entry.abstract[:ABSTRACT_CUTOFF]
    )


def get_full_prompt_for_title_filtering(base_prompt, topic_prompt, batch_str):
    postfix_prompt = (
        '## Instruction\n\n'
        'Identify any papers that are absolutely and completely irrelevant to the criteria, and you are absolutely sure your friend will not enjoy, formatted as a list of arxiv ids like ["ID1", "ID2", "ID3"..].\n'
        'Be extremely cautious, and if you are unsure at all, do not add a paper in this list. You will check it in detail later.\n'
        'Directly respond with the list, do not add ANY extra text before or after the list.'
    )
    full_prompt = "\n\n".join(
        [
            base_prompt,
            topic_prompt,
            "## Papers",
            "\n\n".join(batch_str),
            postfix_prompt,
        ]
    )
    return full_prompt


def get_full_prompt_for_abstract_filtering(base_prompt, topic_prompt, score_prompt, postfix_prompt, batch_str):
    full_prompt = "\n\n".join(
        [
            base_prompt,
            topic_prompt,
            score_prompt,
            "## Papers",
            "\n\n".join(batch_str),
            postfix_prompt,
        ]
    )
    return full_prompt


def get_batch_size(batch_size, paper_num, config):
    use_adaptive = config["SELECTION"].getboolean("adaptive_batch_size")
    adaptive_threshold = int(config["SELECTION"]["adaptive_threshold"])

    if use_adaptive and adaptive_threshold > 0:
        if paper_num <= adaptive_threshold:
            scale_factor = 1
        else:
            scale_factor = math.ceil(math.log(paper_num / adaptive_threshold, 2) + 1)
    else:
        scale_factor = 1

    print(f"Base batch size: {batch_size}, scale factor: {scale_factor}")
    return int(batch_size * scale_factor)


def batched(items, batch_size):
    # takes a list and returns a list of list with batch_size
    return [items[i: i + batch_size] for i in range(0, len(items), batch_size)]


start_query_time = None
query_cnt = 0


@retry.retry(tries=3, delay=2)
def call_chatgpt(full_prompt, openai_client, model):
    def call():
        return openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.0,
            seed=0,
        )

    if int(CONFIG["SELECTION"]["limit_per_minute"]) <= 0:  # no limit
        return call()

    else:  # limit the query num within a minute
        global start_query_time, query_cnt

        while True:
            now_time = datetime.datetime.now()
            if start_query_time is None or now_time - start_query_time > datetime.timedelta(minutes=1):
                start_query_time = now_time
                query_cnt = 0

            if query_cnt < int(CONFIG["SELECTION"]["limit_per_minute"]):
                query_cnt += 1
                return call()
            else:  # wait for a second and recheck
                time.sleep(1)
                continue


def filter_papers_by_title(
    paper_list, openai_client, base_prompt, topic_prompt, config
) -> Tuple[List[Paper], Dict, float, float, int, int]:
    batch_size = get_batch_size(int(config["SELECTION"]["title_batch_size"]), len(paper_list), config)
    print(f"Using batch size of {batch_size} for title filtering")
    batches_of_papers = batched(paper_list, batch_size)

    new_paper_list = []
    filtered_results = {}
    total_prompt_cost = 0.0
    total_completion_cost = 0.0
    prompt_tokens = 0
    completion_tokens = 0

    for batch in tqdm(batches_of_papers, desc="Filtering title"):
        papers_string = "".join([paper_to_titles(paper) for paper in batch])
        full_prompt = get_full_prompt_for_title_filtering(base_prompt, topic_prompt, papers_string)
        model = config["SELECTION"]["model"]
        completion = call_chatgpt(full_prompt, openai_client, model)

        prompt_cost, completion_cost = calc_price(model, completion.usage)
        total_prompt_cost += prompt_cost
        total_completion_cost += completion_cost
        prompt_tokens += completion.usage.prompt_tokens
        completion_tokens += completion.usage.completion_tokens
        out_text = completion.choices[0].message.content
        print({"prompt": {"tokens": completion.usage.prompt_tokens, "cost": prompt_cost}, "completion": {"tokens": completion.usage.completion_tokens, "cost": completion_cost}})

        try:
            filtered_set = set(json.loads(out_text))
            for paper in batch:
                if paper.arxiv_id in filtered_set:
                    filtered_results[paper.arxiv_id] = {
                        "COMMENT": f"Title filtered",
                        "SCORE": 0,
                        **dataclasses.asdict(paper),
                    }
                    print(f"Filtered out paper {paper.arxiv_id} by title ({paper.title})")
                else:
                    new_paper_list.append(paper)
        except Exception as ex:
            print("Exception happened " + str(ex))
            print("Failed to parse LM output as list " + out_text)
            print(completion)
            continue

    print(f"Filtered {len(filtered_results)} papers based on title with cost of ${total_prompt_cost + total_completion_cost}, remaining {len(new_paper_list)} papers:\n"
          f"({prompt_tokens} prompt tokens cost ${total_prompt_cost})\n"
          f"({completion_tokens} completion tokens cost ${total_completion_cost})")

    return new_paper_list, filtered_results, total_prompt_cost, total_completion_cost, prompt_tokens, completion_tokens


def parse_chatgpt(raw_out_text, config):
    # just runs the chatgpt prompt, tries to parse the resulting JSON
    out_text = re.sub("```jsonl\n", "", raw_out_text)
    out_text = re.sub("```", "", out_text)
    out_text = re.sub(r"\n+", "\n", out_text)
    out_text = re.sub("},", "}", out_text).strip()

    # split out_text line by line and parse each as a json.
    json_dicts = []
    invalid_cnt = 0  # the number of papers that cannot be identified according to the model output

    for line in out_text.split("\n"):
        # try catch block to attempt to parse json
        try:
            json_dicts.append(json.loads(line))
        except Exception as ex:
            invalid_cnt += 1
            if config["OUTPUT"].getboolean("debug_messages"):
                print("Exception happened " + str(ex))
                print("Failed to parse LM output as json")
                print(out_text)
                print("RAW output")
                print(raw_out_text)
            continue
    return json_dicts, invalid_cnt


def filter_papers_by_abstract(
    paper_list, id_paper_mapping, openai_client, base_prompt, topic_prompt, score_prompt, postfix_prompt, config
) -> Tuple[List[List[Dict]], Dict, Dict, float, float, int, int]:
    batch_size = get_batch_size(int(config["SELECTION"]["abstract_batch_size"]), len(paper_list), config)
    print(f"Using batch size of {batch_size} for abstract filtering")
    batches_of_papers = batched(paper_list, batch_size)

    invalid_cnt = 0  # the number of papers that cannot be identified according to the model output
    scored_batches = []
    selected_results = {}
    filtered_results = {}
    total_prompt_cost = 0.0
    total_completion_cost = 0.0
    prompt_tokens = 0
    completion_tokens = 0

    for batch in tqdm(batches_of_papers, desc="Filtering abstract"):
        scored_in_batch = []
        batch_str = [paper_to_string(paper) for paper in batch]
        full_prompt = get_full_prompt_for_abstract_filtering(base_prompt, topic_prompt, score_prompt, postfix_prompt, batch_str)
        model = config["SELECTION"]["model"]
        completion = call_chatgpt(full_prompt, openai_client, model)

        prompt_cost, completion_cost = calc_price(model, completion.usage)
        total_prompt_cost += prompt_cost
        total_completion_cost += completion_cost
        prompt_tokens += completion.usage.prompt_tokens
        completion_tokens += completion.usage.completion_tokens
        out_text = completion.choices[0].message.content
        print({"prompt": {"tokens": completion.usage.prompt_tokens, "cost": prompt_cost}, "completion": {"tokens": completion.usage.completion_tokens, "cost": completion_cost}})

        json_dicts, this_invalid_cnt = parse_chatgpt(out_text, config)
        invalid_cnt += this_invalid_cnt

        for jdict in json_dicts:
            if jdict["ARXIVID"] not in id_paper_mapping:
                invalid_cnt += 1
                if config["OUTPUT"].getboolean("debug_messages"):
                    print("Exception happened:")
                    print(f"ARXIVID {jdict['ARXIVID']} not found in `id_paper_mapping`")
                continue

            result = {
                "SCORE": jdict["RELEVANCE"] + jdict["NOVELTY"],
                "RELEVANCE": jdict["RELEVANCE"],
                "NOVELTY": jdict["NOVELTY"],
                **jdict,
                **dataclasses.asdict(id_paper_mapping[jdict["ARXIVID"]]),
            }
            scored_in_batch.append(result)

            filtered = (
                int(jdict["RELEVANCE"]) < int(config["FILTERING"]["relevance_cutoff"]) or
                int(jdict["NOVELTY"]) < int(config["FILTERING"]["novelty_cutoff"])
            )
            if filtered:
                filtered_results[jdict["ARXIVID"]] = result
                print(f"Filtered out paper {jdict['ARXIVID']} by score (RELEVANCE={jdict['RELEVANCE']}, NOVELTY={jdict['NOVELTY']}) ({id_paper_mapping[jdict['ARXIVID']].title})")
            else:
                selected_results[jdict["ARXIVID"]] = result

        scored_batches.append(scored_in_batch)

    print(f"Filtered {len(filtered_results)} papers based on abstract with cost of ${total_prompt_cost + total_completion_cost}, remaining {len(selected_results)} papers:\n"
          f"({prompt_tokens} prompt tokens cost ${total_prompt_cost})\n"
          f"({completion_tokens} completion tokens cost ${total_completion_cost})")

    return scored_batches, selected_results, filtered_results, total_prompt_cost, total_completion_cost, prompt_tokens, completion_tokens


def filter_by_gpt(paper_list, base_prompt, topic_prompt, score_prompt, postfix_prompt, config):
    total_filtered_results = {}
    total_prompt_cost = 0.0
    total_completion_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    id_paper_mapping = {paper.arxiv_id: paper for paper in paper_list}

    # filter papers by titles
    paper_list, filtered_results, prompt_cost, completion_cost, prompt_tokens, completion_tokens = filter_papers_by_title(
        paper_list,
        openai_client,
        base_prompt,
        topic_prompt,
        config
    )
    total_filtered_results.update(filtered_results)
    total_prompt_cost += prompt_cost
    total_completion_cost += completion_cost
    total_prompt_tokens += prompt_tokens
    total_completion_tokens += completion_tokens

    # filter remaining papers by abstracts
    scored_batches, selected_results, filtered_results, prompt_cost, completion_cost, prompt_tokens, completion_tokens = filter_papers_by_abstract(
        paper_list,
        id_paper_mapping,
        openai_client,
        base_prompt,
        topic_prompt,
        score_prompt,
        postfix_prompt,
        config
    )
    total_filtered_results.update(filtered_results)
    total_prompt_cost += prompt_cost
    total_completion_cost += completion_cost
    total_prompt_tokens += prompt_tokens
    total_completion_tokens += completion_tokens

    if config["OUTPUT"].getboolean("dump_debug_file"):
        with open(OUTPUT_DEBUG_FILE_FORMAT.format("gpt_paper_batches.json"), "w") as outfile:
            json.dump(scored_batches, outfile, cls=EnhancedJSONEncoder, indent=4)

    print(f"Total cost is ${total_prompt_cost + total_completion_cost}:\n"
          f"({total_prompt_tokens} prompt tokens cost ${total_prompt_cost})\n"
          f"({total_completion_tokens} completion tokens cost ${total_completion_cost})")

    return selected_results, total_filtered_results, total_prompt_cost, total_completion_cost, total_prompt_tokens, total_completion_tokens


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
        batch_str = [paper_to_string(paper) for paper in batch]
        full_prompt = get_full_prompt_for_abstract_filtering(BASE_PROMPT, TOPIC_PROMPT, SCORE_PROMPT, POSTFIX_PROMPT, batch_str)
        model = CONFIG["SELECTION"]["model"]
        completion = call_chatgpt(full_prompt, openai_client, model)

        prompt_cost, completion_cost = calc_price(model, completion.usage)
        total_cost += prompt_cost + completion_cost
        out_text = completion.choices[0].message.content

        json_dicts = parse_chatgpt(out_text, CONFIG)
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
