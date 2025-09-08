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

from arxiv_assistant.environment import OPENAI_API_KEY, OPENAI_BASE_URL, OUTPUT_DEBUG_FILE_FORMAT
from arxiv_assistant.utils.pricing import MODEL_PRICING
from arxiv_assistant.utils.utils import EnhancedJSONEncoder, Paper, batched

ABSTRACT_CUTOFF = 4000


def calc_price(model, usage):
    if model not in MODEL_PRICING:
        print(f"Model \"{model}\" not found in pricing table, skip pricing calculation")
        return 0, 0

    cached_tokens = usage.model_extra.get("prompt_tokens_details", {}).get("cached_tokens", 0)
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


def get_user_prompt_for_title_filtering(topic_prompt, postfix_prompt, batch_str):
    user_prompt = "\n\n".join(
        [
            topic_prompt,
            "## Papers",
            "\n\n".join(batch_str),
            postfix_prompt,
        ]
    )
    return user_prompt


def get_user_prompt_for_abstract_filtering(topic_prompt, score_prompt, postfix_prompt, batch_str):
    user_prompt = "\n\n".join(
        [
            topic_prompt,
            score_prompt,
            "## Papers",
            "\n\n".join(batch_str),
            postfix_prompt,
        ]
    )
    return user_prompt


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


start_query_time = None
query_cnt = 0


@retry.retry(tries=3, delay=30.0)
def call_chatgpt(system_prompt, user_prompt, openai_client, model, limit_per_minute=-1):
    def call():
        return openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            seed=0,
        )

    if limit_per_minute <= 0:  # no limit
        return call()

    else:  # limit the query num within a minute
        global start_query_time, query_cnt

        while True:
            now_time = datetime.datetime.now()
            if start_query_time is None or now_time - start_query_time > datetime.timedelta(minutes=1):
                start_query_time = now_time
                query_cnt = 0
            if query_cnt < limit_per_minute:
                query_cnt += 1
                return call()
            else:  # wait for a second and recheck
                time.sleep(1)
                continue


def filter_papers_by_title(
    paper_list,
    openai_client,
    system_prompt,
    topic_prompt,
    postfix_prompt,
    config,
    retry=3,
) -> Tuple[List[Paper], Dict, float, float, int, int]:
    batch_size = get_batch_size(int(config["SELECTION"]["title_batch_size"]), len(paper_list), config)
    print(f"Using batch size of {batch_size} for title filtering")
    batches_of_papers = batched(paper_list, batch_size)

    invalid_paper_list = []  # papers failed to be filtered by GPT, recorded for retrying
    new_paper_list = []
    filtered_results = {}
    total_prompt_cost = 0.0
    total_completion_cost = 0.0
    prompt_tokens = 0
    completion_tokens = 0

    for batch in tqdm(batches_of_papers, desc="Filtering title"):
        # prepare input
        papers_string = [paper_to_titles(paper) for paper in batch]
        user_prompt = get_user_prompt_for_title_filtering(topic_prompt, postfix_prompt, papers_string)
        model = config["SELECTION"]["model"]
        try:
            completion = call_chatgpt(system_prompt, user_prompt, openai_client, model)
        except Exception as ex:
            # if config["OUTPUT"].getboolean("debug_messages"):
            print(f"Exception happened: Failed to call GPT with batch size {len(batch)} ({ex.args})")
            invalid_paper_list.extend(batch)
            continue

        # get GPT output
        prompt_cost, completion_cost = calc_price(model, completion.usage)
        total_prompt_cost += prompt_cost
        total_completion_cost += completion_cost
        prompt_tokens += completion.usage.prompt_tokens
        completion_tokens += completion.usage.completion_tokens
        out_text = completion.choices[0].message.content
        print({"prompt": {"tokens": completion.usage.prompt_tokens, "cost": prompt_cost}, "completion": {"tokens": completion.usage.completion_tokens, "cost": completion_cost}})

        # parse output
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
            invalid_paper_list.extend(batch)
            if config["OUTPUT"].getboolean("debug_messages"):
                print(f"Exception happened: Failed to parse LM output as list ({ex.args})")
                print(f"`out_text`: {out_text}")
            continue

    print(f"Filtered {len(filtered_results)} papers based on title with cost of ${total_prompt_cost + total_completion_cost}, remaining {len(new_paper_list)} papers:\n"
          f"({prompt_tokens} prompt tokens cost ${total_prompt_cost})\n"
          f"({completion_tokens} completion tokens cost ${total_completion_cost})")

    if len(invalid_paper_list) > 0:
        if retry > 0:
            print(f"Retrying {len(invalid_paper_list)} papers failed to be filtered by GPT through title filtering (left {retry - 1} retries)")
            retried_new_paper_list, retried_filtered_results, retried_total_prompt_cost, retried_total_completion_cost, retried_prompt_tokens, retried_completion_tokens = filter_papers_by_title(
                invalid_paper_list,
                openai_client,
                system_prompt,
                topic_prompt,
                postfix_prompt,
                config,
                retry - 1,
            )
            new_paper_list.extend(retried_new_paper_list)
            filtered_results.update(retried_filtered_results)
            total_prompt_cost += retried_total_prompt_cost
            total_completion_cost += retried_total_completion_cost
            prompt_tokens += retried_prompt_tokens
            completion_tokens += retried_completion_tokens
        else:
            print(f"Maximum retries reached, skip retrying")
            print(f"Left {len(invalid_paper_list)} papers failed to be filtered by GPT through title filtering")
            print(f"Invalid paper titles:")
            for paper in invalid_paper_list:
                print(f"{paper.title}")

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
                print(f"Exception happened: Failed to parse LM output as json ({ex.args})")
                print(f"RAW output: {raw_out_text}")
                print(f"`out_text`: {out_text}")
            continue
    return json_dicts, invalid_cnt


def filter_papers_by_abstract(
    paper_list,
    id_paper_mapping,
    openai_client,
    system_prompt,
    topic_prompt,
    score_prompt,
    postfix_prompt,
    config,
    retry=3,
    limit_per_minute=-1,
) -> Tuple[List[List[Dict]], Dict, Dict, float, float, int, int]:
    batch_size = get_batch_size(int(config["SELECTION"]["abstract_batch_size"]), len(paper_list), config)
    print(f"Using batch size of {batch_size} for abstract filtering")
    batches_of_papers = batched(paper_list, batch_size)

    invalid_arxiv_ids = set()  # arxiv ids of papers failed to be scored by GPT, recorded for retrying
    scored_batches = []
    selected_results = {}
    filtered_results = {}
    total_prompt_cost = 0.0
    total_completion_cost = 0.0
    prompt_tokens = 0
    completion_tokens = 0

    for batch in tqdm(batches_of_papers, desc="Filtering abstract"):
        # temp values
        this_scored_batch = []
        all_arxiv_ids = {paper.arxiv_id for paper in batch}
        finished_arxiv_ids = set()

        # prepare input
        batch_str = [paper_to_string(paper) for paper in batch]
        user_prompt = get_user_prompt_for_abstract_filtering(topic_prompt, score_prompt, postfix_prompt, batch_str)
        model = config["SELECTION"]["model"]
        try:
            completion = call_chatgpt(system_prompt, user_prompt, openai_client, model, limit_per_minute=limit_per_minute)
        except Exception as ex:
            # if config["OUTPUT"].getboolean("debug_messages"):
            print(f"Exception happened: Failed to call GPT with batch size {len(batch)} ({ex.args})")
            invalid_arxiv_ids.update(all_arxiv_ids)
            continue

        # get GPT output
        prompt_cost, completion_cost = calc_price(model, completion.usage)
        total_prompt_cost += prompt_cost
        total_completion_cost += completion_cost
        prompt_tokens += completion.usage.prompt_tokens
        completion_tokens += completion.usage.completion_tokens
        out_text = completion.choices[0].message.content
        print({"prompt": {"tokens": completion.usage.prompt_tokens, "cost": prompt_cost}, "completion": {"tokens": completion.usage.completion_tokens, "cost": completion_cost}})

        # parse output
        json_dicts, _ = parse_chatgpt(out_text, config)

        for jdict in json_dicts:
            if jdict["ARXIVID"] not in id_paper_mapping:
                if config["OUTPUT"].getboolean("debug_messages"):
                    print(f"Exception happened: ARXIVID \"{jdict['ARXIVID']}\" not found in `id_paper_mapping`")
                continue

            result = {
                "SCORE": jdict["RELEVANCE"] + jdict["NOVELTY"],
                **jdict,
                **dataclasses.asdict(id_paper_mapping[jdict["ARXIVID"]]),
            }
            this_scored_batch.append(result)

            filtered = (
                int(jdict["RELEVANCE"]) < int(config["FILTERING"]["relevance_cutoff"]) or
                int(jdict["NOVELTY"]) < int(config["FILTERING"]["novelty_cutoff"])
            )
            if filtered:
                filtered_results[jdict["ARXIVID"]] = result
                print(f"Filtered out paper {jdict['ARXIVID']} by score (RELEVANCE={jdict['RELEVANCE']}, NOVELTY={jdict['NOVELTY']}) ({id_paper_mapping[jdict['ARXIVID']].title})")
            else:
                selected_results[jdict["ARXIVID"]] = result

            finished_arxiv_ids.add(jdict["ARXIVID"])
        scored_batches.append(this_scored_batch)

        # check if all papers are finished
        this_invalid_arxiv_ids = all_arxiv_ids - finished_arxiv_ids
        if len(this_invalid_arxiv_ids) > 0:
            invalid_arxiv_ids.update(this_invalid_arxiv_ids)

    print(f"Filtered {len(filtered_results)} papers based on abstract with cost of ${total_prompt_cost + total_completion_cost}, remaining {len(selected_results)} papers:\n"
          f"({prompt_tokens} prompt tokens cost ${total_prompt_cost})\n"
          f"({completion_tokens} completion tokens cost ${total_completion_cost})")

    # retry invalid arxiv ids
    if len(invalid_arxiv_ids) > 0:
        if retry > 0:
            print(f"Retrying {len(invalid_arxiv_ids)} papers failed to be scored by GPT through abstract filtering (left {retry - 1} retries)")
            retried_scored_batches, retried_selected_results, retried_filtered_results, retried_total_prompt_cost, retried_total_completion_cost, retried_prompt_tokens, retried_completion_tokens = filter_papers_by_abstract(
                [id_paper_mapping[arxiv_id] for arxiv_id in invalid_arxiv_ids],
                id_paper_mapping,
                openai_client,
                system_prompt,
                topic_prompt,
                score_prompt,
                postfix_prompt,
                config,
                retry - 1,
            )
            scored_batches.extend(retried_scored_batches)
            selected_results.update(retried_selected_results)
            filtered_results.update(retried_filtered_results)
            total_prompt_cost += retried_total_prompt_cost
            total_completion_cost += retried_total_completion_cost
            prompt_tokens += retried_prompt_tokens
            completion_tokens += retried_completion_tokens
        else:
            print(f"Maximum retries reached, skip retrying")
            print(f"Left {len(invalid_arxiv_ids)} papers failed to be scored by GPT through abstract filtering")
            print(f"Invalid paper titles:")
            for arxiv_id in invalid_arxiv_ids:
                print(f"{id_paper_mapping[arxiv_id].title}")

    return scored_batches, selected_results, filtered_results, total_prompt_cost, total_completion_cost, prompt_tokens, completion_tokens


def filter_by_gpt(
    paper_list,
    system_prompt,
    topic_prompt,
    score_prompt,
    postfix_prompt_title,
    postfix_prompt_abstract,
    config,
):
    total_filtered_results = {}
    total_prompt_cost = 0.0
    total_completion_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    id_paper_mapping: Dict[str, Paper] = {paper.arxiv_id: paper for paper in paper_list}

    # filter papers by titles
    if config["SELECTION"].getboolean("run_title_filter"):
        paper_list, filtered_results, prompt_cost, completion_cost, prompt_tokens, completion_tokens = filter_papers_by_title(
            paper_list,
            openai_client,
            system_prompt,
            topic_prompt,
            postfix_prompt_title,
            config,
            retry=int(config["SELECTION"]["title_retry"]),
        )
    else:
        filtered_results = {}
        prompt_cost, completion_cost, prompt_tokens, completion_tokens = 0.0, 0.0, 0, 0
        print("Skipping GPT title filtering")

    total_filtered_results.update(filtered_results)
    total_prompt_cost += prompt_cost
    total_completion_cost += completion_cost
    total_prompt_tokens += prompt_tokens
    total_completion_tokens += completion_tokens

    # filter remaining papers by abstracts
    if config["SELECTION"].getboolean("run_abstract_filter"):
        scored_batches, selected_results, filtered_results, prompt_cost, completion_cost, prompt_tokens, completion_tokens = filter_papers_by_abstract(
            paper_list,
            id_paper_mapping,
            openai_client,
            system_prompt,
            topic_prompt,
            score_prompt,
            postfix_prompt_abstract,
            config,
            retry=int(config["SELECTION"]["abstract_retry"]),
            limit_per_minute=int(config["SELECTION"]["limit_per_minute"]),
        )
    else:
        scored_batches = []
        selected_results = {paper.arxiv_id: {**dataclasses.asdict(paper)} for paper in paper_list}
        filtered_results = {}
        prompt_cost, completion_cost, prompt_tokens, completion_tokens = 0.0, 0.0, 0, 0
        print("Skipping GPT abstract filtering")

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

# if __name__ == "__main__":
#     openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
#
#     # loads papers from 'in/debug_papers.json' and filters them
#     with open("../../in/debug_papers.json", "r") as f:
#         paper_list_in_dict = json.load(f)
#
#     papers = [
#         [
#             Paper(
#                 arxiv_id=paper["arxiv_id"],
#                 authors=paper["authors"],
#                 title=paper["title"],
#                 abstract=paper["abstract"],
#             )
#             for paper in batch
#         ]
#         for batch in paper_list_in_dict
#     ]
#     all_papers = {}
#     paper_outputs = {}
#     sort_dict = {}
#     total_cost = 0
#     for batch in tqdm(papers):
#         batch_str = [paper_to_string(paper) for paper in batch]
#         user_prompt = get_full_prompt_for_abstract_filtering(SYSTEM_PROMPT, TOPIC_PROMPT, SCORE_PROMPT, POSTFIX_PROMPT_ABSTRACT, batch_str)
#         model = CONFIG["SELECTION"]["model"]
#         completion = call_chatgpt(user_prompt, openai_client, model)
#
#         prompt_cost, completion_cost = calc_price(model, completion.usage)
#         total_cost += prompt_cost + completion_cost
#         out_text = completion.choices[0].message.content
#
#         json_dicts = parse_chatgpt(out_text, CONFIG)
#         for paper in batch:
#             all_papers[paper.arxiv_id] = paper
#         for jdict in json_dicts:
#             paper_outputs[jdict["ARXIVID"]] = {
#                 **dataclasses.asdict(all_papers[jdict["ARXIVID"]]),
#                 **jdict,
#             }
#             sort_dict[jdict["ARXIVID"]] = jdict["RELEVANCE"] + jdict["NOVELTY"]
#
#     # sort the papers by relevance and novelty
#     print("total cost:" + str(total_cost))
#     keys = list(sort_dict.keys())
#     values = list(sort_dict.values())
#
#
#     def argsort(seq):
#         return sorted(range(len(seq)), key=seq.__getitem__)
#
#
#     sorted_keys = [keys[idx] for idx in argsort(values)[::-1]]
#     selected_papers = {key: paper_outputs[key] for key in sorted_keys}
#
#     with open(OUTPUT_DEBUG_FILE_FORMAT.format("filter_paper_test.json"), "w") as outfile:
#         json.dump(selected_papers, outfile, cls=EnhancedJSONEncoder, indent=4)
