import json
import os

from arxiv_assistant.apis.arxiv import get_papers_from_arxiv
from arxiv_assistant.apis.semantic_scholar import get_authors
from arxiv_assistant.environment import AUTHOR_ID_SET, SYSTEM_PROMPT, CONFIG, NOW_DAY, NOW_MONTH, NOW_YEAR, OUTPUT_DEBUG_FILE_FORMAT, OUTPUT_JSON_FILE_FORMAT, OUTPUT_MD_FILE_FORMAT, POSTFIX_PROMPT_ABSTRACT, POSTFIX_PROMPT_TITLE, S2_API_KEY, SCORE_PROMPT, SLACK_KEY, TOPIC_PROMPT
from arxiv_assistant.filters.filter_author import filter_papers_by_hindex, select_by_author
from arxiv_assistant.filters.filter_gpt import filter_by_gpt
from arxiv_assistant.paper_topics import build_daily_topic_bundle, build_hotspot_paper_bundle, ensure_topic_fields_for_mapping, sort_paper_mapping_for_daily_display
from arxiv_assistant.push_to_slack import push_to_slack
from arxiv_assistant.renderers.paper.render_daily import render_daily_md, render_summary_table
from arxiv_assistant.utils.io import copy_file_or_dir, delete_file_or_dir
from arxiv_assistant.utils.utils import EnhancedJSONEncoder

if __name__ == "__main__":
    # get the paper list from arxiv
    all_entries, arxiv_paper_dict = get_papers_from_arxiv(CONFIG, source="rss")
    paper_list = list(set(v for area_papers in arxiv_paper_dict.values() for v in area_papers))
    print("Total number of papers:" + str(len(paper_list)))
    if len(paper_list) == 0:
        print("No papers found")
        exit(0)

    # get the author list from papers
    if CONFIG["SELECTION"].getboolean("run_author_match"):
        all_authors = set()
        for paper in paper_list:
            all_authors.update(set(paper.authors))
        print("Getting author info for " + str(len(all_authors)) + " authors")
        all_authors = get_authors(list(all_authors), S2_API_KEY, config=CONFIG)
    else:
        print("Skipping author info")
        all_authors = {}

    # dump all papers for debugging
    if CONFIG["OUTPUT"].getboolean("dump_debug_file"):
        with open(OUTPUT_DEBUG_FILE_FORMAT.format("config.json"), "w") as outfile:
            json.dump({section: dict(CONFIG[section]) for section in CONFIG.sections()}, outfile, cls=EnhancedJSONEncoder, indent=4)
        with open(OUTPUT_DEBUG_FILE_FORMAT.format("author_id_set.json"), "w") as outfile:
            json.dump(list(AUTHOR_ID_SET), outfile, cls=EnhancedJSONEncoder, indent=4)
        with open(OUTPUT_DEBUG_FILE_FORMAT.format("all_papers.json"), "w") as outfile:
            json.dump(paper_list, outfile, cls=EnhancedJSONEncoder, indent=4)
        with open(OUTPUT_DEBUG_FILE_FORMAT.format("all_authors.json"), "w") as outfile:
            json.dump(all_authors, outfile, cls=EnhancedJSONEncoder, indent=4)

    # initialize vars for filtering
    selected_paper_dict = {}
    filtered_paper_dict = {}  # NOTE: NOT USED HERE

    # select papers by author
    if CONFIG["SELECTION"].getboolean("run_author_match"):
        paper_list, selected_results = select_by_author(
            all_authors,
            paper_list,
            AUTHOR_ID_SET,
            CONFIG
        )
        selected_paper_dict.update(selected_results)
    else:
        print("Skipping selection by author")

    # filter papers by h-index
    if CONFIG["SELECTION"].getboolean("run_author_match"):
        paper_list, filtered_results = filter_papers_by_hindex(
            all_authors,
            paper_list,
            CONFIG
        )
        filtered_paper_dict.update(filtered_results)
    else:
        print("Skipping h-index filtering")

    hotspot_paper_bundle = build_hotspot_paper_bundle((NOW_YEAR, NOW_MONTH, NOW_DAY), {})

    # filter papers by GPT
    if CONFIG["SELECTION"].getboolean("run_openai"):
        selected_results, filtered_results, total_prompt_cost, total_completion_cost, total_prompt_tokens, total_completion_tokens = filter_by_gpt(
            paper_list,
            SYSTEM_PROMPT,
            TOPIC_PROMPT,
            SCORE_PROMPT,
            POSTFIX_PROMPT_TITLE,
            POSTFIX_PROMPT_ABSTRACT,
            CONFIG,
        )
        selected_paper_dict.update(selected_results)
        filtered_paper_dict.update(filtered_results)

        scored_results_for_hotspot = ensure_topic_fields_for_mapping(
            {
                arxiv_id: paper_entry
                for arxiv_id, paper_entry in {**selected_results, **filtered_results}.items()
                if "RELEVANCE" in paper_entry or "NOVELTY" in paper_entry
            }
        )
        hotspot_paper_bundle = build_hotspot_paper_bundle(
            (NOW_YEAR, NOW_MONTH, NOW_DAY),
            scored_results_for_hotspot,
            max_daily_hot=CONFIG["HOTSPOTS"].getint("paper_spotlight_max_daily_hot", fallback=6),
            max_new_frontier=CONFIG["HOTSPOTS"].getint("paper_spotlight_max_new_frontier", fallback=4),
            daily_hot_score_cutoff=CONFIG["HOTSPOTS"].getint("paper_spotlight_daily_hot_score_cutoff", fallback=15),
            daily_hot_relevance_cutoff=CONFIG["HOTSPOTS"].getint("paper_spotlight_daily_hot_relevance_cutoff", fallback=7),
            new_frontier_score_cutoff=CONFIG["HOTSPOTS"].getint("paper_spotlight_new_frontier_score_cutoff", fallback=15),
            new_frontier_novelty_cutoff=CONFIG["HOTSPOTS"].getint("paper_spotlight_new_frontier_novelty_cutoff", fallback=8),
        )
        hotspot_paper_ids = set(hotspot_paper_bundle["papers"].keys())
        if hotspot_paper_ids:
            selected_paper_dict = {
                arxiv_id: paper_entry
                for arxiv_id, paper_entry in selected_paper_dict.items()
                if arxiv_id not in hotspot_paper_ids
            }
            for arxiv_id in hotspot_paper_ids:
                spotlight_entry = dict(scored_results_for_hotspot[arxiv_id])
                spotlight_entry["DIVERTED_TO_HOTSPOT_PAPERS"] = True
                spotlight_entry["FILTER_REASON"] = "Diverted to the daily hotspot paper spotlight."
                filtered_paper_dict[arxiv_id] = {
                    **filtered_paper_dict.get(arxiv_id, {}),
                    **spotlight_entry,
                }
    else:
        total_prompt_cost, total_completion_cost, total_prompt_tokens, total_completion_tokens = 0.0, 0.0, 0, 0
        print("Skipping GPT filtering")

    selected_paper_dict = sort_paper_mapping_for_daily_display(selected_paper_dict)
    filtered_paper_dict = ensure_topic_fields_for_mapping(filtered_paper_dict)
    daily_topic_bundle = build_daily_topic_bundle(
        (NOW_YEAR, NOW_MONTH, NOW_DAY),
        selected_paper_dict,
        usage={
            "model": CONFIG["SELECTION"]["model"],
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "prompt_cost": total_prompt_cost,
            "completion_cost": total_completion_cost,
            "total_arxiv_papers": len(all_entries),
            "total_scanned_papers": sum(len(area_papers) for area_papers in arxiv_paper_dict.values()),
            "total_relevant_papers": len(selected_paper_dict),
        },
    )

    # dump filtered & selected papers for debugging
    if CONFIG["OUTPUT"].getboolean("dump_debug_file"):
        with open(OUTPUT_DEBUG_FILE_FORMAT.format("selected_paper_dict.json"), "w") as outfile:
            json.dump(selected_paper_dict, outfile, cls=EnhancedJSONEncoder, indent=4)
        with open(OUTPUT_DEBUG_FILE_FORMAT.format("filtered_paper_dict.json"), "w") as outfile:
            json.dump(filtered_paper_dict, outfile, cls=EnhancedJSONEncoder, indent=4)
        with open(OUTPUT_DEBUG_FILE_FORMAT.format("topic_diagnostics.json"), "w") as outfile:
            json.dump(daily_topic_bundle["diagnostics"], outfile, indent=4)
        with open(OUTPUT_DEBUG_FILE_FORMAT.format("hotspot_paper_bundle.json"), "w") as outfile:
            json.dump(hotspot_paper_bundle, outfile, indent=4)

    if CONFIG["OUTPUT"].getboolean("dump_json"):
        with open(OUTPUT_JSON_FILE_FORMAT.format("output.json"), "w") as outfile:
            json.dump(selected_paper_dict, outfile, indent=4)
        with open(OUTPUT_JSON_FILE_FORMAT.format("daily-papers.json"), "w") as outfile:
            json.dump(daily_topic_bundle, outfile, indent=4)
        with open(OUTPUT_JSON_FILE_FORMAT.format("hotspot-papers.json"), "w") as outfile:
            json.dump(hotspot_paper_bundle, outfile, indent=4)

    if CONFIG["OUTPUT"].getboolean("dump_md"):
        head_table = {
            "html": render_summary_table(
                model=CONFIG["SELECTION"]["model"],
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                prompt_cost=total_prompt_cost,
                completion_cost=total_completion_cost,
                total_arxiv_papers=len(all_entries),
                total_scanned_papers=sum(len(area_papers) for area_papers in arxiv_paper_dict.values()),
                total_relevant_papers=len(selected_paper_dict),
            )
        }
        with open(OUTPUT_MD_FILE_FORMAT.format("latest.md"), "w") as f:
            f.write(render_daily_md(all_entries, arxiv_paper_dict, selected_paper_dict, now_date=(NOW_YEAR, NOW_MONTH, NOW_DAY), prompts=(SYSTEM_PROMPT, TOPIC_PROMPT, SCORE_PROMPT, POSTFIX_PROMPT_ABSTRACT), head_table=head_table))

    # only push to slack for non-empty dicts
    if CONFIG["OUTPUT"].getboolean("push_to_slack"):
        if SLACK_KEY is None:
            print("Warning: push_to_slack is true, but SLACK_KEY is not set - not pushing to slack")
        else:
            push_to_slack(selected_paper_dict)

    # copy files
    copy_file_or_dir(OUTPUT_MD_FILE_FORMAT.format("latest.md"), CONFIG["OUTPUT"]["output_path"])
    delete_file_or_dir(os.path.join(CONFIG["OUTPUT"]["output_path"], "latest.md"))
    os.rename(
        os.path.join(CONFIG["OUTPUT"]["output_path"], os.path.basename(OUTPUT_MD_FILE_FORMAT.format("latest.md"))),
        os.path.join(CONFIG["OUTPUT"]["output_path"], "latest.md"),
    )
