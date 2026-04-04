import json
from html import escape
from typing import Dict, List, Tuple

from tabulate import tabulate

from arxiv_assistant.filters.filter_gpt import get_user_prompt_for_abstract_filtering
from arxiv_assistant.paper_topics import (
    group_sorted_papers_by_topic,
    paper_anchor_id,
    sort_paper_mapping_for_daily_display,
    topic_anchor_id,
)
from arxiv_assistant.utils.io import add_prefix_to_lines
from arxiv_assistant.utils.utils import Paper, align_markdown_table


def render_summary_table(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    prompt_cost: float,
    completion_cost: float,
    total_arxiv_papers: int,
    total_scanned_papers: int,
    total_relevant_papers: int,
) -> str:
    total_tokens = prompt_tokens + completion_tokens
    total_cost = prompt_cost + completion_cost

    return "\n".join(
        [
            "<table>",
            "    <thead>",
            "        <tr>",
            "            <th rowspan=\"2\">Model</th>",
            "            <th rowspan=\"2\">Metric</th>",
            "            <th colspan=\"3\">Usage</th>",
            "            <th colspan=\"3\">Papers</th>",
            "        </tr>",
            "        <tr>",
            "            <th>Prompt</th>",
            "            <th>Completion</th>",
            "            <th>Total</th>",
            "            <th>Total arXiv</th>",
            "            <th>Scanned</th>",
            "            <th>Relevant</th>",
            "        </tr>",
            "    </thead>",
            "    <tbody>",
            "        <tr>",
            f"            <td rowspan=\"2\" align=\"center\"><code>{escape(model)}</code></td>",
            "            <td align=\"center\"><strong>Tokens</strong></td>",
            f"            <td align=\"center\">{prompt_tokens}</td>",
            f"            <td align=\"center\">{completion_tokens}</td>",
            f"            <td align=\"center\">{total_tokens}</td>",
            f"            <td rowspan=\"2\" align=\"center\">{total_arxiv_papers}</td>",
            f"            <td rowspan=\"2\" align=\"center\">{total_scanned_papers}</td>",
            f"            <td rowspan=\"2\" align=\"center\">{total_relevant_papers}</td>",
            "        </tr>",
            "        <tr>",
            "            <td align=\"center\"><strong>Cost</strong></td>",
            f"            <td align=\"center\">${prompt_cost:.2f}</td>",
            f"            <td align=\"center\">${completion_cost:.2f}</td>",
            f"            <td align=\"center\">${total_cost:.2f}</td>",
            "        </tr>",
            "    </tbody>",
            "</table>",
        ]
    )


def render_title_and_author(paper_entry: Dict, idx: int) -> str:
    title = paper_entry["title"]
    authors = paper_entry["authors"]
    paper_string = f"{idx}. [{title}](#{paper_anchor_id(paper_entry['arxiv_id'])})\n"
    paper_string += f'**Authors:** {", ".join(authors)}'
    return paper_string


def render_paper_content(paper_entry: Dict, idx: int) -> str:
    """
    :param paper_entry: is a dict from a json. an example is
    {"paperId": "2754e70eaa0c2d40972c47c4c23210f0cece8bfc", "externalIds": {"ArXiv": "2310.16834", "CorpusId": 264451832}, "title": "Discrete Diffusion Language Modeling by Estimating the Ratios of the Data Distribution", "abstract": "Despite their groundbreaking performance for many generative modeling tasks, diffusion models have fallen short on discrete data domains such as natural language. Crucially, standard diffusion models rely on the well-established theory of score matching, but efforts to generalize this to discrete structures have not yielded the same empirical gains. In this work, we bridge this gap by proposing score entropy, a novel discrete score matching loss that is more stable than existing methods, forms an ELBO for maximum likelihood training, and can be efficiently optimized with a denoising variant. We scale our Score Entropy Discrete Diffusion models (SEDD) to the experimental setting of GPT-2, achieving highly competitive likelihoods while also introducing distinct algorithmic advantages. In particular, when comparing similarly sized SEDD and GPT-2 models, SEDD attains comparable perplexities (normally within $+10\\%$ of and sometimes outperforming the baseline). Furthermore, SEDD models learn a more faithful sequence distribution (around $4\\times$ better compared to GPT-2 models with ancestral sampling as measured by large models), can trade off compute for generation quality (needing only $16\\times$ fewer network evaluations to match GPT-2), and enables arbitrary infilling beyond the standard left to right prompting.", "year": 2023, "authors": [{"authorId": "2261494043", "name": "Aaron Lou"}, {"authorId": "83262128", "name": "Chenlin Meng"}, {"authorId": "2490652", "name": "Stefano Ermon"}], "ARXIVID": "2310.16834", "COMMENT": "The paper shows a significant advance in the performance of diffusion language models, directly meeting one of the criteria.", "RELEVANCE": 10, "NOVELTY": 8}, "2310.16779": {"paperId": "edc8953d559560d3237fc0b27175cdb1114c0ca5", "externalIds": {"ArXiv": "2310.16779", "CorpusId": 264451949}, "title": "Multi-scale Diffusion Denoised Smoothing", "abstract": "Along with recent diffusion models, randomized smoothing has become one of a few tangible approaches that offers adversarial robustness to models at scale, e.g., those of large pre-trained models. Specifically, one can perform randomized smoothing on any classifier via a simple\"denoise-and-classify\"pipeline, so-called denoised smoothing, given that an accurate denoiser is available - such as diffusion model. In this paper, we investigate the trade-off between accuracy and certified robustness of denoised smoothing: for example, we question on which representation of diffusion model would maximize the certified robustness of denoised smoothing. We consider a new objective that aims collective robustness of smoothed classifiers across multiple noise levels at a shared diffusion model, which also suggests a new way to compensate the cost of accuracy in randomized smoothing for its certified robustness. This objective motivates us to fine-tune diffusion model (a) to perform consistent denoising whenever the original image is recoverable, but (b) to generate rather diverse outputs otherwise. Our experiments show that this fine-tuning scheme of diffusion models combined with the multi-scale smoothing enables a strong certified robustness possible at highest noise level while maintaining the accuracy closer to non-smoothed classifiers.", "year": 2023, "authors": [{"authorId": "83125078", "name": "Jongheon Jeong"}, {"authorId": "2261688831", "name": "Jinwoo Shin"}], "ARXIVID": "2310.16779", "COMMENT": "The paper presents an advancement in the performance of diffusion models, specifically in the context of denoised smoothing.", "RELEVANCE": 9, "NOVELTY": 7}
    :return: a markdown formatted string showing the arxiv id, title, arxiv url, abstract, authors, score and comment (if those fields exist)
    """
    # get the arxiv id
    arxiv_id = paper_entry["arxiv_id"]
    # get the title
    title = paper_entry["title"]
    # get the arxiv url
    arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
    # get the abstract
    abstract = paper_entry["abstract"]
    # get the authors
    authors = paper_entry["authors"]
    paper_string = f'### {idx}. [{title}]({arxiv_url}) <a id="{paper_anchor_id(arxiv_id)}"></a>\n\n'
    paper_string += f"**ArXiv ID:** {arxiv_id}\n\n"
    paper_string += f"**Primary Topic:** {paper_entry.get('PRIMARY_TOPIC_LABEL', '')}\n\n"
    matched_topic_labels = [
        label
        for label in paper_entry.get("MATCHED_TOPIC_LABELS", [])
        if label != paper_entry.get("PRIMARY_TOPIC_LABEL", "")
    ]
    if matched_topic_labels:
        paper_string += f"**Also Matches:** {', '.join(matched_topic_labels)}\n\n"
    paper_string += f'**Authors:** {", ".join(authors)}\n\n'
    paper_string += f"**Abstract:** {abstract}\n\n"
    if "COMMENT" in paper_entry:
        comment = paper_entry["COMMENT"]
        paper_string += f"**Comment:** {comment}\n\n"
    topic_match_comment = paper_entry.get("TOPIC_MATCH_COMMENT", "")
    if topic_match_comment and topic_match_comment != paper_entry.get("COMMENT", ""):
        paper_string += f"**Topic Match:** {topic_match_comment}\n\n"
    if "RELEVANCE" in paper_entry and "NOVELTY" in paper_entry:
        # get the relevance and novelty scores
        relevance = paper_entry["RELEVANCE"]
        novelty = paper_entry["NOVELTY"]
        paper_string += f"**Relevance:** {relevance}\n"
        paper_string += f"**Novelty:** {novelty}"
    return paper_string


def render_topic_coverage_table(topic_sections: List[Dict]) -> str:
    lines = [
        "<table>",
        "    <thead>",
        "        <tr><th>Topic</th><th>Papers</th></tr>",
        "    </thead>",
        "    <tbody>",
    ]
    for section in topic_sections:
        lines.append(
            "        <tr>"
            f"<td><a href=\"#{topic_anchor_id(section['topic_id'])}\">{section['topic_label']}</a></td>"
            f"<td align=\"center\">{section['paper_count']}</td>"
            "</tr>"
        )
    lines.extend(["    </tbody>", "</table>"])
    return "\n".join(lines)


def render_topic_toc_section(topic_section: Dict) -> str:
    title_strings = [
        render_title_and_author(paper, idx + 1)
        for idx, paper in enumerate(topic_section["papers"])
    ]
    return "\n\n".join(
        [
            f"## [{topic_section['topic_label']}](#{topic_anchor_id(topic_section['topic_id'])}) ({topic_section['paper_count']})",
            "\n\n".join(title_strings),
        ]
    )


def render_topic_section(topic_section: Dict) -> str:
    paper_strings = [
        render_paper_content(paper, idx + 1)
        for idx, paper in enumerate(topic_section["papers"])
    ]
    return "\n\n".join(
        [
            f"## {topic_section['topic_label']} ({topic_section['paper_count']}) <a id=\"{topic_anchor_id(topic_section['topic_id'])}\"></a>",
            "\n\n---\n\n".join(paper_strings),
        ]
    )


def render_daily_md(
    all_entries: List,
    arxiv_paper_dict: Dict[str, List[Paper]],
    selected_paper_dict: Dict[str, Dict],
    now_date: Tuple[int, int, int] = None,  # year, month, day
    prompts: Tuple[str, str, str, str] = None,  # base, topic, score, postfix
    head_table: Dict = None,
):
    # render date content
    if now_date is not None:
        now_year, now_month, now_day = now_date
        date_string = f"{now_year}-{format(now_month, '02d')}-{format(now_day, '02d')}"
    else:
        date_string = ""

    # render head table
    if head_table is not None:
        if "html" in head_table:
            head_table_strings = head_table["html"]
        else:
            head_table_strings = tabulate(head_table["data"], headers=head_table["headers"], tablefmt="github")
            head_table_strings = align_markdown_table(head_table_strings, "center")
    else:
        head_table_strings = ""

    selected_paper_dict = sort_paper_mapping_for_daily_display(selected_paper_dict)
    sorted_papers = list(selected_paper_dict.values())
    topic_sections = group_sorted_papers_by_topic(sorted_papers)
    toc_sections = [
        render_topic_toc_section(section)
        for section in topic_sections
    ]
    paper_sections = [
        render_topic_section(section)
        for section in topic_sections
    ]

    # render prompt
    if prompts is not None:
        system_prompt, topic_prompt, score_prompt, postfix_prompt = prompts
        user_prompt = get_user_prompt_for_abstract_filtering(
            topic_prompt,
            score_prompt,
            postfix_prompt,
            ['[PAPER LIST HERE]']
        )
    else:
        system_prompt = ""
        user_prompt = ""

    # cat output string
    output_string = "\n\n".join([
        f"# Personalized Daily ArXiv Papers {date_string}",
        head_table_strings,
        "**Topic Coverage:**",
        render_topic_coverage_table(topic_sections) if topic_sections else "No papers selected.",
        "**Table of contents by topic:**",
        "\n\n".join(toc_sections),
        "---",
        "\n\n---\n\n".join(paper_sections),
        "---",
        "# Paper Selection Prompt",
        "## System Prompt",
        add_prefix_to_lines(system_prompt, "> "),
        "## User Prompt",
        add_prefix_to_lines(user_prompt, "> "),
    ])
    return output_string


if __name__ == "__main__":
    # parse output.json into a dict
    with open("../../../out_local/json/2025-01/2025-01-17-output.json", "r") as f:
        output = json.load(f)
    # simulate head table
    head_table = {
        "headers": ["", "Prompt", "Completion", "Total"],
        "data": [
            ["**Token**", 100, 100, 200],
            ["**Cost**", f"${0.5}", f"${0.5}", f"${1.0}"],
        ]
    }
    # write to latest.md
    with open("out/latest.md", "w") as f:
        f.write(render_daily_md(output, head_table=head_table))
