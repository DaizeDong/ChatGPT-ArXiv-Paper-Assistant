import json
from datetime import datetime

from tabulate import tabulate

from environment import BASE_PROMPT, POSTFIX_PROMPT, SCORE_PROMPT, TOPIC_PROMPT
from filter_papers import get_full_prompt_for_abstract_filtering


def render_title_and_author(paper_entry: dict, idx: int) -> str:
    title = paper_entry["title"]
    authors = paper_entry["authors"]
    paper_string = f"{idx}. [{title}](#user-content-link{idx})\n"
    paper_string += f'**Authors:** {", ".join(authors)}'
    return paper_string


def render_paper(paper_entry: dict, idx: int) -> str:
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
    paper_string = f'## {idx}. [{title}]({arxiv_url}) <a id="link{idx}"></a>\n\n'
    paper_string += f"**ArXiv ID:** {arxiv_id}\n\n"
    paper_string += f'**Authors:** {", ".join(authors)}\n\n'
    paper_string += f"**Abstract:** {abstract}\n\n"
    if "COMMENT" in paper_entry:
        comment = paper_entry["COMMENT"]
        paper_string += f"**Comment:** {comment}\n\n"
    if "RELEVANCE" in paper_entry and "NOVELTY" in paper_entry:
        # get the relevance and novelty scores
        relevance = paper_entry["RELEVANCE"]
        novelty = paper_entry["NOVELTY"]
        paper_string += f"**Relevance:** {relevance}\n"
        paper_string += f"**Novelty:** {novelty}"
    return paper_string


def render_md_string(papers_dict, head_table=None):
    # render head table
    headers = head_table["headers"]
    data = head_table["data"]
    head_table_strings = tabulate(data, headers=headers, tablefmt="github")

    # render each paper
    title_strings = [
        render_title_and_author(paper, i + 1)
        for i, paper in enumerate(papers_dict.values())
    ]
    paper_strings = [
        render_paper(paper, i + 1)
        for i, paper in enumerate(papers_dict.values())
    ]

    # render prompt
    prompt_strings = get_full_prompt_for_abstract_filtering(
        BASE_PROMPT,
        TOPIC_PROMPT,
        SCORE_PROMPT,
        POSTFIX_PROMPT,
        ['[PAPER LIST HERE]']
    )

    # cat output string
    output_string = "\n\n".join([
        f"# Personalized Daily Arxiv Papers {datetime.today().strftime('%m/%d/%Y')}",
        "" if head_table is None else head_table_strings,
        f"Total relevant papers: {len(papers_dict)}",
        "**Table of contents with paper titles:**",
        "\n\n".join(title_strings),
        "---",
        "\n\n---\n\n".join(paper_strings),
        "---",
        "# Paper Selection Prompt",
        prompt_strings
    ])
    return output_string


if __name__ == "__main__":
    # parse output.json into a dict
    with open("out_local/json/2025-01/2025-01-17-output.json", "r") as f:
        output = json.load(f)
    # simulate head table
    head_table = {
        "headers": ["", "Prompt", "Completion", "Total"],
        "data": [
            ["**Token**", 100, 100, 200],
            ["**Cost**", f"${0.5}", f"${0.5}", f"${1.0}"],
        ]
    }
    # write to output.md
    with open("out/output.md", "w") as f:
        f.write(render_md_string(output, head_table=head_table))
