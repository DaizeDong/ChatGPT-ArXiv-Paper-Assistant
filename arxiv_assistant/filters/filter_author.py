import dataclasses


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
