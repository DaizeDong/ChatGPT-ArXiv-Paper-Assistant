from typing import Dict, List, Tuple

from arxiv_assistant.renderers.paper.monthly_summary import MONTH_CATEGORY_ORDER
from arxiv_assistant.renderers.site_paths import (
    relative_site_href,
    site_day_page_path,
    site_month_page_path,
    site_month_summary_page_path,
)

HEADER_SEPARATOR = "&nbsp;|&nbsp;"


def _render_link(href: str, text: str) -> str:
    return f'<a href="{href}">{text}</a>'


def _join_nav_parts(parts: list[str]) -> str:
    return HEADER_SEPARATOR.join(part for part in parts if part)


def _render_summary_header(
    current_page_path: str,
    now_month: Tuple[int, int],
    all_months_list: List[Tuple[int, int]],
) -> str:
    now_idx = all_months_list.index(now_month)
    previous_month = all_months_list[now_idx - 1] if now_idx > 0 else None
    next_month = all_months_list[now_idx + 1] if now_idx + 1 < len(all_months_list) else None

    previous_href = (
        relative_site_href(site_month_summary_page_path(previous_month), current_page_path)
        if previous_month is not None
        else None
    )
    month_href = relative_site_href(site_month_page_path(now_month), current_page_path)
    next_href = (
        relative_site_href(site_month_summary_page_path(next_month), current_page_path)
        if next_month is not None
        else None
    )

    line_one = _join_nav_parts(
        [
            _render_link(previous_href, "&larr; Previous Summary") if previous_href else "",
            _render_link(month_href, "Monthly Overview"),
            _render_link(next_href, "Next Summary &rarr;") if next_href else "",
        ]
    )
    line_two = _join_nav_parts(
        [
            _render_link(previous_href, f"{previous_month[0]}-{previous_month[1]:02d}") if previous_month else "",
            _render_link(month_href, f"{now_month[0]}-{now_month[1]:02d}"),
            _render_link(next_href, f"{next_month[0]}-{next_month[1]:02d}") if next_month else "",
        ]
    )

    return f'<div align="center">{line_one}<br>{line_two}</div>'


def _render_summary_stats(category_buckets: Dict[str, List[Dict]]) -> str:
    total_papers = sum(len(papers) for papers in category_buckets.values())
    lines = [
        "<table>",
        "    <thead>",
        "        <tr><th>Metric</th><th>Value</th></tr>",
        "    </thead>",
        "    <tbody>",
        f"        <tr><td><strong>Total Papers</strong></td><td align=\"center\">{total_papers}</td></tr>",
    ]

    for category in MONTH_CATEGORY_ORDER:
        lines.append(
            f"        <tr><td>{category}</td><td align=\"center\">{len(category_buckets.get(category, []))}</td></tr>"
        )

    lines.extend(["    </tbody>", "</table>"])
    return "\n".join(lines)


def _render_paper_line(current_page_path: str, paper_entry: Dict, idx: int) -> str:
    date = paper_entry.get("SOURCE_DATE", (0, 0, 0))
    arxiv_id = paper_entry["arxiv_id"]
    day_page_href = relative_site_href(paper_entry.get("SOURCE_DAY_PAGE", site_day_page_path(date)), current_page_path)
    arxiv_href = f"https://arxiv.org/abs/{arxiv_id}"
    score = paper_entry.get("SCORE", 0)
    relevance = paper_entry.get("RELEVANCE", 0)
    novelty = paper_entry.get("NOVELTY", 0)
    comment = paper_entry.get("MONTHLY_COMMENT", "") or paper_entry.get("COMMENT", "")

    lines = [
        f"{idx}. [{paper_entry['title']}]({arxiv_href})",
        f"   - **Score:** {score} (R={relevance}, N={novelty})",
        f"   - **Date:** [{date[0]}-{date[1]:02d}-{date[2]:02d}]({day_page_href})",
    ]
    if comment:
        lines.append(f"   - **Comment:** {comment}")
    return "\n".join(lines)


def _render_category_section(
    category: str,
    papers: List[Dict],
    current_page_path: str,
) -> str:
    if not papers:
        return ""

    paper_lines = [
        _render_paper_line(current_page_path, paper_entry, idx + 1)
        for idx, paper_entry in enumerate(papers)
    ]
    return "\n\n".join(
        [
            f"## {category} ({len(papers)})",
            "\n\n".join(paper_lines),
        ]
    )


def render_monthly_summary_md_with_hyperlink(
    now_month: Tuple[int, int],
    current_page_path: str,
    category_buckets: Dict[str, List[Dict]],
    all_summary_months_list: List[Tuple[int, int]],
) -> str:
    year, month_num = now_month
    head_string = _render_summary_header(current_page_path, now_month, all_summary_months_list)
    stats_table = _render_summary_stats(category_buckets)
    sections = [
        _render_category_section(category, category_buckets.get(category, []), current_page_path)
        for category in MONTH_CATEGORY_ORDER
    ]
    rendered_sections = [section for section in sections if section]

    return "\n\n".join(
        [
            head_string,
            f"# Personalized Monthly Topic Summary {year}/{month_num:02d}",
            stats_table,
            *rendered_sections,
        ]
    )
