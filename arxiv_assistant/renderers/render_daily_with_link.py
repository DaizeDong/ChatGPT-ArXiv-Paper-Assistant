from typing import Dict, List, Set, Tuple

from arxiv_assistant.renderers.site_paths import (
    month_from_date,
    relative_site_href,
    site_day_page_path,
    site_month_page_path,
)
from arxiv_assistant.utils.utils import Paper


def _render_day_nav_cell(
    href: str | None,
    title: str,
    subtitle: str,
    arrow_html: str,
    align: str,
) -> str:
    if href is None:
        return '<td style="border: none; padding: 0px; width: 33%;"></td>'

    if align == "left":
        content = f"""
            <div style="display: flex; align-items: center; gap: 5px;">
                <a href="{href}" style="text-decoration: none; color: inherit;">
                    <strong style="color: black; font-size: 18px;">{arrow_html}</strong>
                </a>
                <div style="text-align: center;">
                    <a href="{href}" style="text-decoration: none;">
                        <strong>{title}</strong>
                    </a>
                    <br>
                    <a href="{href}" style="text-decoration: none; font-size: 14px; color: gray;">
                        {subtitle}
                    </a>
                </div>
            </div>
        """
    else:
        content = f"""
            <div style="display: flex; align-items: center; gap: 5px; justify-content: flex-end; width: 100%;">
                <div style="text-align: center;">
                    <a href="{href}" style="text-decoration: none;">
                        <strong>{title}</strong>
                    </a>
                    <br>
                    <a href="{href}" style="text-decoration: none; font-size: 14px; color: gray;">
                        {subtitle}
                    </a>
                </div>
                <a href="{href}" style="text-decoration: none; color: inherit;">
                    <strong style="color: black; font-size: 18px;">{arrow_html}</strong>
                </a>
            </div>
        """

    return f'<td style="border: none; padding: 0px; width: 33%;">{content}</td>'


def _render_month_nav_cell(href: str, title: str, subtitle: str) -> str:
    return f"""
        <td style="border: none; padding: 0px; width: 34%; text-align: center;">
            <div style="display: flex; flex-direction: column; align-items: center; width: 100%;">
                <a href="{href}" style="text-decoration: none;">
                    <strong>{title}</strong>
                </a>
                <a href="{href}" style="text-decoration: none; font-size: 14px; color: gray;">
                    {subtitle}
                </a>
            </div>
        </td>
    """


def _render_day_header(
    current_page_path: str,
    now_date: Tuple[int, int, int],
    all_dates_list: List[Tuple[int, int, int]],
) -> str:
    now_year, now_month, _ = now_date
    now_idx = all_dates_list.index(now_date)

    previous_date = all_dates_list[now_idx - 1] if now_idx > 0 else None
    next_date = all_dates_list[now_idx + 1] if now_idx + 1 < len(all_dates_list) else None
    month_href = relative_site_href(
        site_month_page_path(month_from_date(now_date)),
        current_page_path,
    )

    previous_href = (
        relative_site_href(site_day_page_path(previous_date), current_page_path)
        if previous_date is not None
        else None
    )
    next_href = (
        relative_site_href(site_day_page_path(next_date), current_page_path)
        if next_date is not None
        else None
    )

    return "\n".join(
        [
            '<table style="width: 100%; border-collapse: collapse; border: none;">',
            "    <tr>",
            _render_day_nav_cell(
                previous_href,
                "Previous Day",
                f"{previous_date[0]}-{previous_date[1]:02d}-{previous_date[2]:02d}" if previous_date else "",
                "&larr;",
                "left",
            ),
            _render_month_nav_cell(
                month_href,
                "Monthly Overview",
                f"{now_year}-{now_month:02d}",
            ),
            _render_day_nav_cell(
                next_href,
                "Next Day",
                f"{next_date[0]}-{next_date[1]:02d}-{next_date[2]:02d}" if next_date else "",
                "&rarr;",
                "right",
            ),
            "    </tr>",
            "</table>",
        ]
    )


def render_daily_md_with_hyperlink(
    now_date: Tuple[int, int, int],
    current_page_path: str,
    all_dates: Set[Tuple[int, int, int]] = None,
    content_string: str = None,
    all_entries: List = None,
    arxiv_paper_dict: Dict[str, List[Paper]] = None,
    selected_paper_dict: Dict[str, Dict] = None,
    prompts: Tuple[str, str, str, str] = None,
    head_table: Dict = None,
) -> str:
    all_dates_set = set(all_dates or set())
    all_dates_set.add(now_date)
    all_dates_list = sorted(all_dates_set)

    if content_string is None:
        from arxiv_assistant.renderers.render_daily import render_daily_md

        required_args = [all_entries, arxiv_paper_dict, selected_paper_dict, prompts, head_table]
        assert all(arg is not None for arg in required_args)
        content_string = render_daily_md(
            all_entries,
            arxiv_paper_dict,
            selected_paper_dict,
            now_date=now_date,
            prompts=prompts,
            head_table=head_table,
        )

    head_string = _render_day_header(current_page_path, now_date, all_dates_list)
    return "\n\n".join([head_string, content_string])


if __name__ == "__main__":
    pass
