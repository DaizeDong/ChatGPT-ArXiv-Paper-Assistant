from typing import Dict, List, Set, Tuple

from arxiv_assistant.renderers.site_paths import (
    month_from_date,
    relative_site_href,
    relative_site_path,
    site_day_page_path,
    site_month_page_path,
)
from arxiv_assistant.utils.utils import Paper

FALLBACK_SEPARATOR = "&nbsp;|&nbsp;"


def _render_text_link(href: str, text: str) -> str:
    return f'<a href="{href}">{text}</a>'


def _join_fallback_parts(parts: list[str]) -> str:
    return FALLBACK_SEPARATOR.join(part for part in parts if part)


def _render_day_header(
    current_page_path: str,
    now_date: Tuple[int, int, int],
    all_dates_list: List[Tuple[int, int, int]],
    previous_asset_path: str | None,
    next_asset_path: str | None,
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

    if previous_href and previous_asset_path or next_href and next_asset_path:
        parts = ["<div>"]
        if previous_href and previous_asset_path and previous_date is not None:
            previous_asset_href = relative_site_path(previous_asset_path, current_page_path)
            parts.append(
                f'<a href="{previous_href}"><img align="left" src="{previous_asset_href}" '
                f'alt="Previous Day {previous_date[0]}-{previous_date[1]:02d}-{previous_date[2]:02d}"></a>'
            )
        if next_href and next_asset_path and next_date is not None:
            next_asset_href = relative_site_path(next_asset_path, current_page_path)
            parts.append(
                f'<a href="{next_href}"><img align="right" src="{next_asset_href}" '
                f'alt="Next Day {next_date[0]}-{next_date[1]:02d}-{next_date[2]:02d}"></a>'
            )
        parts.append('<div align="center">')
        parts.append(_render_text_link(month_href, f"Monthly Overview<br>{now_year}-{now_month:02d}"))
        parts.append("</div>")
        parts.append('<br clear="all">')
        parts.append("</div>")
        return "".join(parts)

    line_one = _join_fallback_parts(
        [
            _render_text_link(previous_href, "&larr; Previous Day") if previous_href else "",
            _render_text_link(month_href, "Monthly Overview"),
            _render_text_link(next_href, "Next Day &rarr;") if next_href else "",
        ]
    )
    line_two = _join_fallback_parts(
        [
            _render_text_link(previous_href, f"{previous_date[0]}-{previous_date[1]:02d}-{previous_date[2]:02d}")
            if previous_date
            else "",
            _render_text_link(month_href, f"{now_year}-{now_month:02d}"),
            _render_text_link(next_href, f"{next_date[0]}-{next_date[1]:02d}-{next_date[2]:02d}")
            if next_date
            else "",
        ]
    )

    return f'<div align="center">{line_one}<br>{line_two}</div>'


def render_daily_md_with_hyperlink(
    now_date: Tuple[int, int, int],
    current_page_path: str,
    all_dates: Set[Tuple[int, int, int]] = None,
    previous_asset_path: str | None = None,
    next_asset_path: str | None = None,
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

    head_string = _render_day_header(
        current_page_path,
        now_date,
        all_dates_list,
        previous_asset_path,
        next_asset_path,
    )
    return "\n\n".join([head_string, content_string])


if __name__ == "__main__":
    pass
