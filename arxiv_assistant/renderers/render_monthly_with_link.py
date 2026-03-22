import calendar
from typing import Dict, Tuple

from arxiv_assistant.renderers.site_paths import (
    month_from_date,
    relative_site_href,
    relative_site_path,
    site_month_page_path,
    site_year_page_path,
)

HEADER_SEPARATOR = "&nbsp;|&nbsp;"


def _render_link(href: str, text: str) -> str:
    return f'<a href="{href}">{text}</a>'


def _join_nav_parts(parts: list[str]) -> str:
    return HEADER_SEPARATOR.join(part for part in parts if part)


def _render_month_header(
    current_page_path: str,
    now_month: Tuple[int, int],
    all_months_list: list[Tuple[int, int]],
    previous_asset_path: str | None,
    center_asset_path: str | None,
    next_asset_path: str | None,
) -> str:
    now_idx = all_months_list.index(now_month)
    now_year, _ = now_month
    previous_month = all_months_list[now_idx - 1] if now_idx > 0 else None
    next_month = all_months_list[now_idx + 1] if now_idx + 1 < len(all_months_list) else None
    year_href = relative_site_href(site_year_page_path(now_year), current_page_path)

    previous_href = (
        relative_site_href(site_month_page_path(previous_month), current_page_path)
        if previous_month is not None
        else None
    )
    next_href = (
        relative_site_href(site_month_page_path(next_month), current_page_path)
        if next_month is not None
        else None
    )

    if previous_href and previous_asset_path or center_asset_path or next_href and next_asset_path:
        parts = ["<div>"]
        if previous_href and previous_asset_path and previous_month is not None:
            previous_asset_href = relative_site_path(previous_asset_path, current_page_path)
            parts.append(
                f'<a href="{previous_href}"><img align="left" src="{previous_asset_href}" '
                f'alt="Previous Month {previous_month[0]}-{previous_month[1]:02d}"></a>'
            )
        if next_href and next_asset_path and next_month is not None:
            next_asset_href = relative_site_path(next_asset_path, current_page_path)
            parts.append(
                f'<a href="{next_href}"><img align="right" src="{next_asset_href}" '
                f'alt="Next Month {next_month[0]}-{next_month[1]:02d}"></a>'
            )
        parts.append('<div align="center">')
        if center_asset_path is not None:
            center_asset_href = relative_site_path(center_asset_path, current_page_path)
            parts.append(f'<a href="{year_href}"><img src="{center_asset_href}" alt="Yearly Overview {now_year}"></a>')
        else:
            parts.append(_render_link(year_href, f"Yearly Overview<br>{now_year}"))
        parts.append("</div>")
        parts.append('<br clear="all">')
        parts.append("</div>")
        return "".join(parts)

    line_one = _join_nav_parts(
        [
            _render_link(previous_href, "&larr; Previous Month") if previous_href else "",
            _render_link(year_href, "Yearly Overview"),
            _render_link(next_href, "Next Month &rarr;") if next_href else "",
        ]
    )
    line_two = _join_nav_parts(
        [
            _render_link(previous_href, f"{previous_month[0]}-{previous_month[1]:02d}") if previous_month else "",
            _render_link(year_href, str(now_year)),
            _render_link(next_href, f"{next_month[0]}-{next_month[1]:02d}") if next_month else "",
        ]
    )

    return f'<div align="center">{line_one}<br>{line_two}</div>'


def render_content_table(
    now_date: Tuple[int, int, int],
    current_page_path: str,
    all_date_file_mapping: Dict[Tuple[int, int, int], str] = None,
) -> str:
    all_date_file_mapping = all_date_file_mapping or {}
    now_year, now_month, _ = now_date

    month_calendar = calendar.Calendar(firstweekday=6).monthdayscalendar(now_year, now_month)
    headers = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

    lines = [
        '<table style="margin: 0 auto; border-collapse: collapse; text-align: center;">',
        "    <thead>",
        '        <tr><th colspan="7" style="padding: 8px 0;">'
        f"{now_year}/{now_month:02d}"
        "</th></tr>",
        "        <tr>"
        + "".join(f'<th style="padding: 6px;">{header}</th>' for header in headers)
        + "</tr>",
        "    </thead>",
        "    <tbody>",
    ]

    for week in month_calendar:
        cells = []
        for day in week:
            if day == 0:
                cells.append('<td style="padding: 8px;"></td>')
                continue

            target_path = all_date_file_mapping.get((now_year, now_month, day))
            if target_path is None:
                cell_content = str(day)
            else:
                href = relative_site_href(target_path, current_page_path)
                cell_content = f'<a href="{href}" style="text-decoration: none;">{day}</a>'
            cells.append(f'<td style="padding: 8px;">{cell_content}</td>')

        lines.append("        <tr>" + "".join(cells) + "</tr>")

    lines.extend(["    </tbody>", "</table>"])
    return "\n".join(lines)


def render_monthly_md_with_hyperlink(
    now_date: Tuple[int, int, int],
    current_page_path: str,
    all_date_file_mapping: Dict[Tuple[int, int, int], str] = None,
    summary_page_path: str | None = None,
    previous_asset_path: str | None = None,
    center_asset_path: str | None = None,
    next_asset_path: str | None = None,
) -> str:
    all_date_file_mapping = all_date_file_mapping or {}

    now_year, now_month, _ = now_date
    current_month = (now_year, now_month)
    all_months = {month_from_date(date) for date in all_date_file_mapping.keys()}
    all_months.add(current_month)
    all_months_list = sorted(all_months)

    head_string = _render_month_header(
        current_page_path,
        current_month,
        all_months_list,
        previous_asset_path,
        center_asset_path,
        next_asset_path,
    )
    content_table = render_content_table(now_date, current_page_path, all_date_file_mapping)
    summary_link = ""
    if summary_page_path is not None:
        summary_href = relative_site_href(summary_page_path, current_page_path)
        summary_link = f'<div align="center"><a href="{summary_href}">Monthly Topic Summary</a></div>'

    parts = [
        head_string,
        f"# Personalized Monthly ArXiv Paper Summary {now_year}/{now_month:02d}",
    ]
    if summary_link:
        parts.append(summary_link)
    parts.append(content_table)

    return "\n\n".join(parts)


if __name__ == "__main__":
    pass
