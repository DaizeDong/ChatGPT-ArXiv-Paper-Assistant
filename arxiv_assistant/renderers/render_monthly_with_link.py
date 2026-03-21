import calendar
from typing import Dict, Tuple

from arxiv_assistant.renderers.site_paths import (
    month_from_date,
    relative_site_href,
    site_month_page_path,
)


def _render_month_nav_cell(
    href: str | None,
    title: str,
    subtitle: str,
    arrow_html: str,
    align: str,
) -> str:
    if href is None:
        return '<td style="border: none; padding: 0px; width: 50%;"></td>'

    if align == "left":
        content = (
            '<div style="display: flex; align-items: center; gap: 5px;">'
            f'<a href="{href}" style="text-decoration: none; color: inherit;">'
            f'<strong style="color: black; font-size: 18px;">{arrow_html}</strong>'
            "</a>"
            '<div style="text-align: center;">'
            f'<a href="{href}" style="text-decoration: none;"><strong>{title}</strong></a>'
            "<br>"
            f'<a href="{href}" style="text-decoration: none; font-size: 14px; color: gray;">{subtitle}</a>'
            "</div>"
            "</div>"
        )
    else:
        content = (
            '<div style="display: flex; align-items: center; gap: 5px; justify-content: flex-end; width: 100%;">'
            '<div style="text-align: center;">'
            f'<a href="{href}" style="text-decoration: none;"><strong>{title}</strong></a>'
            "<br>"
            f'<a href="{href}" style="text-decoration: none; font-size: 14px; color: gray;">{subtitle}</a>'
            "</div>"
            f'<a href="{href}" style="text-decoration: none; color: inherit;">'
            f'<strong style="color: black; font-size: 18px;">{arrow_html}</strong>'
            "</a>"
            "</div>"
        )

    return f'<td style="border: none; padding: 0px; width: 50%;">{content}</td>'


def _render_month_header(
    current_page_path: str,
    now_month: Tuple[int, int],
    all_months_list: list[Tuple[int, int]],
) -> str:
    now_idx = all_months_list.index(now_month)
    previous_month = all_months_list[now_idx - 1] if now_idx > 0 else None
    next_month = all_months_list[now_idx + 1] if now_idx + 1 < len(all_months_list) else None

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

    return "\n".join(
        [
            '<table style="width: 100%; border-collapse: collapse; border: none;">',
            "    <tr>",
            _render_month_nav_cell(
                previous_href,
                "Previous Month",
                f"{previous_month[0]}-{previous_month[1]:02d}" if previous_month else "",
                "&larr;",
                "left",
            ),
            _render_month_nav_cell(
                next_href,
                "Next Month",
                f"{next_month[0]}-{next_month[1]:02d}" if next_month else "",
                "&rarr;",
                "right",
            ),
            "    </tr>",
            "</table>",
        ]
    )


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
        '<table style="width: 100%; border-collapse: collapse; text-align: center;">',
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
) -> str:
    all_date_file_mapping = all_date_file_mapping or {}

    now_year, now_month, _ = now_date
    current_month = (now_year, now_month)
    all_months = {month_from_date(date) for date in all_date_file_mapping.keys()}
    all_months.add(current_month)
    all_months_list = sorted(all_months)

    head_string = _render_month_header(current_page_path, current_month, all_months_list)
    content_table = render_content_table(now_date, current_page_path, all_date_file_mapping)

    return "\n\n".join(
        [
            head_string,
            f"# Personalized Monthly ArXiv Paper Summary {now_year}/{now_month:02d}",
            content_table,
        ]
    )


if __name__ == "__main__":
    pass
