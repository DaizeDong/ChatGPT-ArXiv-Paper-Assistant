import calendar
from typing import Dict

from arxiv_assistant.renderers.site_paths import (
    relative_site_href,
    relative_site_path,
    site_month_page_path,
    site_year_page_path,
)

HEADER_SEPARATOR = "&nbsp;|&nbsp;"
MONTHS_PER_ROW = 4


def _render_link(href: str, text: str) -> str:
    return f'<a href="{href}">{text}</a>'


def _join_nav_parts(parts: list[str]) -> str:
    return HEADER_SEPARATOR.join(part for part in parts if part)


def _render_fixed_nav_row(left_html: str = "", center_html: str = "", right_html: str = "") -> str:
    return "".join(
        [
            '<div style="display: flex; align-items: flex-start; justify-content: space-between; width: 100%;">',
            f'<div style="width: 33.33%; text-align: left;">{left_html}</div>',
            f'<div style="width: 33.33%; text-align: center;">{center_html}</div>',
            f'<div style="width: 33.33%; text-align: right;">{right_html}</div>',
            "</div>",
        ]
    )


def _render_year_header(
    current_page_path: str,
    now_year: int,
    all_years_list: list[int],
    previous_asset_path: str | None,
    next_asset_path: str | None,
) -> str:
    now_idx = all_years_list.index(now_year)
    previous_year = all_years_list[now_idx - 1] if now_idx > 0 else None
    next_year = all_years_list[now_idx + 1] if now_idx + 1 < len(all_years_list) else None

    previous_href = relative_site_href(site_year_page_path(previous_year), current_page_path) if previous_year is not None else None
    next_href = relative_site_href(site_year_page_path(next_year), current_page_path) if next_year is not None else None

    if previous_href is None and next_href is None:
        return ""

    if previous_href and previous_asset_path or next_href and next_asset_path:
        previous_html = ""
        if previous_href and previous_asset_path and previous_year is not None:
            previous_asset_href = relative_site_path(previous_asset_path, current_page_path)
            previous_html = (
                f'<a href="{previous_href}"><img src="{previous_asset_href}" alt="Previous Year {previous_year}"></a>'
            )
        next_html = ""
        if next_href and next_asset_path and next_year is not None:
            next_asset_href = relative_site_path(next_asset_path, current_page_path)
            next_html = (
                f'<a href="{next_href}"><img src="{next_asset_href}" alt="Next Year {next_year}"></a>'
            )
        return _render_fixed_nav_row(previous_html, "", next_html)

    line_one = _join_nav_parts(
        [
            _render_link(previous_href, "&larr; Previous Year") if previous_href else "",
            _render_link(next_href, "Next Year &rarr;") if next_href else "",
        ]
    )
    line_two = _join_nav_parts(
        [
            _render_link(previous_href, str(previous_year)) if previous_year is not None else "",
            _render_link(next_href, str(next_year)) if next_year is not None else "",
        ]
    )
    return f'<div align="center">{line_one}<br>{line_two}</div>'


def render_content_table(
    now_year: int,
    current_page_path: str,
    all_month_file_mapping: Dict[tuple[int, int], str] = None,
    all_month_day_counts: Dict[tuple[int, int], int] = None,
) -> str:
    all_month_file_mapping = all_month_file_mapping or {}
    all_month_day_counts = all_month_day_counts or {}

    month_labels = list(calendar.month_name)[1:]
    lines = [
        '<table style="width: 100%; border-collapse: collapse; text-align: center;">',
        "    <thead>",
        f'        <tr><th colspan="{MONTHS_PER_ROW}" style="padding: 8px 0;">{now_year}</th></tr>',
        "    </thead>",
        "    <tbody>",
    ]

    for row_start in range(0, 12, MONTHS_PER_ROW):
        cells = []
        for month_num in range(row_start + 1, row_start + MONTHS_PER_ROW + 1):
            month_key = (now_year, month_num)
            target_path = all_month_file_mapping.get(month_key)
            month_label = month_labels[month_num - 1]
            day_count = all_month_day_counts.get(month_key, 0)

            if target_path is None:
                month_content = f"<strong>{month_label}</strong><br>{day_count} days"
            else:
                href = relative_site_href(target_path, current_page_path)
                month_content = f'<a href="{href}" style="text-decoration: none;"><strong>{month_label}</strong><br>{day_count} days</a>'

            cells.append(f'<td style="padding: 10px 6px;">{month_content}</td>')

        lines.append("        <tr>" + "".join(cells) + "</tr>")

    lines.extend(["    </tbody>", "</table>"])
    return "\n".join(lines)


def render_yearly_md_with_hyperlink(
    now_year: int,
    current_page_path: str,
    all_month_file_mapping: Dict[tuple[int, int], str] = None,
    all_month_day_counts: Dict[tuple[int, int], int] = None,
    related_page_path: str | None = None,
    related_label: str = "Daily AI Hotspots",
    previous_asset_path: str | None = None,
    next_asset_path: str | None = None,
) -> str:
    all_month_file_mapping = all_month_file_mapping or {}
    all_month_day_counts = all_month_day_counts or {}

    all_years = {year for year, _ in all_month_file_mapping.keys()}
    all_years.add(now_year)
    all_years_list = sorted(all_years)

    head_string = _render_year_header(
        current_page_path,
        now_year,
        all_years_list,
        previous_asset_path,
        next_asset_path,
    )
    content_table = render_content_table(
        now_year,
        current_page_path,
        all_month_file_mapping,
        all_month_day_counts,
    )

    parts = [f"# Personalized Yearly ArXiv Paper Summary {now_year}", content_table]
    if head_string:
        parts.insert(0, head_string)
    if related_page_path is not None:
        related_href = relative_site_href(related_page_path, current_page_path)
        parts.insert(1 if head_string else 0, f'<div align="center" class="site-jump-links"><a href="{related_href}">{related_label}</a></div>')
    return "\n\n".join(parts)


if __name__ == "__main__":
    pass
