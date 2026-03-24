from __future__ import annotations

import calendar
from typing import Any

from arxiv_assistant.renderers.site_paths import (
    relative_site_href,
    relative_site_path,
    site_hot_day_page_path,
    site_hot_month_page_path,
    site_hot_year_page_path,
)

HEADER_SEPARATOR = "&nbsp;|&nbsp;"


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


def _render_month_header(
    current_page_path: str,
    now_month: tuple[int, int],
    all_months_list: list[tuple[int, int]],
    previous_asset_path: str | None,
    center_asset_path: str | None,
    next_asset_path: str | None,
) -> str:
    now_idx = all_months_list.index(now_month)
    now_year, _ = now_month
    previous_month = all_months_list[now_idx - 1] if now_idx > 0 else None
    next_month = all_months_list[now_idx + 1] if now_idx + 1 < len(all_months_list) else None
    year_href = relative_site_href(site_hot_year_page_path(now_year), current_page_path)

    previous_href = (
        relative_site_href(site_hot_month_page_path(previous_month), current_page_path)
        if previous_month is not None
        else None
    )
    next_href = (
        relative_site_href(site_hot_month_page_path(next_month), current_page_path)
        if next_month is not None
        else None
    )

    if (previous_href and previous_asset_path) or center_asset_path or (next_href and next_asset_path):
        previous_html = ""
        if previous_href and previous_asset_path and previous_month is not None:
            previous_asset_href = relative_site_path(previous_asset_path, current_page_path)
            previous_html = (
                f'<a href="{previous_href}"><img src="{previous_asset_href}" '
                f'alt="Previous Hotspot Month {previous_month[0]}-{previous_month[1]:02d}"></a>'
            )
        center_html = ""
        if center_asset_path is not None:
            center_asset_href = relative_site_path(center_asset_path, current_page_path)
            center_html = (
                f'<a href="{year_href}"><img src="{center_asset_href}" alt="Yearly Hotspots {now_year}"></a>'
            )
        else:
            center_html = _render_link(year_href, f"Yearly Hotspots<br>{now_year}")
        next_html = ""
        if next_href and next_asset_path and next_month is not None:
            next_asset_href = relative_site_path(next_asset_path, current_page_path)
            next_html = (
                f'<a href="{next_href}"><img src="{next_asset_href}" '
                f'alt="Next Hotspot Month {next_month[0]}-{next_month[1]:02d}"></a>'
            )
        return _render_fixed_nav_row(previous_html, center_html, next_html)

    line_one = _join_nav_parts(
        [
            _render_link(previous_href, "&larr; Previous Month") if previous_href else "",
            _render_link(year_href, "Yearly Hotspots"),
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


def _render_stats_table(report_summaries: dict[tuple[int, int, int], dict[str, Any]]) -> str:
    total_days = len(report_summaries)
    total_topics = sum(int(row.get("top_topic_count", 0)) for row in report_summaries.values())
    total_watchlist = sum(int(row.get("watchlist_count", 0)) for row in report_summaries.values())
    return "\n".join(
        [
            '<table class="archive-summary-table">',
            "  <thead><tr><th>Metric</th><th>Value</th></tr></thead>",
            "  <tbody>",
            f"    <tr><td><strong>Report days</strong></td><td align=\"center\">{total_days}</td></tr>",
            f"    <tr><td><strong>Top topics</strong></td><td align=\"center\">{total_topics}</td></tr>",
            f"    <tr><td><strong>Watchlist items</strong></td><td align=\"center\">{total_watchlist}</td></tr>",
            "  </tbody>",
            "</table>",
        ]
    )


def _render_calendar(
    now_month: tuple[int, int],
    current_page_path: str,
    report_summaries: dict[tuple[int, int, int], dict[str, Any]],
) -> str:
    now_year, now_month_num = now_month
    month_calendar = calendar.Calendar(firstweekday=6).monthdayscalendar(now_year, now_month_num)
    headers = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    lines = [
        '<table style="margin: 0 auto; border-collapse: collapse; text-align: center;">',
        "    <thead>",
        '        <tr><th colspan="7" style="padding: 8px 0;">'
        f"{now_year}/{now_month_num:02d}"
        "</th></tr>",
        "        <tr>" + "".join(f'<th style="padding: 6px;">{header}</th>' for header in headers) + "</tr>",
        "    </thead>",
        "    <tbody>",
    ]
    for week in month_calendar:
        cells = []
        for day in week:
            if day == 0:
                cells.append('<td style="padding: 8px;"></td>')
                continue
            day_key = (now_year, now_month_num, day)
            report_summary = report_summaries.get(day_key)
            if report_summary is None:
                cell_content = str(day)
            else:
                href = relative_site_href(site_hot_day_page_path(day_key), current_page_path)
                topic_count = int(report_summary.get("top_topic_count", 0))
                cell_content = (
                    f'<a href="{href}" style="text-decoration: none;"><strong>{day}</strong><br>'
                    f'<span class="archive-day-meta">{topic_count} topics</span></a>'
                )
            cells.append(f'<td style="padding: 8px;">{cell_content}</td>')
        lines.append("        <tr>" + "".join(cells) + "</tr>")
    lines.extend(["    </tbody>", "</table>"])
    return "\n".join(lines)


def _render_daily_briefs(
    current_page_path: str,
    report_summaries: dict[tuple[int, int, int], dict[str, Any]],
) -> str:
    if not report_summaries:
        return ""
    lines = ["## Daily Briefs", ""]
    for date in sorted(report_summaries.keys(), reverse=True):
        summary_row = report_summaries[date]
        href = relative_site_href(site_hot_day_page_path(date), current_page_path)
        summary = str(summary_row.get("summary", "")).strip() or "No summary available."
        topic_count = int(summary_row.get("top_topic_count", 0))
        lines.append(f'- [{date[0]}-{date[1]:02d}-{date[2]:02d}]({href}) · {topic_count} top topics · {summary}')
    return "\n".join(lines)


def render_hot_monthly_md_with_hyperlink(
    now_month: tuple[int, int],
    current_page_path: str,
    all_months_list: list[tuple[int, int]],
    report_summaries: dict[tuple[int, int, int], dict[str, Any]],
    previous_asset_path: str | None = None,
    center_asset_path: str | None = None,
    next_asset_path: str | None = None,
    related_page_path: str | None = None,
    related_label: str = "Personalized Daily Arxiv Paper",
) -> str:
    year, month_num = now_month
    parts = [
        _render_month_header(
            current_page_path=current_page_path,
            now_month=now_month,
            all_months_list=all_months_list,
            previous_asset_path=previous_asset_path,
            center_asset_path=center_asset_path,
            next_asset_path=next_asset_path,
        )
    ]
    if related_page_path is not None:
        related_href = relative_site_href(related_page_path, current_page_path)
        parts.append(f'<div align="center" class="site-jump-links"><a href="{related_href}">{related_label}</a></div>')
    parts.extend(
        [
            f"# Daily AI Hotspots Archive {year}/{month_num:02d}",
            _render_stats_table(report_summaries),
            _render_calendar(now_month, current_page_path, report_summaries),
            _render_daily_briefs(current_page_path, report_summaries),
        ]
    )
    return "\n\n".join(parts).strip()
