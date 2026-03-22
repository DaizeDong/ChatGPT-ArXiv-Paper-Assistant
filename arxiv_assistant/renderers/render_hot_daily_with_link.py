from __future__ import annotations

from typing import Any

from arxiv_assistant.renderers.render_hot_daily import render_hot_daily_md
from arxiv_assistant.renderers.site_paths import (
    month_from_date,
    relative_site_href,
    relative_site_path,
    site_hot_day_page_path,
    site_hot_month_page_path,
)

FALLBACK_SEPARATOR = "&nbsp;|&nbsp;"


def _render_text_link(href: str, text: str) -> str:
    return f'<a href="{href}">{text}</a>'


def _join_fallback_parts(parts: list[str]) -> str:
    return FALLBACK_SEPARATOR.join(part for part in parts if part)


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


def _render_hot_day_header(
    current_page_path: str,
    now_date: tuple[int, int, int],
    all_dates_list: list[tuple[int, int, int]],
    previous_asset_path: str | None,
    center_asset_path: str | None,
    next_asset_path: str | None,
) -> str:
    now_year, now_month, _ = now_date
    now_idx = all_dates_list.index(now_date)
    previous_date = all_dates_list[now_idx - 1] if now_idx > 0 else None
    next_date = all_dates_list[now_idx + 1] if now_idx + 1 < len(all_dates_list) else None
    month_href = relative_site_href(site_hot_month_page_path(month_from_date(now_date)), current_page_path)

    previous_href = (
        relative_site_href(site_hot_day_page_path(previous_date), current_page_path)
        if previous_date is not None
        else None
    )
    next_href = (
        relative_site_href(site_hot_day_page_path(next_date), current_page_path)
        if next_date is not None
        else None
    )

    if (previous_href and previous_asset_path) or center_asset_path or (next_href and next_asset_path):
        previous_html = ""
        if previous_href and previous_asset_path and previous_date is not None:
            previous_asset_href = relative_site_path(previous_asset_path, current_page_path)
            previous_html = (
                f'<a href="{previous_href}"><img src="{previous_asset_href}" '
                f'alt="Previous Hotspot Day {previous_date[0]}-{previous_date[1]:02d}-{previous_date[2]:02d}"></a>'
            )
        center_html = ""
        if center_asset_path is not None:
            center_asset_href = relative_site_path(center_asset_path, current_page_path)
            center_html = (
                f'<a href="{month_href}"><img src="{center_asset_href}" alt="Monthly Hotspots {now_year}-{now_month:02d}"></a>'
            )
        else:
            center_html = _render_text_link(month_href, f"Monthly Hotspots<br>{now_year}-{now_month:02d}")
        next_html = ""
        if next_href and next_asset_path and next_date is not None:
            next_asset_href = relative_site_path(next_asset_path, current_page_path)
            next_html = (
                f'<a href="{next_href}"><img src="{next_asset_href}" '
                f'alt="Next Hotspot Day {next_date[0]}-{next_date[1]:02d}-{next_date[2]:02d}"></a>'
            )
        return _render_fixed_nav_row(previous_html, center_html, next_html)

    line_one = _join_fallback_parts(
        [
            _render_text_link(previous_href, "&larr; Previous Hotspots") if previous_href else "",
            _render_text_link(month_href, "Monthly Hotspots"),
            _render_text_link(next_href, "Next Hotspots &rarr;") if next_href else "",
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


def render_hot_daily_md_with_hyperlink(
    now_date: tuple[int, int, int],
    current_page_path: str,
    all_dates: set[tuple[int, int, int]] | None = None,
    previous_asset_path: str | None = None,
    center_asset_path: str | None = None,
    next_asset_path: str | None = None,
    related_page_path: str | None = None,
    related_label: str = "Daily Paper Digest",
    content_string: str | None = None,
    report: dict[str, Any] | None = None,
) -> str:
    all_dates_set = set(all_dates or set())
    all_dates_set.add(now_date)
    all_dates_list = sorted(all_dates_set)

    if content_string is None:
        if report is None:
            raise ValueError("Either content_string or report must be provided.")
        content_string = render_hot_daily_md(report)

    parts = [
        _render_hot_day_header(
            current_page_path=current_page_path,
            now_date=now_date,
            all_dates_list=all_dates_list,
            previous_asset_path=previous_asset_path,
            center_asset_path=center_asset_path,
            next_asset_path=next_asset_path,
        )
    ]

    if related_page_path is not None:
        related_href = relative_site_href(related_page_path, current_page_path)
        parts.append(f'<div align="center" class="site-jump-links"><a href="{related_href}">{related_label}</a></div>')

    parts.append(content_string)
    return "\n\n".join(parts)
