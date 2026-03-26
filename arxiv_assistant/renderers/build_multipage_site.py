import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

from arxiv_assistant.renderers.nav_assets import write_nav_button_svg
from arxiv_assistant.renderers.hotspot.render_hot_daily_with_link import render_hot_daily_md_with_hyperlink
from arxiv_assistant.renderers.hotspot.render_hot_monthly_with_link import render_hot_monthly_md_with_hyperlink
from arxiv_assistant.renderers.hotspot.render_hot_yearly_with_link import render_hot_yearly_md_with_hyperlink
from arxiv_assistant.renderers.paper.monthly_summary import build_monthly_summary_data
from arxiv_assistant.renderers.paper.render_daily_with_link import render_daily_md_with_hyperlink
from arxiv_assistant.renderers.paper.render_monthly_summary_with_link import render_monthly_summary_md_with_hyperlink
from arxiv_assistant.renderers.paper.render_monthly_with_link import render_monthly_md_with_hyperlink
from arxiv_assistant.renderers.paper.render_yearly_with_link import render_yearly_md_with_hyperlink
from arxiv_assistant.renderers.site_paths import (
    HOT_ROOT_SITE_PAGE,
    ROOT_SITE_PAGE,
    month_from_date,
    site_hot_day_nav_asset_path,
    site_hot_day_page_path,
    site_hot_month_nav_asset_path,
    site_hot_month_page_path,
    site_hot_year_nav_asset_path,
    site_hot_year_page_path,
    site_year_nav_asset_path,
    site_year_page_path,
    site_day_nav_asset_path,
    site_day_page_path,
    site_month_nav_asset_path,
    site_month_page_path,
    site_month_summary_page_path,
    year_from_date,
)
from arxiv_assistant.utils.hotspot.hotspot_dates import is_supported_hotspot_date_key

DAY_FILE_PATTERN = re.compile(r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})-(?P<suffix>[^/\\\\]+)\.md$")
HOT_DAY_FILE_PATTERN = re.compile(r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})-hotspots\.md$")
HOT_REPORT_PATTERN = re.compile(r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})\.json$")
SUFFIX_PRIORITY = {"latest": 0, "output": 1}


def _candidate_priority(path: Path) -> Tuple[int, str]:
    match = DAY_FILE_PATTERN.match(path.name)
    suffix = match.group("suffix") if match else path.stem
    return SUFFIX_PRIORITY.get(suffix, 99), path.name


def discover_daily_markdown(md_root: Path) -> Dict[Tuple[int, int, int], Path]:
    candidates: Dict[Tuple[int, int, int], list[Path]] = {}
    if not md_root.exists():
        return {}

    for file_path in sorted(md_root.glob("*/*.md")):
        if file_path.name == "index.md":
            continue
        match = DAY_FILE_PATTERN.match(file_path.name)
        if match is None:
            continue
        date = (
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
        )
        candidates.setdefault(date, []).append(file_path)

    resolved: Dict[Tuple[int, int, int], Path] = {}
    for date, file_paths in candidates.items():
        resolved[date] = sorted(file_paths, key=_candidate_priority)[0]
    return resolved


def discover_hotspot_markdown(md_root: Path) -> Dict[Tuple[int, int, int], Path]:
    resolved: Dict[Tuple[int, int, int], Path] = {}
    if not md_root.exists():
        return resolved

    for file_path in sorted(md_root.glob("*/*.md")):
        match = HOT_DAY_FILE_PATTERN.match(file_path.name)
        if match is None:
            continue
        date = (
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
        )
        if not is_supported_hotspot_date_key(date):
            continue
        resolved[date] = file_path
    return resolved


def discover_hotspot_report_summaries(report_root: Path) -> Dict[Tuple[int, int, int], Dict[str, Any]]:
    summaries: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    if not report_root.exists():
        return summaries

    for file_path in sorted(report_root.glob("*.json")):
        match = HOT_REPORT_PATTERN.match(file_path.name)
        if match is None:
            continue
        date = (
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
        )
        if not is_supported_hotspot_date_key(date):
            continue
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        summaries[date] = {
            "summary": payload.get("summary", ""),
            "top_topic_count": len(payload.get("top_topics", [])),
            "watchlist_count": len(payload.get("watchlist", [])),
            "source_stats": payload.get("source_stats", {}),
        }
    return summaries


def _write_text(target_path: Path, content: str) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(content, encoding="utf-8")


def _write_day_nav_asset(site_root: Path, current_date: Tuple[int, int, int], target_date: Tuple[int, int, int], direction: str) -> str:
    asset_path = site_day_nav_asset_path(current_date, direction)
    title = "Previous Day" if direction == "prev" else "Next Day"
    subtitle = f"{target_date[0]}-{target_date[1]:02d}-{target_date[2]:02d}"
    arrow = "\u2190" if direction == "prev" else "\u2192"
    arrow_side = "left" if direction == "prev" else "right"
    write_nav_button_svg(site_root / Path(asset_path), title, subtitle, arrow=arrow, arrow_side=arrow_side)
    return asset_path


def _write_day_center_asset(site_root: Path, current_date: Tuple[int, int, int]) -> str:
    asset_path = site_day_nav_asset_path(current_date, "center")
    write_nav_button_svg(
        site_root / Path(asset_path),
        "Monthly Overview",
        f"{current_date[0]}-{current_date[1]:02d}",
    )
    return asset_path


def _write_hot_day_nav_asset(site_root: Path, current_date: Tuple[int, int, int], target_date: Tuple[int, int, int], direction: str) -> str:
    asset_path = site_hot_day_nav_asset_path(current_date, direction)
    title = "Previous Hotspots" if direction == "prev" else "Next Hotspots"
    subtitle = f"{target_date[0]}-{target_date[1]:02d}-{target_date[2]:02d}"
    arrow = "\u2190" if direction == "prev" else "\u2192"
    arrow_side = "left" if direction == "prev" else "right"
    write_nav_button_svg(site_root / Path(asset_path), title, subtitle, arrow=arrow, arrow_side=arrow_side)
    return asset_path


def _write_hot_day_center_asset(site_root: Path, current_date: Tuple[int, int, int]) -> str:
    asset_path = site_hot_day_nav_asset_path(current_date, "center")
    write_nav_button_svg(
        site_root / Path(asset_path),
        "Monthly Hotspots",
        f"{current_date[0]}-{current_date[1]:02d}",
    )
    return asset_path


def _write_month_nav_asset(site_root: Path, current_month: Tuple[int, int], target_month: Tuple[int, int], direction: str) -> str:
    asset_path = site_month_nav_asset_path(current_month, direction)
    title = "Previous Month" if direction == "prev" else "Next Month"
    subtitle = f"{target_month[0]}-{target_month[1]:02d}"
    arrow = "\u2190" if direction == "prev" else "\u2192"
    arrow_side = "left" if direction == "prev" else "right"
    write_nav_button_svg(site_root / Path(asset_path), title, subtitle, arrow=arrow, arrow_side=arrow_side)
    return asset_path


def _write_month_center_asset(site_root: Path, current_month: Tuple[int, int]) -> str:
    asset_path = site_month_nav_asset_path(current_month, "center")
    write_nav_button_svg(
        site_root / Path(asset_path),
        "Yearly Overview",
        str(current_month[0]),
    )
    return asset_path


def _write_hot_month_nav_asset(site_root: Path, current_month: Tuple[int, int], target_month: Tuple[int, int], direction: str) -> str:
    asset_path = site_hot_month_nav_asset_path(current_month, direction)
    title = "Previous Month" if direction == "prev" else "Next Month"
    subtitle = f"{target_month[0]}-{target_month[1]:02d}"
    arrow = "\u2190" if direction == "prev" else "\u2192"
    arrow_side = "left" if direction == "prev" else "right"
    write_nav_button_svg(site_root / Path(asset_path), title, subtitle, arrow=arrow, arrow_side=arrow_side)
    return asset_path


def _write_hot_month_center_asset(site_root: Path, current_month: Tuple[int, int]) -> str:
    asset_path = site_hot_month_nav_asset_path(current_month, "center")
    write_nav_button_svg(
        site_root / Path(asset_path),
        "Yearly Hotspots",
        str(current_month[0]),
    )
    return asset_path


def _write_year_nav_asset(site_root: Path, current_year: int, target_year: int, direction: str) -> str:
    asset_path = site_year_nav_asset_path(current_year, direction)
    title = "Previous Year" if direction == "prev" else "Next Year"
    subtitle = str(target_year)
    arrow = "\u2190" if direction == "prev" else "\u2192"
    arrow_side = "left" if direction == "prev" else "right"
    write_nav_button_svg(site_root / Path(asset_path), title, subtitle, arrow=arrow, arrow_side=arrow_side)
    return asset_path


def _write_hot_year_nav_asset(site_root: Path, current_year: int, target_year: int, direction: str) -> str:
    asset_path = site_hot_year_nav_asset_path(current_year, direction)
    title = "Previous Year" if direction == "prev" else "Next Year"
    subtitle = str(target_year)
    arrow = "\u2190" if direction == "prev" else "\u2192"
    arrow_side = "left" if direction == "prev" else "right"
    write_nav_button_svg(site_root / Path(asset_path), title, subtitle, arrow=arrow, arrow_side=arrow_side)
    return asset_path


def build_multipage_site(output_root: str | Path) -> Path | None:
    output_root = Path(output_root)
    daily_md_root = output_root / "md"
    daily_json_root = output_root / "json"
    monthly_summary_root = output_root / "monthly"
    hot_md_root = output_root / "hot" / "md"
    hot_report_root = output_root / "hot" / "reports"
    site_root = output_root / "site"
    day_sources = discover_daily_markdown(daily_md_root)
    hot_day_sources = discover_hotspot_markdown(hot_md_root)
    hot_report_summaries = discover_hotspot_report_summaries(hot_report_root)

    if not day_sources:
        return None

    if site_root.exists():
        shutil.rmtree(site_root)
    site_root.mkdir(parents=True, exist_ok=True)

    all_dates = set(day_sources.keys())
    all_dates_list = sorted(all_dates)
    day_page_mapping = {date: site_day_page_path(date) for date in all_dates}
    month_page_mapping = {month_from_date(date): site_month_page_path(month_from_date(date)) for date in all_dates}
    month_summary_page_mapping = {month_from_date(date): site_month_summary_page_path(month_from_date(date)) for date in all_dates}
    month_day_counts: Dict[Tuple[int, int], int] = {}
    for date in all_dates:
        month_day_counts[month_from_date(date)] = month_day_counts.get(month_from_date(date), 0) + 1
    monthly_summary_data = build_monthly_summary_data(daily_json_root, monthly_summary_root, day_page_mapping)
    hot_dates = sorted(hot_day_sources.keys())
    hot_dates_set = set(hot_dates)
    hot_months_set = {month_from_date(date) for date in hot_dates}
    hot_years_set = {year_from_date(date) for date in hot_dates}
    latest_hot_date = max(hot_dates_set) if hot_dates_set else None
    latest_hot_month = month_from_date(latest_hot_date) if latest_hot_date is not None else None
    latest_hot_year = year_from_date(latest_hot_date) if latest_hot_date is not None else None

    def best_hotspot_route_for_day(date: Tuple[int, int, int]) -> str | None:
        if date in hot_dates_set:
            return site_hot_day_page_path(date)
        month = month_from_date(date)
        if month in hot_months_set:
            return site_hot_month_page_path(month)
        year = year_from_date(date)
        if year in hot_years_set:
            return site_hot_year_page_path(year)
        if latest_hot_date is not None:
            return site_hot_day_page_path(latest_hot_date)
        return None

    def best_hotspot_route_for_month(month: Tuple[int, int]) -> str | None:
        if month in hot_months_set:
            return site_hot_month_page_path(month)
        year = month[0]
        if year in hot_years_set:
            return site_hot_year_page_path(year)
        if latest_hot_month is not None:
            return site_hot_month_page_path(latest_hot_month)
        return None

    def best_hotspot_route_for_year(year: int) -> str | None:
        if year in hot_years_set:
            return site_hot_year_page_path(year)
        if latest_hot_year is not None:
            return site_hot_year_page_path(latest_hot_year)
        return None

    for date_idx, date in enumerate(all_dates_list):
        source_path = day_sources[date]
        content = source_path.read_text(encoding="utf-8")
        target_rel_path = site_day_page_path(date)
        previous_date = all_dates_list[date_idx - 1] if date_idx > 0 else None
        next_date = all_dates_list[date_idx + 1] if date_idx + 1 < len(all_dates_list) else None
        previous_asset_path = (
            _write_day_nav_asset(site_root, date, previous_date, "prev") if previous_date is not None else None
        )
        next_asset_path = _write_day_nav_asset(site_root, date, next_date, "next") if next_date is not None else None
        rendered = render_daily_md_with_hyperlink(
            now_date=date,
            current_page_path=target_rel_path,
            all_dates=all_dates,
            previous_asset_path=previous_asset_path,
            center_asset_path=_write_day_center_asset(site_root, date),
            next_asset_path=next_asset_path,
            related_page_path=best_hotspot_route_for_day(date),
            content_string=content,
        )
        _write_text(site_root / Path(target_rel_path), rendered)

    latest_date = max(all_dates)
    latest_content = day_sources[latest_date].read_text(encoding="utf-8")
    latest_idx = all_dates_list.index(latest_date)
    latest_previous = all_dates_list[latest_idx - 1] if latest_idx > 0 else None
    latest_next = all_dates_list[latest_idx + 1] if latest_idx + 1 < len(all_dates_list) else None
    latest_root_page = render_daily_md_with_hyperlink(
        now_date=latest_date,
        current_page_path=ROOT_SITE_PAGE,
        all_dates=all_dates,
        previous_asset_path=(
            _write_day_nav_asset(site_root, latest_date, latest_previous, "prev") if latest_previous is not None else None
        ),
        center_asset_path=_write_day_center_asset(site_root, latest_date),
        next_asset_path=(
            _write_day_nav_asset(site_root, latest_date, latest_next, "next") if latest_next is not None else None
        ),
        related_page_path=best_hotspot_route_for_day(latest_date),
        content_string=latest_content,
    )
    _write_text(site_root / ROOT_SITE_PAGE, latest_root_page)

    all_months_list = sorted({month_from_date(date) for date in all_dates})
    for month_idx, month in enumerate(all_months_list):
        year, month_num = month
        previous_month = all_months_list[month_idx - 1] if month_idx > 0 else None
        next_month = all_months_list[month_idx + 1] if month_idx + 1 < len(all_months_list) else None
        rendered = render_monthly_md_with_hyperlink(
            now_date=(year, month_num, 1),
            current_page_path=site_month_page_path(month),
            all_date_file_mapping=day_page_mapping,
            summary_page_path=month_summary_page_mapping.get(month) if month in monthly_summary_data else None,
            related_page_path=best_hotspot_route_for_month(month),
            previous_asset_path=(
                _write_month_nav_asset(site_root, month, previous_month, "prev") if previous_month is not None else None
            ),
            center_asset_path=_write_month_center_asset(site_root, month),
            next_asset_path=(
                _write_month_nav_asset(site_root, month, next_month, "next") if next_month is not None else None
            ),
        )
        _write_text(site_root / Path(site_month_page_path(month)), rendered)

    all_summary_months_list = sorted(monthly_summary_data.keys())
    for month in all_summary_months_list:
        rendered = render_monthly_summary_md_with_hyperlink(
            now_month=month,
            current_page_path=site_month_summary_page_path(month),
            category_buckets=monthly_summary_data[month],
            all_summary_months_list=all_summary_months_list,
        )
        _write_text(site_root / Path(site_month_summary_page_path(month)), rendered)

    all_years_list = sorted({year_from_date(date) for date in all_dates})
    for year_idx, year in enumerate(all_years_list):
        previous_year = all_years_list[year_idx - 1] if year_idx > 0 else None
        next_year = all_years_list[year_idx + 1] if year_idx + 1 < len(all_years_list) else None
        rendered = render_yearly_md_with_hyperlink(
            now_year=year,
            current_page_path=site_year_page_path(year),
            all_month_file_mapping=month_page_mapping,
            all_month_day_counts=month_day_counts,
            related_page_path=best_hotspot_route_for_year(year),
            previous_asset_path=(
                _write_year_nav_asset(site_root, year, previous_year, "prev") if previous_year is not None else None
            ),
            next_asset_path=(
                _write_year_nav_asset(site_root, year, next_year, "next") if next_year is not None else None
            ),
        )
        _write_text(site_root / Path(site_year_page_path(year)), rendered)

    if hot_day_sources:
        for hot_idx, date in enumerate(hot_dates):
            content = hot_day_sources[date].read_text(encoding="utf-8")
            target_rel_path = site_hot_day_page_path(date)
            previous_date = hot_dates[hot_idx - 1] if hot_idx > 0 else None
            next_date = hot_dates[hot_idx + 1] if hot_idx + 1 < len(hot_dates) else None
            rendered = render_hot_daily_md_with_hyperlink(
                now_date=date,
                current_page_path=target_rel_path,
                all_dates=hot_dates_set,
                previous_asset_path=(
                    _write_hot_day_nav_asset(site_root, date, previous_date, "prev") if previous_date is not None else None
                ),
                center_asset_path=_write_hot_day_center_asset(site_root, date),
                next_asset_path=(
                    _write_hot_day_nav_asset(site_root, date, next_date, "next") if next_date is not None else None
                ),
                related_page_path=site_day_page_path(date) if date in day_page_mapping else ROOT_SITE_PAGE,
                related_label="Personalized Daily Arxiv Paper",
                content_string=content,
            )
            _write_text(site_root / Path(target_rel_path), rendered)

        latest_hot_content = hot_day_sources[latest_hot_date].read_text(encoding="utf-8")
        latest_hot_idx = hot_dates.index(latest_hot_date)
        latest_hot_previous = hot_dates[latest_hot_idx - 1] if latest_hot_idx > 0 else None
        latest_hot_next = hot_dates[latest_hot_idx + 1] if latest_hot_idx + 1 < len(hot_dates) else None
        hot_root_page = render_hot_daily_md_with_hyperlink(
            now_date=latest_hot_date,
            current_page_path=HOT_ROOT_SITE_PAGE,
            all_dates=hot_dates_set,
            previous_asset_path=(
                _write_hot_day_nav_asset(site_root, latest_hot_date, latest_hot_previous, "prev")
                if latest_hot_previous is not None
                else None
            ),
            center_asset_path=_write_hot_day_center_asset(site_root, latest_hot_date),
            next_asset_path=(
                _write_hot_day_nav_asset(site_root, latest_hot_date, latest_hot_next, "next")
                if latest_hot_next is not None
                else None
            ),
            related_page_path=site_day_page_path(latest_hot_date) if latest_hot_date in day_page_mapping else ROOT_SITE_PAGE,
            related_label="Personalized Daily Arxiv Paper",
            content_string=latest_hot_content,
        )
        _write_text(site_root / HOT_ROOT_SITE_PAGE, hot_root_page)

        hot_months = sorted({month_from_date(date) for date in hot_dates})
        for month_idx, month in enumerate(hot_months):
            month_reports = {
                date: hot_report_summaries.get(
                    date,
                    {"summary": "", "top_topic_count": 0, "watchlist_count": 0, "source_stats": {}},
                )
                for date in hot_dates
                if month_from_date(date) == month
            }
            rendered = render_hot_monthly_md_with_hyperlink(
                now_month=month,
                current_page_path=site_hot_month_page_path(month),
                all_months_list=hot_months,
                report_summaries=month_reports,
                previous_asset_path=(
                    _write_hot_month_nav_asset(site_root, month, hot_months[month_idx - 1], "prev")
                    if month_idx > 0
                    else None
                ),
                center_asset_path=_write_hot_month_center_asset(site_root, month),
                next_asset_path=(
                    _write_hot_month_nav_asset(site_root, month, hot_months[month_idx + 1], "next")
                    if month_idx + 1 < len(hot_months)
                    else None
                ),
                related_page_path=site_month_page_path(month) if month in month_page_mapping else site_year_page_path(month[0]) if month[0] in all_years_list else ROOT_SITE_PAGE,
            )
            _write_text(site_root / Path(site_hot_month_page_path(month)), rendered)

        hot_years = sorted({year_from_date(date) for date in hot_dates})
        for year_idx, year in enumerate(hot_years):
            month_stats: Dict[Tuple[int, int], Dict[str, Any]] = {}
            for month in hot_months:
                if month[0] != year:
                    continue
                month_dates = [date for date in hot_dates if month_from_date(date) == month]
                month_stats[month] = {
                    "day_count": len(month_dates),
                    "top_topic_count": sum(
                        int(hot_report_summaries.get(date, {}).get("top_topic_count", 0)) for date in month_dates
                    ),
                }
            rendered = render_hot_yearly_md_with_hyperlink(
                now_year=year,
                current_page_path=site_hot_year_page_path(year),
                all_years_list=hot_years,
                month_stats=month_stats,
                previous_asset_path=(
                    _write_hot_year_nav_asset(site_root, year, hot_years[year_idx - 1], "prev") if year_idx > 0 else None
                ),
                next_asset_path=(
                    _write_hot_year_nav_asset(site_root, year, hot_years[year_idx + 1], "next")
                    if year_idx + 1 < len(hot_years)
                    else None
                ),
                related_page_path=site_year_page_path(year) if year in all_years_list else ROOT_SITE_PAGE,
            )
            _write_text(site_root / Path(site_hot_year_page_path(year)), rendered)

    return site_root


if __name__ == "__main__":
    site_path = build_multipage_site("out")
    print(site_path if site_path is not None else "No daily markdown files found.")
