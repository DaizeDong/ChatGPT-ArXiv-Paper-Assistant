import re
import shutil
from pathlib import Path
from typing import Dict, Tuple

from arxiv_assistant.renderers.render_daily_with_link import render_daily_md_with_hyperlink
from arxiv_assistant.renderers.render_monthly_with_link import render_monthly_md_with_hyperlink
from arxiv_assistant.renderers.site_paths import ROOT_SITE_PAGE, month_from_date, site_day_page_path, site_month_page_path

DAY_FILE_PATTERN = re.compile(r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})-(?P<suffix>[^/\\\\]+)\.md$")
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


def _write_text(target_path: Path, content: str) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(content, encoding="utf-8")


def build_multipage_site(output_root: str | Path) -> Path | None:
    output_root = Path(output_root)
    daily_md_root = output_root / "md"
    site_root = output_root / "site"
    day_sources = discover_daily_markdown(daily_md_root)

    if not day_sources:
        return None

    if site_root.exists():
        shutil.rmtree(site_root)
    site_root.mkdir(parents=True, exist_ok=True)

    all_dates = set(day_sources.keys())
    day_page_mapping = {date: site_day_page_path(date) for date in all_dates}

    for date, source_path in sorted(day_sources.items()):
        content = source_path.read_text(encoding="utf-8")
        target_rel_path = site_day_page_path(date)
        rendered = render_daily_md_with_hyperlink(
            now_date=date,
            current_page_path=target_rel_path,
            all_dates=all_dates,
            content_string=content,
        )
        _write_text(site_root / Path(target_rel_path), rendered)

    latest_date = max(all_dates)
    latest_content = day_sources[latest_date].read_text(encoding="utf-8")
    latest_root_page = render_daily_md_with_hyperlink(
        now_date=latest_date,
        current_page_path=ROOT_SITE_PAGE,
        all_dates=all_dates,
        content_string=latest_content,
    )
    _write_text(site_root / ROOT_SITE_PAGE, latest_root_page)

    for month in sorted({month_from_date(date) for date in all_dates}):
        year, month_num = month
        rendered = render_monthly_md_with_hyperlink(
            now_date=(year, month_num, 1),
            current_page_path=site_month_page_path(month),
            all_date_file_mapping=day_page_mapping,
        )
        _write_text(site_root / Path(site_month_page_path(month)), rendered)

    return site_root


if __name__ == "__main__":
    site_path = build_multipage_site("out")
    print(site_path if site_path is not None else "No daily markdown files found.")
