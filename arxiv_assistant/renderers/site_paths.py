import posixpath
from typing import Tuple

DayTuple = Tuple[int, int, int]
MonthTuple = Tuple[int, int]

ROOT_SITE_PAGE = "index.md"


def month_from_date(date: DayTuple) -> MonthTuple:
    return date[0], date[1]


def site_month_page_path(month: MonthTuple) -> str:
    year, month_num = month
    return f"archive/{year:04d}-{month_num:02d}/index.md"


def site_day_page_path(date: DayTuple) -> str:
    year, month_num, day = date
    return f"archive/{year:04d}-{month_num:02d}/{day:02d}/index.md"


def relative_site_href(target_page_path: str, current_page_path: str) -> str:
    current_dir = posixpath.dirname(current_page_path) or "."
    relative_path = posixpath.relpath(target_page_path, current_dir)

    if relative_path == "index.md":
        return "."
    if relative_path.endswith("/index.md"):
        return relative_path[: -len("/index.md")]
    if relative_path.endswith(".md"):
        return relative_path[:-len(".md")]
    return relative_path
