import posixpath
from typing import Tuple

DayTuple = Tuple[int, int, int]
MonthTuple = Tuple[int, int]
YearTuple = int

ROOT_SITE_PAGE = "index.md"


def month_from_date(date: DayTuple) -> MonthTuple:
    return date[0], date[1]


def year_from_date(date: DayTuple) -> YearTuple:
    return date[0]


def site_year_page_path(year: YearTuple) -> str:
    return f"archive/{year:04d}/index.md"


def site_month_page_path(month: MonthTuple) -> str:
    year, month_num = month
    return f"archive/{year:04d}-{month_num:02d}/index.md"


def site_day_page_path(date: DayTuple) -> str:
    year, month_num, day = date
    return f"archive/{year:04d}-{month_num:02d}/{day:02d}/index.md"


def site_day_nav_asset_path(date: DayTuple, direction: str) -> str:
    year, month_num, day = date
    return f"assets/nav/day/{year:04d}-{month_num:02d}-{day:02d}-{direction}.svg"


def site_month_nav_asset_path(month: MonthTuple, direction: str) -> str:
    year, month_num = month
    return f"assets/nav/month/{year:04d}-{month_num:02d}-{direction}.svg"


def site_year_nav_asset_path(year: YearTuple, direction: str) -> str:
    return f"assets/nav/year/{year:04d}-{direction}.svg"


def relative_site_path(target_path: str, current_page_path: str) -> str:
    current_dir = posixpath.dirname(current_page_path) or "."
    return posixpath.relpath(target_path, current_dir)


def relative_site_href(target_page_path: str, current_page_path: str) -> str:
    relative_path = relative_site_path(target_page_path, current_page_path)

    if relative_path == "index.md":
        return "."
    if relative_path.endswith("/index.md"):
        return relative_path[: -len("/index.md")]
    if relative_path.endswith(".md"):
        return relative_path[:-len(".md")]
    return relative_path
