from __future__ import annotations

HOTSPOT_ARCHIVE_START_DATE = "2026-03-17"
HOTSPOT_ARCHIVE_START_MONTH = HOTSPOT_ARCHIVE_START_DATE[:7]
HOTSPOT_ARCHIVE_START_YEAR = HOTSPOT_ARCHIVE_START_DATE[:4]
HOTSPOT_ARCHIVE_START_KEY = (2026, 3, 17)


def is_supported_hotspot_date(date_str: str | None) -> bool:
    return bool(date_str) and str(date_str) >= HOTSPOT_ARCHIVE_START_DATE


def is_supported_hotspot_month(month: str | None) -> bool:
    return bool(month) and str(month) >= HOTSPOT_ARCHIVE_START_MONTH


def is_supported_hotspot_year(year: str | None) -> bool:
    return bool(year) and str(year) >= HOTSPOT_ARCHIVE_START_YEAR


def is_supported_hotspot_date_key(date_key: tuple[int, int, int]) -> bool:
    return date_key >= HOTSPOT_ARCHIVE_START_KEY
