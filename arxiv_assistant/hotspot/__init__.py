"""The hotspot feed.

The three pipeline entry points below are re-exported LAZILY, on purpose.
Importing them eagerly made every leaf of this package drag in the whole
subsystem: `from arxiv_assistant.hotspot.support.dates import ...` ran
pipeline.py, which builds `ZoneInfo("America/New_York")` at module scope.
On a Windows runner there is no system time zone database, so a script that
wanted one date helper died with ZoneInfoNotFoundError and took the published
site down with it. A package root should not decide what its callers pay for.
"""

from typing import TYPE_CHECKING

__all__ = [
    "detect_latest_local_output_date",
    "generate_daily_hotspot_report",
    "parse_target_datetime",
]

if TYPE_CHECKING:  # pragma: no cover - import shape for type checkers only
    from .pipeline import (
        detect_latest_local_output_date,
        generate_daily_hotspot_report,
        parse_target_datetime,
    )


def __getattr__(name: str):
    if name in __all__:
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__():
    return sorted(set(globals()) | set(__all__))
