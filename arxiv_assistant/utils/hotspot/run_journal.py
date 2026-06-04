from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Default per-run journal location (spec §E). Callers may override.
DEFAULT_JOURNAL_PATH = Path("out/hot/state/run_journal.jsonl")


class RunJournal:
    """Accumulates one run's observability record and flushes it as one JSONL line.

    Stage 0 ships the container + append/flush only. The 2nd-order pollution
    alert thresholds over `intentionally_dropped_stale_competitor` are stage 4;
    here that field is a plain placeholder list with no threshold logic.
    """

    def __init__(self, run_date: str, journal_path: Path | None = None):
        self.run_date = run_date
        self.journal_path = Path(journal_path) if journal_path is not None else DEFAULT_JOURNAL_PATH
        self.source_counts: dict[str, int] = {}
        self.stage_timings: dict[str, float] = {}
        self.intentionally_dropped_stale_competitor: list[dict[str, Any]] = []
        self.extra: dict[str, Any] = {}

    def append(self, key: str, value: Any) -> None:
        """Generic append. `source_counts`/`stage_timings` merge dicts; any other key goes to `extra`.

        Contract: callers pass a dict for the two special keys. A non-dict value for them
        falls through to `extra[key]` rather than raising — this is an observability helper,
        so a malformed call degrades to a visible record (under `extra`) instead of crashing a run.
        """
        if key == "source_counts" and isinstance(value, dict):
            self.source_counts.update(value)
        elif key == "stage_timings" and isinstance(value, dict):
            self.stage_timings.update(value)
        else:
            self.extra[key] = value

    def append_stage_timing(self, stage: str, seconds: float) -> None:
        self.stage_timings[stage] = float(seconds)

    def record_dropped_stale_competitor(self, entry: dict[str, Any]) -> None:
        """Append one intentionally-dropped stale competitor item (spec §D.3 / §E)."""
        self.intentionally_dropped_stale_competitor.append(dict(entry))

    def to_record(self) -> dict[str, Any]:
        return {
            "run_date": self.run_date,
            "flushed_at": datetime.now(timezone.utc).isoformat(),
            "source_counts": dict(self.source_counts),
            "stage_timings": dict(self.stage_timings),
            "intentionally_dropped_stale_competitor": list(
                self.intentionally_dropped_stale_competitor
            ),
            "extra": dict(self.extra),
        }

    def flush(self) -> Path:
        """Append this run's record as one JSON line; create parent dirs as needed."""
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(self.to_record(), ensure_ascii=False)
        with self.journal_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return self.journal_path
