from __future__ import annotations

import configparser
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from arxiv_assistant.hotspots import date_verify, kernel
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem

FIXT = Path(__file__).resolve().parent / "fixtures" / "replay" / "raw_2026-05-20.json"


def _load_items() -> list[HotspotItem]:
    rows = json.loads(FIXT.read_text(encoding="utf-8"))
    return [kernel._deserialize_item(r) for r in rows]


def _config() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg["HOTSPOTS"] = {
        "enabled": "true", "mode": "heuristic", "max_raw_items": "120",
        "max_item_age_days": "14", "target_topics": "5", "target_watchlist_topics": "3",
        "max_topics_per_category": "4",
    }
    cfg["HOTSPOT_SOURCES"] = {}
    return cfg


def _stub_verify(item, store):  # noqa: ARG001 - store unused by the stub
    """Deterministic stand-in for date_verify.verify.

    WHY THIS PATCH EXISTS. Without it this test was not measuring the kernel, it
    was measuring the internet. ``date_verify.verify`` calls the live arXiv API,
    Crossref and the Wayback CDX, and may dispatch a subagent; the test patched
    only ``_fetch_source_payloads``, so whichever of those answered (or timed
    out) decided an item's verified_first_date. Two runs minutes apart could land
    on a verifier answer and a published_at fallback respectively, and the
    checkpoint would differ by exactly that item's timestamp.

    Measured: with the network held constant this way, score.json is
    byte-identical across runs; with it live, it intermittently is not. A
    bit-stability test has to hold its inputs constant or it reports the weather.
    The kernel's determinism is the claim under test; date_verify's network
    behaviour is an input, and it is non-deterministic by nature.
    """
    return {"verified_first_date": item.published_at, "confidence": 0.9, "evidence": []}


class TestReplayDiff(unittest.TestCase):
    def _run_once(self, root: Path) -> tuple[str, str]:
        td = datetime(2026, 5, 20, tzinfo=timezone.utc)
        items = _load_items()
        with patch.object(kernel, "_fetch_source_payloads",
                          return_value=(items, {"hf_papers": 1, "ainews": 2}, {})), \
             patch.object(date_verify, "verify", side_effect=_stub_verify):
            kernel.run(root, td, _config(), force=True)
        score = (root / "hot" / "state" / "checkpoint" / "2026-05-20" / "score.json").read_text("utf-8")
        report = json.loads((root / "hot" / "reports" / "2026-05-20.json").read_text("utf-8"))
        report.pop("generated_at", None)  # only non-deterministic field by design
        return score, json.dumps(report, sort_keys=True, ensure_ascii=False)

    def test_two_runs_are_bit_stable(self) -> None:
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            score_a, report_a = self._run_once(Path(a))
            score_b, report_b = self._run_once(Path(b))
        self.assertEqual(score_a, score_b)
        self.assertEqual(report_a, report_b)
