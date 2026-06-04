from __future__ import annotations

import configparser
import unittest
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import patch

from arxiv_assistant.hotspots.pipeline import _apply_freshness_gates
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem


def _paper(arxiv_id: str, published_at: str, verified: str | None) -> HotspotItem:
    item = HotspotItem(
        source_id="hf_papers", source_name="HF", source_role="paper_trending",
        source_type="paper", title=f"paper {arxiv_id}", summary="s",
        url=f"https://huggingface.co/papers/{arxiv_id}",
        canonical_url=f"https://arxiv.org/abs/{arxiv_id}",
        published_at=published_at, metadata={"arxiv_id": arxiv_id},
    )
    item.verified_first_date = verified
    return item


class TestStage1FreshnessGate(unittest.TestCase):
    def test_stale_verified_paper_dropped_zero_agents(self) -> None:
        target = datetime(2026, 4, 4, tzinfo=UTC)
        # Source claims today's date, but verified v1 is 2023 → must be dropped.
        stale = _paper("2301.00001", "2026-04-04T00:00:00Z", "2023-01-02T00:00:00Z")
        fresh = _paper("2604.00002", "2026-04-04T00:00:00Z", "2026-04-04T00:00:00Z")
        kept = _apply_freshness_gates([stale, fresh], target, max_item_age_days=14,
                                      freshness_hours=24)
        kept_ids = {i.metadata["arxiv_id"] for i in kept}
        self.assertIn("2604.00002", kept_ids)
        self.assertNotIn("2301.00001", kept_ids)

    def test_github_trend_exempt_from_max_age(self) -> None:
        target = datetime(2026, 4, 4, tzinfo=UTC)
        trend = HotspotItem(
            source_id="github_trend", source_name="GH", source_role="repo_trending",
            source_type="repo", title="old repo", summary="s",
            url="https://github.com/a/b", canonical_url="https://github.com/a/b",
            published_at="2020-01-01T00:00:00Z",
            metadata={"fetched_at": "2026-04-04T00:00:00Z"},
        )
        kept = _apply_freshness_gates([trend], target, max_item_age_days=14,
                                      freshness_hours=24)
        self.assertEqual(len(kept), 1)  # exempt → kept despite 2020 published_at

    def test_golden_eight_of_fortyone_sink_without_agents(self) -> None:
        # 41 papers: 8 are stale-verified (v1 > 14d old), 33 are genuinely fresh.
        target = datetime(2026, 4, 4, tzinfo=UTC)
        items = []
        for n in range(33):
            items.append(_paper(f"2604.1{n:04d}", "2026-04-04T00:00:00Z",
                                 "2026-04-04T00:00:00Z"))
        for n in range(8):
            items.append(_paper(f"2301.0{n:04d}", "2026-04-04T00:00:00Z",
                                 "2023-01-02T00:00:00Z"))
        kept = _apply_freshness_gates(items, target, max_item_age_days=14,
                                      freshness_hours=24)
        self.assertEqual(len(kept), 33)  # exactly the 8 stale ones sank, zero agents


class TestConfigMaxItemAge(unittest.TestCase):
    def test_config_has_max_item_age_days(self) -> None:
        cfg = configparser.ConfigParser()
        cfg.read(Path("configs/config.ini"), encoding="utf-8")
        self.assertEqual(cfg["HOTSPOTS"].getint("max_item_age_days", fallback=-1), 14)

    def test_template_documents_max_item_age_days(self) -> None:
        text = Path("configs/templates/config.template.ini").read_text(encoding="utf-8")
        self.assertIn("max_item_age_days", text)


if __name__ == "__main__":
    unittest.main()
