from __future__ import annotations

import configparser
import unittest
from datetime import UTC, date, datetime
from pathlib import Path

from arxiv_assistant.hotspots.pipeline import _apply_freshness_gates
from arxiv_assistant.utils.hotspot.gate_date import gate_date
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem
from arxiv_assistant.utils.hotspot.hotspot_sources import get_freshness_date


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
        kept = _apply_freshness_gates([stale, fresh], target, max_item_age_days=14)
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
        kept = _apply_freshness_gates([trend], target, max_item_age_days=14)
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
        kept = _apply_freshness_gates(items, target, max_item_age_days=14)
        self.assertEqual(len(kept), 33)  # exactly the 8 stale ones sank, zero agents


class TestConfigMaxItemAge(unittest.TestCase):
    def test_config_has_max_item_age_days(self) -> None:
        cfg = configparser.ConfigParser()
        cfg.read(Path("configs/config.ini"), encoding="utf-8")
        self.assertEqual(cfg["HOTSPOTS"].getint("max_item_age_days", fallback=-1), 14)

    def test_template_documents_max_item_age_days(self) -> None:
        text = Path("configs/templates/config.template.ini").read_text(encoding="utf-8")
        self.assertIn("max_item_age_days", text)


class TestInvariants(unittest.TestCase):
    def test_inv1_gates_use_verified_not_claimed(self) -> None:
        # INV1: a 2023-verified paper claiming 2026 must gate as 2023 everywhere.
        item = _paper("2301.00001", "2026-04-04T00:00:00Z", "2023-01-02T00:00:00Z")
        self.assertEqual(gate_date(item), date(2023, 1, 2))
        self.assertEqual(get_freshness_date(item), "2023-01-02T00:00:00Z")

    def test_inv2_subday_jitter_cannot_flip_gate(self) -> None:
        # INV2: two sub-day-jittered verified dates on the same UTC day → identical
        # gate decision (both kept or both dropped), never split.
        target = datetime(2026, 4, 18, tzinfo=UTC)  # 14 days after 2026-04-04
        a = _paper("2604.0000a", "2026-04-04T00:00:01Z", "2026-04-04T00:00:01Z")
        b = _paper("2604.0000b", "2026-04-04T23:59:59Z", "2026-04-04T23:59:59Z")
        kept = _apply_freshness_gates([a, b], target, max_item_age_days=14)
        # Both have gate_date 2026-04-04, exactly 14 days old → both kept together.
        self.assertEqual(len(kept), 2)

    def test_inv2_boundary_is_day_not_instant(self) -> None:
        target = datetime(2026, 4, 19, tzinfo=UTC)  # 15 days after 2026-04-04
        a = _paper("2604.0000a", "2026-04-04T00:00:01Z", "2026-04-04T00:00:01Z")
        b = _paper("2604.0000b", "2026-04-04T23:59:59Z", "2026-04-04T23:59:59Z")
        kept = _apply_freshness_gates([a, b], target, max_item_age_days=14)
        # Both gate_date 2026-04-04, now 15 days old → both dropped together.
        self.assertEqual(len(kept), 0)


if __name__ == "__main__":
    unittest.main()
