from __future__ import annotations

import unittest
from datetime import date

from arxiv_assistant.utils.hotspot.gate_date import floor_to_utc_day, gate_date
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem


class TestFloorToUtcDay(unittest.TestCase):
    def test_none_returns_none(self) -> None:
        self.assertIsNone(floor_to_utc_day(None))

    def test_empty_string_returns_none(self) -> None:
        self.assertIsNone(floor_to_utc_day(""))

    def test_unparseable_returns_none(self) -> None:
        self.assertIsNone(floor_to_utc_day("not-a-date"))

    def test_date_only_string(self) -> None:
        self.assertEqual(floor_to_utc_day("2026-04-04"), date(2026, 4, 4))

    def test_zulu_timestamp_truncates_time(self) -> None:
        self.assertEqual(floor_to_utc_day("2026-04-04T23:59:59Z"), date(2026, 4, 4))

    def test_subday_jitter_same_day(self) -> None:
        # Two sub-day-different timestamps on the same UTC day floor identically (INV2).
        a = floor_to_utc_day("2026-04-04T00:00:01Z")
        b = floor_to_utc_day("2026-04-04T23:59:58Z")
        self.assertEqual(a, b)
        self.assertEqual(a, date(2026, 4, 4))

    def test_offset_converted_to_utc_before_flooring(self) -> None:
        # 2026-04-04T01:00:00+09:00 == 2026-04-03T16:00:00Z → floors to the 3rd.
        self.assertEqual(floor_to_utc_day("2026-04-04T01:00:00+09:00"), date(2026, 4, 3))

    def test_naive_timestamp_assumed_utc(self) -> None:
        self.assertEqual(floor_to_utc_day("2026-04-04 12:30:00"), date(2026, 4, 4))


# ---------------------------------------------------------------------------
# Task 2: gate_date over HotspotItem
# ---------------------------------------------------------------------------


def _item(**kw) -> HotspotItem:
    base = dict(
        source_id="hf_papers",
        source_name="HF",
        source_role="paper_trending",
        source_type="paper",
        title="t",
        summary="s",
        url="https://huggingface.co/papers/2301.00001",
        canonical_url="https://arxiv.org/abs/2301.00001",
        published_at="2026-04-04T12:00:00Z",
    )
    base.update(kw)
    item = HotspotItem(**{k: v for k, v in base.items() if k != "verified_first_date"})
    if "verified_first_date" in kw:
        item.verified_first_date = kw["verified_first_date"]
    return item


class TestGateDate(unittest.TestCase):
    def test_no_credible_date_returns_none(self) -> None:
        # verified_first_date unset, no anchors → None (cannot verify, do not drop).
        item = _item(metadata={})
        item.verified_first_date = None
        self.assertIsNone(gate_date(item))

    def test_uses_verified_first_date(self) -> None:
        item = _item(verified_first_date="2023-01-02T09:00:00Z", metadata={})
        self.assertEqual(gate_date(item), date(2023, 1, 2))

    def test_ignores_published_at_even_when_earlier_than_verified(self) -> None:
        # Adversarial backdating: source claims an EARLIER published_at than the
        # verified first date. published_at must never enter the credible set (INV1),
        # so the verified date still wins — not the earlier source-claimed one.
        item = _item(
            verified_first_date="2023-01-02T09:00:00Z",
            published_at="2020-01-01T00:00:00Z",
            metadata={},
        )
        self.assertEqual(gate_date(item), date(2023, 1, 2))

    def test_ignores_published_at_when_verified_present(self) -> None:
        # INV1: source-claimed published_at (2026) must NOT win over verified (2023).
        item = _item(verified_first_date="2023-01-02T00:00:00Z",
                     published_at="2026-04-04T12:00:00Z", metadata={})
        self.assertEqual(gate_date(item), date(2023, 1, 2))

    def test_arxiv_announced_anchor_min_wins(self) -> None:
        # Authoritative whole-day anchor earlier than verified → min wins (§B.3.1).
        item = _item(verified_first_date="2023-01-05T00:00:00Z",
                     metadata={"arxiv_announced_date": "2023-01-02"})
        self.assertEqual(gate_date(item), date(2023, 1, 2))

    def test_crossref_anchor_min_wins(self) -> None:
        item = _item(verified_first_date="2024-06-10T00:00:00Z",
                     metadata={"crossref_registered_date": "2024-06-09"})
        self.assertEqual(gate_date(item), date(2024, 6, 9))

    def test_min_is_monotone_earliest_wins(self) -> None:
        # All three credible dates present; earliest (anchor) wins regardless of order.
        item = _item(verified_first_date="2025-03-03T00:00:00Z",
                     metadata={"arxiv_announced_date": "2025-03-01",
                               "crossref_registered_date": "2025-03-02"})
        self.assertEqual(gate_date(item), date(2025, 3, 1))

    def test_subday_jitter_absorbed_by_floor(self) -> None:
        # INV2: two sub-day-jittered verified dates on same UTC day → same gate_date.
        a = gate_date(_item(verified_first_date="2026-04-04T00:00:30Z", metadata={}))
        b = gate_date(_item(verified_first_date="2026-04-04T22:10:00Z", metadata={}))
        self.assertEqual(a, b)
        self.assertEqual(a, date(2026, 4, 4))

    def test_anchor_only_no_verified(self) -> None:
        item = _item(metadata={"arxiv_announced_date": "2023-01-02"})
        item.verified_first_date = None
        self.assertEqual(gate_date(item), date(2023, 1, 2))


if __name__ == "__main__":
    unittest.main()
