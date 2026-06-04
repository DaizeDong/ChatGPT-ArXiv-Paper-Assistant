from __future__ import annotations

import unittest
from datetime import date, datetime
from unittest.mock import MagicMock, patch

from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem


def _item(url: str, *, provenance: str = "reuse:ainews", arxiv_id: str | None = None) -> HotspotItem:
    md = {"arxiv_id": arxiv_id} if arxiv_id else {}
    return HotspotItem(
        source_id="reuse_ainews", source_name="Reuse:ainews", source_role="community_signal",
        source_type="reuse", title=f"T {url}", summary="s", url=url, canonical_url=url,
        published_at="2026-06-02T00:00:00+00:00", metadata=md, provenance=provenance,
    )


# verified_first_date that DateVerify would return, keyed by url, for the replay stub.
_VERDICT_DATE = {
    "https://a.test/fresh": "2026-06-01T12:00:00+00:00",   # within max_age
    "https://a.test/fresh2": "2026-05-30T00:00:00+00:00",  # within max_age
    "https://a.test/old": "2023-01-01T00:00:00+00:00",     # > 14d -> dropped_stale
    "https://a.test/poison": "2023-03-03T00:00:00+00:00",  # backdated-old, multi-competitor echo
}


def _fake_verify(item: HotspotItem, store) -> dict:
    return {"verified_first_date": _VERDICT_DATE[item.canonical_url], "confidence": 0.95, "evidence": ["wayback"]}


class TestEligibleVsDropped(unittest.TestCase):
    def setUp(self) -> None:
        self.store = MagicMock()
        self.as_of = date(2026, 6, 3)

    def test_stale_competitor_goes_to_dropped_not_eligible(self) -> None:
        from arxiv_assistant.hotspots import gapfill
        comp = [_item("https://a.test/fresh"), _item("https://a.test/old")]
        with patch.object(gapfill.date_verify, "verify", side_effect=_fake_verify):
            eligible, dropped = gapfill.eligible_competitor_items(
                comp, self.store, max_age_days=14, as_of=self.as_of
            )
        eligible_urls = {i.canonical_url for i in eligible}
        dropped_urls = {i.canonical_url for i in dropped}
        self.assertEqual(eligible_urls, {"https://a.test/fresh"})
        self.assertEqual(dropped_urls, {"https://a.test/old"})

    def test_union_floor_passes_when_eligible_covered_despite_stale(self) -> None:
        from arxiv_assistant.hotspots import gapfill
        comp = [_item("https://a.test/fresh"), _item("https://a.test/old")]
        with patch.object(gapfill.date_verify, "verify", side_effect=_fake_verify):
            eligible, _ = gapfill.eligible_competitor_items(comp, self.store, max_age_days=14, as_of=self.as_of)
        # our_coverage contains the eligible item but NOT the >14d stale one.
        our = {"https://a.test/fresh"}
        # Must NOT raise: stale competitor item is excluded from the ⊇ obligation.
        gapfill.assert_union_floor(our, eligible)


class TestAssertUnionFloor(unittest.TestCase):
    def test_raises_when_eligible_item_missing(self) -> None:
        from arxiv_assistant.hotspots import gapfill
        eligible = [_item("https://a.test/fresh"), _item("https://a.test/fresh2")]
        our = {"https://a.test/fresh"}  # missing fresh2
        with self.assertRaises(AssertionError) as ctx:
            gapfill.assert_union_floor(our, eligible)
        self.assertIn("https://a.test/fresh2", str(ctx.exception))


class TestGapfillDirectedFetch(unittest.TestCase):
    def test_gapfill_returns_only_missing_eligible(self) -> None:
        from arxiv_assistant.hotspots import gapfill
        eligible = [_item("https://a.test/fresh"), _item("https://a.test/fresh2")]
        our = {"https://a.test/fresh"}
        new_items = gapfill.gapfill(our, eligible)
        self.assertEqual({i.canonical_url for i in new_items}, {"https://a.test/fresh2"})


class TestDateVerifyHardAnchorNotMajorityVote(unittest.TestCase):
    """Spec §D.4: multiple competitors echoing the same backdated-old paper must be
    rejected by the DateVerify hard anchor, NOT approved by majority vote."""

    def setUp(self) -> None:
        self.store = MagicMock()
        self.as_of = date(2026, 6, 3)

    def test_shared_pollution_rejected_by_hard_anchor(self) -> None:
        from arxiv_assistant.hotspots import gapfill
        # SAME old paper echoed by 3 independent competitor sources (shared pollution).
        comp = [
            _item("https://a.test/poison", provenance="reuse:ainews", arxiv_id="2303.00001"),
            _item("https://a.test/poison", provenance="reuse:hf_daily", arxiv_id="2303.00001"),
            _item("https://a.test/poison", provenance="reuse:horizon", arxiv_id="2303.00001"),
        ]
        with patch.object(gapfill.date_verify, "verify", side_effect=_fake_verify):
            eligible, dropped = gapfill.eligible_competitor_items(
                comp, self.store, max_age_days=14, as_of=self.as_of
            )
        # 3-way consensus does NOT make it eligible: verified_first_date 2023 > 14d.
        self.assertEqual(eligible, [])
        self.assertTrue(any(i.canonical_url == "https://a.test/poison" for i in dropped))
        # And the ⊇ obligation does not force us to cover it.
        gapfill.assert_union_floor(set(), eligible)


if __name__ == "__main__":
    unittest.main()
