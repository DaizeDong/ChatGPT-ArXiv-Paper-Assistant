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


class TestSecondOrderPollutionAlert(unittest.TestCase):
    """Spec §E: single-source dropped-ratio spike vs trailing-14-run median baseline."""

    def _run(self, ratio: float, src: str = "reuse:ainews") -> dict:
        seen, dropped = 100, int(round(ratio * 100))
        return {
            "channel": "intentionally_dropped_stale_competitor",
            "per_source": {src: {"seen": seen, "dropped": dropped, "drop_ratio": ratio}},
        }

    def test_spike_triggers_alert(self) -> None:
        from arxiv_assistant.hotspots import gapfill
        history = [self._run(0.05) for _ in range(14)]          # stable 5% baseline
        today = self._run(0.40)                                  # spike: 8x baseline AND >=30%
        alerts = gapfill.second_order_pollution_alerts(today, history, multiplier=2.0, abs_floor=0.30)
        self.assertEqual([a["source"] for a in alerts], ["reuse:ainews"])

    def test_stable_does_not_trigger(self) -> None:
        from arxiv_assistant.hotspots import gapfill
        history = [self._run(0.20) for _ in range(14)]          # baseline 20%
        today = self._run(0.25)                                  # below 2x AND not a big jump
        alerts = gapfill.second_order_pollution_alerts(today, history, multiplier=2.0, abs_floor=0.30)
        self.assertEqual(alerts, [])

    def test_high_abs_but_below_2x_baseline_does_not_trigger(self) -> None:
        from arxiv_assistant.hotspots import gapfill
        # Source legitimately curates many old items every day: 35% steady baseline.
        history = [self._run(0.35) for _ in range(14)]
        today = self._run(0.40)                                  # >=30% abs but only 1.14x baseline
        alerts = gapfill.second_order_pollution_alerts(today, history, multiplier=2.0, abs_floor=0.30)
        self.assertEqual(alerts, [])

    def test_zero_baseline_uses_abs_floor_guard(self) -> None:
        from arxiv_assistant.hotspots import gapfill
        history = [self._run(0.0) for _ in range(14)]            # never dropped before
        today = self._run(0.40)                                  # first big drop
        alerts = gapfill.second_order_pollution_alerts(today, history, multiplier=2.0, abs_floor=0.30)
        self.assertEqual([a["source"] for a in alerts], ["reuse:ainews"])


if __name__ == "__main__":
    unittest.main()
