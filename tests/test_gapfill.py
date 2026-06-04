from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
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


class TestHarvestReuseLayer(unittest.TestCase):
    """Task 11: harvest_reuse_layer — config-driven dispatch + per-source fault tolerance."""

    def _fresh_item(self, url: str, src: str = "reuse:hf_daily") -> HotspotItem:
        return _item(url, provenance=src)

    def test_dispatches_only_enabled_sources(self) -> None:
        """Only sources present in reuse_sources are harvested; others are skipped."""
        from arxiv_assistant.hotspots import gapfill

        target = datetime(2026, 6, 2, 12, 0, 0, tzinfo=timezone.utc)
        hf_item = self._fresh_item("https://arxiv.org/abs/2406.00001", "reuse:hf_daily")
        ainews_item = self._fresh_item("https://news.smol.ai/issues/1", "reuse:ainews")

        def _fake_import(mod_path: str):
            class FakeMod:
                pass
            m = FakeMod()
            if "hf_daily" in mod_path:
                m.fetch_hotspot_items = lambda *a, **kw: [hf_item]
            elif "ainews" in mod_path:
                m.fetch_hotspot_items = lambda *a, **kw: [ainews_item]
            else:
                m.fetch_hotspot_items = lambda *a, **kw: []
            return m

        with patch("importlib.import_module", side_effect=_fake_import):
            # Only hf_daily enabled — ainews must NOT be harvested
            result = gapfill.harvest_reuse_layer(["hf_daily"], target, 30)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].provenance, "reuse:hf_daily")

    def test_unknown_source_name_is_silently_skipped(self) -> None:
        """A source name not in REUSE_ADAPTERS registry is ignored without error."""
        from arxiv_assistant.hotspots import gapfill

        target = datetime(2026, 6, 2, 12, 0, 0, tzinfo=timezone.utc)
        with patch("importlib.import_module", side_effect=Exception("should not be called")):
            result = gapfill.harvest_reuse_layer(["totally_unknown_source"], target, 30)
        self.assertEqual(result, [])

    def test_one_source_raising_does_not_kill_others(self) -> None:
        """Fault tolerance: one adapter raising must not prevent other adapters from harvesting."""
        from arxiv_assistant.hotspots import gapfill

        target = datetime(2026, 6, 2, 12, 0, 0, tzinfo=timezone.utc)
        good_item = self._fresh_item("https://arxiv.org/abs/2406.99999", "reuse:ainews")

        def _fake_import(mod_path: str):
            class FakeMod:
                pass
            m = FakeMod()
            if "hf_daily" in mod_path:
                def _boom(*a, **kw):
                    raise RuntimeError("hf_daily network error")
                m.fetch_hotspot_items = _boom
            elif "ainews" in mod_path:
                m.fetch_hotspot_items = lambda *a, **kw: [good_item]
            return m

        with patch("importlib.import_module", side_effect=_fake_import):
            result = gapfill.harvest_reuse_layer(["hf_daily", "ainews"], target, 30)

        # hf_daily failed but ainews must still contribute
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].provenance, "reuse:ainews")

    def test_all_enabled_sources_are_harvested(self) -> None:
        """All sources listed in reuse_sources are dispatched (happy path)."""
        from arxiv_assistant.hotspots import gapfill

        target = datetime(2026, 6, 2, 12, 0, 0, tzinfo=timezone.utc)
        call_log: list[str] = []

        def _fake_import(mod_path: str):
            name = mod_path.split(".")[-1]
            call_log.append(name)

            class FakeMod:
                pass
            m = FakeMod()
            m.fetch_hotspot_items = lambda *a, **kw: []
            return m

        with patch("importlib.import_module", side_effect=_fake_import):
            gapfill.harvest_reuse_layer(["hf_daily", "ainews", "agents_radar"], target, 30)

        self.assertIn("reuse_hf_daily", call_log)
        self.assertIn("reuse_ainews", call_log)
        self.assertIn("reuse_agents_radar", call_log)


class TestRunGapfillFloor(unittest.TestCase):
    """Task 11: run_gapfill_floor seam — gap+journal+alerts, competitor_items=eligible+dropped,
    floor holds post-gapfill, journal record written."""

    # Verdicts keyed by canonical_url
    _VERDICTS = {
        "https://a.test/fresh1": "2026-06-01T10:00:00+00:00",   # within 14d -> eligible
        "https://a.test/fresh2": "2026-05-30T00:00:00+00:00",   # within 14d -> eligible
        "https://a.test/stale1": "2023-01-01T00:00:00+00:00",   # >14d -> dropped
    }

    def _fake_verify(self, item, store) -> dict:
        return {"verified_first_date": self._VERDICTS[item.canonical_url], "confidence": 0.9}

    def _run_seam(self, our_coverage, competitor_items, *, journal_path):
        from arxiv_assistant.hotspots import gapfill

        store = MagicMock()
        with patch.object(gapfill.date_verify, "verify", side_effect=self._fake_verify):
            return gapfill.run_gapfill_floor(
                our_coverage,
                competitor_items,
                store,
                max_age_days=14,
                as_of=date(2026, 6, 3),
                run_date="2026-06-03",
                journal_path=journal_path,
            )

    def test_returns_gap_eligible_dropped_alerts(self) -> None:
        """run_gapfill_floor result must contain new_items, eligible, dropped, alerts."""
        comp = [
            _item("https://a.test/fresh1", provenance="reuse:hf_daily"),
            _item("https://a.test/fresh2", provenance="reuse:ainews"),
            _item("https://a.test/stale1", provenance="reuse:ainews"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            jpath = Path(tmp) / "journal.jsonl"
            result = self._run_seam(set(), comp, journal_path=jpath)

        self.assertIn("new_items", result)
        self.assertIn("eligible", result)
        self.assertIn("dropped", result)
        self.assertIn("alerts", result)

    def test_alert_baseline_excludes_today_so_first_run_spike_fires(self) -> None:
        """Ordering fix: the alert baseline is read from PRIOR runs BEFORE appending
        today's record. First run with an all-stale source (drop_ratio=1.0) must FIRE
        (baseline=0.0). With the old append-then-read ordering, today's 1.0 would be its
        OWN baseline -> median([1.0])=1.0 and 1.0 >= 2*1.0 is False -> the spike would be
        self-suppressed. This test locks the fix."""
        comp = [_item("https://a.test/stale1", provenance="reuse:ainews")]  # all stale -> ratio 1.0
        with tempfile.TemporaryDirectory() as tmp:
            jpath = Path(tmp) / "journal.jsonl"
            result = self._run_seam(set(), comp, journal_path=jpath)
        sources = [a["source"] for a in result["alerts"]]
        self.assertIn("reuse:ainews", sources)

    def test_gap_contains_only_items_missing_from_our_coverage(self) -> None:
        """new_items = eligible \\ our_coverage."""
        comp = [
            _item("https://a.test/fresh1", provenance="reuse:hf_daily"),
            _item("https://a.test/fresh2", provenance="reuse:ainews"),
            _item("https://a.test/stale1", provenance="reuse:ainews"),
        ]
        # We already have fresh1
        our_coverage = {"https://a.test/fresh1"}
        with tempfile.TemporaryDirectory() as tmp:
            jpath = Path(tmp) / "journal.jsonl"
            result = self._run_seam(our_coverage, comp, journal_path=jpath)

        new_urls = {i.canonical_url for i in result["new_items"]}
        self.assertEqual(new_urls, {"https://a.test/fresh2"})

    def test_floor_holds_after_gapfill(self) -> None:
        """After gapfill補入, our_coverage ∪ new_items ⊇ eligible (floor satisfied)."""
        from arxiv_assistant.hotspots import gapfill

        comp = [
            _item("https://a.test/fresh1", provenance="reuse:hf_daily"),
            _item("https://a.test/fresh2", provenance="reuse:ainews"),
            _item("https://a.test/stale1", provenance="reuse:ainews"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            jpath = Path(tmp) / "journal.jsonl"
            result = self._run_seam(set(), comp, journal_path=jpath)

        # Should not raise: floor holds after gapfill补入
        covered = {i.canonical_url for i in result["new_items"]}
        gapfill.assert_union_floor(covered, result["eligible"])

    def test_competitor_items_is_full_set_for_correct_ratios(self) -> None:
        """Journal record per_source seen/dropped ratios must use full competitor_items
        (eligible + dropped), not just dropped.  If competitor_items only contained
        dropped, the ratio would be 1.0 for every source; correct behaviour is <1.0 when
        some items are eligible."""
        comp = [
            _item("https://a.test/fresh1", provenance="reuse:ainews"),   # -> eligible
            _item("https://a.test/fresh2", provenance="reuse:ainews"),   # -> eligible
            _item("https://a.test/stale1", provenance="reuse:ainews"),   # -> dropped
        ]
        with tempfile.TemporaryDirectory() as tmp:
            jpath = Path(tmp) / "journal.jsonl"
            result = self._run_seam(set(), comp, journal_path=jpath)

            # Read back the written journal record (must stay inside tmp context)
            lines = jpath.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            rec = json.loads(lines[0])
            ps = rec["per_source"]["reuse:ainews"]
            self.assertEqual(ps["seen"], 3)    # all 3 items counted (eligible+dropped)
            self.assertEqual(ps["dropped"], 1)
            self.assertAlmostEqual(ps["drop_ratio"], round(1 / 3, 4))

    def test_journal_record_written_to_disk(self) -> None:
        """run_gapfill_floor must persist the journal record via run_journal.append."""
        comp = [_item("https://a.test/fresh1", provenance="reuse:hf_daily")]
        with tempfile.TemporaryDirectory() as tmp:
            jpath = Path(tmp) / "journal.jsonl"
            self.assertFalse(jpath.exists())
            self._run_seam(set(), comp, journal_path=jpath)
            self.assertTrue(jpath.exists())
            lines = jpath.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            rec = json.loads(lines[0])
            self.assertEqual(rec["channel"], "intentionally_dropped_stale_competitor")

    def test_stale_item_goes_to_dropped_not_new_items(self) -> None:
        """Stale competitor item (old verified_first_date) must NOT appear in new_items."""
        comp = [_item("https://a.test/stale1", provenance="reuse:ainews")]
        with tempfile.TemporaryDirectory() as tmp:
            jpath = Path(tmp) / "journal.jsonl"
            result = self._run_seam(set(), comp, journal_path=jpath)
        self.assertEqual(result["new_items"], [])
        dropped_urls = {i.canonical_url for i in result["dropped"]}
        self.assertIn("https://a.test/stale1", dropped_urls)

    def test_alerts_is_list(self) -> None:
        """alerts field must always be a list (empty when no spike detected)."""
        comp = [_item("https://a.test/fresh1", provenance="reuse:hf_daily")]
        with tempfile.TemporaryDirectory() as tmp:
            jpath = Path(tmp) / "journal.jsonl"
            result = self._run_seam(set(), comp, journal_path=jpath)
        self.assertIsInstance(result["alerts"], list)


class TestInvariants(unittest.TestCase):
    """Stage-4 acceptance gate: named assertions for each spec §D/§E invariant.

    §D.1 / INV5  — Reuse items inherit recall, not staleness.  They carry
                   provenance="reuse:..." and pass through the IDENTICAL DateVerify
                   + max_age gate as native items.  No separate reuse-only bypass.
    §D.3          — The ⊇ (union-floor) obligation is scoped to *eligible* items only.
                   Items dropped as dropped_stale carry no ⊇ obligation.
    §D.4          — Multi-competitor consensus on a backdated paper is REJECTED by the
                   DateVerify hard anchor; majority vote never approves.
    §E            — The pollution alert reads journal aggregates; baseline excludes the
                   current run so a first-run spike is not self-suppressed.
    INV2          — Sub-day timestamp jitter (e.g. 23:59:59 vs 00:00:01 on the same UTC
                   day) cannot flip the day-granular eligibility gate.
    """

    def test_inv5_reuse_items_carry_reuse_provenance_and_pass_same_gate(self) -> None:
        # Reuse items inherit recall, not staleness: they go through the identical gate.
        from arxiv_assistant.hotspots import gapfill
        store = MagicMock()
        comp = [_item("https://a.test/fresh", provenance="reuse:hf_daily")]
        with patch.object(gapfill.date_verify, "verify", side_effect=_fake_verify):
            eligible, dropped = gapfill.eligible_competitor_items(
                comp, store, max_age_days=14, as_of=date(2026, 6, 3)
            )
        self.assertTrue(all(i.provenance.startswith("reuse:") for i in eligible + dropped))
        self.assertEqual(len(eligible), 1)

    def test_inv2_subday_jitter_cannot_flip_gate(self) -> None:
        from arxiv_assistant.hotspots import gapfill
        store = MagicMock()
        # Two verifies of the SAME url differing only by sub-day time -> same eligibility.
        def jitter_verify(item, _store):
            t = "23:59:59" if item.title.endswith("Z") else "00:00:01"
            return {"verified_first_date": f"2026-05-20T{t}+00:00", "confidence": 0.9, "evidence": []}
        a = _item("https://a.test/jit"); a.title = "X"
        b = _item("https://a.test/jit"); b.title = "XZ"
        with patch.object(gapfill.date_verify, "verify", side_effect=jitter_verify):
            ea, _ = gapfill.eligible_competitor_items([a], store, max_age_days=14, as_of=date(2026, 6, 3))
            eb, _ = gapfill.eligible_competitor_items([b], store, max_age_days=14, as_of=date(2026, 6, 3))
        self.assertEqual(len(ea), len(eb))  # day-granular gate: identical verdict


if __name__ == "__main__":
    unittest.main()
