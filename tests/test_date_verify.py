from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import patch

from arxiv_assistant.hotspots.date_verify import (
    _content_hash,
    _fetch_arxiv_v1_date,
    _fetch_crossref_date,
    poll_arxiv_versions,
    verify,
)
from arxiv_assistant.hotspots.story import _freshness_weight
from arxiv_assistant.utils.hotspot.gate_date import floor_to_utc_day, gate_date
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem
from arxiv_assistant.utils.hotspot.hotspot_sources import get_freshness_date

_ARXIV_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00001v3</id>
    <published>2023-01-02T18:00:00Z</published>
    <updated>2026-03-30T10:00:00Z</updated>
    <title>An old paper resurfacing on HF</title>
  </entry>
</feed>"""


class TestFetchArxivV1Date(unittest.TestCase):
    @patch("arxiv_assistant.hotspots.date_verify.fetch_text")
    def test_reads_v1_published_not_updated(self, mock_fetch) -> None:
        mock_fetch.return_value = _ARXIV_ATOM
        result = _fetch_arxiv_v1_date("2301.00001")
        # v1 submission date (published), NOT the v3 updated date.
        self.assertEqual(result, "2023-01-02T18:00:00Z")

    @patch("arxiv_assistant.hotspots.date_verify.fetch_text")
    def test_strips_version_suffix_in_query(self, mock_fetch) -> None:
        mock_fetch.return_value = _ARXIV_ATOM
        _fetch_arxiv_v1_date("2301.00001v3")
        called_url = mock_fetch.call_args[0][0]
        self.assertIn("id_list=2301.00001", called_url)
        self.assertNotIn("v3", called_url.split("id_list=")[-1])

    @patch("arxiv_assistant.hotspots.date_verify.fetch_text")
    def test_no_entry_returns_none(self, mock_fetch) -> None:
        mock_fetch.return_value = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        self.assertIsNone(_fetch_arxiv_v1_date("9999.99999"))

    @patch("arxiv_assistant.hotspots.date_verify.fetch_text", side_effect=RuntimeError("net"))
    def test_network_error_returns_none(self, mock_fetch) -> None:
        self.assertIsNone(_fetch_arxiv_v1_date("2301.00001"))


# ---------------------------------------------------------------------------
# Task 4: Crossref registration-date reader
# ---------------------------------------------------------------------------
_CROSSREF_JSON = {
    "message": {
        "created": {"date-parts": [[2024, 6, 9]], "date-time": "2024-06-09T08:00:00Z"}
    }
}


class TestFetchCrossrefDate(unittest.TestCase):
    @patch("arxiv_assistant.hotspots.date_verify.fetch_json")
    def test_reads_created_date(self, mock_json) -> None:
        mock_json.return_value = _CROSSREF_JSON
        self.assertEqual(_fetch_crossref_date("10.1145/1234.5678"), "2024-06-09")

    @patch("arxiv_assistant.hotspots.date_verify.fetch_json")
    def test_missing_created_returns_none(self, mock_json) -> None:
        mock_json.return_value = {"message": {}}
        self.assertIsNone(_fetch_crossref_date("10.1145/1234.5678"))

    @patch("arxiv_assistant.hotspots.date_verify.fetch_json", side_effect=RuntimeError("net"))
    def test_network_error_returns_none(self, mock_json) -> None:
        self.assertIsNone(_fetch_crossref_date("10.1145/1234.5678"))


# ---------------------------------------------------------------------------
# Task 5: verify() — Tier-0 dispatch, earliest-credible, write-once cache
# ---------------------------------------------------------------------------


class _FakeStore:
    """Minimal write-once verdict cache matching StoryStore.get/put_verdict."""

    def __init__(self) -> None:
        self._verdicts: dict[str, dict] = {}
        self.put_calls = 0

    def get_verdict(self, content_hash: str):
        return self._verdicts.get(content_hash)

    def put_verdict(self, content_hash: str, verdict: dict) -> None:
        self.put_calls += 1
        self._verdicts.setdefault(content_hash, verdict)  # write-once


def _hf_item(**kw) -> HotspotItem:
    item = HotspotItem(
        source_id=kw.get("source_id", "hf_papers"),
        source_name="HF",
        source_role="paper_trending",
        source_type="paper",
        title="t",
        summary="s",
        url="https://huggingface.co/papers/2301.00001",
        canonical_url="https://arxiv.org/abs/2301.00001",
        published_at=kw.get("published_at", "2026-04-04T12:00:00Z"),
        metadata=kw.get("metadata", {"arxiv_id": "2301.00001"}),
    )
    return item


class TestVerify(unittest.TestCase):
    @patch("arxiv_assistant.hotspots.date_verify._fetch_arxiv_v1_date",
           return_value="2023-01-02T18:00:00Z")
    def test_arxiv_id_yields_v1_old_date(self, _m) -> None:
        store = _FakeStore()
        verdict = verify(_hf_item(), store)
        # The 2023 v1 date beats the 2026 source-claimed published_at (INV1).
        self.assertEqual(verdict["verified_first_date"], "2023-01-02T18:00:00Z")
        self.assertGreaterEqual(verdict["confidence"], 0.9)
        self.assertTrue(any("arxiv" in e.lower() for e in verdict["evidence"]))

    @patch("arxiv_assistant.hotspots.date_verify._fetch_arxiv_v1_date",
           return_value="2023-01-02T18:00:00Z")
    def test_cache_written_once_and_frozen(self, _m) -> None:
        store = _FakeStore()
        first = verify(_hf_item(), store)
        # Second call hits cache; even if the network would now return a different
        # date, the frozen verdict is returned unchanged (INV3 freeze).
        with patch("arxiv_assistant.hotspots.date_verify._fetch_arxiv_v1_date",
                   return_value="2099-12-31T00:00:00Z"):
            second = verify(_hf_item(), store)
        self.assertEqual(first, second)
        self.assertEqual(first["verified_first_date"], "2023-01-02T18:00:00Z")
        self.assertEqual(store.put_calls, 1)  # written exactly once

    def test_github_trend_uses_observed_trending_date(self) -> None:
        store = _FakeStore()
        item = _hf_item(source_id="github_trend",
                        published_at="2026-04-04T00:00:00Z",
                        metadata={})
        verdict = verify(item, store)
        self.assertEqual(verdict["verified_first_date"], "2026-04-04T00:00:00Z")
        self.assertGreaterEqual(verdict["confidence"], 0.9)

    @patch("arxiv_assistant.hotspots.date_verify._fetch_arxiv_v1_date", return_value=None)
    def test_no_anchor_conservative_low_confidence(self, _m) -> None:
        store = _FakeStore()
        item = _hf_item(metadata={}, published_at="2026-04-04T12:00:00Z")
        verdict = verify(item, store)
        # Conservative Stage-1 fallback: min(claimed, fetched), low confidence.
        self.assertEqual(verdict["verified_first_date"], "2026-04-04T12:00:00Z")
        self.assertLessEqual(verdict["confidence"], 0.5)

    def test_content_hash_prefers_arxiv_id_version_stripped(self) -> None:
        item = _hf_item(metadata={"arxiv_id": "2301.00001v3"})
        self.assertEqual(_content_hash(item), "arxiv:2301.00001")

    @patch("arxiv_assistant.hotspots.date_verify._fetch_arxiv_v1_date", return_value=None)
    @patch("arxiv_assistant.hotspots.date_verify._fetch_crossref_date",
           return_value="2024-01-15")
    def test_doi_fallback_to_crossref(self, _cr, _ax) -> None:
        """When no arxiv_id but a DOI is present, Crossref date is used as anchor."""
        store = _FakeStore()
        item = HotspotItem(
            source_id="hf_papers",
            source_name="HF",
            source_role="paper_trending",
            source_type="paper",
            title="t",
            summary="s",
            url="https://example.com/paper",
            canonical_url="https://example.com/paper",
            published_at="2026-04-04T12:00:00Z",
            metadata={"doi": "10.1234/test.5678"},
        )
        verdict = verify(item, store)
        self.assertEqual(verdict["verified_first_date"], "2024-01-15")
        self.assertGreaterEqual(verdict["confidence"], 0.9)
        self.assertTrue(any("crossref" in e.lower() for e in verdict["evidence"]))

    @patch("arxiv_assistant.hotspots.date_verify._fetch_arxiv_v1_date",
           return_value="2023-01-02T18:00:00Z")
    def test_earliest_credible_date_wins_over_claimed(self, _m) -> None:
        """Earliest-credible-date-wins: arXiv v1 2023 must beat source-claimed 2026."""
        store = _FakeStore()
        item = _hf_item(published_at="2026-06-01T00:00:00Z",
                        metadata={"arxiv_id": "2301.00001"})
        verdict = verify(item, store)
        self.assertEqual(verdict["verified_first_date"], "2023-01-02T18:00:00Z")

    def test_content_hash_doi_when_no_arxiv(self) -> None:
        item = HotspotItem(
            source_id="hf_papers",
            source_name="HF",
            source_role="paper_trending",
            source_type="paper",
            title="t",
            summary="s",
            url="https://example.com/paper",
            canonical_url="https://example.com/paper",
            published_at="2026-04-04T12:00:00Z",
            metadata={"doi": "10.1234/test.5678"},
        )
        self.assertEqual(_content_hash(item), "doi:10.1234/test.5678")

    def test_content_hash_url_when_no_arxiv_or_doi(self) -> None:
        item = HotspotItem(
            source_id="hf_papers",
            source_name="HF",
            source_role="paper_trending",
            source_type="paper",
            title="t",
            summary="s",
            url="https://example.com/paper",
            canonical_url="https://example.com/paper",
            published_at="2026-04-04T12:00:00Z",
            metadata={},
        )
        self.assertIn("url:", _content_hash(item))

    @patch("arxiv_assistant.hotspots.date_verify._fetch_arxiv_v1_date",
           return_value="2023-01-02T18:00:00Z")
    def test_verify_signature_accepts_will_be_featured_kwarg(self, _m) -> None:
        """verify() must accept will_be_featured as a keyword arg (§2.5 contract)."""
        store = _FakeStore()
        # should not raise — just passes stage-3 gating flag through
        verdict = verify(_hf_item(), store, will_be_featured=True)
        self.assertIn("verified_first_date", verdict)

    @patch("arxiv_assistant.hotspots.date_verify._fetch_arxiv_v1_date",
           return_value="2023-01-02T18:00:00Z")
    def test_verdict_persisted_to_store(self, _m) -> None:
        """put_verdict must be called so the verdict freezes in the store."""
        store = _FakeStore()
        verify(_hf_item(), store)
        self.assertEqual(store.put_calls, 1)
        cached = store.get_verdict("arxiv:2301.00001")
        self.assertIsNotNone(cached)
        self.assertEqual(cached["verified_first_date"], "2023-01-02T18:00:00Z")


# ---------------------------------------------------------------------------
# Task 6: poll_arxiv_versions — batched version-count read (INV3 decoupled)
# ---------------------------------------------------------------------------

_VERSIONS_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><id>http://arxiv.org/abs/2301.00001v3</id></entry>
  <entry><id>http://arxiv.org/abs/2302.00002v1</id></entry>
</feed>"""


class TestPollArxivVersions(unittest.TestCase):
    @patch("arxiv_assistant.hotspots.date_verify.fetch_text")
    def test_parses_version_counts(self, mock_fetch) -> None:
        mock_fetch.return_value = _VERSIONS_ATOM
        result = poll_arxiv_versions(["2301.00001", "2302.00002"])
        self.assertEqual(result, {"2301.00001": 3, "2302.00002": 1})

    @patch("arxiv_assistant.hotspots.date_verify.fetch_text")
    def test_empty_input_no_fetch(self, mock_fetch) -> None:
        self.assertEqual(poll_arxiv_versions([]), {})
        mock_fetch.assert_not_called()

    @patch("arxiv_assistant.hotspots.date_verify.fetch_text", side_effect=RuntimeError("net"))
    def test_network_error_returns_empty(self, _m) -> None:
        self.assertEqual(poll_arxiv_versions(["2301.00001"]), {})

    @patch("arxiv_assistant.hotspots.date_verify.fetch_text")
    def test_dedups_ids_in_query(self, mock_fetch) -> None:
        mock_fetch.return_value = _VERSIONS_ATOM
        poll_arxiv_versions(["2301.00001v2", "2301.00001"])  # same bare id twice
        called_url = mock_fetch.call_args[0][0]
        self.assertEqual(called_url.split("id_list=")[-1].split("&")[0], "2301.00001")

    @patch("arxiv_assistant.hotspots.date_verify.fetch_text")
    def test_batching_splits_large_input(self, mock_fetch) -> None:
        """More than 100 ids must produce multiple fetch_text calls (<=100 per batch)."""
        mock_fetch.return_value = _VERSIONS_ATOM
        ids = [f"23{i:02d}.{j:05d}" for i in range(10) for j in range(11)]  # 110 ids
        poll_arxiv_versions(ids)
        self.assertEqual(mock_fetch.call_count, 2)  # batch 1: 100, batch 2: 10

    @patch("arxiv_assistant.hotspots.date_verify.fetch_text")
    def test_inv3_no_verdict_write(self, mock_fetch) -> None:
        """poll_arxiv_versions must NEVER touch date_verdicts or put_verdict (INV3).

        The structural guarantee is that the function has no `store` parameter and so
        cannot write a verdict. We assert the signature directly (guards against a future
        refactor accidentally threading a store in), plus that it returns plain counts.
        """
        import inspect

        params = inspect.signature(poll_arxiv_versions).parameters
        self.assertEqual(list(params), ["arxiv_ids"])  # no `store` param → cannot freeze

        mock_fetch.return_value = _VERSIONS_ATOM
        result = poll_arxiv_versions(["2301.00001", "2302.00002"])
        self.assertEqual(result, {"2301.00001": 3, "2302.00002": 1})


class TestFreshnessWeight(unittest.TestCase):
    def test_same_day_full_weight(self) -> None:
        run = date(2026, 4, 4)
        w = _freshness_weight(date(2026, 4, 4), run_date=run)
        self.assertAlmostEqual(w, 1.0, places=6)

    def test_decays_with_age(self) -> None:
        run = date(2026, 4, 10)
        fresh = _freshness_weight(date(2026, 4, 10), run_date=run)
        old = _freshness_weight(date(2026, 4, 4), run_date=run)  # 6 days → sinks
        self.assertLess(old, fresh)
        self.assertLess(old, 0.2)  # 6-day-old story folds below the line

    def test_none_gate_date_neutral(self) -> None:
        # Unverifiable date → neutral 0.6 (matches legacy unknown-date behavior).
        self.assertAlmostEqual(_freshness_weight(None, run_date=date(2026, 4, 4)), 0.6, places=6)

    def test_subday_jitter_floors_to_same_weight(self) -> None:
        # Two timestamps on the SAME UTC day but different times floor to the same
        # gate_day, so the gravity weight is identical — sub-day jitter cannot flip
        # the discrete freshness outcome (INV2). Exercises floor + weight together.
        run = date(2026, 4, 6)
        early = _freshness_weight(floor_to_utc_day("2026-04-04T00:00:01Z"), run_date=run)
        late = _freshness_weight(floor_to_utc_day("2026-04-04T23:59:59Z"), run_date=run)
        self.assertEqual(early, late)


class TestGetFreshnessDate(unittest.TestCase):
    def test_prefers_verified_first_date(self) -> None:
        item = _hf_item(published_at="2026-04-04T12:00:00Z")
        item.verified_first_date = "2023-01-02T18:00:00Z"
        # INV1: verified date wins over source-claimed published_at.
        self.assertEqual(get_freshness_date(item), "2023-01-02T18:00:00Z")

    def test_falls_back_to_published_at_when_unverified(self) -> None:
        item = _hf_item(published_at="2026-04-04T12:00:00Z")
        item.verified_first_date = None
        self.assertEqual(get_freshness_date(item), "2026-04-04T12:00:00Z")

    def test_github_trend_still_uses_fetched_at(self) -> None:
        item = _hf_item(source_id="github_trend",
                        published_at="2026-04-01T00:00:00Z",
                        metadata={"fetched_at": "2026-04-04T00:00:00Z"})
        item.verified_first_date = None
        self.assertEqual(get_freshness_date(item), "2026-04-04T00:00:00Z")


class TestAntiPollutionReads(unittest.TestCase):
    def test_wayback_earliest_snapshot_parses_first_timestamp(self):
        from arxiv_assistant.hotspots import date_verify
        # CDX returns rows [["timestamp"], ["20231114083012"], ["20240101000000"]]
        cdx_rows = [["timestamp"], ["20231114083012"], ["20240101000000"]]
        with patch.object(date_verify, "fetch_json", return_value=cdx_rows):
            earliest = date_verify._wayback_earliest_snapshot("https://example.com/x")
        self.assertEqual(earliest, "2023-11-14T08:30:12Z")

    def test_wayback_earliest_snapshot_returns_none_on_empty(self):
        from arxiv_assistant.hotspots import date_verify
        with patch.object(date_verify, "fetch_json", return_value=[["timestamp"]]):
            self.assertIsNone(date_verify._wayback_earliest_snapshot("https://example.com/x"))

    def test_wayback_earliest_snapshot_returns_none_on_network_error(self):
        from arxiv_assistant.hotspots import date_verify
        with patch.object(date_verify, "fetch_json", side_effect=RuntimeError("boom")):
            self.assertIsNone(date_verify._wayback_earliest_snapshot("https://example.com/x"))

    def test_page_published_time_reads_meta_property(self):
        from arxiv_assistant.hotspots import date_verify
        html = '<html><head><meta property="article:published_time" content="2023-11-14T08:30:00Z"></head></html>'
        with patch.object(date_verify, "fetch_text", return_value=html):
            self.assertEqual(date_verify._page_published_time("https://example.com/x"), "2023-11-14T08:30:00Z")

    def test_page_published_time_reads_jsonld_datepublished(self):
        from arxiv_assistant.hotspots import date_verify
        html = (
            '<html><head><script type="application/ld+json">'
            '{"@type":"Article","datePublished":"2024-02-01T00:00:00Z"}'
            '</script></head></html>'
        )
        with patch.object(date_verify, "fetch_text", return_value=html):
            self.assertEqual(date_verify._page_published_time("https://example.com/x"), "2024-02-01T00:00:00Z")

    def test_page_published_time_reversed_meta_attribute_order(self):
        """Fix 3: content attr before property attr must still be parsed."""
        from arxiv_assistant.hotspots import date_verify
        # reversed order: content first, property second
        html = '<html><head><meta content="2025-03-15T10:00:00Z" property="article:published_time"></head></html>'
        with patch.object(date_verify, "fetch_text", return_value=html):
            self.assertEqual(date_verify._page_published_time("https://example.com/x"), "2025-03-15T10:00:00Z")

    def test_page_published_time_jsonld_graph_wrapped(self):
        """Fix 3: @graph-wrapped JSON-LD must yield datePublished from nested objects."""
        from arxiv_assistant.hotspots import date_verify
        html = (
            '<html><head><script type="application/ld+json">'
            '{"@context":"https://schema.org","@graph":[{"@type":"Article","datePublished":"2024-05-20T00:00:00Z"}]}'
            '</script></head></html>'
        )
        with patch.object(date_verify, "fetch_text", return_value=html):
            self.assertEqual(date_verify._page_published_time("https://example.com/x"), "2024-05-20T00:00:00Z")


class TestClampVerdict(unittest.TestCase):
    def test_clamp_picks_earliest_credible_and_floors_to_day(self):
        from arxiv_assistant.hotspots.date_verify import _clamp_verdict
        clamped = _clamp_verdict(
            claimed_iso="2026-06-02T09:00:00Z",
            agent_out={
                "verified_first_date": "2023-11-14T08:30:00Z",
                "confidence": 0.9,
                "evidence": ["wayback_cdx:20231114083012"],
                "stale_date_pollution": True,
            },
            wayback_earliest="2023-11-14T00:00:00Z",
            page_published_time="2023-11-14T08:30:00Z",
        )
        # earliest-credible-date-wins -> the 2023 day, not the claimed 2026 day
        self.assertEqual(clamped["verified_first_date"], "2023-11-14T00:00:00Z")
        self.assertTrue(clamped["stale_date_pollution"])
        self.assertGreaterEqual(clamped["confidence"], 0.0)

    def test_clamp_ignores_agent_date_later_than_evidence(self):
        from arxiv_assistant.hotspots.date_verify import _clamp_verdict
        # agent hallucinates a LATER date than Wayback proves -> verifier overrides with the earlier
        clamped = _clamp_verdict(
            claimed_iso="2026-06-02T09:00:00Z",
            agent_out={
                "verified_first_date": "2026-06-02T00:00:00Z",
                "confidence": 0.95,
                "evidence": [],
                "stale_date_pollution": False,
            },
            wayback_earliest="2023-11-14T00:00:00Z",
            page_published_time=None,
        )
        self.assertEqual(clamped["verified_first_date"], "2023-11-14T00:00:00Z")
        self.assertTrue(clamped["stale_date_pollution"])

    def test_clamp_falls_back_to_min_claimed_fetched_when_no_signals(self):
        from arxiv_assistant.hotspots.date_verify import _clamp_verdict
        clamped = _clamp_verdict(
            claimed_iso="2026-06-02T09:00:00Z",
            agent_out=None,            # agent failed / unparseable
            wayback_earliest=None,
            page_published_time=None,
        )
        # no credible earlier signal -> conservative claimed day, low confidence, not flagged
        self.assertEqual(clamped["verified_first_date"], "2026-06-02T00:00:00Z")
        self.assertLess(clamped["confidence"], 0.5)
        self.assertFalse(clamped["stale_date_pollution"])


class TestClampVerdictINV6AntiHallucination(unittest.TestCase):
    """Fix 4: INV6 — agent proposes a future date with no external signal → confidence capped LOW."""

    def test_clamp_rejects_hallucinated_future_agent_date_no_external_signal(self):
        """Agent claims 2099 with no Wayback/page signal: min picks the claimed date,
        confidence must be LOW (not the agent's high 0.95)."""
        from arxiv_assistant.hotspots.date_verify import _clamp_verdict, _CONFIDENCE_LOW
        claimed = "2023-11-14T00:00:00Z"
        clamped = _clamp_verdict(
            claimed_iso=claimed,
            agent_out={
                "verified_first_date": "2099-01-01T00:00:00Z",  # hallucinated future
                "confidence": 0.95,
                "evidence": [],
            },
            wayback_earliest=None,
            page_published_time=None,
        )
        # min picks the claimed date (2023), not the hallucinated 2099
        self.assertEqual(clamped["verified_first_date"], "2023-11-14T00:00:00Z")
        # stale_date_pollution must be False: claimed IS the earliest, not stale
        self.assertFalse(clamped["stale_date_pollution"])
        # confidence must be capped to LOW — agent's 0.95 must not leak through
        self.assertLessEqual(clamped["confidence"], _CONFIDENCE_LOW)

    def test_clamp_hallucinated_future_agent_wayback_proves_earlier(self):
        """Agent claims 2099 but Wayback shows 2023: min picks 2023, stale=True, confidence
        should reflect the external Wayback signal (agent was overridden but Wayback is real)."""
        from arxiv_assistant.hotspots.date_verify import _clamp_verdict, _CONFIDENCE_LOW
        claimed = "2026-06-02T09:00:00Z"
        clamped = _clamp_verdict(
            claimed_iso=claimed,
            agent_out={
                "verified_first_date": "2099-01-01T00:00:00Z",  # hallucinated future
                "confidence": 0.95,
                "evidence": [],
            },
            wayback_earliest="2023-11-14T00:00:00Z",
            page_published_time=None,
        )
        # Wayback 2023 is the earliest credible signal
        self.assertEqual(clamped["verified_first_date"], "2023-11-14T00:00:00Z")
        # claimed (2026) > earliest (2023) → stale_date_pollution
        self.assertTrue(clamped["stale_date_pollution"])
        # Wayback IS a real external signal, so confidence is carried (not hard-capped to LOW)
        # but it is a valid clamped-to-[0,1] value
        self.assertGreaterEqual(clamped["confidence"], 0.0)
        self.assertLessEqual(clamped["confidence"], 1.0)


class TestGateDateAuthoritativeAnchor(unittest.TestCase):
    def _item(self, *, published_at, metadata=None):
        return HotspotItem(
            source_id="hf_papers",
            source_name="HF",
            source_role="paper_trending",
            source_type="paper",
            title="Old paper resurfaced as new",
            summary="",
            url="https://huggingface.co/papers/2311.01234",
            canonical_url="https://arxiv.org/abs/2311.01234",
            published_at=published_at,
            metadata=metadata or {},
        )

    def test_arxiv_announced_day_pulls_gate_earlier_than_claimed(self):
        # claimed (HF publishedAt) is today; arXiv announced day is 2023 -> gate must use the 2023 day
        item = self._item(
            published_at="2026-06-02T09:00:00Z",
            metadata={"arxiv_id": "2311.01234", "arxiv_announced_day": "2023-11-14"},
        )
        item.verified_first_date = "2023-11-14T00:00:00Z"
        self.assertEqual(gate_date(item), date(2023, 11, 14))

    def test_anchor_never_pulls_gate_later(self):
        # verified_first_date earlier than the anchor -> earliest-min keeps the earlier verified date
        item = self._item(
            published_at="2026-06-02T09:00:00Z",
            metadata={"arxiv_id": "2311.01234", "arxiv_announced_day": "2023-11-20"},
        )
        item.verified_first_date = "2023-11-14T00:00:00Z"
        self.assertEqual(gate_date(item), date(2023, 11, 14))

    def test_crossref_registration_day_used_when_doi_present(self):
        item = self._item(
            published_at="2026-06-02T09:00:00Z",
            metadata={"doi": "10.1234/x", "crossref_registered_day": "2024-02-01"},
        )
        item.verified_first_date = None
        self.assertEqual(gate_date(item), date(2024, 2, 1))


if __name__ == "__main__":
    unittest.main()
