"""Tests for reuse_hf_daily and reuse_ainews adapters.

Key contract assertions per spec §D.1/§D.2:
  - provenance="reuse:<name>"  (stamped by build_reuse_item)
  - verified_first_date is None  (DateVerify is downstream, never source-set)
  - Recall-first: low-upvote HF papers are KEPT (no quality gate in reuse adapter)
  - Degrades to [] on fetch failure, never crashes
  - All HTTP is mocked; no network calls.
"""
from __future__ import annotations

import json
import textwrap
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from arxiv_assistant.apis.hotspot import reuse_ainews, reuse_hf_daily
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_TARGET_DATE = datetime(2026, 6, 2, 12, 0, 0, tzinfo=timezone.utc)
_FRESHNESS_HOURS = 30


# ---------------------------------------------------------------------------
# Sample HF Daily Papers HTML (JSON-island payload).
# The native hotspot_hf_papers adapter drops upvotes < 5; the reuse adapter
# must keep ALL papers — including the one with upvotes=1 below.
# ---------------------------------------------------------------------------

def _make_hf_html(papers: list[dict]) -> str:
    """Wrap a list of paper dicts in the DailyPapers JSON-island HTML fragment."""
    import html as _html
    payload = json.dumps({"dailyPapers": papers})
    escaped = _html.escape(payload, quote=True)
    return f'<div data-target="DailyPapers" data-props="{escaped}"></div>'


_HF_PAPER_HIGH_UPVOTES = {
    "paper": {
        "id": "2406.00001",
        "title": "High Upvote Paper",
        "summary": "Great paper with lots of upvotes.",
        "publishedAt": "2026-06-02T08:00:00.000Z",
        "upvotes": 50,
        "authors": [{"name": "Alice"}],
        "ai_keywords": ["LLM", "reasoning"],
    }
}

# This paper has only 1 upvote — the native adapter (MIN_UPVOTES=5) would skip it,
# but the reuse adapter must keep it (recall-first).
_HF_PAPER_LOW_UPVOTES = {
    "paper": {
        "id": "2406.00002",
        "title": "Low Upvote Paper",
        "summary": "A niche paper that is not yet popular.",
        "publishedAt": "2026-06-02T06:00:00.000Z",
        "upvotes": 1,
        "authors": [{"name": "Bob"}],
        "ai_keywords": [],
    }
}

# Paper with no publishedAt date — must still be included (recall-first; DateVerify handles this)
_HF_PAPER_NO_DATE = {
    "paper": {
        "id": "2406.00003",
        "title": "Paper Without Date",
        "summary": "This paper lacks a publication date.",
        "upvotes": 0,
        "authors": [],
        "ai_keywords": [],
    }
}

_HF_HTML_ALL = _make_hf_html([_HF_PAPER_HIGH_UPVOTES, _HF_PAPER_LOW_UPVOTES, _HF_PAPER_NO_DATE])
_HF_HTML_ONE_HIGH = _make_hf_html([_HF_PAPER_HIGH_UPVOTES])
_HF_HTML_EMPTY = "<html><body>No papers today.</body></html>"


# ---------------------------------------------------------------------------
# Sample AINews RSS feed
# ---------------------------------------------------------------------------

_AINEWS_RSS = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>AINews</title>
        <link>https://news.smol.ai</link>
        <description>AI News Recap</description>
        <item>
          <title>AINews Issue: Big LLM Week</title>
          <link>https://news.smol.ai/issues/big-llm-week</link>
          <description>This week in AI: new models, benchmarks, and more.</description>
          <pubDate>Mon, 02 Jun 2026 10:00:00 +0000</pubDate>
        </item>
        <item>
          <title>AINews Issue: Agents Taking Over</title>
          <link>https://news.smol.ai/issues/agents-taking-over</link>
          <description>Agentic systems are everywhere this week.</description>
          <pubDate>Mon, 02 Jun 2026 08:00:00 +0000</pubDate>
        </item>
        <item>
          <title>Old AINews Issue From 2020</title>
          <link>https://news.smol.ai/issues/old-2020</link>
          <description>Stale content from 2020.</description>
          <pubDate>Sat, 01 Jan 2020 00:00:00 +0000</pubDate>
        </item>
      </channel>
    </rss>
""")

_AINEWS_RSS_WITH_CONTENT = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">
      <channel>
        <title>AINews</title>
        <link>https://news.smol.ai</link>
        <item>
          <title>AINews with HTML Content</title>
          <link>https://news.smol.ai/issues/html-content</link>
          <content:encoded><![CDATA[<p>This is <strong>HTML content</strong> that should be stripped.</p>]]></content:encoded>
          <pubDate>Mon, 02 Jun 2026 09:00:00 +0000</pubDate>
        </item>
      </channel>
    </rss>
""")


# ===========================================================================
# Tests for reuse_hf_daily
# ===========================================================================

class TestReuseHfDailyModule(unittest.TestCase):
    """Module-level sanity checks."""

    def test_module_has_fetch_hotspot_items(self) -> None:
        self.assertTrue(hasattr(reuse_hf_daily, "fetch_hotspot_items"))

    def test_reuse_name_constant(self) -> None:
        self.assertEqual(reuse_hf_daily.REUSE_NAME, "hf_daily")


_HF_FETCH_PATCH = "arxiv_assistant.apis.hotspot.reuse_hf_daily.fetch_text"


class TestReuseHfDailyFetch(unittest.TestCase):
    """fetch_hotspot_items behaviour.

    All HTTP mocked at the adapter's own fetch_text binding to avoid network calls.
    """

    @patch(_HF_FETCH_PATCH, return_value=_HF_HTML_ALL)
    def test_returns_list_of_hotspot_items(self, _mock) -> None:
        items = reuse_hf_daily.fetch_hotspot_items(_TARGET_DATE, _FRESHNESS_HOURS)
        self.assertIsInstance(items, list)
        for item in items:
            self.assertIsInstance(item, HotspotItem)

    @patch(_HF_FETCH_PATCH, return_value=_HF_HTML_ALL)
    def test_provenance_is_reuse_hf_daily(self, _mock) -> None:
        items = reuse_hf_daily.fetch_hotspot_items(_TARGET_DATE, _FRESHNESS_HOURS)
        self.assertGreater(len(items), 0)
        for item in items:
            self.assertEqual(item.provenance, "reuse:hf_daily")

    @patch(_HF_FETCH_PATCH, return_value=_HF_HTML_ALL)
    def test_verified_first_date_is_none(self, _mock) -> None:
        """DateVerify is downstream; reuse adapter must NOT set verified_first_date."""
        items = reuse_hf_daily.fetch_hotspot_items(_TARGET_DATE, _FRESHNESS_HOURS)
        self.assertGreater(len(items), 0)
        for item in items:
            self.assertIsNone(item.verified_first_date)

    @patch(_HF_FETCH_PATCH, return_value=_HF_HTML_ALL)
    def test_recall_first_low_upvote_paper_is_kept(self, _mock) -> None:
        """Core recall-first contract: upvotes=1 paper must appear.

        The native hotspot_hf_papers adapter filters out upvotes < MIN_UPVOTES (5).
        The reuse adapter must NOT apply that cutoff — it's for breadth/recall.
        """
        items = reuse_hf_daily.fetch_hotspot_items(_TARGET_DATE, _FRESHNESS_HOURS)
        arxiv_ids = [item.metadata.get("arxiv_id") for item in items]
        self.assertIn(
            "2406.00002",
            arxiv_ids,
            msg="Low-upvote paper (upvotes=1) must be included by recall-first reuse adapter",
        )

    @patch(_HF_FETCH_PATCH, return_value=_HF_HTML_ALL)
    def test_recall_first_no_date_paper_is_kept(self, _mock) -> None:
        """A paper without publishedAt must still be included by the reuse adapter.

        The native adapter skips papers without a date; the reuse adapter must not.
        """
        items = reuse_hf_daily.fetch_hotspot_items(_TARGET_DATE, _FRESHNESS_HOURS)
        arxiv_ids = [item.metadata.get("arxiv_id") for item in items]
        self.assertIn(
            "2406.00003",
            arxiv_ids,
            msg="Paper without publishedAt must be included by recall-first reuse adapter",
        )

    @patch(_HF_FETCH_PATCH, return_value=_HF_HTML_ALL)
    def test_high_upvote_paper_also_included(self, _mock) -> None:
        items = reuse_hf_daily.fetch_hotspot_items(_TARGET_DATE, _FRESHNESS_HOURS)
        arxiv_ids = [item.metadata.get("arxiv_id") for item in items]
        self.assertIn("2406.00001", arxiv_ids)

    @patch(_HF_FETCH_PATCH, return_value=_HF_HTML_ALL)
    def test_upvotes_stored_in_metadata(self, _mock) -> None:
        items = reuse_hf_daily.fetch_hotspot_items(_TARGET_DATE, _FRESHNESS_HOURS)
        by_id = {item.metadata["arxiv_id"]: item for item in items}
        self.assertEqual(by_id["2406.00001"].metadata["upvotes"], 50)
        self.assertEqual(by_id["2406.00002"].metadata["upvotes"], 1)

    @patch(_HF_FETCH_PATCH, return_value=_HF_HTML_ALL)
    def test_canonical_url_is_arxiv(self, _mock) -> None:
        items = reuse_hf_daily.fetch_hotspot_items(_TARGET_DATE, _FRESHNESS_HOURS)
        for item in items:
            arxiv_id = item.metadata.get("arxiv_id")
            self.assertEqual(item.canonical_url, f"https://arxiv.org/abs/{arxiv_id}")

    @patch(_HF_FETCH_PATCH, return_value=_HF_HTML_ALL)
    def test_source_id_is_reuse_hf_daily(self, _mock) -> None:
        items = reuse_hf_daily.fetch_hotspot_items(_TARGET_DATE, _FRESHNESS_HOURS)
        for item in items:
            self.assertEqual(item.source_id, "reuse_hf_daily")

    @patch(_HF_FETCH_PATCH, return_value=_HF_HTML_ALL)
    def test_source_type_is_reuse(self, _mock) -> None:
        items = reuse_hf_daily.fetch_hotspot_items(_TARGET_DATE, _FRESHNESS_HOURS)
        for item in items:
            self.assertEqual(item.source_type, "reuse")

    @patch(_HF_FETCH_PATCH, return_value=_HF_HTML_ALL)
    def test_source_role_is_trusted_research(self, _mock) -> None:
        items = reuse_hf_daily.fetch_hotspot_items(_TARGET_DATE, _FRESHNESS_HOURS)
        for item in items:
            self.assertEqual(item.source_role, "trusted_research")

    @patch(_HF_FETCH_PATCH, return_value=_HF_HTML_ALL)
    def test_result_limit_honoured(self, _mock) -> None:
        items = reuse_hf_daily.fetch_hotspot_items(_TARGET_DATE, _FRESHNESS_HOURS, result_limit=2)
        self.assertLessEqual(len(items), 2)

    @patch(_HF_FETCH_PATCH, return_value=_HF_HTML_ALL)
    def test_authors_passed_through(self, _mock) -> None:
        items = reuse_hf_daily.fetch_hotspot_items(_TARGET_DATE, _FRESHNESS_HOURS)
        by_id = {item.metadata["arxiv_id"]: item for item in items}
        self.assertIn("Alice", by_id["2406.00001"].authors)

    @patch(_HF_FETCH_PATCH, return_value=_HF_HTML_ALL)
    def test_tags_passed_through(self, _mock) -> None:
        items = reuse_hf_daily.fetch_hotspot_items(_TARGET_DATE, _FRESHNESS_HOURS)
        by_id = {item.metadata["arxiv_id"]: item for item in items}
        self.assertIn("LLM", by_id["2406.00001"].tags)

    @patch(_HF_FETCH_PATCH, return_value=_HF_HTML_EMPTY)
    def test_empty_page_returns_empty_list(self, _mock) -> None:
        items = reuse_hf_daily.fetch_hotspot_items(_TARGET_DATE, _FRESHNESS_HOURS)
        self.assertEqual(items, [])

    @patch(_HF_FETCH_PATCH, side_effect=Exception("connection refused"))
    def test_both_fetches_fail_returns_empty_list(self, _mock) -> None:
        """Both date-specific and fallback fetches failing must degrade to [] (spec §E)."""
        items = reuse_hf_daily.fetch_hotspot_items(_TARGET_DATE, _FRESHNESS_HOURS)
        self.assertEqual(items, [])

    def test_fallback_to_trending_url_on_date_fetch_failure(self) -> None:
        """On date-URL failure, trending URL is tried; on its success items are returned."""
        def selective_fail(url: str) -> str:
            from arxiv_assistant.apis.hotspot.hotspot_hf_papers import HF_DATE_URL
            if url == HF_DATE_URL.format(date=_TARGET_DATE.strftime("%Y-%m-%d")):
                raise ConnectionError("date URL failed")
            return _HF_HTML_ONE_HIGH  # trending URL succeeds

        with patch(_HF_FETCH_PATCH, side_effect=selective_fail):
            items = reuse_hf_daily.fetch_hotspot_items(_TARGET_DATE, _FRESHNESS_HOURS)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].metadata["arxiv_id"], "2406.00001")


# ===========================================================================
# Tests for reuse_ainews
# ===========================================================================

class TestReuseAinewsModule(unittest.TestCase):
    """Module-level sanity checks."""

    def test_module_has_fetch_hotspot_items(self) -> None:
        self.assertTrue(hasattr(reuse_ainews, "fetch_hotspot_items"))

    def test_reuse_name_constant(self) -> None:
        self.assertEqual(reuse_ainews.REUSE_NAME, "ainews")


class TestReuseAinewsFetch(unittest.TestCase):
    """fetch_hotspot_items behaviour."""

    @patch("arxiv_assistant.apis.hotspot.reuse_common.fetch_text", return_value=_AINEWS_RSS)
    def test_returns_list_of_hotspot_items(self, _mock) -> None:
        items = reuse_ainews.fetch_hotspot_items(_TARGET_DATE, _FRESHNESS_HOURS)
        self.assertIsInstance(items, list)
        for item in items:
            self.assertIsInstance(item, HotspotItem)

    @patch("arxiv_assistant.apis.hotspot.reuse_common.fetch_text", return_value=_AINEWS_RSS)
    def test_provenance_is_reuse_ainews(self, _mock) -> None:
        items = reuse_ainews.fetch_hotspot_items(_TARGET_DATE, _FRESHNESS_HOURS)
        self.assertGreater(len(items), 0)
        for item in items:
            self.assertEqual(item.provenance, "reuse:ainews")

    @patch("arxiv_assistant.apis.hotspot.reuse_common.fetch_text", return_value=_AINEWS_RSS)
    def test_verified_first_date_is_none(self, _mock) -> None:
        """DateVerify is downstream; verified_first_date must stay None."""
        items = reuse_ainews.fetch_hotspot_items(_TARGET_DATE, _FRESHNESS_HOURS)
        self.assertGreater(len(items), 0)
        for item in items:
            self.assertIsNone(item.verified_first_date)

    @patch("arxiv_assistant.apis.hotspot.reuse_common.fetch_text", return_value=_AINEWS_RSS)
    def test_fresh_items_returned_stale_filtered(self, _mock) -> None:
        """The 2020 item must be filtered by freshness; 2026 items returned."""
        items = reuse_ainews.fetch_hotspot_items(_TARGET_DATE, _FRESHNESS_HOURS)
        urls = [item.url for item in items]
        self.assertNotIn("https://news.smol.ai/issues/old-2020", urls)
        # At least one of the 2026 items should be present
        self.assertTrue(
            any("big-llm-week" in u or "agents-taking-over" in u for u in urls),
            msg="Fresh 2026 AINews issues should be returned",
        )

    @patch("arxiv_assistant.apis.hotspot.reuse_common.fetch_text", return_value=_AINEWS_RSS)
    def test_source_id_is_reuse_ainews(self, _mock) -> None:
        items = reuse_ainews.fetch_hotspot_items(_TARGET_DATE, _FRESHNESS_HOURS)
        for item in items:
            self.assertEqual(item.source_id, "reuse_ainews")

    @patch("arxiv_assistant.apis.hotspot.reuse_common.fetch_text", return_value=_AINEWS_RSS)
    def test_source_type_is_reuse(self, _mock) -> None:
        items = reuse_ainews.fetch_hotspot_items(_TARGET_DATE, _FRESHNESS_HOURS)
        for item in items:
            self.assertEqual(item.source_type, "reuse")

    @patch("arxiv_assistant.apis.hotspot.reuse_common.fetch_text", return_value=_AINEWS_RSS)
    def test_source_role_is_community_signal(self, _mock) -> None:
        items = reuse_ainews.fetch_hotspot_items(_TARGET_DATE, _FRESHNESS_HOURS)
        for item in items:
            self.assertEqual(item.source_role, "community_signal")

    @patch("arxiv_assistant.apis.hotspot.reuse_common.fetch_text", return_value=_AINEWS_RSS)
    def test_freshness_window_widened_to_36h(self, _mock) -> None:
        """AINews publishes weekdays only; effective freshness is widened to max(hours, 36)."""
        # With freshness_hours=1 (too short), effective should become 36
        # so items published 30h ago should still be in range.
        short_freshness = 1
        # With short_freshness=1, effective becomes 36, so the Jun 02 items (within 36h) pass.
        items = reuse_ainews.fetch_hotspot_items(_TARGET_DATE, short_freshness)
        # Items exist because effective window is widened to 36h
        self.assertGreater(len(items), 0)

    @patch("arxiv_assistant.apis.hotspot.reuse_common.fetch_text", return_value=_AINEWS_RSS)
    def test_result_limit_honoured(self, _mock) -> None:
        items = reuse_ainews.fetch_hotspot_items(_TARGET_DATE, _FRESHNESS_HOURS, result_limit=1)
        self.assertLessEqual(len(items), 1)

    @patch("arxiv_assistant.apis.hotspot.reuse_common.fetch_text",
           side_effect=Exception("network error"))
    def test_fetch_failure_returns_empty_list(self, _mock) -> None:
        """Fetch failure must degrade to [] without crashing (spec §E)."""
        items = reuse_ainews.fetch_hotspot_items(_TARGET_DATE, _FRESHNESS_HOURS)
        self.assertEqual(items, [])

    @patch("arxiv_assistant.apis.hotspot.reuse_common.fetch_text", return_value="not valid xml <<<>>>")
    def test_malformed_feed_returns_empty_list(self, _mock) -> None:
        items = reuse_ainews.fetch_hotspot_items(_TARGET_DATE, _FRESHNESS_HOURS)
        self.assertEqual(items, [])

    def test_summary_helper_strips_html(self) -> None:
        """_summary() should return plain text with HTML tags removed."""
        class FakeEntry(dict):
            pass

        entry = FakeEntry({"summary": "<p>Hello <strong>world</strong></p>"})
        result = reuse_ainews._summary(entry)
        self.assertNotIn("<", result)
        self.assertIn("Hello", result)
        self.assertIn("world", result)

    def test_summary_helper_prefers_content_over_summary(self) -> None:
        """_summary() should prefer content[0].value over summary when both present."""
        entry = {
            "content": [{"value": "<p>Content value</p>"}],
            "summary": "Summary value",
        }
        result = reuse_ainews._summary(entry)
        self.assertIn("Content value", result)
        self.assertNotIn("Summary value", result)

    def test_summary_helper_falls_back_to_description(self) -> None:
        """_summary() falls back to description when content and summary are absent."""
        entry = {"description": "<p>Description value</p>"}
        result = reuse_ainews._summary(entry)
        self.assertIn("Description value", result)


if __name__ == "__main__":
    unittest.main()
