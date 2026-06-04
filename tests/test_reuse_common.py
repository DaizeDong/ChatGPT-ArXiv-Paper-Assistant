from __future__ import annotations

import textwrap
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from arxiv_assistant.apis.hotspot.reuse_common import (
    REUSE_SOURCE_TIER_ANCHOR,
    build_reuse_item,
    harvest_rss_reuse,
    reuse_source_role,
)
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem

# ---------------------------------------------------------------------------
# Sample RSS feed for mocked fetch_text responses.
# ---------------------------------------------------------------------------
_SAMPLE_RSS = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>AI News Feed</title>
        <link>https://example.com</link>
        <description>Test feed</description>
        <item>
          <title>LLM Breakthrough Announced</title>
          <link>https://example.com/llm-breakthrough</link>
          <description>A new LLM has been released with record benchmark scores.</description>
          <pubDate>Mon, 02 Jun 2026 10:00:00 +0000</pubDate>
        </item>
        <item>
          <title>Open Source Model Release</title>
          <link>https://example.com/open-source-model</link>
          <description>New open-source model published on HuggingFace.</description>
          <pubDate>Mon, 02 Jun 2026 08:30:00 +0000</pubDate>
        </item>
        <item>
          <title>Old Paper From 2020</title>
          <link>https://example.com/old-paper-2020</link>
          <description>This paper is from 2020 and should be filtered out by freshness.</description>
          <pubDate>Sat, 01 Jan 2020 00:00:00 +0000</pubDate>
        </item>
      </channel>
    </rss>
""")

# A deliberately malformed / empty RSS snippet to test bozo-feed handling.
_EMPTY_BOZO_RSS = "not valid xml at all <<<>>>"

# Target date: 2026-06-02T12:00:00+00:00 — the two June items are within 30h freshness.
_TARGET_DATE = datetime(2026, 6, 2, 12, 0, 0, tzinfo=timezone.utc)
_FRESHNESS_HOURS = 30


# ---------------------------------------------------------------------------
# reuse_source_role
# ---------------------------------------------------------------------------

class TestReuseSourceRole(unittest.TestCase):
    def test_known_sources_map_to_expected_tiers(self) -> None:
        expected = {
            "hf_daily": "trusted_research",
            "ainews": "community_signal",
            "agents_radar": "builder_ecosystem",
            "horizon": "trusted_analysis",
            "scholar_inbox": "trusted_research",
        }
        for name, tier in expected.items():
            with self.subTest(name=name):
                self.assertEqual(reuse_source_role(name), tier)

    def test_unknown_source_falls_back_to_community_signal(self) -> None:
        self.assertEqual(reuse_source_role("totally_new_source"), "community_signal")

    def test_openalex_maps_to_trusted_research(self) -> None:
        self.assertEqual(reuse_source_role("openalex"), "trusted_research")

    def test_all_anchor_names_resolve_to_a_known_tier(self) -> None:
        # Every anchor in REUSE_SOURCE_TIER_ANCHOR must map to a real tier in source_tiers.json,
        # not just the generic "community_signal" fallback used for *unrecognised* sources.
        # (Note: ainews legitimately belongs to community_signal; the fallback is only for sources
        # with NO anchor entry at all.  We verify the anchor itself exists in source_tiers.json.)
        import json
        from pathlib import Path
        tiers_path = Path(__file__).resolve().parents[1] / "configs" / "hotspot" / "source_tiers.json"
        tier_map = json.loads(tiers_path.read_text(encoding="utf-8")).get("source_id_to_tier", {})
        for reuse_name, anchor in REUSE_SOURCE_TIER_ANCHOR.items():
            self.assertIn(anchor, tier_map,
                          msg=f"{reuse_name!r} anchor {anchor!r} not found in source_tiers.json source_id_to_tier")


# ---------------------------------------------------------------------------
# build_reuse_item
# ---------------------------------------------------------------------------

class TestBuildReuseItem(unittest.TestCase):
    def _make(self, reuse_name: str = "ainews", **kwargs) -> HotspotItem:
        defaults = dict(
            title="A test story",
            url="https://example.com/story",
            summary="A brief summary of the test story.",
            published_at="2026-06-02T10:00:00+00:00",
        )
        defaults.update(kwargs)
        return build_reuse_item(reuse_name, **defaults)

    def test_provenance_is_reuse_colon_name(self) -> None:
        item = self._make("ainews")
        self.assertEqual(item.provenance, "reuse:ainews")

    def test_source_id_is_reuse_underscore_name(self) -> None:
        item = self._make("hf_daily")
        self.assertEqual(item.source_id, "reuse_hf_daily")

    def test_source_type_is_reuse(self) -> None:
        item = self._make("ainews")
        self.assertEqual(item.source_type, "reuse")

    def test_source_role_matches_tier_for_hf_daily(self) -> None:
        item = self._make("hf_daily")
        self.assertEqual(item.source_role, "trusted_research")

    def test_source_role_matches_tier_for_ainews(self) -> None:
        item = self._make("ainews")
        self.assertEqual(item.source_role, "community_signal")

    def test_source_role_matches_tier_for_agents_radar(self) -> None:
        item = self._make("agents_radar")
        self.assertEqual(item.source_role, "builder_ecosystem")

    def test_source_role_matches_tier_for_horizon(self) -> None:
        item = self._make("horizon")
        self.assertEqual(item.source_role, "trusted_analysis")

    def test_canonical_url_defaults_to_url_when_not_provided(self) -> None:
        item = self._make(url="https://example.com/story")
        self.assertEqual(item.canonical_url, "https://example.com/story")

    def test_canonical_url_overrides_url_when_provided(self) -> None:
        item = self._make(
            url="https://huggingface.co/papers/2406.00001",
            canonical_url="https://arxiv.org/abs/2406.00001",
        )
        self.assertEqual(item.canonical_url, "https://arxiv.org/abs/2406.00001")

    def test_summary_is_clipped_to_520_chars(self) -> None:
        long_summary = "x" * 600
        item = self._make(summary=long_summary)
        self.assertLessEqual(len(item.summary), 520)

    def test_metadata_contains_reuse_name_and_host(self) -> None:
        item = self._make("ainews", url="https://news.smol.ai/issue/123")
        self.assertEqual(item.metadata["reuse_name"], "ainews")
        self.assertEqual(item.metadata["host"], "news.smol.ai")

    def test_extra_metadata_merged_into_metadata(self) -> None:
        item = self._make("hf_daily", extra_metadata={"arxiv_id": "2406.00001", "upvotes": 42})
        self.assertEqual(item.metadata["arxiv_id"], "2406.00001")
        self.assertEqual(item.metadata["upvotes"], 42)

    def test_tags_and_authors_default_to_empty_lists(self) -> None:
        item = self._make()
        self.assertEqual(item.tags, [])
        self.assertEqual(item.authors, [])

    def test_tags_and_authors_passed_through(self) -> None:
        item = self._make(tags=["llm", "agents"], authors=["Alice", "Bob"])
        self.assertIn("llm", item.tags)
        self.assertIn("Alice", item.authors)

    def test_provenance_not_set_by_date_verify_here(self) -> None:
        # DateVerify/max_age gate is downstream; verified_first_date stays None here.
        item = self._make("ainews")
        self.assertIsNone(item.verified_first_date)

    def test_source_name_is_reuse_colon_name(self) -> None:
        item = self._make("horizon")
        self.assertEqual(item.source_name, "Reuse:horizon")


# ---------------------------------------------------------------------------
# harvest_rss_reuse
# ---------------------------------------------------------------------------

class TestHarvestRssReuse(unittest.TestCase):
    @patch("arxiv_assistant.apis.hotspot.reuse_common.fetch_text", return_value=_SAMPLE_RSS)
    def test_returns_fresh_items_only(self, _mock_fetch) -> None:
        items = harvest_rss_reuse("ainews", "https://example.com/rss", _TARGET_DATE, _FRESHNESS_HOURS)
        # Only the two June 2026 items are within the 30h window; 2020 item is stale.
        self.assertEqual(len(items), 2)

    @patch("arxiv_assistant.apis.hotspot.reuse_common.fetch_text", return_value=_SAMPLE_RSS)
    def test_items_have_correct_provenance(self, _mock_fetch) -> None:
        items = harvest_rss_reuse("ainews", "https://example.com/rss", _TARGET_DATE, _FRESHNESS_HOURS)
        for item in items:
            self.assertEqual(item.provenance, "reuse:ainews")

    @patch("arxiv_assistant.apis.hotspot.reuse_common.fetch_text", return_value=_SAMPLE_RSS)
    def test_items_have_correct_source_role(self, _mock_fetch) -> None:
        items = harvest_rss_reuse("ainews", "https://example.com/rss", _TARGET_DATE, _FRESHNESS_HOURS)
        for item in items:
            self.assertEqual(item.source_role, "community_signal")

    @patch("arxiv_assistant.apis.hotspot.reuse_common.fetch_text", return_value=_SAMPLE_RSS)
    def test_result_limit_is_honoured(self, _mock_fetch) -> None:
        items = harvest_rss_reuse("ainews", "https://example.com/rss", _TARGET_DATE, _FRESHNESS_HOURS, result_limit=1)
        self.assertEqual(len(items), 1)

    @patch("arxiv_assistant.apis.hotspot.reuse_common.fetch_text", return_value=_SAMPLE_RSS)
    def test_deduplication_by_url(self, _mock_fetch) -> None:
        # Feed with duplicate URLs: only one item should be kept.
        duplicate_rss = textwrap.dedent("""\
            <?xml version="1.0" encoding="UTF-8"?>
            <rss version="2.0">
              <channel>
                <title>Dup Feed</title>
                <item>
                  <title>First Mention</title>
                  <link>https://example.com/same-url</link>
                  <pubDate>Mon, 02 Jun 2026 10:00:00 +0000</pubDate>
                </item>
                <item>
                  <title>Second Mention Same URL</title>
                  <link>https://example.com/same-url</link>
                  <pubDate>Mon, 02 Jun 2026 11:00:00 +0000</pubDate>
                </item>
              </channel>
            </rss>
        """)
        with patch("arxiv_assistant.apis.hotspot.reuse_common.fetch_text", return_value=duplicate_rss):
            items = harvest_rss_reuse("ainews", "https://example.com/rss", _TARGET_DATE, _FRESHNESS_HOURS)
        self.assertEqual(len(items), 1)

    @patch("arxiv_assistant.apis.hotspot.reuse_common.fetch_text", side_effect=Exception("network error"))
    def test_fetch_failure_returns_empty_list(self, _mock_fetch) -> None:
        items = harvest_rss_reuse("ainews", "https://example.com/rss", _TARGET_DATE, _FRESHNESS_HOURS)
        self.assertEqual(items, [])

    @patch("arxiv_assistant.apis.hotspot.reuse_common.fetch_text", return_value=_EMPTY_BOZO_RSS)
    def test_bozo_feed_with_no_entries_returns_empty_list(self, _mock_fetch) -> None:
        items = harvest_rss_reuse("ainews", "https://example.com/rss", _TARGET_DATE, _FRESHNESS_HOURS)
        self.assertEqual(items, [])

    @patch("arxiv_assistant.apis.hotspot.reuse_common.fetch_text", return_value=_SAMPLE_RSS)
    def test_custom_summary_of_callable_is_used(self, _mock_fetch) -> None:
        def custom_summary(entry):
            return f"CUSTOM:{entry.get('title', '')}"

        items = harvest_rss_reuse(
            "ainews", "https://example.com/rss", _TARGET_DATE, _FRESHNESS_HOURS,
            summary_of=custom_summary,
        )
        for item in items:
            self.assertTrue(item.summary.startswith("CUSTOM:"))

    @patch("arxiv_assistant.apis.hotspot.reuse_common.fetch_text", return_value=_SAMPLE_RSS)
    def test_titles_cleaned(self, _mock_fetch) -> None:
        items = harvest_rss_reuse("ainews", "https://example.com/rss", _TARGET_DATE, _FRESHNESS_HOURS)
        for item in items:
            # clean_text applied: no leading/trailing whitespace, no double spaces
            self.assertEqual(item.title, item.title.strip())
            self.assertNotIn("  ", item.title)

    @patch("arxiv_assistant.apis.hotspot.reuse_common.fetch_text", return_value=_SAMPLE_RSS)
    def test_verified_first_date_not_set_here(self, _mock_fetch) -> None:
        # DateVerify is downstream; reuse_common never stamps verified_first_date.
        items = harvest_rss_reuse("ainews", "https://example.com/rss", _TARGET_DATE, _FRESHNESS_HOURS)
        for item in items:
            self.assertIsNone(item.verified_first_date)


if __name__ == "__main__":
    unittest.main()
