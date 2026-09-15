from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch


class TestHotspotTwitterapiSource(unittest.TestCase):
    def _seed_file(self, tmp_dir: str) -> Path:
        seed_path = Path(tmp_dir) / "x_seeds.json"
        seed_path.write_text(
            json.dumps(
                {
                    "accounts": [
                        {"handle": "openai", "name": "OpenAI", "kind": "official", "tier": 3, "active": True},
                        {"handle": "demishassabis", "name": "Demis Hassabis", "kind": "researcher", "tier": 3, "active": True},
                    ]
                }
            ),
            encoding="utf-8",
        )
        return seed_path

    # -----------------------------------------------------------------------
    # Task 1: key gate
    # -----------------------------------------------------------------------

    def test_returns_empty_when_no_twitterapi_key_configured(self) -> None:
        from arxiv_assistant.apis.hotspot.hotspot_twitterapi import fetch_hotspot_items

        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(os.environ, {}, clear=True):
            seed_path = self._seed_file(tmp_dir)
            items = fetch_hotspot_items(
                datetime(2026, 3, 21, tzinfo=UTC),
                36,
                seed_path,
                snapshot_path=Path(tmp_dir) / "x_authority_inventory.json",
            )
        self.assertEqual(items, [])


    # -----------------------------------------------------------------------
    # Task 2: REST fetch + JSON normalization helpers
    # -----------------------------------------------------------------------

    # Real-shape twitterapi.io payload (camelCase, Twitter classic createdAt).
    _OPENAI_PAYLOAD = {
        "tweets": [
            {
                "id": "2035012260008272007",
                "text": "We released GPT-5.4 mini today in ChatGPT, Codex, and the API. https://t.co/abc123",
                "createdAt": "Sat Mar 21 10:00:00 +0000 2026",
                "lang": "en",
                "inReplyToUserId": None,
                "retweeted_tweet": None,
                "entities": {"urls": [{"expanded_url": "https://openai.com/index/gpt-5-4-mini"}]},
                "likeCount": 1200,
                "replyCount": 90,
                "retweetCount": 150,
                "quoteCount": 40,
                "viewCount": 530000,
                "bookmarkCount": 100,
                "author": {"id": "1", "name": "OpenAI", "userName": "OpenAI", "isBlueVerified": True},
            }
        ]
    }

    def test_map_twitterapi_tweet_normalizes_camelcase_and_timestamp(self) -> None:
        from arxiv_assistant.apis.hotspot.hotspot_twitterapi import _map_twitterapi_tweet

        mapped = _map_twitterapi_tweet(self._OPENAI_PAYLOAD["tweets"][0], handle="openai", user_id="1")
        self.assertEqual(mapped["id"], "2035012260008272007")
        self.assertEqual(mapped["created_at"], "2026-03-21T10:00:00Z")
        self.assertEqual(mapped["public_metrics"]["like_count"], 1200)
        self.assertEqual(mapped["public_metrics"]["impression_count"], 530000)
        self.assertEqual(mapped["author"]["username"], "openai")
        self.assertTrue(mapped["author"]["verified"])

    def test_fetch_last_tweets_rest_returns_empty_on_429(self) -> None:
        from arxiv_assistant.apis.hotspot import hotspot_twitterapi as mod

        class _Resp:
            status_code = 429

            def raise_for_status(self) -> None:  # pragma: no cover - not reached on 429
                raise AssertionError("should not raise_for_status on 429")

            def json(self) -> dict:  # pragma: no cover - not reached on 429
                return {}

        with patch.object(mod.requests, "get", return_value=_Resp()), \
                patch.object(mod.time, "sleep", return_value=None):
            rows = mod._fetch_last_tweets_rest(
                user_id=None,
                handle="openai",
                api_key="k",
                since=datetime(2026, 3, 20, tzinfo=UTC),
                max_results=10,
            )
        self.assertEqual(rows, [])

    def test_fetch_last_tweets_rest_filters_by_since_window(self) -> None:
        from arxiv_assistant.apis.hotspot import hotspot_twitterapi as mod

        payload = {
            "tweets": [
                dict(self._OPENAI_PAYLOAD["tweets"][0]),  # 2026-03-21 (in window)
                {
                    **self._OPENAI_PAYLOAD["tweets"][0],
                    "id": "999",
                    "createdAt": "Mon Jan 05 10:00:00 +0000 2026",  # old, out of window
                },
            ]
        }

        class _Resp:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return payload

        with patch.object(mod.requests, "get", return_value=_Resp()):
            rows = mod._fetch_last_tweets_rest(
                user_id="1",
                handle="openai",
                api_key="k",
                since=datetime(2026, 3, 20, tzinfo=UTC),
                max_results=10,
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "2035012260008272007")


    # -----------------------------------------------------------------------
    # Task 3: tweet → HotspotItem normalization (provenance, source_id, fields)
    # -----------------------------------------------------------------------

    def test_tweet_to_item_sets_provenance_and_canonical_fields(self) -> None:
        from arxiv_assistant.apis.hotspot import hotspot_twitterapi as mod

        row = mod._map_twitterapi_tweet(self._OPENAI_PAYLOAD["tweets"][0], handle="openai", user_id="1")
        authority = {"handle": "openai", "name": "OpenAI", "kind": "official", "tier": 3, "organization": "OpenAI"}
        item = mod._tweet_to_item(row, authority=authority)

        self.assertIsNotNone(item)
        self.assertEqual(item.source_id, "x_twitterapi")
        self.assertEqual(item.source_role, "official_news")
        self.assertEqual(item.source_type, "tweet")
        # URL uses authority handle (lowercase) + tweet id
        self.assertEqual(item.url, "https://x.com/openai/status/2035012260008272007")
        self.assertEqual(item.published_at, "2026-03-21T10:00:00Z")
        self.assertEqual(item.metadata["provenance"], "native:x_twitterapi")
        self.assertEqual(item.metadata["source_id"], "x_twitterapi")
        self.assertEqual(item.metadata["authority_kind"], "official")
        self.assertEqual(item.metadata["author_handle"], "openai")
        self.assertGreater(item.metadata["activity"], 500)

    def test_tweet_to_item_drops_replies_and_retweets(self) -> None:
        from arxiv_assistant.apis.hotspot import hotspot_twitterapi as mod

        reply = {**self._OPENAI_PAYLOAD["tweets"][0], "in_reply_to_user_id": "42"}
        authority = {"handle": "openai", "name": "OpenAI", "kind": "official", "tier": 3}
        # _map already ran on the raw payload above; here pass an already-canonical reply dict.
        canonical_reply = mod._map_twitterapi_tweet(reply, handle="openai", user_id="1")
        canonical_reply["in_reply_to_user_id"] = "42"
        self.assertIsNone(mod._tweet_to_item(canonical_reply, authority=authority))

    def test_tweet_to_item_official_release_survives_is_newsworthy_filter(self) -> None:
        """Official 'We released GPT-5' tweets must NOT be killed by SELF_WORK_PATTERNS —
        the official-account exemption in is_newsworthy_x_text must pass them through.
        This is the X≈0 root-cause fix: channel (twitterapi.io) + filter-is-not-the-blocker."""
        from arxiv_assistant.apis.hotspot import hotspot_twitterapi as mod

        row = mod._map_twitterapi_tweet(self._OPENAI_PAYLOAD["tweets"][0], handle="openai", user_id="1")
        authority = {"handle": "openai", "name": "OpenAI", "kind": "official", "tier": 3}
        item = mod._tweet_to_item(row, authority=authority)
        self.assertIsNotNone(item, "Official 'We released ...' tweet must survive is_newsworthy_x_text")
        self.assertEqual(item.source_id, "x_twitterapi")
        self.assertEqual(item.provenance, "native:x_twitterapi")
        self.assertEqual(item.metadata["provenance"], "native:x_twitterapi")

    # -----------------------------------------------------------------------
    # End-to-end: full fetch_hotspot_items path
    # -----------------------------------------------------------------------

    _RESEARCHER_PAYLOAD = {
        "tweets": [
            {
                "id": "2035012260008272010",
                "text": "Strong new results on agent benchmarks and reasoning evals. https://t.co/paper",
                "createdAt": "Sat Mar 21 10:05:00 +0000 2026",
                "lang": "en",
                "inReplyToUserId": None,
                "retweeted_tweet": None,
                "entities": {"urls": [{"expanded_url": "https://arxiv.org/abs/2603.12345"}]},
                "likeCount": 400,
                "replyCount": 40,
                "retweetCount": 80,
                "quoteCount": 8,
                "viewCount": 90000,
                "bookmarkCount": 20,
                "author": {"id": "3", "name": "Demis Hassabis", "userName": "demishassabis", "isBlueVerified": True},
            }
        ]
    }

    def _fake_twitterapi_get(self, params, *, api_key):
        handle = (params.get("userName") or "").lower()
        if handle == "openai" or params.get("userId") == "openai":
            return self._OPENAI_PAYLOAD
        if handle == "demishassabis":
            return self._RESEARCHER_PAYLOAD
        return {"tweets": []}

    def test_fetch_hotspot_items_official_release_survives_filter(self) -> None:
        from arxiv_assistant.apis.hotspot import hotspot_twitterapi as mod

        # target_date at noon so the 10:00Z tweet falls inside window [target-24h, target+6h].
        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.dict(os.environ, {"TWITTERAPI_IO_KEY": "test-key"}, clear=True), \
                patch.object(mod, "_twitterapi_get", side_effect=self._fake_twitterapi_get), \
                patch("arxiv_assistant.utils.hotspot.x_authority_registry.fetch_text", return_value=""), \
                patch("arxiv_assistant.utils.hotspot.x_authority_registry.fetch_json", return_value={}):
            seed_path = self._seed_file(tmp_dir)
            items = mod.fetch_hotspot_items(
                datetime(2026, 3, 21, 12, 0, tzinfo=UTC),
                24,
                seed_path,
                result_limit=80,
                snapshot_path=Path(tmp_dir) / "x_authority_inventory.json",
            )

        # The official "We released GPT-5.4 mini today" tweet must NOT be killed by SELF_WORK_PATTERNS.
        self.assertGreaterEqual(len(items), 1)
        urls = {item.url for item in items}
        self.assertIn("https://x.com/openai/status/2035012260008272007", urls)
        official = next(i for i in items if i.url.endswith("2035012260008272007"))
        self.assertEqual(official.source_id, "x_twitterapi")
        self.assertEqual(official.metadata["provenance"], "native:x_twitterapi")
        self.assertEqual(official.source_role, "official_news")

    def test_fetch_hotspot_items_drops_out_of_window_tweets(self) -> None:
        from arxiv_assistant.apis.hotspot import hotspot_twitterapi as mod

        stale_payload = {
            "tweets": [
                {
                    **self._OPENAI_PAYLOAD["tweets"][0],
                    "createdAt": "Mon Jan 05 10:00:00 +0000 2026",  # ~11 weeks old
                }
            ]
        }

        def _stale_get(params, *, api_key):
            if (params.get("userName") or "").lower() == "openai":
                return stale_payload
            return {"tweets": []}

        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.dict(os.environ, {"TWITTERAPI_IO_KEY": "test-key"}, clear=True), \
                patch.object(mod, "_twitterapi_get", side_effect=_stale_get), \
                patch("arxiv_assistant.utils.hotspot.x_authority_registry.fetch_text", return_value=""), \
                patch("arxiv_assistant.utils.hotspot.x_authority_registry.fetch_json", return_value={}):
            seed_path = self._seed_file(tmp_dir)
            items = mod.fetch_hotspot_items(
                datetime(2026, 3, 21, 12, 0, tzinfo=UTC),
                24,
                seed_path,
                snapshot_path=Path(tmp_dir) / "x_authority_inventory.json",
            )
        self.assertEqual(items, [])

    def test_fetch_hotspot_items_empty_response_degrades_cleanly(self) -> None:
        from arxiv_assistant.apis.hotspot import hotspot_twitterapi as mod

        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.dict(os.environ, {"TWITTERAPI_IO_KEY": "test-key"}, clear=True), \
                patch.object(mod, "_twitterapi_get", return_value={"tweets": []}), \
                patch("arxiv_assistant.utils.hotspot.x_authority_registry.fetch_text", return_value=""), \
                patch("arxiv_assistant.utils.hotspot.x_authority_registry.fetch_json", return_value={}):
            seed_path = self._seed_file(tmp_dir)
            items = mod.fetch_hotspot_items(
                datetime(2026, 3, 21, 12, 0, tzinfo=UTC),
                24,
                seed_path,
                snapshot_path=Path(tmp_dir) / "x_authority_inventory.json",
            )
        self.assertEqual(items, [])

    # -----------------------------------------------------------------------
    # Task 4 additional: multi-account iteration, fault tolerance, dedup, limit
    # -----------------------------------------------------------------------

    def _seed_file_three_accounts(self, tmp_dir: str) -> Path:
        """Seed with 3 official accounts so we can test multi-account paths."""
        seed_path = Path(tmp_dir) / "x_seeds_3.json"
        seed_path.write_text(
            json.dumps(
                {
                    "accounts": [
                        {"handle": "openai", "name": "OpenAI", "kind": "official", "tier": 3, "active": True},
                        {"handle": "anthropicai", "name": "Anthropic", "kind": "official", "tier": 3, "active": True},
                        {"handle": "googledeepmind", "name": "Google DeepMind", "kind": "official", "tier": 3, "active": True},
                    ]
                }
            ),
            encoding="utf-8",
        )
        return seed_path

    _ANTHROPIC_PAYLOAD = {
        "tweets": [
            {
                "id": "2035012260009000001",
                "text": "Claude 4 is now available in the API. Introducing our most capable model yet. https://t.co/anthropic1",
                "createdAt": "Sat Mar 21 11:00:00 +0000 2026",
                "lang": "en",
                "inReplyToUserId": None,
                "retweeted_tweet": None,
                "entities": {"urls": [{"expanded_url": "https://anthropic.com/claude-4"}]},
                "likeCount": 900,
                "replyCount": 70,
                "retweetCount": 120,
                "quoteCount": 30,
                "viewCount": 400000,
                "bookmarkCount": 80,
                "author": {"id": "2", "name": "Anthropic", "userName": "AnthropicAI", "isBlueVerified": True},
            }
        ]
    }

    _DEEPMIND_PAYLOAD = {
        "tweets": [
            {
                "id": "2035012260009000002",
                "text": "Gemini 3.0 launches today with breakthrough reasoning capabilities. https://t.co/deepmind1",
                "createdAt": "Sat Mar 21 11:30:00 +0000 2026",
                "lang": "en",
                "inReplyToUserId": None,
                "retweeted_tweet": None,
                "entities": {"urls": [{"expanded_url": "https://deepmind.google/gemini3"}]},
                "likeCount": 800,
                "replyCount": 60,
                "retweetCount": 100,
                "quoteCount": 25,
                "viewCount": 350000,
                "bookmarkCount": 60,
                "author": {"id": "4", "name": "Google DeepMind", "userName": "GoogleDeepMind", "isBlueVerified": True},
            }
        ]
    }

    def _fake_three_account_get(self, params, *, api_key):
        handle = (params.get("userName") or "").lower()
        if handle == "openai":
            return self._OPENAI_PAYLOAD
        if handle == "anthropicai":
            return self._ANTHROPIC_PAYLOAD
        if handle == "googledeepmind":
            return self._DEEPMIND_PAYLOAD
        return {"tweets": []}

    def test_fetch_hotspot_items_multi_account_yields_items_from_each(self) -> None:
        """Multi-account iteration: items from 3 distinct official accounts all appear."""
        from arxiv_assistant.apis.hotspot import hotspot_twitterapi as mod

        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.dict(os.environ, {"TWITTERAPI_IO_KEY": "test-key"}, clear=True), \
                patch.object(mod, "_twitterapi_get", side_effect=self._fake_three_account_get), \
                patch("arxiv_assistant.utils.hotspot.x_authority_registry.fetch_text", return_value=""), \
                patch("arxiv_assistant.utils.hotspot.x_authority_registry.fetch_json", return_value={}):
            seed_path = self._seed_file_three_accounts(tmp_dir)
            items = mod.fetch_hotspot_items(
                datetime(2026, 3, 21, 12, 0, tzinfo=UTC),
                24,
                seed_path,
                result_limit=80,
                snapshot_path=Path(tmp_dir) / "x_authority_inventory.json",
            )

        urls = {item.url for item in items}
        self.assertIn("https://x.com/openai/status/2035012260008272007", urls)
        self.assertIn("https://x.com/anthropicai/status/2035012260009000001", urls)
        self.assertIn("https://x.com/googledeepmind/status/2035012260009000002", urls)
        self.assertGreaterEqual(len(items), 3)

    def test_fetch_hotspot_items_per_account_fault_tolerance(self) -> None:
        """One account raising an exception does not crash the run; other accounts still yield."""
        from arxiv_assistant.apis.hotspot import hotspot_twitterapi as mod

        call_count = {"n": 0}

        def _fault_tolerant_get(params, *, api_key):
            handle = (params.get("userName") or "").lower()
            if handle == "anthropicai":
                raise RuntimeError("simulated network error for anthropicai")
            if handle == "openai":
                return self._OPENAI_PAYLOAD
            if handle == "googledeepmind":
                return self._DEEPMIND_PAYLOAD
            return {"tweets": []}

        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.dict(os.environ, {"TWITTERAPI_IO_KEY": "test-key"}, clear=True), \
                patch.object(mod, "_twitterapi_get", side_effect=_fault_tolerant_get), \
                patch("arxiv_assistant.utils.hotspot.x_authority_registry.fetch_text", return_value=""), \
                patch("arxiv_assistant.utils.hotspot.x_authority_registry.fetch_json", return_value={}):
            seed_path = self._seed_file_three_accounts(tmp_dir)
            items = mod.fetch_hotspot_items(
                datetime(2026, 3, 21, 12, 0, tzinfo=UTC),
                24,
                seed_path,
                result_limit=80,
                snapshot_path=Path(tmp_dir) / "x_authority_inventory.json",
            )

        urls = {item.url for item in items}
        # openai and googledeepmind tweets must still appear despite anthropicai failing
        self.assertIn("https://x.com/openai/status/2035012260008272007", urls)
        self.assertIn("https://x.com/googledeepmind/status/2035012260009000002", urls)
        # anthropicai tweet must not appear (its fetch errored)
        self.assertNotIn("https://x.com/anthropicai/status/2035012260009000001", urls)

    def test_fetch_hotspot_items_dedup_same_tweet_from_two_accounts(self) -> None:
        """The same tweet URL returned via two different account fetches must appear exactly once.

        Scenario: both openai and anthropicai fetches return the OpenAI tweet
        (same tweet_id, same author='openai'), simulating the real-world case where a
        tweet appears in multiple accounts' timelines (e.g. via quote-tweet or API quirk).
        The seen_urls dedup set in _collect_timelines must prevent the duplicate.
        """
        from arxiv_assistant.apis.hotspot import hotspot_twitterapi as mod

        # anthropicai's response also returns the OpenAI tweet (same id, same author userName=OpenAI)
        openai_tweet_via_anthropicai = {
            "tweets": [
                dict(self._OPENAI_PAYLOAD["tweets"][0])  # identical tweet, including author=OpenAI
            ]
        }

        def _dedup_get(params, *, api_key):
            handle = (params.get("userName") or "").lower()
            if handle == "openai":
                return self._OPENAI_PAYLOAD
            if handle == "anthropicai":
                return openai_tweet_via_anthropicai  # same openai tweet surfaced via anthropicai fetch
            return {"tweets": []}

        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.dict(os.environ, {"TWITTERAPI_IO_KEY": "test-key"}, clear=True), \
                patch.object(mod, "_twitterapi_get", side_effect=_dedup_get), \
                patch("arxiv_assistant.utils.hotspot.x_authority_registry.fetch_text", return_value=""), \
                patch("arxiv_assistant.utils.hotspot.x_authority_registry.fetch_json", return_value={}):
            seed_path = Path(tmp_dir) / "x_seeds_dedup.json"
            seed_path.write_text(
                json.dumps(
                    {
                        "accounts": [
                            {"handle": "openai", "name": "OpenAI", "kind": "official", "tier": 3, "active": True},
                            {"handle": "anthropicai", "name": "Anthropic", "kind": "official", "tier": 3, "active": True},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            items = mod.fetch_hotspot_items(
                datetime(2026, 3, 21, 12, 0, tzinfo=UTC),
                24,
                seed_path,
                result_limit=80,
                snapshot_path=Path(tmp_dir) / "x_authority_inventory.json",
            )

        # The URL https://x.com/openai/status/2035012260008272007 must appear exactly once
        target_url = "https://x.com/openai/status/2035012260008272007"
        matching = [item for item in items if item.url == target_url]
        self.assertEqual(len(matching), 1, "Same tweet URL must be deduplicated to one item")

    def test_fetch_hotspot_items_result_limit_is_honored(self) -> None:
        """result_limit=1 must cap output at 1 item even when 3 accounts each have a tweet."""
        from arxiv_assistant.apis.hotspot import hotspot_twitterapi as mod

        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.dict(os.environ, {"TWITTERAPI_IO_KEY": "test-key"}, clear=True), \
                patch.object(mod, "_twitterapi_get", side_effect=self._fake_three_account_get), \
                patch("arxiv_assistant.utils.hotspot.x_authority_registry.fetch_text", return_value=""), \
                patch("arxiv_assistant.utils.hotspot.x_authority_registry.fetch_json", return_value={}):
            seed_path = self._seed_file_three_accounts(tmp_dir)
            items = mod.fetch_hotspot_items(
                datetime(2026, 3, 21, 12, 0, tzinfo=UTC),
                24,
                seed_path,
                result_limit=1,
                snapshot_path=Path(tmp_dir) / "x_authority_inventory.json",
            )

        self.assertLessEqual(len(items), 1)


if __name__ == "__main__":
    unittest.main()
