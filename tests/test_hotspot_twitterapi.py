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


if __name__ == "__main__":
    unittest.main()
