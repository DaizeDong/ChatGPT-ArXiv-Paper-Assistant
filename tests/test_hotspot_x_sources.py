from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from arxiv_assistant.apis.hotspot.hotspot_x_ainews import _extract_twitter_section_items
from arxiv_assistant.apis.hotspot.hotspot_x_official import fetch_hotspot_items as fetch_x_official_items
from arxiv_assistant.apis.hotspot.hotspot_x_paperpulse import fetch_hotspot_items as fetch_x_paperpulse_items
from arxiv_assistant.utils.hotspot.x_authority_registry import build_x_authority_registry, load_x_authority_registry, refresh_x_authority_registry


class TestHotspotXSources(unittest.TestCase):
    def test_ainews_twitter_recap_extracts_x_items(self) -> None:
        content_html = """
        <blockquote>
          <p>We checked <a href="https://twitter.com/i/lists/1585430245762441216">544 Twitters</a>.</p>
        </blockquote>
        <h1>AI Twitter Recap</h1>
        <p><strong>Coding Agents, Model Attribution, and the Cursor/Kimi Composer 2 Controversy</strong></p>
        <ul>
          <li>
            <strong>Cursor's Composer 2 is built on Kimi K2.5</strong>:
            Attribution questions dominated the day
            <a href="https://x.com/OpenAI/status/2035012260008272007">@OpenAI</a>
            <a href="https://x.com/AnthropicAI/status/2035041428535939535">@AnthropicAI</a>.
          </li>
        </ul>
        <h1>AI Reddit Recap</h1>
        """
        items = _extract_twitter_section_items(
            content_html,
            "AINews issue",
            "https://news.smol.ai/issues/test",
            "2026-03-21T08:00:00+00:00",
        )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].source_id, "ainews_twitter")
        self.assertEqual(items[0].source_role, "community_heat")
        self.assertEqual(items[0].url, "https://x.com/OpenAI/status/2035012260008272007")
        self.assertEqual(items[0].metadata["host"], "x.com")
        self.assertGreaterEqual(items[0].metadata["activity"], 80)

    @patch("arxiv_assistant.apis.hotspot.hotspot_x_official._iter_recent_search")
    def test_x_official_adapter_builds_items_from_recent_search(self, mock_iter_recent_search) -> None:
        mock_iter_recent_search.return_value = [
            {
                "id": "2035012260008272007",
                "text": "GPT-5.4 mini is available today in ChatGPT, Codex, and the API. https://t.co/abc123",
                "created_at": "2026-03-21T10:00:00.000Z",
                "author_id": "1",
                "author": {"username": "OpenAI", "name": "OpenAI", "verified": True},
                "entities": {"urls": [{"expanded_url": "https://openai.com/index/gpt-5-4-mini"}]},
                "public_metrics": {"like_count": 120, "reply_count": 9, "retweet_count": 15, "quote_count": 4, "bookmark_count": 10, "impression_count": 53000},
                "referenced_tweets": [],
            },
            {
                "id": "2035012260008272009",
                "text": "Are you up for a challenge? https://t.co/demo",
                "created_at": "2026-03-21T10:02:00.000Z",
                "author_id": "2",
                "author": {"username": "OpenAI", "name": "OpenAI", "verified": True},
                "entities": {"urls": [{"expanded_url": "https://openai.com/index/parameter-golf"}]},
                "public_metrics": {"like_count": 900, "reply_count": 12, "retweet_count": 20, "quote_count": 3, "bookmark_count": 1, "impression_count": 80000},
                "referenced_tweets": [],
            },
            {
                "id": "2035012260008272010",
                "text": "Our new paper on agents is out today: https://t.co/paper",
                "created_at": "2026-03-21T10:05:00.000Z",
                "author_id": "3",
                "author": {"username": "demishassabis", "name": "Demis Hassabis", "verified": True},
                "entities": {"urls": [{"expanded_url": "https://arxiv.org/abs/2603.12345"}]},
                "public_metrics": {"like_count": 400, "reply_count": 40, "retweet_count": 80, "quote_count": 8, "bookmark_count": 20, "impression_count": 90000},
                "referenced_tweets": [],
            },
        ]
        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(os.environ, {"X_BEARER_TOKEN": "test-token"}):
            config_path = Path(tmp_dir) / "x_seeds.json"
            config_path.write_text(
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
            items = fetch_x_official_items(
                datetime(2026, 3, 21, tzinfo=UTC),
                36,
                config_path,
                default_result_limit=80,
                snapshot_path=Path(tmp_dir) / "x_authority_inventory.json",
            )
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].url, "https://x.com/OpenAI/status/2035012260008272007")
        self.assertEqual(items[0].source_name, "OpenAI")
        self.assertEqual(items[0].source_role, "official_news")
        self.assertEqual(items[0].metadata["authority_kind"], "official")
        self.assertGreater(items[0].metadata["activity"], 200)

    @patch("arxiv_assistant.apis.hotspot.hotspot_x_paperpulse.fetch_json")
    def test_paperpulse_adapter_builds_researcher_feed_items(self, mock_fetch_json) -> None:
        mock_fetch_json.return_value = {
            "count": 1,
            "tweets": [
                {
                    "tweet_id": "2035012260008273000",
                    "text": "A useful benchmark roundup on reasoning models is worth reading https://example.com/report",
                    "created_at": "2026-03-21T09:30:00+00:00",
                    "author_handle": "demishassabis",
                    "author_name": "Demis Hassabis",
                    "public_metrics": {"like_count": 900, "reply_count": 45, "retweet_count": 110, "quote_count": 18, "bookmark_count": 120, "impression_count": 240000},
                    "referenced_tweets": [],
                }
            ],
        }
        items = fetch_x_paperpulse_items(datetime(2026, 3, 21, tzinfo=UTC), 36, result_limit=10)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].source_name, "PaperPulse Researcher Feed")
        self.assertEqual(items[0].url, "https://x.com/demishassabis/status/2035012260008273000")
        self.assertEqual(items[0].metadata["proxy_source"], "paperpulse")
        self.assertGreater(items[0].metadata["activity"], 1000)

    @patch("arxiv_assistant.utils.hotspot.x_authority_registry._get_bearer_token", return_value=None)
    @patch("arxiv_assistant.utils.hotspot.x_authority_registry.fetch_json")
    @patch("arxiv_assistant.utils.hotspot.x_authority_registry.fetch_text")
    def test_x_authority_registry_merges_manual_and_external_seeds(self, mock_fetch_text, mock_fetch_json, _mock_token) -> None:
        mock_fetch_text.return_value = "[Demis](https://x.com/demishassabis) [OpenAI](https://x.com/OpenAI)"
        mock_fetch_json.return_value = {"authors": ["demishassabis", "JeffDean"]}
        with tempfile.TemporaryDirectory() as tmp_dir:
            seed_path = Path(tmp_dir) / "x_seeds.json"
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
            payload = build_x_authority_registry(seed_path)

        accounts = {row["handle"]: row for row in payload["accounts"]}
        self.assertIn("openai", accounts)
        self.assertIn("demishassabis", accounts)
        self.assertIn("jeffdean", accounts)
        self.assertTrue(accounts["demishassabis"]["active"])
        self.assertGreaterEqual(accounts["demishassabis"]["tier"], 2)
        self.assertEqual(accounts["openai"]["kind"], "official")

    @patch("arxiv_assistant.utils.hotspot.x_authority_registry.fetch_json")
    @patch("arxiv_assistant.utils.hotspot.x_authority_registry.fetch_text")
    @patch("arxiv_assistant.utils.hotspot.x_authority_registry._fetch_x_following")
    @patch("arxiv_assistant.utils.hotspot.x_authority_registry._fetch_x_user")
    @patch("arxiv_assistant.utils.hotspot.x_authority_registry._get_bearer_token", return_value="token")
    def test_x_authority_registry_expands_following_graph(
        self,
        _mock_token,
        mock_fetch_x_user,
        mock_fetch_x_following,
        mock_fetch_text,
        mock_fetch_json,
    ) -> None:
        mock_fetch_text.return_value = ""
        mock_fetch_json.return_value = {"authors": []}
        mock_fetch_x_user.side_effect = [
            {"id": "1", "username": "jeffdean", "name": "Jeff Dean", "description": "Chief Scientist, Google DeepMind"},
            {"id": "2", "username": "hongyiwang10", "name": "Hongyi Wang", "description": "Assistant Professor, AI infrastructure"},
        ]
        mock_fetch_x_following.side_effect = [
            [
                {
                    "username": "sh_research",
                    "name": "Sharon Research",
                    "description": "Research scientist working on multimodal AI and agents at a frontier lab",
                    "verified": True,
                    "verified_type": "blue",
                    "public_metrics": {"followers_count": 24000, "listed_count": 420},
                },
                {
                    "username": "random_finance",
                    "name": "Random Finance",
                    "description": "Macro investor and markets commentator",
                    "verified": True,
                    "verified_type": "blue",
                    "public_metrics": {"followers_count": 80000, "listed_count": 90},
                },
            ],
            [
                {
                    "username": "sh_research",
                    "name": "Sharon Research",
                    "description": "Research scientist working on multimodal AI and agents at a frontier lab",
                    "verified": True,
                    "verified_type": "blue",
                    "public_metrics": {"followers_count": 24000, "listed_count": 420},
                },
                {
                    "username": "hongyi_friend",
                    "name": "Hongyi Friend",
                    "description": "AI engineer building LLM systems and inference infrastructure",
                    "verified": False,
                    "verified_type": "none",
                    "public_metrics": {"followers_count": 2800, "listed_count": 35},
                },
            ],
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            seed_path = Path(tmp_dir) / "x_seeds.json"
            seed_path.write_text(
                json.dumps(
                    {
                        "accounts": [
                            {"handle": "jeffdean", "name": "Jeff Dean", "kind": "researcher", "tier": 3, "active": True},
                            {"handle": "hongyiwang10", "name": "Hongyi Wang", "kind": "researcher", "tier": 2, "active": True},
                        ],
                        "following_graph": {
                            "seed_handles": ["jeffdean", "hongyiwang10"],
                            "max_following_per_seed": 20,
                            "min_support_count": 1,
                            "min_active_support_count": 2,
                            "min_watchlist_score": 4.2,
                            "min_active_score": 6.0,
                            "min_followers_count": 1000,
                            "min_listed_count": 20,
                        },
                    }
                ),
                encoding="utf-8",
            )
            payload = build_x_authority_registry(seed_path)

        accounts = {row["handle"]: row for row in payload["accounts"]}
        self.assertIn("sh_research", accounts)
        self.assertTrue(accounts["sh_research"]["active"])
        self.assertIn("following:jeffdean", accounts["sh_research"]["source_refs"])
        self.assertIn("following:hongyiwang10", accounts["sh_research"]["source_refs"])
        self.assertIn("hongyi_friend", accounts)
        self.assertFalse(accounts["hongyi_friend"]["active"])
        self.assertNotIn("random_finance", accounts)
        self.assertEqual(payload["graph_expansion"]["selected_candidates"], 2)

    def test_x_authority_registry_can_fallback_to_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            snapshot_path = Path(tmp_dir) / "x_authority_inventory.json"
            snapshot_payload = {
                "generated_at": "2026-03-23T00:00:00+00:00",
                "accounts": [
                    {
                        "handle": "openai",
                        "name": "OpenAI",
                        "kind": "official",
                        "tier": 3,
                        "active": True,
                        "source_refs": ["manual_seed"],
                    }
                ],
            }
            snapshot_path.write_text(json.dumps(snapshot_payload), encoding="utf-8")

            payload = load_x_authority_registry(
                snapshot_path=snapshot_path,
                max_age_hours=24,
            )

        self.assertEqual(len(payload["accounts"]), 1)
        self.assertEqual(payload["accounts"][0]["handle"], "openai")

    @patch("arxiv_assistant.utils.hotspot.x_authority_registry.build_x_authority_registry")
    def test_refresh_x_authority_registry_is_stable_when_payload_is_equivalent(self, mock_build_registry) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            seed_path = Path(tmp_dir) / "x_seeds.json"
            seed_path.write_text(json.dumps({"accounts": []}), encoding="utf-8")
            snapshot_path = Path(tmp_dir) / "x_authority_inventory.json"
            existing_payload = {
                "generated_at": "2026-03-23T00:00:00+00:00",
                "seed_path": str(seed_path),
                "seed_sources": {"follow_the_ai_leaders": 1, "paperpulse_authors": 1, "overlap": 0},
                "graph_expansion": {"enabled": True},
                "errors": [],
                "accounts": [{"handle": "openai", "active": True, "tier": 3, "kind": "official", "source_refs": ["manual_seed"]}],
            }
            snapshot_path.write_text(json.dumps(existing_payload, indent=2), encoding="utf-8")
            mock_build_registry.return_value = {
                **existing_payload,
                "generated_at": "2026-03-24T00:00:00+00:00",
            }

            payload = refresh_x_authority_registry(
                seed_path=seed_path,
                snapshot_path=snapshot_path,
                force=True,
            )

            persisted = json.loads(snapshot_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["generated_at"], "2026-03-23T00:00:00+00:00")
        self.assertEqual(persisted["generated_at"], "2026-03-23T00:00:00+00:00")
        self.assertEqual(persisted["accounts"][0]["handle"], "openai")


if __name__ == "__main__":
    unittest.main()
