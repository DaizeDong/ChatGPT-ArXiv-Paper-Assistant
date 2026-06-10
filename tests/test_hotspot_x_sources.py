from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import configparser

from arxiv_assistant.apis.hotspot.hotspot_x_ainews import _extract_twitter_section_items
from arxiv_assistant.apis.hotspot.hotspot_x_paperpulse import fetch_hotspot_items as fetch_x_paperpulse_items
from arxiv_assistant.utils.hotspot.x_authority_registry import build_x_authority_registry, load_x_authority_registry, refresh_x_authority_registry
from arxiv_assistant.hotspots import pipeline as hp


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
        items = fetch_x_paperpulse_items(datetime(2026, 3, 21, 12, 0, tzinfo=UTC), 36, result_limit=10)
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

    @patch("arxiv_assistant.utils.hotspot.x_authority_registry.build_x_authority_registry")
    def test_refresh_x_authority_registry_preserves_existing_snapshot_on_catastrophic_graph_failure(self, mock_build_registry) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            seed_path = Path(tmp_dir) / "x_seeds.json"
            seed_path.write_text(json.dumps({"accounts": []}), encoding="utf-8")
            snapshot_path = Path(tmp_dir) / "x_authority_inventory.json"
            existing_payload = {
                "generated_at": "2026-03-23T00:00:00+00:00",
                "seed_path": str(seed_path),
                "seed_sources": {"follow_the_ai_leaders": 31, "paperpulse_authors": 10, "overlap": 9},
                "graph_expansion": {
                    "enabled": True,
                    "seed_handles": ["jeffdean", "drjimfan", "hongyiwang10"],
                    "resolved_seeds": 12,
                    "selected_candidates": 510,
                },
                "errors": [],
                "accounts": [{"handle": "openai", "active": True, "tier": 3, "kind": "official", "source_refs": ["manual_seed"]}] * 300,
            }
            snapshot_path.write_text(json.dumps(existing_payload, indent=2), encoding="utf-8")
            mock_build_registry.return_value = {
                "generated_at": "2026-03-24T00:00:00+00:00",
                "seed_path": str(seed_path),
                "seed_sources": {"follow_the_ai_leaders": 31, "paperpulse_authors": 10, "overlap": 9},
                "graph_expansion": {
                    "enabled": True,
                    "seed_handles": ["jeffdean", "drjimfan", "hongyiwang10"],
                    "resolved_seeds": 0,
                    "selected_candidates": 0,
                },
                "errors": [
                    {"source": "following:jeffdean", "error": "403"},
                    {"source": "following:drjimfan", "error": "403"},
                    {"source": "following:hongyiwang10", "error": "403"},
                ],
                "accounts": [{"handle": "openai", "active": True, "tier": 3, "kind": "official", "source_refs": ["manual_seed"]}] * 68,
            }

            payload = refresh_x_authority_registry(
                seed_path=seed_path,
                snapshot_path=snapshot_path,
                force=True,
            )

            persisted = json.loads(snapshot_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["generated_at"], "2026-03-23T00:00:00+00:00")
        self.assertEqual(persisted["generated_at"], "2026-03-23T00:00:00+00:00")
        self.assertEqual(persisted["graph_expansion"]["selected_candidates"], 510)


class TestAgentScoutRegistration(unittest.TestCase):
    """Assert agent_scout is registered as a kernel source iff use_agent_scout is on.

    The scout's agent_fn is NEVER called here: every adapter fetch function is
    patched to return [] so the unit test touches no network and no claude CLI.
    We only assert which source ids the pipeline registers under the flag.
    """

    def _make_config(self, *, use_agent_scout: bool) -> configparser.ConfigParser:
        cfg = configparser.ConfigParser()
        cfg.read_dict(
            {
                "HOTSPOTS": {
                    "freshness_hours": "30",
                    "agent_scout_result_limit": "40",
                    "agent_scout_timeout_s": "300",
                    "source_registry_path": "configs/hotspot/roundup_sites.json",
                },
                "HOTSPOT_SOURCES": {
                    "use_local_papers": "false",
                    "use_hf_papers": "false",
                    "use_ainews": "false",
                    "use_official_blogs": "false",
                    "use_roundup_sites": "false",
                    "use_analysis_feeds": "false",
                    "use_reddit": "false",
                    "use_x_ainews_twitter": "false",
                    "use_twitterapi": "false",
                    "use_x_paperpulse": "false",
                    "use_x_official": "false",
                    "use_github": "false",
                    "use_hn": "false",
                    "use_agent_scout": "true" if use_agent_scout else "false",
                    "reuse_cached_raw": "false",
                },
                "HOTSPOT_X": {},
                "HOTSPOT_GITHUB": {},
                "HOTSPOT_HN": {},
            }
        )
        return cfg

    def _registered_source_ids(self, *, use_agent_scout: bool) -> set[str]:
        cfg = self._make_config(use_agent_scout=use_agent_scout)
        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.object(hp, "fetch_agent_scout_items", return_value=[]) as mock_scout:
            _items, source_stats, _usage = hp.fetch_source_payloads(
                datetime(2026, 6, 8, tzinfo=UTC),
                Path(tmp_dir),
                cfg,
                force=True,
            )
            # The scout agent_fn is never invoked in this unit test; if the flag
            # is off, the adapter is not even registered/called.
            if not use_agent_scout:
                mock_scout.assert_not_called()
        return set(source_stats.keys())

    def test_agent_scout_registered_when_flag_on(self) -> None:
        self.assertIn("agent_scout", self._registered_source_ids(use_agent_scout=True))

    def test_agent_scout_absent_when_flag_off(self) -> None:
        self.assertNotIn("agent_scout", self._registered_source_ids(use_agent_scout=False))


class TestSubagentRouteRegistration(unittest.TestCase):
    """use_subagent_routes: reddit served by the browser subagent (not the 403 scraper).

    The browser fetcher + the direct reddit fetcher are patched to [] -> no network,
    no claude/playwright. We only assert which source ids the pipeline registers and
    that the direct reddit scraper is suppressed when the subagent route is on.
    """

    def _make_config(self, *, use_subagent_routes: bool) -> configparser.ConfigParser:
        cfg = configparser.ConfigParser()
        cfg.read_dict(
            {
                "HOTSPOTS": {
                    "freshness_hours": "30",
                    "source_registry_path": "configs/hotspot/roundup_sites.json",
                },
                "HOTSPOT_SOURCES": {
                    "use_local_papers": "false", "use_hf_papers": "false", "use_ainews": "false",
                    "use_official_blogs": "false", "use_roundup_sites": "false", "use_analysis_feeds": "false",
                    "use_reddit": "true",  # direct reddit ON; subagent route should supersede it
                    "use_x_ainews_twitter": "false", "use_twitterapi": "false",
                    "use_x_paperpulse": "false", "use_x_official": "false",
                    "use_github": "false", "use_hn": "false", "use_agent_scout": "false",
                    "use_subagent_routes": "true" if use_subagent_routes else "false",
                    "reuse_cached_raw": "false",
                },
                "HOTSPOT_X": {}, "HOTSPOT_GITHUB": {}, "HOTSPOT_HN": {},
            }
        )
        return cfg

    def _run(self, *, use_subagent_routes: bool):
        cfg = self._make_config(use_subagent_routes=use_subagent_routes)
        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch("arxiv_assistant.apis.hotspot.browser_source_fetch.fetch_source_via_browser",
                      return_value=[]) as mock_browser, \
                patch.object(hp, "fetch_reddit_items", return_value=[]) as mock_reddit:
            _items, source_stats, _usage = hp.fetch_source_payloads(
                datetime(2026, 6, 8, tzinfo=UTC), Path(tmp_dir), cfg, force=True,
            )
        return set(source_stats.keys()), mock_browser, mock_reddit

    def test_reddit_via_browser_subagent_when_on(self) -> None:
        ids, mock_browser, mock_reddit = self._run(use_subagent_routes=True)
        # reddit served by the browser-route source(s); the direct reddit scraper is suppressed.
        self.assertIn("reddit_localllama", ids)
        self.assertNotIn("reddit", ids)
        mock_reddit.assert_not_called()
        self.assertTrue(mock_browser.called)

    def test_direct_reddit_when_off(self) -> None:
        ids, mock_browser, mock_reddit = self._run(use_subagent_routes=False)
        # default: direct reddit scraper registered, browser fetcher never imported/called.
        self.assertIn("reddit", ids)
        self.assertNotIn("reddit_localllama", ids)
        mock_browser.assert_not_called()

    def test_config_flag_and_stepfun_url(self) -> None:
        root = Path(__file__).resolve().parents[1]
        c = configparser.ConfigParser(); c.read(root / "configs" / "config.ini", encoding="utf-8")
        self.assertFalse(c.getboolean("HOTSPOT_SOURCES", "use_subagent_routes"))
        p = configparser.ConfigParser(); p.read(root / "configs" / "profiles" / "agent-native.ini", encoding="utf-8")
        self.assertTrue(p.getboolean("HOTSPOT_SOURCES", "use_subagent_routes"))
        blogs = json.loads((root / "configs" / "hotspot" / "official_blogs.json").read_text(encoding="utf-8"))
        stepfun = next(b for b in blogs if b.get("source_id") == "stepfun_blog")
        self.assertEqual(stepfun["url"], "https://www.stepfun.com/")


if __name__ == "__main__":
    unittest.main()
