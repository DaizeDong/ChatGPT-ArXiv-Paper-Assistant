from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem
from arxiv_assistant.utils.hotspot.hotspot_web_data import build_daily_hotspot_web_payload, write_hotspot_web_data


class TestHotspotWebData(unittest.TestCase):
    def test_daily_payload_groups_items_by_source_family_and_links_topics(self) -> None:
        report = {
            "date": "2026-03-21",
            "generated_at": "2026-03-21T12:00:00+00:00",
            "mode": "heuristic",
            "summary": "A dense day across social, papers, and official updates.",
            "totals": {"raw_items": 3, "clusters": 3, "candidate_clusters": 3, "radar_clusters": 3},
            "costs": {"prompt": 0.0, "completion": 0.0, "total": 0.0},
            "usage": {
                "llm": {
                    "provider": "OpenAI",
                    "billing_model": "quota",
                    "screen_model": "gpt-5.4",
                    "summary_model": "gpt-5.4-mini",
                    "requests": 2,
                    "prompt_tokens": 1200,
                    "completion_tokens": 300,
                    "total_tokens": 1500,
                    "prompt_cost": 0.01,
                    "completion_cost": 0.02,
                    "total_cost": 0.03,
                },
                "external": {
                    "x_official": {
                        "provider": "X API",
                        "billing_model": "quota",
                        "requests": 8,
                        "items": 2,
                        "estimated_cost": 0.0,
                        "cache_hit": False,
                    }
                },
                "summary": {"external_requests": 8, "x_requests": 8, "estimated_external_cost": 0.0},
            },
            "source_stats": {"ainews": 1, "hf_papers": 1, "openai_news": 1},
            "featured_topics": [
                {
                    "TOPIC_ID": "cursor-news",
                    "HEADLINE": "Cursor Composer 2 debate",
                    "PRIMARY_CATEGORY": "Community Signal",
                    "DISPLAY_PRIORITY": 8.2,
                    "FINAL_SCORE": 8.0,
                    "HEAT": 9,
                    "QUALITY": 7,
                    "IMPORTANCE": 7,
                    "OCCURRENCE_SCORE": 7.4,
                    "source_names": ["AINews", "Hacker News"],
                    "source_roles": ["community_heat", "hn_discussion"],
                    "source_types": ["roundup", "discussion"],
                    "items": [
                        {
                            "title": "Cursor's Composer 2 is stirring debate",
                            "url": "https://www.reddit.com/r/LocalLLaMA/comments/123",
                            "source_name": "AINews",
                        }
                    ],
                    "WHY_IT_MATTERS": "Proxy social debate around agent coding quality.",
                    "SHORT_COMMENT": "Social debate around agent coding quality.",
                }
            ],
            "category_sections": [
                {
                    "category": "Research",
                    "total_candidates": 1,
                    "topics": [
                        {
                            "TOPIC_ID": "world-sim",
                            "title": "Grounded world simulation",
                            "PRIMARY_CATEGORY": "Research",
                            "DISPLAY_PRIORITY": 7.4,
                            "FINAL_SCORE": 7.3,
                            "HEAT": 6,
                            "QUALITY": 8,
                            "IMPORTANCE": 7,
                            "OCCURRENCE_SCORE": 6.1,
                            "source_names": ["Hugging Face Trending Papers"],
                            "source_roles": ["paper_trending"],
                            "source_types": ["paper"],
                            "items": [
                                {
                                    "title": "Grounding World Simulation Models in a Real-World Metropolis",
                                    "url": "https://huggingface.co/papers/2603.00001",
                                    "source_name": "Hugging Face Trending Papers",
                                }
                            ],
                            "summary": "A paper about world simulation grounding.",
                            "SHORT_COMMENT": "Paper about world simulation grounding.",
                        }
                    ],
                }
            ],
            "long_tail_sections": [],
            "watchlist": [],
            "x_buzz": [],
        }
        raw_items = [
            HotspotItem(
                source_id="ainews",
                source_name="AINews",
                source_role="community_heat",
                source_type="roundup",
                title="Cursor's Composer 2 is stirring debate",
                summary="AINews social proxy item.",
                url="https://www.reddit.com/r/LocalLLaMA/comments/123",
                canonical_url="https://www.reddit.com/r/LocalLLaMA/comments/123",
                metadata={"activity": 900},
            ),
            HotspotItem(
                source_id="hf_papers",
                source_name="Hugging Face Trending Papers",
                source_role="paper_trending",
                source_type="paper",
                title="Grounding World Simulation Models in a Real-World Metropolis",
                summary="A paper about world simulation grounding.",
                url="https://huggingface.co/papers/2603.00001",
                canonical_url="https://arxiv.org/abs/2603.00001",
                metadata={"arxiv_id": "2603.00001", "upvotes": 120},
            ),
            HotspotItem(
                source_id="openai_news",
                source_name="OpenAI News",
                source_role="official_news",
                source_type="official_blog",
                title="Introducing a new agent benchmark",
                summary="Official benchmark launch.",
                url="https://openai.com/index/new-agent-benchmark",
                canonical_url="https://openai.com/index/new-agent-benchmark",
                metadata={"is_official": True},
            ),
        ]

        payload = build_daily_hotspot_web_payload(report, raw_items)

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["meta"]["counts"]["featured_topics"], 1)
        self.assertEqual(payload["meta"]["counts"]["source_items"], 3)
        self.assertEqual(payload["usage"]["llm"]["prompt_tokens"], 1200)
        self.assertEqual(payload["usage"]["external"]["x_official"]["requests"], 8)
        section_lookup = {section["slug"]: section for section in payload["source_sections"]}
        self.assertEqual(section_lookup["x-buzz"]["count"], 1)
        self.assertEqual(section_lookup["papers"]["count"], 1)
        self.assertEqual(section_lookup["official"]["count"], 1)
        self.assertEqual(section_lookup["blogs"]["count"], 0)
        x_item = section_lookup["x-buzz"]["items"][0]
        self.assertEqual(x_item["topic_refs"][0]["topic_id"], "cursor-news")
        featured = payload["featured_topics"][0]
        self.assertEqual(featured["headline"], "Cursor Composer 2 debate")
        self.assertEqual(featured["route"], "/hot/2026-03-21/topic/cursor-composer-2-debate/")

    def test_daily_payload_orders_sections_by_signal_strength_and_family_priority(self) -> None:
        report = {
            "date": "2026-03-21",
            "generated_at": "2026-03-21T12:00:00+00:00",
            "mode": "heuristic",
            "summary": "A broad day across social buzz, official news, and research.",
            "totals": {"raw_items": 4, "clusters": 4, "candidate_clusters": 4, "radar_clusters": 4},
            "costs": {"prompt": 0.0, "completion": 0.0, "total": 0.0},
            "source_stats": {"ainews": 1, "openai_news": 1, "hf_papers": 1, "github_trend": 1},
            "featured_topics": [
                {
                    "TOPIC_ID": "cursor-news",
                    "HEADLINE": "Cursor Composer 2 debate",
                    "PRIMARY_CATEGORY": "Community Signal",
                    "DISPLAY_PRIORITY": 8.3,
                    "FINAL_SCORE": 8.0,
                    "HEAT": 9,
                    "QUALITY": 7,
                    "IMPORTANCE": 7,
                    "OCCURRENCE_SCORE": 7.1,
                    "source_names": ["AINews", "Hacker News"],
                    "source_roles": ["community_heat", "hn_discussion"],
                    "source_types": ["roundup", "discussion"],
                    "items": [{"title": "Cursor's Composer 2 is stirring debate", "url": "https://www.reddit.com/r/LocalLLaMA/comments/123", "source_name": "AINews"}],
                    "WHY_IT_MATTERS": "Proxy social debate around agent coding quality.",
                    "SHORT_COMMENT": "Social debate around agent coding quality.",
                }
            ],
            "category_sections": [
                {
                    "category": "Product Release",
                    "total_candidates": 1,
                    "topics": [
                        {
                            "TOPIC_ID": "official-release",
                            "title": "OpenAI announces a new agent benchmark",
                            "PRIMARY_CATEGORY": "Product Release",
                            "DISPLAY_PRIORITY": 7.1,
                            "FINAL_SCORE": 7.0,
                            "HEAT": 6,
                            "QUALITY": 7,
                            "IMPORTANCE": 8,
                            "OCCURRENCE_SCORE": 5.8,
                            "source_names": ["OpenAI News"],
                            "source_roles": ["official_news"],
                            "source_types": ["official_blog"],
                            "items": [{"title": "Introducing a new agent benchmark", "url": "https://openai.com/index/new-agent-benchmark", "source_name": "OpenAI News"}],
                            "summary": "Official benchmark launch.",
                            "SHORT_COMMENT": "Official benchmark launch.",
                        }
                    ],
                }
            ],
            "long_tail_sections": [
                {
                    "category": "Tooling",
                    "total_candidates": 1,
                    "topics": [
                        {
                            "TOPIC_ID": "repo-trend",
                            "title": "Open-source agent runtime",
                            "PRIMARY_CATEGORY": "Tooling",
                            "DISPLAY_PRIORITY": 6.3,
                            "FINAL_SCORE": 6.1,
                            "HEAT": 6,
                            "QUALITY": 6,
                            "IMPORTANCE": 6,
                            "OCCURRENCE_SCORE": 4.9,
                            "source_names": ["GitHub Trending Repos"],
                            "source_roles": ["github_trend"],
                            "source_types": ["repo"],
                            "items": [{"title": "acme/agent-runtime", "url": "https://github.com/acme/agent-runtime", "source_name": "GitHub Trending Repos"}],
                            "summary": "Repo trend.",
                            "SHORT_COMMENT": "Repo trend.",
                        }
                    ],
                }
            ],
            "watchlist": [],
            "x_buzz": [],
        }
        raw_items = [
            HotspotItem(
                source_id="ainews",
                source_name="AINews",
                source_role="community_heat",
                source_type="roundup",
                title="Cursor's Composer 2 is stirring debate",
                summary="AINews social proxy item.",
                url="https://www.reddit.com/r/LocalLLaMA/comments/123",
                canonical_url="https://www.reddit.com/r/LocalLLaMA/comments/123",
                metadata={"activity": 900},
            ),
            HotspotItem(
                source_id="openai_news",
                source_name="OpenAI News",
                source_role="official_news",
                source_type="official_blog",
                title="Introducing a new agent benchmark",
                summary="Official benchmark launch.",
                url="https://openai.com/index/new-agent-benchmark",
                canonical_url="https://openai.com/index/new-agent-benchmark",
                metadata={"is_official": True},
            ),
            HotspotItem(
                source_id="github_trend",
                source_name="GitHub Trending Repos",
                source_role="github_trend",
                source_type="repo",
                title="acme/agent-runtime",
                summary="Repo trend.",
                url="https://github.com/acme/agent-runtime",
                canonical_url="https://github.com/acme/agent-runtime",
                metadata={"stars": 4200},
            ),
            HotspotItem(
                source_id="hf_papers",
                source_name="Hugging Face Trending Papers",
                source_role="paper_trending",
                source_type="paper",
                title="Long-context agent memory paper",
                summary="A paper about agent memory.",
                url="https://huggingface.co/papers/2603.00001",
                canonical_url="https://arxiv.org/abs/2603.00001",
                metadata={"arxiv_id": "2603.00001", "upvotes": 120},
            ),
        ]

        payload = build_daily_hotspot_web_payload(report, raw_items)
        ordered_slugs = [section["slug"] for section in payload["source_sections"] if section["count"] > 0]

        self.assertEqual(ordered_slugs, ["x-buzz", "official", "github", "papers"])

    def test_write_hotspot_web_data_builds_root_month_and_year_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "out"
            (output_root / "md" / "2026-03").mkdir(parents=True, exist_ok=True)
            (output_root / "md" / "2026-03" / "2026-03-20-output.md").write_text("# Paper day", encoding="utf-8")
            (output_root / "md" / "2026-03" / "2026-03-21-output.md").write_text("# Paper day", encoding="utf-8")
            raw_items = [
                HotspotItem(
                    source_id="ainews",
                    source_name="AINews",
                    source_role="community_heat",
                    source_type="roundup",
                    title="Social item",
                    summary="Proxy social item.",
                    url="https://www.reddit.com/r/LocalLLaMA/comments/aaa",
                    canonical_url="https://www.reddit.com/r/LocalLLaMA/comments/aaa",
                    metadata={"activity": 300},
                )
            ]

            for date in ("2026-03-20", "2026-03-21"):
                report = {
                    "date": date,
                    "generated_at": f"{date}T12:00:00+00:00",
                    "mode": "heuristic",
                    "summary": f"Summary for {date}",
                    "totals": {"raw_items": 1, "clusters": 1, "candidate_clusters": 1, "radar_clusters": 1},
                    "costs": {"prompt": 0.0, "completion": 0.0, "total": 0.0},
                    "source_stats": {"ainews": 1},
                    "featured_topics": [
                        {
                            "TOPIC_ID": f"topic-{date}",
                            "HEADLINE": f"Headline {date}",
                            "PRIMARY_CATEGORY": "Community Signal",
                            "DISPLAY_PRIORITY": 7.0,
                            "FINAL_SCORE": 7.0,
                            "HEAT": 7,
                            "QUALITY": 6,
                            "IMPORTANCE": 6,
                            "OCCURRENCE_SCORE": 5.0,
                            "source_names": ["AINews"],
                            "source_roles": ["community_heat"],
                            "source_types": ["roundup"],
                            "items": [{"title": "Social item", "url": "https://www.reddit.com/r/LocalLLaMA/comments/aaa", "source_name": "AINews"}],
                            "WHY_IT_MATTERS": "Social signal.",
                            "SHORT_COMMENT": "Social signal.",
                        }
                    ],
                    "category_sections": [],
                    "long_tail_sections": [],
                    "watchlist": [],
                    "x_buzz": [],
                }
                write_hotspot_web_data(output_root, report, raw_items)

            root_index = json.loads((output_root / "web_data" / "hot" / "index.json").read_text(encoding="utf-8"))
            month_index = json.loads((output_root / "web_data" / "hot" / "2026-03" / "index.json").read_text(encoding="utf-8"))
            year_index = json.loads((output_root / "web_data" / "hot" / "2026" / "index.json").read_text(encoding="utf-8"))
            latest_daily = json.loads((output_root / "web_data" / "hot" / "2026-03-21.json").read_text(encoding="utf-8"))

            self.assertEqual(root_index["latest_date"], "2026-03-21")
            self.assertEqual(len(root_index["dates"]), 2)
            self.assertEqual(month_index["month"], "2026-03")
            self.assertEqual(len(month_index["days"]), 2)
            self.assertEqual(month_index["totals"]["days"], 2)
            self.assertEqual(month_index["source_section_totals"]["x-buzz"], 2)
            self.assertEqual(year_index["year"], "2026")
            self.assertEqual(len(year_index["months"]), 1)
            self.assertEqual(year_index["totals"]["days"], 2)
            self.assertEqual(year_index["months"][0]["source_section_totals"]["x-buzz"], 2)
            self.assertEqual(latest_daily["meta"]["previous_date"], "2026-03-20")
            self.assertEqual(latest_daily["meta"]["paper_routes"]["day"], "/archive/2026-03/21/")
            self.assertEqual(month_index["paper_routes"]["month"], "/archive/2026-03/")
            self.assertEqual(year_index["paper_routes"]["year"], "/archive/2026/")
            self.assertEqual(root_index["dates"][-1]["paper_routes"]["day"], "/archive/2026-03/21/")

    def test_item_ranking_prefers_multi_source_topic_support_inside_a_section(self) -> None:
        report = {
            "date": "2026-03-21",
            "generated_at": "2026-03-21T12:00:00+00:00",
            "mode": "heuristic",
            "summary": "Daily summary.",
            "totals": {"raw_items": 2, "clusters": 2, "candidate_clusters": 2, "radar_clusters": 2},
            "costs": {"prompt": 0.0, "completion": 0.0, "total": 0.0},
            "source_stats": {"roundup_sites": 2},
            "featured_topics": [
                {
                    "TOPIC_ID": "google-stitch",
                    "HEADLINE": "Google bets on vibe design with Stitch",
                    "PRIMARY_CATEGORY": "Tooling",
                    "DISPLAY_PRIORITY": 8.0,
                    "FINAL_SCORE": 7.8,
                    "HEAT": 7,
                    "QUALITY": 7,
                    "IMPORTANCE": 7,
                    "OCCURRENCE_SCORE": 6.8,
                    "LLM_STATUS": "featured",
                    "source_names": ["Superhuman AI", "The Rundown AI"],
                    "source_roles": ["headline_consensus"],
                    "source_types": ["roundup"],
                    "items": [
                        {"title": "Google bets on vibe design with Stitch", "url": "https://www.rundown.ai/articles/google-bets-on-vibe-design-with-stitch", "source_name": "The Rundown AI"}
                    ],
                    "WHY_IT_MATTERS": "Strong multi-source tooling signal.",
                    "SHORT_COMMENT": "Strong multi-source tooling signal.",
                }
            ],
            "category_sections": [],
            "long_tail_sections": [],
            "watchlist": [],
            "x_buzz": [],
        }
        raw_items = [
            HotspotItem(
                source_id="the_rundown_ai",
                source_name="The Rundown AI",
                source_role="headline_consensus",
                source_type="roundup",
                title="Google bets on vibe design with Stitch",
                summary="The linked consensus item.",
                url="https://www.rundown.ai/articles/google-bets-on-vibe-design-with-stitch",
                canonical_url="https://www.rundown.ai/articles/google-bets-on-vibe-design-with-stitch",
                metadata={"score": 12},
            ),
            HotspotItem(
                source_id="superhuman_ai",
                source_name="Superhuman AI",
                source_role="headline_consensus",
                source_type="roundup",
                title="Anthropic doubles Claude usage limits",
                summary="A weaker standalone newsletter item.",
                url="https://www.superhuman.ai/p/anthropic-doubles-claude-s-usage-limits",
                canonical_url="https://www.superhuman.ai/p/anthropic-doubles-claude-s-usage-limits",
                metadata={"score": 1},
            ),
        ]

        payload = build_daily_hotspot_web_payload(report, raw_items)
        blogs_section = next(section for section in payload["source_sections"] if section["slug"] == "blogs")

        self.assertEqual(blogs_section["items"][0]["title"], "Google bets on vibe design with Stitch")
        self.assertGreater(blogs_section["items"][0]["signal_score"], blogs_section["items"][1]["signal_score"])

    def test_topic_summary_prefers_cross_source_topics_over_isolated_papers(self) -> None:
        report = {
            "date": "2026-03-21",
            "generated_at": "2026-03-21T12:00:00+00:00",
            "mode": "heuristic",
            "summary": "Daily summary.",
            "totals": {"raw_items": 2, "clusters": 2, "candidate_clusters": 2, "radar_clusters": 2},
            "costs": {"prompt": 0.0, "completion": 0.0, "total": 0.0},
            "source_stats": {"hf_papers": 1, "roundup_sites": 1},
            "featured_topics": [
                {
                    "TOPIC_ID": "isolated-paper",
                    "HEADLINE": "Isolated research paper",
                    "PRIMARY_CATEGORY": "Research",
                    "DISPLAY_PRIORITY": 7.8,
                    "FINAL_SCORE": 7.7,
                    "HEAT": 6,
                    "QUALITY": 8,
                    "IMPORTANCE": 7,
                    "OCCURRENCE_SCORE": 5.1,
                    "source_names": ["Hugging Face Trending Papers"],
                    "source_roles": ["paper_trending"],
                    "source_types": ["paper"],
                    "items": [{"title": "Isolated research paper", "url": "https://huggingface.co/papers/2603.00001", "source_name": "Hugging Face Trending Papers"}],
                    "WHY_IT_MATTERS": "Strong paper.",
                    "SHORT_COMMENT": "Strong paper.",
                },
                {
                    "TOPIC_ID": "google-stitch",
                    "HEADLINE": "Google bets on vibe design with Stitch",
                    "PRIMARY_CATEGORY": "Tooling",
                    "DISPLAY_PRIORITY": 7.2,
                    "FINAL_SCORE": 6.9,
                    "HEAT": 7,
                    "QUALITY": 6,
                    "IMPORTANCE": 7,
                    "OCCURRENCE_SCORE": 6.7,
                    "LLM_STATUS": "featured",
                    "source_names": ["Superhuman AI", "The Rundown AI"],
                    "source_roles": ["headline_consensus", "headline_consensus"],
                    "source_types": ["roundup", "roundup"],
                    "items": [{"title": "Google bets on vibe design with Stitch", "url": "https://www.rundown.ai/articles/google-bets-on-vibe-design-with-stitch", "source_name": "The Rundown AI"}],
                    "WHY_IT_MATTERS": "Repeated multi-source tooling signal.",
                    "SHORT_COMMENT": "Repeated multi-source tooling signal.",
                },
            ],
            "category_sections": [],
            "long_tail_sections": [],
            "watchlist": [],
            "x_buzz": [],
        }
        raw_items = [
            HotspotItem(
                source_id="hf_papers",
                source_name="Hugging Face Trending Papers",
                source_role="paper_trending",
                source_type="paper",
                title="Isolated research paper",
                summary="Paper item.",
                url="https://huggingface.co/papers/2603.00001",
                canonical_url="https://arxiv.org/abs/2603.00001",
                metadata={"upvotes": 140},
            ),
            HotspotItem(
                source_id="the_rundown_ai",
                source_name="The Rundown AI",
                source_role="headline_consensus",
                source_type="roundup",
                title="Google bets on vibe design with Stitch",
                summary="Roundup item.",
                url="https://www.rundown.ai/articles/google-bets-on-vibe-design-with-stitch",
                canonical_url="https://www.rundown.ai/articles/google-bets-on-vibe-design-with-stitch",
                metadata={"score": 12},
            ),
        ]

        payload = build_daily_hotspot_web_payload(report, raw_items)

        self.assertEqual(payload["topic_summary"][0]["headline"], "Google bets on vibe design with Stitch")


if __name__ == "__main__":
    unittest.main()
