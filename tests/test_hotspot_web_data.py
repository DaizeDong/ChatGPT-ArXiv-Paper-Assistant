from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from arxiv_assistant.utils.hotspot_schema import HotspotItem
from arxiv_assistant.utils.hotspot_web_data import build_daily_hotspot_web_payload, write_hotspot_web_data


class TestHotspotWebData(unittest.TestCase):
    def test_daily_payload_groups_items_by_source_family_and_links_topics(self) -> None:
        report = {
            "date": "2026-03-21",
            "generated_at": "2026-03-21T12:00:00+00:00",
            "mode": "heuristic",
            "summary": "A dense day across social, papers, and official updates.",
            "totals": {"raw_items": 3, "clusters": 3, "candidate_clusters": 3, "radar_clusters": 3},
            "costs": {"prompt": 0.0, "completion": 0.0, "total": 0.0},
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

    def test_write_hotspot_web_data_builds_root_month_and_year_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "out"
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


if __name__ == "__main__":
    unittest.main()
