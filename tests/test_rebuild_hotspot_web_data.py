from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.rebuild_hotspot_web_data import rebuild_hotspot_web_data


class TestRebuildHotspotWebData(unittest.TestCase):
    def test_rebuild_hotspot_web_data_regenerates_daily_and_root_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "out"
            reports_root = output_root / "hot" / "reports"
            normalized_root = output_root / "hot" / "normalized"
            reports_root.mkdir(parents=True, exist_ok=True)
            normalized_root.mkdir(parents=True, exist_ok=True)

            report_template = {
                "generated_at": "2026-03-22T12:00:00+00:00",
                "mode": "heuristic",
                "summary": "Summary",
                "totals": {"raw_items": 1, "clusters": 1, "candidate_clusters": 1, "radar_clusters": 1},
                "costs": {"prompt": 0.0, "completion": 0.0, "total": 0.0},
                "source_stats": {"ainews": 1},
                "featured_topics": [
                    {
                        "TOPIC_ID": "topic",
                        "HEADLINE": "Headline",
                        "PRIMARY_CATEGORY": "Community Signal",
                        "DISPLAY_PRIORITY": 5.0,
                        "FINAL_SCORE": 5.0,
                        "HEAT": 4,
                        "QUALITY": 4,
                        "IMPORTANCE": 4,
                        "OCCURRENCE_SCORE": 3.0,
                        "source_names": ["AINews"],
                        "source_roles": ["community_heat"],
                        "source_types": ["roundup"],
                        "items": [{"title": "Social item", "url": "https://www.reddit.com/r/LocalLLaMA/comments/abc", "source_name": "AINews"}],
                        "WHY_IT_MATTERS": "Social signal.",
                        "SHORT_COMMENT": "Social signal.",
                    }
                ],
                "category_sections": [],
                "long_tail_sections": [],
                "watchlist": [],
                "x_buzz": [],
            }
            raw_item = [
                {
                    "source_id": "ainews",
                    "source_name": "AINews",
                    "source_role": "community_heat",
                    "source_type": "roundup",
                    "title": "Social item",
                    "summary": "Proxy social item.",
                    "url": "https://www.reddit.com/r/LocalLLaMA/comments/abc",
                    "canonical_url": "https://www.reddit.com/r/LocalLLaMA/comments/abc",
                    "published_at": None,
                    "tags": [],
                    "authors": [],
                    "metadata": {"activity": 120},
                }
            ]

            for date in ("2026-03-21", "2026-03-22"):
                report = dict(report_template)
                report["date"] = date
                (reports_root / f"{date}.json").write_text(json.dumps(report), encoding="utf-8")
                (normalized_root / f"{date}.json").write_text(json.dumps(raw_item), encoding="utf-8")

            rebuilt = rebuild_hotspot_web_data(output_root)

            self.assertEqual(rebuilt, ["2026-03-21", "2026-03-22"])
            root_index = json.loads((output_root / "web_data" / "hot" / "index.json").read_text(encoding="utf-8"))
            latest_daily = json.loads((output_root / "web_data" / "hot" / "2026-03-22.json").read_text(encoding="utf-8"))

            self.assertEqual(root_index["latest_date"], "2026-03-22")
            self.assertEqual(len(root_index["dates"]), 2)
            self.assertEqual(latest_daily["meta"]["previous_date"], "2026-03-21")

    def test_rebuild_hotspot_web_data_skips_dates_before_archive_start(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir) / "out"
            reports_root = output_root / "hot" / "reports"
            normalized_root = output_root / "hot" / "normalized"
            reports_root.mkdir(parents=True, exist_ok=True)
            normalized_root.mkdir(parents=True, exist_ok=True)

            report_template = {
                "generated_at": "2026-03-17T12:00:00+00:00",
                "mode": "heuristic",
                "summary": "Summary",
                "totals": {"raw_items": 1, "clusters": 1, "candidate_clusters": 1, "radar_clusters": 1},
                "costs": {"prompt": 0.0, "completion": 0.0, "total": 0.0},
                "source_stats": {"ainews": 1},
                "featured_topics": [
                    {
                        "TOPIC_ID": "topic",
                        "HEADLINE": "Headline",
                        "PRIMARY_CATEGORY": "Community Signal",
                        "DISPLAY_PRIORITY": 5.0,
                        "FINAL_SCORE": 5.0,
                        "HEAT": 4,
                        "QUALITY": 4,
                        "IMPORTANCE": 4,
                        "OCCURRENCE_SCORE": 3.0,
                        "source_names": ["AINews"],
                        "source_roles": ["community_heat"],
                        "source_types": ["roundup"],
                        "items": [{"title": "Social item", "url": "https://www.reddit.com/r/LocalLLaMA/comments/abc", "source_name": "AINews"}],
                        "WHY_IT_MATTERS": "Social signal.",
                        "SHORT_COMMENT": "Social signal.",
                    }
                ],
                "category_sections": [],
                "long_tail_sections": [],
                "watchlist": [],
                "x_buzz": [],
            }
            raw_item = [
                {
                    "source_id": "ainews",
                    "source_name": "AINews",
                    "source_role": "community_heat",
                    "source_type": "roundup",
                    "title": "Social item",
                    "summary": "Proxy social item.",
                    "url": "https://www.reddit.com/r/LocalLLaMA/comments/abc",
                    "canonical_url": "https://www.reddit.com/r/LocalLLaMA/comments/abc",
                    "published_at": None,
                    "tags": [],
                    "authors": [],
                    "metadata": {"activity": 120},
                }
            ]

            for date in ("2026-03-16", "2026-03-17"):
                report = dict(report_template)
                report["date"] = date
                (reports_root / f"{date}.json").write_text(json.dumps(report), encoding="utf-8")
                (normalized_root / f"{date}.json").write_text(json.dumps(raw_item), encoding="utf-8")

            rebuilt = rebuild_hotspot_web_data(output_root)

            self.assertEqual(rebuilt, ["2026-03-17"])
            root_index = json.loads((output_root / "web_data" / "hot" / "index.json").read_text(encoding="utf-8"))

            self.assertEqual(root_index["latest_date"], "2026-03-17")
            self.assertEqual([entry["date"] for entry in root_index["dates"]], ["2026-03-17"])
            self.assertFalse((output_root / "web_data" / "hot" / "2026-03-16.json").exists())
            self.assertTrue((output_root / "web_data" / "hot" / "2026-03-17.json").exists())


if __name__ == "__main__":
    unittest.main()
