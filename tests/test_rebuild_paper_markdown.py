import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from scripts.rebuild_paper_markdown import rebuild_paper_markdown


class RebuildPaperMarkdownTests(unittest.TestCase):
    def test_rebuild_paper_markdown_backfills_topics_and_grouped_markdown(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out_root = Path(temp_dir) / "out"
            json_dir = out_root / "json" / "2026-03"
            json_dir.mkdir(parents=True)
            (out_root / "md").mkdir(parents=True)

            source_payload = {
                "2501.00001": {
                    "arxiv_id": "2501.00001",
                    "title": "World model planning",
                    "authors": ["A"],
                    "abstract": "world model for exploration",
                    "COMMENT": "Model-based RL with imagination.",
                    "SCORE": 18,
                    "RELEVANCE": 9,
                    "NOVELTY": 9,
                }
            }
            (json_dir / "2026-03-31-output.json").write_text(
                json.dumps(source_payload, indent=2),
                encoding="utf-8",
            )

            rebuilt_dates = rebuild_paper_markdown(out_root)

            self.assertEqual(rebuilt_dates, ["2026-03-31"])
            enriched_payload = json.loads((json_dir / "2026-03-31-output.json").read_text(encoding="utf-8"))
            bundle_payload = json.loads((json_dir / "2026-03-31-daily-papers.json").read_text(encoding="utf-8"))
            daily_md = (out_root / "md" / "2026-03" / "2026-03-31-latest.md").read_text(encoding="utf-8")
            latest_md = (out_root / "latest.md").read_text(encoding="utf-8")

            self.assertIn("PRIMARY_TOPIC_ID", enriched_payload["2501.00001"])
            self.assertEqual(bundle_payload["meta"]["date"], "2026-03-31")
            self.assertIn("World Models, Exploration, and Open-Ended Reinforcement Learning", daily_md)
            self.assertEqual(daily_md, latest_md)


if __name__ == "__main__":
    unittest.main()
