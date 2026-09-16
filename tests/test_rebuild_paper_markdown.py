import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from arxiv_assistant.paper_topics import get_topic_registry
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
                    "title": "Expert routing and load balancing for sparse MoE training",
                    "authors": ["A"],
                    "abstract": "A mixture of experts router with a new load balancing auxiliary loss.",
                    "COMMENT": "New MoE routing and load balancing mechanism.",
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
            # Ask the registry for the label rather than pinning it: topic labels
            # are reworded whenever the feed is retuned, and a literal here fails
            # for a rename that the renderer handled correctly.
            self.assertIn(get_topic_registry().get("moe_training").label, daily_md)
            self.assertEqual(
                enriched_payload["2501.00001"]["PRIMARY_TOPIC_ID"], "moe_training"
            )
            self.assertEqual(daily_md, latest_md)


if __name__ == "__main__":
    unittest.main()
