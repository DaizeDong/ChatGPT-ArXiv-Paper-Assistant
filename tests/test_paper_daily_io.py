import json
import tempfile
import unittest
from pathlib import Path

from arxiv_assistant.paper_daily_io import discover_daily_json, extract_paper_mapping, write_daily_json_outputs


class PaperDailyIOTests(unittest.TestCase):
    def test_extract_paper_mapping_supports_flat_and_bundle_payloads(self):
        flat_payload = {
            "2501.00001": {"arxiv_id": "2501.00001", "title": "Flat"},
        }
        bundle_payload = {
            "schema_version": 2,
            "papers": {
                "2501.00002": {"arxiv_id": "2501.00002", "title": "Bundle"},
            },
        }

        self.assertEqual(list(extract_paper_mapping(flat_payload).keys()), ["2501.00001"])
        self.assertEqual(list(extract_paper_mapping(bundle_payload).keys()), ["2501.00002"])

    def test_discover_daily_json_prefers_output_payload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            json_root = Path(temp_dir) / "json" / "2026-03"
            json_root.mkdir(parents=True)
            (json_root / "2026-03-31-daily-papers.json").write_text("{}", encoding="utf-8")
            (json_root / "2026-03-31-hotspot-papers.json").write_text("{}", encoding="utf-8")
            (json_root / "2026-03-31-output.json").write_text("{}", encoding="utf-8")

            discovered = discover_daily_json(Path(temp_dir) / "json")

            self.assertEqual(discovered[(2026, 3, 31)].name, "2026-03-31-output.json")

    def test_write_daily_json_outputs_writes_bundle_with_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = write_daily_json_outputs(
                temp_dir,
                (2026, 3, 31),
                {
                    "2501.00001": {
                        "arxiv_id": "2501.00001",
                        "title": "Memory paper",
                        "authors": ["A"],
                        "abstract": "episodic memory",
                        "SCORE": 18,
                        "RELEVANCE": 9,
                        "NOVELTY": 9,
                        "PRIMARY_TOPIC_ID": "memory_systems",
                    }
                },
            )

            json_dir = Path(temp_dir) / "json" / "2026-03"
            output_payload = json.loads((json_dir / "2026-03-31-output.json").read_text(encoding="utf-8"))
            bundle_payload = json.loads((json_dir / "2026-03-31-daily-papers.json").read_text(encoding="utf-8"))

            self.assertIn("PRIMARY_TOPIC_ID", output_payload["2501.00001"])
            self.assertEqual(bundle_payload["diagnostics"]["total_papers"], 1)
            self.assertEqual(bundle["diagnostics"]["topic_counts"][3]["paper_count"], 1)


if __name__ == "__main__":
    unittest.main()
