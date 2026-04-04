import json
import tempfile
import unittest
from pathlib import Path

from arxiv_assistant.renderers.paper.monthly_summary import (
    MONTH_TOPIC_ORDER,
    build_monthly_summary_data,
)


class MonthlySummaryTopicsTests(unittest.TestCase):
    def test_generated_monthly_summary_accepts_old_and_new_topic_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            json_root = temp_root / "json"
            monthly_root = temp_root / "monthly"
            monthly_root.mkdir(parents=True)

            payload = {
                "papers": [
                    {
                        "arxiv_id": "2501.00001",
                        "title": "Memory paper",
                        "SOURCE_DATE": [2026, 3, 30],
                        "KEEP_IN_MONTHLY": True,
                        "PRIMARY_TOPIC_ID": "memory_systems",
                    },
                    {
                        "arxiv_id": "2501.00002",
                        "title": "Efficiency paper",
                        "abstract": "distributed training and memory-efficient inference",
                        "SOURCE_DATE": [2026, 3, 29],
                        "KEEP_IN_MONTHLY": True,
                        "PRIMARY_CATEGORY": "High Performance Computing",
                    },
                ]
            }
            (monthly_root / "2026-03-summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

            summary = build_monthly_summary_data(json_root, monthly_root, {})

            self.assertIn((2026, 3), summary)
            self.assertEqual(list(summary[(2026, 3)].keys()), MONTH_TOPIC_ORDER)
            self.assertEqual(len(summary[(2026, 3)]["memory_systems"]), 1)
            self.assertEqual(len(summary[(2026, 3)]["efficiency_scaling"]), 1)


if __name__ == "__main__":
    unittest.main()
