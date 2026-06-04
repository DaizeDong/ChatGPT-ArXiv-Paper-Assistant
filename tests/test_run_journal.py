from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from arxiv_assistant.utils.hotspot.run_journal import RunJournal


class TestRunJournal(unittest.TestCase):
    def test_append_and_flush_writes_one_jsonl_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            j = RunJournal(run_date="2026-06-03", journal_path=path)
            j.append("source_counts", {"hf_papers": 12, "ainews": 4})
            j.append_stage_timing("harvest", 1.5)
            j.append_stage_timing("embed", 0.25)
            j.record_dropped_stale_competitor(
                {"source_id": "agents_radar", "gate_date": "2023-01-01", "reason": "stale_curated"}
            )
            out = j.flush()
            self.assertEqual(out, path)

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            rec = json.loads(lines[0])
            self.assertEqual(rec["run_date"], "2026-06-03")
            self.assertEqual(rec["source_counts"], {"hf_papers": 12, "ainews": 4})
            self.assertEqual(rec["stage_timings"], {"harvest": 1.5, "embed": 0.25})
            self.assertEqual(len(rec["intentionally_dropped_stale_competitor"]), 1)
            self.assertEqual(
                rec["intentionally_dropped_stale_competitor"][0]["source_id"], "agents_radar"
            )

    def test_second_run_appends_a_new_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            RunJournal(run_date="2026-06-01", journal_path=path).flush()
            RunJournal(run_date="2026-06-02", journal_path=path).flush()
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["run_date"], "2026-06-01")
            self.assertEqual(json.loads(lines[1])["run_date"], "2026-06-02")

    def test_append_unknown_key_goes_to_extra(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            j = RunJournal(run_date="2026-06-03", journal_path=path)
            j.append("custom_flag", True)
            j.append("note", "x_yield=0")
            rec = json.loads(j.flush().read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(rec["extra"], {"custom_flag": True, "note": "x_yield=0"})

    def test_empty_journal_flushes_default_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            RunJournal(run_date="2026-06-03", journal_path=path).flush()
            rec = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(rec["source_counts"], {})
            self.assertEqual(rec["stage_timings"], {})
            self.assertEqual(rec["intentionally_dropped_stale_competitor"], [])
            self.assertEqual(rec["extra"], {})


class TestRecordDroppedStaleCompetitor(unittest.TestCase):
    """Task 8: module-level record_dropped_stale_competitor builds per-source aggregates."""

    def _item(self, provenance: str, url: str = "https://x.test/1", vfd: str | None = None):
        obj = type("I", (), {
            "provenance": provenance,
            "canonical_url": url,
            "url": url,
            "verified_first_date": vfd,
        })()
        return obj

    def test_per_source_seen_dropped_ratio(self) -> None:
        from arxiv_assistant.utils.hotspot.run_journal import record_dropped_stale_competitor

        competitor_items = [
            self._item("reuse:ainews", "https://x.test/1"),
            self._item("reuse:ainews", "https://x.test/2"),
        ]
        dropped = [self._item("reuse:ainews", "https://x.test/1", "2023-01-01")]
        eligible = [self._item("reuse:ainews", "https://x.test/2")]

        rec = record_dropped_stale_competitor("2026-06-03", eligible, dropped, competitor_items)

        self.assertEqual(rec["channel"], "intentionally_dropped_stale_competitor")
        self.assertEqual(rec["run_date"], "2026-06-03")
        self.assertEqual(rec["eligible_count"], 1)
        self.assertEqual(rec["dropped_count"], 1)
        ps = rec["per_source"]["reuse:ainews"]
        self.assertEqual(ps["seen"], 2)
        self.assertEqual(ps["dropped"], 1)
        self.assertAlmostEqual(ps["drop_ratio"], 0.5)

    def test_multiple_sources_isolated(self) -> None:
        from arxiv_assistant.utils.hotspot.run_journal import record_dropped_stale_competitor

        competitor_items = [
            self._item("reuse:ainews", "https://a.test/1"),
            self._item("reuse:ainews", "https://a.test/2"),
            self._item("reuse:horizon", "https://b.test/1"),
        ]
        dropped = [self._item("reuse:ainews", "https://a.test/1")]
        eligible = [
            self._item("reuse:ainews", "https://a.test/2"),
            self._item("reuse:horizon", "https://b.test/1"),
        ]

        rec = record_dropped_stale_competitor("2026-06-03", eligible, dropped, competitor_items)

        self.assertIn("reuse:ainews", rec["per_source"])
        self.assertIn("reuse:horizon", rec["per_source"])
        self.assertEqual(rec["per_source"]["reuse:ainews"]["dropped"], 1)
        self.assertEqual(rec["per_source"]["reuse:horizon"]["dropped"], 0)
        self.assertAlmostEqual(rec["per_source"]["reuse:horizon"]["drop_ratio"], 0.0)

    def test_dropped_items_carry_provenance_gate_date_reason(self) -> None:
        from arxiv_assistant.utils.hotspot.run_journal import record_dropped_stale_competitor

        dropped = [self._item("reuse:ainews", "https://x.test/1", "2023-01-01")]
        rec = record_dropped_stale_competitor("2026-06-03", [], dropped, dropped)

        di = rec["dropped_items"][0]
        self.assertEqual(di["provenance"], "reuse:ainews")
        self.assertEqual(di["gate_date"], "2023-01-01")
        self.assertEqual(di["reason"], "stale_beyond_max_age_or_unverified")
        self.assertEqual(di["canonical_url"], "https://x.test/1")

    def test_instance_method_still_works(self) -> None:
        """Stage-0 RunJournal.record_dropped_stale_competitor(entry) must remain intact."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            j = RunJournal(run_date="2026-06-04", journal_path=path)
            j.record_dropped_stale_competitor(
                {"source_id": "agents_radar", "gate_date": "2023-01-01", "reason": "stale_curated"}
            )
            rec = json.loads(j.flush().read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(len(rec["intentionally_dropped_stale_competitor"]), 1)
            self.assertEqual(rec["intentionally_dropped_stale_competitor"][0]["source_id"], "agents_radar")


if __name__ == "__main__":
    unittest.main()
