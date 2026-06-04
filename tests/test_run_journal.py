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


if __name__ == "__main__":
    unittest.main()
