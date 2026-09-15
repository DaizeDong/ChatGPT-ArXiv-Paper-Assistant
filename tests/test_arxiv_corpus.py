from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from arxiv_assistant.apis.corpus import (
    CorpusUnavailable,
    get_papers_from_corpus,
)


def _record(arxiv_id, created, categories=("cs.LG",), title=None):
    return {
        "arxiv_id": arxiv_id,
        "created": created,
        "title": title or f"Paper {arxiv_id}",
        "abstract": f"Abstract for {arxiv_id}",
        "authors": ["Ada Example"],
        "categories": list(categories),
    }


class CorpusReaderTest(unittest.TestCase):
    """The corpus is the source a backfill reads for old dates.

    Its whole point is that it selects on SUBMISSION date, so the tests that
    matter are the ones that fail if it ever drifts back to selecting on
    something else, or if it starts reporting an absent corpus as a quiet day.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.corpus = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _write(self, name, records):
        with open(self.corpus / name, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

    def test_only_papers_created_inside_the_window_come_back(self):
        self._write("cs_202607.jsonl", [
            _record("2607.00001", "2026-07-19"),
            _record("2607.00002", "2026-07-20"),
            _record("2607.00003", "2026-07-21"),
        ])
        _, papers = get_papers_from_corpus("cs.LG", (2026, 7, 20), (2026, 7, 20), corpus_dir=self.corpus)
        self.assertEqual([p.arxiv_id for p in papers], ["2607.00002"])

    def test_the_window_is_inclusive_at_both_ends(self):
        self._write("cs_202607.jsonl", [
            _record("2607.00001", "2026-07-20"),
            _record("2607.00002", "2026-07-21"),
            _record("2607.00003", "2026-07-22"),
        ])
        _, papers = get_papers_from_corpus("cs.LG", (2026, 7, 20), (2026, 7, 22), corpus_dir=self.corpus)
        self.assertEqual(len(papers), 3)

    def test_area_filter_matches_any_category_and_force_primary_narrows_it(self):
        self._write("cs_202607.jsonl", [
            _record("2607.00001", "2026-07-20", categories=("cs.AI", "cs.LG")),
            _record("2607.00002", "2026-07-20", categories=("cs.LG", "cs.AI")),
            _record("2607.00003", "2026-07-20", categories=("cs.CV",)),
        ])
        _, any_cat = get_papers_from_corpus("cs.LG", (2026, 7, 20), (2026, 7, 20), corpus_dir=self.corpus)
        self.assertEqual({p.arxiv_id for p in any_cat}, {"2607.00001", "2607.00002"})

        _, primary = get_papers_from_corpus(
            "cs.LG", (2026, 7, 20), (2026, 7, 20), force_primary=True, corpus_dir=self.corpus
        )
        self.assertEqual([p.arxiv_id for p in primary], ["2607.00002"])

    def test_a_paper_present_in_two_parts_is_returned_once(self):
        self._write("cs_202607.jsonl", [_record("2607.00001", "2026-07-20")])
        self._write("cs_202608.jsonl", [_record("2607.00001", "2026-07-20")])
        _, papers = get_papers_from_corpus("cs.LG", (2026, 7, 20), (2026, 7, 20), corpus_dir=self.corpus)
        self.assertEqual(len(papers), 1)

    def test_a_truncated_final_line_costs_one_paper_not_the_day(self):
        with open(self.corpus / "cs_202607.jsonl", "w", encoding="utf-8") as handle:
            handle.write(json.dumps(_record("2607.00001", "2026-07-20")) + "\n")
            handle.write('{"arxiv_id": "2607.00002", "creat')
        _, papers = get_papers_from_corpus("cs.LG", (2026, 7, 20), (2026, 7, 20), corpus_dir=self.corpus)
        self.assertEqual([p.arxiv_id for p in papers], ["2607.00001"])

    def test_a_missing_corpus_raises_instead_of_reporting_an_empty_day(self):
        # The whole failure mode this guards: "no corpus" rendering as "no
        # papers today", which writes a healthy-looking empty bundle.
        with self.assertRaises(CorpusUnavailable):
            get_papers_from_corpus("cs.LG", (2026, 7, 20), (2026, 7, 20), corpus_dir=self.corpus)

    def test_an_empty_day_inside_a_present_corpus_is_not_an_error(self):
        self._write("cs_202607.jsonl", [_record("2607.00001", "2026-07-20")])
        entries, papers = get_papers_from_corpus("cs.LG", (2026, 7, 25), (2026, 7, 25), corpus_dir=self.corpus)
        self.assertEqual((entries, papers), ([], []))


if __name__ == "__main__":
    unittest.main()
