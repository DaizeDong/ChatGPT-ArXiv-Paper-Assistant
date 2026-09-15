"""Tests for the BM25 retriever and the archive query tool."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arxiv_assistant.reader.retrieval import BM25Index, Document, tokenize
from arxiv_assistant.utils import llm_gateway
from arxiv_assistant.utils.agent_runner import AgentError
from scripts.query_archive import (
    AnswerStatus,
    _verify_agent_answer,
    load_corpus,
    query_archive,
    render_text,
)


def _doc(doc_id: str, text: str, url: str = "", title: str = "", date: str = "2026-08-01") -> Document:
    return Document(doc_id=doc_id, kind="hotspot", date=date, title=title or doc_id, text=text, url=url)


class TokenizeTest(unittest.TestCase):
    def test_word_tokens_are_case_folded(self):
        self.assertEqual(tokenize("Mixture of EXPERTS"), ["mixture", "of", "experts"])

    def test_punctuation_and_underscores_split(self):
        self.assertEqual(tokenize("agent_memory, v2.1"), ["agent", "memory", "v2", "1"])

    def test_cjk_run_becomes_bigrams(self):
        # The bug this guards: \w+ would return ["多智能体"] as ONE token, so the query
        # "智能体" could never match. Bigrams make the overlap visible.
        self.assertEqual(tokenize("多智能体"), ["多智", "智能", "能体"])

    def test_single_cjk_character_survives_as_unigram(self):
        self.assertEqual(tokenize("模"), ["模"])

    def test_mixed_script_does_not_swallow_cjk_into_a_word_token(self):
        self.assertEqual(tokenize("Claude智能体x"), ["claude", "智能", "能体", "x"])


class BM25Test(unittest.TestCase):
    def setUp(self):
        self.docs = [
            _doc("a", "mixture of experts routing collapse", url="https://example.com/a"),
            _doc("b", "mixture of experts", url="https://example.com/b"),
            _doc("c", "diffusion image generation", url="https://example.com/c"),
        ]
        self.index = BM25Index(self.docs)

    def test_ranks_the_denser_match_first(self):
        hits = self.index.search("routing collapse", top_k=5)
        self.assertEqual([hit.document.doc_id for hit in hits], ["a"])

    def test_non_matching_documents_are_excluded_not_scored_zero(self):
        hits = self.index.search("mixture experts", top_k=5)
        self.assertEqual({hit.document.doc_id for hit in hits}, {"a", "b"})

    def test_ranks_are_dense_and_one_based(self):
        hits = self.index.search("mixture", top_k=5)
        self.assertEqual([hit.rank for hit in hits], list(range(1, len(hits) + 1)))

    def test_idf_is_never_negative_so_a_match_never_scores_below_zero(self):
        # "mixture" appears in 2 of 3 docs; the textbook IDF would go negative here.
        for hit in self.index.search("mixture", top_k=5):
            self.assertGreater(hit.score, 0.0)

    def test_matched_terms_report_what_actually_fired(self):
        hit = self.index.search("mixture unicorn", top_k=1)[0]
        self.assertEqual(hit.matched_terms, ("mixture",))

    def test_chinese_query_retrieves_chinese_document(self):
        index = BM25Index([
            _doc("zh", "阿里杀进Agent上下文战场", url="https://example.com/zh"),
            _doc("en", "context engineering for agents", url="https://example.com/en"),
        ])
        hits = index.search("上下文", top_k=5)
        self.assertEqual([hit.document.doc_id for hit in hits], ["zh"])

    def test_empty_index_is_distinguishable_from_no_hits(self):
        empty = BM25Index([])
        self.assertTrue(empty.is_empty)
        self.assertEqual(empty.search("anything"), [])
        self.assertFalse(self.index.is_empty)
        self.assertEqual(self.index.search("nothing matches this"), [])


class ArchiveFixture:
    """Builds a throwaway archive tree with the real on-disk layout."""

    def __init__(self, root: Path):
        self.root = root

    def paper_day(self, day: str, mapping: dict) -> None:
        path = self.root / "out" / "json" / day[:7] / f"{day}-output.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(mapping), encoding="utf-8")

    def hotspot_day(self, day: str, report: dict) -> None:
        path = self.root / "out" / "hot" / "reports" / f"{day}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")


class CorpusLoadTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.fixture = ArchiveFixture(self.root)
        self.addCleanup(self._tmp.cleanup)

    def test_counts_missing_and_empty_days_separately(self):
        # 08-01 has papers, 08-02 is the literal "{}" outage fingerprint, 08-03 is absent.
        self.fixture.paper_day("2026-08-01", {"2608.00001": {"title": "T", "abstract": "A", "COMMENT": "C"}})
        self.fixture.paper_day("2026-08-02", {})
        from datetime import date

        docs, coverage = load_corpus(self.root, date(2026, 8, 1), date(2026, 8, 3))
        self.assertEqual(len(docs), 1)
        self.assertEqual(coverage.paper_days_present, 1)
        self.assertEqual(coverage.paper_days_empty, 1)
        self.assertEqual(coverage.paper_days_missing, 1)
        self.assertEqual(coverage.hotspot_days_missing, 3)
        self.assertEqual(docs[0].url, "https://arxiv.org/abs/2608.00001")

    def test_reads_every_topic_bucket_and_dedupes_by_topic_id(self):
        topic = {
            "TOPIC_ID": "t1",
            "HEADLINE": "Router collapse",
            "summary": "s",
            "WHY_IT_MATTERS": "w",
            "KEY_TAKEAWAYS": ["k1", "k2"],
            "items": [{"url": "https://example.com/t1"}],
        }
        self.fixture.hotspot_day(
            "2026-08-01",
            {
                "featured_topics": [topic],
                "category_sections": [{"topics": [topic]}],  # same topic id -> deduped
                "long_tail_sections": [{"topics": [{"TOPIC_ID": "t2", "title": "Tail", "summary": "tail"}]}],
                "watchlist": [{"TOPIC_ID": "t3", "title": "Watch", "summary": "watch"}],
            },
        )
        from datetime import date

        docs, coverage = load_corpus(self.root, date(2026, 8, 1), date(2026, 8, 1))
        self.assertEqual(sorted(doc.doc_id for doc in docs), [
            "hotspot:2026-08-01:t1",
            "hotspot:2026-08-01:t2",
            "hotspot:2026-08-01:t3",
        ])
        self.assertEqual(coverage.hotspot_docs, 3)
        first = next(doc for doc in docs if doc.doc_id.endswith("t1"))
        self.assertIn("k1 k2", first.text)
        self.assertEqual(first.url, "https://example.com/t1")

    def test_missing_optional_keys_do_not_crash(self):
        self.fixture.hotspot_day("2026-08-01", {"date": "2026-08-01", "summary": "only a summary"})
        from datetime import date

        docs, coverage = load_corpus(self.root, date(2026, 8, 1), date(2026, 8, 1))
        self.assertEqual(docs, [])
        self.assertEqual(coverage.hotspot_days_present, 1)

    def test_corrupt_file_is_recorded_not_swallowed(self):
        path = self.root / "out" / "hot" / "reports" / "2026-08-01.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        from datetime import date

        _docs, coverage = load_corpus(self.root, date(2026, 8, 1), date(2026, 8, 1))
        self.assertEqual(len(coverage.unreadable_files), 1)


class VerifierTest(unittest.TestCase):
    def setUp(self):
        docs = [_doc("a", "text a", url="https://example.com/a"), _doc("b", "text b", url="https://example.com/b")]
        self.hits = BM25Index(docs).search("text", top_k=5)

    def test_keeps_a_conclusion_whose_citation_was_retrieved(self):
        clean = _verify_agent_answer(
            {"conclusions": [{"text": "A is true.", "citations": ["https://example.com/a"]}]}, self.hits
        )
        self.assertEqual(len(clean), 1)
        self.assertEqual(clean[0]["citations"][0]["doc_id"], "a")

    def test_drops_a_fabricated_url_and_keeps_the_real_one(self):
        clean = _verify_agent_answer(
            {
                "conclusions": [
                    {"text": "A is true.", "citations": ["https://evil.example/made-up", "https://example.com/b"]}
                ]
            },
            self.hits,
        )
        self.assertEqual([c["url"] for c in clean[0]["citations"]], ["https://example.com/b"])

    def test_rejects_when_every_citation_was_fabricated(self):
        self.assertIsNone(
            _verify_agent_answer(
                {"conclusions": [{"text": "A is true.", "citations": ["https://evil.example/x"]}]}, self.hits
            )
        )

    def test_drops_an_uncited_conclusion(self):
        clean = _verify_agent_answer(
            {
                "conclusions": [
                    {"text": "unsourced", "citations": []},
                    {"text": "sourced", "citations": ["https://example.com/a"]},
                ]
            },
            self.hits,
        )
        self.assertEqual([c["text"] for c in clean], ["sourced"])

    def test_empty_conclusion_list_is_an_answer_not_a_rejection(self):
        self.assertEqual(_verify_agent_answer({"conclusions": []}, self.hits), [])

    def test_malformed_payload_is_rejected(self):
        self.assertIsNone(_verify_agent_answer({"conclusions": "nope"}, self.hits))
        self.assertIsNone(_verify_agent_answer(["conclusions"], self.hits))


class QueryArchiveTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        fixture = ArchiveFixture(self.root)
        fixture.hotspot_day(
            "2026-08-01",
            {
                "featured_topics": [
                    {
                        "TOPIC_ID": "t1",
                        "HEADLINE": "Router collapse measured",
                        "summary": "mixture of experts routing collapse threshold",
                        "items": [{"url": "https://example.com/t1"}],
                    }
                ]
            },
        )
        fixture.paper_day("2026-08-01", {})

    def test_no_llm_returns_hits_and_skipped_status(self):
        result = query_archive(
            "routing collapse", since="2026-08-01", until="2026-08-01", archive_root=self.root, use_llm=False
        )
        self.assertEqual(result["answer"]["status"], AnswerStatus.SKIPPED)
        self.assertEqual(result["hits"][0]["doc_id"], "hotspot:2026-08-01:t1")
        self.assertEqual(result["coverage"]["paper_days_empty"], 1)

    def test_answered_path_carries_verified_citations(self):
        def agent(prompt, **kwargs):
            self.assertIn("https://example.com/t1", prompt)
            return {"conclusions": [{"text": "The threshold was measured.", "citations": ["https://example.com/t1"]}]}

        result = query_archive(
            "routing collapse",
            since="2026-08-01",
            until="2026-08-01",
            archive_root=self.root,
            model="test-model",
            agent_fn=agent,
        )
        self.assertEqual(result["answer"]["status"], AnswerStatus.ANSWERED)
        self.assertEqual(result["answer"]["conclusions"][0]["citations"][0]["url"], "https://example.com/t1")

    def test_empty_window_says_no_data_not_no_hits(self):
        result = query_archive(
            "routing collapse", since="2020-01-01", until="2020-01-03", archive_root=self.root, use_llm=False
        )
        self.assertEqual(result["answer"]["status"], AnswerStatus.NO_DATA)
        self.assertIn("NOTHING was searched", result["answer"]["note"])
        self.assertEqual(result["coverage"]["total_docs"], 0)

    def test_real_corpus_with_no_match_is_no_hits_not_no_data(self):
        result = query_archive(
            "zzzzz nonexistent term",
            since="2026-08-01",
            until="2026-08-01",
            archive_root=self.root,
            model="test-model",
            agent_fn=lambda *a, **k: self.fail("agent must not run with zero hits"),
        )
        self.assertEqual(result["answer"]["status"], AnswerStatus.NO_HITS)
        self.assertGreater(result["coverage"]["total_docs"], 0)

    def test_agent_failure_is_unavailable_not_an_empty_answer(self):
        def dead_agent(prompt, **kwargs):
            raise AgentError("claude binary not found")

        result = query_archive(
            "routing collapse",
            since="2026-08-01",
            until="2026-08-01",
            archive_root=self.root,
            model="test-model",
            agent_fn=dead_agent,
        )
        self.assertEqual(result["answer"]["status"], AnswerStatus.UNAVAILABLE)
        self.assertIn("claude binary not found", result["answer"]["note"])

    def test_fully_hallucinated_answer_is_rejected(self):
        result = query_archive(
            "routing collapse",
            since="2026-08-01",
            until="2026-08-01",
            archive_root=self.root,
            model="test-model",
            agent_fn=lambda *a, **k: {"conclusions": [{"text": "x", "citations": ["https://evil.example/x"]}]},
        )
        self.assertEqual(result["answer"]["status"], AnswerStatus.REJECTED)

    def test_answer_says_which_backend_produced_it(self):
        """An answer with no provenance is unauditable: a week later there is no way
        to tell whether it came from the chain or from the local agent fallback."""
        result = query_archive(
            "routing collapse",
            since="2026-08-01",
            until="2026-08-01",
            archive_root=self.root,
            model="test-model",
            agent_fn=lambda *a, **k: {
                "conclusions": [
                    {"text": "The threshold was measured.", "citations": ["https://example.com/t1"]}
                ]
            },
        )
        self.assertEqual(result["answer"]["status"], AnswerStatus.ANSWERED)
        self.assertEqual(result["answer"]["backend"], llm_gateway.BACKEND_AGENT)
        self.assertEqual(result["answer"]["provider"], "claude")
        self.assertIn("answered by: backend=agent", render_text(result))

    def test_llmcall_backend_is_used_and_reported_when_injected(self):
        """The llmcall chain path, end to end, without touching the network. The
        INV6 verifier still runs on it: the fabricated citation is dropped and the
        conclusion with a real one survives."""

        class _Result:
            data = {
                "conclusions": [
                    {"text": "Measured.", "citations": ["https://example.com/t1"]},
                    {"text": "Invented.", "citations": ["https://evil.example/x"]},
                ]
            }
            text = ""
            provider = "codexg"
            error = ""

            def __bool__(self):
                return True

        result = query_archive(
            "routing collapse",
            since="2026-08-01",
            until="2026-08-01",
            archive_root=self.root,
            llmcall_fn=lambda prompt, **kwargs: _Result(),
        )
        answer = result["answer"]
        self.assertEqual(answer["status"], AnswerStatus.ANSWERED)
        self.assertEqual(answer["backend"], llm_gateway.BACKEND_LLMCALL)
        self.assertEqual(answer["provider"], "codexg")
        self.assertEqual(len(answer["conclusions"]), 1)
        self.assertEqual(answer["conclusions"][0]["text"], "Measured.")

    def test_dead_chain_is_unavailable_not_an_empty_answer(self):
        """Same guarantee as the agent path, on the other backend: a chain where
        every provider failed must not render as "nothing to say"."""

        class _Dead:
            data = None
            text = ""
            provider = ""
            error = "all providers exhausted"

            def __bool__(self):
                return False

        result = query_archive(
            "routing collapse",
            since="2026-08-01",
            until="2026-08-01",
            archive_root=self.root,
            llmcall_fn=lambda prompt, **kwargs: _Dead(),
        )
        self.assertEqual(result["answer"]["status"], AnswerStatus.UNAVAILABLE)
        self.assertNotEqual(result["answer"]["status"], AnswerStatus.NO_HITS)
        self.assertIn("all providers exhausted", result["answer"]["note"])

    def test_no_llm_never_touches_the_gateway(self):
        """--no-llm must work with no model access at all, so the ledger must show
        that not one call was even attempted."""
        llm_gateway.LEDGER.reset()
        result = query_archive(
            "routing collapse",
            since="2026-08-01",
            until="2026-08-01",
            archive_root=self.root,
            use_llm=False,
            call_fn=lambda *a, **k: self.fail("--no-llm must not reach the gateway"),
        )
        self.assertEqual(result["answer"]["status"], AnswerStatus.SKIPPED)
        self.assertEqual(llm_gateway.LEDGER.attempted, 0)
        self.assertNotIn("backend", result["answer"])

    def test_operator_errors_raise_instead_of_returning_nothing(self):
        with self.assertRaises(ValueError):
            query_archive("q", since="08/01/2026", archive_root=self.root, use_llm=False)
        with self.assertRaises(ValueError):
            query_archive("q", since="2026-08-05", until="2026-08-01", archive_root=self.root, use_llm=False)
        with tempfile.TemporaryDirectory() as bogus:
            with self.assertRaises(ValueError):
                query_archive("q", since="2026-08-01", archive_root=bogus, use_llm=False)


if __name__ == "__main__":
    unittest.main()
