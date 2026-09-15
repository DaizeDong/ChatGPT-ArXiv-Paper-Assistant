"""Tests for ``arxiv_assistant.reader.delta``.

The invariant under test, and the reason this file is long: "nothing crossed the
threshold" and "scoring never ran" must be DIFFERENT outputs. This repo's paper
pipeline emitted an empty ``{}`` archive every day for three months because every
model call raised, was swallowed, and left behind something that looked exactly
like a quiet day. So every failure path below asserts ``DeltaStatus.UNAVAILABLE``
with a note that says WHY, and explicitly asserts it is not a score of zero.

No network, no subprocess: ``score_candidates`` takes ``agent_fn``, and the fakes
here are the only transport.

Archive fixtures live in ``tests/fixtures/reader/`` and are trimmed excerpts of
real files from the data branch (``out/json/2026-06/2026-06-04-output.json`` and
``out/hot/reports/2026-09-08.json``). They are copied in rather than read from a
sibling worktree so the suite runs on a bare clone; ``ArchiveWorktreeTest`` below
re-runs the same assertions against the full files when that worktree happens to
be present.
"""
from __future__ import annotations

import json
import unittest
import os
from pathlib import Path

from arxiv_assistant.reader.delta import (
    DeltaCandidate,
    DeltaStatus,
    DeltaVerdict,
    MAX_REASON_WORDS,
    annotate,
    candidates_from_hotspot_report,
    candidates_from_paper_mapping,
    dumps_verdicts,
    rank,
    score_candidates,
    verify_verdict_row,
)
from arxiv_assistant.reader.questions import parse_question_document
from arxiv_assistant.utils import llm_gateway
from arxiv_assistant.utils.agent_runner import AgentError

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "reader"
# Drift detection against the FULL archive. A hardcoded absolute path would make
# this layer silently inert on any other machine, and a skip is not a pass -- so
# the path comes from the environment, and when the variable IS set but wrong we
# fail loudly instead of skipping. Set it to a checkout of the auto_update branch:
#   ARXIV_ASSISTANT_ARCHIVE_ROOT=/path/to/auto_update/worktree
_ARCHIVE_ENV = "ARXIV_ASSISTANT_ARCHIVE_ROOT"
_archive_raw = os.environ.get(_ARCHIVE_ENV, "").strip()
ARCHIVE_ROOT = Path(_archive_raw) if _archive_raw else None
if ARCHIVE_ROOT is not None and not ARCHIVE_ROOT.is_dir():
    raise RuntimeError(
        f"{_ARCHIVE_ENV} is set to {ARCHIVE_ROOT}, which is not a directory. "
        "Unset it to skip the archive drift checks, or point it at a real "
        "auto_update checkout -- silently skipping a check you asked for is worse "
        "than not asking for it."
    )

QUESTION_DOC = """# Q1: Do continuous bit-widths beat integer quantization?

## 当前看法

Integer bit-widths are a deployment artifact, not an accuracy limit.

## 什么会让我改看法

A kernel benchmark showing decode overhead eats the memory win.
"""

EMPTY_QUESTION_DOC = """# Q1: untouched template

## 当前看法

<!-- write your belief here -->

## 想做的实验

<在这里写下你想做的实验>
"""


def _questions(n: int = 1):
    """n populated question documents, ids q1..qn."""
    return [
        parse_question_document(
            QUESTION_DOC.replace("# Q1:", f"# Q{i}:"), fallback_id=f"q{i}"
        )
        for i in range(1, n + 1)
    ]


def _candidates(n: int = 2):
    return [
        DeltaCandidate(
            candidate_id=f"paper:{i}",
            kind="paper",
            title=f"Title {i}",
            text=f"Abstract {i}",
            url=f"https://arxiv.org/abs/{i}",
            tiebreak=float(i),
        )
        for i in range(n)
    ]


GOOD_REASON = "reports 3.4 effective bits at 2-bit memory cost"


def _row(index: int = 0, **overrides):
    row = {
        "index": index,
        "question_id": "q1",
        "field": "current_view",
        "delta_score": 8,
        "one_line_reason": GOOD_REASON,
    }
    row.update(overrides)
    return row


class VerifyVerdictRowTest(unittest.TestCase):
    """One assertion per rejection rule: a verifier that rejects everything and a
    verifier that rejects nothing both pass a suite that only checks the happy path."""

    def setUp(self) -> None:
        self.candidates = _candidates(2)
        self.kwargs = {"valid_question_ids": ["q1"], "candidates": self.candidates}

    def _verify(self, row):
        return verify_verdict_row(row, **self.kwargs)

    def test_accepts_a_valid_row(self) -> None:
        verdict = self._verify(_row(index=1))
        self.assertIsNotNone(verdict)
        self.assertEqual(verdict.status, DeltaStatus.SCORED)
        self.assertTrue(verdict.passes)
        self.assertEqual(verdict.candidate_id, "paper:1")
        self.assertEqual(verdict.delta_score, 8)
        self.assertEqual(verdict.question_id, "q1")
        self.assertEqual(verdict.field, "current_view")
        self.assertEqual(verdict.one_line_reason, GOOD_REASON)

    def test_question_id_is_normalised_before_matching(self) -> None:
        verdict = self._verify(_row(question_id=" Q1 "))
        self.assertEqual(verdict.status, DeltaStatus.SCORED)
        self.assertEqual(verdict.question_id, "q1")

    def test_rejects_unknown_question_id(self) -> None:
        verdict = self._verify(_row(question_id="q9"))
        self.assertEqual(verdict.status, DeltaStatus.REJECTED)
        self.assertIn("question_id", verdict.note)

    def test_rejects_unknown_field(self) -> None:
        verdict = self._verify(_row(field="vibes"))
        self.assertEqual(verdict.status, DeltaStatus.REJECTED)
        self.assertIn("field", verdict.note)

    def test_rejects_score_above_range(self) -> None:
        verdict = self._verify(_row(delta_score=11))
        self.assertEqual(verdict.status, DeltaStatus.REJECTED)
        self.assertIn("out of range", verdict.note)

    def test_rejects_score_below_range(self) -> None:
        verdict = self._verify(_row(delta_score=-1))
        self.assertEqual(verdict.status, DeltaStatus.REJECTED)
        self.assertIn("out of range", verdict.note)

    def test_rejects_bool_score(self) -> None:
        """True is an int in Python; a model that returns a boolean has not scored."""
        verdict = self._verify(_row(delta_score=True))
        self.assertEqual(verdict.status, DeltaStatus.REJECTED)
        self.assertIn("not an integer", verdict.note)

    def test_rejects_float_score(self) -> None:
        verdict = self._verify(_row(delta_score=7.5))
        self.assertEqual(verdict.status, DeltaStatus.REJECTED)
        self.assertIn("not an integer", verdict.note)

    def test_rejects_string_score(self) -> None:
        verdict = self._verify(_row(delta_score="8"))
        self.assertEqual(verdict.status, DeltaStatus.REJECTED)
        self.assertIn("not an integer", verdict.note)

    def test_rejects_empty_reason(self) -> None:
        verdict = self._verify(_row(one_line_reason="   "))
        self.assertEqual(verdict.status, DeltaStatus.REJECTED)
        self.assertIn("empty", verdict.note)

    def test_rejects_over_long_reason(self) -> None:
        verdict = self._verify(_row(one_line_reason="word " * (MAX_REASON_WORDS + 1)))
        self.assertEqual(verdict.status, DeltaStatus.REJECTED)
        self.assertIn("too long", verdict.note)

    def test_rejects_generic_reason_at_high_score(self) -> None:
        verdict = self._verify(
            _row(delta_score=9, one_line_reason="generally interesting for this line of work")
        )
        self.assertEqual(verdict.status, DeltaStatus.REJECTED)
        self.assertIn("generic", verdict.note)

    def test_allows_generic_reason_at_low_score(self) -> None:
        """A low score may honestly say "not much here"; only a high score must name
        something. Without this the previous test would also pass if the verifier
        rejected generic reasons unconditionally."""
        verdict = self._verify(
            _row(delta_score=2, one_line_reason="generally interesting for this line of work")
        )
        self.assertEqual(verdict.status, DeltaStatus.SCORED)

    def test_rejects_index_out_of_range(self) -> None:
        self.assertIsNone(self._verify(_row(index=5)))

    def test_rejects_negative_index(self) -> None:
        self.assertIsNone(self._verify(_row(index=-1)))

    def test_rejects_non_int_index(self) -> None:
        self.assertIsNone(self._verify(_row(index="0")))
        self.assertIsNone(self._verify(_row(index=True)))

    def test_rejects_non_dict_row(self) -> None:
        self.assertIsNone(self._verify(["index", 0]))
        self.assertIsNone(self._verify("index 0"))
        self.assertIsNone(self._verify(None))


# ---------------------------------------------------------------------------
# score_candidates: transport fakes
# ---------------------------------------------------------------------------


class _Recorder:
    """agent_fn stand-in: records prompts, replays a scripted response per call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        response = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(response, Exception):
            raise response
        return response


class ScoreCandidatesTest(unittest.TestCase):
    def test_happy_path_scores_every_candidate(self) -> None:
        candidates = _candidates(2)
        agent = _Recorder([{"verdicts": [_row(0), _row(1, delta_score=3)]}])
        verdicts = score_candidates(
            candidates, _questions(), model="m", agent_fn=agent
        )
        self.assertEqual(set(verdicts), {"paper:0", "paper:1"})
        self.assertTrue(all(v.status == DeltaStatus.SCORED for v in verdicts.values()))
        self.assertEqual(verdicts["paper:0"].delta_score, 8)
        self.assertEqual(verdicts["paper:1"].delta_score, 3)
        self.assertEqual(len(agent.calls), 1)
        # The prompt carries the populated question, not an empty shell.
        self.assertIn("q1", agent.calls[0][0])

    def test_no_candidates_returns_empty_without_calling_the_agent(self) -> None:
        agent = _Recorder([{"verdicts": []}])
        self.assertEqual(score_candidates([], _questions(), model="m", agent_fn=agent), {})
        self.assertEqual(agent.calls, [])

    def test_agent_error_yields_unavailable_not_zero(self) -> None:
        """The whole point of DeltaStatus.UNAVAILABLE. An outage must not render as
        a batch of well-behaved zeroes."""
        candidates = _candidates(3)
        agent = _Recorder([AgentError("claude -p exited 1")])
        verdicts = score_candidates(
            candidates, _questions(), model="m", agent_fn=agent
        )
        self.assertEqual(len(verdicts), 3)
        for verdict in verdicts.values():
            self.assertEqual(verdict.status, DeltaStatus.UNAVAILABLE)
            self.assertFalse(verdict.passes)
            self.assertNotEqual(verdict.status, DeltaStatus.SCORED)
            self.assertIn("delta agent failed", verdict.note)
            self.assertIn("claude -p exited 1", verdict.note)

    def test_payload_without_verdicts_array_yields_unavailable(self) -> None:
        candidates = _candidates(2)
        for payload in ({"notes": "sorry"}, {"verdicts": "nope"}, {"verdicts": None}):
            with self.subTest(payload=payload):
                verdicts = score_candidates(
                    candidates, _questions(), model="m", agent_fn=_Recorder([payload])
                )
                self.assertEqual(len(verdicts), 2)
                for verdict in verdicts.values():
                    self.assertEqual(verdict.status, DeltaStatus.UNAVAILABLE)
                    self.assertIn("no verdicts array", verdict.note)

    def test_omitted_item_is_rejected_with_a_note(self) -> None:
        """A missing row is the model's fault, not an outage: the batch DID run, so
        the other items keep their scores and this one says why it has none."""
        candidates = _candidates(2)
        agent = _Recorder([{"verdicts": [_row(0)]}])
        verdicts = score_candidates(candidates, _questions(), model="m", agent_fn=agent)
        self.assertEqual(verdicts["paper:0"].status, DeltaStatus.SCORED)
        self.assertEqual(verdicts["paper:1"].status, DeltaStatus.REJECTED)
        self.assertEqual(verdicts["paper:1"].delta_score, 0)
        self.assertIn("no verdict row", verdicts["paper:1"].note)

    def test_malformed_row_does_not_take_down_its_neighbours(self) -> None:
        candidates = _candidates(2)
        agent = _Recorder([{"verdicts": ["garbage", _row(1, delta_score=6)]}])
        verdicts = score_candidates(candidates, _questions(), model="m", agent_fn=agent)
        self.assertEqual(verdicts["paper:1"].delta_score, 6)
        self.assertEqual(verdicts["paper:0"].status, DeltaStatus.REJECTED)

    def test_duplicate_index_keeps_the_first_row(self) -> None:
        """A repeated index means the model scored one item twice and skipped
        another; the second row is about an item the model never actually read."""
        candidates = _candidates(2)
        agent = _Recorder(
            [{"verdicts": [_row(0, delta_score=9), _row(0, delta_score=1)]}]
        )
        verdicts = score_candidates(candidates, _questions(), model="m", agent_fn=agent)
        self.assertEqual(verdicts["paper:0"].delta_score, 9)
        self.assertEqual(verdicts["paper:1"].status, DeltaStatus.REJECTED)

    def test_empty_reader_model_is_unavailable_and_says_so(self) -> None:
        """Five untouched templates is not "nothing moved me this week"."""
        empty = [parse_question_document(EMPTY_QUESTION_DOC, fallback_id="q1")]
        self.assertFalse(empty[0].is_populated)
        agent = _Recorder([{"verdicts": [_row(0)]}])
        verdicts = score_candidates(_candidates(2), empty, model="m", agent_fn=agent)
        self.assertEqual(len(verdicts), 2)
        for verdict in verdicts.values():
            self.assertEqual(verdict.status, DeltaStatus.UNAVAILABLE)
            self.assertNotEqual(verdict.status, DeltaStatus.SCORED)
            self.assertIn("reader model is empty", verdict.note)
        # And it never even reached the transport.
        self.assertEqual(agent.calls, [])

    def test_unpopulated_questions_are_not_shown_to_the_model(self) -> None:
        questions = _questions(1) + [
            parse_question_document(EMPTY_QUESTION_DOC.replace("Q1", "Q2"), fallback_id="q2")
        ]
        agent = _Recorder([{"verdicts": [_row(0)]}])
        score_candidates(_candidates(1), questions, model="m", agent_fn=agent)
        prompt = agent.calls[0][0]
        # Match the rendered heading, not a bare id: the prompt template's own
        # prose mentions q2 as an example of a bad reason.
        self.assertIn("### q1:", prompt)
        self.assertNotIn("### q2:", prompt)

    def test_verdict_naming_an_unpopulated_question_is_rejected(self) -> None:
        questions = _questions(1) + [
            parse_question_document(EMPTY_QUESTION_DOC.replace("Q1", "Q2"), fallback_id="q2")
        ]
        agent = _Recorder([{"verdicts": [_row(0, question_id="q2")]}])
        verdicts = score_candidates(_candidates(1), questions, model="m", agent_fn=agent)
        self.assertEqual(verdicts["paper:0"].status, DeltaStatus.REJECTED)

    def test_batches_of_ten_over_twenty_five_candidates(self) -> None:
        candidates = _candidates(25)
        agent = _Recorder([{"verdicts": [_row(i) for i in range(10)]}] * 3)
        verdicts = score_candidates(
            candidates, _questions(), model="m", batch_size=10, agent_fn=agent
        )
        self.assertEqual(len(agent.calls), 3)
        self.assertEqual(len(verdicts), 25)
        self.assertEqual(set(verdicts), {c.candidate_id for c in candidates})
        # The last batch holds five items, so its five surplus rows fall out of
        # range and the five real items are the ones scored.
        self.assertTrue(all(v.status == DeltaStatus.SCORED for v in verdicts.values()))

    def test_model_and_timeout_reach_the_transport(self) -> None:
        agent = _Recorder([{"verdicts": [_row(0)]}])
        score_candidates(
            _candidates(1), _questions(), model="opus-x", timeout_s=42, agent_fn=agent
        )
        _, kwargs = agent.calls[0]
        self.assertEqual(kwargs["model"], "opus-x")
        self.assertEqual(kwargs["timeout_s"], 42)
        self.assertIn("schema", kwargs)

    def test_one_bad_batch_does_not_poison_the_good_one(self) -> None:
        candidates = _candidates(4)
        agent = _Recorder([AgentError("timeout"), {"verdicts": [_row(0), _row(1)]}])
        verdicts = score_candidates(
            candidates, _questions(), model="m", batch_size=2, agent_fn=agent
        )
        self.assertEqual(verdicts["paper:0"].status, DeltaStatus.UNAVAILABLE)
        self.assertEqual(verdicts["paper:1"].status, DeltaStatus.UNAVAILABLE)
        self.assertEqual(verdicts["paper:2"].status, DeltaStatus.SCORED)
        self.assertEqual(verdicts["paper:3"].status, DeltaStatus.SCORED)


class _FakeLlmcallResult:
    """What ``llmcall.call`` hands back: truthy on success, with .data/.provider."""

    def __init__(self, data=None, text="", provider="cc", error=""):
        self.data = data
        self.text = text
        self.provider = provider
        self.error = error

    def __bool__(self):
        return not self.error


class GatewayRoutingTest(unittest.TestCase):
    """score_candidates now calls through utils.llm_gateway.

    These tests exist because the routing change is invisible to every test
    above: a gateway that ignored its backend argument and always used the agent
    transport would pass all of them. So each one asserts something only true of
    the gateway -- which backend ran, and what the ledger recorded.
    """

    def setUp(self) -> None:
        llm_gateway.LEDGER.reset()

    def tearDown(self) -> None:
        llm_gateway.LEDGER.reset()

    def test_llmcall_fn_pins_the_llmcall_backend(self) -> None:
        seen = {}

        def fake_llmcall(prompt, **kwargs):
            seen.update(kwargs)
            return _FakeLlmcallResult(data={"verdicts": [_row(0)]}, provider="codexg")

        verdicts = score_candidates(
            _candidates(1), _questions(), model="", llmcall_fn=fake_llmcall
        )
        self.assertEqual(verdicts["paper:0"].status, DeltaStatus.SCORED)
        self.assertEqual(seen["mode"], "judge")
        self.assertIn("schema", seen)
        self.assertEqual(llm_gateway.LEDGER.by_backend, {"llmcall": 1})
        self.assertEqual(llm_gateway.LEDGER.by_provider, {"codexg": 1})

    def test_agent_fn_pins_the_agent_backend_even_when_llmcall_exists(self) -> None:
        """The seam that keeps this suite off the network. If injecting agent_fn
        did not pin the backend, a machine with llmcall installed would resolve to
        it and make the real call."""
        agent = _Recorder([{"verdicts": [_row(0)]}])
        score_candidates(_candidates(1), _questions(), model="m", agent_fn=agent)
        self.assertEqual(len(agent.calls), 1)
        self.assertEqual(llm_gateway.LEDGER.by_backend, {"agent": 1})

    def test_llmcall_returning_json_text_is_parsed(self) -> None:
        """llmcall may leave the payload as a JSON string in .text. That is a
        successful call, not an unparseable one."""
        result = _FakeLlmcallResult(data=None, text=json.dumps({"verdicts": [_row(0)]}))
        verdicts = score_candidates(
            _candidates(1), _questions(), model="", llmcall_fn=lambda p, **k: result
        )
        self.assertEqual(verdicts["paper:0"].status, DeltaStatus.SCORED)

    def test_whole_chain_failing_is_unavailable_and_lands_in_the_ledger(self) -> None:
        """The outage this module exists for, on the new backend. The verdicts say
        unavailable AND the ledger can prove nothing succeeded -- the digest reads
        the second one, because unavailable verdicts alone cannot distinguish a
        dead chain from an empty reader model."""
        dead = _FakeLlmcallResult(error="all providers exhausted")
        verdicts = score_candidates(
            _candidates(2), _questions(), model="", llmcall_fn=lambda p, **k: dead
        )
        for verdict in verdicts.values():
            self.assertEqual(verdict.status, DeltaStatus.UNAVAILABLE)
            self.assertNotEqual(verdict.status, DeltaStatus.SCORED)
            self.assertIn("all providers exhausted", verdict.note)
        ledger = llm_gateway.LEDGER.to_dict()
        self.assertEqual(ledger["attempted"], 1)
        self.assertEqual(ledger["succeeded"], 0)
        self.assertEqual(ledger["failed"], 1)
        self.assertTrue(ledger["errors"])

    def test_a_successful_run_leaves_a_ledger_that_proves_it_ran(self) -> None:
        """Negative control for the test above: the same assertions must NOT hold
        on a healthy run, or "attempted > 0, succeeded == 0" would be meaningless."""
        score_candidates(
            _candidates(1),
            _questions(),
            model="",
            llmcall_fn=lambda p, **k: _FakeLlmcallResult(data={"verdicts": [_row(0)]}),
        )
        ledger = llm_gateway.LEDGER.to_dict()
        self.assertEqual(ledger["attempted"], 1)
        self.assertEqual(ledger["succeeded"], 1)
        self.assertEqual(ledger["failed"], 0)

    def test_call_fn_replaces_the_gateway_entrypoint(self) -> None:
        calls = []

        def fake_call(prompt, **kwargs):
            calls.append(kwargs)
            return llm_gateway.GatewayResult(
                text="", data={"verdicts": [_row(0)]}, backend="agent", provider="claude"
            )

        verdicts = score_candidates(
            _candidates(1), _questions(), model="m", timeout_s=42, call_fn=fake_call
        )
        self.assertEqual(verdicts["paper:0"].status, DeltaStatus.SCORED)
        self.assertEqual(calls[0]["model"], "m")
        self.assertEqual(calls[0]["timeout_s"], 42)


# ---------------------------------------------------------------------------
# Candidate extraction, against real archive shapes
# ---------------------------------------------------------------------------


def _load(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class PaperCandidateTest(unittest.TestCase):
    """Excerpt of a real ``out/json/<month>/<date>-output.json``."""

    def setUp(self) -> None:
        self.mapping = _load("paper-2026-06-04-output.json")

    def test_real_paper_day_produces_usable_candidates(self) -> None:
        candidates = candidates_from_paper_mapping(self.mapping, date="2026-06-04")
        self.assertEqual(len(candidates), len(self.mapping))
        for candidate in candidates:
            with self.subTest(candidate=candidate.candidate_id):
                arxiv_id = candidate.candidate_id.split("paper:", 1)[1]
                self.assertIn(arxiv_id, self.mapping)
                self.assertEqual(candidate.kind, "paper")
                self.assertTrue(candidate.title.strip())
                self.assertTrue(candidate.text.strip())
                self.assertEqual(candidate.url, f"https://arxiv.org/abs/{arxiv_id}")
                self.assertEqual(candidate.date, "2026-06-04")
                # NOVELTY is the paper-side tiebreak; RELEVANCE and SCORE are not.
                self.assertEqual(
                    candidate.tiebreak, float(self.mapping[arxiv_id]["NOVELTY"])
                )
                self.assertIs(candidate.payload, self.mapping[arxiv_id])

    def test_text_carries_abstract_and_filter_comment(self) -> None:
        arxiv_id, entry = next(iter(self.mapping.items()))
        candidate = next(
            c
            for c in candidates_from_paper_mapping(self.mapping)
            if c.candidate_id == f"paper:{arxiv_id}"
        )
        self.assertIn(entry["abstract"][:60], candidate.text)
        self.assertIn(entry["COMMENT"][:40], candidate.text)

    def test_empty_day_produces_no_candidates(self) -> None:
        """An archived empty day is the literal two-byte ``{}``; that is a real day
        with no papers, and it must not raise."""
        self.assertEqual(candidates_from_paper_mapping(json.loads("{}")), [])
        self.assertEqual(candidates_from_paper_mapping({}, date="2026-06-06"), [])

    def test_non_mapping_entry_is_skipped_not_fatal(self) -> None:
        polluted = dict(self.mapping)
        polluted["junk"] = "not an entry"
        self.assertEqual(len(candidates_from_paper_mapping(polluted)), len(self.mapping))

    def test_missing_novelty_falls_back_to_zero(self) -> None:
        entry = {"title": "t", "abstract": "a"}
        candidate = candidates_from_paper_mapping({"2606.1": entry})[0]
        self.assertEqual(candidate.tiebreak, 0.0)


class HotspotCandidateTest(unittest.TestCase):
    """Excerpt of a real ``out/hot/reports/<date>.json`` (flat, no month dir)."""

    def setUp(self) -> None:
        self.report = _load("hotspot-2026-09-08.json")

    def test_real_report_produces_usable_candidates(self) -> None:
        candidates = candidates_from_hotspot_report(self.report)
        expected = len(self.report["featured_topics"]) + len(self.report["watchlist"])
        self.assertEqual(len(candidates), expected)
        by_id = {t["TOPIC_ID"]: t for t in self.report["featured_topics"] + self.report["watchlist"]}
        for candidate in candidates:
            with self.subTest(candidate=candidate.candidate_id):
                topic_id = candidate.candidate_id.rsplit(":", 1)[1]
                topic = by_id[topic_id]
                self.assertEqual(candidate.kind, "hotspot")
                self.assertEqual(candidate.candidate_id, f"hotspot:2026-09-08:{topic_id}")
                self.assertTrue(candidate.title.strip())
                self.assertTrue(candidate.text.strip())
                self.assertTrue(candidate.url.startswith("http"))
                self.assertEqual(candidate.date, "2026-09-08")
                # FINAL_SCORE is the hotspot-side tiebreak. Topics have no NOVELTY.
                self.assertEqual(candidate.tiebreak, float(topic["FINAL_SCORE"]))
                self.assertNotIn("NOVELTY", topic)

    def test_text_carries_summary_and_why_it_matters(self) -> None:
        topic = self.report["featured_topics"][0]
        candidate = candidates_from_hotspot_report(self.report)[0]
        self.assertIn(topic["WHY_IT_MATTERS"][:40], candidate.text)

    def test_older_reports_without_manifest_or_resurgence_are_fine(self) -> None:
        """Archived reports predate those keys; extraction must not read them."""
        self.assertNotIn("manifest", self.report)
        self.assertNotIn("resurgence", self.report)
        self.assertTrue(candidates_from_hotspot_report(self.report))

    def test_a_topic_in_both_buckets_is_emitted_once(self) -> None:
        doubled = dict(self.report)
        doubled["watchlist"] = list(self.report["watchlist"]) + [
            self.report["featured_topics"][0]
        ]
        self.assertEqual(
            len(candidates_from_hotspot_report(doubled)),
            len(candidates_from_hotspot_report(self.report)),
        )

    def test_report_with_no_topics_is_empty_not_fatal(self) -> None:
        self.assertEqual(candidates_from_hotspot_report({"date": "2026-09-08"}), [])
        self.assertEqual(
            candidates_from_hotspot_report(
                {"date": "d", "featured_topics": None, "watchlist": []}
            ),
            [],
        )


class ArchiveWorktreeTest(unittest.TestCase):
    """Same assertions against the FULL archive files, when that checkout exists.

    Skipped on a bare clone. This is the check that the trimmed fixtures above
    still describe the real archive rather than a shape that drifted away from it.
    """

    def _archive_json(self, *parts: str):
        """Load a file from the full archive, or skip when it was not requested.

        Skips only when the env var is unset. When it IS set the path must exist:
        _ARCHIVE_ENV validation at import time already proved the root is real, so
        a missing file here means the archive drifted, which must fail, not skip.
        """
        if ARCHIVE_ROOT is None:
            self.skipTest(f"{_ARCHIVE_ENV} not set; skipping archive drift checks")
        path = ARCHIVE_ROOT.joinpath("out", *parts)
        self.assertTrue(path.is_file(), f"archive drifted: missing {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_full_paper_day(self) -> None:
        mapping = self._archive_json("json", "2026-06", "2026-06-04-output.json")
        candidates = candidates_from_paper_mapping(mapping, date="2026-06-04")
        self.assertEqual(len(candidates), len(mapping))
        self.assertTrue(all(c.title.strip() and c.text.strip() for c in candidates))
        self.assertTrue(all(c.url.startswith("https://arxiv.org/abs/") for c in candidates))

    def test_full_hotspot_report(self) -> None:
        report = self._archive_json("hot", "reports", "2026-09-08.json")
        candidates = candidates_from_hotspot_report(report)
        self.assertTrue(candidates)
        self.assertTrue(all(c.title.strip() and c.text.strip() for c in candidates))
        self.assertTrue(all(c.tiebreak > 0 for c in candidates))


# ---------------------------------------------------------------------------
# Ranking and annotation
# ---------------------------------------------------------------------------


class RankTest(unittest.TestCase):
    def setUp(self) -> None:
        self.high = DeltaCandidate("a", "paper", "A", "", tiebreak=1.0)
        self.mid_hi_tie = DeltaCandidate("b", "paper", "B", "", tiebreak=9.0)
        self.mid_lo_tie = DeltaCandidate("c", "paper", "C", "", tiebreak=2.0)
        self.unscored = DeltaCandidate("d", "paper", "D", "", tiebreak=99.0)
        self.unavailable = DeltaCandidate("e", "paper", "E", "", tiebreak=98.0)
        self.rejected = DeltaCandidate("f", "paper", "F", "", tiebreak=97.0)
        self.verdicts = {
            "a": DeltaVerdict("a", DeltaStatus.SCORED, delta_score=9),
            "b": DeltaVerdict("b", DeltaStatus.SCORED, delta_score=5),
            "c": DeltaVerdict("c", DeltaStatus.SCORED, delta_score=5),
            "e": DeltaVerdict("e", DeltaStatus.UNAVAILABLE, note="agent down"),
            "f": DeltaVerdict("f", DeltaStatus.REJECTED, note="bad row"),
        }

    def test_orders_by_score_then_tiebreak_and_sinks_the_unscored(self) -> None:
        ordered = rank(
            [self.unscored, self.mid_lo_tie, self.unavailable, self.high,
             self.rejected, self.mid_hi_tie],
            self.verdicts,
        )
        self.assertEqual([c.candidate_id for c in ordered[:3]], ["a", "b", "c"])
        # Everything without a passing verdict lands behind everything with one,
        # regardless of how high its pipeline score was.
        self.assertEqual(set(c.candidate_id for c in ordered[3:]), {"d", "e", "f"})
        self.assertEqual([c.candidate_id for c in ordered[3:]], ["d", "e", "f"])

    def test_a_high_pipeline_score_never_outranks_a_delta_score(self) -> None:
        ordered = rank([self.unavailable, self.mid_lo_tie], self.verdicts)
        self.assertEqual(ordered[0].candidate_id, "c")

    def test_empty_inputs(self) -> None:
        self.assertEqual(rank([], {}), [])
        self.assertEqual([c.candidate_id for c in rank([self.high], {})], ["a"])


class AnnotateAndDumpTest(unittest.TestCase):
    def test_annotate_does_not_mutate_the_source_payload(self) -> None:
        payload = {"title": "T"}
        candidate = DeltaCandidate("a", "paper", "T", "", payload=payload)
        verdict = DeltaVerdict("a", DeltaStatus.SCORED, delta_score=7, question_id="q1")
        annotated = annotate(candidate, verdict)
        self.assertEqual(annotated.payload["DELTA"]["delta_score"], 7)
        self.assertNotIn("DELTA", payload, "annotate mutated the archived entry")
        self.assertIsNot(annotated.payload, payload)

    def test_annotate_handles_a_candidate_without_a_payload(self) -> None:
        annotated = annotate(
            DeltaCandidate("a", "paper", "T", ""),
            DeltaVerdict("a", DeltaStatus.UNAVAILABLE, note="agent down"),
        )
        self.assertEqual(annotated.payload["DELTA"]["status"], DeltaStatus.UNAVAILABLE)

    def test_dumps_verdicts_keeps_the_status_and_note(self) -> None:
        """Serialisation must preserve WHY a verdict is empty, or the distinction
        this module exists to make dies at the file boundary."""
        blob = dumps_verdicts(
            {
                "a": DeltaVerdict("a", DeltaStatus.UNAVAILABLE, note="agent down"),
                "b": DeltaVerdict("b", DeltaStatus.SCORED, delta_score=6),
            }
        )
        loaded = json.loads(blob)
        self.assertEqual(loaded["a"]["status"], DeltaStatus.UNAVAILABLE)
        self.assertEqual(loaded["a"]["note"], "agent down")
        self.assertEqual(loaded["a"]["delta_score"], 0)
        self.assertEqual(loaded["b"]["delta_score"], 6)


if __name__ == "__main__":
    unittest.main()
