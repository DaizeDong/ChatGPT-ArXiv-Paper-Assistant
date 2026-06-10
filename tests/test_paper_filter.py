from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from arxiv_assistant.filters.paper_filter import (
    FilterVerdict,
    PaperFilter,
    RuleFilter,
    ApiScoreFilter,
    AgentFilter,
    cascade_filter,
)
from arxiv_assistant.utils.utils import Paper


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "agent" / "paper_filter"


def _paper(arxiv_id: str, title: str = "T", abstract: str = "A", authors=None) -> Paper:
    return Paper(authors=list(authors or ["Ada Lovelace"]), title=title, abstract=abstract, arxiv_id=arxiv_id)


class FilterVerdictTests(unittest.TestCase):
    def test_verdict_fields_and_defaults(self) -> None:
        v = FilterVerdict(keep=True, relevance=9.0, novelty=8.0, rationale="strong fit", evidence=["http://a"])
        self.assertTrue(v.keep)
        self.assertEqual(v.relevance, 9.0)
        self.assertEqual(v.novelty, 8.0)
        self.assertEqual(v.rationale, "strong fit")
        self.assertEqual(v.evidence, ["http://a"])

    def test_evidence_defaults_to_empty_list(self) -> None:
        v = FilterVerdict(keep=False, relevance=2.0, novelty=3.0, rationale="off-topic")
        self.assertEqual(v.evidence, [])
        # default must not be a shared mutable
        v.evidence.append("x")
        v2 = FilterVerdict(keep=False, relevance=1.0, novelty=1.0, rationale="r")
        self.assertEqual(v2.evidence, [])

    def test_protocol_is_runtime_checkable(self) -> None:
        class Dummy:
            def judge(self, paper, criteria):  # noqa: ARG002
                return FilterVerdict(keep=True, relevance=10.0, novelty=10.0, rationale="ok")

        self.assertIsInstance(Dummy(), PaperFilter)

        class NotAFilter:
            pass

        self.assertNotIsInstance(NotAFilter(), PaperFilter)


# ---------------------------------------------------------------------------
# Task 2: RuleFilter
# ---------------------------------------------------------------------------

class _Cfg(dict):
    """Minimal config shim: c['SECTION']['key'] like configparser, no I/O."""


def _config(**overrides) -> dict:
    cfg = {
        "FILTERING": {"h_cutoff": "10", "relevance_cutoff": "8", "novelty_cutoff": "8"},
        "PAPER_FILTER": {"mode": "cascade", "agent_borderline_low": "6.0", "agent_borderline_high": "8.0"},
        "SELECTION": {"run_openai": "true"},
    }
    for section, kv in overrides.items():
        cfg.setdefault(section, {}).update(kv)
    return cfg


class RuleFilterTests(unittest.TestCase):
    def test_keeps_paper_with_high_hindex_author(self) -> None:
        all_authors = {"Ada Lovelace": [{"authorId": "1", "hIndex": 42}]}
        rf = RuleFilter(all_authors=all_authors, config=_config())
        v = rf.judge(_paper("2501.1", authors=["Ada Lovelace"]), criteria="")
        self.assertTrue(v.keep)

    def test_drops_paper_below_hindex_cutoff(self) -> None:
        all_authors = {"Bob Nobody": [{"authorId": "2", "hIndex": 3}]}
        rf = RuleFilter(all_authors=all_authors, config=_config())
        v = rf.judge(_paper("2501.2", authors=["Bob Nobody"]), criteria="")
        self.assertFalse(v.keep)
        self.assertIn("H-index", v.rationale)

    def test_unknown_author_treated_as_hindex_zero(self) -> None:
        # run_author_match=false world: all_authors empty -> max_hindex 0
        rf = RuleFilter(all_authors={}, config=_config())
        # h_cutoff 10 > 0 -> dropped, matching filter_papers_by_hindex
        self.assertFalse(rf.judge(_paper("2501.3"), criteria="").keep)
        # h_cutoff 0 -> 0 < 0 is False -> kept (matches gate)
        rf0 = RuleFilter(all_authors={}, config=_config(FILTERING={"h_cutoff": "0"}))
        self.assertTrue(rf0.judge(_paper("2501.4"), criteria="").keep)

    def test_rulefilter_equivalent_to_filter_papers_by_hindex(self) -> None:
        # Equivalence proof: per-paper RuleFilter.keep == "survived filter_papers_by_hindex"
        from arxiv_assistant.filters.filter_author import filter_papers_by_hindex

        all_authors = {
            "Ada Lovelace": [{"authorId": "1", "hIndex": 42}],
            "Bob Nobody": [{"authorId": "2", "hIndex": 3}],
        }
        papers = [
            _paper("2501.a", authors=["Ada Lovelace"]),
            _paper("2501.b", authors=["Bob Nobody"]),
            _paper("2501.c", authors=["Carol Unknown"]),
        ]
        cfg = _config()
        survivors, _filtered = filter_papers_by_hindex(all_authors, list(papers), cfg)
        survivor_ids = {p.arxiv_id for p in survivors}
        rf = RuleFilter(all_authors=all_authors, config=cfg)
        for p in papers:
            self.assertEqual(rf.judge(p, "").keep, p.arxiv_id in survivor_ids, p.arxiv_id)


# ---------------------------------------------------------------------------
# Task 3: ApiScoreFilter
# ---------------------------------------------------------------------------

def _fake_gpt_factory(scores: dict):
    """Return a stand-in for filter_by_gpt using the real selected/filtered split rule.

    scores: arxiv_id -> (relevance, novelty, comment)
    Reproduces filter_gpt.py:346-354 splitting and the 6-tuple return shape.
    """

    def _fake_filter_by_gpt(paper_list, system_prompt, topic_prompt, score_prompt,
                            postfix_title, postfix_abstract, config):
        selected, filtered = {}, {}
        rc = int(config["FILTERING"]["relevance_cutoff"])
        nc = int(config["FILTERING"]["novelty_cutoff"])
        for p in paper_list:
            rel, nov, comment = scores[p.arxiv_id]
            entry = {
                "ARXIVID": p.arxiv_id, "COMMENT": comment,
                "RELEVANCE": rel, "NOVELTY": nov, "SCORE": rel + nov,
                "arxiv_id": p.arxiv_id, "title": p.title, "abstract": p.abstract, "authors": p.authors,
            }
            if rel < rc or nov < nc:
                filtered[p.arxiv_id] = entry
            else:
                selected[p.arxiv_id] = entry
        return selected, filtered, 0.0, 0.0, 0, 0

    return _fake_filter_by_gpt


class ApiScoreFilterTests(unittest.TestCase):
    def _prompts(self):
        return ("sys", "topic", "score", "ptitle", "pabstract")

    def test_keep_split_matches_filter_by_gpt_selected_set(self) -> None:
        scores = {
            "2501.keep": (9, 9, "strong"),       # >= cutoffs -> selected
            "2501.lowrel": (5, 9, "weak rel"),   # rel<8 -> filtered
            "2501.lownov": (9, 4, "weak nov"),   # nov<8 -> filtered
        }
        cfg = _config()
        papers = [_paper(i) for i in scores]
        gpt_fn = _fake_gpt_factory(scores)
        f = ApiScoreFilter(prompts=self._prompts(), config=cfg, gpt_fn=gpt_fn)
        verdicts = f.judge_batch(papers, criteria="topic")
        by_id = {v_id: v for v_id, v in verdicts.items()}
        self.assertTrue(by_id["2501.keep"].keep)
        self.assertFalse(by_id["2501.lowrel"].keep)
        self.assertFalse(by_id["2501.lownov"].keep)

    def test_verdict_carries_same_scale_scores_and_comment(self) -> None:
        scores = {"2501.x": (7, 8, "matched topic A")}
        f = ApiScoreFilter(prompts=self._prompts(), config=_config(), gpt_fn=_fake_gpt_factory(scores))
        v = f.judge_batch([_paper("2501.x")], "topic")["2501.x"]
        self.assertEqual(v.relevance, 7.0)
        self.assertEqual(v.novelty, 8.0)
        self.assertEqual(v.rationale, "matched topic A")
        self.assertEqual(v.evidence, [])  # API modality has no fetched evidence URLs

    def test_zero_behavior_change_against_real_split_rule(self) -> None:
        # The selected/filtered partition produced by ApiScoreFilter MUST equal the one the
        # historical filter_by_gpt would produce on the same scores+cutoffs. We assert it over
        # a truth-table of relevance/novelty around the cutoff (8/8).
        cfg = _config()
        rc, nc = 8, 8
        scores, papers = {}, []
        for rel in range(6, 11):
            for nov in range(6, 11):
                aid = f"2501.{rel}_{nov}"
                scores[aid] = (rel, nov, f"r{rel}n{nov}")
                papers.append(_paper(aid))
        gpt_fn = _fake_gpt_factory(scores)
        # ground truth from the fake (which encodes filter_gpt.py:346-354 literally)
        gt_selected, _gt_filtered, *_ = gpt_fn(papers, *self._prompts(), cfg)
        gt_keep = set(gt_selected.keys())
        f = ApiScoreFilter(prompts=self._prompts(), config=cfg, gpt_fn=gpt_fn)
        verdicts = f.judge_batch(papers, "topic")
        got_keep = {aid for aid, v in verdicts.items() if v.keep}
        self.assertEqual(got_keep, gt_keep)

    def test_judge_single_paper_runs_one_paper_batch(self) -> None:
        scores = {"2501.solo": (9, 9, "ok")}
        f = ApiScoreFilter(prompts=self._prompts(), config=_config(), gpt_fn=_fake_gpt_factory(scores))
        v = f.judge(_paper("2501.solo"), "topic")
        self.assertTrue(v.keep)
        self.assertEqual(v.relevance, 9.0)


# ---------------------------------------------------------------------------
# Task 4 (fixtures) + Task 5: AgentFilter
# ---------------------------------------------------------------------------

def _replay(fixture_name: str):
    data = json.loads((FIXTURE_DIR / fixture_name).read_text(encoding="utf-8"))

    def _agent_fn(paper, criteria, **kwargs):  # noqa: ARG001
        return json.dumps(data["response"])

    return data, _agent_fn


class AgentFilterTests(unittest.TestCase):
    def test_keep_strong_passes_verifier(self) -> None:
        data, agent_fn = _replay("keep_strong.json")
        af = AgentFilter(config=_config(), agent_fn=agent_fn)
        v = af.judge(_paper(data["arxiv_id"]), data["criteria"])
        self.assertTrue(v.keep)
        self.assertEqual(v.relevance, 9.0)
        self.assertEqual(v.novelty, 8.0)
        # self-referential arxiv evidence survives the verifier
        self.assertIn("https://arxiv.org/abs/2501.12345", v.evidence)

    def test_confident_drop(self) -> None:
        data, agent_fn = _replay("drop_offtopic.json")
        af = AgentFilter(config=_config(), agent_fn=agent_fn)
        v = af.judge(_paper(data["arxiv_id"]), data["criteria"])
        self.assertFalse(v.keep)

    def test_verifier_strips_hallucinated_evidence(self) -> None:
        data, agent_fn = _replay("hallucinated_evidence.json")
        af = AgentFilter(config=_config(), agent_fn=agent_fn)
        v = af.judge(_paper(data["arxiv_id"]), data["criteria"])
        # Neither the wrong-arxiv-id URL nor the invented domain may survive.
        self.assertNotIn("https://arxiv.org/abs/2407.00001", v.evidence)
        self.assertNotIn("https://totally-made-up-citation-portal.example/paper/xyz", v.evidence)
        # All evidence hallucinated -> keep=True with external claim collapses to conservative reject.
        self.assertFalse(v.keep)
        self.assertIn("verifier", v.rationale.lower())

    def test_malformed_schema_falls_back_conservatively(self) -> None:
        data, agent_fn = _replay("malformed_schema.json")
        af = AgentFilter(config=_config(), agent_fn=agent_fn)
        v = af.judge(_paper(data["arxiv_id"]), data["criteria"])
        self.assertFalse(v.keep)
        self.assertEqual(v.evidence, [])
        self.assertIn("verifier", v.rationale.lower())

    def test_reuse_signal_urls_are_allowlisted_evidence(self) -> None:
        # An evidence URL not referencing the paper id is accepted IF it was a supplied reuse signal.
        data = json.loads((FIXTURE_DIR / "keep_strong.json").read_text(encoding="utf-8"))
        resp = dict(data["response"])
        resp["evidence"] = resp["evidence"] + ["https://huggingface.co/papers/2501.12345#votes"]

        def agent_fn(paper, criteria, **kwargs):  # noqa: ARG001
            return json.dumps(resp)

        af = AgentFilter(
            config=_config(),
            agent_fn=agent_fn,
            reuse_signal_fn=lambda paper: ["https://huggingface.co/papers/2501.12345#votes"],
        )
        v = af.judge(_paper(data["arxiv_id"]), data["criteria"])
        self.assertIn("https://huggingface.co/papers/2501.12345#votes", v.evidence)

    def test_reuse_signal_fn_absent_is_noop(self) -> None:
        # When the Store/reuse layer is not wired, reuse_signal_fn defaults to a no-op stub.
        data, agent_fn = _replay("keep_strong.json")
        af = AgentFilter(config=_config(), agent_fn=agent_fn)  # no reuse_signal_fn
        v = af.judge(_paper(data["arxiv_id"]), data["criteria"])
        self.assertTrue(v.keep)  # still works without reuse signals


# ---------------------------------------------------------------------------
# Task 6: cascade_filter router
# ---------------------------------------------------------------------------

class _StubAgent:
    """Records which papers were escalated; returns a deterministic high-keep verdict."""

    def __init__(self):
        self.seen = []

    def judge(self, paper, criteria):
        self.seen.append(paper.arxiv_id)
        return FilterVerdict(keep=True, relevance=8.5, novelty=8.5,
                             rationale="agent deep-judged keep", evidence=["https://arxiv.org/abs/" + paper.arxiv_id])


class CascadeRoutingTests(unittest.TestCase):
    def _api(self, scores):
        return ApiScoreFilter(prompts=("s", "t", "sc", "pt", "pa"),
                              config=_config(), gpt_fn=_fake_gpt_factory(scores))

    def test_api_only_never_calls_agent(self) -> None:
        scores = {"2501.hi": (9, 9, "keep"), "2501.lo": (4, 4, "drop")}
        papers = [_paper("2501.hi"), _paper("2501.lo")]
        agent = _StubAgent()
        verdicts = cascade_filter(
            papers, "topic", _config(PAPER_FILTER={"mode": "api_only"}),
            rule_filter=None, api_filter=self._api(scores), agent_filter=agent,
        )
        self.assertEqual([v.keep for v in verdicts], [True, False])
        self.assertEqual(agent.seen, [])

    def test_cascade_escalates_only_borderline_band(self) -> None:
        # band [6.0, 8.0): rel=7 is borderline; rel=9 confident keep; rel=4 confident drop.
        scores = {
            "2501.conf_keep": (9, 9, "keep"),
            "2501.border": (7, 9, "uncertain"),
            "2501.conf_drop": (4, 9, "drop"),
        }
        papers = [_paper(i) for i in scores]
        agent = _StubAgent()
        verdicts = cascade_filter(
            papers, "topic", _config(PAPER_FILTER={"mode": "cascade", "agent_borderline_low": "6.0", "agent_borderline_high": "8.0"}),
            rule_filter=None, api_filter=self._api(scores), agent_filter=agent,
        )
        by_id = {v.rationale: v for v in verdicts}
        # only the borderline paper was escalated
        self.assertEqual(agent.seen, ["2501.border"])
        # confident verdicts retain their API decision
        keeps = {p.arxiv_id: v.keep for p, v in zip(papers, verdicts)}
        self.assertTrue(keeps["2501.conf_keep"])
        self.assertFalse(keeps["2501.conf_drop"])
        # borderline now carries the agent's verdict
        self.assertTrue(keeps["2501.border"])

    def test_agent_only_routes_every_survivor(self) -> None:
        scores = {"2501.a": (5, 5, "x"), "2501.b": (9, 9, "y")}
        papers = [_paper("2501.a"), _paper("2501.b")]
        agent = _StubAgent()
        verdicts = cascade_filter(
            papers, "topic", _config(PAPER_FILTER={"mode": "agent_only"}),
            rule_filter=None, api_filter=self._api(scores), agent_filter=agent,
        )
        self.assertEqual(sorted(agent.seen), ["2501.a", "2501.b"])
        self.assertTrue(all(v.keep for v in verdicts))

    def test_rule_filter_drops_before_scoring(self) -> None:
        scores = {"2501.keep": (9, 9, "k")}  # only the survivor is scored
        papers = [_paper("2501.keep", authors=["Ada"]), _paper("2501.lowh", authors=["Bob"])]
        rule = RuleFilter(all_authors={"Ada": [{"authorId": "1", "hIndex": 50}],
                                       "Bob": [{"authorId": "2", "hIndex": 1}]}, config=_config())
        agent = _StubAgent()
        verdicts = cascade_filter(
            papers, "topic", _config(PAPER_FILTER={"mode": "cascade"}),
            rule_filter=rule, api_filter=self._api(scores), agent_filter=agent,
        )
        keeps = {p.arxiv_id: v.keep for p, v in zip(papers, verdicts)}
        self.assertTrue(keeps["2501.keep"])
        self.assertFalse(keeps["2501.lowh"])  # dropped by rule, never scored
        self.assertEqual(agent.seen, [])      # confident API keep (9 >= 8) -> no escalation


# ---------------------------------------------------------------------------
# Task 7: main.py wiring (cascade-mode api_only projection)
# ---------------------------------------------------------------------------

class MainWiringProjectionTests(unittest.TestCase):
    def test_api_only_projection_matches_historical_partition(self) -> None:
        # Simulate the main.py agent-mode projection but in api_only-equivalent terms:
        # cascade_filter with mode=api_only must reproduce the selected/filtered split.
        scores = {"2501.k": (9, 9, "k"), "2501.f": (5, 9, "f")}
        papers = [_paper("2501.k"), _paper("2501.f")]
        api = ApiScoreFilter(prompts=("s", "t", "sc", "pt", "pa"), config=_config(), gpt_fn=_fake_gpt_factory(scores))
        verdicts = cascade_filter(papers, "topic", _config(PAPER_FILTER={"mode": "api_only"}),
                                  rule_filter=None, api_filter=api, agent_filter=None)
        keeps = {p.arxiv_id: v.keep for p, v in zip(papers, verdicts)}
        self.assertEqual(keeps, {"2501.k": True, "2501.f": False})


# ---------------------------------------------------------------------------
# Task 8: reuse-signal bridge
# ---------------------------------------------------------------------------

class ReuseSignalBridgeTests(unittest.TestCase):
    def test_none_store_is_noop(self) -> None:
        from arxiv_assistant.filters.paper_filter import make_reuse_signal_fn
        fn = make_reuse_signal_fn(store=None)
        self.assertEqual(fn(_paper("2501.x")), [])

    def test_store_without_method_is_noop(self) -> None:
        from arxiv_assistant.filters.paper_filter import make_reuse_signal_fn

        class BareStore:  # plan-01 store that has not implemented the reuse method yet
            pass

        fn = make_reuse_signal_fn(store=BareStore())
        self.assertEqual(fn(_paper("2501.x")), [])

    def test_store_signals_flow_into_allowlist(self) -> None:
        from arxiv_assistant.filters.paper_filter import make_reuse_signal_fn

        class Store:
            def reuse_signal_urls_for_arxiv(self, arxiv_id):
                return ["https://huggingface.co/papers/%s#votes" % arxiv_id]

        fn = make_reuse_signal_fn(store=Store())
        self.assertEqual(fn(_paper("2501.7")), ["https://huggingface.co/papers/2501.7#votes"])


# ---------------------------------------------------------------------------
# Task 9: gray-compare
# ---------------------------------------------------------------------------

class GrayCompareTests(unittest.TestCase):
    def test_cascade_differs_from_api_only_only_on_borderline(self) -> None:
        # Confident keep/drop must be identical between api_only and cascade; only the borderline
        # band may differ (because the agent re-judged it).
        #
        # Use relevance_cutoff=6 so that borderline papers (rel 6-7) PASS the api_only filter
        # (they are above the cutoff), letting the agent override them in cascade mode.
        # The borderline band [6.0, 8.0) overlaps with api_only-kept papers when cutoff=6.
        scores = {
            "2501.ck": (10, 10, "ck"),   # confident keep (well above band)
            "2501.cd": (3, 3, "cd"),     # confident drop (below cutoff=6)
            "2501.b1": (7, 9, "b1"),     # borderline (rel 7 in [6,8)) AND above cutoff=6 -> api_only keeps
            "2501.b2": (6, 9, "b2"),     # borderline (rel 6 in [6,8)) AND at cutoff=6 -> api_only keeps
        }
        papers = [_paper(i) for i in scores]
        # low cutoffs so borderline papers pass the API filter (proving that cascade can diverge)
        low_cutoff_cfg = _config(FILTERING={"h_cutoff": "10", "relevance_cutoff": "6", "novelty_cutoff": "6"})

        api = ApiScoreFilter(prompts=("s", "t", "sc", "pt", "pa"), config=low_cutoff_cfg, gpt_fn=_fake_gpt_factory(scores))
        api_only = cascade_filter(papers, "topic",
                                  _config(PAPER_FILTER={"mode": "api_only"},
                                          FILTERING={"h_cutoff": "10", "relevance_cutoff": "6", "novelty_cutoff": "6"}),
                                  rule_filter=None, api_filter=api, agent_filter=None)

        # Agent flips both borderline papers to drop (a plausible "deeper read says off-topic").
        class FlipAgent:
            def judge(self, paper, criteria):
                return FilterVerdict(keep=False, relevance=5.0, novelty=5.0,
                                     rationale="agent: borderline -> drop after full-text", evidence=[])

        api2 = ApiScoreFilter(prompts=("s", "t", "sc", "pt", "pa"), config=low_cutoff_cfg, gpt_fn=_fake_gpt_factory(scores))
        cascade = cascade_filter(papers, "topic",
                                 _config(PAPER_FILTER={"mode": "cascade", "agent_borderline_low": "6.0", "agent_borderline_high": "8.0"},
                                         FILTERING={"h_cutoff": "10", "relevance_cutoff": "6", "novelty_cutoff": "6"}),
                                 rule_filter=None, api_filter=api2, agent_filter=FlipAgent())

        api_keep = {p.arxiv_id: v.keep for p, v in zip(papers, api_only)}
        casc_keep = {p.arxiv_id: v.keep for p, v in zip(papers, cascade)}
        # confident bands identical
        self.assertEqual(api_keep["2501.ck"], casc_keep["2501.ck"])
        self.assertEqual(api_keep["2501.cd"], casc_keep["2501.cd"])
        # borderline band changed (agent overrode api_only keep -> cascade drop)
        self.assertNotEqual(api_keep["2501.b1"], casc_keep["2501.b1"])
        self.assertNotEqual(api_keep["2501.b2"], casc_keep["2501.b2"])
