# Paper Pipeline — Agent Filter Modality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement each task below in order. Steps use checkbox (`- [ ]`) syntax; check each off only when its test command is green.
>
> **Contract:** This stage implements §2.9 of `docs/superpowers/plans/2026-06-03-agent-native-rewrite-00-overview.md` (`FilterVerdict` / `PaperFilter` Protocol / `RuleFilter` / `ApiScoreFilter` / `AgentFilter` / `cascade_filter`), the `[PAPER_FILTER]` config block of §3, the record/replay test convention of §4, and invariant **INV6** (every random agent is followed by a deterministic verifier; temperature 0; pinned model id). It realizes spec §H.1 (unified `PaperFilter.judge` interface), §H.2 (cascade routing Rule→Api→Agent-on-borderline), and §H.3 (boundary / reuse-first-class-source / shared-stack / migration: wrap current behavior with zero change first, then add `AgentFilter` defaulting to `cascade`, gray compare).
>
> **Stop-the-line:** This stage is "done" only when every test command listed below is green, including the **zero-behavior-change equivalence proof** (`ApiScoreFilter` ≡ `filter_by_gpt` on the same input) and the record/replay verifier-hallucination tests. Depends on plan 01 (`StoryStore`) **optionally** — the reuse-signal lookup in §H.3 is a no-op stub when the Store is absent.

---

## 0. Scope, ground truth, and locked signatures

**Your phase = the paper-pipeline Agent-filter modality. Do NOT rewrite the current pipeline.** You wrap the existing deterministic + API logic behind a strategy interface (zero behavior change), then add an agent strategy and a cascade router, and wire them into `main.py` behind a config switch whose default reproduces today's behavior exactly.

### 0.1 Ground-truth code facts (verified in-repo this round)

| Fact | File:lines | Consequence for this plan |
|---|---|---|
| `Paper` dataclass = `authors, title, abstract, arxiv_id`; hashed by `arxiv_id` | `arxiv_assistant/utils/utils.py:15-25` | `PaperFilter.judge(paper, criteria)` takes a `Paper`; verdicts keyed by `paper.arxiv_id`. |
| `filter_papers_by_hindex(all_authors, paper_list, config)` → `(new_paper_list, filtered_results)`; gate is `max_hindex < float(config["FILTERING"]["h_cutoff"])` | `arxiv_assistant/filters/filter_author.py:28-52` | `RuleFilter` wraps this **per-paper**: a paper is dropped (`keep=False`) iff it would land in `filtered_results`. |
| `filter_by_gpt(paper_list, system_prompt, topic_prompt, score_prompt, postfix_prompt_title, postfix_prompt_abstract, config)` → `(selected_results, total_filtered_results, prompt_cost, completion_cost, prompt_tokens, completion_tokens)`. Selected/filtered split: `relevance < relevance_cutoff or novelty < novelty_cutoff` ⇒ filtered | `arxiv_assistant/filters/filter_gpt.py:346-354, 400-478` | `ApiScoreFilter` calls `filter_by_gpt` **once** over the batch and projects each per-paper result into a `FilterVerdict`. `keep` = "ended up in `selected_results`". `relevance`/`novelty` come straight from the API scores (same scale). |
| Per-paper scored entry carries `RELEVANCE`, `NOVELTY`, `SCORE = relevance+novelty`, `COMMENT` | `filter_gpt.py:335-344` | `FilterVerdict.relevance = RELEVANCE`, `.novelty = NOVELTY`, `.rationale = COMMENT`. **Same scale guaranteed** because it is literally the same integer. |
| `main.py` flow: `get_papers_from_arxiv` → author match → `filter_papers_by_hindex` → `filter_by_gpt` → `build_hotspot_paper_bundle` → render | `main.py:15-190` | Insert the modality switch at the `run_openai` block (`main.py:76-120`), preserving `selected_paper_dict` / `filtered_paper_dict` / `scored_results_for_hotspot` exactly so rendering/archival/bilingual are untouched. |
| Config sections loaded once into `CONFIG` (configparser); prompts loaded via `read_prompt(...)` | `arxiv_assistant/environment.py:30-43` | New `[PAPER_FILTER]` section read with `fallback=` so old config files keep working. |
| Tests use `unittest.TestCase`, `@patch`, `tempfile`; run via `pytest tests/test_*.py -v` | `tests/test_prompt_loader.py`, `tests/test_paper_topics.py` | Match this style; never hit the network. |

### 0.2 Locked signatures (from contract §2.9 — do NOT invent variants)

```python
@dataclass
class FilterVerdict:
    keep: bool
    relevance: float        # same scale as existing GPT score (integer 1..10, as float)
    novelty: float
    rationale: str
    evidence: list[str]

class PaperFilter(Protocol):
    def judge(self, paper, criteria: str) -> FilterVerdict: ...

class RuleFilter:      ...  # wraps filter_author h-index gate
class ApiScoreFilter:  ...  # wraps existing filter_gpt single-call scoring — ZERO behavior change
class AgentFilter:     ...  # Claude Code subagent; temp0 + structured + deterministic verifier
def cascade_filter(papers, criteria, config) -> list[FilterVerdict]: ...  # Rule→Api→(Agent on borderline)
```

### 0.3 Config block (contract §3, add to `configs/config.ini` and `configs/templates/config.template.ini`)

```ini
[PAPER_FILTER]
mode = api_only                         # api_only|agent_only|cascade  (DEFAULT api_only = today's behavior, zero change)
agent_borderline_low = 6.0
agent_borderline_high = 8.0
```

> **Migration safety:** the repo default ships `mode = api_only` so a user who does not touch config gets byte-identical output to today. The contract §3 sets `mode = cascade` as the *eventual* recommended value; we flip the default only after the gray-compare in Task 9 passes. Document both in the template.

### 0.4 Files this plan creates / touches

```
arxiv_assistant/filters/paper_filter.py   NEW  — FilterVerdict + Protocol + RuleFilter/ApiScoreFilter/AgentFilter + cascade_filter
main.py                                    MOD  — modality switch at the run_openai block (api_only default = no-op)
configs/config.ini                         MOD  — add [PAPER_FILTER]
configs/templates/config.template.ini      MOD  — add [PAPER_FILTER] (documented)
tests/test_paper_filter.py                 NEW  — equivalence + record/replay + cascade-routing tests
tests/fixtures/agent/paper_filter/*.json   NEW  — captured AgentFilter responses (record/replay)
```

File-size discipline: `paper_filter.py` stays < ~300 lines, one responsibility (filter strategies + router). The agent transport is a single injectable callable so tests never touch the network.

---

## 1. Task 0 — Add the `[PAPER_FILTER]` config block (no code yet)

- [ ] Add the section to `configs/config.ini` (active runtime config) **after** the `[FILTERING]` block:

```ini
[PAPER_FILTER]
# Paper-filtering modality. api_only reproduces the historical pipeline exactly (default, zero behavior change).
# agent_only routes every surviving paper through the Claude Code agent (highest quality, highest cost).
# cascade runs Rule -> Api cheap scoring, and only escalates the borderline band [low, high) to the agent.
mode = api_only
# Borderline band on the API RELEVANCE score (same 1..10 scale). A paper whose API relevance is
# >= agent_borderline_low and < agent_borderline_high is "uncertain" and (in cascade mode) gets the agent.
agent_borderline_low = 6.0
agent_borderline_high = 8.0
```

- [ ] Mirror it in `configs/templates/config.template.ini` (same keys; the template is the documented onboarding copy). Keep `mode = api_only` in both for now.

- [ ] **Test command** (config parses, defaults resolve):

```bash
python -c "import configparser; c=configparser.ConfigParser(); c.read('configs/config.ini'); print(c['PAPER_FILTER']['mode'], c['PAPER_FILTER'].getfloat('agent_borderline_low'), c['PAPER_FILTER'].getfloat('agent_borderline_high'))"
```

Expected stdout: `api_only 6.0 8.0`

- [ ] **Commit:** `feat(paper-filter): add [PAPER_FILTER] config block (mode/borderline band), default api_only`

---

## 2. Task 1 — `FilterVerdict` + `PaperFilter` Protocol (skeleton, TDD)

Write the test first, watch it fail (module missing), then create the module.

- [ ] Create `tests/test_paper_filter.py` with the dataclass/protocol tests:

```python
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
```

- [ ] Run it, confirm `ImportError` (module not yet created):

```bash
pytest tests/test_paper_filter.py -v
```

- [ ] Create `arxiv_assistant/filters/paper_filter.py` — header + verdict + protocol:

```python
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from arxiv_assistant.utils.utils import Paper


@dataclass
class FilterVerdict:
    """Unified, strategy-agnostic decision for one paper.

    `relevance`/`novelty` are on the SAME 1..10 scale the existing GPT scorer emits
    (stored as float so agent strategies may return fractional confidence), so the
    historical ranking/thresholds keep working unchanged.
    """

    keep: bool
    relevance: float
    novelty: float
    rationale: str
    evidence: List[str] = field(default_factory=list)


@runtime_checkable
class PaperFilter(Protocol):
    def judge(self, paper: Paper, criteria: str) -> FilterVerdict: ...
```

- [ ] Re-run; the three skeleton tests pass (the `RuleFilter/ApiScoreFilter/AgentFilter/cascade_filter` imports still fail — that is expected, they are added next; keep them imported at the top so the import error guides the next task):

```bash
pytest tests/test_paper_filter.py::FilterVerdictTests -v
```

Expected: 3 passed.

- [ ] **Commit:** `feat(paper-filter): FilterVerdict dataclass + runtime-checkable PaperFilter protocol`

---

## 3. Task 2 — `RuleFilter` (wrap the h-index gate, TDD)

`RuleFilter` is the cheap hard pre-filter. It wraps `filter_papers_by_hindex` semantics **per paper** so the protocol is uniform. A paper is **dropped** (`keep=False`) exactly when the existing gate would have moved it into `filtered_results`: `max_hindex < float(config["FILTERING"]["h_cutoff"])`. When author info is unavailable (`run_author_match=false`, the repo default), `all_authors={}` so `max_hindex=0` and the gate trips only if `h_cutoff>0` — identical to today.

- [ ] Add to `tests/test_paper_filter.py`:

```python
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
```

- [ ] Run, watch `RuleFilter` import/attr errors. Then append `RuleFilter` to `paper_filter.py`:

```python
class RuleFilter:
    """Hard pre-filter: drops papers whose best author h-index is below the cutoff.

    Mirrors arxiv_assistant.filters.filter_author.filter_papers_by_hindex EXACTLY,
    one paper at a time, so the cascade can reuse the historical gate as a protocol member.
    """

    def __init__(self, all_authors: dict, config) -> None:
        self._all_authors = all_authors or {}
        self._config = config

    def judge(self, paper: Paper, criteria: str) -> FilterVerdict:  # noqa: ARG002
        cutoff = float(self._config["FILTERING"]["h_cutoff"])
        max_hindex = max(
            [
                alias["hIndex"]
                for author in paper.authors if author in self._all_authors
                for alias in self._all_authors[author]
            ]
            + [0]
        )
        dropped = max_hindex < cutoff
        if dropped:
            return FilterVerdict(
                keep=False,
                relevance=0.0,
                novelty=0.0,
                rationale=f"H-index filtered (max is {max_hindex}<{cutoff})",
                evidence=[],
            )
        # Pass-through: RuleFilter only gates; scoring is the API/agent's job.
        return FilterVerdict(keep=True, relevance=0.0, novelty=0.0, rationale="passed h-index gate", evidence=[])
```

- [ ] **Test command:**

```bash
pytest tests/test_paper_filter.py::RuleFilterTests -v
```

Expected: 4 passed (including the equivalence proof against `filter_papers_by_hindex`).

- [ ] **Commit:** `feat(paper-filter): RuleFilter wraps filter_papers_by_hindex per-paper (equivalence-tested)`

---

## 4. Task 3 — `ApiScoreFilter` (wrap `filter_by_gpt`, ZERO behavior change, TDD)

This is the load-bearing equivalence task. `ApiScoreFilter` runs the existing batch scorer **once** and projects each per-paper result into a `FilterVerdict`. `keep` is determined by membership in `selected_results` (i.e. NOT `relevance < relevance_cutoff or novelty < novelty_cutoff`). The verdict's `relevance`/`novelty`/`rationale` are the literal `RELEVANCE`/`NOVELTY`/`COMMENT` the API returned — **same scale, same values** ⇒ zero behavior change is provable by construction.

Because `filter_by_gpt` is a batch function (not per-paper), `ApiScoreFilter` exposes a batch entry point `judge_batch(papers, criteria)` used by the cascade and by `main.py`, plus a `judge(paper, criteria)` that runs a one-paper batch (used only for the uniform protocol / tests). The expensive call is injected as `gpt_fn` so tests stub it deterministically and never touch OpenAI.

- [ ] Add to `tests/test_paper_filter.py` a fake `filter_by_gpt` and the equivalence tests:

```python
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
```

- [ ] Append `ApiScoreFilter` to `paper_filter.py`:

```python
class ApiScoreFilter:
    """Wraps the existing single-call GPT scorer (filter_by_gpt) behind the PaperFilter protocol.

    ZERO behavior change: it calls filter_by_gpt once over the whole batch and projects each
    per-paper result into a FilterVerdict. `keep` == "paper landed in selected_results", which is
    exactly NOT (RELEVANCE < relevance_cutoff or NOVELTY < novelty_cutoff). RELEVANCE/NOVELTY/COMMENT
    are passed through verbatim, so the score scale and the selected/filtered partition are identical
    to the historical pipeline.

    `gpt_fn` is injected (defaults to the real filter_by_gpt) so tests run offline.
    """

    def __init__(self, prompts: Sequence[str], config, gpt_fn: Optional[Callable] = None) -> None:
        (self._system_prompt, self._topic_prompt, self._score_prompt,
         self._postfix_title, self._postfix_abstract) = prompts
        self._config = config
        if gpt_fn is None:
            from arxiv_assistant.filters.filter_gpt import filter_by_gpt as gpt_fn  # lazy: avoid OpenAI import at module load
        self._gpt_fn = gpt_fn
        # Populated by judge_batch so main.py can recover the rich scored entries for the hotspot bundle.
        self.last_selected: Dict[str, dict] = {}
        self.last_filtered: Dict[str, dict] = {}
        self.last_costs: tuple = (0.0, 0.0, 0, 0)

    def judge_batch(self, papers: Sequence[Paper], criteria: str) -> Dict[str, FilterVerdict]:  # noqa: ARG002
        selected, filtered, p_cost, c_cost, p_tok, c_tok = self._gpt_fn(
            list(papers),
            self._system_prompt,
            self._topic_prompt,
            self._score_prompt,
            self._postfix_title,
            self._postfix_abstract,
            self._config,
        )
        self.last_selected, self.last_filtered = dict(selected), dict(filtered)
        self.last_costs = (p_cost, c_cost, p_tok, c_tok)

        verdicts: Dict[str, FilterVerdict] = {}
        for aid, entry in {**selected, **filtered}.items():
            verdicts[aid] = FilterVerdict(
                keep=aid in selected,
                relevance=float(entry.get("RELEVANCE", 0)),
                novelty=float(entry.get("NOVELTY", 0)),
                rationale=str(entry.get("COMMENT", "")),
                evidence=[],
            )
        return verdicts

    def judge(self, paper: Paper, criteria: str) -> FilterVerdict:
        verdicts = self.judge_batch([paper], criteria)
        return verdicts.get(
            paper.arxiv_id,
            FilterVerdict(keep=False, relevance=0.0, novelty=0.0, rationale="no API verdict", evidence=[]),
        )
```

- [ ] **Test command:**

```bash
pytest tests/test_paper_filter.py::ApiScoreFilterTests -v
```

Expected: 4 passed. The `test_zero_behavior_change_against_real_split_rule` truth-table is the proof of equivalence with the current pipeline.

- [ ] **Commit:** `feat(paper-filter): ApiScoreFilter wraps filter_by_gpt with proven zero behavior change`

---

## 5. Task 4 — record/replay fixtures for `AgentFilter`

The agent is a stateless Claude Code subagent: temperature 0, forced structured (JSON) output, allowed to call tools (WebFetch full-text PDF, Semantic Scholar / OpenAlex citation lookup, arXiv version check). Per INV6 it is **always** followed by a deterministic verifier. We never hit the network in tests; instead we record one captured JSON response per scenario under `tests/fixtures/agent/paper_filter/` and replay it through an injected `agent_fn`.

The agent's raw response contract (what `agent_fn(paper, criteria)` returns — a JSON string the agent emitted):

```json
{
  "keep": <bool>,
  "relevance": <number 1..10>,
  "novelty": <number 1..10>,
  "rationale": "<short reason>",
  "evidence": ["<url that the agent actually fetched/cited>", ...]
}
```

- [ ] Create `tests/fixtures/agent/paper_filter/keep_strong.json` (a clear keep, evidence URLs all reference the paper under judgement so the verifier accepts them):

```json
{
  "arxiv_id": "2501.12345",
  "criteria": "Foundational topics: architecture & training of large models.",
  "response": {
    "keep": true,
    "relevance": 9,
    "novelty": 8,
    "rationale": "Introduces a new attention factorization; full-text confirms a genuine architecture contribution, not an application paper.",
    "evidence": [
      "https://arxiv.org/abs/2501.12345",
      "https://www.semanticscholar.org/arxiv/2501.12345"
    ]
  }
}
```

- [ ] Create `tests/fixtures/agent/paper_filter/drop_offtopic.json` (a confident drop):

```json
{
  "arxiv_id": "2501.99999",
  "criteria": "Foundational topics: architecture & training of large models.",
  "response": {
    "keep": false,
    "relevance": 3,
    "novelty": 4,
    "rationale": "Full-text reading shows the core contribution is a domain medical-imaging application; only tangentially uses a pretrained backbone.",
    "evidence": [
      "https://arxiv.org/abs/2501.99999"
    ]
  }
}
```

- [ ] Create `tests/fixtures/agent/paper_filter/hallucinated_evidence.json` (the agent cites a URL it did not legitimately derive — used to prove the verifier intercepts hallucinations). The fabricated evidence references a *different* arXiv id than the paper under judgement and an obviously invented domain:

```json
{
  "arxiv_id": "2501.55555",
  "criteria": "Foundational topics: architecture & training of large models.",
  "response": {
    "keep": true,
    "relevance": 9,
    "novelty": 9,
    "rationale": "Claims a breakthrough, citing sources that do not correspond to this paper.",
    "evidence": [
      "https://arxiv.org/abs/2407.00001",
      "https://totally-made-up-citation-portal.example/paper/xyz"
    ]
  }
}
```

- [ ] Create `tests/fixtures/agent/paper_filter/malformed_schema.json` (agent returned a structurally invalid object — missing `novelty`, `relevance` out of range — used to prove the verifier rejects and the deterministic fallback fires):

```json
{
  "arxiv_id": "2501.77777",
  "criteria": "Foundational topics: architecture & training of large models.",
  "response": {
    "keep": true,
    "relevance": 42,
    "rationale": "no novelty field, relevance out of 1..10 range"
  }
}
```

- [ ] **Test command** (fixtures are valid JSON and present):

```bash
python -c "import json,glob; [json.load(open(f, encoding='utf-8')) for f in glob.glob('tests/fixtures/agent/paper_filter/*.json')]; print('fixtures ok')"
```

Expected stdout: `fixtures ok`

- [ ] **Commit:** `test(paper-filter): record/replay fixtures for AgentFilter (keep/drop/hallucination/malformed)`

---

## 6. Task 5 — `AgentFilter` + deterministic verifier (INV6, TDD)

`AgentFilter.judge` does: build a typed prompt for the paper+criteria → call the injected `agent_fn` (the Claude Code subagent transport; temperature 0, model id pinned in the verdict provenance) → parse the JSON → run a **deterministic verifier** that (a) validates the schema (types + `1 <= relevance,novelty <= 10` + `keep` bool) and (b) checks every `evidence` URL is *legitimately derived*: an evidence URL is accepted iff it references the paper's own `arxiv_id` (e.g. an `arxiv.org/abs/<id>` or a known scholarly host carrying that id) OR it was supplied to the agent in the reuse-signal allowlist (§H.3). Any evidence URL that fails this is **dropped from the verdict**; if that empties a `keep=True` verdict's evidence *and* the verdict claimed external corroboration, or if the schema is invalid, the verifier rejects the agent output and the deterministic fallback verdict is returned (conservative: `keep=False`, low scores, rationale flags the rejection). This realizes INV6: a random agent is always followed by a deterministic verifier.

- [ ] Add to `tests/test_paper_filter.py`:

```python
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
        # §H.3: when the Store/reuse layer is not wired, reuse_signal_fn defaults to a no-op stub.
        data, agent_fn = _replay("keep_strong.json")
        af = AgentFilter(config=_config(), agent_fn=agent_fn)  # no reuse_signal_fn
        v = af.judge(_paper(data["arxiv_id"]), data["criteria"])
        self.assertTrue(v.keep)  # still works without reuse signals
```

- [ ] Append `AgentFilter` (and helpers) to `paper_filter.py`:

```python
AGENT_MODEL_ID = "claude-code-subagent"   # pinned; recorded in provenance for INV6 auditability
AGENT_TEMPERATURE = 0.0


def _evidence_references_paper(url: str, arxiv_id: str) -> bool:
    """Deterministic check: does this URL legitimately reference the paper under judgement?

    Accepts URLs that carry the paper's own arxiv id (arxiv.org abs/pdf, or scholarly hosts
    that embed the id). The id may appear with or without a version suffix.
    """
    base_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
    needle = base_id.lower()
    u = url.lower()
    return needle in u and ("arxiv.org" in u or "semanticscholar.org" in u or "openalex.org" in u)


def _verify_agent_response(raw: str, paper: Paper, allowlist: Sequence[str]) -> Optional[FilterVerdict]:
    """Deterministic verifier (INV6). Returns a clean FilterVerdict, or None to signal rejection.

    Rules:
      - JSON must parse and contain keep(bool), relevance(number 1..10), novelty(number 1..10).
      - Each evidence URL is kept iff it references the paper id OR is in the reuse-signal allowlist.
      - If schema invalid -> None (reject).
      - If keep is True but every evidence URL was stripped as hallucinated AND the agent supplied
        any evidence at all (i.e. it CLAIMED external corroboration that did not survive) -> None.
    """
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    keep = data.get("keep")
    rel = data.get("relevance")
    nov = data.get("novelty")
    if not isinstance(keep, bool):
        return None
    if not isinstance(rel, (int, float)) or not isinstance(nov, (int, float)):
        return None
    if not (1 <= rel <= 10) or not (1 <= nov <= 10):
        return None

    raw_evidence = data.get("evidence") or []
    if not isinstance(raw_evidence, list):
        raw_evidence = []
    allow = set(allowlist or [])
    clean_evidence = [
        url for url in raw_evidence
        if isinstance(url, str) and (_evidence_references_paper(url, paper.arxiv_id) or url in allow)
    ]
    claimed_external = len(raw_evidence) > 0
    if keep and claimed_external and not clean_evidence:
        return None  # all corroboration was hallucinated -> reject

    return FilterVerdict(
        keep=keep,
        relevance=float(rel),
        novelty=float(nov),
        rationale=str(data.get("rationale", "")),
        evidence=clean_evidence,
    )


class AgentFilter:
    """Claude Code stateless subagent strategy (temp 0, forced JSON, tool-using), followed by a
    deterministic verifier (INV6). The subagent transport is injected as `agent_fn` so tests replay
    captured JSON. `reuse_signal_fn` (§H.3) optionally supplies corroborating reuse-source URLs
    (HF votes / Scholar Inbox / Altmetric); when absent it is a no-op stub.
    """

    def __init__(self, config, agent_fn: Callable, reuse_signal_fn: Optional[Callable] = None) -> None:
        self._config = config
        self._agent_fn = agent_fn
        self._reuse_signal_fn = reuse_signal_fn or (lambda paper: [])

    def judge(self, paper: Paper, criteria: str) -> FilterVerdict:
        allowlist = list(self._reuse_signal_fn(paper) or [])
        raw = self._agent_fn(
            paper,
            criteria,
            reuse_signals=allowlist,
            temperature=AGENT_TEMPERATURE,
            model=AGENT_MODEL_ID,
        )
        verdict = _verify_agent_response(raw, paper, allowlist)
        if verdict is None:
            return FilterVerdict(
                keep=False,
                relevance=0.0,
                novelty=0.0,
                rationale="Rejected by deterministic verifier (schema invalid or hallucinated evidence).",
                evidence=[],
            )
        return verdict
```

- [ ] **Test command:**

```bash
pytest tests/test_paper_filter.py::AgentFilterTests -v
```

Expected: 6 passed (keep, drop, hallucination-stripped, malformed-fallback, reuse-allowlist, reuse-noop).

- [ ] **Commit:** `feat(paper-filter): AgentFilter subagent + deterministic verifier (INV6, evidence-URL truth check)`

---

## 7. Task 6 — `cascade_filter` router (Rule → Api → Agent-on-borderline, TDD)

`cascade_filter(papers, criteria, config)` returns `list[FilterVerdict]` in input order. It dispatches by `config["PAPER_FILTER"]["mode"]`:

- **`api_only`** (default): RuleFilter hard pre-filter (only when the rule filter is enabled, i.e. author info present) then ApiScoreFilter batch; never invokes the agent. Output identical to today's `selected/filtered` partition.
- **`agent_only`**: RuleFilter pre-filter, then AgentFilter on every survivor.
- **`cascade`**: RuleFilter pre-filter → ApiScoreFilter batch → for each survivor whose API relevance ∈ `[agent_borderline_low, agent_borderline_high)` (the uncertain band), escalate to AgentFilter and use its verdict; confident-high and confident-low API verdicts are taken as final.

The strategies are injected (`rule_filter`, `api_filter`, `agent_filter`) so the router is unit-tested without network. Papers dropped by `RuleFilter` get the rule's `keep=False` verdict and are NOT scored further (matching the historical "h-index filtered papers never reach GPT").

- [ ] Add to `tests/test_paper_filter.py`:

```python
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
```

- [ ] Append `cascade_filter` to `paper_filter.py`:

```python
def cascade_filter(
    papers: Sequence[Paper],
    criteria: str,
    config,
    *,
    rule_filter: Optional["RuleFilter"] = None,
    api_filter: Optional["ApiScoreFilter"] = None,
    agent_filter: Optional["AgentFilter"] = None,
) -> List[FilterVerdict]:
    """Confidence-aware router (spec §H.2). Returns verdicts in input order.

    mode=api_only  : Rule pre-filter (if provided) -> Api batch. No agent.
    mode=agent_only: Rule pre-filter -> Agent on every survivor.
    mode=cascade   : Rule pre-filter -> Api batch -> Agent only on the borderline relevance band.
    """
    mode = config["PAPER_FILTER"]["mode"].strip().lower()
    low = float(config["PAPER_FILTER"]["agent_borderline_low"])
    high = float(config["PAPER_FILTER"]["agent_borderline_high"])

    verdicts: Dict[str, FilterVerdict] = {}
    survivors: List[Paper] = []

    # Stage 1: hard rule pre-filter (h-index). Dropped papers never get scored.
    for paper in papers:
        if rule_filter is not None:
            rv = rule_filter.judge(paper, criteria)
            if not rv.keep:
                verdicts[paper.arxiv_id] = rv
                continue
        survivors.append(paper)

    if mode == "agent_only":
        if agent_filter is None:
            raise ValueError("mode=agent_only requires an agent_filter")
        for paper in survivors:
            verdicts[paper.arxiv_id] = agent_filter.judge(paper, criteria)
    else:
        # api_only and cascade both start from the cheap batch score.
        if api_filter is None:
            raise ValueError(f"mode={mode} requires an api_filter")
        api_verdicts = api_filter.judge_batch(survivors, criteria)
        for paper in survivors:
            av = api_verdicts.get(
                paper.arxiv_id,
                FilterVerdict(keep=False, relevance=0.0, novelty=0.0, rationale="no API verdict", evidence=[]),
            )
            if mode == "cascade" and agent_filter is not None and low <= av.relevance < high:
                verdicts[paper.arxiv_id] = agent_filter.judge(paper, criteria)
            else:
                verdicts[paper.arxiv_id] = av

    return [verdicts[paper.arxiv_id] for paper in papers]
```

- [ ] **Test command:**

```bash
pytest tests/test_paper_filter.py::CascadeRoutingTests -v
```

Expected: 4 passed (api_only no agent, cascade borderline-only escalation, agent_only all, rule pre-drop).

- [ ] **Full module test command:**

```bash
pytest tests/test_paper_filter.py -v
```

Expected: all classes green (FilterVerdict 3, RuleFilter 4, ApiScoreFilter 4, AgentFilter 6, Cascade 4 = 21 passed).

- [ ] **Commit:** `feat(paper-filter): cascade_filter router (api_only/agent_only/cascade) with borderline-only agent escalation`

---

## 8. Task 7 — Wire `cascade_filter` into `main.py` behind `[PAPER_FILTER]` mode

Insert the modality at the existing `run_openai` block (`main.py:76-120`). Critically, **`mode=api_only` must reproduce today's behavior byte-for-byte**: in that mode we keep calling `filter_by_gpt` exactly as before and use its existing `selected_results`/`filtered_results`/cost tuple/`scored_results_for_hotspot`. The cascade is engaged only for `agent_only`/`cascade`. This preserves the hotspot bundle, the diversion logic, rendering, archival, and bilingual output untouched.

- [ ] Read the current block once more to anchor the edit (`main.py:76-120`). The edit replaces the body of `if CONFIG["SELECTION"].getboolean("run_openai"):` with a mode dispatch. For `api_only`, call the existing path verbatim. For the agent modes, build the strategies and run `cascade_filter`, then **reconstruct** `selected_results` / `filtered_results` / `scored_results_for_hotspot` from the verdicts so all downstream code (the `build_hotspot_paper_bundle` call, the diversion loop, the dumps) is unchanged.

- [ ] Apply this edit to `main.py` — replace the `filter_by_gpt(...)` invocation and the two lines computing `selected_paper_dict.update`/`filtered_paper_dict.update` (lines 77-87) with the mode dispatch below. The hotspot block (lines 89-120) stays as-is and consumes the same variables:

```python
        paper_filter_mode = CONFIG["PAPER_FILTER"]["mode"].strip().lower() if CONFIG.has_section("PAPER_FILTER") else "api_only"

        if paper_filter_mode == "api_only":
            # Historical path, unchanged: single-call GPT scoring over the batch.
            selected_results, filtered_results, total_prompt_cost, total_completion_cost, total_prompt_tokens, total_completion_tokens = filter_by_gpt(
                paper_list,
                SYSTEM_PROMPT,
                TOPIC_PROMPT,
                SCORE_PROMPT,
                POSTFIX_PROMPT_TITLE,
                POSTFIX_PROMPT_ABSTRACT,
                CONFIG,
            )
        else:
            from arxiv_assistant.filters.paper_filter import ApiScoreFilter, AgentFilter, cascade_filter
            from arxiv_assistant.apis.claude_agent import judge_paper_with_agent  # subagent transport (added in this stage)

            api_filter = ApiScoreFilter(
                prompts=(SYSTEM_PROMPT, TOPIC_PROMPT, SCORE_PROMPT, POSTFIX_PROMPT_TITLE, POSTFIX_PROMPT_ABSTRACT),
                config=CONFIG,
            )
            agent_filter = AgentFilter(config=CONFIG, agent_fn=judge_paper_with_agent)
            verdicts = cascade_filter(
                paper_list, TOPIC_PROMPT, CONFIG,
                rule_filter=None,          # h-index pre-filter already ran above (main.py:62-71)
                api_filter=api_filter,
                agent_filter=agent_filter,
            )
            # Project verdicts back into the historical selected/filtered mappings so all
            # downstream rendering/archival/bilingual code is untouched.
            import dataclasses as _dc
            from arxiv_assistant.paper_topics import ensure_topic_fields as _ensure
            id_to_paper = {p.arxiv_id: p for p in paper_list}
            selected_results, filtered_results = {}, {}
            for paper, v in zip(paper_list, verdicts):
                entry = _ensure({
                    "ARXIVID": paper.arxiv_id,
                    "COMMENT": v.rationale,
                    "RELEVANCE": int(round(v.relevance)),
                    "NOVELTY": int(round(v.novelty)),
                    "SCORE": int(round(v.relevance)) + int(round(v.novelty)),
                    "FILTER_EVIDENCE": v.evidence,
                    **_dc.asdict(id_to_paper[paper.arxiv_id]),
                }, arxiv_id=paper.arxiv_id)
                if v.keep:
                    selected_results[paper.arxiv_id] = entry
                else:
                    filtered_results[paper.arxiv_id] = entry
            # Costs: agent runs on the subscription quota (not per-call billed), so report API costs only.
            total_prompt_cost, total_completion_cost, total_prompt_tokens, total_completion_tokens = api_filter.last_costs[0], api_filter.last_costs[1], api_filter.last_costs[2], api_filter.last_costs[3]

        selected_paper_dict.update(selected_results)
        filtered_paper_dict.update(filtered_results)
```

> **Note for the implementer:** `judge_paper_with_agent` (the real Claude Code subagent transport) is a thin adapter living at `arxiv_assistant/apis/claude_agent.py`. Its production implementation (spawning `claude -p` headless, temperature 0, forced JSON) belongs to the runtime stage (plan 07) which owns the shared Claude Code stack (spec §H.3 "shared stack"). For this stage, ship a minimal stub that raises `NotImplementedError("agent transport wired in stage 6 runtime")` if called, since the repo default `mode=api_only` never reaches it. The unit tests inject their own `agent_fn` and never import this module. Document this clearly in the stub so the default run is unaffected.

- [ ] Create the stub `arxiv_assistant/apis/claude_agent.py`:

```python
"""Claude Code subagent transport for the paper Agent-filter modality (spec §H).

The real implementation (headless `claude -p`, temperature 0, forced structured JSON,
tool access to WebFetch/Semantic Scholar/OpenAlex/arXiv) is owned by the shared-runtime
stage (plan 07). Until then, only the default mode=api_only is supported in production;
the agent modes require this transport. Unit tests inject their own agent_fn and never call this.
"""

from __future__ import annotations


def judge_paper_with_agent(paper, criteria, *, reuse_signals=None, temperature=0.0, model="claude-code-subagent") -> str:
    raise NotImplementedError(
        "Claude Code agent transport is wired in the stage-6 runtime (plan 07). "
        "Use mode=api_only until then, or inject agent_fn in tests."
    )
```

- [ ] **Test command** (default mode keeps `main.py` importable and the api_only branch intact; we assert the module imports and the api_only path does not require the agent transport):

```bash
python -c "import ast; ast.parse(open('main.py', encoding='utf-8').read()); print('main.py parses')"
python -c "import importlib; m=importlib.import_module('arxiv_assistant.apis.claude_agent'); print(hasattr(m,'judge_paper_with_agent'))"
```

Expected: `main.py parses` then `True`.

- [ ] Add a focused integration test to `tests/test_paper_filter.py` proving the `main.py` projection is decision-preserving for `api_only` vs the historical split (no agent transport touched):

```python
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
```

- [ ] **Test command:**

```bash
pytest tests/test_paper_filter.py -v
```

Expected: 22 passed.

- [ ] **Commit:** `feat(paper-filter): wire cascade modality into main.py (api_only default unchanged; agent transport stub)`

---

## 9. Task 8 — §H.3 reuse-signal consumption (no-op stub when Store absent)

The `AgentFilter` already accepts `reuse_signal_fn`; this task documents and tests the bridge to the §D reuse layer (HF votes / Scholar Inbox / Altmetric) so the paper pipeline also enjoys "lower bound = union of the market". When plan 01's `StoryStore` and the reuse adapters are not yet present, `reuse_signal_fn` is omitted and the agent runs with no corroborating URLs (no-op stub — already covered by `test_reuse_signal_fn_absent_is_noop`).

- [ ] Add a stub reuse-signal provider to `paper_filter.py` (kept here, not in `main.py`, so it is unit-testable):

```python
def make_reuse_signal_fn(store=None) -> Callable:
    """Build a reuse_signal_fn for AgentFilter from a StoryStore (spec §H.3, §D).

    Returns evidence-grade corroboration URLs (HF votes / Scholar Inbox / Altmetric) for a paper,
    used by the verifier's allowlist so legitimately-corroborating reuse URLs are not stripped as
    hallucinations. When `store` is None (plan 01 not yet wired) this is a no-op stub returning [].
    """
    if store is None:
        return lambda paper: []

    def _signals(paper: Paper) -> List[str]:
        # Store is the single reader for reuse signals; method name is owned by plan 01/04.
        getter = getattr(store, "reuse_signal_urls_for_arxiv", None)
        if getter is None:
            return []
        try:
            return list(getter(paper.arxiv_id) or [])
        except Exception:
            return []

    return _signals
```

- [ ] Add tests to `tests/test_paper_filter.py`:

```python
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
```

- [ ] **Test command:**

```bash
pytest tests/test_paper_filter.py::ReuseSignalBridgeTests -v
```

Expected: 3 passed.

- [ ] **Commit:** `feat(paper-filter): reuse-signal bridge for AgentFilter (no-op stub when StoryStore absent, §H.3)`

---

## 10. Task 9 — Gray-compare harness + flip default to `cascade` (gated)

Spec §H.3 migration: "first wrap current behavior (zero change, add test net), then add `AgentFilter` defaulting to `cascade`, gray-compare the retention difference of the agent modality vs pure API before tuning thresholds." We add an offline gray-compare test that, using replayed agent fixtures, demonstrates the cascade changes only the borderline band's decisions and never the confident bands. Only after this is green do we flip the repo default.

- [ ] Add the gray-compare test to `tests/test_paper_filter.py`:

```python
class GrayCompareTests(unittest.TestCase):
    def test_cascade_differs_from_api_only_only_on_borderline(self) -> None:
        # Confident keep/drop must be identical between api_only and cascade; only the borderline
        # band may differ (because the agent re-judged it).
        scores = {
            "2501.ck": (10, 10, "ck"),   # confident keep
            "2501.cd": (3, 3, "cd"),     # confident drop
            "2501.b1": (7, 9, "b1"),     # borderline (rel 7 in [6,8))
            "2501.b2": (6, 9, "b2"),     # borderline (rel 6 in [6,8))
        }
        papers = [_paper(i) for i in scores]

        api = ApiScoreFilter(prompts=("s", "t", "sc", "pt", "pa"), config=_config(), gpt_fn=_fake_gpt_factory(scores))
        api_only = cascade_filter(papers, "topic", _config(PAPER_FILTER={"mode": "api_only"}),
                                  rule_filter=None, api_filter=api, agent_filter=None)

        # Agent flips both borderline papers to drop (a plausible "deeper read says off-topic").
        class FlipAgent:
            def judge(self, paper, criteria):
                return FilterVerdict(keep=False, relevance=5.0, novelty=5.0,
                                     rationale="agent: borderline -> drop after full-text", evidence=[])

        api2 = ApiScoreFilter(prompts=("s", "t", "sc", "pt", "pa"), config=_config(), gpt_fn=_fake_gpt_factory(scores))
        cascade = cascade_filter(papers, "topic",
                                 _config(PAPER_FILTER={"mode": "cascade", "agent_borderline_low": "6.0", "agent_borderline_high": "8.0"}),
                                 rule_filter=None, api_filter=api2, agent_filter=FlipAgent())

        api_keep = {p.arxiv_id: v.keep for p, v in zip(papers, api_only)}
        casc_keep = {p.arxiv_id: v.keep for p, v in zip(papers, cascade)}
        # confident bands identical
        self.assertEqual(api_keep["2501.ck"], casc_keep["2501.ck"])
        self.assertEqual(api_keep["2501.cd"], casc_keep["2501.cd"])
        # borderline band changed (agent overrode)
        self.assertNotEqual(api_keep["2501.b1"], casc_keep["2501.b1"])
        self.assertNotEqual(api_keep["2501.b2"], casc_keep["2501.b2"])
```

- [ ] **Test command:**

```bash
pytest tests/test_paper_filter.py::GrayCompareTests -v
```

Expected: 1 passed.

- [ ] **Flip the default** in `configs/config.ini` and `configs/templates/config.template.ini` to `mode = cascade` (the contract §3 recommended value), now that the gray-compare proves confident bands are preserved. Update the inline comment to note the change and that `api_only` remains available for byte-identical legacy behavior.

> **Note:** flipping requires the stage-6 runtime (`judge_paper_with_agent`) to be live, otherwise a `cascade` run hits the `NotImplementedError` stub. If this stage lands **before** plan 07, keep the repo default at `api_only` and record a follow-up checkbox: "flip `[PAPER_FILTER] mode` to `cascade` once plan 07 ships the agent transport." Do not flip a default that would break the production run.

- [ ] **Decision checkbox:**
  - [ ] If plan 07 (agent transport) is live → flip default to `cascade`, commit `feat(paper-filter): default to cascade modality after gray-compare`.
  - [ ] Else → leave default `api_only`, add the deferred-flip note to `docs/superpowers/plans/2026-06-03-agent-native-rewrite-00-overview.md` execution notes (or this file's tail), commit `docs(paper-filter): record deferred cascade-default flip pending stage-6 transport`.

---

## 11. Task 10 — Final invariant + full-suite green gate

- [ ] Confirm INV6 holds end-to-end: every `AgentFilter.judge` path returns a verdict that passed `_verify_agent_response` or the conservative fallback (covered by `AgentFilterTests`), the model id and temperature are pinned (`AGENT_MODEL_ID`, `AGENT_TEMPERATURE`) and passed to `agent_fn`, and the cascade never bypasses the verifier (the agent is only reached via `AgentFilter.judge`).

- [ ] Confirm zero-behavior-change: `mode=api_only` (repo default unless Task 9 flipped it) routes through the unchanged `filter_by_gpt` call in `main.py`; the equivalence proof (`ApiScoreFilterTests.test_zero_behavior_change_against_real_split_rule`, `RuleFilterTests.test_rulefilter_equivalent_to_filter_papers_by_hindex`) is green.

- [ ] **Full project test command** (this module + adjacent suites must stay green):

```bash
pytest tests/test_paper_filter.py tests/test_paper_topics.py tests/test_prompt_loader.py -v
```

Expected: all green; `tests/test_paper_filter.py` reports 27 passed (3+4+4+6+4+1+3+1+1 across the classes).

- [ ] **Lint / import sanity:**

```bash
python -c "import arxiv_assistant.filters.paper_filter as m; print([n for n in ('FilterVerdict','PaperFilter','RuleFilter','ApiScoreFilter','AgentFilter','cascade_filter','make_reuse_signal_fn') if hasattr(m,n)])"
```

Expected: all seven names listed.

- [ ] **Commit:** `test(paper-filter): full-suite green gate + INV6/zero-change assertions`

---

## 12. Acceptance checklist (stage "done" criteria)

- [ ] `arxiv_assistant/filters/paper_filter.py` exists, < ~300 lines, defines the six locked symbols + `make_reuse_signal_fn`, using only the contract §2.9 signatures.
- [ ] `RuleFilter` per-paper decision == `filter_papers_by_hindex` survivorship (proven).
- [ ] `ApiScoreFilter` selected/filtered partition == `filter_by_gpt` partition over the cutoff truth-table (proven); scores pass through verbatim (same scale).
- [ ] `AgentFilter` is temperature 0, structured, pinned model id, **always** followed by the deterministic verifier; verifier strips hallucinated evidence URLs and rejects malformed schema → conservative fallback (INV6).
- [ ] `cascade_filter` routes `api_only`/`agent_only`/`cascade`; in `cascade` only the `[low, high)` relevance band reaches the agent; confident bands keep the API decision.
- [ ] `main.py` engages the modality behind `[PAPER_FILTER] mode`; `api_only` reproduces today's `selected/filtered/hotspot/render/archival/bilingual` behavior with no change.
- [ ] §H.3 reuse-signal bridge is a no-op stub when `StoryStore` is absent; flows real URLs into the verifier allowlist when present.
- [ ] record/replay fixtures committed under `tests/fixtures/agent/paper_filter/`; tests never hit the network.
- [ ] `[PAPER_FILTER]` block added to `configs/config.ini` and `configs/templates/config.template.ini` with documented defaults.
- [ ] Default `mode` is `cascade` **iff** the stage-6 agent transport is live; otherwise `api_only` with a recorded deferred-flip note.
