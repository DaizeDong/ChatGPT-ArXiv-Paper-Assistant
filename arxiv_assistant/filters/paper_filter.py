"""Paper-filter strategy module (spec H.1-H.3).

Defines the unified PaperFilter protocol and three concrete strategies:
  - RuleFilter:     hard h-index pre-filter (wraps filter_author.filter_papers_by_hindex)
  - ApiScoreFilter: GPT-batch scorer (wraps filter_gpt.filter_by_gpt, zero behavior change)
  - AgentFilter:    Claude Code subagent + deterministic verifier (INV6)
  - cascade_filter: confidence-aware router (api_only / agent_only / cascade)
  - make_reuse_signal_fn: reuse-signal bridge for AgentFilter (no-op when StoryStore absent)

Default mode is api_only (byte-identical to the historical pipeline).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from arxiv_assistant.utils.utils import Paper


# ---------------------------------------------------------------------------
# FilterVerdict: unified, strategy-agnostic decision for one paper
# ---------------------------------------------------------------------------

@dataclass
class FilterVerdict:
    """Unified, strategy-agnostic decision for one paper.

    relevance/novelty are on the SAME 1..10 scale the existing GPT scorer emits
    (stored as float so agent strategies may return fractional confidence), so the
    historical ranking/thresholds keep working unchanged.
    """

    keep: bool
    relevance: float
    novelty: float
    rationale: str
    evidence: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PaperFilter: runtime-checkable Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class PaperFilter(Protocol):
    def judge(self, paper: Paper, criteria: str) -> FilterVerdict: ...


# ---------------------------------------------------------------------------
# RuleFilter: hard pre-filter wrapping h-index gate
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ApiScoreFilter: wraps filter_by_gpt, ZERO behavior change
# ---------------------------------------------------------------------------

class ApiScoreFilter:
    """Wraps the existing single-call GPT scorer (filter_by_gpt) behind the PaperFilter protocol.

    ZERO behavior change: it calls filter_by_gpt once over the whole batch and projects each
    per-paper result into a FilterVerdict. keep == "paper landed in selected_results", which is
    exactly NOT (RELEVANCE < relevance_cutoff or NOVELTY < novelty_cutoff). RELEVANCE/NOVELTY/COMMENT
    are passed through verbatim, so the score scale and the selected/filtered partition are identical
    to the historical pipeline.

    gpt_fn is injected (defaults to the real filter_by_gpt) so tests run offline.
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


# ---------------------------------------------------------------------------
# AgentFilter: Claude Code subagent + deterministic verifier (INV6)
# ---------------------------------------------------------------------------

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
    deterministic verifier (INV6). The subagent transport is injected as agent_fn so tests replay
    captured JSON. reuse_signal_fn (spec H.3) optionally supplies corroborating reuse-source URLs
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


# ---------------------------------------------------------------------------
# cascade_filter: confidence-aware router
# ---------------------------------------------------------------------------

def cascade_filter(
    papers: Sequence[Paper],
    criteria: str,
    config,
    *,
    rule_filter: Optional["RuleFilter"] = None,
    api_filter: Optional["ApiScoreFilter"] = None,
    agent_filter: Optional["AgentFilter"] = None,
) -> List[FilterVerdict]:
    """Confidence-aware router (spec H.2). Returns verdicts in input order.

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


# ---------------------------------------------------------------------------
# make_reuse_signal_fn: reuse-signal bridge (spec H.3)
# ---------------------------------------------------------------------------

def make_reuse_signal_fn(store=None) -> Callable:
    """Build a reuse_signal_fn for AgentFilter from a StoryStore (spec H.3, D).

    Returns evidence-grade corroboration URLs (HF votes / Scholar Inbox / Altmetric) for a paper,
    used by the verifier's allowlist so legitimately-corroborating reuse URLs are not stripped as
    hallucinations. When store is None (plan 01 not yet wired) this is a no-op stub returning [].
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
