"""Claude Code subagent transport for the paper Agent-filter modality (spec §H).

Provides ``judge_paper_with_agent``, a plug-in adapter that satisfies the
``agent_fn`` contract expected by ``AgentFilter``:

    agent_fn(paper, criteria, *, reuse_signals, temperature, model) -> str

The function builds a concise verdict prompt, dispatches a headless
``claude -p`` subagent via ``agent_runner.run_agent``, and returns the raw
JSON string so that ``AgentFilter._verify_agent_response`` (INV6) validates
it deterministically before accepting it.

On any transport failure (``AgentError``) the function degrades gracefully:
it returns a conservative ``keep=False`` JSON string and never propagates the
error to the pipeline.

No import side-effects: this module is imported lazily by main.py on the
non-default (cascade / agent_only) code paths only.
"""

from __future__ import annotations

import json

from arxiv_assistant.utils.agent_runner import AgentError, run_agent

# ---------------------------------------------------------------------------
# Verdict schema for run_agent structural validation
# ---------------------------------------------------------------------------

_VERDICT_SCHEMA = {
    "required": ["keep", "relevance", "novelty"],
    "properties": {
        "keep": {"type": "boolean"},
        "relevance": {"type": "number"},
        "novelty": {"type": "number"},
        "rationale": {"type": "string"},
        "evidence": {"type": "array"},
    },
}

# Placeholder model id used by AgentFilter (recorded in provenance); the adapter
# maps it to a real model id before passing to run_agent.
_PLACEHOLDER_MODEL = "claude-code-subagent"
_DEFAULT_REAL_MODEL = "claude-sonnet-4-6"


def _resolve_model(model: str) -> str:
    """Map the AgentFilter placeholder to a real model id.

    If *model* is the placeholder sentinel ``"claude-code-subagent"``, resolve
    it to the configured ``agent_model`` (read from the pipeline CONFIG if
    importable), falling back to ``"claude-sonnet-4-6"`` when the config is
    absent or the key is not present.  Any other value (a real model id) is
    returned unchanged.
    """
    if model != _PLACEHOLDER_MODEL:
        return model

    try:
        from arxiv_assistant.environment import CONFIG  # lazy; no side-effects on import path
        return CONFIG["PAPER_FILTER"].get("agent_model", _DEFAULT_REAL_MODEL)
    except Exception:
        return _DEFAULT_REAL_MODEL


def _build_prompt(paper, criteria: str, reuse_signals) -> str:
    """Build a concise verdict prompt for the paper-filter agent."""
    signals_block = ""
    if reuse_signals:
        urls = "\n".join(f"  - {u}" for u in reuse_signals)
        signals_block = (
            f"\nCorroborating reuse signals (pre-verified; you MAY cite these as evidence):\n{urls}\n"
        )

    return (
        "You are a research-paper relevance judge.  Judge whether the paper below "
        "should be KEPT for a curated daily digest, given the stated criteria.\n\n"
        f"## Criteria\n{criteria}\n\n"
        f"## Paper\n"
        f"Title: {paper.title}\n"
        f"ArXiv ID: {paper.arxiv_id}\n"
        f"Authors: {', '.join(paper.authors or [])}\n"
        f"Abstract:\n{paper.abstract}\n"
        f"{signals_block}\n"
        "## Task\n"
        "Return a JSON object with these fields (and ONLY these fields):\n"
        '  "keep": true/false\n'
        '  "relevance": integer 1-10 (how relevant to the criteria)\n'
        '  "novelty": integer 1-10 (how novel/impactful)\n'
        '  "rationale": one-sentence explanation\n'
        '  "evidence": list of URLs (cite ONLY the paper\'s own arXiv URL '
        f'https://arxiv.org/abs/{paper.arxiv_id} or any pre-verified reuse signals above; '
        "DO NOT invent external citations — hallucinated evidence will cause automatic rejection)\n\n"
        "Output ONLY the JSON object, no markdown fences, no prose."
    )


def judge_paper_with_agent(
    paper,
    criteria: str,
    *,
    reuse_signals=None,
    temperature: float = 0.0,  # noqa: ARG001 — recorded in provenance; run_agent runs temp-0 by design
    model: str = _PLACEHOLDER_MODEL,
) -> str:
    """Adapter satisfying AgentFilter's agent_fn contract.

    Calls ``run_agent`` with a structured verdict prompt and returns the result
    as a JSON string for ``_verify_agent_response`` (INV6) to validate.

    On ``AgentError``: returns a conservative ``keep=False`` fallback JSON
    string so the pipeline degrades safely without crashing.

    Args:
        paper:          The ``Paper`` dataclass instance to judge.
        criteria:       Topic/relevance criteria string from the pipeline config.
        reuse_signals:  Optional list of pre-verified corroborating URLs; these
                        are added to the prompt and will pass the evidence verifier.
        temperature:    Ignored — ``claude -p`` subagents run deterministically.
                        Accepted for call-site compatibility with AgentFilter.
        model:          Model id string.  The placeholder ``"claude-code-subagent"``
                        is resolved to the configured real model before dispatch.

    Returns:
        A JSON string whose shape satisfies ``_verify_agent_response``.
    """
    real_model = _resolve_model(model)
    prompt = _build_prompt(paper, criteria, reuse_signals or [])

    try:
        result = run_agent(prompt, schema=_VERDICT_SCHEMA, model=real_model, timeout_s=120)
        return json.dumps(result)
    except AgentError as exc:
        return json.dumps({
            "keep": False,
            "rationale": f"agent transport failed: {exc}",
            "evidence": [],
        })
