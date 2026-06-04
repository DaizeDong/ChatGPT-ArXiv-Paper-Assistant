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
