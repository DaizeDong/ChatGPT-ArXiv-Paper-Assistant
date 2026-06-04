"""Stage 6 (Synthesize) Claude Code headless transport (spec §G.3 / INV6).

Wires the bilingual Synthesize agent to ``agent_runner.run_agent`` (``claude -p``
headless, deterministic / temperature 0).  For each input topic the agent is
asked to produce a bilingual (en + zh) headline + summary and to echo back ONLY
that topic's own real evidence URLs — those URLs are fed INTO the prompt so the
downstream verifier ``kernel._synthesis_row_valid`` (anti-hallucination, INV6)
accepts the row.

The verifier is NEVER bypassed: this transport merely produces candidate rows;
``_stage_synthesize`` re-checks every returned row's bilingual fields and cited
evidence against the topic's real evidence-URL set before applying it, and
degrades any rejected topic to the deterministic heuristic fallback.

On any ``AgentError`` this returns ``{"topics": []}`` so ``_stage_synthesize``
rejects all rows and falls back to heuristic — degrade-not-crash, identical to
today's behaviour.  No import side-effects beyond ``json`` + ``agent_runner``.
"""

from __future__ import annotations

import json
from typing import Any

from arxiv_assistant.utils.agent_runner import AgentError, run_agent

# Default real model id when the caller passes a falsy / placeholder value.
_DEFAULT_REAL_MODEL = "claude-sonnet-4-6"
_PLACEHOLDER_MODELS = {"", "claude-code-subagent"}

# JSON-Schema-like structural contract validated by run_agent's _validate_schema.
_SYNTH_SCHEMA = {
    "required": ["topics"],
    "properties": {
        "topics": {"type": "array"},
    },
}


def _evidence_urls(topic: dict[str, Any]) -> list[str]:
    """Collect a topic's real evidence URLs (mirrors kernel._story_evidence_urls).

    The agent is told to cite ONLY from this set, so the kernel verifier passes.
    """
    urls: list[str] = []
    seen: set[str] = set()
    for key in ("EVIDENCE_URLS", "evidence_urls", "SOURCE_URLS"):
        for u in topic.get(key, []) or []:
            if u and str(u) not in seen:
                seen.add(str(u))
                urls.append(str(u))
    if topic.get("URL") and str(topic["URL"]) not in seen:
        seen.add(str(topic["URL"]))
        urls.append(str(topic["URL"]))
    return urls


def _resolve_model(model: str) -> str:
    """Default a falsy/placeholder model to the real Synthesize model id."""
    if not model or str(model) in _PLACEHOLDER_MODELS:
        return _DEFAULT_REAL_MODEL
    return str(model)


def _build_prompt(topics: list[dict[str, Any]]) -> str:
    """Build the bilingual Synthesize prompt, embedding each topic's real
    evidence URLs so the agent cites only from them."""
    blocks: list[str] = []
    for topic in topics:
        tid = str(topic.get("TOPIC_ID", ""))
        headline = str(topic.get("HEADLINE") or topic.get("title", "")).strip()
        why = str(topic.get("WHY_IT_MATTERS", "")).strip()
        urls = _evidence_urls(topic)
        url_lines = "\n".join(f"    - {u}" for u in urls) or "    (none)"
        blocks.append(
            f"### TOPIC_ID: {tid}\n"
            f"English headline: {headline}\n"
            f"Why it matters (English): {why}\n"
            f"EVIDENCE_URLS (cite ONLY these, verbatim):\n{url_lines}"
        )
    topics_block = "\n\n".join(blocks)

    return (
        "You are a bilingual (English + Simplified Chinese) tech-news editor for an "
        "AI research daily digest.  For EACH topic below, write a crisp bilingual "
        "headline and a 1-2 sentence bilingual summary.\n\n"
        "## Topics\n"
        f"{topics_block}\n\n"
        "## Task\n"
        "Return a JSON object with this exact shape (and ONLY this shape):\n"
        '  {"topics": [\n'
        '    {"TOPIC_ID": "<echo the topic id verbatim>",\n'
        '     "headline_en": "<English headline, non-empty>",\n'
        '     "headline_zh": "<Simplified Chinese headline, non-empty>",\n'
        '     "summary_en": "<English summary, non-empty>",\n'
        '     "summary_zh": "<Simplified Chinese summary, non-empty>",\n'
        '     "evidence": ["<echo ONLY this topic\'s provided EVIDENCE_URLS, verbatim>"]}\n'
        "  ]}\n\n"
        "Rules:\n"
        "- Produce exactly one object per input topic, echoing its TOPIC_ID verbatim.\n"
        "- All four bilingual fields must be non-empty.\n"
        "- The 'evidence' list MUST contain only URLs copied verbatim from that "
        "topic's own EVIDENCE_URLS above.  DO NOT invent, modify, or borrow URLs "
        "from other topics — hallucinated evidence causes automatic rejection.\n"
        "- Output ONLY the JSON object: no markdown fences, no prose."
    )


def synthesize_bilingual(topics: list[dict[str, Any]], *, model: str, temperature: float) -> dict[str, Any]:
    """Run the bilingual Synthesize subagent via ``claude -p`` and return its rows.

    Args:
        topics:      Featured topic dicts (each carrying TOPIC_ID, English
                     headline / why-it-matters, and real evidence URLs).
        model:       Pinned model id; falsy / placeholder defaults to the real
                     ``claude-sonnet-4-6``.  Recorded into the manifest by the
                     caller (``_stage_synthesize``).
        temperature: Accepted for call-site compatibility and recorded in the
                     manifest; ``claude -p`` is deterministic so it is NOT
                     passed to ``run_agent``.

    Returns:
        ``{"topics": [{TOPIC_ID, headline_en, headline_zh, summary_en,
        summary_zh, evidence}, ...]}`` — already a dict, gated downstream by
        ``kernel._synthesis_row_valid``.  On ``AgentError`` returns
        ``{"topics": []}`` so the caller degrades every topic to heuristic.
    """
    real_model = _resolve_model(model)
    prompt = _build_prompt(topics)
    try:
        return run_agent(prompt, schema=_SYNTH_SCHEMA, model=real_model, timeout_s=180)
    except AgentError:
        return {"topics": []}
