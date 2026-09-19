"""Stage 6 (Synthesize) Claude Code headless transport (spec §G.3 / INV6)."""

from __future__ import annotations

import json
from typing import Any

from arxiv_assistant.utils.models import DEFAULT_AGENT_MODEL
from arxiv_assistant.utils.agent_runner import AgentError, run_agent

# Default real model id when the caller passes a falsy / placeholder value.
_DEFAULT_REAL_MODEL = DEFAULT_AGENT_MODEL
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
    """Run the bilingual Synthesize subagent via ``claude -p`` and return its rows."""
    real_model = _resolve_model(model)
    prompt = _build_prompt(topics)
    try:
        return run_agent(prompt, schema=_SYNTH_SCHEMA, model=real_model, timeout_s=180)
    except AgentError:
        return {"topics": []}
