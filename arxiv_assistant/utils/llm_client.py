"""Shared LLM + agent backend resolution for BOTH pipelines (paper digest + hotspots).

Single source of truth for:
  - the OpenAI client / api-key / base-url (``get_openai_client`` / ``resolve_openai_config``)
  - the OpenAI model id (``resolve_llm_model`` -> ``[LLM] model``)
  - the Claude Code (``claude -p``) agent model id (``resolve_agent_model`` -> ``[AGENT] model``)

Back-compatibility is intentional: the legacy per-section keys (``[SELECTION] model``,
``[HOTSPOTS] model_enrich/model_screen``, ``[PAPER_FILTER] agent_model``) still WIN when present
(callers pass them as ``override=``), so an existing config produces byte-identical behaviour.
The new ``[LLM]`` / ``[AGENT]`` sections only supply the shared *fallback*.
"""
from __future__ import annotations

import os
from typing import Any

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_LLM_MODEL = "gpt-5.4"
_DEFAULT_AGENT_MODEL = "claude-sonnet-4-6"


def resolve_openai_config(*, api_key: str | None = None, base_url: str | None = None) -> tuple[str, str]:
    """Resolve the OpenAI ``(api_key, base_url)`` -- the ONE place env defaults live.

    ``api_key`` defaults to ``$OPENAI_API_KEY`` (empty string if unset, matching the
    historical ``enrich`` behaviour); ``base_url`` defaults to ``$OPENAI_BASE_URL`` then
    ``https://api.openai.com/v1``, trailing slash stripped.
    """
    key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
    url = (base_url or os.environ.get("OPENAI_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")
    return key, url


def get_openai_client(*, api_key: str | None = None, base_url: str | None = None) -> Any:
    """The SINGLE place the ``openai.OpenAI`` client is constructed."""
    from openai import OpenAI  # lazy: keep import cost off modules that only resolve config

    key, url = resolve_openai_config(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=key, base_url=url)


def _section_get(config: Any, section: str, key: str) -> str | None:
    """``config[section].get(key)`` that tolerates a missing section/key and blank values."""
    try:
        value = config[section].get(key)
    except (KeyError, TypeError):
        return None
    value = (value or "").strip() if isinstance(value, str) else value
    return value or None


def resolve_llm_model(config: Any, *, override: str | None = None, fallback: str = _DEFAULT_LLM_MODEL) -> str:
    """OpenAI model id: explicit ``override`` (legacy per-section key) wins, then ``[LLM] model``, then fallback."""
    if override and str(override).strip():
        return str(override).strip()
    return _section_get(config, "LLM", "model") or fallback


def resolve_agent_model(config: Any, *, override: str | None = None, fallback: str = _DEFAULT_AGENT_MODEL) -> str:
    """``claude -p`` agent model id: explicit ``override`` (legacy per-section key) wins, then ``[AGENT] model``, then fallback."""
    if override and str(override).strip():
        return str(override).strip()
    return _section_get(config, "AGENT", "model") or fallback
