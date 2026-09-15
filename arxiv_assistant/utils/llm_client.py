"""Shared LLM + agent backend resolution for BOTH pipelines (paper digest + hotspots)."""
from __future__ import annotations
from arxiv_assistant.utils.models import DEFAULT_AGENT_MODEL

import os
from typing import Any

_DEFAULT_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_LLM_MODEL = "gpt-5.4"
_DEFAULT_AGENT_MODEL = DEFAULT_AGENT_MODEL


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
