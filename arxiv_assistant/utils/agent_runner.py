"""Shared Claude Code headless subagent transport (spec §2.11 / §4 INV6).

Single pin-point for:
- Model id and temperature-0 invocation (callers pass the model string).
- ``claude -p`` envelope parsing: the real ``--output-format json`` result has
  shape ``{"type":"result","subtype":"success","result":"<text>"}`` where
  ``result`` holds the model's final message (a JSON string for structured
  agents).
- Minimal schema validation: required top-level keys are checked; callers run
  their own deterministic verifier on top (INV6).
- ``AgentError`` on every failure mode; callers degrade deterministically and
  NEVER let ``AgentError`` propagate to end-users uncaught.

Used by:
- Stage 3 (DateVerify Tier-1/2 wired in task 5).
- Stage 6 (Synthesize).
- Stage 8 (AgentFilter).

Tests ``@patch("arxiv_assistant.utils.agent_runner.subprocess.run")`` to replay
fixtures — zero real subprocess in the test suite.
"""
from __future__ import annotations

import json
import subprocess
from typing import Optional


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------


class AgentError(Exception):
    """Raised by run_agent on any failure mode.

    Callers MUST catch this and degrade deterministically (spec §E / §4 INV6).
    Subcases:
    - Non-zero subprocess exit code.
    - subprocess.TimeoutExpired.
    - Unparseable JSON envelope or empty stdout.
    - Malformed inner payload (not a dict / missing required schema keys).
    - Schema-validation failure (returned dict does not satisfy ``schema``).
    """


# ---------------------------------------------------------------------------
# Minimal schema validator (jsonschema is not a project dependency)
# ---------------------------------------------------------------------------


def _validate_schema(data: dict, schema: dict) -> None:
    """Minimal structural validator — checks required keys and their types.

    ``schema`` format (subset of JSON Schema, sufficient for our structured
    agent outputs):

    .. code-block:: python

        {
          "required": ["key1", "key2"],          # optional list
          "properties": {                         # optional dict
            "key1": {"type": "string"},           # "string" | "number" | "boolean" | "array" | "object"
            "key2": {"type": "number"},
          }
        }

    Raises ``AgentError`` on the first violation found.
    """
    # 1. Required-key check.
    required = schema.get("required", [])
    for key in required:
        if key not in data:
            raise AgentError(
                f"Schema validation failed: required key {key!r} missing from agent output. "
                f"Got keys: {list(data.keys())}"
            )

    # 2. Type check for declared properties.
    type_map: dict[str, type] = {
        "string": str,
        "number": (int, float),  # type: ignore[dict-item]
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    properties = schema.get("properties", {})
    for key, prop_schema in properties.items():
        if key not in data:
            continue  # only validate present keys (required-check already covered missing ones)
        declared_type = prop_schema.get("type")
        if declared_type is None:
            continue
        expected = type_map.get(declared_type)
        if expected is None:
            continue
        value = data[key]
        # Booleans are a subclass of int in Python; treat bool as boolean-only.
        if declared_type == "number" and isinstance(value, bool):
            raise AgentError(
                f"Schema validation failed: key {key!r} expected number, got bool."
            )
        if declared_type == "boolean":
            if not isinstance(value, bool):
                raise AgentError(
                    f"Schema validation failed: key {key!r} expected boolean, got {type(value).__name__}."
                )
        elif not isinstance(value, expected):  # type: ignore[arg-type]
            raise AgentError(
                f"Schema validation failed: key {key!r} expected {declared_type}, "
                f"got {type(value).__name__}."
            )


# ---------------------------------------------------------------------------
# Envelope parser
# ---------------------------------------------------------------------------


def _parse_envelope(raw_stdout: str) -> dict:
    """Parse the ``claude -p --output-format json`` envelope and return the
    inner structured dict.

    The real envelope shape is::

        {"type": "result", "subtype": "success", "result": "<text>", ...}

    where ``result`` is the model's final message text.  For structured agents
    the text is itself a JSON object string.

    Raises ``AgentError`` for any parse failure.
    """
    raw = (raw_stdout or "").strip()
    if not raw:
        raise AgentError("Agent returned empty stdout.")

    try:
        envelope = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise AgentError(f"Failed to parse JSON envelope: {exc}. stdout={raw[:200]!r}") from exc

    if not isinstance(envelope, dict):
        raise AgentError(
            f"Expected JSON envelope to be a dict, got {type(envelope).__name__}."
        )

    # Reject error envelopes even when the exit code was 0: claude can report an
    # in-band error (is_error / subtype="error_*") with a zero return code.
    subtype = envelope.get("subtype", "")
    if envelope.get("is_error") or (isinstance(subtype, str) and subtype.startswith("error")):
        raise AgentError(
            f"Agent reported an error envelope: is_error={envelope.get('is_error')!r}, "
            f"subtype={subtype!r}, result={str(envelope.get('result'))[:200]!r}"
        )

    # Extract the inner payload from envelope["result"].
    inner_raw = envelope.get("result", envelope)

    # If "result" is a string, it is the model's text output — parse it as JSON.
    if isinstance(inner_raw, str):
        inner_raw = inner_raw.strip()
        if not inner_raw:
            raise AgentError("Agent envelope 'result' field is an empty string.")
        try:
            inner = json.loads(inner_raw)
        except (ValueError, TypeError) as exc:
            raise AgentError(
                f"Agent 'result' field is not valid JSON: {exc}. "
                f"result={inner_raw[:200]!r}"
            ) from exc
    elif isinstance(inner_raw, dict):
        inner = inner_raw
    else:
        raise AgentError(
            f"Agent envelope 'result' is neither a string nor a dict: "
            f"{type(inner_raw).__name__}."
        )

    if not isinstance(inner, dict):
        raise AgentError(
            f"Parsed inner agent output is not a dict: {type(inner).__name__}."
        )

    return inner


# ---------------------------------------------------------------------------
# Public transport function
# ---------------------------------------------------------------------------


def run_agent(
    prompt: str,
    *,
    schema: dict,
    model: str,
    tools: Optional[list[str]] = None,
    timeout_s: int = 120,
) -> dict:
    """Dispatch a stateless Claude Code headless subagent and return its output.

    Invokes ``claude -p <prompt> --output-format json --model <model>``
    (plus optional ``--allowedTools`` when *tools* is given) via subprocess.
    Parses the JSON envelope, validates the inner dict against *schema*, and
    returns the dict.

    Temperature is always 0 (Claude Code CLI default for ``-p`` / headless
    mode; no flag needed).

    Args:
        prompt:    The full prompt string passed to ``claude -p``.
        schema:    Minimal JSON-Schema-like dict used by ``_validate_schema``
                   to check the returned dict (required keys + property types).
        model:     Model identifier string (e.g. ``"claude-opus-4-8"``).
                   Pinned per-consumer so a single config field controls it.
        tools:     Optional list of tool names passed via ``--allowedTools``.
                   When *None* or empty, the flag is omitted.
        timeout_s: Subprocess timeout in seconds (default 120).

    Returns:
        The inner structured dict extracted from the ``claude -p`` JSON
        envelope, validated against *schema*.

    Raises:
        AgentError: On non-zero exit code, subprocess.TimeoutExpired,
                    unparseable envelope, malformed inner payload, or
                    schema-validation failure.
    """
    cmd: list[str] = ["claude", "-p", prompt, "--output-format", "json", "--model", model]
    if tools:
        cmd += ["--allowedTools"] + list(tools)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise AgentError(
            f"Agent subprocess timed out after {timeout_s}s for model={model!r}."
        ) from exc
    except OSError as exc:
        raise AgentError(
            f"Agent subprocess failed to start (OSError): {exc}."
        ) from exc

    if proc.returncode != 0:
        raise AgentError(
            f"Agent subprocess exited with code {proc.returncode}. "
            f"stderr={proc.stderr[:300]!r}. stdout={proc.stdout[:200]!r}."
        )

    inner = _parse_envelope(proc.stdout)
    _validate_schema(inner, schema)
    return inner
