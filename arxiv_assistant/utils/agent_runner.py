"""Shared Claude Code headless subagent transport (spec §2.11 / §4 INV6)."""
from __future__ import annotations

import json
import re
import subprocess
from typing import Optional


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------


class AgentError(Exception):
    """Raised by run_agent on any failure mode."""


# ---------------------------------------------------------------------------
# Minimal schema validator (jsonschema is not a project dependency)
# ---------------------------------------------------------------------------


def _validate_schema(data: dict, schema: dict) -> None:
    """Minimal structural validator — checks required keys and their types."""
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
# JSON extraction from fenced / prose-wrapped result text
# ---------------------------------------------------------------------------

# Matches the FIRST fenced code block, optionally tagged ```json (or any tag),
# capturing the block's inner content. DOTALL so the body can span newlines.
_FENCE_RE = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)


def _iter_balanced_objects(text: str):
    """Yield each top-level brace-balanced ``{...}`` substring, in order."""
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    yield text[start:i + 1]
                    start = -1


def _extract_json_object(text: str) -> Optional[str]:
    """Best-effort extraction of a JSON-object candidate from result *text*."""
    for fence in _FENCE_RE.finditer(text):
        candidate = fence.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except (ValueError, TypeError):
            continue
    for candidate in _iter_balanced_objects(text):
        try:
            json.loads(candidate)
            return candidate
        except (ValueError, TypeError):
            continue
    return None


# ---------------------------------------------------------------------------
# Envelope parser
# ---------------------------------------------------------------------------


def _parse_envelope(raw_stdout: str) -> dict:
    """Parse the ``claude -p --output-format json`` envelope and return the
    inner structured dict.
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
            # Fast path: pure-JSON result (unchanged behavior for structured agents).
            inner = json.loads(inner_raw)
        except (ValueError, TypeError) as exc:
            # Agentic tasks often narrate before/after the JSON (prose + a fenced
            # code block). Try to extract a JSON-object candidate before giving up.
            candidate = _extract_json_object(inner_raw)
            if candidate is not None:
                try:
                    inner = json.loads(candidate)
                except (ValueError, TypeError):
                    candidate = None
            if candidate is None:
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
    """Dispatch a stateless Claude Code headless subagent and return its output."""
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
