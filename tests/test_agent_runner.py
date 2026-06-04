"""Tests for arxiv_assistant/utils/agent_runner.py (spec §2.11 / §4 INV6).

Record/replay strategy
----------------------
All tests patch ``arxiv_assistant.utils.agent_runner.subprocess.run`` to return
a fake ``CompletedProcess``-like object whose ``stdout`` is a real-shape
``claude -p --output-format json`` envelope JSON.  Zero real subprocesses are
spawned.

Fixtures live under ``tests/fixtures/agent/`` (real-shape captured envelopes):
  - dateverify_tier1_stale_pollution.json  — inner payload, wrapped in envelope
  - dateverify_tier1_clean.json
  - dateverify_tier2_deep.json
"""
from __future__ import annotations

import json as _json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from arxiv_assistant.utils.agent_runner import AgentError, run_agent

FIXTURES = Path(__file__).parent / "fixtures" / "agent"

# Minimal schema matching dateverify.out.v1 inner payloads.
_DATEVERIFY_SCHEMA = {
    "required": ["schema", "verified_first_date", "confidence", "evidence", "stale_date_pollution"],
    "properties": {
        "schema": {"type": "string"},
        "verified_first_date": {"type": "string"},
        "confidence": {"type": "number"},
        "evidence": {"type": "array"},
        "stale_date_pollution": {"type": "boolean"},
    },
}

# A permissive schema that accepts any dict (used to isolate envelope-parsing tests).
_ANY_SCHEMA: dict = {}


# ---------------------------------------------------------------------------
# Helper: build a fake subprocess.CompletedProcess-like object
# ---------------------------------------------------------------------------


class _FakeProc:
    """Minimal stand-in for subprocess.CompletedProcess."""

    def __init__(self, *, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _wrap_fixture(fixture_name: str) -> str:
    """Load an inner-payload fixture and wrap it in a real-shape claude envelope."""
    inner_text = (FIXTURES / fixture_name).read_text(encoding="utf-8")
    # Real ``claude -p --output-format json`` shape: result field is the model's
    # final text output (a JSON string for structured agents).
    envelope = {"type": "result", "subtype": "success", "result": inner_text}
    return _json.dumps(envelope)


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


class TestRunAgentHappyPath(unittest.TestCase):
    def test_parses_stale_pollution_fixture(self) -> None:
        """Full happy path: envelope → validated dict with correct field values."""
        envelope_stdout = _wrap_fixture("dateverify_tier1_stale_pollution.json")

        with patch("arxiv_assistant.utils.agent_runner.subprocess.run",
                   return_value=_FakeProc(stdout=envelope_stdout)):
            out = run_agent(
                "test prompt",
                schema=_DATEVERIFY_SCHEMA,
                model="claude-opus-4-8",
            )

        self.assertEqual(out["verified_first_date"], "2023-11-14T00:00:00Z")
        self.assertAlmostEqual(out["confidence"], 0.9)
        self.assertTrue(out["stale_date_pollution"])
        self.assertIsInstance(out["evidence"], list)
        self.assertEqual(out["schema"], "dateverify.out.v1")

    def test_parses_clean_fixture(self) -> None:
        """Clean (non-stale) fixture round-trips correctly."""
        envelope_stdout = _wrap_fixture("dateverify_tier1_clean.json")

        with patch("arxiv_assistant.utils.agent_runner.subprocess.run",
                   return_value=_FakeProc(stdout=envelope_stdout)):
            out = run_agent(
                "test prompt",
                schema=_DATEVERIFY_SCHEMA,
                model="claude-opus-4-8",
            )

        self.assertEqual(out["verified_first_date"], "2026-06-02T00:00:00Z")
        self.assertFalse(out["stale_date_pollution"])

    def test_parses_tier2_deep_fixture(self) -> None:
        """Tier-2 deep fixture round-trips correctly."""
        envelope_stdout = _wrap_fixture("dateverify_tier2_deep.json")

        with patch("arxiv_assistant.utils.agent_runner.subprocess.run",
                   return_value=_FakeProc(stdout=envelope_stdout)):
            out = run_agent(
                "test prompt",
                schema=_DATEVERIFY_SCHEMA,
                model="claude-opus-4-8",
            )

        self.assertEqual(out["verified_first_date"], "2024-09-30T00:00:00Z")
        self.assertTrue(out["stale_date_pollution"])
        self.assertIn("deep_search:semantic_scholar:corpusId=987654", out["evidence"])

    def test_returns_dict(self) -> None:
        """Return type is always dict on success."""
        envelope_stdout = _wrap_fixture("dateverify_tier1_clean.json")

        with patch("arxiv_assistant.utils.agent_runner.subprocess.run",
                   return_value=_FakeProc(stdout=envelope_stdout)):
            out = run_agent("prompt", schema=_ANY_SCHEMA, model="claude-opus-4-8")

        self.assertIsInstance(out, dict)

    def test_allowed_tools_passed_in_command(self) -> None:
        """When tools is given, --allowedTools flag appears in the subprocess cmd."""
        envelope_stdout = _wrap_fixture("dateverify_tier1_clean.json")
        captured_cmd: list = []

        def _capture(*args, **kwargs):
            captured_cmd.extend(args[0])
            return _FakeProc(stdout=envelope_stdout)

        with patch("arxiv_assistant.utils.agent_runner.subprocess.run", side_effect=_capture):
            run_agent(
                "prompt",
                schema=_ANY_SCHEMA,
                model="claude-opus-4-8",
                tools=["WebSearch", "WebFetch"],
            )

        self.assertIn("--allowedTools", captured_cmd)
        self.assertIn("WebSearch", captured_cmd)
        self.assertIn("WebFetch", captured_cmd)

    def test_no_allowed_tools_when_tools_none(self) -> None:
        """When tools is None, --allowedTools must NOT appear in the cmd."""
        envelope_stdout = _wrap_fixture("dateverify_tier1_clean.json")
        captured_cmd: list = []

        def _capture(*args, **kwargs):
            captured_cmd.extend(args[0])
            return _FakeProc(stdout=envelope_stdout)

        with patch("arxiv_assistant.utils.agent_runner.subprocess.run", side_effect=_capture):
            run_agent("prompt", schema=_ANY_SCHEMA, model="claude-opus-4-8", tools=None)

        self.assertNotIn("--allowedTools", captured_cmd)

    def test_model_id_in_command(self) -> None:
        """The --model flag must carry the caller-supplied model id."""
        envelope_stdout = _wrap_fixture("dateverify_tier1_clean.json")
        captured_cmd: list = []

        def _capture(*args, **kwargs):
            captured_cmd.extend(args[0])
            return _FakeProc(stdout=envelope_stdout)

        with patch("arxiv_assistant.utils.agent_runner.subprocess.run", side_effect=_capture):
            run_agent("prompt", schema=_ANY_SCHEMA, model="claude-haiku-3-5")

        model_idx = captured_cmd.index("--model")
        self.assertEqual(captured_cmd[model_idx + 1], "claude-haiku-3-5")


# ---------------------------------------------------------------------------
# AgentError: non-zero exit
# ---------------------------------------------------------------------------


class TestRunAgentNonZeroExit(unittest.TestCase):
    def test_nonzero_exit_raises_agent_error(self) -> None:
        with patch("arxiv_assistant.utils.agent_runner.subprocess.run",
                   return_value=_FakeProc(returncode=1, stderr="model error")):
            with self.assertRaises(AgentError) as ctx:
                run_agent("prompt", schema=_ANY_SCHEMA, model="claude-opus-4-8")
        self.assertIn("1", str(ctx.exception))

    def test_nonzero_exit_includes_stderr_in_message(self) -> None:
        with patch("arxiv_assistant.utils.agent_runner.subprocess.run",
                   return_value=_FakeProc(returncode=2, stderr="rate limit exceeded")):
            with self.assertRaises(AgentError) as ctx:
                run_agent("prompt", schema=_ANY_SCHEMA, model="claude-opus-4-8")
        self.assertIn("rate limit exceeded", str(ctx.exception))

    def test_raises_agent_error_not_other_exception(self) -> None:
        """Must raise AgentError specifically, not a raw subprocess or ValueError."""
        with patch("arxiv_assistant.utils.agent_runner.subprocess.run",
                   return_value=_FakeProc(returncode=1)):
            with self.assertRaises(AgentError):
                run_agent("prompt", schema=_ANY_SCHEMA, model="claude-opus-4-8")


# ---------------------------------------------------------------------------
# AgentError: timeout
# ---------------------------------------------------------------------------


class TestRunAgentTimeout(unittest.TestCase):
    def test_timeout_raises_agent_error(self) -> None:
        with patch("arxiv_assistant.utils.agent_runner.subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd=["claude"], timeout=120)):
            with self.assertRaises(AgentError) as ctx:
                run_agent("prompt", schema=_ANY_SCHEMA, model="claude-opus-4-8", timeout_s=120)
        self.assertIn("timed out", str(ctx.exception).lower())

    def test_timeout_message_contains_duration(self) -> None:
        with patch("arxiv_assistant.utils.agent_runner.subprocess.run",
                   side_effect=subprocess.TimeoutExpired(cmd=["claude"], timeout=30)):
            with self.assertRaises(AgentError) as ctx:
                run_agent("prompt", schema=_ANY_SCHEMA, model="claude-opus-4-8", timeout_s=30)
        self.assertIn("30", str(ctx.exception))

    def test_oserror_when_binary_missing_raises_agent_error(self) -> None:
        # claude not on PATH / cannot exec -> OSError -> AgentError (not a raw OSError).
        with patch("arxiv_assistant.utils.agent_runner.subprocess.run",
                   side_effect=OSError("[Errno 2] No such file or directory: 'claude'")):
            with self.assertRaises(AgentError):
                run_agent("prompt", schema=_ANY_SCHEMA, model="claude-opus-4-8")


# ---------------------------------------------------------------------------
# AgentError: malformed / unparseable JSON envelope
# ---------------------------------------------------------------------------


class TestRunAgentMalformedEnvelope(unittest.TestCase):
    def test_non_json_stdout_raises_agent_error(self) -> None:
        with patch("arxiv_assistant.utils.agent_runner.subprocess.run",
                   return_value=_FakeProc(stdout="not json at all")):
            with self.assertRaises(AgentError):
                run_agent("prompt", schema=_ANY_SCHEMA, model="claude-opus-4-8")

    def test_in_band_error_envelope_with_exit0_raises_agent_error(self) -> None:
        # claude can report an error in-band (is_error/subtype) with returncode 0;
        # this must still raise, not pass the error envelope through as data.
        envelope = _json.dumps(
            {"type": "result", "subtype": "error_during_execution", "is_error": True,
             "result": "the model failed"}
        )
        with patch("arxiv_assistant.utils.agent_runner.subprocess.run",
                   return_value=_FakeProc(returncode=0, stdout=envelope)):
            with self.assertRaises(AgentError):
                run_agent("prompt", schema=_ANY_SCHEMA, model="claude-opus-4-8")

    def test_empty_stdout_raises_agent_error(self) -> None:
        with patch("arxiv_assistant.utils.agent_runner.subprocess.run",
                   return_value=_FakeProc(stdout="")):
            with self.assertRaises(AgentError):
                run_agent("prompt", schema=_ANY_SCHEMA, model="claude-opus-4-8")

    def test_whitespace_only_stdout_raises_agent_error(self) -> None:
        with patch("arxiv_assistant.utils.agent_runner.subprocess.run",
                   return_value=_FakeProc(stdout="   \n  ")):
            with self.assertRaises(AgentError):
                run_agent("prompt", schema=_ANY_SCHEMA, model="claude-opus-4-8")

    def test_envelope_result_is_non_json_string_raises_agent_error(self) -> None:
        """Envelope parses as JSON but result field contains non-JSON text."""
        envelope = _json.dumps({"type": "result", "result": "This is plain prose."})
        with patch("arxiv_assistant.utils.agent_runner.subprocess.run",
                   return_value=_FakeProc(stdout=envelope)):
            with self.assertRaises(AgentError):
                run_agent("prompt", schema=_ANY_SCHEMA, model="claude-opus-4-8")

    def test_envelope_result_is_json_array_raises_agent_error(self) -> None:
        """Agent returns a JSON array instead of an object — not a dict, must fail."""
        envelope = _json.dumps({"type": "result", "result": _json.dumps([1, 2, 3])})
        with patch("arxiv_assistant.utils.agent_runner.subprocess.run",
                   return_value=_FakeProc(stdout=envelope)):
            with self.assertRaises(AgentError):
                run_agent("prompt", schema=_ANY_SCHEMA, model="claude-opus-4-8")

    def test_envelope_missing_result_key_falls_back_to_envelope_itself(self) -> None:
        """If the envelope has no 'result' key, the envelope itself is tried as inner dict."""
        # If envelope IS already the structured dict (no wrapping), it should succeed.
        inner = {"key": "value"}
        envelope_str = _json.dumps(inner)
        with patch("arxiv_assistant.utils.agent_runner.subprocess.run",
                   return_value=_FakeProc(stdout=envelope_str)):
            out = run_agent("prompt", schema=_ANY_SCHEMA, model="claude-opus-4-8")
        self.assertEqual(out, inner)


# ---------------------------------------------------------------------------
# AgentError: schema validation failure
# ---------------------------------------------------------------------------


class TestRunAgentSchemaValidation(unittest.TestCase):
    def test_missing_required_key_raises_agent_error(self) -> None:
        """Inner dict missing a required schema key must raise AgentError."""
        # dateverify_tier1_clean fixture is missing 'verified_first_date' after we
        # demand an extra required key 'nonexistent_key'.
        strict_schema = {
            "required": ["verified_first_date", "nonexistent_key"],
        }
        envelope_stdout = _wrap_fixture("dateverify_tier1_clean.json")
        with patch("arxiv_assistant.utils.agent_runner.subprocess.run",
                   return_value=_FakeProc(stdout=envelope_stdout)):
            with self.assertRaises(AgentError) as ctx:
                run_agent("prompt", schema=strict_schema, model="claude-opus-4-8")
        self.assertIn("nonexistent_key", str(ctx.exception))

    def test_wrong_type_raises_agent_error(self) -> None:
        """Inner dict with wrong type for a declared property raises AgentError."""
        # confidence should be 'number'; declare it as 'string' to force a mismatch.
        wrong_type_schema = {
            "properties": {
                "confidence": {"type": "string"},
            }
        }
        envelope_stdout = _wrap_fixture("dateverify_tier1_clean.json")
        with patch("arxiv_assistant.utils.agent_runner.subprocess.run",
                   return_value=_FakeProc(stdout=envelope_stdout)):
            with self.assertRaises(AgentError) as ctx:
                run_agent("prompt", schema=wrong_type_schema, model="claude-opus-4-8")
        self.assertIn("confidence", str(ctx.exception))

    def test_valid_schema_does_not_raise(self) -> None:
        """Full valid schema against clean fixture must succeed without raising."""
        envelope_stdout = _wrap_fixture("dateverify_tier1_clean.json")
        with patch("arxiv_assistant.utils.agent_runner.subprocess.run",
                   return_value=_FakeProc(stdout=envelope_stdout)):
            out = run_agent("prompt", schema=_DATEVERIFY_SCHEMA, model="claude-opus-4-8")
        self.assertIn("verified_first_date", out)

    def test_boolean_is_not_accepted_as_number(self) -> None:
        """Python booleans are subclass of int; schema type 'number' must reject bool."""
        bool_envelope = _json.dumps({
            "type": "result",
            "result": _json.dumps({"key": True}),
        })
        schema_wants_number = {"properties": {"key": {"type": "number"}}}
        with patch("arxiv_assistant.utils.agent_runner.subprocess.run",
                   return_value=_FakeProc(stdout=bool_envelope)):
            with self.assertRaises(AgentError):
                run_agent("prompt", schema=schema_wants_number, model="claude-opus-4-8")

    def test_empty_schema_accepts_any_dict(self) -> None:
        """An empty schema ({}) must accept any well-formed dict without raising."""
        envelope_stdout = _wrap_fixture("dateverify_tier1_stale_pollution.json")
        with patch("arxiv_assistant.utils.agent_runner.subprocess.run",
                   return_value=_FakeProc(stdout=envelope_stdout)):
            out = run_agent("prompt", schema={}, model="claude-opus-4-8")
        self.assertIsInstance(out, dict)


# ---------------------------------------------------------------------------
# Signature contract test (§2.11 binding)
# ---------------------------------------------------------------------------


class TestRunAgentSignature(unittest.TestCase):
    def test_signature_matches_contract(self) -> None:
        """run_agent must have exact signature: (prompt, *, schema, model, tools=None, timeout_s=120)."""
        import inspect
        sig = inspect.signature(run_agent)
        params = sig.parameters
        self.assertEqual(list(params.keys()), ["prompt", "schema", "model", "tools", "timeout_s"])
        # All after 'prompt' must be keyword-only.
        for name in ["schema", "model", "tools", "timeout_s"]:
            self.assertEqual(
                params[name].kind,
                inspect.Parameter.KEYWORD_ONLY,
                msg=f"Parameter {name!r} must be keyword-only.",
            )
        # Default values.
        self.assertIsNone(params["tools"].default)
        self.assertEqual(params["timeout_s"].default, 120)

    def test_agent_error_is_exception(self) -> None:
        """AgentError must subclass Exception."""
        self.assertTrue(issubclass(AgentError, Exception))


if __name__ == "__main__":
    unittest.main()
