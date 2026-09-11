"""Tests for arxiv_assistant/utils/llm_gateway.py -- the ONE place this repo
talks to a model.

No network, no subprocess, no llmcall import: every test either injects
``llmcall_fn=``/``agent_fn=`` or patches the detection helpers. The backend
env var is cleared with ``patch.dict`` in every test that resolves a backend,
because a developer machine may legitimately have it exported.

What these tests exist to protect
---------------------------------
1. ``auto`` must NEVER resolve to ``openai``, even when a key is present. That
   rule is the reason a dead key cannot quietly become the default again, so it
   gets an explicit negative control (``test_auto_never_resolves_to_openai``).
2. The ledger must count exactly. Health detection was moved off OpenAI token
   counts onto ledger attempts, so a lost update under concurrency would
   understate an outage -- hence the ThreadPool test asserting an exact total.
3. A failed call must raise AND leave a trace. "nothing matched" and "the model
   never ran" have to stay different outputs; every failure path below asserts
   that ``succeeded`` did not move and an error string was recorded.
"""
from __future__ import annotations

import configparser
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from arxiv_assistant.utils import llm_gateway
from arxiv_assistant.utils.agent_runner import AgentError
from arxiv_assistant.utils.llm_gateway import (
    BACKEND_AGENT,
    BACKEND_ENV,
    BACKEND_LLMCALL,
    BACKEND_OPENAI,
    CallLedger,
    GatewayResult,
)


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class FakeResult:
    """Duck-typed stand-in for an llmcall Result (truthy or falsy)."""

    def __init__(self, *, text="", data=None, provider="", ok=True, error=""):
        self.text = text
        self.data = data
        self.provider = provider
        self.error = error
        self._ok = ok

    def __bool__(self):
        return self._ok


def make_config(**llm_keys) -> configparser.ConfigParser:
    """A ConfigParser with a ``[LLM]`` section (omit kwargs for no section)."""
    config = configparser.ConfigParser()
    if llm_keys:
        config["LLM"] = {k: str(v) for k, v in llm_keys.items()}
    return config


def _clear_backend_env():
    import os

    os.environ.pop(BACKEND_ENV, None)


class GatewayTestCase(unittest.TestCase):
    """Shared setUp: the module-level LEDGER is process-global state."""

    def setUp(self):
        llm_gateway.LEDGER.reset()
        self.addCleanup(llm_gateway.LEDGER.reset)


# ---------------------------------------------------------------------------
# resolve_backend
# ---------------------------------------------------------------------------


class TestResolveBackend(GatewayTestCase):
    def test_env_override_wins_over_config(self):
        config = make_config(backend=BACKEND_LLMCALL)
        with patch.dict("os.environ", {BACKEND_ENV: BACKEND_AGENT}):
            self.assertEqual(llm_gateway.resolve_backend(config), BACKEND_AGENT)

    def test_env_override_is_case_and_space_insensitive(self):
        with patch.dict("os.environ", {BACKEND_ENV: "  AGENT  "}):
            self.assertEqual(llm_gateway.resolve_backend(None), BACKEND_AGENT)

    def test_config_backend_used_when_env_unset(self):
        config = make_config(backend=BACKEND_AGENT)
        with patch.dict("os.environ", {}, clear=False):
            _clear_backend_env()
            # Detection would say llmcall; the config must still win.
            with patch.object(llm_gateway, "llmcall_available", return_value=True):
                self.assertEqual(llm_gateway.resolve_backend(config), BACKEND_AGENT)

    def test_detection_used_when_no_env_and_no_config(self):
        with patch.dict("os.environ", {}, clear=False):
            _clear_backend_env()
            with patch.object(llm_gateway, "llmcall_available", return_value=True):
                self.assertEqual(llm_gateway.resolve_backend(None), BACKEND_LLMCALL)
            with patch.object(llm_gateway, "llmcall_available", return_value=False):
                self.assertEqual(llm_gateway.resolve_backend(None), BACKEND_AGENT)

    def test_auto_falls_through_to_detection(self):
        config = make_config(backend="auto")
        with patch.dict("os.environ", {}, clear=False):
            _clear_backend_env()
            with patch.object(llm_gateway, "llmcall_available", return_value=True):
                self.assertEqual(llm_gateway.resolve_backend(config), BACKEND_LLMCALL)
            with patch.object(llm_gateway, "llmcall_available", return_value=False):
                self.assertEqual(llm_gateway.resolve_backend(config), BACKEND_AGENT)

    def test_blank_config_backend_falls_through_to_detection(self):
        config = make_config(backend="")
        with patch.dict("os.environ", {}, clear=False):
            _clear_backend_env()
            with patch.object(llm_gateway, "llmcall_available", return_value=False):
                self.assertEqual(llm_gateway.resolve_backend(config), BACKEND_AGENT)

    def test_missing_llm_section_is_tolerated(self):
        config = make_config()  # no [LLM] at all
        self.assertFalse(config.has_section("LLM"))
        with patch.dict("os.environ", {}, clear=False):
            _clear_backend_env()
            with patch.object(llm_gateway, "llmcall_available", return_value=False):
                self.assertEqual(llm_gateway.resolve_backend(config), BACKEND_AGENT)

    def test_unknown_env_value_raises_value_error(self):
        with patch.dict("os.environ", {BACKEND_ENV: "gpt9"}):
            with self.assertRaises(ValueError) as ctx:
                llm_gateway.resolve_backend(None)
        self.assertIn("gpt9", str(ctx.exception))

    def test_unknown_config_value_raises_value_error(self):
        config = make_config(backend="anthropic-direct")
        with patch.dict("os.environ", {}, clear=False):
            _clear_backend_env()
            with self.assertRaises(ValueError):
                llm_gateway.resolve_backend(config)

    def test_openai_is_still_reachable_when_explicitly_selected(self):
        """Legacy opt-in must keep working -- it is only never *automatic*."""
        with patch.dict("os.environ", {BACKEND_ENV: BACKEND_OPENAI}):
            self.assertEqual(llm_gateway.resolve_backend(None), BACKEND_OPENAI)

    # -- the negative control -------------------------------------------------

    def test_auto_never_resolves_to_openai(self):
        """NEGATIVE CONTROL: a present OpenAI key must not become the default.

        This is the rule that stopped a dead key from silently owning every
        call for three months. With llmcall absent AND a key available, ``auto``
        must still pick the repo-local agent transport.
        """
        with patch.dict("os.environ", {}, clear=False):
            _clear_backend_env()
            with patch.object(llm_gateway, "openai_available", return_value=True), \
                    patch.object(llm_gateway, "llmcall_available", return_value=False):
                # Detection sees openai as *available*...
                self.assertIn(BACKEND_OPENAI, llm_gateway.available_backends())
                # ...and still refuses to select it automatically.
                for config in (None, make_config(backend="auto"), make_config()):
                    resolved = llm_gateway.resolve_backend(config)
                    self.assertNotEqual(resolved, BACKEND_OPENAI)
                    self.assertEqual(resolved, BACKEND_AGENT)

            # Same with llmcall present: still not openai.
            with patch.object(llm_gateway, "openai_available", return_value=True), \
                    patch.object(llm_gateway, "llmcall_available", return_value=True):
                self.assertEqual(llm_gateway.resolve_backend(None), BACKEND_LLMCALL)


class TestAvailableBackends(GatewayTestCase):
    def test_agent_is_always_listed(self):
        with patch.object(llm_gateway, "llmcall_available", return_value=False), \
                patch.object(llm_gateway, "openai_available", return_value=False):
            self.assertEqual(llm_gateway.available_backends(), [BACKEND_AGENT])

    def test_all_three_when_all_present(self):
        with patch.object(llm_gateway, "llmcall_available", return_value=True), \
                patch.object(llm_gateway, "openai_available", return_value=True):
            self.assertEqual(
                llm_gateway.available_backends(),
                [BACKEND_LLMCALL, BACKEND_AGENT, BACKEND_OPENAI],
            )


# ---------------------------------------------------------------------------
# call() -- llmcall backend
# ---------------------------------------------------------------------------


class TestCallLlmcallBackend(GatewayTestCase):
    def test_truthy_result_returns_gateway_result_and_records_one_success(self):
        fake = FakeResult(text="yes", data={"verdict": "yes"}, provider="cc")
        result = llm_gateway.call(
            "is 17 prime?",
            backend=BACKEND_LLMCALL,
            llmcall_fn=lambda prompt, **kwargs: fake,
        )

        self.assertIsInstance(result, GatewayResult)
        self.assertTrue(result)
        self.assertEqual(result.text, "yes")
        self.assertEqual(result.data, {"verdict": "yes"})
        self.assertEqual(result.backend, BACKEND_LLMCALL)
        self.assertEqual(result.provider, "cc")

        ledger = llm_gateway.LEDGER.to_dict()
        self.assertEqual(ledger["attempted"], 1)
        self.assertEqual(ledger["succeeded"], 1)
        self.assertEqual(ledger["failed"], 0)
        self.assertEqual(ledger["by_backend"], {BACKEND_LLMCALL: 1})
        self.assertEqual(ledger["by_provider"], {"cc": 1})
        self.assertEqual(ledger["errors"], [])

    def test_falsy_result_raises_agent_error_and_records_an_error_not_a_success(self):
        fake = FakeResult(ok=False, error="all four transports refused")

        with self.assertRaises(AgentError) as ctx:
            llm_gateway.call(
                "prompt",
                backend=BACKEND_LLMCALL,
                llmcall_fn=lambda prompt, **kwargs: fake,
            )

        self.assertIn("all four transports refused", str(ctx.exception))
        ledger = llm_gateway.LEDGER.to_dict()
        self.assertEqual(ledger["attempted"], 1)
        self.assertEqual(ledger["succeeded"], 0)
        self.assertEqual(ledger["failed"], 1)  # the outage is visible afterwards
        self.assertEqual(ledger["by_backend"], {BACKEND_LLMCALL: 1})
        self.assertEqual(ledger["by_provider"], {})
        self.assertEqual(len(ledger["errors"]), 1)

    def test_falsy_result_without_error_attribute_still_records_something(self):
        class Bare:
            def __bool__(self):
                return False

        with self.assertRaises(AgentError):
            llm_gateway.call(
                "prompt", backend=BACKEND_LLMCALL, llmcall_fn=lambda p, **k: Bare()
            )
        self.assertEqual(len(llm_gateway.LEDGER.errors), 1)
        self.assertNotEqual(llm_gateway.LEDGER.errors[0].strip(), "")

    def test_missing_provider_is_recorded_as_unknown(self):
        fake = FakeResult(text="ok", provider="")
        result = llm_gateway.call(
            "p", backend=BACKEND_LLMCALL, llmcall_fn=lambda p, **k: fake
        )
        self.assertEqual(result.provider, "")
        self.assertEqual(llm_gateway.LEDGER.by_provider, {"unknown": 1})

    def test_judge_mode_and_timeout_and_effort_reach_the_backend(self):
        seen = {}

        def fake_call(prompt, **kwargs):
            seen["prompt"] = prompt
            seen.update(kwargs)
            return FakeResult(text="ok", provider="codex")

        llm_gateway.call(
            "the prompt",
            backend=BACKEND_LLMCALL,
            timeout_s=42.0,
            config=make_config(effort="high"),
            model="some-model",
            llmcall_fn=fake_call,
        )

        self.assertEqual(seen["prompt"], "the prompt")
        self.assertEqual(seen["mode"], "judge")
        self.assertEqual(seen["timeout"], 42.0)
        self.assertEqual(seen["effort"], "high")
        self.assertEqual(seen["model"], "some-model")

    def test_default_effort_is_max_when_config_is_silent(self):
        seen = {}

        def fake_call(prompt, **kwargs):
            seen.update(kwargs)
            return FakeResult(text="ok", provider="cc")

        llm_gateway.call("p", backend=BACKEND_LLMCALL, llmcall_fn=fake_call)
        self.assertEqual(seen["effort"], llm_gateway.DEFAULT_EFFORT)
        self.assertEqual(llm_gateway.DEFAULT_EFFORT, "max")


# ---------------------------------------------------------------------------
# call() -- agent backend
# ---------------------------------------------------------------------------


class TestCallAgentBackend(GatewayTestCase):
    def test_success_path(self):
        payload = {"text": "the answer", "verdict": "yes"}
        result = llm_gateway.call(
            "prompt",
            backend=BACKEND_AGENT,
            agent_fn=lambda prompt, **kwargs: payload,
        )

        self.assertEqual(result.text, "the answer")
        self.assertEqual(result.data, payload)
        self.assertEqual(result.backend, BACKEND_AGENT)
        self.assertEqual(result.provider, "claude")

        ledger = llm_gateway.LEDGER.to_dict()
        self.assertEqual(ledger["attempted"], 1)
        self.assertEqual(ledger["succeeded"], 1)
        self.assertEqual(ledger["by_backend"], {BACKEND_AGENT: 1})
        self.assertEqual(ledger["by_provider"], {BACKEND_AGENT: 1})

    def test_agent_error_propagates_with_attempt_recorded_but_no_success(self):
        def boom(prompt, **kwargs):
            raise AgentError("claude -p exited 1")

        with self.assertRaises(AgentError) as ctx:
            llm_gateway.call("prompt", backend=BACKEND_AGENT, agent_fn=boom)

        self.assertIn("claude -p exited 1", str(ctx.exception))
        ledger = llm_gateway.LEDGER.to_dict()
        self.assertEqual(ledger["attempted"], 1)
        self.assertEqual(ledger["succeeded"], 0)
        self.assertEqual(ledger["failed"], 1)
        self.assertEqual(ledger["by_backend"], {BACKEND_AGENT: 1})
        self.assertEqual(ledger["by_provider"], {})
        self.assertEqual(len(ledger["errors"]), 1)
        self.assertIn("claude -p exited 1", ledger["errors"][0])

    def test_model_and_timeout_and_tools_reach_the_runner(self):
        seen = {}

        def fake_agent(prompt, **kwargs):
            seen["prompt"] = prompt
            seen.update(kwargs)
            return {"text": "ok"}

        llm_gateway.call(
            "prompt",
            backend=BACKEND_AGENT,
            model="claude-sonnet-4-6",
            timeout_s=90.0,
            tools=["WebSearch"],
            agent_fn=fake_agent,
        )
        self.assertEqual(seen["model"], "claude-sonnet-4-6")
        self.assertEqual(seen["timeout_s"], 90)
        self.assertEqual(seen["tools"], ["WebSearch"])

    def test_non_dict_payload_yields_empty_text_but_still_counts_as_success(self):
        result = llm_gateway.call(
            "p", backend=BACKEND_AGENT, agent_fn=lambda p, **k: ["a", "b"]
        )
        self.assertEqual(result.text, "")
        self.assertEqual(result.data, ["a", "b"])
        self.assertEqual(llm_gateway.LEDGER.succeeded, 1)


class TestCallDispatch(GatewayTestCase):
    def test_unknown_explicit_backend_raises_agent_error(self):
        with self.assertRaises(AgentError):
            llm_gateway.call("p", backend="mystery")
        # The attempt is still on the ledger -- a bad backend is an outage too.
        self.assertEqual(llm_gateway.LEDGER.attempted, 1)
        self.assertEqual(llm_gateway.LEDGER.succeeded, 0)

    def test_resolved_backend_is_used_when_none_is_passed(self):
        with patch.dict("os.environ", {BACKEND_ENV: BACKEND_AGENT}):
            result = llm_gateway.call("p", agent_fn=lambda p, **k: {"text": "x"})
        self.assertEqual(result.backend, BACKEND_AGENT)
        self.assertEqual(llm_gateway.LEDGER.by_backend, {BACKEND_AGENT: 1})


# ---------------------------------------------------------------------------
# schema passthrough
# ---------------------------------------------------------------------------


class TestSchemaPassthrough(GatewayTestCase):
    SCHEMA = {
        "required": ["verdict", "why"],
        "properties": {"verdict": {"type": "string"}, "why": {"type": "string"}},
    }

    def test_schema_reaches_the_llmcall_backend(self):
        seen = {}

        def fake_call(prompt, **kwargs):
            seen.update(kwargs)
            return FakeResult(text="", data={"verdict": "yes", "why": "because"}, provider="cc")

        llm_gateway.call(
            "p", schema=self.SCHEMA, backend=BACKEND_LLMCALL, llmcall_fn=fake_call
        )
        self.assertIn("schema", seen)
        self.assertEqual(seen["schema"], self.SCHEMA)

    def test_no_schema_means_no_schema_kwarg_for_llmcall(self):
        seen = {}

        def fake_call(prompt, **kwargs):
            seen.update(kwargs)
            return FakeResult(text="ok", provider="cc")

        llm_gateway.call("p", backend=BACKEND_LLMCALL, llmcall_fn=fake_call)
        self.assertNotIn("schema", seen)

    def test_schema_reaches_the_agent_backend(self):
        seen = {}

        def fake_agent(prompt, **kwargs):
            seen.update(kwargs)
            return {"verdict": "yes", "why": "because"}

        llm_gateway.call(
            "p", schema=self.SCHEMA, backend=BACKEND_AGENT, agent_fn=fake_agent
        )
        self.assertEqual(seen["schema"], self.SCHEMA)

    def test_agent_gets_a_permissive_schema_when_caller_wants_raw_text(self):
        seen = {}

        def fake_agent(prompt, **kwargs):
            seen.update(kwargs)
            return {"text": "raw"}

        llm_gateway.call("p", backend=BACKEND_AGENT, agent_fn=fake_agent)
        self.assertEqual(seen["schema"], {"required": [], "properties": {}})


# ---------------------------------------------------------------------------
# CallLedger
# ---------------------------------------------------------------------------


class TestCallLedger(unittest.TestCase):
    """Uses fresh CallLedger instances -- never the module-level LEDGER."""

    def test_arithmetic(self):
        ledger = CallLedger()
        self.assertEqual((ledger.attempted, ledger.succeeded, ledger.failed), (0, 0, 0))

        ledger.record_attempt(BACKEND_LLMCALL)
        ledger.record_attempt(BACKEND_LLMCALL)
        ledger.record_attempt(BACKEND_AGENT)
        ledger.record_success("cc")
        ledger.record_success("cc")

        self.assertEqual(ledger.attempted, 3)
        self.assertEqual(ledger.succeeded, 2)
        self.assertEqual(ledger.failed, 1)
        self.assertEqual(ledger.by_backend, {BACKEND_LLMCALL: 2, BACKEND_AGENT: 1})
        self.assertEqual(ledger.by_provider, {"cc": 2})

    def test_failed_never_goes_negative(self):
        ledger = CallLedger()
        ledger.record_success("cc")  # success with no recorded attempt
        self.assertEqual(ledger.failed, 0)

    def test_blank_provider_bucketed_as_unknown(self):
        ledger = CallLedger()
        ledger.record_success("")
        self.assertEqual(ledger.by_provider, {"unknown": 1})

    def test_error_cap(self):
        ledger = CallLedger()
        for i in range(CallLedger.MAX_ERRORS + 17):
            ledger.record_error(f"error {i}")
        self.assertEqual(len(ledger.errors), CallLedger.MAX_ERRORS)
        # The cap keeps the FIRST errors, which are the ones nearest the cause.
        self.assertEqual(ledger.errors[0], "error 0")

    def test_long_error_is_truncated(self):
        ledger = CallLedger()
        ledger.record_error("x" * 5000)
        self.assertEqual(len(ledger.errors[0]), 300)

    def test_to_dict_is_a_snapshot_not_a_live_view(self):
        ledger = CallLedger()
        ledger.record_attempt(BACKEND_AGENT)
        snapshot = ledger.to_dict()
        ledger.record_attempt(BACKEND_AGENT)
        self.assertEqual(snapshot["attempted"], 1)
        self.assertEqual(snapshot["by_backend"], {BACKEND_AGENT: 1})

    def test_reset_clears_everything(self):
        # Exhaustive on purpose: comparing the WHOLE dict means a field added
        # later without a matching reset shows up here rather than silently
        # leaking one run's numbers into the next run's cost report.
        ledger = CallLedger()
        ledger.record_attempt(BACKEND_AGENT)
        ledger.record_success("claude")
        ledger.record_error("boom")
        ledger.record_trace("codexg: unavailable after 3.9s")
        ledger.record_seconds(12.5)
        ledger.reset()
        self.assertEqual(
            ledger.to_dict(),
            {
                "attempted": 0,
                "succeeded": 0,
                "failed": 0,
                "by_backend": {},
                "by_provider": {},
                "errors": [],
                "seconds": 0.0,
                "trace": [],
            },
        )

    def test_concurrent_record_attempt_loses_nothing(self):
        """An exact count matters: a lost update would UNDERSTATE an outage.

        pipeline_health reads ``attempted``; if 2000 calls were recorded as 1900
        a partial outage could read as a smaller one, or a total outage as a
        partial one.
        """
        ledger = CallLedger()
        workers, per_worker = 16, 250
        total = workers * per_worker

        def hammer():
            for _ in range(per_worker):
                ledger.record_attempt(BACKEND_LLMCALL)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in [pool.submit(hammer) for _ in range(workers)]:
                future.result()

        self.assertEqual(ledger.attempted, total)
        self.assertEqual(ledger.by_backend[BACKEND_LLMCALL], total)
        self.assertEqual(ledger.failed, total)

    def test_concurrent_mixed_records(self):
        ledger = CallLedger()
        workers, per_worker = 12, 200

        def hammer(index):
            for _ in range(per_worker):
                ledger.record_attempt(BACKEND_LLMCALL if index % 2 else BACKEND_AGENT)
                ledger.record_success("cc")
                ledger.record_error("e")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in [pool.submit(hammer, i) for i in range(workers)]:
                future.result()

        self.assertEqual(ledger.attempted, workers * per_worker)
        self.assertEqual(ledger.succeeded, workers * per_worker)
        self.assertEqual(sum(ledger.by_backend.values()), workers * per_worker)
        self.assertEqual(ledger.by_provider["cc"], workers * per_worker)
        self.assertEqual(len(ledger.errors), CallLedger.MAX_ERRORS)


class TestModuleLedgerIsShared(GatewayTestCase):
    def test_module_ledger_accumulates_across_calls(self):
        for _ in range(3):
            llm_gateway.call("p", backend=BACKEND_AGENT, agent_fn=lambda p, **k: {"text": "x"})
        self.assertEqual(llm_gateway.LEDGER.attempted, 3)
        self.assertEqual(llm_gateway.LEDGER.succeeded, 3)

    def test_reset_makes_tests_order_independent(self):
        llm_gateway.LEDGER.record_attempt(BACKEND_AGENT)
        llm_gateway.LEDGER.reset()
        self.assertEqual(llm_gateway.LEDGER.attempted, 0)


# ---------------------------------------------------------------------------
# describe_backend
# ---------------------------------------------------------------------------


class TestDescribeBackend(GatewayTestCase):
    def test_names_the_resolved_backend(self):
        with patch.dict("os.environ", {BACKEND_ENV: BACKEND_AGENT}):
            line = llm_gateway.describe_backend(None)
        self.assertIsInstance(line, str)
        self.assertTrue(line.startswith("LLM backend: "))
        self.assertIn(BACKEND_AGENT, line)

    def test_names_llmcall_when_llmcall_is_resolved(self):
        with patch.dict("os.environ", {BACKEND_ENV: BACKEND_LLMCALL}):
            line = llm_gateway.describe_backend(None)
        self.assertIn(BACKEND_LLMCALL, line)

    def test_invalid_backend_says_so_instead_of_raising(self):
        with patch.dict("os.environ", {BACKEND_ENV: "nope"}):
            line = llm_gateway.describe_backend(None)
        self.assertIn("INVALID", line)


# ---------------------------------------------------------------------------
# GatewayResult
# ---------------------------------------------------------------------------


class TestGatewayResult(unittest.TestCase):
    def test_truthiness(self):
        self.assertFalse(GatewayResult(text="", data=None))
        self.assertTrue(GatewayResult(text="hi"))
        self.assertTrue(GatewayResult(text="", data={"a": 1}))
        self.assertTrue(GatewayResult(text="", data=[]))  # empty list is still a payload

    def test_is_frozen(self):
        result = GatewayResult(text="hi")
        with self.assertRaises(Exception):
            result.text = "changed"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
