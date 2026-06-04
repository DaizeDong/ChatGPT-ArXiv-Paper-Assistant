"""Tests for arxiv_assistant/apis/claude_agent.py — judge_paper_with_agent.

Three test cases (TDD, strict offline):

1. Valid run_agent result → adapter returns a JSON string that, when fed
   through AgentFilter (with this adapter as agent_fn), yields the expected
   keep=True verdict. Proves the adapter↔verifier handshake works end-to-end.

2. run_agent raises AgentError → adapter returns conservative fallback JSON
   string → AgentFilter.judge returns keep=False (degrade-not-crash).

3. The model placeholder "claude-code-subagent" is remapped to the real
   default ("claude-sonnet-4-6") before run_agent is called.

All tests @patch agent_runner.run_agent — zero real subprocesses.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from arxiv_assistant.apis.claude_agent import judge_paper_with_agent
from arxiv_assistant.filters.paper_filter import AgentFilter
from arxiv_assistant.utils.agent_runner import AgentError
from arxiv_assistant.utils.utils import Paper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _paper(arxiv_id: str = "2501.12345") -> Paper:
    return Paper(
        authors=["Alice Example"],
        title="A Novel Approach to Testing",
        abstract="We present a novel approach that achieves state-of-the-art results.",
        arxiv_id=arxiv_id,
    )


def _config(**overrides) -> dict:
    cfg: dict = {
        "FILTERING": {"h_cutoff": "10", "relevance_cutoff": "8", "novelty_cutoff": "8"},
        "PAPER_FILTER": {
            "mode": "agent_only",
            "agent_borderline_low": "6.0",
            "agent_borderline_high": "8.0",
        },
    }
    for section, kv in overrides.items():
        cfg.setdefault(section, {}).update(kv)
    return cfg


# A valid run_agent return dict (shape validated by run_agent's _validate_schema).
_VALID_AGENT_RESULT = {
    "keep": True,
    "relevance": 9,
    "novelty": 8,
    "rationale": "Highly relevant and novel contribution.",
    "evidence": ["https://arxiv.org/abs/2501.12345"],
}

# What run_agent returns for the fallback test (AgentError is raised instead,
# so this is never actually used — but kept for documentation clarity).
_CONSERVATIVE_FALLBACK_SHAPE = {
    "keep": False,
    "rationale": "agent transport failed: ...",
    "evidence": [],
}


# ---------------------------------------------------------------------------
# Test 1: valid run_agent result → AgentFilter.judge returns keep=True
# ---------------------------------------------------------------------------

class TestJudgePaperWithAgentHappyPath(unittest.TestCase):
    """End-to-end handshake: adapter → verifier → FilterVerdict."""

    def test_valid_result_flows_through_verifier_to_keep_true(self) -> None:
        """A valid run_agent dict → JSON string → verifier accepts → keep=True verdict."""
        paper = _paper("2501.12345")

        with patch(
            "arxiv_assistant.apis.claude_agent.run_agent",
            return_value=dict(_VALID_AGENT_RESULT),
        ):
            af = AgentFilter(config=_config(), agent_fn=judge_paper_with_agent)
            verdict = af.judge(paper, criteria="deep learning for computer vision")

        self.assertTrue(verdict.keep)
        self.assertEqual(verdict.relevance, 9.0)
        self.assertEqual(verdict.novelty, 8.0)
        self.assertIn("https://arxiv.org/abs/2501.12345", verdict.evidence)

    def test_adapter_returns_json_string(self) -> None:
        """judge_paper_with_agent must return a str (not a dict) for the verifier."""
        paper = _paper("2501.99999")
        agent_result = {
            "keep": False,
            "relevance": 3,
            "novelty": 2,
            "rationale": "Off-topic.",
            "evidence": [],
        }

        with patch(
            "arxiv_assistant.apis.claude_agent.run_agent",
            return_value=dict(agent_result),
        ):
            raw = judge_paper_with_agent(paper, "machine learning")

        self.assertIsInstance(raw, str)
        parsed = json.loads(raw)
        self.assertFalse(parsed["keep"])
        self.assertEqual(parsed["relevance"], 3)


# ---------------------------------------------------------------------------
# Test 2: AgentError → conservative fallback → AgentFilter yields keep=False
# ---------------------------------------------------------------------------

class TestJudgePaperWithAgentFallback(unittest.TestCase):
    """Degrade-not-crash: AgentError → keep=False, pipeline continues."""

    def test_agent_error_returns_fallback_json_string(self) -> None:
        """run_agent raising AgentError → adapter returns JSON string (not an exception)."""
        paper = _paper("2501.12345")

        with patch(
            "arxiv_assistant.apis.claude_agent.run_agent",
            side_effect=AgentError("subprocess timed out after 120s"),
        ):
            raw = judge_paper_with_agent(paper, "some criteria")

        self.assertIsInstance(raw, str)
        fallback = json.loads(raw)
        self.assertFalse(fallback["keep"])
        self.assertIn("agent transport failed", fallback["rationale"])
        self.assertIsInstance(fallback["evidence"], list)

    def test_agent_error_via_agent_filter_yields_keep_false(self) -> None:
        """Full AgentFilter path: transport failure → conservative keep=False verdict."""
        paper = _paper("2501.12345")

        with patch(
            "arxiv_assistant.apis.claude_agent.run_agent",
            side_effect=AgentError("OSError: claude not on PATH"),
        ):
            af = AgentFilter(config=_config(), agent_fn=judge_paper_with_agent)
            verdict = af.judge(paper, criteria="any topic")

        # The fallback JSON has no relevance/novelty keys → verifier rejects → safe keep=False.
        self.assertFalse(verdict.keep)
        self.assertEqual(verdict.evidence, [])

    def test_agent_error_does_not_propagate(self) -> None:
        """AgentError must NEVER escape judge_paper_with_agent — no exception at call site."""
        paper = _paper("2501.12345")

        with patch(
            "arxiv_assistant.apis.claude_agent.run_agent",
            side_effect=AgentError("fatal: model quota exceeded"),
        ):
            try:
                result = judge_paper_with_agent(paper, "criteria")
            except AgentError:
                self.fail("AgentError must be caught inside judge_paper_with_agent, not propagated.")
        self.assertIsInstance(result, str)


# ---------------------------------------------------------------------------
# Test 3: model placeholder resolution
# ---------------------------------------------------------------------------

class TestModelPlaceholderResolution(unittest.TestCase):
    """The "claude-code-subagent" placeholder is remapped to a real model id."""

    def test_placeholder_remapped_to_real_default(self) -> None:
        """model="claude-code-subagent" → run_agent is called with a real model id, not the placeholder."""
        paper = _paper("2501.12345")
        captured_kwargs: dict = {}

        def _fake_run_agent(prompt, *, schema, model, tools=None, timeout_s=120):
            captured_kwargs["model"] = model
            return dict(_VALID_AGENT_RESULT)

        with patch("arxiv_assistant.apis.claude_agent.run_agent", side_effect=_fake_run_agent):
            judge_paper_with_agent(paper, "criteria", model="claude-code-subagent")

        real_model = captured_kwargs["model"]
        self.assertNotEqual(
            real_model,
            "claude-code-subagent",
            "Placeholder must be resolved to a real model id before calling run_agent.",
        )
        # Must be a plausible Claude model string (contains "claude").
        self.assertIn("claude", real_model.lower())

    def test_real_model_id_passed_through_unchanged(self) -> None:
        """A real model id (not the placeholder) must reach run_agent verbatim."""
        paper = _paper("2501.12345")
        captured_kwargs: dict = {}

        def _fake_run_agent(prompt, *, schema, model, tools=None, timeout_s=120):
            captured_kwargs["model"] = model
            return dict(_VALID_AGENT_RESULT)

        with patch("arxiv_assistant.apis.claude_agent.run_agent", side_effect=_fake_run_agent):
            judge_paper_with_agent(paper, "criteria", model="claude-opus-4-8")

        self.assertEqual(captured_kwargs["model"], "claude-opus-4-8")

    def test_default_real_model_is_sonnet(self) -> None:
        """Without config override, the placeholder resolves to claude-sonnet-4-6."""
        from arxiv_assistant.apis.claude_agent import _resolve_model, _DEFAULT_REAL_MODEL
        self.assertEqual(_DEFAULT_REAL_MODEL, "claude-sonnet-4-6")

        # Patch the environment import to simulate missing config.
        with patch.dict("sys.modules", {"arxiv_assistant.environment": None}):
            resolved = _resolve_model("claude-code-subagent")
        self.assertEqual(resolved, "claude-sonnet-4-6")


if __name__ == "__main__":
    unittest.main()
