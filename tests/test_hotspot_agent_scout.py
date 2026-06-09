from __future__ import annotations

import unittest
from datetime import UTC, datetime
from unittest.mock import patch

from arxiv_assistant.apis.hotspot import hotspot_agent_scout as scout
from arxiv_assistant.utils.agent_runner import AgentError


def _always_alive(url: str) -> bool:
    return True


def _always_dead(url: str) -> bool:
    return False


class TestHotspotAgentScout(unittest.TestCase):
    def setUp(self) -> None:
        self.target_date = datetime(2026, 6, 8, tzinfo=UTC)
        self.freshness_hours = 24

    def _run(self, payload, *, url_check_fn, result_limit=40):
        def fake_agent(prompt, *, schema, model, tools, timeout_s):
            # Record the call args so other tests can assert on them.
            self._last_call = {
                "prompt": prompt,
                "schema": schema,
                "model": model,
                "tools": tools,
                "timeout_s": timeout_s,
            }
            return payload

        return scout.fetch_hotspot_items(
            self.target_date,
            self.freshness_hours,
            result_limit=result_limit,
            use_market_intel=False,  # hermetic: built-in venues, no on-disk skill dependency
            agent_fn=fake_agent,
            url_check_fn=url_check_fn,
        )

    # ---------------------------------------------------------------------
    # Happy path: 2 live URLs -> 2 items with correct fields + provenance.
    # ---------------------------------------------------------------------
    def test_valid_items_mapped(self) -> None:
        payload = {
            "items": [
                {
                    "title": "New diffusion model released",
                    "url": "https://openai.com/blog/new-model",
                    "summary": "A faster image model.",
                    "source_kind": "release",
                },
                {
                    "title": "Notable paper on transformers",
                    "url": "https://arxiv.org/abs/2406.01234",
                    "summary": "Improves attention efficiency.",
                    "source_kind": "paper",
                    "published_at": "2026-06-07T12:00:00Z",
                },
            ]
        }
        items = self._run(payload, url_check_fn=_always_alive)
        self.assertEqual(len(items), 2)
        for item in items:
            self.assertEqual(item.source_id, "agent_scout")
            self.assertEqual(item.source_name, "Agent Scout")
            self.assertEqual(item.source_role, "agent_discovery")
            self.assertEqual(item.metadata["provenance"], "agent:scout")
            self.assertEqual(item.metadata["source_id"], "agent_scout")
            self.assertEqual(item.provenance, "agent:scout")
        self.assertEqual(items[0].title, "New diffusion model released")
        self.assertEqual(items[0].url, "https://openai.com/blog/new-model")
        self.assertEqual(items[1].canonical_url, "https://arxiv.org/abs/2406.01234")
        self.assertEqual(items[1].published_at, "2026-06-07T12:00:00Z")
        self.assertIn("agent-scout", items[1].tags)
        self.assertIn("paper", items[1].tags)
        # Confirm the transport contract: web tools were requested.
        self.assertEqual(self._last_call["tools"], ["WebSearch", "WebFetch"])
        self.assertEqual(self._last_call["schema"]["required"], ["items"])
        # Loose regression guard: the prompt should carry the market-intel-style
        # research discipline (name a couple of the key primary venues + source
        # tiering). Case-insensitive substring checks only -- do NOT over-couple
        # to exact wording.
        prompt_lc = self._last_call["prompt"].lower()
        self.assertIn("arxiv", prompt_lc)
        self.assertIn("hugging face", prompt_lc)
        self.assertIn("lab blog", prompt_lc)

    # ---------------------------------------------------------------------
    # Anti-hallucination: url_check_fn returns False -> item dropped.
    # ---------------------------------------------------------------------
    def test_dead_url_dropped(self) -> None:
        payload = {
            "items": [
                {"title": "Real one", "url": "https://blog.real.com/post", "source_kind": "news"},
                {"title": "Fabricated", "url": "https://blog.fake.com/hallucinated", "source_kind": "news"},
            ]
        }

        def half_alive(url: str) -> bool:
            return "real" in url

        items = self._run(payload, url_check_fn=half_alive)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].title, "Real one")

    # ---------------------------------------------------------------------
    # Garbage / non-http url -> dropped BEFORE url_check_fn (scheme gate first).
    # ---------------------------------------------------------------------
    def test_non_http_url_dropped(self) -> None:
        for bad in ("not a url", "ftp://x/y", ""):
            payload = {"items": [{"title": "Bad", "url": bad, "source_kind": "news"}]}
            # url_check_fn would say True for everything, but the scheme/host gate
            # must reject these first.
            items = self._run(payload, url_check_fn=_always_alive)
            self.assertEqual(items, [], f"expected drop for url={bad!r}")

    # ---------------------------------------------------------------------
    # Blocklisted host (search-results link) -> dropped.
    # ---------------------------------------------------------------------
    def test_blocklisted_host_dropped(self) -> None:
        payload = {
            "items": [
                {"title": "Search link", "url": "https://www.google.com/search?q=ai", "source_kind": "news"},
            ]
        }
        items = self._run(payload, url_check_fn=_always_alive)
        self.assertEqual(items, [])

    # ---------------------------------------------------------------------
    # AgentError -> [] (degrade, not crash).
    # ---------------------------------------------------------------------
    def test_agent_error_degrades(self) -> None:
        def raising_agent(prompt, *, schema, model, tools, timeout_s):
            raise AgentError("boom")

        items = scout.fetch_hotspot_items(
            self.target_date,
            self.freshness_hours,
            agent_fn=raising_agent,
            url_check_fn=_always_alive,
        )
        self.assertEqual(items, [])

    # ---------------------------------------------------------------------
    # result_limit honored + dedupe by canonical_url.
    # ---------------------------------------------------------------------
    def test_result_limit_and_dedupe(self) -> None:
        payload = {
            "items": [
                {"title": "A", "url": "https://a.example2.com/1", "source_kind": "news"},
                {"title": "B", "url": "https://b.example3.com/2", "source_kind": "news"},
            ]
        }
        items = self._run(payload, url_check_fn=_always_alive, result_limit=1)
        self.assertEqual(len(items), 1)

        dup_payload = {
            "items": [
                {"title": "First", "url": "https://dup.host.com/x", "source_kind": "news"},
                {"title": "Second (same url)", "url": "https://dup.host.com/x", "source_kind": "news"},
            ]
        }
        items = self._run(dup_payload, url_check_fn=_always_alive, result_limit=40)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].title, "First")

    # ---------------------------------------------------------------------
    # Malformed payloads / per-item failures degrade gracefully.
    # ---------------------------------------------------------------------
    def test_malformed_payloads(self) -> None:
        self.assertEqual(self._run({}, url_check_fn=_always_alive), [])
        self.assertEqual(self._run({"items": "nope"}, url_check_fn=_always_alive), [])
        self.assertEqual(self._run("not a dict", url_check_fn=_always_alive), [])
        # A non-dict item in the list is skipped, the good one survives.
        payload = {
            "items": [
                "garbage",
                {"title": "Good", "url": "https://ok.host.com/p", "source_kind": "news"},
            ]
        }
        items = self._run(payload, url_check_fn=_always_alive)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].title, "Good")

    # ---------------------------------------------------------------------
    # _default_url_alive: arXiv ids resolve WITHOUT a network call.
    # ---------------------------------------------------------------------
    def test_default_url_alive_arxiv_no_network(self) -> None:
        with patch("arxiv_assistant.apis.hotspot.hotspot_agent_scout.requests.head") as mock_head:
            self.assertTrue(scout._default_url_alive("https://arxiv.org/abs/2406.01234"))
            self.assertTrue(scout._default_url_alive("https://doi.org/10.1234/abcd"))
            mock_head.assert_not_called()

    def test_default_url_alive_uses_head_for_other_hosts(self) -> None:
        class _Resp:
            status_code = 200

        with patch(
            "arxiv_assistant.apis.hotspot.hotspot_agent_scout.requests.head",
            return_value=_Resp(),
        ) as mock_head:
            self.assertTrue(scout._default_url_alive("https://blog.host.com/post"))
            mock_head.assert_called_once()

    def test_default_url_alive_exception_is_dead(self) -> None:
        with patch(
            "arxiv_assistant.apis.hotspot.hotspot_agent_scout.requests.head",
            side_effect=RuntimeError("network down"),
        ):
            self.assertFalse(scout._default_url_alive("https://blog.host.com/post"))


class TestScoutMarketIntelReuse(unittest.TestCase):
    """The scout reuses the market-intel curated source matrix when available,
    and falls back to the built-in venue list otherwise."""

    def setUp(self) -> None:
        self.target_date = datetime(2026, 6, 8, tzinfo=UTC)

    def test_build_prompt_injects_curated_matrix(self) -> None:
        prompt = scout._build_prompt(
            self.target_date, 24, 40, source_guidance="CURATED MATRIX XYZ"
        )
        self.assertIn("CURATED MATRIX XYZ", prompt)
        self.assertIn("market-intel skill", prompt)

    def test_build_prompt_falls_back_to_builtin_when_no_guidance(self) -> None:
        prompt = scout._build_prompt(self.target_date, 24, 40, source_guidance=None)
        self.assertIn("arXiv recent listings", prompt)  # built-in venue list
        self.assertNotIn("CURATED MATRIX", prompt)

    def test_fetch_uses_bridge_guidance_in_agent_prompt(self) -> None:
        seen = {}

        def fake_agent(prompt, *, schema, model, tools, timeout_s):
            seen["prompt"] = prompt
            return {"items": []}

        with patch.object(
            scout.market_intel_bridge, "load_source_guidance", return_value="REUSED-MATRIX-42"
        ) as mock_load:
            scout.fetch_hotspot_items(
                self.target_date, 24, use_market_intel=True,
                agent_fn=fake_agent, url_check_fn=_always_alive,
            )
        mock_load.assert_called_once()
        self.assertIn("REUSED-MATRIX-42", seen["prompt"])

    def test_fetch_skips_bridge_when_disabled(self) -> None:
        def fake_agent(prompt, *, schema, model, tools, timeout_s):
            return {"items": []}

        with patch.object(scout.market_intel_bridge, "load_source_guidance") as mock_load:
            scout.fetch_hotspot_items(
                self.target_date, 24, use_market_intel=False,
                agent_fn=fake_agent, url_check_fn=_always_alive,
            )
        mock_load.assert_not_called()

    def test_fetch_degrades_to_builtin_when_bridge_returns_none(self) -> None:
        seen = {}

        def fake_agent(prompt, *, schema, model, tools, timeout_s):
            seen["prompt"] = prompt
            return {"items": []}

        with patch.object(
            scout.market_intel_bridge, "load_source_guidance", return_value=None
        ):
            scout.fetch_hotspot_items(
                self.target_date, 24, use_market_intel=True,
                agent_fn=fake_agent, url_check_fn=_always_alive,
            )
        self.assertIn("arXiv recent listings", seen["prompt"])  # built-in fallback


if __name__ == "__main__":
    unittest.main()
