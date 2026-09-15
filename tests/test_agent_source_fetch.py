from __future__ import annotations

import unittest
from datetime import UTC, datetime

from arxiv_assistant.apis.hotspot import agent_source_fetch as mod
from arxiv_assistant.utils.agent_runner import AgentError


def _always_alive(_url: str) -> bool:
    return True


def _always_dead(_url: str) -> bool:
    return False


class _CaptureAgent:
    """Fake agent_fn that records its call and returns a canned payload."""

    def __init__(self, payload):
        self._payload = payload
        self.last = None

    def __call__(self, prompt, *, schema, model, tools, timeout_s):
        self.last = {"prompt": prompt, "schema": schema, "model": model, "tools": tools, "timeout_s": timeout_s}
        return self._payload


_NOW = datetime(2026, 6, 9, tzinfo=UTC)
_INDEX = "https://openai.com/news/"


class TestAgentSourceFetch(unittest.TestCase):
    def _fetch(self, payload, **kw):
        agent = _CaptureAgent(payload)
        items = mod.fetch_source_via_agent(
            "openai_news", _INDEX, "blog", _NOW, 168,
            agent_fn=agent, url_check_fn=kw.pop("url_check_fn", _always_alive), **kw,
        )
        return items, agent

    def test_valid_permalinks_mapped(self):
        payload = {"items": [
            {"title": "GPT-X released", "url": "https://openai.com/news/gpt-x", "summary": "A new model.", "published_at": "2026-06-09"},
            {"title": "Sora 2", "url": "https://openai.com/news/sora-2", "summary": "Video model."},
        ]}
        items, agent = self._fetch(payload)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].source_id, "agent_src:openai_news")
        self.assertEqual(items[0].source_role, "agent_source")
        self.assertEqual(items[0].metadata["provenance"], "agent:source:openai_news")
        self.assertEqual(items[0].metadata["origin_url"], _INDEX)
        self.assertEqual(items[0].provenance, "agent:source:openai_news")
        # transport contract: WebFetch present + JSON-only contract + the source url in the prompt
        self.assertIn("WebFetch", agent.last["tools"])
        self.assertIn("https://openai.com/news/", agent.last["prompt"])
        self.assertIn("ONLY a single raw JSON object", agent.last["prompt"])
        self.assertEqual(agent.last["schema"]["required"], ["items"])

    def test_index_url_itself_is_dropped(self):
        # An item whose url is the index page (not a permalink) must be dropped.
        payload = {"items": [
            {"title": "Index", "url": "https://openai.com/news", "summary": "the listing page"},
            {"title": "Real post", "url": "https://openai.com/news/real-post", "summary": "a permalink"},
        ]}
        items, _ = self._fetch(payload)
        self.assertEqual(len(items), 1)
        self.assertTrue(items[0].url.endswith("/real-post"))

    def test_dead_and_garbage_and_blocklisted_urls_dropped(self):
        payload = {"items": [
            {"title": "Live", "url": "https://openai.com/news/live", "summary": "ok"},
            {"title": "Dead", "url": "https://openai.com/news/dead", "summary": "unreachable"},
            {"title": "Garbage", "url": "not a url", "summary": "no scheme"},
            {"title": "Search link", "url": "https://www.google.com/search?q=ai", "summary": "blocklisted"},
        ]}
        # url_check_fn marks only the /live permalink as alive
        items, _ = self._fetch(payload, url_check_fn=lambda u: u.endswith("/live"))
        self.assertEqual(len(items), 1)
        self.assertTrue(items[0].url.endswith("/live"))

    def test_agent_error_degrades_to_empty(self):
        def boom(*a, **k):
            raise AgentError("transport down")
        items = mod.fetch_source_via_agent("openai_news", _INDEX, "blog", _NOW, 168, agent_fn=boom, url_check_fn=_always_alive)
        self.assertEqual(items, [])

    def test_malformed_payloads_degrade(self):
        for payload in ({}, {"items": "nope"}, ["not", "a", "dict"], None):
            items, _ = self._fetch(payload)
            self.assertEqual(items, [])

    def test_result_limit_and_dedupe(self):
        payload = {"items": [
            {"title": "A", "url": "https://openai.com/news/a", "summary": "x"},
            {"title": "A dup", "url": "https://openai.com/news/a", "summary": "same canonical"},
            {"title": "B", "url": "https://openai.com/news/b", "summary": "y"},
        ]}
        items, _ = self._fetch(payload, result_limit=1)
        self.assertEqual(len(items), 1)
        # dedupe: without the limit, the duplicate canonical collapses
        items2, _ = self._fetch(payload, result_limit=10)
        self.assertEqual(len(items2), 2)

    def test_all_dead_returns_empty(self):
        payload = {"items": [{"title": "X", "url": "https://openai.com/news/x", "summary": "x"}]}
        items, _ = self._fetch(payload, url_check_fn=_always_dead)
        self.assertEqual(items, [])


if __name__ == "__main__":
    unittest.main()
