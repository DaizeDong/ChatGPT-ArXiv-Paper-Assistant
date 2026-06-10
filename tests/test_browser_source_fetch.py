from __future__ import annotations

import unittest
from datetime import UTC, datetime

from arxiv_assistant.apis.hotspot.browser_source_fetch import (
    PLAYWRIGHT_TOOLS,
    fetch_source_via_browser,
)
from arxiv_assistant.utils.agent_runner import AgentError

NOW = datetime(2026, 6, 9, tzinfo=UTC)
INDEX = "https://www.reddit.com/r/LocalLLaMA/"


class _RecordingAgent:
    """Fake agent_fn that records the kwargs it was called with and returns a fixed payload."""

    def __init__(self, payload):
        self._payload = payload
        self.last = None

    def __call__(self, prompt, **kwargs):
        self.last = {"prompt": prompt, **kwargs}
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _always_alive(_url: str) -> bool:
    return True


def _always_dead(_url: str) -> bool:
    return False


class TestBrowserSourceFetch(unittest.TestCase):
    def test_valid_permalinks_mapped(self) -> None:
        agent = _RecordingAgent({
            "items": [
                {"title": "Llama 4 fine-tuning thread", "url": "https://www.reddit.com/r/LocalLLaMA/comments/aaa/llama4/", "summary": "discussion", "published_at": "2026-06-09T01:00:00Z"},
                {"title": "New 1B model dropped", "url": "https://www.reddit.com/r/LocalLLaMA/comments/bbb/new1b/", "summary": "release"},
            ]
        })
        items = fetch_source_via_browser(
            "reddit_localllama", INDEX, "social", NOW, 48,
            agent_fn=agent, url_check_fn=_always_alive,
        )
        self.assertEqual(len(items), 2)
        it = items[0]
        self.assertEqual(it.source_id, "browser_src:reddit_localllama")
        self.assertEqual(it.source_name, "reddit_localllama")
        self.assertEqual(it.source_role, "browser_source")
        self.assertEqual(it.metadata["provenance"], "browser:source:reddit_localllama")
        self.assertEqual(it.metadata["source_id"], "browser_src:reddit_localllama")
        self.assertEqual(it.metadata["fetch_route"], "browser")
        self.assertEqual(it.metadata["origin_url"], INDEX)
        self.assertEqual(it.provenance, "browser:source:reddit_localllama")
        self.assertEqual(it.published_at, "2026-06-09T01:00:00Z")

    def test_playwright_tools_and_browser_prompt(self) -> None:
        agent = _RecordingAgent({"items": []})
        fetch_source_via_browser("reddit_localllama", INDEX, "social", NOW, 48, agent_fn=agent, url_check_fn=_always_alive)
        # tools passed to the transport are the playwright browser tools.
        self.assertEqual(agent.last["tools"], PLAYWRIGHT_TOOLS)
        self.assertIn("mcp__plugin_playwright_playwright__browser_navigate", agent.last["tools"])
        self.assertIn("mcp__plugin_playwright_playwright__browser_snapshot", agent.last["tools"])
        prompt = agent.last["prompt"].lower()
        self.assertIn("browser", prompt)
        self.assertIn("navigate", prompt)
        self.assertIn(INDEX, agent.last["prompt"])
        # JSON-only output contract present.
        self.assertIn("only", prompt)
        self.assertTrue("no markdown" in prompt or "raw json" in prompt)
        # schema reached the transport.
        self.assertEqual(agent.last["schema"]["required"], ["items"])

    def test_index_url_dropped(self) -> None:
        agent = _RecordingAgent({
            "items": [
                {"title": "the index itself", "url": INDEX, "summary": "not an item"},
                {"title": "real permalink", "url": "https://www.reddit.com/r/LocalLLaMA/comments/ccc/x/", "summary": "ok"},
            ]
        })
        items = fetch_source_via_browser("reddit_localllama", INDEX, "social", NOW, 48, agent_fn=agent, url_check_fn=_always_alive)
        self.assertEqual(len(items), 1)
        self.assertNotEqual(items[0].url, INDEX)

    def test_dead_and_garbage_urls_dropped(self) -> None:
        agent = _RecordingAgent({
            "items": [
                {"title": "garbage url", "url": "not a url", "summary": "x"},
                {"title": "blocklisted", "url": "https://www.google.com/search?q=ai", "summary": "x"},
                {"title": "live one", "url": "https://www.reddit.com/r/LocalLLaMA/comments/ddd/y/", "summary": "ok"},
            ]
        })
        # url_check passes everything; the scheme/host gate + blocklist still drop the bad ones.
        items = fetch_source_via_browser("reddit_localllama", INDEX, "social", NOW, 48, agent_fn=agent, url_check_fn=_always_alive)
        self.assertEqual(len(items), 1)
        # And a live-but-dead-per-check url is dropped by liveness.
        items2 = fetch_source_via_browser(
            "reddit_localllama", INDEX, "social", NOW, 48,
            agent_fn=_RecordingAgent({"items": [{"title": "t", "url": "https://www.reddit.com/r/LocalLLaMA/comments/eee/z/", "summary": "s"}]}),
            url_check_fn=_always_dead,
        )
        self.assertEqual(items2, [])

    def test_agent_error_degrades(self) -> None:
        agent = _RecordingAgent(AgentError("playwright unavailable"))
        items = fetch_source_via_browser("reddit_localllama", INDEX, "social", NOW, 48, agent_fn=agent, url_check_fn=_always_alive)
        self.assertEqual(items, [])

    def test_result_limit_and_dedupe(self) -> None:
        dup = "https://www.reddit.com/r/LocalLLaMA/comments/fff/dup/"
        agent = _RecordingAgent({
            "items": [
                {"title": "a", "url": "https://www.reddit.com/r/LocalLLaMA/comments/g1/a/", "summary": "s"},
                {"title": "b", "url": dup, "summary": "s"},
                {"title": "b again", "url": dup, "summary": "s"},
            ]
        })
        items = fetch_source_via_browser("reddit_localllama", INDEX, "social", NOW, 48, result_limit=1, agent_fn=agent, url_check_fn=_always_alive)
        self.assertEqual(len(items), 1)
        # dedupe: without the cap, the two duplicate urls collapse to one.
        items2 = fetch_source_via_browser("reddit_localllama", INDEX, "social", NOW, 48, result_limit=10, agent_fn=agent, url_check_fn=_always_alive)
        self.assertEqual(len(items2), 2)


if __name__ == "__main__":
    unittest.main()
