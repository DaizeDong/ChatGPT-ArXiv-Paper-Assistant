from __future__ import annotations

import unittest
from datetime import datetime, timezone

from arxiv_assistant.utils.hotspot import source_escalation as se


def _item(n):
    # lightweight stand-in for a HotspotItem (the framework only counts / passes them through)
    return {"id": n}


_TD = datetime(2026, 6, 9, tzinfo=timezone.utc)


class TestDetectProtection(unittest.TestCase):
    def test_403_exception_is_protected(self):
        self.assertEqual(se.detect_protection(item_count=0, exception=Exception("403 Client Error: Forbidden")), se.PROTECTED)

    def test_429_exception_is_protected(self):
        self.assertEqual(se.detect_protection(item_count=0, exception=Exception("429 Too Many Requests")), se.PROTECTED)

    def test_blocked_keyword_is_protected(self):
        self.assertEqual(se.detect_protection(item_count=0, exception=Exception("Blocked by Cloudflare captcha")), se.PROTECTED)

    def test_timeout_exception_is_protected(self):
        self.assertEqual(se.detect_protection(item_count=0, exception=TimeoutError("read timeout")), se.PROTECTED)

    def test_401_exception_is_needs_agent(self):
        self.assertEqual(se.detect_protection(item_count=0, exception=Exception("401 Unauthorized: LLM extraction needs api key")), se.NEEDS_AGENT)

    def test_http_status_403_is_protected(self):
        self.assertEqual(se.detect_protection(item_count=0, http_status=403), se.PROTECTED)

    def test_http_status_401_is_needs_agent(self):
        self.assertEqual(se.detect_protection(item_count=0, http_status=401), se.NEEDS_AGENT)

    def test_items_present_is_ok(self):
        self.assertEqual(se.detect_protection(item_count=3), se.OK)

    def test_zero_items_small_body_is_js_walled(self):
        self.assertEqual(se.detect_protection(item_count=0, body_len=200), se.JS_WALLED)

    def test_zero_items_large_body_is_empty(self):
        self.assertEqual(se.detect_protection(item_count=0, body_len=50000), se.EMPTY)

    def test_other_exception_is_error(self):
        self.assertEqual(se.detect_protection(item_count=0, exception=ValueError("weird parse failure")), se.ERROR)


class TestEscalationRoute(unittest.TestCase):
    def test_ok_no_escalation(self):
        self.assertIsNone(se.escalation_route(se.OK))

    def test_protected_to_browser(self):
        self.assertEqual(se.escalation_route(se.PROTECTED), se.ROUTE_BROWSER)

    def test_js_walled_to_browser(self):
        self.assertEqual(se.escalation_route(se.JS_WALLED), se.ROUTE_BROWSER)

    def test_needs_agent_to_agent(self):
        self.assertEqual(se.escalation_route(se.NEEDS_AGENT), se.ROUTE_AGENT)

    def test_empty_to_agent(self):
        self.assertEqual(se.escalation_route(se.EMPTY), se.ROUTE_AGENT)

    def test_error_to_agent(self):
        self.assertEqual(se.escalation_route(se.ERROR), se.ROUTE_AGENT)


class TestFetchSourceResilient(unittest.TestCase):
    def _call(self, **kw):
        return se.fetch_source_resilient("src", "https://x.example/feed", "news", _TD, 168, **kw)

    def test_direct_ok_no_escalation(self):
        agent_calls, browser_calls = [], []

        def direct():
            return [_item(1), _item(2)]

        def agent(*a, **k):
            agent_calls.append(1); return [_item(9)]

        def browser(*a, **k):
            browser_calls.append(1); return [_item(9)]

        res = self._call(direct_fn=direct, agent_fn=agent, browser_fn=browser)
        self.assertEqual(res["route_used"], se.ROUTE_DIRECT)
        self.assertEqual(res["status"], se.OK)
        self.assertEqual(len(res["items"]), 2)
        self.assertEqual(agent_calls, [])
        self.assertEqual(browser_calls, [])

    def test_direct_403_escalates_to_browser(self):
        def direct():
            raise Exception("403 Forbidden")

        def browser(*a, **k):
            return [_item(7)]

        res = self._call(direct_fn=direct, agent_fn=lambda *a, **k: [_item(8)], browser_fn=browser)
        self.assertEqual(res["route_used"], se.ROUTE_BROWSER)
        self.assertEqual(res["status"], se.PROTECTED)
        self.assertEqual(len(res["items"]), 1)

    def test_direct_empty_escalates_to_agent(self):
        def direct():
            return []  # empty, no exception -> EMPTY -> agent

        agent_called = []

        def agent(*a, **k):
            agent_called.append(1); return [_item(5)]

        res = self._call(direct_fn=direct, agent_fn=agent, browser_fn=lambda *a, **k: [_item(0)])
        self.assertEqual(res["route_used"], se.ROUTE_AGENT)
        self.assertEqual(agent_called, [1])
        self.assertEqual(len(res["items"]), 1)

    def test_agent_empty_then_browser(self):
        # EMPTY -> agent returns [] -> escalate once more to browser
        res = self._call(direct_fn=lambda: [], agent_fn=lambda *a, **k: [], browser_fn=lambda *a, **k: [_item(3)])
        self.assertEqual(res["route_used"], se.ROUTE_BROWSER)
        self.assertEqual(len(res["items"]), 1)

    def test_all_fail_returns_empty_no_raise(self):
        res = self._call(direct_fn=lambda: [], agent_fn=lambda *a, **k: [], browser_fn=lambda *a, **k: [])
        self.assertEqual(res["items"], [])
        self.assertEqual(res["route_used"], "none")
        # attempts recorded each tier tried
        self.assertGreaterEqual(len(res["attempts"]), 2)

    def test_prefer_browser_skips_direct(self):
        direct_called = []

        def direct():
            direct_called.append(1); return [_item(1)]

        res = self._call(prefer="browser", direct_fn=direct, browser_fn=lambda *a, **k: [_item(2)])
        self.assertEqual(direct_called, [])
        self.assertEqual(res["route_used"], se.ROUTE_BROWSER)

    def test_tier_raises_is_caught_chain_continues(self):
        def direct():
            return []  # EMPTY -> agent

        def agent(*a, **k):
            raise RuntimeError("agent blew up")  # must be caught

        res = self._call(direct_fn=direct, agent_fn=agent, browser_fn=lambda *a, **k: [_item(4)])
        # agent raised -> recorded -> escalate to browser
        self.assertEqual(res["route_used"], se.ROUTE_BROWSER)
        self.assertTrue(any("error" in a for a in res["attempts"]))

    def test_browser_missing_falls_back_to_agent(self):
        # PROTECTED -> browser, but no browser_fn -> weaker agent fallback
        res = self._call(direct_fn=lambda: (_ for _ in ()).throw(Exception("403")), agent_fn=lambda *a, **k: [_item(6)], browser_fn=None)
        self.assertEqual(res["route_used"], se.ROUTE_AGENT)
        self.assertEqual(len(res["items"]), 1)


if __name__ == "__main__":
    unittest.main()
