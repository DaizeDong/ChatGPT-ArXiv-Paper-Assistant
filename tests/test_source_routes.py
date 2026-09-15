from __future__ import annotations

import json
import unittest
from pathlib import Path

from arxiv_assistant.utils.hotspot.source_routes import (
    SUBAGENT_ROUTES,
    iter_subagent_sources,
    route_for,
)

_OFFICIAL_BLOGS = Path(__file__).resolve().parents[1] / "configs" / "hotspot" / "official_blogs.json"
# CN AI-lab SPA blogs now served by the zero-key browser subagent (and disabled in official_blogs).
_CN_BROWSER_BLOGS = ("zhipu_blog", "bytedance_seed_blog", "baichuan_blog", "01ai_blog", "stepfun_blog")


class TestSourceRoutes(unittest.TestCase):
    def test_route_for_known(self) -> None:
        self.assertEqual(route_for("reddit_localllama"), "browser")
        self.assertEqual(route_for("reddit_machinelearning"), "browser")
        self.assertEqual(route_for("jiqizhixin"), "browser")
        # CN-lab SPA blogs are routed to the browser subagent (zero-key render+extract).
        for name in _CN_BROWSER_BLOGS:
            self.assertEqual(route_for(name), "browser", name)

    def test_cn_spa_blogs_disabled_in_official_blogs(self) -> None:
        """The CN-SPA blogs routed to the browser subagent must be DISABLED in
        official_blogs.json so they are not double-fetched (and don't 401)."""
        blogs = {b["source_id"]: b for b in json.loads(_OFFICIAL_BLOGS.read_text(encoding="utf-8"))}
        for name in _CN_BROWSER_BLOGS:
            self.assertIn(name, blogs, name)
            self.assertIs(blogs[name].get("enabled"), False, "{0} must be enabled=false".format(name))
        # a mainstream blog stays enabled
        self.assertIs(blogs["openai_news"].get("enabled"), True)

    def test_route_for_unknown_is_none(self) -> None:
        self.assertIsNone(route_for("nope"))
        self.assertIsNone(route_for(""))

    def test_iter_entries_have_required_keys(self) -> None:
        entries = iter_subagent_sources()
        self.assertEqual(len(entries), len(SUBAGENT_ROUTES))
        for name, spec in entries:
            self.assertIsInstance(name, str)
            for key in ("url", "kind", "route", "reason"):
                self.assertIn(key, spec, "{0} missing {1}".format(name, key))
            self.assertIn(spec["route"], ("browser", "agent"))
            self.assertTrue(spec["url"].startswith("http"))


if __name__ == "__main__":
    unittest.main()
