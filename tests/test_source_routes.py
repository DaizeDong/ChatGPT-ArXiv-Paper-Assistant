from __future__ import annotations

import unittest

from arxiv_assistant.utils.hotspot.source_routes import (
    SUBAGENT_ROUTES,
    iter_subagent_sources,
    route_for,
)


class TestSourceRoutes(unittest.TestCase):
    def test_route_for_known(self) -> None:
        self.assertEqual(route_for("reddit_localllama"), "browser")
        self.assertEqual(route_for("reddit_machinelearning"), "browser")
        self.assertEqual(route_for("jiqizhixin"), "browser")
        self.assertEqual(route_for("zhipu_blog"), "agent")
        self.assertEqual(route_for("bytedance_seed_blog"), "agent")

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
