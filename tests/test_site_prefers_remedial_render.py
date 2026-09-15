from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from arxiv_assistant.paper_daily_io import JSON_SUFFIX_PRIORITY
from arxiv_assistant.renderers.build_multipage_site import (
    SUFFIX_PRIORITY,
    discover_daily_markdown,
)

#: What a failed original run leaves behind: a page that renders but says nothing.
EMPTY_PAGE = """# Personalized Daily ArXiv Papers 06/01/2026

Total relevant papers: 0

No papers selected.
"""

#: What the remedial re-run writes beside it.
REBUILT_PAGE = """# Personalized Daily ArXiv Papers 06/01/2026

> This is a remedial run for missed papers from 2026-06-01.

Total relevant papers: 2

## 1. Expert Routing Without Auxiliary Losses
[arxiv link](https://arxiv.org/abs/2606.00001)

## 2. Overlapping All-to-All With Expert Compute
[arxiv link](https://arxiv.org/abs/2606.00002)
"""


class RemedialRenderWinsTest(unittest.TestCase):
    """A rebuilt day must be the one that gets published."""

    def _day_dir(self, *names: str) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        month = Path(tmp.name) / "2026-06"
        month.mkdir(parents=True)
        for name in names:
            body = EMPTY_PAGE if name.endswith("-latest.md") else REBUILT_PAGE
            (month / name).write_text(body, encoding="utf-8")
        return Path(tmp.name)

    def test_the_rebuilt_render_wins_when_both_exist(self):
        root = self._day_dir("2026-06-01-latest.md", "2026-06-01-output.md")
        picked = discover_daily_markdown(root)[(2026, 6, 1)]
        self.assertEqual(picked.name, "2026-06-01-output.md")
        # The point of the fix is the CONTENT, not the filename.
        self.assertIn("arxiv.org/abs/2606.00001", picked.read_text(encoding="utf-8"))
        self.assertNotIn("No papers selected", picked.read_text(encoding="utf-8"))

    def test_an_ordinary_day_with_only_latest_is_untouched(self):
        root = self._day_dir("2026-06-02-latest.md")
        picked = discover_daily_markdown(root)[(2026, 6, 2)]
        self.assertEqual(picked.name, "2026-06-02-latest.md")

    def test_a_rebuilt_day_with_only_output_is_found(self):
        root = self._day_dir("2026-06-03-output.md")
        picked = discover_daily_markdown(root)[(2026, 6, 3)]
        self.assertEqual(picked.name, "2026-06-03-output.md")

    def test_markdown_and_json_discovery_agree_on_which_run_supersedes(self):
        # They disagreed for months: JSON preferred "output" (so the monthly
        # summaries were right) while markdown preferred "latest" (so the day
        # pages were wrong). Same data, two answers.
        self.assertLess(
            SUFFIX_PRIORITY["output"], SUFFIX_PRIORITY["latest"],
            "the remedial render must outrank the original",
        )
        self.assertLess(
            JSON_SUFFIX_PRIORITY["output"], JSON_SUFFIX_PRIORITY["latest"],
            "the JSON side must keep preferring the remedial bundle",
        )


if __name__ == "__main__":
    unittest.main()
