from __future__ import annotations

import unittest

from arxiv_assistant.renderers.paper.render_daily import render_summary_table


def _table(**over):
    args = dict(
        model="llmcall:codexg+cc",
        prompt_tokens=0,
        completion_tokens=0,
        prompt_cost=0.0,
        completion_cost=0.0,
        total_arxiv_papers=474,
        total_scanned_papers=474,
        total_relevant_papers=11,
    )
    args.update(over)
    return render_summary_table(**args)


class UsageTableHonestyTest(unittest.TestCase):
    """An unmeasured run must not render as a free one.

    WHY THIS EXISTS. CLI-backed providers answer without reporting tokens, so
    the counters stay at zero. The table printed "0" and "$0.00", which reads as
    "this cost nothing" rather than "nobody counted". 112 rebuilt days were
    published that way, each also crediting a model that never ran.
    """

    def test_unreported_usage_is_labelled_not_priced_at_zero(self):
        html = _table()
        self.assertIn("not reported", html)
        self.assertNotIn("$0.00", html)

    def test_real_usage_still_renders_as_numbers(self):
        html = _table(prompt_tokens=1200, completion_tokens=300, prompt_cost=0.6, completion_cost=0.45)
        self.assertIn("1200", html)
        self.assertIn("$0.60", html)
        self.assertIn("$1.05", html)
        self.assertNotIn("not reported", html)

    def test_paper_counts_survive_either_way(self):
        # The counts are measured even when the usage is not, so they must not
        # disappear with it -- the remedial renderer used to drop them entirely.
        for html in (_table(), _table(prompt_tokens=10, completion_tokens=5)):
            self.assertIn("474", html)
            self.assertIn("11", html)

    def test_the_model_cell_shows_whatever_it_was_given(self):
        # The caller decides; this asserts the table does not substitute a
        # default of its own, which is how "gpt-5.4" outlived the OpenAI path.
        self.assertIn("llmcall:codexg+cc", _table())
        self.assertIn("llmcall:none", _table(model="llmcall:none"))


if __name__ == "__main__":
    unittest.main()
