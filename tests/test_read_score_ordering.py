import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from arxiv_assistant.filters.filter_gpt import read_score
from arxiv_assistant.paper_topics import daily_sort_key


class ReadScoreTests(unittest.TestCase):
    def test_novelty_does_not_move_the_reading_order(self):
        # The measured reason for the formula: novelty is flat across the middle
        # tiers and highest on the tier that is too complicated to act on, so it
        # must not enter the ordering. A regression that adds it back fails here.
        low = read_score({"NOVELTY": 1, "PROXIMITY": 7, "EVIDENCE": 7, "LOAD": 4}, 8)
        high = read_score({"NOVELTY": 10, "PROXIMITY": 7, "EVIDENCE": 7, "LOAD": 4}, 8)
        self.assertEqual(low, high)

    def test_a_heavy_paper_sorts_below_an_equal_but_lighter_one(self):
        light = read_score({"PROXIMITY": 7, "EVIDENCE": 7, "LOAD": 2}, 8)
        heavy = read_score({"PROXIMITY": 7, "EVIDENCE": 7, "LOAD": 10}, 8)
        self.assertGreater(light, heavy)

    def test_missing_axes_fall_back_to_the_middle_rather_than_zero(self):
        # A scorer that omits the axes must not send the paper to the bottom of
        # the day; absent is not the same as bad.
        self.assertEqual(read_score({}, 8), round(8 + 5 + 5 - 2.5, 1))

    def test_archived_entries_keep_their_original_order(self):
        # Days scored before READ_SCORE existed have only SCORE. They must still
        # order by it, not collapse to a tie at zero.
        old_high = daily_sort_key({"SCORE": 18, "RELEVANCE": 9})
        old_low = daily_sort_key({"SCORE": 12, "RELEVANCE": 9})
        self.assertGreater(old_high, old_low)

    def test_read_score_wins_over_score_when_present(self):
        ranked = sorted(
            [
                {"SCORE": 20, "READ_SCORE": 9.0, "RELEVANCE": 10},
                {"SCORE": 12, "READ_SCORE": 19.5, "RELEVANCE": 6},
            ],
            key=daily_sort_key,
            reverse=True,
        )
        self.assertEqual(ranked[0]["READ_SCORE"], 19.5)


if __name__ == "__main__":
    unittest.main()
