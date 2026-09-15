"""The outage detector is tested against REAL rows from the published archive."""
import unittest

from arxiv_assistant.utils.pipeline_health import (
    STATUS_LLM_UNAVAILABLE,
    STATUS_NO_INPUT,
    STATUS_OK,
    STATUS_SKIPPED,
    assess_paper_filter_health,
    format_banner,
)

# Verbatim from origin/auto_update:out/json/2026-09/2026-09-07-daily-papers.json
REAL_OUTAGE_DAY = dict(
    scanned_papers=342, selected_papers=0, prompt_tokens=0, completion_tokens=0
)
# Verbatim from origin/auto_update:out/json/2026-06/2026-06-04-daily-papers.json
REAL_HEALTHY_DAY = dict(
    scanned_papers=443, selected_papers=16, prompt_tokens=173454, completion_tokens=25482
)


class PipelineHealthTests(unittest.TestCase):
    def test_real_outage_day_is_flagged(self):
        health = assess_paper_filter_health(**REAL_OUTAGE_DAY, llm_filtering_enabled=True)
        self.assertEqual(health.status, STATUS_LLM_UNAVAILABLE)
        self.assertTrue(health.is_outage)
        self.assertIn("ZERO tokens", health.message)

    def test_real_healthy_day_is_not_flagged(self):
        health = assess_paper_filter_health(**REAL_HEALTHY_DAY, llm_filtering_enabled=True)
        self.assertEqual(health.status, STATUS_OK)
        self.assertFalse(health.is_outage)

    def test_negative_control_zero_selected_but_tokens_spent_is_healthy(self):
        """The gate must NOT fire on a genuinely quiet day.

        Scoring ran, cost real tokens, and simply kept nothing. That is the case
        the outage must never be confused with, in either direction.
        """
        health = assess_paper_filter_health(
            scanned_papers=400,
            selected_papers=0,
            prompt_tokens=150000,
            completion_tokens=20000,
            llm_filtering_enabled=True,
        )
        self.assertEqual(health.status, STATUS_OK)
        self.assertFalse(health.is_outage)

    def test_disabled_llm_filtering_is_not_an_outage(self):
        health = assess_paper_filter_health(
            **REAL_OUTAGE_DAY, llm_filtering_enabled=False
        )
        self.assertEqual(health.status, STATUS_SKIPPED)
        self.assertFalse(health.is_outage)

    def test_no_papers_scanned_is_not_an_outage(self):
        health = assess_paper_filter_health(
            scanned_papers=0,
            selected_papers=0,
            prompt_tokens=0,
            completion_tokens=0,
            llm_filtering_enabled=True,
        )
        self.assertEqual(health.status, STATUS_NO_INPUT)
        self.assertFalse(health.is_outage)

    def test_partial_tokens_still_counts_as_ran(self):
        health = assess_paper_filter_health(
            scanned_papers=342,
            selected_papers=0,
            prompt_tokens=0,
            completion_tokens=7,
            llm_filtering_enabled=True,
        )
        self.assertEqual(health.status, STATUS_OK)

    def test_banner_is_loud_for_outage_and_empty_otherwise(self):
        outage = assess_paper_filter_health(**REAL_OUTAGE_DAY, llm_filtering_enabled=True)
        healthy = assess_paper_filter_health(**REAL_HEALTHY_DAY, llm_filtering_enabled=True)
        banner = format_banner(outage)
        self.assertIn("PAPER FILTER OUTAGE", banner)
        self.assertIn("OPENAI_API_KEY", banner)
        self.assertEqual(format_banner(healthy), "")

class CallLedgerSignalTests(unittest.TestCase):
    """On a token-less backend the ledger is the only honest signal.

    The llmcall chain reports no OpenAI tokens, so a token-based detector would
    call every healthy run an outage. These tests pin the inversion: with a
    ledger present, tokens are irrelevant in BOTH directions.
    """

    def test_calls_attempted_none_succeeded_is_an_outage(self):
        health = assess_paper_filter_health(
            scanned_papers=342,
            selected_papers=0,
            prompt_tokens=0,
            completion_tokens=0,
            llm_filtering_enabled=True,
            llm_calls_attempted=35,
            llm_calls_succeeded=0,
        )
        self.assertEqual(health.status, STATUS_LLM_UNAVAILABLE)
        self.assertIn("NONE succeeded", health.message)

    def test_zero_calls_attempted_while_papers_waited_is_an_outage(self):
        health = assess_paper_filter_health(
            scanned_papers=342,
            selected_papers=0,
            prompt_tokens=0,
            completion_tokens=0,
            llm_filtering_enabled=True,
            llm_calls_attempted=0,
            llm_calls_succeeded=0,
        )
        self.assertEqual(health.status, STATUS_LLM_UNAVAILABLE)
        self.assertIn("ZERO model", health.message)

    def test_successful_calls_with_zero_tokens_is_HEALTHY(self):
        """THE REGRESSION THIS EXISTS FOR."""
        health = assess_paper_filter_health(
            scanned_papers=443,
            selected_papers=16,
            prompt_tokens=0,
            completion_tokens=0,
            llm_filtering_enabled=True,
            llm_calls_attempted=35,
            llm_calls_succeeded=35,
        )
        self.assertEqual(health.status, STATUS_OK)
        self.assertFalse(health.is_outage)

    def test_partial_success_is_not_an_outage(self):
        health = assess_paper_filter_health(
            scanned_papers=443,
            selected_papers=4,
            prompt_tokens=0,
            completion_tokens=0,
            llm_filtering_enabled=True,
            llm_calls_attempted=35,
            llm_calls_succeeded=3,
        )
        self.assertEqual(health.status, STATUS_OK)

    def test_ledger_beats_tokens_when_both_are_present(self):
        """Tokens spent but every call failed: still an outage."""
        health = assess_paper_filter_health(
            scanned_papers=443,
            selected_papers=0,
            prompt_tokens=173454,
            completion_tokens=25482,
            llm_filtering_enabled=True,
            llm_calls_attempted=35,
            llm_calls_succeeded=0,
        )
        self.assertEqual(health.status, STATUS_LLM_UNAVAILABLE)

    def test_absent_ledger_falls_back_to_the_token_heuristic(self):
        """A legacy OpenAI-only run reports no ledger; the old path must remain."""
        health = assess_paper_filter_health(
            **REAL_OUTAGE_DAY, llm_filtering_enabled=True
        )
        self.assertEqual(health.status, STATUS_LLM_UNAVAILABLE)


class PipelineHealthSerialisationTests(unittest.TestCase):
    def test_health_is_serializable_into_the_archive_bundle(self):
        payload = assess_paper_filter_health(
            **REAL_OUTAGE_DAY, llm_filtering_enabled=True
        ).to_dict()
        self.assertEqual(
            sorted(payload),
            ["llm_tokens", "message", "scanned_papers", "selected_papers", "status"],
        )
        self.assertEqual(payload["status"], STATUS_LLM_UNAVAILABLE)


if __name__ == "__main__":
    unittest.main()
