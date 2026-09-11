"""The weekly digest must never dress an outage up as a quiet week.

There is exactly ONE state in which the page may print the tidy line
"本周没有改变看法的内容": scoring really ran, over a real window, against a
reader model that really has content, and nothing cleared the cutoff on the
merits. Every other empty page has a cause, and the cause has to be on the page.

This file is the guard for that. It is deliberately a table test: each row is a
state, and the expectation is a boolean about the tidy line. Without it, nothing
in CI stops someone from simplifying the loud branch away, and the resulting
report would look correct every single week while the pipeline was dead -- which
is precisely what happened to the paper pipeline here for three months.
"""
import json
import unittest
from pathlib import Path

from arxiv_assistant.renderers.weekly.render_weekly_digest import (
    QUIET_WEEK_LINE,
    render_weekly_digest_md,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _diagnostics(**overrides):
    """A diagnostics block describing a HEALTHY, genuinely quiet week."""
    base = {
        "days_requested": 7,
        "days_found": 7,
        "days_missing": [],
        "paper_days_empty": [],
        "paper_llm_tokens_seen": 198936,
        "paper_scanned_seen": 443,
        "paper_papers_seen": 16,
        "candidates_scored": 40,
        "verdict_status_counts": {"scored": 40, "rejected": 0, "unavailable": 0},
        "reader_questions_populated": 3,
        "reader_questions_total": 5,
        "delta_score_cutoff": 7,
        "max_deep_read": 5,
        "max_skim": 15,
        "items_over_cutoff": 0,
        # Added when every model call moved onto utils/llm_gateway. A healthy
        # week PROVES calls were made and answered; the token counters above are
        # OpenAI-only and read zero on the llmcall chain even when it is working,
        # which is exactly why an outage test cannot rest on them alone.
        "llm_backend": "LLM backend: llmcall chain=codexg->codex->cc->claude",
        "llm_ledger": {
            "attempted": 4,
            "succeeded": 4,
            "failed": 0,
            "by_backend": {"llmcall": 4},
            "by_provider": {"cc": 4},
            "errors": [],
        },
    }
    base.update(overrides)
    return base


def _digest(deep_read=(), skim=(), **diag_overrides):
    return {
        "week_start": "2026-09-03",
        "week_end": "2026-09-09",
        "deep_read": list(deep_read),
        "skim": list(skim),
        "diagnostics": _diagnostics(**diag_overrides),
    }


def _item(**overrides):
    item = {
        "candidate_id": "hotspot:2026-09-05:abc123456789",
        "kind": "hotspot",
        "title": "A routing change that removes the auxiliary loss",
        "url": "https://example.com/post",
        "date": "2026-09-05",
        "delta_score": 9,
        "question_id": "q2",
        "field": "would_change_my_mind",
        "one_line_reason": "Reports stable training to 128 experts without a load balancing loss.",
        "tiebreak": 8.5,
    }
    item.update(overrides)
    return item


class QuietLineIsReachableOnlyWhenHonest(unittest.TestCase):
    """The table. Each row: a state, and whether the tidy line may appear."""

    def test_states(self):
        cases = [
            (
                "healthy quiet week",
                _digest(),
                True,
            ),
            (
                "reader model empty",
                _digest(
                    reader_questions_populated=0,
                    verdict_status_counts={"scored": 0, "rejected": 0, "unavailable": 40},
                ),
                False,
            ),
            (
                "scoring unavailable",
                _digest(
                    verdict_status_counts={"scored": 10, "rejected": 0, "unavailable": 30}
                ),
                False,
            ),
            (
                "no archive days in window",
                _digest(
                    days_found=0,
                    days_missing=["2026-09-03", "2026-09-04"],
                    candidates_scored=0,
                    verdict_status_counts={"scored": 0, "rejected": 0, "unavailable": 0},
                ),
                False,
            ),
            (
                # The ledger's own alarm: the chain was reached N times and not
                # one provider answered. Distinct from the row above it, because
                # `unavailable` verdicts alone cannot say whether the transport
                # was dead or the reader model was empty.
                "every provider in the chain failed",
                _digest(
                    verdict_status_counts={"scored": 0, "rejected": 0, "unavailable": 40},
                    llm_ledger={
                        "attempted": 4,
                        "succeeded": 0,
                        "failed": 4,
                        "by_backend": {"llmcall": 4},
                        "by_provider": {},
                        "errors": ["llmcall chain failed: all providers exhausted"],
                    },
                ),
                False,
            ),
            (
                # Negative control for the row above. Same shape of diagnostics,
                # a ledger that says calls SUCCEEDED, and the tidy line is allowed
                # again. Without this row a renderer that shouted on every ledger,
                # or ignored the ledger entirely, would still pass.
                "ledger proves the chain answered and nothing cleared the bar",
                _digest(
                    llm_ledger={
                        "attempted": 6,
                        "succeeded": 6,
                        "failed": 0,
                        "by_backend": {"llmcall": 6},
                        "by_provider": {"codexg": 6},
                        "errors": [],
                    }
                ),
                True,
            ),
            (
                # A digest built before the ledger existed. Absence is not an
                # alarm: the older causes still cover it.
                "no ledger at all (pre-gateway digest)",
                _digest(llm_ledger=None, llm_backend=None),
                True,
            ),
            (
                "paper pipeline outage (scanned but zero tokens)",
                _digest(
                    paper_scanned_seen=996,
                    paper_papers_seen=0,
                    paper_llm_tokens_seen=0,
                    paper_days_empty=["2026-09-03", "2026-09-04", "2026-09-07"],
                ),
                False,
            ),
        ]
        for name, digest, quiet_allowed in cases:
            with self.subTest(state=name):
                md = render_weekly_digest_md(digest)
                self.assertEqual(
                    QUIET_WEEK_LINE in md,
                    quiet_allowed,
                    f"state {name!r}: tidy line presence was wrong.\n{md}",
                )

    def test_dead_chain_block_names_the_backend_and_the_error(self):
        """Loud is not enough: the page has to say WHICH backend and WHY, or the
        reader is left running the diagnosis from scratch."""
        md = render_weekly_digest_md(
            _digest(
                verdict_status_counts={"scored": 0, "rejected": 0, "unavailable": 40},
                llm_backend="LLM backend: llmcall chain=codexg->codex->cc->claude",
                llm_ledger={
                    "attempted": 4,
                    "succeeded": 0,
                    "failed": 4,
                    "by_backend": {"llmcall": 4},
                    "by_provider": {},
                    "errors": ["llmcall chain failed: all providers exhausted"],
                },
            )
        )
        self.assertNotIn(QUIET_WEEK_LINE, md)
        self.assertIn("codexg->codex->cc->claude", md)
        self.assertIn("all providers exhausted", md)
        self.assertIn("4", md)

    def test_missing_diagnostics_block_renders_loud_not_tidy(self):
        """A digest that LOST its diagnostics must not read as a calm week.

        A renderer that treated absent counters as confident zeros would print
        the tidy line here, which is the same class of bug as reading an empty
        archive as "nothing was relevant".
        """
        md = render_weekly_digest_md(
            {"week_start": "2026-09-03", "week_end": "2026-09-09", "deep_read": [], "skim": []}
        )
        self.assertNotIn(QUIET_WEEK_LINE, md)

    def test_outage_warning_still_shows_when_some_items_were_kept(self):
        """A partial outage must not be hidden by a non-empty result."""
        md = render_weekly_digest_md(
            _digest(
                deep_read=[_item()],
                verdict_status_counts={"scored": 10, "rejected": 0, "unavailable": 30},
            )
        )
        self.assertNotIn(QUIET_WEEK_LINE, md)
        self.assertIn(_item()["title"], md)


_CONFIG_TEXT = """[READER]
enabled = true
delta_score_cutoff = 7

[LLM]
backend = auto
effort = max
"""

_QUESTION_DOC = """# Q1: Do continuous bit-widths beat integer quantization?

## 当前看法

Integer bit-widths are a deployment artifact, not an accuracy limit.
"""


class LedgerReachesTheDigest(unittest.TestCase):
    """The renderer above is only useful if the SCRIPT actually puts a ledger in
    the diagnostics. A test of the renderer alone would pass forever while the
    producer emitted nothing -- which is the same "the check was fed nothing"
    failure this whole feature exists to prevent."""

    def _build(self, transport):
        import configparser
        import tempfile
        from datetime import date

        from arxiv_assistant.utils import llm_gateway
        from scripts.generate_weekly_digest import build_weekly_digest

        config = configparser.ConfigParser()
        config.read_string(_CONFIG_TEXT)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            questions = root / "questions"
            questions.mkdir()
            (questions / "q1.md").write_text(
                _QUESTION_DOC,
                encoding="utf-8",
            )
            hot = root / "hot" / "reports"
            hot.mkdir(parents=True)
            (hot / "2026-09-09.json").write_text(
                json.dumps(
                    {
                        "date": "2026-09-09",
                        "featured_topics": [
                            {
                                "TOPIC_ID": "t1",
                                "HEADLINE": "A routing change",
                                "WHY_IT_MATTERS": "Removes the auxiliary loss.",
                                "FINAL_SCORE": 8.5,
                                "items": [{"url": "https://example.com/post"}],
                            }
                        ],
                        "watchlist": [],
                    }
                ),
                encoding="utf-8",
            )
            llm_gateway.LEDGER.reset()
            digest = build_weekly_digest(
                archive_dir=root,
                end_date=date(2026, 9, 9),
                days=1,
                config=config,
                agent_fn=transport,
                questions_dir=questions,
            )
        return digest["diagnostics"]

    def test_a_healthy_run_records_its_calls(self):
        def transport(prompt, **kwargs):
            return {
                "verdicts": [
                    {
                        "index": 0,
                        "question_id": "q1",
                        "field": "current_view",
                        "delta_score": 9,
                        "one_line_reason": "reports 3.4 effective bits at 2-bit memory cost",
                    }
                ]
            }

        diagnostics = self._build(transport)
        self.assertIn("llm_backend", diagnostics)
        self.assertEqual(diagnostics["llm_ledger"]["attempted"], 1)
        self.assertEqual(diagnostics["llm_ledger"]["succeeded"], 1)
        self.assertEqual(diagnostics["llm_ledger"]["failed"], 0)
        self.assertEqual(diagnostics["llm_ledger"]["by_backend"], {"agent": 1})
        self.assertTrue(str(diagnostics["llm_backend"]).startswith("LLM backend:"))

    def test_a_dead_transport_produces_a_ledger_the_renderer_shouts_about(self):
        """End to end: every call fails -> the digest's own diagnostics make the
        page loud. This is the path the three-month outage took, on the new backend."""
        from arxiv_assistant.utils.agent_runner import AgentError

        def dead(prompt, **kwargs):
            raise AgentError("claude -p exited 1")

        diagnostics = self._build(dead)
        self.assertGreater(diagnostics["llm_ledger"]["attempted"], 0)
        self.assertEqual(diagnostics["llm_ledger"]["succeeded"], 0)
        md = render_weekly_digest_md(
            {
                "week_start": "2026-09-09",
                "week_end": "2026-09-09",
                "deep_read": [],
                "skim": [],
                "diagnostics": diagnostics,
            }
        )
        self.assertNotIn(QUIET_WEEK_LINE, md)
        self.assertIn("claude -p exited 1", md)


class DigestShapeTests(unittest.TestCase):
    def test_caps_are_respected_by_the_renderer_output(self):
        deep = [_item(candidate_id=f"d{i}", title=f"Deep {i}") for i in range(5)]
        skim = [_item(candidate_id=f"s{i}", title=f"Skim {i}") for i in range(15)]
        md = render_weekly_digest_md(_digest(deep_read=deep, skim=skim, items_over_cutoff=20))
        for entry in deep + skim:
            self.assertIn(entry["title"], md)
        self.assertNotIn(QUIET_WEEK_LINE, md)

    def test_digest_has_no_source_first_or_category_expansion_tables(self):
        """Spec: those views stay on the daily archive pages, not in the digest."""
        md = render_weekly_digest_md(
            _digest(deep_read=[_item()], skim=[_item(candidate_id="s1", title="Skim one")])
        )
        for banned in (
            "Source Stats",
            "Topic Radar By Category",
            "Long-tail Signals",
            "Table of contents by topic",
            "Topic Coverage",
        ):
            self.assertNotIn(banned, md, f"digest must not contain {banned!r}")

    def test_every_entry_carries_score_question_and_reason(self):
        md = render_weekly_digest_md(_digest(deep_read=[_item()]))
        self.assertIn("q2", md)
        self.assertIn("would_change_my_mind", md)
        self.assertIn("Reports stable training to 128 experts", md)
        self.assertIn("https://example.com/post", md)


class DailyRenderersUntouched(unittest.TestCase):
    """The weekly renderer must be additive: the daily markdown is unchanged."""

    def test_weekly_renderer_does_not_import_the_daily_assemblers(self):
        source = (
            REPO_ROOT
            / "arxiv_assistant"
            / "renderers"
            / "weekly"
            / "render_weekly_digest.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("render_daily_md", source)
        self.assertNotIn("render_hot_daily_md", source)


class ReaderConfigIsDeclaredEverywhere(unittest.TestCase):
    """All three ini files must carry [READER] and the weekly Slack key.

    Asserted through ConfigParser, not by substring: a key that appears only
    inside a comment would satisfy an `assertIn` on the raw text while being
    invisible to every reader at runtime. configs/profiles/agent-native.ini
    matters most -- README.md documents copying it OVER config.ini, so a section
    present only in config.ini does not exist on the zero-key deployment.
    """

    INI_FILES = (
        "configs/config.ini",
        "configs/templates/config.template.ini",
        "configs/profiles/agent-native.ini",
    )

    def test_reader_section_parses_in_every_config(self):
        import configparser

        for rel in self.INI_FILES:
            with self.subTest(config=rel):
                parser = configparser.ConfigParser()
                read = parser.read(REPO_ROOT / rel, encoding="utf-8")
                # ConfigParser.read() does not raise on a missing file; it just
                # returns an empty list and every getter falls back. Check it.
                self.assertTrue(read, f"{rel} was not read at all")
                self.assertTrue(parser.has_section("READER"), f"{rel} has no [READER]")
                reader = parser["READER"]
                self.assertEqual(reader.getint("delta_score_cutoff"), 7)
                self.assertEqual(reader.getint("max_deep_read"), 5)
                self.assertEqual(reader.getint("max_skim"), 15)
                self.assertEqual(
                    reader.get("questions_dir"), "configs/reader/questions"
                )

    def test_daily_slack_stays_off_and_weekly_has_its_own_switch(self):
        import configparser

        for rel in self.INI_FILES:
            with self.subTest(config=rel):
                parser = configparser.ConfigParser()
                parser.read(REPO_ROOT / rel, encoding="utf-8")
                # main.py and scripts/remedy_missed_dates.py read push_to_slack.
                # Flipping it for the weekly would restart the daily posts.
                self.assertFalse(parser["OUTPUT"].getboolean("push_to_slack"))
                self.assertIn("push_weekly_to_slack", parser["OUTPUT"])

    def test_narrowed_thresholds_are_applied_everywhere(self):
        import configparser

        for rel in self.INI_FILES:
            with self.subTest(config=rel):
                parser = configparser.ConfigParser()
                parser.read(REPO_ROOT / rel, encoding="utf-8")
                self.assertEqual(parser["FILTERING"].getint("relevance_cutoff"), 8)
                self.assertEqual(parser["FILTERING"].getint("novelty_cutoff"), 7)
                self.assertEqual(parser["HOTSPOTS"].getint("target_topics"), 3)
                self.assertEqual(
                    parser["HOTSPOTS"].getint("target_watchlist_topics"), 2
                )


class WeeklyOutputStaysOutOfTheSiteBuilder(unittest.TestCase):
    """out/weekly/ must not collide with the day-page discovery in the site build.

    build_multipage_site's DAY_FILE_PATTERN matches `<date>-<anything>.md` under
    out/md/, so a weekly page written there would either hijack a day with no
    daily markdown or be silently dropped on a day that has one. The frontend is
    out of scope for this change, so the weekly writer must stay out of out/md/.
    """

    def test_script_never_writes_under_out_md(self):
        source = (REPO_ROOT / "scripts" / "generate_weekly_digest.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"weekly"', source)
        self.assertNotIn('"md"', source)

    def test_site_builder_is_untouched_by_this_change(self):
        source = (
            REPO_ROOT
            / "arxiv_assistant"
            / "renderers"
            / "build_multipage_site.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("weekly", source.lower())


if __name__ == "__main__":
    unittest.main()
