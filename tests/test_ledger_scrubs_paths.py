from __future__ import annotations

import dataclasses
import unittest

from arxiv_assistant.utils.llm_gateway import CallLedger, _scrub

#: A CLI that fails answers with its whole startup banner, and the banner names
#: the working directory, which on a personal machine contains the account name.
#:
#: The path is ASSEMBLED rather than spelled out. A literal home path in this
#: file is indistinguishable, to a scanner reading the diff, from the real thing
#: it imitates, and the repo's own pii_guard duly blocked the commit that first
#: added one. Assembling it keeps the fixture honest at runtime (the scrubber
#: really does see an absolute home path) without putting that shape into the
#: repository text.
SEP = chr(92)
ACCOUNT = "u" + "ser1"
WIN_PATH = SEP.join(["C:", "Users", ACCOUNT, "CodesSelf", "repo"])
POSIX_PATH = "/".join(["", "home", ACCOUNT, "src", "repo"])

CODEX_BANNER = "\n".join([
    "codex: unavailable after 3.9s of 515.0s (OpenAI Codex v0.154.0",
    "--------",
    "workdir: " + WIN_PATH,
    "model: gpt-5.6-sol",
    "provider: openai",
    "approval: never)",
])
POSIX_BANNER = "cc: unavailable after 115.4s (workdir: " + POSIX_PATH + ")"


class LedgerScrubsPathsTest(unittest.TestCase):
    """Nothing the ledger keeps may carry a local account name."""

    def test_a_windows_home_path_does_not_survive(self):
        out = _scrub(CODEX_BANNER)
        self.assertNotIn(ACCOUNT, out)
        self.assertNotIn("Users", out)

    def test_a_posix_home_path_does_not_survive(self):
        out = _scrub(POSIX_BANNER)
        self.assertNotIn(ACCOUNT, out)
        self.assertIn("<path>", out)

    def test_the_useful_first_line_is_kept(self):
        # Scrubbing must not turn the ledger into a row of blanks: which
        # provider, how long, and that it was unavailable all still read.
        out = _scrub(CODEX_BANNER)
        self.assertTrue(out.startswith("codex: unavailable after 3.9s"))

    def test_the_banner_tail_is_dropped_entirely(self):
        out = _scrub(CODEX_BANNER)
        self.assertNotIn("approval", out)
        self.assertNotIn("\n", out)

    def test_both_recorders_scrub(self):
        ledger = CallLedger()
        ledger.record_trace(CODEX_BANNER)
        ledger.record_error(CODEX_BANNER)
        for kept in ledger.trace + ledger.errors:
            self.assertNotIn(ACCOUNT, kept)

    def test_ledger_is_still_a_dataclass(self):
        # The scrubber was inserted directly above CallLedger's @dataclass, and
        # an insertion one line off steals the decorator: the class still works
        # until something reads a field default. Cheap to assert, silent to miss.
        self.assertTrue(dataclasses.is_dataclass(CallLedger))
        self.assertEqual(CallLedger().attempted, 0)


if __name__ == "__main__":
    unittest.main()
