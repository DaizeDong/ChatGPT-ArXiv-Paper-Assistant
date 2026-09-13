from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNED_DIRS = ("arxiv_assistant", "scripts")


class UndefinedNameTest(unittest.TestCase):
    """No shipped module may reference a name that is not in scope.

    WHY THIS EXISTS. A CLI flag was added to the argument parser and read inside
    a function as `args.source`, but that function takes explicit parameters and
    has no `args` in scope. Python only notices at RUNTIME, on the line that
    runs it, so the mistake survived a syntax check and an import, and the
    backfill it drives failed on all 39 dates at once -- each one spending a
    process start to reach the same NameError.

    Nothing in the test suite could have caught it: the failing line only runs
    inside a full pipeline execution that talks to models. A static
    undefined-name check does catch it, and catches the whole class rather than
    this instance.
    """

    def test_no_undefined_names_in_shipped_code(self):
        targets = [str(REPO_ROOT / d) for d in SCANNED_DIRS]
        result = subprocess.run(
            [sys.executable, "-m", "pyflakes", *targets],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        findings = [
            line for line in (result.stdout or "").splitlines()
            if "undefined name" in line
        ]
        self.assertEqual(findings, [], "\n".join(findings))


if __name__ == "__main__":
    unittest.main()
