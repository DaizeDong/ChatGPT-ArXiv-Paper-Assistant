from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNED_DIRS = ("arxiv_assistant", "scripts")


class UndefinedNameTest(unittest.TestCase):
    """No shipped module may reference a name that is not in scope."""

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
