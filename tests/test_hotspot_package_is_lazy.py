"""The hotspot package root must not drag the pipeline into every import.

2026-09-19: the published site went down for a day. `hotspot/__init__.py`
eagerly re-exported three names from pipeline.py, and pipeline.py builds
`ZoneInfo("America/New_York")` at module scope. `rebuild_hotspot_web_data.py`
imports one date helper out of `hotspot.support.dates`; on the Windows runner,
which has no system time zone database and had no `tzdata` installed, that one
import raised ZoneInfoNotFoundError.

Two separate faults, so two separate checks: the dependency was missing, and
the coupling meant a missing dependency in one module could break every other.
Either one alone would have kept the site up.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "test-key")

REPO_ROOT = Path(__file__).resolve().parents[1]

# Run in a fresh interpreter: the test session has almost certainly imported
# pipeline already, so checking sys.modules in-process would pass no matter
# what the package root does.
PROBE = """
import sys
sys.path.insert(0, %r)
import %s
print("pipeline" if "arxiv_assistant.hotspot.pipeline" in sys.modules else "clean")
"""


def _imports_pipeline(module: str) -> bool:
    proc = subprocess.run(
        [sys.executable, "-c", PROBE % (str(REPO_ROOT), module)],
        capture_output=True, text=True,
        env={**os.environ, "OPENAI_API_KEY": "test-key"},
    )
    if proc.returncode != 0:
        raise AssertionError("importing %s failed:\n%s" % (module, proc.stderr))
    return proc.stdout.strip() == "pipeline"


class HotspotPackageIsLazyTests(unittest.TestCase):
    def test_a_leaf_import_does_not_pull_in_the_pipeline(self):
        for module in ("arxiv_assistant.hotspot.support.dates",
                       "arxiv_assistant.hotspot"):
            with self.subTest(module=module):
                self.assertFalse(
                    _imports_pipeline(module),
                    "%s pulled in pipeline.py; a module-scope failure there now "
                    "breaks every caller of this package" % module,
                )

    def test_the_re_exports_still_resolve(self):
        # Lazy must not mean gone: scripts/generate_daily_hotspots.py imports
        # these three off the package root.
        import arxiv_assistant.hotspot as hotspot

        for name in ("detect_latest_local_output_date",
                     "generate_daily_hotspot_report",
                     "parse_target_datetime"):
            self.assertTrue(callable(getattr(hotspot, name)), name)

    def test_the_timezone_the_pipeline_needs_is_actually_available(self):
        # The runner has no system tz database. Without `tzdata` in
        # requirements.txt this raises, which is what took the site down.
        from zoneinfo import ZoneInfo

        self.assertIsNotNone(ZoneInfo("America/New_York"))

    def test_tzdata_is_pinned_in_requirements(self):
        text = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertRegex(text, r"(?m)^tzdata==")


if __name__ == "__main__":
    unittest.main()
