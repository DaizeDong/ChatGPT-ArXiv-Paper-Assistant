"""A build that runs for an hour must not be cancellable by the next trigger.

2026-09-15 to 2026-09-19: the published site went stale for four days. Five
consecutive Pages builds died at the translation step with conclusion
`cancelled`, not `failure`, so nothing looked broken -- no error, no failing
step, just a run that stopped. The cause was `cancel-in-progress: true` on a
67-minute build whose triggers include pushes to main under `arxiv_assistant/**`
and `scripts/**`. Routine development on main was killing the publish.

`cancel-in-progress` is right for a fast check where only the newest commit
matters. It is wrong for a slow publish, which can then never reach the end.
The threshold below is deliberately crude: any workflow that talks to a model
per item is in the slow class.
"""

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"

#: Workflows whose runtime is dominated by per-item model calls.
SLOW_WORKFLOWS = ("publish_md.yml", "weekly_digest.yaml")


def _concurrency_block(text: str) -> str:
    match = re.search(r"(?ms)^concurrency:\n((?:[ \t]+.*\n|\n)*)", text)
    return match.group(1) if match else ""


class LongBuildsAreNotCancelledTests(unittest.TestCase):
    def test_slow_workflows_queue_rather_than_cancel(self):
        for name in SLOW_WORKFLOWS:
            path = WORKFLOW_DIR / name
            with self.subTest(workflow=name):
                self.assertTrue(path.exists(), "%s is gone; update this test" % name)
                block = _concurrency_block(path.read_text(encoding="utf-8"))
                self.assertTrue(block.strip(), "%s has no concurrency block" % name)
                setting = re.search(r"(?m)^\s*cancel-in-progress:\s*(\S+)", block)
                self.assertIsNotNone(
                    setting, "%s does not state cancel-in-progress" % name
                )
                self.assertEqual(
                    setting.group(1), "false",
                    "%s can be cancelled mid-build. This run takes about an hour "
                    "and its own triggers fire more often than that, so it would "
                    "never reach the end -- and it would report `cancelled`, "
                    "which reads like nothing went wrong." % name,
                )

    def test_every_workflow_that_sets_concurrency_says_which_it_wants(self):
        # An omitted cancel-in-progress defaults to false, which is the safe
        # side -- but silence makes the two classes indistinguishable to a
        # reader deciding whether a given build may be interrupted.
        for path in sorted(WORKFLOW_DIR.glob("*.y*ml")):
            text = path.read_text(encoding="utf-8")
            block = _concurrency_block(text)
            if not block.strip():
                continue
            with self.subTest(workflow=path.name):
                self.assertRegex(
                    block, r"(?m)^\s*cancel-in-progress:",
                    "%s groups runs but never says whether one may be killed "
                    "mid-flight" % path.name,
                )


if __name__ == "__main__":
    unittest.main()
