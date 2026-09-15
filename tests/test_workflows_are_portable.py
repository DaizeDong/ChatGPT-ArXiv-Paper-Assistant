from __future__ import annotations

import unittest
from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _workflow_files():
    return sorted(p for p in WORKFLOWS.glob("*.y*ml"))


class WorkflowPortabilityTest(unittest.TestCase):
    """The workflows must run on whichever self-hosted runner is registered."""

    def test_there_are_workflows_to_check(self):
        # Guards the whole file: a glob that silently matches nothing would
        # make every assertion below vacuously true.
        self.assertGreaterEqual(len(_workflow_files()), 5)

    def test_no_workflow_is_pinned_to_a_hosted_runner(self):
        offenders = []
        for path in _workflow_files():
            spec = yaml.safe_load(path.read_text(encoding="utf-8"))
            for job_name, job in (spec.get("jobs") or {}).items():
                runs_on = (job or {}).get("runs-on")
                flat = runs_on if isinstance(runs_on, str) else " ".join(map(str, runs_on or []))
                if "self-hosted" not in flat:
                    offenders.append(f"{path.name}:{job_name} -> {runs_on}")
        self.assertEqual(offenders, [], "\n".join(offenders))

    def test_every_run_step_declares_its_shell(self):
        offenders = []
        for path in _workflow_files():
            spec = yaml.safe_load(path.read_text(encoding="utf-8"))
            for job_name, job in (spec.get("jobs") or {}).items():
                for step in ((job or {}).get("steps") or []):
                    if "run" in step and not step.get("shell"):
                        label = step.get("name") or step["run"].strip().splitlines()[0][:40]
                        offenders.append(f"{path.name}:{job_name}: {label}")
        self.assertEqual(offenders, [], "\n".join(offenders))


if __name__ == "__main__":
    unittest.main()
