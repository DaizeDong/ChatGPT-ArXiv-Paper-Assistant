import ast
import unittest
from pathlib import Path

from arxiv_assistant.utils.models import DEFAULT_AGENT_MODEL, DEFAULT_DEEP_MODEL

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "arxiv_assistant"
CENTRAL = PACKAGE / "utils" / "models.py"
# Generated price table: it names every model on the market by design.
GENERATED = {PACKAGE / "utils" / "pricing.py"}


DOC_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _string_constants(path):
    # Only real string literals: a docstring or comment naming a model is prose,
    # not a second source of truth.
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, DOC_OWNERS) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docstrings.add(id(first.value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                yield node.value


class ModelIdsAreCentralTest(unittest.TestCase):
    # models.py was added as "the one place model ids live" and then nothing
    # imported it: the nine modules kept their own literals. Comparing the two
    # constants would have passed anyway, so this counts residuals instead.

    def test_no_module_repeats_a_central_model_id(self):
        offenders = []
        for path in PACKAGE.rglob("*.py"):
            if path == CENTRAL or path in GENERATED:
                continue
            for value in _string_constants(path):
                if value in (DEFAULT_AGENT_MODEL, DEFAULT_DEEP_MODEL):
                    offenders.append(f"{path.relative_to(REPO).as_posix()}: {value}")
        self.assertEqual(offenders, [], "\n".join(offenders))

    def test_the_central_ids_are_current(self):
        self.assertNotIn(DEFAULT_AGENT_MODEL, {"claude-sonnet-4-6", "claude-sonnet-4-5"})
        self.assertNotIn(DEFAULT_DEEP_MODEL, {"claude-opus-4-8", "claude-opus-4-6"})

    def test_the_scan_can_see_literals_at_all(self):
        # Without this, a broken parser would report a clean tree forever.
        found = list(_string_constants(PACKAGE / "utils" / "llm_gateway.py"))
        self.assertIn("llmcall", found)


if __name__ == "__main__":
    unittest.main()
