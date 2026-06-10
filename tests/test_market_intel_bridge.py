from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from arxiv_assistant.utils import market_intel_bridge as bridge

_SHARD = """# Domain: frontier-research

**Triage signals:** AI/ML papers, arXiv, SOTA, 论文.

> Real-run lesson: arXiv + HF Daily Papers are the L1/L2 floor.

| source | route | capability | detect | note/risk |
|---|---|---|---|---|
| **arXiv API** | (1) free | recent by category | REST, no key | be polite |
| **Hugging Face Daily Papers** | (1) free | curated daily papers | REST | no key for read |

**Default pick:** Recent papers -> arXiv API + HF Daily Papers (free).

**Browser route:** lab blogs via playwright.
"""

_NO_TABLE_SHARD = "# Domain: x-twitter\n\nSome prose about X.\nMore prose.\n"


def _make_domains_dir(tmp: str, *, with_anchor=True, extra=None) -> Path:
    domains = Path(tmp) / "reference" / "domains"
    domains.mkdir(parents=True, exist_ok=True)
    if with_anchor:
        (domains / "frontier-research.md").write_text(_SHARD, encoding="utf-8")
    for name, content in (extra or {}).items():
        (domains / name).write_text(content, encoding="utf-8")
    return domains


class TestFindSkillDomainsDir(unittest.TestCase):
    def test_explicit_domains_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            domains = _make_domains_dir(tmp)
            found = bridge.find_skill_domains_dir(explicit=str(domains))
            self.assertEqual(found, domains)

    def test_explicit_skill_root_is_expanded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            domains = _make_domains_dir(tmp)  # tmp is the "skill root"
            found = bridge.find_skill_domains_dir(explicit=tmp)
            self.assertEqual(found, domains)

    def test_explicit_repo_root_is_expanded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Repo layout: <root>/skills/market-intel/reference/domains
            inner = Path(tmp) / "skills" / "market-intel"
            domains = _make_domains_dir(str(inner))
            found = bridge.find_skill_domains_dir(explicit=tmp)
            self.assertEqual(found, domains)

    def test_env_var_used_when_no_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            domains = _make_domains_dir(tmp)
            with patch.dict(os.environ, {"MARKET_INTEL_SKILL_DIR": str(domains)}, clear=False):
                found = bridge.find_skill_domains_dir()
            self.assertEqual(found, domains)

    def test_missing_anchor_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _make_domains_dir(tmp, with_anchor=False)  # dir exists but no frontier-research.md
            # Clear env + point HOME somewhere empty so the default probes miss too.
            with patch.dict(os.environ, {"MARKET_INTEL_SKILL_DIR": ""}, clear=False), \
                    patch.object(bridge.Path, "home", return_value=Path(tmp) / "nohome"):
                found = bridge.find_skill_domains_dir(explicit=tmp)
            self.assertIsNone(found)


class TestLoadSourceGuidance(unittest.TestCase):
    def test_returns_table_and_default_pick(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            domains = _make_domains_dir(tmp)
            out = bridge.load_source_guidance(("frontier-research",), explicit_dir=str(domains))
        self.assertIsNotNone(out)
        self.assertIn("arXiv API", out)
        self.assertIn("| source | route |", out)
        self.assertIn("Default pick", out)
        # The leading prose/lesson should NOT be pulled in (table-scoped).
        self.assertNotIn("Real-run lesson", out)

    def test_missing_shard_falls_back_then_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            domains = _make_domains_dir(tmp)  # only frontier-research present
            # Requesting only a non-existent domain -> no shard -> None.
            out = bridge.load_source_guidance(("does-not-exist",), explicit_dir=str(domains))
        self.assertIsNone(out)

    def test_no_skill_dir_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"MARKET_INTEL_SKILL_DIR": ""}, clear=False), \
                    patch.object(bridge.Path, "home", return_value=Path(tmp) / "nohome"):
                out = bridge.load_source_guidance(explicit_dir=str(Path(tmp) / "nope"))
        self.assertIsNone(out)

    def test_malformed_shard_falls_back_to_first_lines_no_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            domains = _make_domains_dir(tmp, extra={"x-twitter.md": _NO_TABLE_SHARD})
            out = bridge.load_source_guidance(
                ("x-twitter",), explicit_dir=str(domains)
            )
        # No table -> first-N-lines fallback; still non-None, no exception.
        self.assertIsNotNone(out)
        self.assertIn("prose about X", out)

    def test_truncates_to_max_chars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            domains = _make_domains_dir(tmp)
            out = bridge.load_source_guidance(
                ("frontier-research",), explicit_dir=str(domains), max_chars=50
            )
        self.assertIsNotNone(out)
        self.assertLessEqual(len(out), 50 + len("\n... (truncated)"))
        self.assertTrue(out.endswith("(truncated)"))


if __name__ == "__main__":
    unittest.main()
