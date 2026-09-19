import configparser
import os
import shutil
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from arxiv_assistant.utils.config_loader import CONFIG_DIR, CONFIG_FILES, load_repo_config

PAPER_SECTIONS = ("SELECTION", "FILTERING", "PAPER_FILTER", "OUTPUT", "MONTHLY_SUMMARY", "READER", "LLM")
HOTSPOT_SECTIONS = ("HOTSPOTS", "HOTSPOT_SOURCES", "HOTSPOT_X", "HOTSPOT_GITHUB",
                    "HOTSPOT_HN", "HOTSPOT_REUSE", "HOTSPOT_RUNTIME")


class ConfigLoaderTests(unittest.TestCase):
    def test_both_files_merge_into_one_config(self):
        config = load_repo_config()
        for section in PAPER_SECTIONS + HOTSPOT_SECTIONS:
            self.assertIn(section, config, "%s went missing when the config was split" % section)

    def test_the_split_actually_separates_the_two_subsystems(self):
        # If everything drifted back into one file the split bought nothing, and
        # the paper pipeline's settings are buried again.
        paper = configparser.ConfigParser()
        paper.read(CONFIG_DIR / "config.ini", encoding="utf-8")
        hot = configparser.ConfigParser()
        hot.read(CONFIG_DIR / "hotspot.ini", encoding="utf-8")
        self.assertFalse([s for s in paper.sections() if s.startswith("HOTSPOT")])
        self.assertTrue(all(s.startswith("HOTSPOT") for s in hot.sections()))

    def test_a_missing_file_fails_loudly_rather_than_defaulting(self):
        # configparser answers a missing section with whatever the caller passed
        # as a fallback, so a lost file would change what the feed collects
        # without breaking anything visibly. Refusing to start is the point.
        for dropped in CONFIG_FILES:
            with tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp)
                for name in CONFIG_FILES:
                    if name != dropped:
                        shutil.copy(CONFIG_DIR / name, target / name)
                with self.assertRaises(FileNotFoundError, msg="%s may vanish silently" % dropped):
                    load_repo_config(target / CONFIG_FILES[0])

    def test_every_entry_point_sees_the_whole_config(self):
        # The three loaders that existed before this were not equivalent: one
        # read a relative path, one passed no encoding. Reading through any of
        # them must now give the same sections.
        from arxiv_assistant.environment import CONFIG
        from arxiv_assistant.hotspot.support.config import load_repo_config as hotspot_loader

        expected = set(load_repo_config().sections())
        self.assertEqual(set(CONFIG.sections()), expected)
        self.assertEqual(set(hotspot_loader().sections()), expected)


if __name__ == "__main__":
    unittest.main()
