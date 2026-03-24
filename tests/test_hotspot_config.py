from __future__ import annotations

import unittest
from pathlib import Path

from arxiv_assistant.utils.hotspot.hotspot_config import load_repo_config, repo_root


class TestHotspotConfig(unittest.TestCase):
    def test_repo_root_points_to_repository_root(self) -> None:
        root = repo_root()
        self.assertTrue((root / "configs" / "config.ini").exists())
        self.assertEqual(root.name, "ChatGPT-ArXiv-Paper-Assistant")

    def test_load_repo_config_reads_hotspot_sections(self) -> None:
        config = load_repo_config(Path("configs") / "config.ini")
        self.assertIn("HOTSPOTS", config.sections())
        self.assertTrue(config["HOTSPOTS"].getboolean("enabled"))


if __name__ == "__main__":
    unittest.main()
