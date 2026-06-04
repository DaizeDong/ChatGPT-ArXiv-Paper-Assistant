from __future__ import annotations

import configparser
import unittest
from pathlib import Path

from arxiv_assistant.utils.hotspot.hotspot_config import load_repo_config, load_reuse_config, repo_root


class TestHotspotConfig(unittest.TestCase):
    def test_repo_root_points_to_repository_root(self) -> None:
        root = repo_root()
        self.assertTrue((root / "configs" / "config.ini").exists())
        self.assertEqual(root.name, "ChatGPT-ArXiv-Paper-Assistant")

    def test_load_repo_config_reads_hotspot_sections(self) -> None:
        config = load_repo_config(Path("configs") / "config.ini")
        self.assertIn("HOTSPOTS", config.sections())
        self.assertTrue(config["HOTSPOTS"].getboolean("enabled"))


    def test_stage2_dedup_keys_present_with_defaults(self) -> None:
        cfg = configparser.ConfigParser()
        cfg.read(Path(__file__).resolve().parents[1] / "configs" / "config.ini")
        hot = cfg["HOTSPOTS"]
        self.assertEqual(hot.getint("cross_day_window_days"), 14)
        self.assertAlmostEqual(hot.getfloat("cross_day_cosine_threshold"), 0.72)
        self.assertEqual(hot.getint("resurge_min_competitors"), 3)
        self.assertEqual(hot.getint("resurge_cooldown_days"), 7)
        self.assertTrue(hot.get("embed_model_id", fallback="").strip())

    def test_load_reuse_config_reads_from_config_ini(self) -> None:
        cfg = configparser.ConfigParser()
        cfg.read(Path(__file__).resolve().parents[1] / "configs" / "config.ini")
        use, sources = load_reuse_config(cfg)
        self.assertIs(use, True)
        self.assertEqual(sources, ["hf_daily", "ainews", "agents_radar", "horizon", "scholar_inbox"])

    def test_load_reuse_config_fallback_when_section_missing(self) -> None:
        cfg = configparser.ConfigParser()
        # Config with no [HOTSPOT_REUSE] section at all
        use, sources = load_reuse_config(cfg)
        self.assertIs(use, True)
        self.assertEqual(sources, ["hf_daily", "ainews", "agents_radar", "horizon", "scholar_inbox"])

    def test_load_reuse_config_honours_use_reuse_layer_false(self) -> None:
        cfg = configparser.ConfigParser()
        cfg.read_dict({"HOTSPOT_REUSE": {"use_reuse_layer": "false", "reuse_sources": "hf_daily,ainews"}})
        use, sources = load_reuse_config(cfg)
        self.assertIs(use, False)
        self.assertEqual(sources, ["hf_daily", "ainews"])

    def test_load_reuse_config_strips_whitespace_from_sources(self) -> None:
        cfg = configparser.ConfigParser()
        cfg.read_dict({"HOTSPOT_REUSE": {"use_reuse_layer": "true", "reuse_sources": " hf_daily , ainews , agents_radar "}})
        _, sources = load_reuse_config(cfg)
        self.assertEqual(sources, ["hf_daily", "ainews", "agents_radar"])


if __name__ == "__main__":
    unittest.main()
