from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.merge_hotspot_web_dist import merge_hotspot_web_dist


class TestHotspotWebPublish(unittest.TestCase):
    def test_merge_hotspot_web_dist_creates_hot_routes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            web_dist = root / "web-dist"
            site_dist = root / "site-dist"
            (web_dist / "assets").mkdir(parents=True, exist_ok=True)
            (web_dist / "web_data" / "hot").mkdir(parents=True, exist_ok=True)
            (site_dist / "hot" / "legacy").mkdir(parents=True, exist_ok=True)
            (site_dist / "hot" / "legacy" / "index.html").write_text("legacy", encoding="utf-8")

            (web_dist / "index.html").write_text("<html><body>app</body></html>", encoding="utf-8")
            (web_dist / "404.html").write_text("<html><body>404</body></html>", encoding="utf-8")
            (web_dist / "assets" / "app.js").write_text("console.log('ok')", encoding="utf-8")
            root_index = {
                "latest_date": "2026-03-21",
                "dates": [{"date": "2026-03-21"}],
                "months": [{"month": "2026-03"}],
                "years": [{"year": "2026"}],
            }
            day_payload = {
                "source_sections": [{"slug": "blogs"}],
                "topic_summary": [{"slug": "google-bets-on-vibe-design-with-stitch"}],
            }
            (web_dist / "web_data" / "hot" / "index.json").write_text(json.dumps(root_index), encoding="utf-8")
            (web_dist / "web_data" / "hot" / "2026-03-21.json").write_text(json.dumps(day_payload), encoding="utf-8")

            merge_hotspot_web_dist(web_dist, site_dist)

            self.assertFalse((site_dist / "hot" / "legacy").exists())
            self.assertTrue((site_dist / "assets" / "app.js").exists())
            self.assertTrue((site_dist / "web_data" / "hot" / "index.json").exists())
            self.assertTrue((site_dist / "hot" / "index.html").exists())
            self.assertTrue((site_dist / "hot" / "2026-03-21" / "index.html").exists())
            self.assertTrue((site_dist / "hot" / "2026-03-21" / "source" / "blogs" / "index.html").exists())
            self.assertTrue((site_dist / "hot" / "2026-03-21" / "topic" / "google-bets-on-vibe-design-with-stitch" / "index.html").exists())
            self.assertTrue((site_dist / "hot" / "404.html").exists())


if __name__ == "__main__":
    unittest.main()
