import json
import tempfile
import unittest
from pathlib import Path

from arxiv_assistant.renderers.build_multipage_site import build_multipage_site
from arxiv_assistant.renderers.render_static_site import render_static_site


class BuildMultipageSiteTests(unittest.TestCase):
    def test_build_multipage_site_adds_hotspot_pages_and_cross_links(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out_root = Path(temp_dir) / "out"
            (out_root / "md" / "2026-03").mkdir(parents=True)
            (out_root / "hot" / "md" / "2026-03").mkdir(parents=True)
            (out_root / "hot" / "reports").mkdir(parents=True)

            (out_root / "md" / "2026-03" / "2026-03-19-output.md").write_text(
                "# Personalized Daily ArXiv Papers 2026-03-19\n\nPaper digest 19.",
                encoding="utf-8",
            )
            (out_root / "md" / "2026-03" / "2026-03-20-output.md").write_text(
                "# Personalized Daily ArXiv Papers 2026-03-20\n\nPaper digest 20.",
                encoding="utf-8",
            )

            (out_root / "hot" / "md" / "2026-03" / "2026-03-19-hotspots.md").write_text(
                "# Daily AI Hotspots 2026-03-19\n\nHotspot digest 19.",
                encoding="utf-8",
            )
            (out_root / "hot" / "md" / "2026-03" / "2026-03-20-hotspots.md").write_text(
                "# Daily AI Hotspots 2026-03-20\n\nHotspot digest 20.",
                encoding="utf-8",
            )

            for day, topics, watchlist, summary in [
                (19, 2, 1, "Agents and evals drove discussion."),
                (20, 3, 0, "Model releases and tooling led the day."),
            ]:
                payload = {
                    "date": f"2026-03-{day:02d}",
                    "summary": summary,
                    "top_topics": [{} for _ in range(topics)],
                    "watchlist": [{} for _ in range(watchlist)],
                }
                (out_root / "hot" / "reports" / f"2026-03-{day:02d}.json").write_text(
                    json.dumps(payload, indent=2),
                    encoding="utf-8",
                )

            site_root = build_multipage_site(out_root)

            self.assertIsNotNone(site_root)
            self.assertTrue((site_root / "hot" / "index.md").exists())
            self.assertTrue((site_root / "hot" / "2026-03-20" / "index.md").exists())
            self.assertTrue((site_root / "hot" / "2026-03" / "index.md").exists())
            self.assertTrue((site_root / "hot" / "2026" / "index.md").exists())

            paper_home = (site_root / "index.md").read_text(encoding="utf-8")
            paper_day = (site_root / "archive" / "2026-03" / "20" / "index.md").read_text(encoding="utf-8")
            hot_home = (site_root / "hot" / "index.md").read_text(encoding="utf-8")
            hot_month = (site_root / "hot" / "2026-03" / "index.md").read_text(encoding="utf-8")
            hot_year = (site_root / "hot" / "2026" / "index.md").read_text(encoding="utf-8")
            paper_month = (site_root / "archive" / "2026-03" / "index.md").read_text(encoding="utf-8")
            paper_year = (site_root / "archive" / "2026" / "index.md").read_text(encoding="utf-8")

            self.assertIn('href="hot/2026-03-20"', paper_home)
            self.assertIn("Daily AI Hotspots", paper_day)
            self.assertIn("Daily AI Hotspots", paper_home)
            self.assertIn("Daily AI Hotspots", paper_month)
            self.assertIn("Daily AI Hotspots", paper_year)
            self.assertIn("Personalized Daily Arxiv Paper", hot_home)
            self.assertIn("Personalized Daily Arxiv Paper", hot_month)
            self.assertIn("Personalized Daily Arxiv Paper", hot_year)
            self.assertIn("Daily Briefs", hot_month)
            self.assertIn("Report days", hot_month)
            self.assertIn("Months active", hot_year)
            self.assertIn("March", hot_year)

            css_source = Path(temp_dir) / "site.css"
            css_source.write_text("body { color: black; }", encoding="utf-8")
            dist_root = Path(temp_dir) / "dist"
            render_static_site(site_root, dist_root, css_source)

            self.assertTrue((dist_root / "hot" / "index.html").exists())
            self.assertTrue((dist_root / "hot" / "2026-03-20" / "index.html").exists())
            self.assertTrue((dist_root / "hot" / "2026-03" / "index.html").exists())

    def test_build_multipage_site_falls_back_to_hotspot_archive_routes_for_historical_paper_pages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out_root = Path(temp_dir) / "out"
            (out_root / "md" / "2025-03").mkdir(parents=True)
            (out_root / "hot" / "md" / "2026-03").mkdir(parents=True)
            (out_root / "hot" / "reports").mkdir(parents=True)

            (out_root / "md" / "2025-03" / "2025-03-03-output.md").write_text(
                "# Personalized Daily ArXiv Papers 2025-03-03\n\nHistorical paper digest.",
                encoding="utf-8",
            )
            (out_root / "hot" / "md" / "2026-03" / "2026-03-23-hotspots.md").write_text(
                "# Daily AI Hotspots 2026-03-23\n\nHotspot digest.",
                encoding="utf-8",
            )
            (out_root / "hot" / "reports" / "2026-03-23.json").write_text(
                json.dumps({"date": "2026-03-23", "summary": "One day", "top_topics": [{}], "watchlist": []}, indent=2),
                encoding="utf-8",
            )

            site_root = build_multipage_site(out_root)
            self.assertIsNotNone(site_root)

            paper_day = (site_root / "archive" / "2025-03" / "03" / "index.md").read_text(encoding="utf-8")
            paper_month = (site_root / "archive" / "2025-03" / "index.md").read_text(encoding="utf-8")
            paper_year = (site_root / "archive" / "2025" / "index.md").read_text(encoding="utf-8")

            self.assertIn('href="../../../hot/2026-03-23"', paper_day)
            self.assertIn("Daily AI Hotspots", paper_day)
            self.assertIn('href="../../hot/2026-03"', paper_month)
            self.assertIn("Daily AI Hotspots", paper_month)
            self.assertIn('href="../../hot/2026"', paper_year)
            self.assertIn("Daily AI Hotspots", paper_year)


if __name__ == "__main__":
    unittest.main()
