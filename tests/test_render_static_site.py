import tempfile
import unittest
from pathlib import Path

from arxiv_assistant.renderers.render_daily_with_link import render_daily_md_with_hyperlink
from arxiv_assistant.renderers.render_hot_daily_with_link import render_hot_daily_md_with_hyperlink
from arxiv_assistant.renderers.render_static_site import render_static_site


class RenderStaticSiteTests(unittest.TestCase):
    def test_render_static_site_converts_markdown_and_copies_assets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            site_root = temp_root / "site"
            css_source = temp_root / "site.css"
            dist_root = temp_root / "dist"

            (site_root / "archive" / "2026-03" / "13").mkdir(parents=True)
            (site_root / "assets").mkdir(parents=True)

            (site_root / "index.md").write_text("# Home\n\n[Daily](archive/2026-03/13)", encoding="utf-8")
            (site_root / "archive" / "2026-03" / "13" / "index.md").write_text(
                "\n".join(
                    [
                        '<div align="center"><a href="..">Monthly Overview</a></div>',
                        "# Daily Page",
                        "",
                        '1. [Paper](#user-content-link1)',
                        "",
                        '## 1. [Paper](https://arxiv.org/abs/1234.5678) <a id="link1"></a>',
                    ]
                ),
                encoding="utf-8",
            )
            (site_root / "assets" / "nav.svg").write_text("<svg></svg>", encoding="utf-8")
            css_source.write_text("body { color: black; }", encoding="utf-8")

            output_path = render_static_site(site_root, dist_root, css_source)

            self.assertEqual(output_path, dist_root)
            self.assertTrue((dist_root / "site.css").exists())
            self.assertTrue((dist_root / "assets" / "nav.svg").exists())

            home_html = (dist_root / "index.html").read_text(encoding="utf-8")
            daily_html = (dist_root / "archive" / "2026-03" / "13" / "index.html").read_text(encoding="utf-8")

            self.assertIn("<title>Home</title>", home_html)
            self.assertIn('href="../../../site.css"', daily_html)
            self.assertIn('rel="icon"', daily_html)
            self.assertIn('data:image/svg+xml,', daily_html)
            self.assertIn('href="#link1"', daily_html)
            self.assertIn('id="link1"', daily_html)
            self.assertIn("Toggle dark mode", daily_html)

    def test_daily_header_keeps_three_column_layout_at_boundary_dates(self):
        rendered = render_daily_md_with_hyperlink(
            now_date=(2026, 3, 20),
            current_page_path="archive/2026-03/20/index.md",
            all_dates={(2026, 3, 19), (2026, 3, 20)},
            previous_asset_path="assets/nav/day/2026-03-20-prev.svg",
            center_asset_path="assets/nav/day/2026-03-20-center.svg",
            next_asset_path=None,
            content_string="# Personalized Daily ArXiv Papers 2026-03-20",
        )

        self.assertIn('width: 33.33%; text-align: left;', rendered)
        self.assertIn('width: 33.33%; text-align: center;', rendered)
        self.assertIn('width: 33.33%; text-align: right;', rendered)
        self.assertIn('Monthly Overview 2026-03', rendered)

    def test_hotspot_header_keeps_three_column_layout_at_boundary_dates(self):
        rendered = render_hot_daily_md_with_hyperlink(
            now_date=(2026, 3, 20),
            current_page_path="hot/2026-03-20/index.md",
            all_dates={(2026, 3, 19), (2026, 3, 20)},
            previous_asset_path="assets/nav/hot/day/2026-03-20-prev.svg",
            center_asset_path="assets/nav/hot/day/2026-03-20-center.svg",
            next_asset_path=None,
            related_page_path="archive/2026-03/20/index.md",
            content_string="# Daily AI Hotspots 2026-03-20",
        )

        self.assertIn('width: 33.33%; text-align: left;', rendered)
        self.assertIn('width: 33.33%; text-align: center;', rendered)
        self.assertIn('width: 33.33%; text-align: right;', rendered)
        self.assertIn("Monthly Hotspots 2026-03", rendered)
        self.assertIn("Daily Paper Digest", rendered)


if __name__ == "__main__":
    unittest.main()
