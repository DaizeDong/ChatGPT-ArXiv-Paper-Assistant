from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from arxiv_assistant.apis.hotspot_ainews import _choose_best_anchor, _derive_segment_title
from arxiv_assistant.apis.hotspot_local_papers import _resolve_best_source_path
from arxiv_assistant.apis.hotspot_official_blogs import _extract_anthropic_rows
from arxiv_assistant.filters.filter_hotspots import _cluster_signal_scores
from arxiv_assistant.utils.hotspot_cluster import build_hotspot_clusters
from arxiv_assistant.utils.hotspot_schema import HotspotCluster, HotspotItem


class TestHotspotPipeline(unittest.TestCase):
    def test_local_papers_uses_latest_available_daily_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_root = Path(tmp_dir) / "json"
            march_dir = json_root / "2026-03"
            march_dir.mkdir(parents=True, exist_ok=True)
            older = march_dir / "2026-03-18-output.json"
            target = march_dir / "2026-03-20-output.json"
            newer = march_dir / "2026-03-21-output.json"
            for path in (older, target, newer):
                path.write_text("{}", encoding="utf-8")

            resolved = _resolve_best_source_path(datetime(2026, 3, 20, tzinfo=UTC), json_root)
            self.assertIsNotNone(resolved)
            resolved_date, resolved_path = resolved
            self.assertEqual(resolved_date.isoformat(), "2026-03-20")
            self.assertEqual(resolved_path, target)

    def test_ainews_prefers_non_media_external_anchor(self) -> None:
        segment_html = """
        <p>
          Prompt Master keeps Claude prompting focused. (Activity: 728):
          <a href="https://i.redd.it/demo.png">View Image</a>
          <a href="https://github.com/example/prompt-master">GitHub Repository</a>
          <a href="https://www.reddit.com/r/ClaudeAI/comments/abc123/example">Reddit thread</a>
        </p>
        """
        self.assertEqual(
            _derive_segment_title("Prompt Master keeps Claude prompting focused. (Activity: 728): details here"),
            "Prompt Master keeps Claude prompting focused.",
        )
        self.assertEqual(_choose_best_anchor(segment_html), "https://github.com/example/prompt-master")

    def test_anthropic_html_extracts_date_title_and_summary(self) -> None:
        html = """
        <html><body>
          <a href="/news/claude-sonnet-4-6">
            <div class="meta"><span>Product</span><time datetime="2026-02-17">Feb 17, 2026</time></div>
            <h4>Introducing Claude Sonnet 4.6</h4>
            <p>Sonnet 4.6 delivers frontier performance across coding and agents.</p>
          </a>
        </body></html>
        """
        rows = _extract_anthropic_rows(html, "https://www.anthropic.com/news")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["title"], "Introducing Claude Sonnet 4.6")
        self.assertTrue(rows[0]["published_at"].startswith("2026-02-17"))
        self.assertIn("frontier performance", rows[0]["summary"])

    def test_cluster_merges_items_with_same_github_repo(self) -> None:
        repo_url = "https://github.com/example/agent-kit"
        hf_item = HotspotItem(
            source_id="hf_papers",
            source_name="HF",
            source_role="paper_trending",
            source_type="paper",
            title="AgentKit: Tooling for Robust Agents",
            summary="A paper with code.",
            url="https://huggingface.co/papers/2603.12345",
            canonical_url="https://arxiv.org/abs/2603.12345",
            published_at="2026-03-20T12:00:00+00:00",
            metadata={"arxiv_id": "2603.12345", "github_url": repo_url, "upvotes": 120},
        )
        community_item = HotspotItem(
            source_id="ainews",
            source_name="AINews",
            source_role="community_heat",
            source_type="roundup",
            title="AgentKit repo is blowing up today",
            summary="Community discussion around the released repo.",
            url=repo_url,
            canonical_url=repo_url,
            published_at="2026-03-20T15:00:00+00:00",
            metadata={"activity": 900},
        )
        clusters = build_hotspot_clusters([hf_item, community_item])
        self.assertEqual(len(clusters), 1)
        self.assertEqual(sorted(clusters[0].source_ids), ["ainews", "hf_papers"])

    def test_official_release_scores_as_meaningful_watchlist_or_better(self) -> None:
        item = HotspotItem(
            source_id="openai_news",
            source_name="OpenAI News",
            source_role="official_news",
            source_type="official_blog",
            title="OpenAI to acquire Astral",
            summary="Accelerates Codex growth to power the next generation of Python developer tools.",
            url="https://openai.com/index/openai-to-acquire-astral",
            canonical_url="https://openai.com/index/openai-to-acquire-astral",
            published_at="2026-03-19T00:00:00+00:00",
            metadata={"is_official": True},
        )
        cluster = HotspotCluster(
            cluster_id="official1",
            title=item.title,
            canonical_url=item.canonical_url,
            summary=item.summary,
            items=[item.to_dict()],
            source_ids=[item.source_id],
            source_names=[item.source_name],
            source_roles=[item.source_role],
            source_types=[item.source_type],
            tags=[],
            published_at=item.published_at,
            deterministic_score=10.0,
        )
        signals = _cluster_signal_scores(cluster)
        self.assertGreaterEqual(signals["FINAL_SCORE"], 3.6)
        self.assertGreaterEqual(signals["IMPORTANCE"], 5)


if __name__ == "__main__":
    unittest.main()
