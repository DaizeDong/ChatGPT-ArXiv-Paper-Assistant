from __future__ import annotations

import configparser
import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from arxiv_assistant.apis.hotspot.hotspot_ainews import _choose_best_anchor, _derive_segment_title
from arxiv_assistant.apis.hotspot.hotspot_github import fetch_hotspot_items as fetch_github_hotspot_items
from arxiv_assistant.apis.hotspot.hotspot_hn import fetch_hotspot_items as fetch_hn_hotspot_items
from arxiv_assistant.apis.hotspot.hotspot_local_papers import _resolve_best_source_path, fetch_hotspot_items as fetch_local_hotspot_items
from arxiv_assistant.apis.hotspot.hotspot_official_blogs import _extract_anthropic_rows
from arxiv_assistant.filters.filter_hotspots import _cluster_signal_scores, _digest_prompt_payload
from arxiv_assistant.renderers.hotspot.render_hot_daily import render_hot_daily_md
from arxiv_assistant.utils.hotspot.hotspot_cluster import build_hotspot_clusters
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotCluster, HotspotItem
from arxiv_assistant.hotspots.pipeline import _build_category_sections, _build_market_signal_items, _merge_display_candidates, _screening_queue, _trim_topics, detect_latest_local_output_date


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

    def test_local_papers_loads_hotspot_spotlight_companion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_root = Path(tmp_dir) / "json" / "2026-04"
            json_root.mkdir(parents=True, exist_ok=True)
            (json_root / "2026-04-01-output.json").write_text(
                json.dumps(
                    {
                        "2604.00001": {
                            "arxiv_id": "2604.00001",
                            "title": "Personalized paper",
                            "abstract": "Foundational paper.",
                            "authors": ["A"],
                            "PRIMARY_TOPIC_ID": "memory_systems",
                            "PRIMARY_TOPIC_LABEL": "Memory Structures and Agent Memory Systems",
                            "SCORE": 18,
                            "RELEVANCE": 9,
                            "NOVELTY": 9,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (json_root / "2026-04-01-hotspot-papers.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "papers": {
                            "2604.00002": {
                                "arxiv_id": "2604.00002",
                                "title": "Hot frontier paper",
                                "abstract": "Opens a new direction.",
                                "authors": ["B"],
                                "PRIMARY_TOPIC_ID": "architecture_training",
                                "PRIMARY_TOPIC_LABEL": "Architecture and Training Dynamics",
                                "SCORE": 17,
                                "RELEVANCE": 8,
                                "NOVELTY": 9,
                                "HOTSPOT_PAPER_TAGS": ["new_frontier"],
                                "HOTSPOT_PAPER_PRIMARY_KIND": "new_frontier",
                                "HOTSPOT_PAPER_PRIMARY_LABEL": "New Frontier Papers",
                                "HOTSPOT_PAPER_COMMENT": "Likely opens a new direction.",
                            }
                        },
                        "sections": [
                            {
                                "kind": "new_frontier",
                                "label": "New Frontier Papers",
                                "paper_ids": ["2604.00002"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            items = fetch_local_hotspot_items(datetime(2026, 4, 1, tzinfo=UTC), tmp_dir)

            self.assertEqual(len(items), 2)
            spotlight_item = next(item for item in items if item.source_id == "local_hotspot_papers")
            self.assertEqual(spotlight_item.source_role, "paper_trending")
            self.assertEqual(spotlight_item.metadata["spotlight_primary_kind"], "new_frontier")
            self.assertIn("New Frontier Papers", spotlight_item.tags)

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

    def test_cluster_does_not_merge_on_generic_ai_words_only(self) -> None:
        left = HotspotItem(
            source_id="hf_papers",
            source_name="HF",
            source_role="paper_trending",
            source_type="paper",
            title="MiroThinker: Pushing the Performance Boundaries of Open-Source Research Agents",
            summary="Research agent paper.",
            url="https://huggingface.co/papers/2511.11793",
            canonical_url="https://arxiv.org/abs/2511.11793",
            published_at="2026-03-20T12:00:00+00:00",
        )
        right = HotspotItem(
            source_id="hn_discussion",
            source_name="Hacker News",
            source_role="hn_discussion",
            source_type="discussion",
            title="OpenCode — Open source AI coding agent",
            summary="Open source coding agent discussion.",
            url="https://opencode.ai",
            canonical_url="https://opencode.ai",
            published_at="2026-03-20T12:30:00+00:00",
        )

        clusters = build_hotspot_clusters([left, right])

        self.assertEqual(len(clusters), 2)

    def test_cluster_does_not_merge_different_papers_with_similar_task_titles(self) -> None:
        left = HotspotItem(
            source_id="hf_papers",
            source_name="HF",
            source_role="paper_trending",
            source_type="paper",
            title="PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model",
            summary="Document parsing paper.",
            url="https://huggingface.co/papers/2510.14528",
            canonical_url="https://arxiv.org/abs/2510.14528",
            published_at="2026-03-20T12:00:00+00:00",
            metadata={"arxiv_id": "2510.14528"},
        )
        right = HotspotItem(
            source_id="hf_papers",
            source_name="HF",
            source_role="paper_trending",
            source_type="paper",
            title="MinerU2.5: A Decoupled Vision-Language Model for Efficient High-Resolution Document Parsing",
            summary="Another document parsing paper.",
            url="https://huggingface.co/papers/2509.22186",
            canonical_url="https://arxiv.org/abs/2509.22186",
            published_at="2026-03-20T12:30:00+00:00",
            metadata={"arxiv_id": "2509.22186"},
        )

        clusters = build_hotspot_clusters([left, right])

        self.assertEqual(len(clusters), 2)

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

    @patch("arxiv_assistant.apis.hotspot_github.fetch_json")
    def test_github_adapter_builds_repo_items(self, mock_fetch_json) -> None:
        mock_fetch_json.return_value = {
            "items": [
                {
                    "full_name": "acme/agent-kit",
                    "html_url": "https://github.com/acme/agent-kit",
                    "description": "Open-source agent toolkit.",
                    "stargazers_count": 420,
                    "forks_count": 42,
                    "language": "Python",
                    "topics": ["agents", "llm"],
                    "created_at": "2026-03-20T00:00:00Z",
                    "updated_at": "2026-03-21T00:00:00Z",
                    "owner": {"login": "acme"},
                }
            ]
        }

        items = fetch_github_hotspot_items(
            target_date=datetime(2026, 3, 21, tzinfo=UTC),
            search_queries=["agent framework"],
            stars_cutoff=20,
            created_within_days=10,
            result_limit=5,
        )

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].source_role, "github_trend")
        self.assertEqual(items[0].metadata["stars"], 420)
        self.assertIn("agent framework", mock_fetch_json.call_args.kwargs["params"]["q"])
        self.assertIn("stars:>=20", mock_fetch_json.call_args.kwargs["params"]["q"])

    @patch("arxiv_assistant.apis.hotspot_hn.fetch_json")
    def test_hn_adapter_filters_to_ai_relevant_story(self, mock_fetch_json) -> None:
        mock_fetch_json.side_effect = [
            [101, 102],
            {
                "id": 101,
                "type": "story",
                "title": "OpenAI launches a new agent benchmark",
                "score": 120,
                "descendants": 60,
                "time": int(datetime(2026, 3, 21, 12, tzinfo=UTC).timestamp()),
                "url": "https://openai.com/index/new-agent-benchmark",
                "by": "alice",
            },
            {
                "id": 102,
                "type": "story",
                "title": "Interesting startup discussion",
                "score": 25,
                "descendants": 3,
                "time": int(datetime(2026, 3, 21, 12, tzinfo=UTC).timestamp()),
                "url": "https://example.com/startup",
                "by": "bob",
            },
        ]

        items = fetch_hn_hotspot_items(
            target_date=datetime(2026, 3, 21, tzinfo=UTC),
            freshness_hours=36,
            keyword_filter=["openai", "agent"],
            story_limit=10,
            score_cutoff=30,
            comments_cutoff=8,
        )

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].source_role, "hn_discussion")
        self.assertEqual(items[0].metadata["hn_score"], 120)
        self.assertEqual(items[0].metadata["hn_comments"], 60)

    def test_detect_latest_local_output_date_uses_newest_daily_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_root = Path(temp_dir) / "out"
            (out_root / "json" / "2026-03").mkdir(parents=True)
            (out_root / "json" / "2026-03" / "2026-03-20-output.json").write_text("{}", encoding="utf-8")
            (out_root / "json" / "2026-03" / "2026-03-21-output.json").write_text("{}", encoding="utf-8")

            detected = detect_latest_local_output_date(out_root)

            self.assertIsNotNone(detected)
            self.assertEqual(detected.strftime("%Y-%m-%d"), "2026-03-21")

    def test_trim_topics_limits_isolated_research_slots(self) -> None:
        config = configparser.ConfigParser()
        config["HOTSPOTS"] = {
            "target_topics": "4",
            "min_topics": "3",
            "target_watchlist_topics": "2",
        }
        top_topics = [
            {
                "TOPIC_ID": "research-a",
                "PRIMARY_CATEGORY": "Research",
                "source_roles": ["paper_trending"],
                "source_names": ["Hugging Face Trending Papers"],
                "source_ids": ["hf_papers"],
                "FINAL_SCORE": 8.2,
                "EVIDENCE_STRENGTH": 4.0,
                "CROSS_SOURCE_RESONANCE": 3.0,
                "QUALITY": 8,
                "HEAT": 6,
                "IMPORTANCE": 7,
                "WATCHLIST": False,
            },
            {
                "TOPIC_ID": "research-b",
                "PRIMARY_CATEGORY": "Research",
                "source_roles": ["paper_trending"],
                "source_names": ["Hugging Face Trending Papers"],
                "source_ids": ["hf_papers"],
                "FINAL_SCORE": 7.9,
                "EVIDENCE_STRENGTH": 3.8,
                "CROSS_SOURCE_RESONANCE": 2.8,
                "QUALITY": 8,
                "HEAT": 5,
                "IMPORTANCE": 6,
                "WATCHLIST": False,
            },
            {
                "TOPIC_ID": "community-a",
                "PRIMARY_CATEGORY": "Industry Update",
                "source_roles": ["community_heat", "headline_consensus"],
                "source_names": ["AINews", "The Rundown AI"],
                "source_ids": ["ainews", "the_rundown_ai"],
                "FINAL_SCORE": 7.1,
                "EVIDENCE_STRENGTH": 6.2,
                "CROSS_SOURCE_RESONANCE": 6.5,
                "QUALITY": 6,
                "HEAT": 8,
                "IMPORTANCE": 6,
                "WATCHLIST": False,
            },
            {
                "TOPIC_ID": "tooling-a",
                "PRIMARY_CATEGORY": "Tooling",
                "source_roles": ["github_trend"],
                "source_names": ["GitHub Trending Repos"],
                "source_ids": ["github_trend"],
                "FINAL_SCORE": 6.9,
                "EVIDENCE_STRENGTH": 5.5,
                "CROSS_SOURCE_RESONANCE": 4.2,
                "QUALITY": 6,
                "HEAT": 6,
                "IMPORTANCE": 6,
                "WATCHLIST": False,
            },
        ]

        selected, watchlist = _trim_topics(top_topics, [], config)

        self.assertEqual([topic["TOPIC_ID"] for topic in selected], ["community-a"])
        self.assertTrue(
            {"research-a", "tooling-a"}.issubset({topic["TOPIC_ID"] for topic in watchlist})
        )

    def test_trim_topics_resorts_selected_featured_topics_by_priority(self) -> None:
        config = configparser.ConfigParser()
        config["HOTSPOTS"] = {
            "target_topics": "3",
            "min_topics": "3",
            "target_watchlist_topics": "2",
        }
        top_topics = [
            {
                "TOPIC_ID": "official-a",
                "PRIMARY_CATEGORY": "Product Release",
                "source_roles": ["official_news"],
                "source_names": ["OpenAI News"],
                "source_ids": ["openai_news"],
                "FINAL_SCORE": 6.4,
                "CONFIDENCE": 5.8,
                "DISPLAY_PRIORITY": 6.2,
                "EVIDENCE_STRENGTH": 5.0,
                "CROSS_SOURCE_RESONANCE": 4.4,
                "QUALITY": 6,
                "HEAT": 6,
                "IMPORTANCE": 8,
                "WATCHLIST": False,
            },
            {
                "TOPIC_ID": "research-a",
                "PRIMARY_CATEGORY": "Research",
                "source_roles": ["paper_trending", "headline_consensus"],
                "source_names": ["Hugging Face Trending Papers", "Superhuman AI"],
                "source_ids": ["hf_papers", "superhuman_ai"],
                "FINAL_SCORE": 6.1,
                "CONFIDENCE": 8.9,
                "DISPLAY_PRIORITY": 7.9,
                "EVIDENCE_STRENGTH": 6.6,
                "CROSS_SOURCE_RESONANCE": 6.0,
                "QUALITY": 7,
                "HEAT": 6,
                "IMPORTANCE": 7,
                "WATCHLIST": False,
            },
            {
                "TOPIC_ID": "community-a",
                "PRIMARY_CATEGORY": "Industry Update",
                "source_roles": ["community_heat", "headline_consensus"],
                "source_names": ["AINews", "The Rundown AI"],
                "source_ids": ["ainews", "the_rundown_ai"],
                "FINAL_SCORE": 5.9,
                "CONFIDENCE": 6.7,
                "DISPLAY_PRIORITY": 6.8,
                "EVIDENCE_STRENGTH": 5.8,
                "CROSS_SOURCE_RESONANCE": 6.3,
                "QUALITY": 6,
                "HEAT": 8,
                "IMPORTANCE": 6,
                "WATCHLIST": False,
            },
        ]

        selected, _ = _trim_topics(top_topics, [], config)

        self.assertEqual([topic["TOPIC_ID"] for topic in selected], ["research-a", "community-a", "official-a"])

    def test_trim_topics_demotes_low_confidence_single_source_featured_items(self) -> None:
        config = configparser.ConfigParser()
        config["HOTSPOTS"] = {
            "target_topics": "3",
            "min_topics": "3",
            "target_watchlist_topics": "3",
        }
        top_topics = [
            {
                "TOPIC_ID": "official-a",
                "PRIMARY_CATEGORY": "Product Release",
                "source_roles": ["official_news"],
                "source_names": ["OpenAI News"],
                "source_ids": ["openai_news"],
                "source_types": ["official"],
                "FINAL_SCORE": 6.2,
                "CONFIDENCE": 6.5,
                "DISPLAY_PRIORITY": 6.7,
                "EVIDENCE_STRENGTH": 5.8,
                "CROSS_SOURCE_RESONANCE": 5.1,
                "QUALITY": 6,
                "HEAT": 6,
                "IMPORTANCE": 7,
                "WATCHLIST": False,
            },
            {
                "TOPIC_ID": "research-a",
                "PRIMARY_CATEGORY": "Research",
                "source_roles": ["paper_trending"],
                "source_names": ["Hugging Face Trending Papers"],
                "source_ids": ["hf_papers"],
                "source_types": ["paper"],
                "FINAL_SCORE": 6.0,
                "CONFIDENCE": 6.1,
                "DISPLAY_PRIORITY": 6.1,
                "EVIDENCE_STRENGTH": 4.2,
                "CROSS_SOURCE_RESONANCE": 3.2,
                "QUALITY": 7,
                "HEAT": 5,
                "IMPORTANCE": 7,
                "WATCHLIST": False,
            },
            {
                "TOPIC_ID": "community-low",
                "PRIMARY_CATEGORY": "Industry Update",
                "source_roles": ["community_heat"],
                "source_names": ["AINews"],
                "source_ids": ["ainews"],
                "source_types": ["roundup"],
                "FINAL_SCORE": 6.9,
                "CONFIDENCE": 2.1,
                "DISPLAY_PRIORITY": 6.3,
                "EVIDENCE_STRENGTH": 3.4,
                "CROSS_SOURCE_RESONANCE": 4.0,
                "QUALITY": 6,
                "HEAT": 8,
                "IMPORTANCE": 7,
                "WATCHLIST": False,
            },
            {
                "TOPIC_ID": "tooling-strong",
                "PRIMARY_CATEGORY": "Tooling",
                "source_roles": ["builder_momentum", "headline_consensus"],
                "source_names": ["GitHub Trending Repos", "AINews"],
                "source_ids": ["github_trend", "ainews"],
                "source_types": ["repo", "roundup"],
                "FINAL_SCORE": 6.1,
                "CONFIDENCE": 5.6,
                "DISPLAY_PRIORITY": 6.0,
                "EVIDENCE_STRENGTH": 4.8,
                "CROSS_SOURCE_RESONANCE": 5.0,
                "QUALITY": 6,
                "HEAT": 6,
                "IMPORTANCE": 6,
                "WATCHLIST": False,
            },
            {
                "TOPIC_ID": "tooling-low",
                "PRIMARY_CATEGORY": "Tooling",
                "source_roles": ["github_trend"],
                "source_names": ["GitHub Trending Repos"],
                "source_ids": ["github_trend"],
                "source_types": ["repo"],
                "FINAL_SCORE": 6.0,
                "CONFIDENCE": 4.4,
                "DISPLAY_PRIORITY": 5.9,
                "EVIDENCE_STRENGTH": 3.0,
                "CROSS_SOURCE_RESONANCE": 3.2,
                "QUALITY": 6,
                "HEAT": 6,
                "IMPORTANCE": 6,
                "WATCHLIST": False,
            },
        ]

        selected, watchlist = _trim_topics(top_topics, [], config)

        self.assertEqual([topic["TOPIC_ID"] for topic in selected], ["official-a", "tooling-strong"])
        self.assertEqual(
            {topic["TOPIC_ID"] for topic in watchlist},
            {"research-a", "community-low", "tooling-low"},
        )
        self.assertTrue(all(topic.get("DEMOTED_LOW_CONFIDENCE") for topic in watchlist))

    def test_market_signals_filters_for_artifacts_and_authoritative_sources(self) -> None:
        raw_items = [
            HotspotItem(
                source_id="ainews",
                source_name="AINews",
                source_role="community_heat",
                source_type="roundup",
                title="Mistral raises $600M Series B at $6B valuation",
                summary="Major funding round for European AI startup.",
                url="https://www.reddit.com/r/MachineLearning/comments/example1",
                canonical_url="https://www.reddit.com/r/MachineLearning/comments/example1",
                published_at="2026-03-21T00:00:00+00:00",
                metadata={"activity": 900, "host": "reddit.com"},
            ),
            HotspotItem(
                source_id="openai_news",
                source_name="OpenAI News",
                source_role="official_news",
                source_type="official_blog",
                title="OpenAI launches GPT-5 Turbo with 2x context",
                summary="New model release with improved capabilities.",
                url="https://openai.com/index/gpt-5-turbo",
                canonical_url="https://openai.com/index/gpt-5-turbo",
                published_at="2026-03-21T00:00:00+00:00",
                metadata={"is_official": True},
            ),
            HotspotItem(
                source_id="hn_discussion",
                source_name="Hacker News",
                source_role="hn_discussion",
                source_type="discussion",
                title="What do you think about local LLM setups?",
                summary="Community opinion thread.",
                url="https://news.ycombinator.com/item?id=12345",
                canonical_url="https://news.ycombinator.com/item?id=12345",
                published_at="2026-03-21T00:00:00+00:00",
                metadata={"hn_score": 120},
            ),
        ]
        top_topics = [
            {
                "TOPIC_ID": "mistral",
                "HEADLINE": "Mistral funding",
                "items": [{"title": raw_items[0].title, "url": raw_items[0].url}],
            }
        ]

        signals = _build_market_signal_items(raw_items, top_topics, [], target_count=3, min_count=1)

        # Should include funding item (artifact match) and official item (authoritative)
        # Should exclude opinion thread (no artifact, not authoritative)
        signal_ids = {item["source_id"] for item in signals}
        self.assertIn("ainews", signal_ids)
        self.assertIn("openai_news", signal_ids)
        self.assertNotIn("hn_discussion", signal_ids)

    def test_category_sections_expand_coverage_without_repeating_featured_topics(self) -> None:
        candidate_topics = [
            {
                "TOPIC_ID": "featured-official",
                "title": "Official model release",
                "PRIMARY_CATEGORY": "Product Release",
                "source_roles": ["official_news", "headline_consensus"],
                "source_names": ["OpenAI News", "The Rundown AI"],
                "source_ids": ["openai_news", "the_rundown_ai"],
                "source_types": ["official_blog", "roundup"],
                "items": [{"title": "Official model release", "url": "https://example.com/release"}],
                "FINAL_SCORE": 8.8,
                "EVIDENCE_STRENGTH": 7.4,
                "CROSS_SOURCE_RESONANCE": 7.1,
                "QUALITY": 8,
                "HEAT": 8,
                "IMPORTANCE": 9,
                "FRONTIERNESS": 7.0,
            },
            {
                "TOPIC_ID": "tooling-1",
                "title": "Open-source agent runtime",
                "PRIMARY_CATEGORY": "Tooling",
                "source_roles": ["github_trend", "headline_consensus"],
                "source_names": ["GitHub Trending Repos", "The Neuron"],
                "source_ids": ["github_trend", "the_neuron"],
                "source_types": ["repo", "roundup"],
                "items": [{"title": "Open-source agent runtime", "url": "https://example.com/runtime"}],
                "FINAL_SCORE": 7.6,
                "EVIDENCE_STRENGTH": 6.3,
                "CROSS_SOURCE_RESONANCE": 6.0,
                "QUALITY": 7,
                "HEAT": 7,
                "IMPORTANCE": 7,
                "FRONTIERNESS": 5.0,
            },
            {
                "TOPIC_ID": "research-1",
                "title": "Reasoning benchmark paper",
                "PRIMARY_CATEGORY": "Research",
                "source_roles": ["paper_trending", "community_heat"],
                "source_names": ["Hugging Face Trending Papers", "AINews"],
                "source_ids": ["hf_papers", "ainews"],
                "source_types": ["paper", "roundup"],
                "items": [{"title": "Reasoning benchmark paper", "url": "https://example.com/paper"}],
                "FINAL_SCORE": 7.2,
                "EVIDENCE_STRENGTH": 5.8,
                "CROSS_SOURCE_RESONANCE": 5.6,
                "QUALITY": 8,
                "HEAT": 6,
                "IMPORTANCE": 7,
                "FRONTIERNESS": 5.5,
            },
            {
                "TOPIC_ID": "community-1",
                "title": "Debate over model provenance",
                "PRIMARY_CATEGORY": "Industry Update",
                "source_roles": ["community_heat", "hn_discussion"],
                "source_names": ["AINews", "Hacker News"],
                "source_ids": ["ainews", "hn_discussion"],
                "source_types": ["roundup", "discussion"],
                "items": [{"title": "Debate over model provenance", "url": "https://example.com/debate"}],
                "FINAL_SCORE": 6.8,
                "EVIDENCE_STRENGTH": 5.2,
                "CROSS_SOURCE_RESONANCE": 6.4,
                "QUALITY": 6,
                "HEAT": 8,
                "IMPORTANCE": 6,
                "FRONTIERNESS": 4.0,
            },
            {
                "TOPIC_ID": "generic-notice",
                "title": "Information Collection Notice",
                "PRIMARY_CATEGORY": "Industry Update",
                "source_roles": ["headline_consensus"],
                "source_names": ["Ben's Bites"],
                "source_ids": ["bens_bites"],
                "source_types": ["roundup"],
                "items": [{"title": "Information Collection Notice", "url": "https://example.com/notice"}],
                "FINAL_SCORE": 3.8,
                "EVIDENCE_STRENGTH": 5.0,
                "CROSS_SOURCE_RESONANCE": 5.4,
                "QUALITY": 4,
                "HEAT": 6,
                "IMPORTANCE": 4,
            },
            {
                "TOPIC_ID": "off-topic-news",
                "title": "Amazon's secret phone project",
                "PRIMARY_CATEGORY": "Industry Update",
                "source_roles": ["headline_consensus"],
                "source_names": ["The Rundown AI"],
                "source_ids": ["the_rundown_ai"],
                "source_types": ["roundup"],
                "items": [{"title": "Amazon's secret phone project", "url": "https://example.com/phone"}],
                "FINAL_SCORE": 3.6,
                "EVIDENCE_STRENGTH": 4.2,
                "CROSS_SOURCE_RESONANCE": 4.0,
                "QUALITY": 4,
                "HEAT": 5,
                "IMPORTANCE": 4,
                "summary": "A hardware story with little direct AI relevance.",
                "SHORT_COMMENT": "A hardware story with little direct AI relevance.",
            },
        ]
        screened_top = [{"TOPIC_ID": "featured-official", "KEEP_IN_DAILY_HOTSPOTS": True, "WATCHLIST": False}]
        screened_watchlist = [{"TOPIC_ID": "community-1", "KEEP_IN_DAILY_HOTSPOTS": False, "WATCHLIST": True}]

        display_candidates = _merge_display_candidates(candidate_topics, screened_top, screened_watchlist)
        display_lookup = {topic["TOPIC_ID"]: topic for topic in display_candidates}
        sections = _build_category_sections(
            display_candidates,
            [display_lookup["featured-official"]],
            target_total_topics=3,
            max_per_category=2,
            min_display_score=2.0,
        )

        self.assertEqual(display_lookup["featured-official"]["LLM_STATUS"], "featured")
        self.assertEqual(display_lookup["community-1"]["LLM_STATUS"], "watchlist")
        displayed_ids = {topic["TOPIC_ID"] for section in sections for topic in section["topics"]}
        self.assertNotIn("featured-official", displayed_ids)
        self.assertIn("tooling-1", displayed_ids)
        self.assertIn("research-1", displayed_ids)
        self.assertIn("community-1", displayed_ids)
        self.assertNotIn("generic-notice", displayed_ids)
        self.assertNotIn("off-topic-news", displayed_ids)

    def test_screening_queue_auto_routes_strong_and_weak_topics_away_from_llm(self) -> None:
        config = configparser.ConfigParser()
        config["HOTSPOTS"] = {
            "screening_auto_keep_score_cutoff": "6.8",
            "screening_auto_keep_confidence_cutoff": "6.4",
            "screening_auto_keep_evidence_cutoff": "5.4",
            "screening_auto_watchlist_score_cutoff": "4.6",
            "screening_auto_watchlist_confidence_cutoff": "5.4",
            "screening_auto_drop_score_cutoff": "2.8",
            "screening_auto_drop_confidence_cutoff": "3.6",
            "screening_auto_drop_evidence_cutoff": "3.2",
            "screening_heuristic_only_score_cutoff": "5.8",
            "screening_heuristic_only_confidence_cutoff": "5.0",
            "screening_heuristic_only_evidence_cutoff": "3.4",
        }
        candidate_topics = [
            {
                "TOPIC_ID": "official-release",
                "PRIMARY_CATEGORY": "Product Release",
                "source_roles": ["official_news", "headline_consensus"],
                "source_names": ["OpenAI News", "The Rundown AI"],
                "source_types": ["official", "roundup"],
                "FINAL_SCORE": 7.4,
                "CONFIDENCE": 7.2,
                "EVIDENCE_STRENGTH": 6.3,
                "CROSS_SOURCE_RESONANCE": 6.4,
                "HYPE_PENALTY": 2.0,
            },
            {
                "TOPIC_ID": "weak-roundup",
                "PRIMARY_CATEGORY": "Industry Update",
                "source_roles": ["headline_consensus"],
                "source_names": ["Newsletter"],
                "source_types": ["roundup"],
                "FINAL_SCORE": 2.1,
                "CONFIDENCE": 2.9,
                "EVIDENCE_STRENGTH": 2.6,
                "CROSS_SOURCE_RESONANCE": 2.4,
                "HYPE_PENALTY": 6.5,
            },
            {
                "TOPIC_ID": "borderline-tooling",
                "PRIMARY_CATEGORY": "Tooling",
                "source_roles": ["github_trend"],
                "source_names": ["GitHub Trending Repos"],
                "source_types": ["repo"],
                "FINAL_SCORE": 5.2,
                "CONFIDENCE": 4.8,
                "EVIDENCE_STRENGTH": 4.4,
                "CROSS_SOURCE_RESONANCE": 4.7,
                "HYPE_PENALTY": 3.4,
            },
            {
                "TOPIC_ID": "repo-tail",
                "PRIMARY_CATEGORY": "Tooling",
                "source_roles": ["github_trend"],
                "source_names": ["GitHub Trending Repos"],
                "source_types": ["repo"],
                "FINAL_SCORE": 5.1,
                "CONFIDENCE": 4.7,
                "EVIDENCE_STRENGTH": 3.0,
                "CROSS_SOURCE_RESONANCE": 3.5,
                "HYPE_PENALTY": 2.8,
            },
        ]

        auto_keep, auto_watch, review, heuristic_only, stats = _screening_queue(candidate_topics, 4.0, 3.3, config)

        self.assertEqual([topic["TOPIC_ID"] for topic in auto_keep], ["official-release"])
        self.assertEqual(auto_watch, [])
        self.assertEqual(review, [])
        self.assertEqual(heuristic_only, [])
        self.assertEqual(stats["auto_drop"], 3)
        self.assertEqual(stats["heuristic_only"], 0)

    def test_merge_display_candidates_uses_confidence_to_break_close_topics(self) -> None:
        candidate_topics = [
            {
                "TOPIC_ID": "high-confidence",
                "title": "High confidence topic",
                "PRIMARY_CATEGORY": "Tooling",
                "source_roles": ["github_trend", "headline_consensus"],
                "source_names": ["GitHub Trending Repos", "The Neuron"],
                "source_ids": ["github_trend", "the_neuron"],
                "source_types": ["repo", "roundup"],
                "items": [{"title": "High confidence topic", "url": "https://example.com/high"}],
                "FINAL_SCORE": 6.4,
                "CONFIDENCE": 7.4,
                "EVIDENCE_STRENGTH": 5.6,
                "CROSS_SOURCE_RESONANCE": 5.7,
                "QUALITY": 6,
                "HEAT": 6,
                "IMPORTANCE": 6,
            },
            {
                "TOPIC_ID": "low-confidence",
                "title": "Low confidence topic",
                "PRIMARY_CATEGORY": "Tooling",
                "source_roles": ["github_trend"],
                "source_names": ["GitHub Trending Repos"],
                "source_ids": ["github_trend"],
                "source_types": ["repo"],
                "items": [{"title": "Low confidence topic", "url": "https://example.com/low"}],
                "FINAL_SCORE": 6.5,
                "CONFIDENCE": 4.1,
                "EVIDENCE_STRENGTH": 4.2,
                "CROSS_SOURCE_RESONANCE": 4.5,
                "QUALITY": 6,
                "HEAT": 6,
                "IMPORTANCE": 6,
            },
        ]

        merged = _merge_display_candidates(candidate_topics, [], [])

        self.assertEqual(merged[0]["TOPIC_ID"], "high-confidence")

    def test_digest_prompt_payload_caps_evidence_and_drops_urls(self) -> None:
        payload = _digest_prompt_payload(
            [
                {
                    "TOPIC_ID": "topic-1",
                    "title": "Dense topic",
                    "PRIMARY_CATEGORY": "Tooling",
                    "QUALITY": 6,
                    "HEAT": 6,
                    "IMPORTANCE": 6,
                    "WHY_IT_MATTERS": "Useful update.",
                    "source_names": ["GitHub", "AINews"],
                    "items": [
                        {"source_name": "A", "title": "One", "url": "https://example.com/1"},
                        {"source_name": "B", "title": "Two", "url": "https://example.com/2"},
                        {"source_name": "C", "title": "Three", "url": "https://example.com/3"},
                        {"source_name": "D", "title": "Four", "url": "https://example.com/4"},
                    ],
                }
            ],
            [],
        )

        parsed = json.loads(payload)
        self.assertEqual(len(parsed["top_topics"][0]["EVIDENCE"]), 3)
        self.assertEqual(parsed["top_topics"][0]["EVIDENCE"][-1]["title"], "Three")
        self.assertNotIn('"url"', payload.lower())

    def test_render_hot_daily_includes_category_radar(self) -> None:
        report = {
            "date": "2026-03-21",
            "summary": "Strong multi-source AI discussion across releases, tooling, and research.",
            "source_stats": {"ainews": 3, "github_trend": 5},
            "totals": {"raw_items": 20, "clusters": 15},
            "paper_spotlight": [
                {
                    "kind": "new_frontier",
                    "label": "New Frontier Papers",
                    "description": "Papers that appear to open a genuinely new direction, paradigm, or field.",
                    "items": [
                        {
                            "title": "Hot frontier paper",
                            "url": "https://arxiv.org/abs/2603.00001",
                            "arxiv_id": "2603.00001",
                            "primary_topic_label": "Architecture and Training Dynamics",
                            "spotlight_comment": "Likely opens a new direction.",
                            "daily_score": 18,
                            "relevance": 9,
                            "novelty": 9,
                        }
                    ],
                }
            ],
            "featured_topics": [
                {
                    "TOPIC_ID": "featured-official",
                    "HEADLINE": "Official model release",
                    "PRIMARY_CATEGORY": "Product Release",
                    "FINAL_SCORE": 8.8,
                    "QUALITY": 8,
                    "HEAT": 8,
                    "IMPORTANCE": 9,
                    "source_names": ["OpenAI News", "The Rundown AI"],
                    "items": [{"title": "Official model release", "url": "https://example.com/release", "source_name": "OpenAI News"}],
                    "WHY_IT_MATTERS": "The release showed up in multiple trusted sources.",
                    "KEY_TAKEAWAYS": ["Broad ecosystem attention."],
                }
            ],
            "category_sections": [
                {
                    "category": "Tooling",
                    "total_candidates": 2,
                    "topics": [
                        {
                            "TOPIC_ID": "tooling-1",
                            "title": "Open-source agent runtime",
                            "PRIMARY_CATEGORY": "Tooling",
                            "FINAL_SCORE": 7.6,
                            "HEAT": 7,
                            "OCCURRENCE_SCORE": 6.4,
                            "source_names": ["GitHub Trending Repos", "The Neuron"],
                            "items": [{"title": "Open-source agent runtime", "url": "https://example.com/runtime"}],
                            "WHY_IT_MATTERS": "Builder momentum remained strong.",
                            "LLM_STATUS": "candidate",
                        }
                    ],
                }
            ],
            "long_tail_sections": [
                {
                    "category": "Industry Update",
                    "total_candidates": 1,
                    "topics": [
                        {
                            "TOPIC_ID": "tail-1",
                            "title": "Open-source AI coding agent",
                            "PRIMARY_CATEGORY": "Industry Update",
                            "FINAL_SCORE": 2.8,
                            "HEAT": 4,
                            "OCCURRENCE_SCORE": 3.9,
                            "source_names": ["Hacker News"],
                            "items": [{"title": "Open-source AI coding agent", "url": "https://example.com/opencode"}],
                            "LLM_STATUS": "candidate",
                        }
                    ],
                }
            ],
            "x_buzz": [],
            "watchlist": [],
        }

        rendered = render_hot_daily_md(report)

        self.assertIn("## Featured Topics", rendered)
        self.assertIn("## Paper Spotlight", rendered)
        self.assertIn("### New Frontier Papers (1)", rendered)
        self.assertIn("## Topic Radar By Category", rendered)
        self.assertIn("## Long-tail Signals", rendered)
        self.assertIn("### Tooling (1 shown / 2 candidates)", rendered)
        self.assertIn("### Industry Update (1 shown / 1 candidates)", rendered)
        self.assertIn("Open-source agent runtime", rendered)


if __name__ == "__main__":
    unittest.main()
