from __future__ import annotations

import configparser
import json
import tempfile
import unittest
import unittest.mock
from datetime import datetime, timezone
from pathlib import Path

from arxiv_assistant.hotspots import kernel
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem


class TestCheckpointIO(unittest.TestCase):
    def test_checkpoint_roundtrip_and_done_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            td = datetime(2026, 5, 20, tzinfo=timezone.utc)
            self.assertFalse(kernel._checkpoint_done(root, td, "harvest"))
            path = kernel._write_checkpoint(root, td, "harvest", {"items": [1, 2, 3]})
            self.assertTrue(path.exists())
            self.assertTrue(kernel._checkpoint_done(root, td, "harvest"))
            self.assertEqual(kernel._read_checkpoint(root, td, "harvest"), {"items": [1, 2, 3]})

    def test_checkpoint_path_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            td = datetime(2026, 5, 20, tzinfo=timezone.utc)
            p = kernel._checkpoint_path(root, td, "score")
            self.assertEqual(p, root / "hot" / "state" / "checkpoint" / "2026-05-20" / "score.json")

    def test_clear_checkpoints_removes_all_stages_for_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            td = datetime(2026, 5, 20, tzinfo=timezone.utc)
            kernel._write_checkpoint(root, td, "harvest", {})
            kernel._write_checkpoint(root, td, "score", {})
            kernel._clear_checkpoints(root, td)
            self.assertFalse(kernel._checkpoint_done(root, td, "harvest"))
            self.assertFalse(kernel._checkpoint_done(root, td, "score"))


class TestContextAndRetry(unittest.TestCase):
    def _config(self) -> configparser.ConfigParser:
        cfg = configparser.ConfigParser()
        cfg["HOTSPOTS"] = {"enabled": "true"}
        return cfg

    def test_context_run_date_is_target_utc_day(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = kernel.KernelContext(
                output_root=Path(tmp),
                target_date=datetime(2026, 5, 20, 13, 30, tzinfo=timezone.utc),
                config=self._config(),
                store=None,
                journal=[],
            )
            self.assertEqual(ctx.run_date, "2026-05-20")

    def test_retry_succeeds_after_transient_failures(self) -> None:
        calls = {"n": 0}

        def flaky() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("transient")
            return "ok"

        out = kernel._with_retry(flaky, attempts=3, base_delay=0.0)
        self.assertEqual(out, "ok")
        self.assertEqual(calls["n"], 3)

    def test_retry_degrades_to_fallback_after_exhaustion(self) -> None:
        def always_fail() -> str:
            raise RuntimeError("boom")

        out = kernel._with_retry(always_fail, attempts=3, base_delay=0.0, fallback=lambda: "degraded")
        self.assertEqual(out, "degraded")


class TestDagDriver(unittest.TestCase):
    def _config(self) -> configparser.ConfigParser:
        cfg = configparser.ConfigParser()
        cfg["HOTSPOTS"] = {"enabled": "true"}
        return cfg

    def test_stage_order_is_fixed(self) -> None:
        self.assertEqual(
            kernel.STAGES,
            ["harvest", "date_verify", "gravity_gate", "embed", "cluster",
             "storystore_match", "gapfill", "score", "synthesize", "render"],
        )

    def test_run_executes_stages_in_order_and_records_each(self) -> None:
        order: list[str] = []

        def make(stage_name: str):
            def _fn(ctx: kernel.KernelContext) -> dict:
                order.append(stage_name)
                return {"stage": stage_name}
            return _fn

        fns = {s: make(s) for s in kernel.STAGES}
        with tempfile.TemporaryDirectory() as tmp, \
                unittest.mock.patch.object(kernel, "_STAGE_FNS", fns):
            manifest = kernel.run(
                Path(tmp), datetime(2026, 5, 20, tzinfo=timezone.utc), self._config(),
            )
        self.assertEqual(order, kernel.STAGES)
        self.assertEqual(manifest["stages_run"], kernel.STAGES)
        self.assertEqual(manifest["date"], "2026-05-20")

    def test_resume_is_noop_on_completed_stages(self) -> None:
        order: list[str] = []

        def make(stage_name: str):
            def _fn(ctx: kernel.KernelContext) -> dict:
                order.append(stage_name)
                return {"stage": stage_name}
            return _fn

        fns = {s: make(s) for s in kernel.STAGES}
        with tempfile.TemporaryDirectory() as tmp, \
                unittest.mock.patch.object(kernel, "_STAGE_FNS", fns):
            root = Path(tmp)
            td = datetime(2026, 5, 20, tzinfo=timezone.utc)
            kernel.run(root, td, self._config())
            order.clear()
            manifest = kernel.run(root, td, self._config())  # second run, all cached
        self.assertEqual(order, [])  # nothing re-executed
        self.assertEqual(manifest["stages_run"], [])
        self.assertEqual(manifest["stages_skipped"], kernel.STAGES)

    def test_single_stage_run_executes_only_that_stage(self) -> None:
        order: list[str] = []
        fns = {s: (lambda ctx, n=s: (order.append(n), {"stage": n})[1]) for s in kernel.STAGES}
        with tempfile.TemporaryDirectory() as tmp, \
                unittest.mock.patch.object(kernel, "_STAGE_FNS", fns):
            kernel.run(Path(tmp), datetime(2026, 5, 20, tzinfo=timezone.utc),
                       self._config(), stage="score")
        self.assertEqual(order, ["score"])

    def test_force_clears_then_reruns(self) -> None:
        order: list[str] = []
        fns = {s: (lambda ctx, n=s: (order.append(n), {"stage": n})[1]) for s in kernel.STAGES}
        with tempfile.TemporaryDirectory() as tmp, \
                unittest.mock.patch.object(kernel, "_STAGE_FNS", fns):
            root = Path(tmp)
            td = datetime(2026, 5, 20, tzinfo=timezone.utc)
            kernel.run(root, td, self._config())
            order.clear()
            kernel.run(root, td, self._config(), force=True)
        self.assertEqual(order, kernel.STAGES)  # all re-run after force

    def test_unknown_stage_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            td = datetime(2026, 5, 20, tzinfo=timezone.utc)
            with self.assertRaises(ValueError):
                kernel.run(root, td, self._config(), stage="nonexistent")


# ---------------------------------------------------------------------------
# Task 4: TestHarvestStages
# ---------------------------------------------------------------------------

def _item(title: str, url: str, published_at: str) -> HotspotItem:
    return HotspotItem(
        source_id="hf_papers", source_name="HF", source_role="papers",
        source_type="papers", title=title, summary="s", url=url,
        canonical_url=url, published_at=published_at, tags=[], authors=[], metadata={},
    )


class TestHarvestStages(unittest.TestCase):
    def _config(self) -> configparser.ConfigParser:
        cfg = configparser.ConfigParser()
        cfg["HOTSPOTS"] = {"enabled": "true", "max_raw_items": "120", "max_item_age_days": "14"}
        return cfg

    def test_harvest_stage_serializes_items(self) -> None:
        items = [_item("A", "https://x/a", "2026-05-20T00:00:00Z")]
        with tempfile.TemporaryDirectory() as tmp, \
                unittest.mock.patch.object(
                    kernel, "_fetch_source_payloads",
                    return_value=(items, {"hf_papers": 1}, {})):
            ctx = kernel.KernelContext(
                output_root=Path(tmp),
                target_date=datetime(2026, 5, 20, tzinfo=timezone.utc),
                config=self._config(), store=None, journal=[],
            )
            payload = kernel._stage_harvest(ctx)
        self.assertEqual(payload["source_stats"], {"hf_papers": 1})
        self.assertEqual(len(payload["items"]), 1)
        self.assertEqual(payload["items"][0]["title"], "A")

    def test_deserialize_preserves_verified_first_date(self) -> None:
        # FIX 1 round-trip test: verified_first_date must survive serialize→deserialize
        original = _item("X", "https://x/x", "2026-01-01T00:00:00Z")
        original.verified_first_date = "2024-01-02T00:00:00Z"
        row = kernel._serialize_item(original)
        restored = kernel._deserialize_item(row)
        self.assertEqual(restored.verified_first_date, "2024-01-02T00:00:00Z")

    def test_gravity_gate_drops_items_older_than_max_age(self) -> None:
        # FIX 2: items here have verified_first_date set as date_verify stage would do,
        # reflecting the actual kernel flow. gate_date() uses verified_first_date as
        # the credible date anchor, so this tests the real anti-staleness mechanism.
        fresh = _item("fresh", "https://x/fresh", "2026-05-20T00:00:00Z")
        fresh.verified_first_date = "2026-05-19T00:00:00Z"  # recent
        stale = _item("stale", "https://x/stale", "2026-04-01T00:00:00Z")
        stale.verified_first_date = "2026-04-01T00:00:00Z"  # older than 14 days from target

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            td = datetime(2026, 5, 20, tzinfo=timezone.utc)
            kernel._write_checkpoint(root, td, "date_verify", {
                "items": [kernel._serialize_item(fresh), kernel._serialize_item(stale)],
            })
            ctx = kernel.KernelContext(output_root=root, target_date=td,
                                       config=self._config(), store=None, journal=[])
            payload = kernel._stage_gravity_gate(ctx)
        titles = {it["title"] for it in payload["items"]}
        self.assertIn("fresh", titles)
        self.assertNotIn("stale", titles)

    def test_gravity_gate_keeps_item_without_credible_date(self) -> None:
        # Canonical None=keep policy: item without verified_first_date and no metadata
        # anchors → gate_date returns None → do not drop.
        unknown = _item("unknown-date", "https://x/u", "2020-01-01T00:00:00Z")
        # no verified_first_date, no metadata anchors → gate_date returns None
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            td = datetime(2026, 5, 20, tzinfo=timezone.utc)
            kernel._write_checkpoint(root, td, "date_verify", {
                "items": [kernel._serialize_item(unknown)],
            })
            ctx = kernel.KernelContext(output_root=root, target_date=td,
                                       config=self._config(), store=None, journal=[])
            payload = kernel._stage_gravity_gate(ctx)
        self.assertEqual(len(payload["items"]), 1)


# ---------------------------------------------------------------------------
# Task 5: TestStoryStages
# ---------------------------------------------------------------------------

class TestStoryStages(unittest.TestCase):
    def _config(self) -> configparser.ConfigParser:
        cfg = configparser.ConfigParser()
        cfg["HOTSPOTS"] = {
            "enabled": "true", "mode": "heuristic", "target_topics": "5",
            "target_watchlist_topics": "3", "max_topics_per_category": "4",
            "cross_day_window_days": "14", "cross_day_cosine_threshold": "0.90",
        }
        return cfg

    def test_score_stage_emits_featured_and_watchlist(self) -> None:
        items = [
            _item("Big Model release", "https://x/a", "2026-05-20T00:00:00Z"),
            _item("Big Model release", "https://x/b", "2026-05-20T00:00:00Z"),
            _item("Other thing", "https://x/c", "2026-05-20T00:00:00Z"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            td = datetime(2026, 5, 20, tzinfo=timezone.utc)
            kernel._write_checkpoint(root, td, "gravity_gate",
                                     {"items": [kernel._serialize_item(i) for i in items]})
            ctx = kernel.KernelContext(output_root=root, target_date=td,
                                       config=self._config(), store=None, journal=[])
            # embed/cluster/storystore_match/gapfill pass-through in degraded mode
            kernel._write_checkpoint(root, td, "embed", kernel._stage_embed(ctx))
            kernel._write_checkpoint(root, td, "cluster", kernel._stage_cluster(ctx))
            kernel._write_checkpoint(root, td, "storystore_match", kernel._stage_storystore_match(ctx))
            kernel._write_checkpoint(root, td, "gapfill", kernel._stage_gapfill(ctx))
            payload = kernel._stage_score(ctx)
        self.assertIn("featured", payload)
        self.assertIn("watchlist", payload)
        self.assertIsInstance(payload["featured"], list)


FIXT = Path(__file__).resolve().parent / "fixtures" / "agent"

# ---------------------------------------------------------------------------
# Task 7: TestRenderStage
# ---------------------------------------------------------------------------


class TestRenderStage(unittest.TestCase):
    def _config(self) -> configparser.ConfigParser:
        cfg = configparser.ConfigParser()
        cfg["HOTSPOTS"] = {
            "enabled": "true", "mode": "heuristic", "max_item_age_days": "14",
            "resurge_min_competitors": "3", "resurge_cooldown_days": "7",
        }
        return cfg

    def test_render_writes_report_with_resurgence_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            td = datetime(2026, 5, 20, tzinfo=timezone.utc)
            kernel._write_checkpoint(root, td, "harvest",
                                     {"items": [], "source_stats": {}, "api_usage": {}})
            kernel._write_checkpoint(root, td, "synthesize",
                                     {"featured": [], "watchlist": [], "all_topics": [],
                                      "manifest": {"synthesize_model": "m", "synthesize_temperature": 0,
                                                   "synthesize_rejected": []}})
            ctx = kernel.KernelContext(output_root=root, target_date=td,
                                       config=self._config(), store=None, journal=[])
            payload = kernel._stage_render(ctx)
            report_path = root / "hot" / "reports" / "2026-05-20.json"
            self.assertTrue(report_path.exists())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn("resurgence", report)
            self.assertEqual(report["resurgence"], [])
            self.assertEqual(payload["report_path"], str(report_path))

    def test_resurgence_lane_built_from_store(self) -> None:
        class FakeStory:
            story_id = "s9"
            resurged_at = "2026-05-20"
            arxiv_versions = {"2301.00001": 3}
            surfaced_arxiv_versions = {"2301.00001": 2}
            verified_first_date = "2023-01-02T00:00:00Z"
            entity_names = {"FooNet"}

        class FakeStore:
            def __init__(self) -> None:
                self.surfaced: list = []

            def active_stories(self, window_days, as_of):
                return [FakeStory()]

            def record_surface(self, story, run_date, *, lane="featured"):
                self.surfaced.append((story.story_id, run_date, lane))

        store = FakeStore()
        with tempfile.TemporaryDirectory() as tmp, \
                unittest.mock.patch.object(kernel, "_resurge", return_value=True):
            root = Path(tmp)
            td = datetime(2026, 5, 20, tzinfo=timezone.utc)
            kernel._write_checkpoint(root, td, "harvest",
                                     {"items": [], "source_stats": {}, "api_usage": {}})
            kernel._write_checkpoint(root, td, "synthesize",
                                     {"featured": [], "watchlist": [], "all_topics": [],
                                      "manifest": {}})
            ctx = kernel.KernelContext(output_root=root, target_date=td,
                                       config=self._config(), store=store, journal=[])
            kernel._stage_render(ctx)
            report = json.loads((root / "hot" / "reports" / "2026-05-20.json").read_text(encoding="utf-8"))
        self.assertEqual(len(report["resurgence"]), 1)
        entry = report["resurgence"][0]
        self.assertEqual(entry["original_first_date"], "2023-01-02T00:00:00Z")
        self.assertEqual(entry["reason"], "arxiv_version_bump")
        self.assertIn(("s9", "2026-05-20", "resurgence"), store.surfaced)


class TestSynthesizeVerifier(unittest.TestCase):
    def _topic(self) -> dict:
        return {
            "TOPIC_ID": "t1", "title": "Old title",
            "WHY_IT_MATTERS": "", "KEY_TAKEAWAYS": [],
            "EVIDENCE_URLS": ["https://x/a", "https://x/b"],
            "items": [
                {"title": "Subitem about coding performance gains", "summary": "Research shows significant improvements in multi-step coding benchmarks."},
                {"title": "Benchmark results released for agentic tasks", "summary": "New evaluation suite reveals state-of-the-art results across diverse coding problems."},
            ],
        }

    def test_schema_check_rejects_missing_zh(self) -> None:
        row = {"TOPIC_ID": "t1", "headline_en": "h", "summary_en": "s",
               "headline_zh": "", "summary_zh": "", "evidence": ["https://x/a"]}
        self.assertFalse(kernel._synthesis_row_valid(row, {"https://x/a"}))

    def test_evidence_url_must_exist_in_story(self) -> None:
        row = {"TOPIC_ID": "t1", "headline_en": "h", "headline_zh": "标题",
               "summary_en": "s", "summary_zh": "摘要", "evidence": ["https://evil/x"]}
        self.assertFalse(kernel._synthesis_row_valid(row, {"https://x/a"}))

    def test_valid_row_passes(self) -> None:
        row = {"TOPIC_ID": "t1", "headline_en": "h", "headline_zh": "标题",
               "summary_en": "s", "summary_zh": "摘要", "evidence": ["https://x/a"]}
        self.assertTrue(kernel._synthesis_row_valid(row, {"https://x/a"}))

    def test_stage_applies_good_agent_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            td = datetime(2026, 5, 20, tzinfo=timezone.utc)
            kernel._write_checkpoint(root, td, "score",
                                     {"featured": [self._topic()], "watchlist": [], "all_topics": []})
            cfg = configparser.ConfigParser()
            cfg["HOTSPOTS"] = {"enabled": "true", "mode": "openai",
                               "model_synthesize": "pinned-model-v1"}
            ctx = kernel.KernelContext(output_root=root, target_date=td, config=cfg,
                                       store=None, journal=[])
            replay = json.loads((FIXT / "synthesize_ok.json").read_text(encoding="utf-8"))
            with unittest.mock.patch.object(kernel, "_call_synthesize_agent", return_value=replay):
                payload = kernel._stage_synthesize(ctx)
        topic = payload["featured"][0]
        self.assertEqual(topic["HEADLINE"], "Frontier lab ships agentic coding model")
        self.assertEqual(topic["HEADLINE_ZH"], "前沿实验室发布智能体编码模型")
        self.assertEqual(payload["manifest"]["synthesize_model"], "pinned-model-v1")
        self.assertEqual(payload["manifest"]["synthesize_temperature"], 0)

    def test_stage_rejects_hallucinated_url_and_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            td = datetime(2026, 5, 20, tzinfo=timezone.utc)
            topic = self._topic()
            kernel._write_checkpoint(root, td, "score",
                                     {"featured": [topic], "watchlist": [], "all_topics": []})
            cfg = configparser.ConfigParser()
            cfg["HOTSPOTS"] = {"enabled": "true", "mode": "openai",
                               "model_synthesize": "pinned-model-v1"}
            ctx = kernel.KernelContext(output_root=root, target_date=td, config=cfg,
                                       store=None, journal=[])
            replay = json.loads((FIXT / "synthesize_halluc.json").read_text(encoding="utf-8"))
            with unittest.mock.patch.object(kernel, "_call_synthesize_agent", return_value=replay):
                payload = kernel._stage_synthesize(ctx)
        out = payload["featured"][0]
        self.assertEqual(out["HEADLINE"], "Old title")          # original title kept
        self.assertTrue(out["KEY_TAKEAWAYS"])                    # heuristic fallback filled
        self.assertIn("t1", payload["manifest"]["synthesize_rejected"])


from arxiv_assistant.utils.hotspot.hotspot_web_data import build_daily_hotspot_web_payload


class TestResurgenceWebData(unittest.TestCase):
    def test_payload_carries_resurgence_section(self) -> None:
        report = {
            "date": "2026-05-20", "generated_at": "2026-05-20T00:00:00Z", "mode": "heuristic",
            "summary": "", "featured_topics": [], "category_sections": [],
            "long_tail_sections": [], "watchlist": [], "x_buzz": [], "paper_spotlight": [],
            "source_stats": {},
            "resurgence": [
                {"story_id": "s9", "original_first_date": "2023-01-02T00:00:00Z",
                 "resurged_at": "2026-05-20", "reason": "arxiv_version_bump",
                 "headline": "FooNet resurfaces with v3", "entities": ["FooNet"]},
            ],
        }
        payload = build_daily_hotspot_web_payload(report, [])
        self.assertEqual(len(payload["resurgence"]), 1)
        self.assertEqual(payload["resurgence"][0]["reason"], "arxiv_version_bump")
        self.assertEqual(payload["resurgence"][0]["original_first_date"], "2023-01-02T00:00:00Z")
        self.assertEqual(payload["meta"]["counts"]["resurgence"], 1)


from arxiv_assistant.renderers.hotspot.render_hot_daily import render_hot_daily_md


# ---------------------------------------------------------------------------
# Stage 6 Batch F0: TestKernelCrossDayParity
# E2E regression guard: _stage_score must suppress ONGOING stories, just like
# the legacy generate_daily_hotspot_report (pipeline.py:1769-1784).
# ---------------------------------------------------------------------------

class TestKernelCrossDayParity(unittest.TestCase):
    """Verify that kernel._stage_score suppresses ONGOING cross-day stories.

    Design note: cluster_intraday is mocked to return a Story with a
    pre-set centroid matching the store-seeded story. This bypasses the
    mpnet model load (which would make the test slow and require network
    access) while still exercising the REAL match_crossday → classify_cross_day
    → ONGOING suppression path inside _stage_score. The mock is applied at the
    dedup module level (where kernel imports from) so _stage_score's import
    resolves to the mock. record_surface must NOT be called for the ONGOING
    story (it is filtered before featured_stories is built).
    """

    def _config(self) -> configparser.ConfigParser:
        cfg = configparser.ConfigParser()
        cfg["HOTSPOTS"] = {
            "enabled": "true",
            "mode": "heuristic",
            "target_topics": "5",
            "target_watchlist_topics": "3",
            "max_topics_per_category": "4",
            "cross_day_cosine_threshold": "0.85",
            "cross_day_window_days": "14",
        }
        return cfg

    @staticmethod
    def _open_store(tmp_dir: str):
        from arxiv_assistant.hotspots.store import StoryStore
        from pathlib import Path
        db_path = Path(tmp_dir) / "hot" / "state" / "story_store.sqlite"
        return StoryStore(db_path)

    @staticmethod
    def _make_story_with_centroid(
        story_id: str,
        title: str,
        url: str,
        centroid: list,
        entity_names: set,
        status: str = "NEW",
        first_seen: str = "2026-04-10",
    ):
        """Build a Story with a pre-set centroid (bypasses mpnet embedding)."""
        from arxiv_assistant.hotspots.enrich import EnrichedItem
        from arxiv_assistant.hotspots.story import Story

        item = HotspotItem(
            source_id="official_news",
            source_name="TestSource",
            source_role="official_news",
            source_type="official_blog",
            title=title,
            summary=f"Summary of {title}.",
            url=url,
            canonical_url=url,
            published_at="2026-04-10T12:00:00+00:00",
            metadata={},
        )
        item.verified_first_date = "2026-04-10T00:00:00+00:00"
        ei = EnrichedItem(
            item=item,
            event_type="product_release",
            entities=[{"name": e} for e in entity_names],
            summary=item.summary,
            importance=8,
        )
        story = Story(
            story_id=story_id,
            canonical_item=ei,
            items=[ei],
            event_type="product_release",
            entity_names=set(entity_names),
            score=7.0,
        )
        story.centroid = list(centroid)
        story.centroid_model_id = "test-model"
        story.status = status
        story.first_seen = first_seen
        story.headline = title
        story.summary = item.summary
        return story

    def test_ongoing_story_suppressed_through_kernel_stage_score(self) -> None:
        """Day N story featured → day N+1 kernel._stage_score excludes ONGOING from featured.

        The test mocks cluster_intraday to avoid loading the mpnet model while
        still exercising the real match_crossday + classify_cross_day +
        record_surface suppression path inside _stage_score.
        """
        from datetime import date as _date
        from arxiv_assistant.hotspots.dedup import match_crossday, classify_cross_day

        with tempfile.TemporaryDirectory() as tmp:
            store = self._open_store(tmp)
            try:
                day_n = _date(2026, 4, 10)
                day_n1 = _date(2026, 4, 11)

                # Day N: seed the story as featured in the store.
                centroid = [1.0, 0.0, 0.0]
                story_n = self._make_story_with_centroid(
                    "story-abc",
                    "OpenAI Launches GPT-Next",
                    "https://openai.com/gpt-next",
                    centroid=centroid,
                    entity_names={"openai"},
                    status="NEW",
                    first_seen=day_n.isoformat(),
                )
                persisted, is_new = store.match_or_create(
                    story_n.centroid, story_n,
                    cosine_threshold=0.85, window_days=14, as_of=day_n,
                )
                self.assertTrue(is_new)
                store.record_surface(persisted, day_n.isoformat(), lane="featured")

                # Day N+1: write a gapfill checkpoint with the same-URL item.
                # cluster_intraday is mocked to return a story with the exact
                # same centroid so match_crossday can do a deterministic L2 match
                # without loading mpnet.
                root = Path(tmp)
                td = datetime(2026, 4, 11, tzinfo=timezone.utc)
                item_n1 = HotspotItem(
                    source_id="official_news",
                    source_name="TestSource",
                    source_role="official_news",
                    source_type="official_blog",
                    title="OpenAI Launches GPT-Next",
                    summary="Same story, second day.",
                    url="https://openai.com/gpt-next",
                    canonical_url="https://openai.com/gpt-next",
                    published_at="2026-04-11T12:00:00+00:00",
                    metadata={},
                )
                kernel._write_checkpoint(root, td, "gapfill", {
                    "items": [kernel._serialize_item(item_n1)],
                })

                # The story returned by cluster_intraday carries the same
                # centroid as the seeded story; same entity_names (no new entity
                # → T3 will NOT fire → ONGOING, not RESURFACE).
                story_n1 = self._make_story_with_centroid(
                    "story-xyz",
                    "OpenAI Launches GPT-Next",
                    "https://openai.com/gpt-next",
                    centroid=centroid,
                    entity_names={"openai"},
                    status="NEW",
                    first_seen=day_n1.isoformat(),
                )

                # Track record_surface calls; wrap the real implementation.
                surfaced_lanes: list[str] = []
                original_record_surface = store.record_surface

                def tracking_record_surface(story, run_date, *, lane="featured"):
                    surfaced_lanes.append(lane)
                    return original_record_surface(story, run_date, lane=lane)

                store.record_surface = tracking_record_surface

                cfg = self._config()
                ctx = kernel.KernelContext(
                    output_root=root,
                    target_date=td,
                    config=cfg,
                    store=store,
                    journal=[],
                )

                # Mock cluster_intraday (in dedup module) to bypass mpnet; the
                # rest of the Stage-2 path (match_crossday, classify_cross_day,
                # record_surface suppression) runs for real.
                import arxiv_assistant.hotspots.dedup as _dedup_mod
                with unittest.mock.patch.object(
                    _dedup_mod, "cluster_intraday", return_value=[story_n1]
                ):
                    payload = kernel._stage_score(ctx)

                featured_ids = {t.get("TOPIC_ID") for t in payload["featured"]}

                # NON-VACUOUS guard (FIX 1): the test seeds exactly ONE story,
                # which is ONGOING on day N+1. The REAL Stage-2 path suppresses
                # it → featured must be EMPTY. A naive group_into_stories revert
                # would keep the day-N+1 story (never matched to the store's
                # persistent id, never classified ONGOING) and feature it →
                # len(featured) == 1, FAILING this assertion. This is the line
                # that actually guards the cross-day-dedup regression.
                self.assertEqual(
                    len(payload["featured"]), 0,
                    "The sole ONGOING story must be suppressed → featured must be "
                    "empty (fails on naive group_into_stories revert)",
                )

                # Title/headline of the suppressed story must be absent too.
                featured_headlines = {
                    t.get("HEADLINE", t.get("title", "")) for t in payload["featured"]
                }
                self.assertNotIn("OpenAI Launches GPT-Next", featured_headlines)

                # The ONGOING story must NOT appear in featured output.
                self.assertNotIn(
                    persisted.story_id,
                    featured_ids,
                    "ONGOING story must be suppressed from kernel._stage_score featured output",
                )

                # record_surface must NOT have been called with lane="featured"
                # for the ONGOING story (it is filtered before selection).
                self.assertEqual(
                    surfaced_lanes.count("featured"), 0,
                    "record_surface(lane='featured') must NOT be called for an ONGOING story",
                )

            finally:
                store.close()


class TestResurgenceMarkdown(unittest.TestCase):
    def _base_report(self) -> dict:
        return {
            "date": "2026-05-20", "summary": "", "source_stats": {},
            "featured_topics": [], "category_sections": [], "long_tail_sections": [],
            "watchlist": [], "x_buzz": [], "paper_spotlight": [],
        }

    def test_no_resurgence_section_when_empty(self) -> None:
        md = render_hot_daily_md({**self._base_report(), "resurgence": []})
        self.assertNotIn("## Resurgence", md)

    def test_resurgence_section_renders_origin_and_reason(self) -> None:
        report = {**self._base_report(), "resurgence": [
            {"headline": "FooNet resurfaces with v3",
             "original_first_date": "2023-01-02T00:00:00Z",
             "reason": "arxiv_version_bump", "entities": ["FooNet"]},
        ]}
        md = render_hot_daily_md(report)
        self.assertIn("## Resurgence", md)
        self.assertIn("FooNet resurfaces with v3", md)
        self.assertIn("2023-01-02", md)            # original first date shown honestly
        self.assertIn("arxiv_version_bump", md)


# ---------------------------------------------------------------------------
# FIX A/B/C: TestRenderFixes — normalized file, totals/usage/costs, watchlist dedup
# ---------------------------------------------------------------------------

def _make_synth_checkpoint(
    featured: list | None = None,
    watchlist: list | None = None,
    all_topics: list | None = None,
) -> dict:
    return {
        "featured": featured or [],
        "watchlist": watchlist or [],
        "all_topics": all_topics or [],
        "manifest": {"synthesize_model": "m", "synthesize_temperature": 0, "synthesize_rejected": []},
    }


class TestRenderFixes(unittest.TestCase):
    def _config(self) -> configparser.ConfigParser:
        cfg = configparser.ConfigParser()
        cfg["HOTSPOTS"] = {
            "enabled": "true", "mode": "heuristic", "max_item_age_days": "14",
            "resurge_min_competitors": "3", "resurge_cooldown_days": "7",
        }
        return cfg

    def _write_harvest(self, root: Path, td: datetime, items: list, api_usage: dict | None = None) -> None:
        kernel._write_checkpoint(root, td, "harvest", {
            "items": [kernel._serialize_item(i) for i in items],
            "source_stats": {},
            "api_usage": api_usage or {},
        })

    # FIX A test
    def test_render_writes_normalized_items_file(self) -> None:
        """_stage_render must write out/hot/normalized/<date>.json as a list of serialized items."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            td = datetime(2026, 5, 20, tzinfo=timezone.utc)
            items = [_item("A", "https://x/a", "2026-05-20T00:00:00Z"),
                     _item("B", "https://x/b", "2026-05-20T00:00:00Z")]
            self._write_harvest(root, td, items)
            kernel._write_checkpoint(root, td, "synthesize", _make_synth_checkpoint())
            ctx = kernel.KernelContext(output_root=root, target_date=td,
                                       config=self._config(), store=None, journal=[])
            kernel._stage_render(ctx)

            normalized_path = root / "hot" / "normalized" / "2026-05-20.json"
            self.assertTrue(normalized_path.exists(),
                            "out/hot/normalized/2026-05-20.json must exist after _stage_render")
            data = json.loads(normalized_path.read_text(encoding="utf-8"))
            self.assertIsInstance(data, list, "normalized file must be a JSON list")
            self.assertEqual(len(data), 2, "normalized list must have one entry per raw item")

    # FIX B test
    def test_render_report_carries_totals_usage_costs(self) -> None:
        """Report written by _stage_render must carry totals/usage/costs with correct raw_items."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            td = datetime(2026, 5, 20, tzinfo=timezone.utc)
            items = [_item("A", "https://x/a", "2026-05-20T00:00:00Z"),
                     _item("B", "https://x/b", "2026-05-20T00:00:00Z"),
                     _item("C", "https://x/c", "2026-05-20T00:00:00Z")]
            self._write_harvest(root, td, items, api_usage={
                "x_twitterapi": {"provider": "twitterapi.io", "billing_model": "metered",
                                  "requests": 5, "items": 20, "estimated_cost": 0.01, "cache_hit": False},
            })
            kernel._write_checkpoint(root, td, "synthesize", _make_synth_checkpoint())
            ctx = kernel.KernelContext(output_root=root, target_date=td,
                                       config=self._config(), store=None, journal=[])
            kernel._stage_render(ctx)

            report = json.loads((root / "hot" / "reports" / "2026-05-20.json").read_text(encoding="utf-8"))
            # totals
            self.assertIn("totals", report, "report must carry 'totals' key")
            self.assertEqual(report["totals"]["raw_items"], 3,
                             "totals.raw_items must equal len(raw_items)")
            # usage
            self.assertIn("usage", report, "report must carry 'usage' key")
            self.assertIn("llm", report["usage"])
            self.assertIn("external", report["usage"])
            self.assertIn("x_twitterapi", report["usage"]["external"],
                          "external usage must propagate harvest api_usage")
            self.assertEqual(report["usage"]["external"]["x_twitterapi"]["requests"], 5)
            # costs
            self.assertIn("costs", report, "report must carry 'costs' key")
            self.assertIn("total", report["costs"])

    # FIX C test
    def test_render_watchlist_excludes_featured_topic_ids(self) -> None:
        """Watchlist must not contain any TOPIC_ID already present in featured_topics."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            td = datetime(2026, 5, 20, tzinfo=timezone.utc)
            self._write_harvest(root, td, [])

            # topic-1 appears in both featured and watchlist — after render it must
            # be stripped from watchlist (waterfall dedup / FIX C).
            shared_topic = {
                "TOPIC_ID": "topic-1", "title": "Shared", "HEADLINE": "Shared",
                "PRIMARY_CATEGORY": "Research", "score": 9.0,
                "source_roles": [], "items": [], "EVIDENCE_URLS": [],
            }
            watchlist_only = {
                "TOPIC_ID": "topic-2", "title": "WL only", "HEADLINE": "WL only",
                "PRIMARY_CATEGORY": "Research", "score": 5.0,
                "source_roles": [], "items": [], "EVIDENCE_URLS": [],
            }
            kernel._write_checkpoint(root, td, "synthesize", _make_synth_checkpoint(
                featured=[shared_topic],
                watchlist=[shared_topic, watchlist_only],
                all_topics=[shared_topic, watchlist_only],
            ))
            ctx = kernel.KernelContext(output_root=root, target_date=td,
                                       config=self._config(), store=None, journal=[])
            kernel._stage_render(ctx)

            report = json.loads((root / "hot" / "reports" / "2026-05-20.json").read_text(encoding="utf-8"))
            watchlist_ids = {t["TOPIC_ID"] for t in report["watchlist"]}
            featured_ids = {t["TOPIC_ID"] for t in report["featured_topics"]}
            self.assertNotIn("topic-1", watchlist_ids,
                             "topic-1 appears in featured_topics → must be removed from watchlist")
            self.assertIn("topic-2", watchlist_ids,
                          "topic-2 is watchlist-only → must remain in watchlist")
            self.assertIn("topic-1", featured_ids, "topic-1 must still be in featured_topics")


# ---------------------------------------------------------------------------
# Task 10: TestStrangler — generate_daily_hotspot_report delegates to kernel.run
# ---------------------------------------------------------------------------

from arxiv_assistant.hotspots import pipeline as hp


class TestStrangler(unittest.TestCase):
    def test_generate_delegates_to_kernel_and_returns_report(self) -> None:
        cfg = configparser.ConfigParser()
        cfg["HOTSPOTS"] = {"enabled": "true", "mode": "heuristic"}
        captured = {}

        def fake_run(output_root, target_date, config, *, stage=None, force=False):
            captured["called"] = True
            report_dir = Path(output_root) / "hot" / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / "2026-05-20.json").write_text(
                json.dumps({"date": "2026-05-20", "resurgence": []}), encoding="utf-8")
            return {"date": "2026-05-20", "stages_run": kernel.STAGES}

        with tempfile.TemporaryDirectory() as tmp, \
                unittest.mock.patch.object(hp, "kernel_run", fake_run):
            report = hp.generate_daily_hotspot_report(
                output_root=tmp,
                target_date=datetime(2026, 5, 20, tzinfo=timezone.utc),
                config=cfg, mode_override="heuristic", force=False)
        self.assertTrue(captured["called"])
        self.assertEqual(report["date"], "2026-05-20")

    def test_generate_returns_none_when_disabled(self) -> None:
        cfg = configparser.ConfigParser()
        cfg["HOTSPOTS"] = {"enabled": "false"}
        with tempfile.TemporaryDirectory() as tmp:
            out = hp.generate_daily_hotspot_report(
                output_root=tmp, target_date=datetime(2026, 5, 20, tzinfo=timezone.utc),
                config=cfg)
        self.assertIsNone(out)

    def test_auto_mode_resolved_via_decide_mode_before_delegating(self) -> None:
        """Regression guard: mode_override="auto" must NOT be handed to the kernel
        literally. It must be resolved through _decide_mode (which maps "auto" →
        "openai"/"heuristic" by OPENAI_API_KEY presence) and persisted into the
        config the kernel reads. Otherwise _stage_synthesize/_enrich silently fall
        through to heuristic even when an API key is present (silent quality regression)."""
        cfg = configparser.ConfigParser()
        cfg["HOTSPOTS"] = {"enabled": "true", "mode": "auto"}
        captured = {}

        def fake_run(output_root, target_date, config, *, stage=None, force=False):
            # Capture the mode the kernel actually receives.
            captured["mode"] = config["HOTSPOTS"]["mode"]
            report_dir = Path(output_root) / "hot" / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / "2026-05-20.json").write_text(
                json.dumps({"date": "2026-05-20"}), encoding="utf-8")
            return {"date": "2026-05-20"}

        with tempfile.TemporaryDirectory() as tmp, \
                unittest.mock.patch.object(hp, "kernel_run", fake_run), \
                unittest.mock.patch.object(hp, "_decide_mode", return_value="openai") as decided:
            hp.generate_daily_hotspot_report(
                output_root=tmp,
                target_date=datetime(2026, 5, 20, tzinfo=timezone.utc),
                config=cfg, mode_override="auto", force=False)

        # _decide_mode was consulted to resolve "auto"...
        decided.assert_called_once_with("auto")
        # ...and the RESOLVED value (not the literal "auto") is what the kernel sees.
        self.assertEqual(captured["mode"], "openai")
        self.assertEqual(cfg["HOTSPOTS"]["mode"], "openai")


# ---------------------------------------------------------------------------
# Task 12: TestSourceFailureDegrade — single-source failure degrades gracefully
# ---------------------------------------------------------------------------

class TestSourceFailureDegrade(unittest.TestCase):
    def _config(self) -> configparser.ConfigParser:
        cfg = configparser.ConfigParser()
        cfg["HOTSPOTS"] = {"enabled": "true", "mode": "heuristic", "max_raw_items": "120",
                           "max_item_age_days": "14", "target_topics": "5",
                           "target_watchlist_topics": "3", "max_topics_per_category": "4"}
        return cfg

    def test_run_completes_when_one_source_payload_partial(self) -> None:
        ok = [_item("Live story", "https://x/a", "2026-05-20T00:00:00Z")]
        # Simulate fetch_source_payloads having already swallowed a failing adapter:
        # returns only the surviving items + a partial source_stats row of 0.
        with tempfile.TemporaryDirectory() as tmp, \
                unittest.mock.patch.object(
                    kernel, "_fetch_source_payloads",
                    return_value=(ok, {"hf_papers": 1, "reddit": 0}, {})):
            manifest = kernel.run(Path(tmp), datetime(2026, 5, 20, tzinfo=timezone.utc),
                                  self._config())
        self.assertEqual(manifest["stages_run"], kernel.STAGES)  # no crash, all stages done
