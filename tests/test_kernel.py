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
