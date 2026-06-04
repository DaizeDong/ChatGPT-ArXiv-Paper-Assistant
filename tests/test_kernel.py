from __future__ import annotations

import json
import tempfile
import unittest
import unittest.mock
from datetime import datetime, timezone
from pathlib import Path

from arxiv_assistant.hotspots import kernel


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


import configparser


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
