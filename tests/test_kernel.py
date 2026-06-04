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
