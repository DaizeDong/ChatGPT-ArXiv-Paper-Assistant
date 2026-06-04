from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch


class TestHotspotTwitterapiSource(unittest.TestCase):
    def _seed_file(self, tmp_dir: str) -> Path:
        seed_path = Path(tmp_dir) / "x_seeds.json"
        seed_path.write_text(
            json.dumps(
                {
                    "accounts": [
                        {"handle": "openai", "name": "OpenAI", "kind": "official", "tier": 3, "active": True},
                        {"handle": "demishassabis", "name": "Demis Hassabis", "kind": "researcher", "tier": 3, "active": True},
                    ]
                }
            ),
            encoding="utf-8",
        )
        return seed_path

    # -----------------------------------------------------------------------
    # Task 1: key gate
    # -----------------------------------------------------------------------

    def test_returns_empty_when_no_twitterapi_key_configured(self) -> None:
        from arxiv_assistant.apis.hotspot.hotspot_twitterapi import fetch_hotspot_items

        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(os.environ, {}, clear=True):
            seed_path = self._seed_file(tmp_dir)
            items = fetch_hotspot_items(
                datetime(2026, 3, 21, tzinfo=UTC),
                36,
                seed_path,
                snapshot_path=Path(tmp_dir) / "x_authority_inventory.json",
            )
        self.assertEqual(items, [])


if __name__ == "__main__":
    unittest.main()
