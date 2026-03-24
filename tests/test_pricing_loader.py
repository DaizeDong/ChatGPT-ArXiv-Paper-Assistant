import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from arxiv_assistant.utils.pricing_loader import (
    clear_cached_model_pricing,
    get_model_pricing,
    load_pricing_cache,
    normalize_model_pricing,
    refresh_model_pricing,
    save_pricing_cache,
)


class PricingLoaderTests(unittest.TestCase):
    def tearDown(self):
        clear_cached_model_pricing()

    def test_normalize_model_pricing_maps_token_costs_to_per_million(self):
        raw_table = {
            "sample_spec": {
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
            },
            "gpt-test": {
                "input_cost_per_token": 2.5e-06,
                "output_cost_per_token": 1e-05,
                "cache_read_input_token_cost": 1.25e-06,
            },
            "image-test": {
                "input_cost_per_token": None,
                "output_cost_per_token": None,
            },
        }

        normalized = normalize_model_pricing(raw_table)

        self.assertEqual(normalized["gpt-test"]["prompt"], 2.5)
        self.assertEqual(normalized["gpt-test"]["completion"], 10.0)
        self.assertEqual(normalized["gpt-test"]["cache"], 1.25)
        self.assertNotIn("image-test", normalized)

    def test_refresh_model_pricing_writes_cache_and_merges_static_fallback(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "pricing_cache.json"

            with (
                patch("arxiv_assistant.utils.pricing_loader.fetch_latest_pricing_commit_sha", return_value="abc123"),
                patch(
                    "arxiv_assistant.utils.pricing_loader.fetch_remote_model_pricing",
                    return_value={"test-model": {"prompt": 1.0, "completion": 2.0, "cache": 0.1}},
                ),
            ):
                payload = refresh_model_pricing(cache_path=cache_path)

            self.assertTrue(cache_path.exists())
            self.assertEqual(payload["commit_sha"], "abc123")
            self.assertEqual(payload["pricing_table"]["test-model"]["prompt"], 1.0)
            self.assertIn("gpt-5", payload["pricing_table"])
            self.assertIsNotNone(load_pricing_cache(cache_path))

    def test_get_model_pricing_uses_fresh_cache_without_refresh(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "pricing_cache.json"
            payload = {
                "fetched_at": datetime.now(UTC).isoformat(),
                "pricing_table": {"cached-model": {"prompt": 3.0, "completion": 4.0}},
            }
            save_pricing_cache(payload, cache_path)

            with patch("arxiv_assistant.utils.pricing_loader.refresh_model_pricing", side_effect=AssertionError("refresh should not run")):
                pricing = get_model_pricing(cache_path=cache_path, max_age_hours=24)

            self.assertEqual(pricing["cached-model"]["prompt"], 3.0)

    def test_refresh_model_pricing_force_download_ignores_matching_commit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "pricing_cache.json"
            save_pricing_cache(
                {
                    "fetched_at": datetime.now(UTC).isoformat(),
                    "commit_sha": "same-sha",
                    "pricing_table": {"cached-model": {"prompt": 3.0, "completion": 4.0}},
                },
                cache_path,
            )

            with (
                patch("arxiv_assistant.utils.pricing_loader.fetch_latest_pricing_commit_sha", return_value="same-sha"),
                patch(
                    "arxiv_assistant.utils.pricing_loader.fetch_remote_model_pricing",
                    return_value={"fresh-model": {"prompt": 7.0, "completion": 8.0}},
                ),
            ):
                payload = refresh_model_pricing(cache_path=cache_path, force_download=True)

            self.assertIn("fresh-model", payload["pricing_table"])

    def test_get_model_pricing_falls_back_to_cached_payload_on_refresh_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "pricing_cache.json"
            payload = {
                "fetched_at": (datetime.now(UTC) - timedelta(days=2)).isoformat(),
                "pricing_table": {"stale-model": {"prompt": 5.0, "completion": 6.0}},
            }
            save_pricing_cache(payload, cache_path)

            with patch("arxiv_assistant.utils.pricing_loader.refresh_model_pricing", side_effect=RuntimeError("boom")):
                pricing = get_model_pricing(cache_path=cache_path, max_age_hours=1)

            self.assertEqual(pricing["stale-model"]["completion"], 6.0)

    def test_get_model_pricing_falls_back_to_static_table_when_no_cache_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "pricing_cache.json"

            with patch("arxiv_assistant.utils.pricing_loader.refresh_model_pricing", side_effect=RuntimeError("boom")):
                pricing = get_model_pricing(cache_path=cache_path, max_age_hours=1)

            self.assertIn("gpt-5", pricing)


if __name__ == "__main__":
    unittest.main()
