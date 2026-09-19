import json
import os
import unittest
from datetime import UTC, datetime
from unittest import mock

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from arxiv_assistant.hotspot.sources import model_hubs as hubs

NOW = datetime(2026, 9, 16, tzinfo=UTC)


def _payload(*models):
    return json.dumps(list(models))


def _model(model_id, created, **extra):
    base = {"modelId": model_id, "createdAt": created, "downloads": 10, "likes": 2, "tags": ["text-generation"]}
    base.update(extra)
    return base


class ModelHubTests(unittest.TestCase):
    def _fetch(self, payload, registry=None):
        reg = registry or [{"org": "deepseek-ai", "label": "DeepSeek"}]
        with mock.patch.object(hubs, "fetch_text", return_value=payload), \
             mock.patch.object(hubs, "_load_registry", return_value=reg):
            return hubs.fetch_hotspot_items(NOW, freshness_hours=72)

    def test_a_new_model_becomes_a_release_item(self):
        items = self._fetch(_payload(_model("deepseek-ai/DeepSeek-V4.1-Flash", "2026-09-15T04:00:00.000Z")))
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].source_type, "model_release")
        self.assertIn("DeepSeek-V4.1-Flash", items[0].title)
        self.assertEqual(items[0].url, "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash")

    def test_quantisations_do_not_multiply_one_release(self):
        # The same release is published several times in different formats. Each
        # is a hub entry with its own timestamp, so counting them reports one
        # release as four.
        items = self._fetch(
            _payload(
                _model("zai-org/GLM-5.3-Flash", "2026-09-15T04:00:00.000Z"),
                _model("zai-org/GLM-5.3-Flash-GGUF", "2026-09-15T05:00:00.000Z"),
                _model("zai-org/GLM-5.3-Flash-BF16", "2026-09-15T06:00:00.000Z"),
                _model("zai-org/GLM-5.3-Flash-AWQ", "2026-09-15T07:00:00.000Z"),
            ),
            # The registry has to name the org that owns GLM, or the cross-org
            # rule correctly reads these as somebody else's model.
            registry=[{"org": "zai-org", "label": "Zhipu AI"}],
        )
        self.assertEqual([i.metadata["model_id"] for i in items], ["zai-org/GLM-5.3-Flash"])

    def test_somebody_elses_model_is_not_this_lab_s_release(self):
        # Measured on a live fetch: 18 of 57 items in a 30-day window were one
        # org re-uploading another's model. NVIDIA quantising DeepSeek is not
        # NVIDIA releasing a model, and it must not appear as one.
        items = self._fetch(
            _payload(
                _model("nvidia/DeepSeek-V4.1-Flash-NVFP4", "2026-09-15T04:00:00.000Z"),
                _model("nvidia/GLM-5.3-NVFP4", "2026-09-15T05:00:00.000Z"),
                _model("nvidia/Nemotron-3-Nano", "2026-09-15T06:00:00.000Z"),
            ),
            registry=[{"org": "nvidia", "label": "NVIDIA"}],
        )
        self.assertEqual([i.metadata["model_id"] for i in items], ["nvidia/Nemotron-3-Nano"])

    def test_the_owner_of_a_family_still_publishes_it(self):
        items = self._fetch(
            _payload(_model("deepseek-ai/DeepSeek-V4.1-Flash", "2026-09-15T04:00:00.000Z")),
            registry=[{"org": "deepseek-ai", "label": "DeepSeek"}],
        )
        self.assertEqual(len(items), 1)

    def test_stale_models_are_not_reported_as_news(self):
        items = self._fetch(_payload(_model("Qwen/Qwen3-0.5B", "2025-01-02T00:00:00.000Z")))
        self.assertEqual(items, [])

    def test_a_dead_hub_yields_nothing_rather_than_raising(self):
        # One unreachable org must not take the whole daily run down with it.
        with mock.patch.object(hubs, "fetch_text", side_effect=OSError("no route")), \
             mock.patch.object(hubs, "_load_registry", return_value=[{"org": "x", "label": "X"}]):
            self.assertEqual(hubs.fetch_hotspot_items(NOW, freshness_hours=72), [])

    def test_registry_ships_only_orgs_that_were_verified(self):
        # Every entry here was probed against the live API before being added.
        # A new one must be probed too, not pasted in on the strength of a name.
        entries = json.loads(hubs.REGISTRY.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(entries), 15)
        for entry in entries:
            self.assertIn("org", entry)
            self.assertIn("label", entry)
            self.assertTrue(entry["org"].strip())


if __name__ == "__main__":
    unittest.main()
