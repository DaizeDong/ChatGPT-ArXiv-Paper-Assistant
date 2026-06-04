"""Tests for scripts/translate_hotspot_web_data.py — resurgence i18n fix."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Load the script module without executing __main__
# ---------------------------------------------------------------------------
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "translate_hotspot_web_data.py"
_SPEC = importlib.util.spec_from_file_location("translate_hotspot_web_data", _SCRIPT)
_MOD = importlib.util.module_from_spec(_SPEC)
sys.modules["translate_hotspot_web_data"] = _MOD
_SPEC.loader.exec_module(_MOD)

collect_and_translate = _MOD.collect_and_translate


class TestResurgenceTranslation(unittest.TestCase):
    """After collect_and_translate, resurgence entries must carry headline_zh."""

    def _make_payload(self) -> dict:
        return {
            "featured_topics": [],
            "watchlist": [],
            "resurgence": [
                {
                    "story_id": "story-1",
                    "headline": "GPT-5 resurfaces in benchmark rankings",
                    "original_first_date": "2026-05-01",
                    "reason": "Multiple new citations observed",
                    "entities": ["GPT-5"],
                },
                {
                    "story_id": "story-2",
                    "headline": "Open-source LLaMA variant achieves record throughput",
                    "original_first_date": "2026-04-20",
                    "reason": "GitHub stars spike",
                    "entities": ["LLaMA"],
                },
            ],
        }

    def test_resurgence_headlines_get_zh_field(self) -> None:
        """collect_and_translate must produce headline_zh for each resurgence entry."""
        payload = self._make_payload()
        expected_translations = [
            "GPT-5在基准测试排名中重新出现",
            "开源LLaMA变体实现创纪录吞吐量",
        ]

        # Stub batch_translate: return a zh string for every collected text.
        # The function collects ALL translatable texts into one list; resurgence
        # headlines are the only texts in this payload, so the stub receives exactly
        # those two strings and returns the expected translations.
        def fake_batch_translate(texts: list[str], model: str, **kwargs) -> list[str]:
            mapping = {
                "GPT-5 resurfaces in benchmark rankings": expected_translations[0],
                "Open-source LLaMA variant achieves record throughput": expected_translations[1],
            }
            return [mapping.get(t, t) for t in texts]

        with patch.object(_MOD, "batch_translate", side_effect=fake_batch_translate):
            result = collect_and_translate(payload, model="stub-model")

        resurgence = result["resurgence"]
        self.assertEqual(len(resurgence), 2)
        self.assertEqual(resurgence[0]["headline_zh"], expected_translations[0])
        self.assertEqual(resurgence[1]["headline_zh"], expected_translations[1])
        # Originals must be preserved
        self.assertEqual(resurgence[0]["headline"], "GPT-5 resurfaces in benchmark rankings")
        self.assertEqual(resurgence[1]["headline"], "Open-source LLaMA variant achieves record throughput")

    def test_resurgence_skips_already_translated_entries(self) -> None:
        """If headline_zh already exists and is non-empty, it is NOT overwritten."""
        payload = self._make_payload()
        payload["resurgence"][0]["headline_zh"] = "已翻译标题"

        call_log: list[list[str]] = []

        def fake_batch_translate(texts: list[str], model: str, **kwargs) -> list[str]:
            call_log.append(list(texts))
            return [f"zh:{t}" for t in texts]

        with patch.object(_MOD, "batch_translate", side_effect=fake_batch_translate):
            result = collect_and_translate(payload, model="stub-model")

        # Only the second entry should have been translated
        self.assertEqual(result["resurgence"][0]["headline_zh"], "已翻译标题")
        self.assertIn("headline_zh", result["resurgence"][1])
        self.assertNotEqual(result["resurgence"][1]["headline_zh"], "")

        # The already-Chinese headline of entry[0] must NOT appear in the translation batch
        if call_log:
            for batch in call_log:
                self.assertNotIn("GPT-5 resurfaces in benchmark rankings", batch)

    def test_empty_resurgence_is_a_noop(self) -> None:
        """A payload with an empty resurgence list must not cause errors."""
        payload = {"featured_topics": [], "watchlist": [], "resurgence": []}

        with patch.object(_MOD, "batch_translate", return_value=[]):
            result = collect_and_translate(payload, model="stub-model")

        # Empty resurgence list must be preserved unchanged
        self.assertEqual(result["resurgence"], [])

    def test_missing_resurgence_key_is_a_noop(self) -> None:
        """A payload without a resurgence key at all must not cause errors."""
        payload = {"featured_topics": [], "watchlist": []}

        with patch.object(_MOD, "batch_translate", return_value=[]):
            result = collect_and_translate(payload, model="stub-model")

        # No resurgence key should be added when the source has none
        self.assertNotIn("resurgence", result)


if __name__ == "__main__":
    unittest.main()
