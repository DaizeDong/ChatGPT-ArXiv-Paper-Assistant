"""Hotspot enrichment now goes through utils.llm_gateway, not raw HTTP."""
from __future__ import annotations

import configparser
import json
import tempfile
import unittest
import unittest.mock
from datetime import datetime, timezone
from pathlib import Path

from arxiv_assistant.hotspots import enrich as E
from arxiv_assistant.hotspots import kernel
from arxiv_assistant.utils.agent_runner import AgentError
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem


def _item(idx: int) -> HotspotItem:
    return HotspotItem(
        source_id="official_blogs",
        source_name="AcmeCorp Blog",
        source_role="official_news",
        source_type="news",
        title=f"AcmeCorp launches widget {idx}",
        summary="AcmeCorp announced a new widget for developers.",
        url=f"https://example.com/post/{idx}",
        canonical_url=f"https://example.com/post/{idx}",
        published_at="2026-05-20T00:00:00+00:00",
    )


class _FakeResult:
    def __init__(self, text: str, *, backend: str = "llmcall", provider: str = "cc", data=None):
        self.text = text
        self.data = data
        self.backend = backend
        self.provider = provider

    def __bool__(self) -> bool:
        return bool(self.text) or self.data is not None


def _rows_for(items: list[HotspotItem], offset: int = 0) -> str:
    return json.dumps([
        {
            "index": offset + i,
            "event_type": "product_release",
            "entities": [{"name": "AcmeCorp", "type": "organization"}],
            "summary": f"LLM summary {offset + i}",
            "importance": 7,
        }
        for i in range(len(items))
    ])


class TestChatCompletionContract(unittest.TestCase):
    """_chat_completion keeps its OpenAI-chat return shape; callers are unchanged."""

    def test_returns_openai_shaped_dict_from_gateway_text(self) -> None:
        seen: dict = {}

        def fake_call(prompt, **kwargs):
            seen["prompt"] = prompt
            seen["kwargs"] = kwargs
            return _FakeResult("[]", backend="llmcall", provider="cc")

        data = E._chat_completion(
            "gpt-4o-mini",
            [{"role": "system", "content": "You classify items."},
             {"role": "user", "content": "ITEM 0"}],
            gateway_call=fake_call,
        )

        self.assertEqual(data["choices"][0]["message"]["content"], "[]")
        self.assertEqual(data["model"], "gpt-4o-mini")
        self.assertEqual(data["_gateway"], {"backend": "llmcall", "provider": "cc"})

    def test_system_and_user_are_concatenated_with_a_visible_delimiter(self) -> None:
        seen: dict = {}

        def fake_call(prompt, **kwargs):
            seen["prompt"] = prompt
            return _FakeResult("[]")

        E._chat_completion(
            "m",
            [{"role": "system", "content": "SYSTEM-TEXT"},
             {"role": "user", "content": "USER-TEXT"}],
            gateway_call=fake_call,
        )
        prompt = seen["prompt"]
        # Both payloads survive verbatim, in order, under a role banner.
        self.assertIn("SYSTEM-TEXT", prompt)
        self.assertIn("USER-TEXT", prompt)
        self.assertIn("===== SYSTEM INSTRUCTIONS =====", prompt)
        self.assertIn("===== USER =====", prompt)
        self.assertLess(prompt.index("SYSTEM-TEXT"), prompt.index("USER-TEXT"))

    def test_openai_model_name_is_not_forwarded_to_the_gateway(self) -> None:
        """gpt-* is an OpenAI catalogue name; asking llmcall or claude for it fails."""
        seen: dict = {}

        def fake_call(prompt, **kwargs):
            seen["kwargs"] = kwargs
            return _FakeResult("[]")

        E._chat_completion("gpt-4o-mini", [{"role": "user", "content": "x"}], gateway_call=fake_call)
        self.assertNotIn("model", seen["kwargs"])

    def test_data_only_result_is_serialised_into_content(self) -> None:
        def fake_call(prompt, **kwargs):
            return _FakeResult("", data=[{"index": 0}])

        data = E._chat_completion("m", [{"role": "user", "content": "x"}], gateway_call=fake_call)
        self.assertEqual(json.loads(data["choices"][0]["message"]["content"]), [{"index": 0}])

    def test_gateway_failure_propagates(self) -> None:
        def boom(prompt, **kwargs):
            raise AgentError("chain down")

        with self.assertRaises(AgentError):
            E._chat_completion("m", [{"role": "user", "content": "x"}], gateway_call=boom)


class TestEnrichStatus(unittest.TestCase):
    def test_successful_llm_batch_is_labelled_llm(self) -> None:
        items = [_item(i) for i in range(3)]

        def fake_call(prompt, **kwargs):
            return _FakeResult(_rows_for(items), backend="llmcall", provider="cc")

        enriched, status = E.enrich_items_batch_with_status(
            items, "gpt-4o-mini", 20, 3, gateway_call=fake_call)

        self.assertEqual(len(enriched), 3)
        self.assertEqual({e.enrich_source for e in enriched}, {"llm"})
        self.assertEqual(status.path, "llm")
        self.assertTrue(status.llm_ok)
        self.assertEqual(status.items_llm, 3)
        self.assertEqual(status.items_heuristic, 0)
        self.assertEqual(status.batches_failed, 0)
        self.assertEqual(status.backend, "llmcall")
        self.assertEqual(status.provider, "cc")

    def test_total_failure_degrades_to_heuristic_but_says_so(self) -> None:
        items = [_item(i) for i in range(3)]

        def boom(prompt, **kwargs):
            raise AgentError("every backend refused")

        enriched, status = E.enrich_items_batch_with_status(
            items, "gpt-4o-mini", 20, 2, gateway_call=boom)

        # The honest degrade still happens: every item comes back.
        self.assertEqual(len(enriched), 3)
        self.assertEqual({e.enrich_source for e in enriched}, {"heuristic_batch_failed"})
        # ...and it is VISIBLE.
        self.assertEqual(status.path, "heuristic_fallback")
        self.assertFalse(status.llm_ok)
        self.assertEqual(status.items_llm, 0)
        self.assertEqual(status.batches_failed, 1)
        self.assertTrue(status.errors)
        self.assertIn("every backend refused", status.errors[0])

    def test_outage_and_heuristic_mode_have_identical_rows_but_different_status(self) -> None:
        """The whole point: the rows cannot tell you, the status must."""
        items = [_item(i) for i in range(3)]

        def boom(prompt, **kwargs):
            raise AgentError("down")

        degraded, degraded_status = E.enrich_items_batch_with_status(
            items, "m", 20, 1, gateway_call=boom)
        plain = E.enrich_items_heuristic(items)
        plain_status = E.heuristic_status(plain)

        self.assertEqual([e.summary for e in degraded], [e.summary for e in plain])
        self.assertEqual([e.importance for e in degraded], [e.importance for e in plain])
        self.assertNotEqual(degraded_status.path, plain_status.path)
        self.assertEqual(plain_status.path, "heuristic")
        self.assertEqual(degraded_status.path, "heuristic_fallback")

    def test_missing_row_is_labelled_and_counted_as_mixed(self) -> None:
        items = [_item(i) for i in range(3)]

        def partial(prompt, **kwargs):
            return _FakeResult(_rows_for(items[:2]))

        enriched, status = E.enrich_items_batch_with_status(
            items, "m", 20, 1, gateway_call=partial)

        sources = sorted(e.enrich_source for e in enriched)
        self.assertEqual(sources, ["heuristic_row_missing", "llm", "llm"])
        self.assertEqual(status.path, "mixed")
        self.assertTrue(status.llm_ok)
        self.assertEqual(status.items_llm, 2)
        self.assertEqual(status.items_heuristic, 1)

    def test_batch_wrapper_keeps_the_plain_list_contract(self) -> None:
        items = [_item(i) for i in range(2)]

        def fake_call(prompt, **kwargs):
            return _FakeResult(_rows_for(items))

        out = E.enrich_items_batch(items, "m", 20, 1, gateway_call=fake_call)
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 2)
        self.assertTrue(all(isinstance(e, E.EnrichedItem) for e in out))

    def test_status_dict_is_json_serialisable(self) -> None:
        status = E.EnrichmentStatus(mode_requested="openai", items=1, items_llm=1, batches=1, batches_llm=1)
        json.dumps(status.to_dict())  # must not raise
        self.assertEqual(status.to_dict()["path"], "llm")


class TestKernelEnrichMode(unittest.TestCase):
    """The mode literal: "openai" survives as an alias, "llm" is the new name."""

    def test_both_literals_mean_llm_enriched(self) -> None:
        self.assertTrue(kernel._is_llm_mode("openai"))
        self.assertTrue(kernel._is_llm_mode("llm"))
        self.assertTrue(kernel._is_llm_mode("  OpenAI "))
        self.assertFalse(kernel._is_llm_mode("heuristic"))
        self.assertFalse(kernel._is_llm_mode(None))

    def _ctx(self, mode: str) -> kernel.KernelContext:
        cfg = configparser.ConfigParser()
        cfg["HOTSPOTS"] = {"enabled": "true", "mode": mode}
        return kernel.KernelContext(
            output_root=Path("."),
            target_date=datetime(2026, 5, 20, tzinfo=timezone.utc),
            config=cfg, store=None, journal=[],
        )

    def test_heuristic_mode_reports_heuristic_path(self) -> None:
        enriched, status = kernel._enrich(self._ctx("heuristic"), [_item(0)])
        self.assertEqual(len(enriched), 1)
        self.assertEqual(status["path"], "heuristic")
        self.assertFalse(status["llm_ok"])

    def test_llm_mode_outage_is_recorded_not_swallowed(self) -> None:
        def boom(prompt, **kwargs):
            raise AgentError("chain down")

        with unittest.mock.patch.object(E.llm_gateway, "call", boom):
            enriched, status = kernel._enrich(self._ctx("llm"), [_item(0)])

        self.assertEqual(len(enriched), 1)          # degrade still produces output
        self.assertEqual(status["path"], "heuristic_fallback")
        self.assertFalse(status["llm_ok"])
        self.assertEqual(status["mode_requested"], "llm")
        self.assertTrue(status["errors"])

    def test_legacy_openai_literal_takes_the_same_llm_branch(self) -> None:
        def boom(prompt, **kwargs):
            raise AgentError("chain down")

        with unittest.mock.patch.object(E.llm_gateway, "call", boom):
            _enriched, status = kernel._enrich(self._ctx("openai"), [_item(0)])
        self.assertEqual(status["mode_requested"], "openai")
        self.assertEqual(status["path"], "heuristic_fallback")


class TestEnrichmentStatusReachesTheReport(unittest.TestCase):
    """The record has to SURVIVE the checkpoint chain, not merely be created."""

    def _config(self) -> configparser.ConfigParser:
        cfg = configparser.ConfigParser()
        cfg["HOTSPOTS"] = {
            "enabled": "true", "mode": "llm", "max_item_age_days": "14",
            "resurge_min_competitors": "3", "resurge_cooldown_days": "7",
        }
        return cfg

    def _render_report(self, enrichment: dict) -> dict:
        """Run score -> synthesize -> render for real, return the written report."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            td = datetime(2026, 5, 20, tzinfo=timezone.utc)
            kernel._write_checkpoint(root, td, "harvest", {
                "items": [kernel._serialize_item(_item(0))],
                "source_stats": {}, "api_usage": {},
            })
            # The score stage's real output shape, carrying the enrichment record.
            kernel._write_checkpoint(root, td, "score", {
                "featured": [], "watchlist": [], "all_topics": [],
                "enrichment": enrichment,
            })
            ctx = kernel.KernelContext(output_root=root, target_date=td,
                                       config=self._config(), store=None, journal=[])
            # Go through synthesize for real: it is where the record was dropped.
            kernel._write_checkpoint(root, td, "synthesize", kernel._stage_synthesize(ctx))
            kernel._stage_render(ctx)
            return json.loads(
                (root / "hot" / "reports" / "2026-05-20.json").read_text(encoding="utf-8")
            )

    def test_synthesize_forwards_the_enrichment_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            td = datetime(2026, 5, 20, tzinfo=timezone.utc)
            record = {"path": "llm", "llm_ok": True, "provider": "cc", "backend": "llmcall"}
            kernel._write_checkpoint(root, td, "score", {
                "featured": [], "watchlist": [], "all_topics": [], "enrichment": record,
            })
            ctx = kernel.KernelContext(output_root=root, target_date=td,
                                       config=self._config(), store=None, journal=[])
            payload = kernel._stage_synthesize(ctx)
        self.assertEqual(payload.get("enrichment"), record)

    def test_outage_is_legible_in_the_published_report(self) -> None:
        report = self._render_report({
            "path": "heuristic_fallback", "mode_requested": "llm", "llm_ok": False,
            "batches": 2, "batches_llm": 0, "batches_failed": 2,
            "items": 5, "items_llm": 0, "items_heuristic": 5,
            "backend": "", "provider": "", "errors": ["batch 0: chain down"],
        })
        self.assertIn("enrichment", report, "report must carry the enrichment record")
        self.assertEqual(report["enrichment"]["path"], "heuristic_fallback")
        self.assertFalse(report["enrichment"]["llm_ok"])
        self.assertTrue(report["enrichment"]["errors"], "the reason must survive too")
        # And the usage row must not name a vendor when none answered.
        self.assertEqual(report["usage"]["llm"]["provider"], "none")
        self.assertEqual(report["usage"]["llm"]["billing_model"], "disabled")

    def test_success_and_outage_are_different_reports(self) -> None:
        """The negative control: the two paths must not render identically."""
        ok = self._render_report({
            "path": "llm", "mode_requested": "llm", "llm_ok": True,
            "batches": 2, "batches_llm": 2, "batches_failed": 0,
            "items": 5, "items_llm": 5, "items_heuristic": 0,
            "backend": "llmcall", "provider": "cc", "errors": [],
        })
        down = self._render_report({
            "path": "heuristic_fallback", "mode_requested": "llm", "llm_ok": False,
            "batches": 2, "batches_llm": 0, "batches_failed": 2,
            "items": 5, "items_llm": 0, "items_heuristic": 5,
            "backend": "", "provider": "", "errors": ["batch 0: chain down"],
        })
        self.assertEqual(ok["usage"]["llm"]["provider"], "cc")
        self.assertEqual(ok["usage"]["llm"]["billing_model"], "quota")
        self.assertNotEqual(ok["enrichment"]["path"], down["enrichment"]["path"])
        self.assertNotEqual(ok["usage"]["llm"], down["usage"]["llm"])

    def test_archive_without_the_key_reads_as_empty_not_a_crash(self) -> None:
        """173 archived reports predate this key. Absent must read empty, not raise."""
        report = self._render_report({})
        self.assertEqual(report["enrichment"], {})
        self.assertEqual(report["usage"]["llm"]["provider"], "none")


if __name__ == "__main__":
    unittest.main()
