from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from arxiv_assistant.hotspots.enrich import EnrichedItem
from arxiv_assistant.hotspots.story import Story
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem


def _make_item(
    *,
    source_id: str = "hf_papers",
    title: str = "A paper",
    url: str = "https://arxiv.org/abs/2606.00001",
    provenance: str = "",
    verified_first_date: str | None = None,
) -> HotspotItem:
    return HotspotItem(
        source_id=source_id,
        source_name="HF",
        source_role="paper_trending",
        source_type="paper",
        title=title,
        summary="A summary.",
        url=url,
        canonical_url=url,
        published_at="2026-06-01T12:00:00+00:00",
        metadata={"arxiv_id": "2606.00001"},
        provenance=provenance,
        verified_first_date=verified_first_date,
    )


class TestHotspotItemFields(unittest.TestCase):
    def test_new_fields_default_safe(self) -> None:
        item = HotspotItem(
            source_id="hf_papers",
            source_name="HF",
            source_role="paper_trending",
            source_type="paper",
            title="A paper",
            summary="A summary.",
            url="https://arxiv.org/abs/2606.00001",
            canonical_url="https://arxiv.org/abs/2606.00001",
        )
        self.assertIsNone(item.verified_first_date)
        self.assertEqual(item.provenance, "")

    def test_new_fields_round_trip_through_to_dict(self) -> None:
        item = _make_item(provenance="native:hf_papers", verified_first_date="2026-05-20T00:00:00+00:00")
        payload = item.to_dict()
        self.assertEqual(payload["provenance"], "native:hf_papers")
        self.assertEqual(payload["verified_first_date"], "2026-05-20T00:00:00+00:00")
        # round-trip back through the dataclass
        restored = HotspotItem(**payload)
        self.assertEqual(restored.provenance, "native:hf_papers")
        self.assertEqual(restored.verified_first_date, "2026-05-20T00:00:00+00:00")


def _make_enriched(item: HotspotItem | None = None) -> EnrichedItem:
    item = item or _make_item()
    return EnrichedItem(
        item=item,
        event_type="research_paper",
        entities=[{"name": "OpenAI"}],
        summary="A summary.",
        importance=7,
    )


def _make_story(story_id: str = "story-1") -> Story:
    ei = _make_enriched()
    return Story(
        story_id=story_id,
        canonical_item=ei,
        items=[ei],
        event_type="research_paper",
        entity_names={"openai"},
    )


class TestStoryFields(unittest.TestCase):
    def test_new_persistent_fields_default_safe(self) -> None:
        story = _make_story()
        self.assertIsNone(story.first_seen)
        self.assertIsNone(story.centroid)
        self.assertEqual(story.centroid_model_id, "")
        self.assertEqual(story.status, "NEW")
        self.assertEqual(story.arxiv_versions, {})
        self.assertIsNone(story.last_surfaced)
        self.assertIsNone(story.surfaced_verified_max)
        self.assertEqual(story.surfaced_entity_names, set())
        self.assertEqual(story.surfaced_max_tier, 0)
        self.assertEqual(story.surfaced_arxiv_versions, {})
        self.assertIsNone(story.resurged_at)
        self.assertIsNone(story.surfaced_resurged_at)

    def test_existing_behavior_unchanged(self) -> None:
        story = _make_story()
        # __post_init__ still derives category/headline/summary from canonical_item
        self.assertEqual(story.headline, "A paper")
        self.assertEqual(story.summary, "A summary.")
        self.assertEqual(story.entity_names, {"openai"})

    def test_dict_defaults_are_independent_instances(self) -> None:
        a = _make_story("a")
        b = _make_story("b")
        a.arxiv_versions["2606.00001"] = 2
        a.surfaced_entity_names.add("anthropic")
        self.assertEqual(b.arxiv_versions, {})
        self.assertEqual(b.surfaced_entity_names, set())

    def test_evidence_ledger_default_safe(self) -> None:
        story = _make_story()
        self.assertEqual(story.evidence_ledger, [])

    def test_evidence_ledger_independent_instances(self) -> None:
        a = _make_story("a")
        b = _make_story("b")
        a.evidence_ledger.append({"canonical_url": "https://x.com/1", "source_id": "x",
                                   "source_role": "social", "provenance": "native:x",
                                   "source_tier": 1, "added_at": "2026-05-01"})
        self.assertEqual(b.evidence_ledger, [])

    def test_evidence_added_since_partition(self) -> None:
        story = _make_story()
        rows = [
            {"canonical_url": "https://a.com", "source_id": "s1", "source_role": "r",
             "provenance": "p", "source_tier": 1, "added_at": "2026-05-01"},
            {"canonical_url": "https://b.com", "source_id": "s2", "source_role": "r",
             "provenance": "p", "source_tier": 2, "added_at": "2026-05-15"},
            {"canonical_url": "https://c.com", "source_id": "s3", "source_role": "r",
             "provenance": "p", "source_tier": 1, "added_at": "2026-06-01"},
        ]
        story.evidence_ledger = rows

        # "since" is strictly greater-than snapshot_date
        result = story.evidence_added_since("2026-05-01")
        self.assertEqual(len(result), 2)
        self.assertIn(rows[1], result)
        self.assertIn(rows[2], result)

        result = story.evidence_added_since("2026-05-14")
        self.assertEqual(len(result), 2)

        result = story.evidence_added_since("2026-05-15")
        self.assertEqual(len(result), 1)
        self.assertIn(rows[2], result)

        result = story.evidence_added_since("2026-06-01")
        self.assertEqual(result, [])

    def test_evidence_added_since_none_returns_all(self) -> None:
        story = _make_story()
        rows = [
            {"canonical_url": "https://a.com", "source_id": "s1", "source_role": "r",
             "provenance": "p", "source_tier": 1, "added_at": "2026-05-01"},
            {"canonical_url": "https://b.com", "source_id": "s2", "source_role": "r",
             "provenance": "p", "source_tier": 2, "added_at": "2026-06-01"},
        ]
        story.evidence_ledger = rows
        self.assertEqual(story.evidence_added_since(None), rows)

    def test_evidence_before_partition(self) -> None:
        story = _make_story()
        rows = [
            {"canonical_url": "https://a.com", "source_id": "s1", "source_role": "r",
             "provenance": "p", "source_tier": 1, "added_at": "2026-05-01"},
            {"canonical_url": "https://b.com", "source_id": "s2", "source_role": "r",
             "provenance": "p", "source_tier": 2, "added_at": "2026-05-15"},
            {"canonical_url": "https://c.com", "source_id": "s3", "source_role": "r",
             "provenance": "p", "source_tier": 1, "added_at": "2026-06-01"},
        ]
        story.evidence_ledger = rows

        # "before" is <= snapshot_date
        result = story.evidence_before("2026-05-01")
        self.assertEqual(len(result), 1)
        self.assertIn(rows[0], result)

        result = story.evidence_before("2026-05-15")
        self.assertEqual(len(result), 2)
        self.assertIn(rows[0], result)
        self.assertIn(rows[1], result)

        result = story.evidence_before("2026-04-30")
        self.assertEqual(result, [])

        result = story.evidence_before("2026-06-01")
        self.assertEqual(len(result), 3)

    def test_evidence_before_none_returns_empty(self) -> None:
        story = _make_story()
        story.evidence_ledger = [
            {"canonical_url": "https://a.com", "source_id": "s1", "source_role": "r",
             "provenance": "p", "source_tier": 1, "added_at": "2026-05-01"},
        ]
        self.assertEqual(story.evidence_before(None), [])

    def test_evidence_partition_complement(self) -> None:
        """evidence_added_since and evidence_before together cover all rows when snapshot_date is not None."""
        story = _make_story()
        rows = [
            {"canonical_url": "https://a.com", "source_id": "s1", "source_role": "r",
             "provenance": "p", "source_tier": 1, "added_at": "2026-05-01"},
            {"canonical_url": "https://b.com", "source_id": "s2", "source_role": "r",
             "provenance": "p", "source_tier": 2, "added_at": "2026-05-15"},
            {"canonical_url": "https://c.com", "source_id": "s3", "source_role": "r",
             "provenance": "p", "source_tier": 1, "added_at": "2026-06-01"},
        ]
        story.evidence_ledger = rows
        snapshot = "2026-05-10"
        before = story.evidence_before(snapshot)
        since = story.evidence_added_since(snapshot)
        self.assertEqual(len(before) + len(since), len(rows))
        # no row in both halves
        self.assertEqual(set(id(r) for r in before) & set(id(r) for r in since), set())


if __name__ == "__main__":
    unittest.main()
