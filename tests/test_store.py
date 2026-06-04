from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date as _date
from pathlib import Path

from arxiv_assistant.hotspots.enrich import EnrichedItem
from arxiv_assistant.hotspots.store import StoryStore, _open_story_store
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


# ---------------------------------------------------------------------------
# Task 3 — StoryStore tests (verbatim from plan §3a + extended §6 invariants)
# ---------------------------------------------------------------------------


class TestStoreLifecycle(unittest.TestCase):
    def test_init_creates_db_file_and_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state" / "story_store.sqlite"
            store = StoryStore(db_path)
            self.assertTrue(db_path.exists())
            names = store._table_names()
            self.assertEqual(
                set(names) & {"stories", "evidence", "date_verdicts", "versions"},
                {"stories", "evidence", "date_verdicts", "versions"},
            )
            store.close()

    def test_dump_and_load_round_trip_includes_date_verdicts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "state" / "story_store.sqlite"
            store = StoryStore(db_path)
            store.put_verdict(
                "hash-abc",
                {"verified_first_date": "2026-05-20T00:00:00+00:00", "confidence": 0.95, "evidence": ["arxiv"]},
            )
            store.refresh_arxiv_versions("2606.00001", 2)
            out_dir = Path(tmp) / "snapshot"
            snapshot = store.dump_text_snapshot(out_dir)
            self.assertTrue(snapshot.exists())
            blob = json.loads(snapshot.read_text(encoding="utf-8"))
            self.assertIn("date_verdicts", blob)
            self.assertIn("hash-abc", {row["content_hash"] for row in blob["date_verdicts"]})
            store.close()

            # rebuild on a fresh store inherits the frozen verdict
            db2 = Path(tmp) / "state2" / "story_store.sqlite"
            store2 = StoryStore(db2)
            store2.load_text_snapshot(snapshot)
            verdict = store2.get_verdict("hash-abc")
            self.assertIsNotNone(verdict)
            self.assertEqual(verdict["verified_first_date"], "2026-05-20T00:00:00+00:00")
            self.assertEqual(verdict["confidence"], 0.95)
            self.assertEqual(verdict["evidence"], ["arxiv"])
            store2.close()


class TestVerdictFreeze(unittest.TestCase):
    """INV3: put_verdict is write-once (permanent freeze)."""

    def test_put_verdict_write_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StoryStore(Path(tmp) / "store.sqlite")
            store.put_verdict("h1", {"verified_first_date": "2026-01-01T00:00:00+00:00", "confidence": 0.9, "evidence": ["a"]})
            # Second write with different data must be silently ignored
            store.put_verdict("h1", {"verified_first_date": "2099-01-01T00:00:00+00:00", "confidence": 0.1, "evidence": ["b"]})
            v = store.get_verdict("h1")
            self.assertIsNotNone(v)
            self.assertEqual(v["verified_first_date"], "2026-01-01T00:00:00+00:00")
            self.assertAlmostEqual(v["confidence"], 0.9)
            self.assertEqual(v["evidence"], ["a"])
            store.close()

    def test_get_verdict_missing_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StoryStore(Path(tmp) / "store.sqlite")
            self.assertIsNone(store.get_verdict("no-such-hash"))
            store.close()


class TestArxivVersionsMonotonic(unittest.TestCase):
    """INV3: refresh_arxiv_versions is monotonic non-decreasing."""

    def test_version_increases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StoryStore(Path(tmp) / "store.sqlite")
            store.refresh_arxiv_versions("2606.00001", 3)
            self.assertEqual(store.get_arxiv_version("2606.00001"), 3)
            store.refresh_arxiv_versions("2606.00001", 5)
            self.assertEqual(store.get_arxiv_version("2606.00001"), 5)
            store.close()

    def test_version_does_not_decrease(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StoryStore(Path(tmp) / "store.sqlite")
            store.refresh_arxiv_versions("2606.00002", 7)
            store.refresh_arxiv_versions("2606.00002", 2)  # lower — must not shrink
            self.assertEqual(store.get_arxiv_version("2606.00002"), 7)
            store.close()

    def test_version_not_in_date_verdicts(self) -> None:
        """refresh_arxiv_versions must never touch date_verdicts."""
        with tempfile.TemporaryDirectory() as tmp:
            store = StoryStore(Path(tmp) / "store.sqlite")
            store.refresh_arxiv_versions("2606.00003", 2)
            self.assertIsNone(store.get_verdict("2606.00003"))
            store.close()


class TestActiveStoriesAndMatchOrCreate(unittest.TestCase):
    def _make_store(self, tmp: str) -> StoryStore:
        return StoryStore(Path(tmp) / "store.sqlite")

    def test_active_stories_empty_on_new_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            self.assertEqual(store.active_stories(14, _date(2026, 6, 1)), [])
            store.close()

    def test_match_or_create_new_story(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            story = _make_story("s1")
            centroid = [1.0, 0.0]
            returned, is_new = store.match_or_create(centroid, story, 0.9, 14, _date(2026, 6, 1))
            self.assertTrue(is_new)
            self.assertEqual(returned.story_id, "s1")
            self.assertEqual(returned.centroid, centroid)
            store.close()

    def test_match_or_create_matches_similar_story(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            s1 = _make_story("s1")
            s1.first_seen = "2026-06-01"
            centroid_a = [1.0, 0.0]
            store.match_or_create(centroid_a, s1, 0.9, 14, _date(2026, 6, 1))

            # New cluster with near-identical centroid
            s2 = _make_story("s2")
            centroid_b = [0.9999, 0.0141]  # very close
            returned, is_new = store.match_or_create(centroid_b, s2, 0.9, 14, _date(2026, 6, 2))
            self.assertFalse(is_new)
            self.assertEqual(returned.story_id, "s1")  # matched existing
            self.assertEqual(returned.status, "ONGOING")
            store.close()

    def test_active_stories_excludes_outside_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            s_old = _make_story("old")
            s_old.first_seen = "2026-05-01"  # 31 days before
            store.match_or_create([1.0, 0.0], s_old, 0.9, 14, _date(2026, 5, 1))

            active = store.active_stories(14, _date(2026, 6, 1))
            ids = {s.story_id for s in active}
            self.assertNotIn("old", ids)
            store.close()


class TestUpsertEvidenceAndLedger(unittest.TestCase):
    """§6 item 1: source_tier column; §6 item 2: evidence_ledger populated on load."""

    def test_upsert_evidence_source_tier_column_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StoryStore(Path(tmp) / "store.sqlite")
            story = _make_story("s1")
            store.match_or_create([1.0, 0.0], story, 0.9, 14, _date(2026, 6, 1))
            ei = _make_enriched()
            store.upsert_evidence("s1", [ei], "2026-06-01")

            row = store._conn.execute(
                "SELECT source_tier FROM evidence WHERE story_id='s1'"
            ).fetchone()
            self.assertIsNotNone(row)
            self.assertIsInstance(row["source_tier"], int)
            store.close()

    def test_evidence_ledger_populated_on_active_stories(self) -> None:
        """active_stories must return stories with evidence_ledger populated."""
        with tempfile.TemporaryDirectory() as tmp:
            store = StoryStore(Path(tmp) / "store.sqlite")
            story = _make_story("s1")
            store.match_or_create([1.0, 0.0], story, 0.9, 14, _date(2026, 6, 1))
            ei = _make_enriched()
            store.upsert_evidence("s1", [ei], "2026-06-01")

            active = store.active_stories(14, _date(2026, 6, 1))
            self.assertEqual(len(active), 1)
            ledger = active[0].evidence_ledger
            self.assertEqual(len(ledger), 1)
            row = ledger[0]
            self.assertIn("canonical_url", row)
            self.assertIn("source_id", row)
            self.assertIn("source_role", row)
            self.assertIn("provenance", row)
            self.assertIn("source_tier", row)
            self.assertIn("added_at", row)
            self.assertIsInstance(row["source_tier"], int)
            store.close()


class TestSeedFirstSeen(unittest.TestCase):
    """§6 item 1: seed_first_seen write-once, not via match_or_create."""

    def test_seed_first_seen_sets_value(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = StoryStore(Path(tmp) / "store.sqlite")
            story = _make_story("s1")
            # Persist story row directly with first_seen=NULL (bypasses match_or_create)
            store._conn.execute(
                "INSERT INTO stories (story_id, first_seen, event_type, headline, "
                "entity_names, surfaced_entity_names, surfaced_arxiv_versions, updated_at) "
                "VALUES (?,NULL,?,?,?,?,?,?)",
                (
                    "s1", "research_paper", "A paper",
                    '["openai"]', '[]', '{}', "2026-06-01T00:00:00+00:00",
                ),
            )
            store._conn.commit()
            # seed_first_seen should now set first_seen to the provided date
            store.seed_first_seen(story, "2026-05-15")
            row = store._conn.execute(
                "SELECT first_seen FROM stories WHERE story_id='s1'"
            ).fetchone()
            self.assertEqual(row["first_seen"], "2026-05-15")
            store.close()

    def test_seed_first_seen_write_once(self) -> None:
        """seed_first_seen is a no-op if first_seen already set."""
        with tempfile.TemporaryDirectory() as tmp:
            store = StoryStore(Path(tmp) / "store.sqlite")
            story = _make_story("s1")
            story.first_seen = "2026-05-10"
            store.match_or_create([1.0, 0.0], story, 0.9, 14, _date(2026, 5, 10))
            # Attempt to overwrite
            store.seed_first_seen(story, "2026-01-01")
            row = store._conn.execute(
                "SELECT first_seen FROM stories WHERE story_id='s1'"
            ).fetchone()
            # Must still be the original value
            self.assertEqual(row["first_seen"], "2026-05-10")
            store.close()


class TestOpenStoryStoreHelper(unittest.TestCase):
    def test_open_story_store_creates_at_expected_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = _open_story_store(Path(tmp))
            expected = Path(tmp) / "hot" / "state" / "story_store.sqlite"
            self.assertEqual(store.db_path, expected)
            self.assertTrue(expected.exists())
            store.close()


if __name__ == "__main__":
    unittest.main()
