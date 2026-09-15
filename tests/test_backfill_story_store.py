"""Tests for scripts/backfill_story_store.py.

Core contract (overview §5 safety rail):
- dedup_history collapses a multi-day history of EnrichedItems using the SAME
  intraday + cross-day dedup logic as the live pipeline.
- A story that appears on N consecutive days (the 6-day-dup bug) collapses to
  ONE entry whose `first_seen` is the EARLIEST day, not N polluted anchors.
- Two genuinely distinct events yield two separate seed records.

Integration contract (§2.3 seed_first_seen write-once):
- Running the full backfill on a synthetic 3-day history of one repeated event
  seeds the StoryStore with exactly ONE story whose `first_seen` is the earliest
  date — proving no polluted anchors reach the Store.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from arxiv_assistant.hotspots.enrich import EnrichedItem
from arxiv_assistant.hotspots.store import StoryStore
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _item(title: str, url: str, *, summary: str = "") -> HotspotItem:
    return HotspotItem(
        source_id="s",
        source_name="S",
        source_role="official_news",
        source_type="news",
        title=title,
        summary=summary,
        url=url,
        canonical_url=url,
        published_at="2026-06-01T00:00:00+00:00",
        tags=[],
        authors=[],
        metadata={},
    )


def _enriched(title: str, url: str, *, summary: str = "") -> EnrichedItem:
    return EnrichedItem(
        item=_item(title, url, summary=summary),
        event_type="product_release",
        entities=[],
        summary=summary or title,
        importance=5,
    )


# ---------------------------------------------------------------------------
# Unit tests — dedup_history pure function
# ---------------------------------------------------------------------------

class TestDedupHistory(unittest.TestCase):
    """Pure-function tests: no StoryStore, no disk I/O."""

    def _patch_embed(self, mapping: dict):
        """Deterministic stub: map text-substring → unit vector direction."""
        def fake_embed(text: str):
            for key, vec in mapping.items():
                if key in text:
                    return list(vec)
            return [0.0, 0.0, 1.0]
        return patch("arxiv_assistant.hotspots.dedup.embed_text", side_effect=fake_embed)

    def test_six_day_duplicate_yields_single_first_seen(self) -> None:
        """Same event re-featured on 6 consecutive days → ONE seed at the earliest date."""
        from scripts.backfill_story_store import dedup_history

        history: dict[str, list] = {}
        for d in range(1, 7):
            history[f"2026-06-0{d}"] = [
                _enriched("Anthropic launches Claude 5", f"https://news.com/c5-day{d}")
            ]

        with self._patch_embed({"Claude 5": (1.0, 0.0, 0.0)}):
            seeds = dedup_history(history)

        # ONE real story → ONE first_seen at the earliest date (not 6 polluted anchors).
        self.assertEqual(len(seeds), 1)
        (only,) = seeds
        self.assertEqual(only["first_seen"], "2026-06-01")

    def test_two_distinct_events_yield_two_seeds(self) -> None:
        """Genuinely different events on different days yield two separate seeds."""
        from scripts.backfill_story_store import dedup_history

        history = {
            "2026-06-01": [_enriched("Event Alpha", "https://a.com/1")],
            "2026-06-02": [_enriched("Event Beta totally unrelated", "https://b.com/1")],
        }

        with self._patch_embed({"Alpha": (1.0, 0.0, 0.0), "Beta": (0.0, 1.0, 0.0)}):
            seeds = dedup_history(history)

        self.assertEqual(len(seeds), 2)

    def test_first_seen_is_earliest_day(self) -> None:
        """Verify `first_seen` is the minimum date, not the last occurrence."""
        from scripts.backfill_story_store import dedup_history

        history = {
            "2026-06-03": [_enriched("Big AI Story", "https://a.com/day3")],
            "2026-06-01": [_enriched("Big AI Story again", "https://a.com/day1")],
            "2026-06-02": [_enriched("Big AI Story variant", "https://a.com/day2")],
        }

        with self._patch_embed({"Big AI Story": (1.0, 0.02, 0.0)}):
            seeds = dedup_history(history)

        self.assertEqual(len(seeds), 1)
        self.assertEqual(seeds[0]["first_seen"], "2026-06-01")

    def test_empty_history_returns_empty(self) -> None:
        from scripts.backfill_story_store import dedup_history

        seeds = dedup_history({})
        self.assertEqual(seeds, [])

    def test_seed_record_has_required_keys(self) -> None:
        """Each seed record must have first_seen, centroid, centroid_model_id, n_days."""
        from scripts.backfill_story_store import dedup_history

        history = {"2026-06-01": [_enriched("Solo event", "https://a.com/solo")]}

        with self._patch_embed({"Solo": (1.0, 0.0, 0.0)}):
            seeds = dedup_history(history)

        self.assertEqual(len(seeds), 1)
        seed = seeds[0]
        for key in ("first_seen", "centroid", "centroid_model_id", "n_days"):
            self.assertIn(key, seed, f"Missing key: {key}")
        # _member_vecs is an internal accumulator and must be stripped from output.
        self.assertNotIn("_member_vecs", seed)


# ---------------------------------------------------------------------------
# Integration test — StoryStore seeding (dedup-first; earliest first_seen wins)
# ---------------------------------------------------------------------------

class TestBackfillStoreIntegration(unittest.TestCase):
    """End-to-end: synthetic 3-day dup history → StoryStore has ONE story, earliest first_seen."""

    def _patch_embed(self, mapping: dict):
        def fake_embed(text: str):
            for key, vec in mapping.items():
                if key in text:
                    return list(vec)
            return [0.0, 0.0, 1.0]
        return patch("arxiv_assistant.hotspots.dedup.embed_text", side_effect=fake_embed)

    def test_three_day_dup_seeds_one_story_earliest_date(self) -> None:
        """3-day duplicate of the same event → Store has ONE story with first_seen on day 1.

        This is the 6-day-dup pattern in miniature: a story that appeared on multiple
        days in the historical reports must NOT create multiple polluted `first_seen`
        anchors. The backfill must collapse them (dedup-first) and seed exactly one
        story at the earliest date.
        """
        from scripts.backfill_story_store import _seed_id_for, dedup_history

        history = {
            "2026-06-01": [_enriched("OpenAI ships GPT-5", "https://openai.com/gpt5-day1")],
            "2026-06-02": [_enriched("OpenAI ships GPT-5", "https://openai.com/gpt5-day2")],
            "2026-06-03": [_enriched("GPT-5 launch coverage", "https://openai.com/gpt5-day3")],
        }

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test_store.sqlite"
            store = StoryStore(db_path)

            with self._patch_embed({"GPT-5": (1.0, 0.0, 0.0), "gpt5": (1.0, 0.0, 0.0)}):
                seeds = dedup_history(history)

                # dedup_history must collapse 3 days → 1 real story
                self.assertEqual(len(seeds), 1,
                    f"Expected 1 collapsed story, got {len(seeds)}: {[s['first_seen'] for s in seeds]}")

                # Seed the store (mimics what main() does)
                from arxiv_assistant.hotspots.story import Story
                for seed in seeds:
                    story = Story(
                        story_id=_seed_id_for(seed["centroid"], seed["centroid_model_id"]),
                        canonical_item=None,  # type: ignore[arg-type]
                        items=[],
                        event_type="other",
                        headline=" ",
                        summary=" ",
                        centroid=seed["centroid"],
                        centroid_model_id=seed["centroid_model_id"],
                    )
                    store.seed_first_seen(story, seed["first_seen"])

            # Verify: exactly ONE story in the store
            rows = store._conn.execute("SELECT story_id, first_seen FROM stories").fetchall()
            self.assertEqual(len(rows), 1,
                f"Expected 1 story in StoryStore, found {len(rows)}: {[dict(r) for r in rows]}")

            # Verify: first_seen is the earliest date (2026-06-01), not a later duplicate
            self.assertEqual(rows[0]["first_seen"], "2026-06-01",
                f"Expected first_seen='2026-06-01', got '{rows[0]['first_seen']}'")

            store.close()

    def test_seed_first_seen_is_write_once(self) -> None:
        """Calling seed_first_seen twice for the same story_id is a no-op (write-once)."""
        from scripts.backfill_story_store import _seed_id_for, dedup_history

        history = {"2026-06-05": [_enriched("One event", "https://x.com/e1")]}

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "test_store.sqlite"
            store = StoryStore(db_path)

            with self._patch_embed({"One event": (1.0, 0.0, 0.0)}):
                seeds = dedup_history(history)

            self.assertEqual(len(seeds), 1)
            seed = seeds[0]

            from arxiv_assistant.hotspots.story import Story

            def _make_shell(sid: str) -> Story:
                return Story(
                    story_id=sid,
                    canonical_item=None,  # type: ignore[arg-type]
                    items=[],
                    event_type="other",
                    headline=" ",
                    summary=" ",
                    centroid=seed["centroid"],
                    centroid_model_id=seed["centroid_model_id"],
                )

            sid = _seed_id_for(seed["centroid"], seed["centroid_model_id"])

            # First seed — should persist first_seen = 2026-06-05
            store.seed_first_seen(_make_shell(sid), "2026-06-05")
            # Second call with a LATER date — write-once, should remain 2026-06-05
            store.seed_first_seen(_make_shell(sid), "2026-06-10")

            row = store._conn.execute(
                "SELECT first_seen FROM stories WHERE story_id=?", (sid,)
            ).fetchone()
            self.assertIsNotNone(row)
            # write-once: first_seen must not be overwritten by the later date
            self.assertEqual(row["first_seen"], "2026-06-05")

            store.close()


if __name__ == "__main__":
    unittest.main()
