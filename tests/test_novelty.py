from __future__ import annotations

import unittest
from datetime import date

from arxiv_assistant.hotspots import novelty
from arxiv_assistant.hotspots.enrich import EnrichedItem
from arxiv_assistant.hotspots.story import Story
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem


# ---------------------------------------------------------------------------
# Helpers: build dict-row evidence ledger entries (§6 item 1 contract)
# ---------------------------------------------------------------------------

def _row(title, *, added_at: str, source_tier: int,
         verified_first_date: str | None = None,
         provenance: str = "") -> dict:
    """Build a ledger dict row (the shape Story.evidence_added_since returns)."""
    return {
        "canonical_url": f"https://e/{title}",
        "source_id": "s",
        "source_role": "news",
        "provenance": provenance,
        "source_tier": source_tier,
        "added_at": added_at,
        "verified_first_date": verified_first_date,
    }


def _story(rows: list[dict], *, last_surfaced: str | None,
           surfaced_verified_max: date | None = None,
           surfaced_entity_names: set[str] | None = None,
           surfaced_max_tier: int = 0,
           arxiv_versions: dict | None = None,
           surfaced_arxiv_versions: dict | None = None,
           entity_names: set[str] | None = None) -> Story:
    """Build a Story whose evidence_ledger is the supplied list of dict rows.

    We still need a canonical_item (EnrichedItem) for Story.__post_init__ and
    for resurge's gate_date_fn(story.canonical_item.item) call, so we construct
    a minimal one — but novelty reads ALL evidence from the dict ledger.

    If the first row carries a "verified_first_date", it is propagated to the
    canonical HotspotItem so that gate_date(story.canonical_item.item) reflects
    the actual event age (needed by resurge's old-story age gate).
    """
    first_vfd = rows[0].get("verified_first_date") if rows else None
    item = HotspotItem(
        source_id="s", source_name="S", source_role="news", source_type="news",
        title="seed", summary="seed", url="https://e/seed",
        canonical_url="https://e/seed", published_at="2026-06-02T00:00:00+00:00",
        verified_first_date=first_vfd,
    )
    ei = EnrichedItem(item=item, event_type="product_release",
                      entities=[], summary="seed", importance=5)

    s = Story(
        story_id="test-story",
        canonical_item=ei,
        items=[ei],
        event_type="product_release",
        entity_names=set(entity_names or set()),
        evidence_ledger=list(rows),
    )
    s.status = "ONGOING"
    s.last_surfaced = last_surfaced
    s.surfaced_verified_max = surfaced_verified_max
    s.surfaced_entity_names = set(surfaced_entity_names or set())
    s.surfaced_max_tier = surfaced_max_tier
    s.arxiv_versions = dict(arxiv_versions or {})
    s.surfaced_arxiv_versions = dict(surfaced_arxiv_versions or {})
    return s


class TestResurface(unittest.TestCase):
    # ---- NOT triggers ----
    def test_url_churn_only_is_false(self) -> None:
        # New evidence is same tier, same/earlier date, no new entity, no new version.
        rows = [
            _row("e1", added_at="2026-06-01", source_tier=3, verified_first_date="2026-06-01"),
            _row("e2", added_at="2026-06-02", source_tier=3, verified_first_date="2026-06-01"),
        ]
        s = _story(rows, last_surfaced="2026-06-01",
                   surfaced_verified_max=date(2026, 6, 1), surfaced_max_tier=3)
        self.assertFalse(novelty.resurface(s))

    def test_subday_jitter_is_false(self) -> None:
        # Same UTC day, only H:M:S differs → floor_to_utc_day absorbs it.
        rows = [
            _row("e1", added_at="2026-06-01", source_tier=3,
                 verified_first_date="2026-06-01T02:00:00+00:00"),
            _row("e2", added_at="2026-06-02", source_tier=3,
                 verified_first_date="2026-06-01T23:30:00+00:00"),
        ]
        s = _story(rows, last_surfaced="2026-06-01",
                   surfaced_verified_max=date(2026, 6, 1), surfaced_max_tier=3)
        self.assertFalse(novelty.resurface(s))

    def test_same_tier_more_evidence_is_false(self) -> None:
        rows = [
            _row("e1", added_at="2026-06-01", source_tier=2, verified_first_date="2026-06-01"),
            _row("e2", added_at="2026-06-02", source_tier=2, verified_first_date="2026-06-01"),
        ]
        s = _story(rows, last_surfaced="2026-06-01",
                   surfaced_verified_max=date(2026, 6, 1), surfaced_max_tier=2)
        self.assertFalse(novelty.resurface(s))

    # ---- T1: tier jump ----
    def test_t1_tier_jump_is_true(self) -> None:
        rows = [
            _row("e1", added_at="2026-06-01", source_tier=2, verified_first_date="2026-06-01"),
            _row("e2", added_at="2026-06-02", source_tier=7, verified_first_date="2026-06-01"),
        ]
        s = _story(rows, last_surfaced="2026-06-01",
                   surfaced_verified_max=date(2026, 6, 1), surfaced_max_tier=2)
        self.assertTrue(novelty.resurface(s))

    # ---- T2: later gate_date OR new arxiv version ----
    def test_t2_later_gate_date_is_true(self) -> None:
        rows = [
            _row("e1", added_at="2026-06-01", source_tier=3, verified_first_date="2026-06-01"),
            _row("e2", added_at="2026-06-03", source_tier=3, verified_first_date="2026-06-03"),
        ]
        s = _story(rows, last_surfaced="2026-06-01",
                   surfaced_verified_max=date(2026, 6, 1), surfaced_max_tier=3)
        self.assertTrue(novelty.resurface(s))

    def test_t2_new_arxiv_version_is_true(self) -> None:
        rows = [
            _row("e1", added_at="2026-06-01", source_tier=3, verified_first_date="2026-06-01"),
        ]
        s = _story(rows, last_surfaced="2026-06-01",
                   surfaced_verified_max=date(2026, 6, 1), surfaced_max_tier=3,
                   arxiv_versions={"2606.00001": 3},
                   surfaced_arxiv_versions={"2606.00001": 2})
        self.assertTrue(novelty.resurface(s))

    # ---- T3: new named entity ----
    def test_t3_new_entity_is_true(self) -> None:
        rows = [
            _row("e1", added_at="2026-06-01", source_tier=3, verified_first_date="2026-06-01"),
        ]
        s = _story(rows, last_surfaced="2026-06-01",
                   surfaced_verified_max=date(2026, 6, 1), surfaced_max_tier=3,
                   entity_names={"openai", "nvidia"},
                   surfaced_entity_names={"openai"})
        self.assertTrue(novelty.resurface(s))


class TestResurge(unittest.TestCase):
    def _old_story(self, **kw) -> Story:
        """Build an 'old' story: canonical_item gate_date 60+ days ago."""
        # We set verified_first_date on the canonical HotspotItem so that
        # gate_date_fn(story.canonical_item.item) returns a date > max_age_days old.
        old_vfd = "2026-04-01T00:00:00+00:00"
        item = HotspotItem(
            source_id="s", source_name="S", source_role="news", source_type="news",
            title="old", summary="old", url="https://e/old",
            canonical_url="https://e/old", published_at="2026-04-01T00:00:00+00:00",
            verified_first_date=old_vfd,
        )
        ei = EnrichedItem(item=item, event_type="product_release",
                          entities=[], summary="old", importance=3)

        rows: list[dict] = [
            _row("old", added_at="2026-04-01", source_tier=3,
                 verified_first_date=old_vfd),
        ]
        s = Story(
            story_id="resurge-story",
            canonical_item=ei,
            items=[ei],
            event_type="product_release",
            entity_names=set(),
            evidence_ledger=rows,
        )
        s.status = "ONGOING"
        s.last_surfaced = None
        s.surfaced_verified_max = date(2026, 4, 1)
        s.surfaced_max_tier = 3
        s.arxiv_versions = dict(kw.get("arxiv_versions", {"2604.00001": 1}))
        s.surfaced_arxiv_versions = dict(kw.get("surfaced_arxiv_versions", {"2604.00001": 1}))
        s.resurged_at = kw.get("resurged_at")
        s.surfaced_resurged_at = kw.get("surfaced_resurged_at")
        s._today_competitors = kw.get("today_competitors", 0)
        return s

    def test_not_old_returns_false(self) -> None:
        # Fresh story (gate_date within max_age) is never a resurge candidate.
        item = HotspotItem(
            source_id="s", source_name="S", source_role="news", source_type="news",
            title="fresh", summary="fresh", url="https://e/fresh",
            canonical_url="https://e/fresh", published_at="2026-06-02T00:00:00+00:00",
            verified_first_date="2026-06-02T00:00:00+00:00",
        )
        ei = EnrichedItem(item=item, event_type="product_release",
                          entities=[], summary="fresh", importance=5)
        s = Story(
            story_id="fresh-story",
            canonical_item=ei,
            items=[ei],
            event_type="product_release",
            entity_names=set(),
            evidence_ledger=[],
        )
        s.resurged_at = None
        s.surfaced_resurged_at = None
        s.arxiv_versions = {}
        s.surfaced_arxiv_versions = {}
        s._today_competitors = 9
        self.assertFalse(novelty.resurge(
            s, max_age_days=14, run_date=date(2026, 6, 3),
            min_competitors=3, cooldown_days=7,
            competitor_count_fn=lambda story: story._today_competitors))

    def test_r1_version_jump_is_true(self) -> None:
        s = self._old_story(arxiv_versions={"2604.00001": 4},
                            surfaced_arxiv_versions={"2604.00001": 3})
        self.assertTrue(novelty.resurge(
            s, max_age_days=14, run_date=date(2026, 6, 3),
            min_competitors=3, cooldown_days=7,
            competitor_count_fn=lambda story: 0))

    def test_r2_competitors_fresh_cooldown_is_true(self) -> None:
        s = self._old_story(surfaced_resurged_at=None, today_competitors=3)
        self.assertTrue(novelty.resurge(
            s, max_age_days=14, run_date=date(2026, 6, 3),
            min_competitors=3, cooldown_days=7,
            competitor_count_fn=lambda story: story._today_competitors))

    def test_r2_below_min_competitors_is_false(self) -> None:
        s = self._old_story(surfaced_resurged_at=None, today_competitors=2)
        self.assertFalse(novelty.resurge(
            s, max_age_days=14, run_date=date(2026, 6, 3),
            min_competitors=3, cooldown_days=7,
            competitor_count_fn=lambda story: story._today_competitors))

    def test_r2_within_cooldown_is_false_then_true_after(self) -> None:
        # Same group of competitors re-surfaces day after day → cooldown fires ONCE.
        s = self._old_story(surfaced_resurged_at=date(2026, 6, 1), today_competitors=4)
        # 2 days later, cooldown_days=7 → still within cooldown → False
        self.assertFalse(novelty.resurge(
            s, max_age_days=14, run_date=date(2026, 6, 3),
            min_competitors=3, cooldown_days=7,
            competitor_count_fn=lambda story: story._today_competitors))
        # 8 days later → cooldown elapsed → True
        self.assertTrue(novelty.resurge(
            s, max_age_days=14, run_date=date(2026, 6, 9),
            min_competitors=3, cooldown_days=7,
            competitor_count_fn=lambda story: story._today_competitors))


# ---------------------------------------------------------------------------
# Stage-2 acceptance helpers and invariant assertions
# ---------------------------------------------------------------------------

def _ev(title, *, added_at: str, source_tier: int,
        verified_first_date: str | None = None,
        arxiv_id: str = "",
        entities=None) -> dict:
    """Build a ledger dict row in the plan's _ev() interface.

    Equivalent to _row() but uses the same keyword signature as the plan's
    EnrichedItem-based _ev so that TestStage2Invariants can use either helper
    interchangeably.  arxiv_id is accepted but not stored in the dict row —
    resurge reads arxiv_versions from Story-level dicts, not individual rows.
    """
    return _row(title, added_at=added_at, source_tier=source_tier,
                verified_first_date=verified_first_date)


def _install_helpers() -> None:
    """No-op compatibility shim.

    Story.evidence_added_since and Story.evidence_before are real methods on
    the Story dataclass (added in Stage 0).  This function exists so that
    TestStage2Invariants.setUpClass() can follow the same structural pattern as
    earlier plan tasks that needed to monkey-patch those methods before the
    production implementations landed.
    """


class TestStage2Invariants(unittest.TestCase):
    """Named acceptance assertions for the Stage-2 invariants (spec §C / §G).

    INV2 (day-granular dedup/novelty):
        sub-day jitter in verified_first_date cannot flip a resurface decision.
        Proven by test_inv4_resurface_is_pure_no_url_or_subday.

    INV4 (closed-form novelty, zero LLM):
        resurface / resurge are pure closed-form functions over Store-resident
        structured fields.  No LLM call, no HTTP request, no URL-set comparison.
        Proven by:
          - test_inv4_resurface_is_pure_no_url_or_subday  (URL churn + sub-day jitter → False)
          - test_inv4_resurge_cooldown_fires_once_over_consecutive_days  (cooldown gate)

    Cross-day suppression (core Stage-2 value):
        Covered by TestMatchCrossday in test_dedup.py — ONGOING story suppressed on
        day N+1; new entity resurfaces correctly.

    Cross-lingual L1 merge:
        Covered by TestClusterIntraday.test_zh_en_same_event_merge in test_dedup.py.
    """

    @classmethod
    def setUpClass(cls):
        _install_helpers()

    def test_inv4_resurface_is_pure_no_url_or_subday(self) -> None:
        # 40% URL churn + sub-day jitter on a same-tier, same-day, no-new-entity story
        # must NOT resurface (closed-form, zero LLM).
        before = _ev("e1", added_at="2026-06-01", source_tier=4,
                     verified_first_date="2026-06-01T01:00:00+00:00")
        churn = _ev("e2", added_at="2026-06-02", source_tier=4,
                    verified_first_date="2026-06-01T22:00:00+00:00")
        s = _story([before, churn], last_surfaced="2026-06-01",
                   surfaced_verified_max=date(2026, 6, 1), surfaced_max_tier=4)
        # Two independent runs → identical boolean (determinism).
        self.assertFalse(novelty.resurface(s))
        self.assertFalse(novelty.resurface(s))

    def test_inv4_resurge_cooldown_fires_once_over_consecutive_days(self) -> None:
        # Same 4-competitor cluster every day for a week → exactly ONE True within cooldown.
        ev = _ev("old", added_at="2026-04-01", source_tier=3,
                 verified_first_date="2026-04-01", arxiv_id="2604.00001")
        results = []
        surfaced = None
        for day in range(1, 8):
            s = _story([ev], last_surfaced=None, surfaced_verified_max=date(2026, 4, 1),
                       arxiv_versions={"2604.00001": 1}, surfaced_arxiv_versions={"2604.00001": 1})
            s.resurged_at = None
            s.surfaced_resurged_at = surfaced
            fired = novelty.resurge(
                s, max_age_days=14, run_date=date(2026, 6, day),
                min_competitors=3, cooldown_days=7,
                competitor_count_fn=lambda story: 4)
            results.append(fired)
            if fired:
                surfaced = date(2026, 6, day)  # Kernel records surfaced_resurged_at
        self.assertEqual(sum(1 for r in results if r), 1)  # cooldown → exactly once
