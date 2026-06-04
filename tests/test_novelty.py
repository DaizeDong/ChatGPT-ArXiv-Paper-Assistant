from __future__ import annotations

import unittest
from datetime import date

from arxiv_assistant.hotspots import novelty
from arxiv_assistant.hotspots.enrich import EnrichedItem
from arxiv_assistant.hotspots.story import Story
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem


def _ev(title, *, added_at, source_tier, verified_first_date=None, arxiv_id="",
        entities=None):
    md = {}
    if arxiv_id:
        md["arxiv_id"] = arxiv_id
    item = HotspotItem(
        source_id="s", source_name="S", source_role="news", source_type="news",
        title=title, summary=title, url=f"https://e/{title}",
        canonical_url=f"https://e/{title}", published_at="2026-06-02T00:00:00+00:00",
        tags=[], authors=[], metadata=md,
    )
    item.verified_first_date = verified_first_date
    ei = EnrichedItem(item=item, event_type="product_release",
                      entities=entities or [], summary=title, importance=5)
    ei.added_at = added_at          # Stage-0 ledger field
    ei.source_tier = source_tier    # Stage-0 ledger field
    return ei


def _story(evidence, *, last_surfaced, surfaced_verified_max=None,
           surfaced_entity_names=None, surfaced_max_tier=0,
           arxiv_versions=None, surfaced_arxiv_versions=None, entity_names=None):
    canon = evidence[0] if evidence else _ev("seed", added_at="2026-05-01", source_tier=1)
    s = Story(
        story_id="persist1",
        canonical_item=canon,
        items=list(evidence),
        event_type="product_release",
        entity_names=set(entity_names or set()),
    )
    s.status = "ONGOING"
    s.last_surfaced = last_surfaced
    s.surfaced_verified_max = surfaced_verified_max
    s.surfaced_entity_names = set(surfaced_entity_names or set())
    s.surfaced_max_tier = surfaced_max_tier
    s.arxiv_versions = dict(arxiv_versions or {})
    s.surfaced_arxiv_versions = dict(surfaced_arxiv_versions or {})
    s._ledger = list(evidence)  # backing list used by the helpers below
    return s


# Stage-0 provides evidence_added_since/evidence_before on Story; for these unit tests
# we exercise novelty.resurface against a Story whose helpers split `_ledger` by added_at.
def _install_helpers():
    def added_since(self, last):
        return [e for e in getattr(self, "_ledger", self.items) if last is None or e.added_at > last]

    def before(self, last):
        return [e for e in getattr(self, "_ledger", self.items) if last is not None and e.added_at <= last]

    Story.evidence_added_since = added_since
    Story.evidence_before = before


class TestResurface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_helpers()

    # ---- NOT triggers ----
    def test_url_churn_only_is_false(self) -> None:
        # New evidence is same tier, same/earlier date, no new entity, no new version.
        before = _ev("e1", added_at="2026-06-01", source_tier=3, verified_first_date="2026-06-01")
        churn = _ev("e2", added_at="2026-06-02", source_tier=3, verified_first_date="2026-06-01")
        s = _story([before, churn], last_surfaced="2026-06-01",
                   surfaced_verified_max=date(2026, 6, 1), surfaced_max_tier=3)
        self.assertFalse(novelty.resurface(s))

    def test_subday_jitter_is_false(self) -> None:
        # Same UTC day, only H:M:S differs → floor_to_utc_day absorbs it.
        before = _ev("e1", added_at="2026-06-01", source_tier=3,
                     verified_first_date="2026-06-01T02:00:00+00:00")
        jitter = _ev("e2", added_at="2026-06-02", source_tier=3,
                     verified_first_date="2026-06-01T23:30:00+00:00")
        s = _story([before, jitter], last_surfaced="2026-06-01",
                   surfaced_verified_max=date(2026, 6, 1), surfaced_max_tier=3)
        self.assertFalse(novelty.resurface(s))

    def test_same_tier_more_evidence_is_false(self) -> None:
        before = _ev("e1", added_at="2026-06-01", source_tier=2, verified_first_date="2026-06-01")
        more = _ev("e2", added_at="2026-06-02", source_tier=2, verified_first_date="2026-06-01")
        s = _story([before, more], last_surfaced="2026-06-01",
                   surfaced_verified_max=date(2026, 6, 1), surfaced_max_tier=2)
        self.assertFalse(novelty.resurface(s))

    # ---- T1: tier jump ----
    def test_t1_tier_jump_is_true(self) -> None:
        before = _ev("e1", added_at="2026-06-01", source_tier=2, verified_first_date="2026-06-01")
        official = _ev("e2", added_at="2026-06-02", source_tier=7, verified_first_date="2026-06-01")
        s = _story([before, official], last_surfaced="2026-06-01",
                   surfaced_verified_max=date(2026, 6, 1), surfaced_max_tier=2)
        self.assertTrue(novelty.resurface(s))

    # ---- T2: later gate_date OR new arxiv version ----
    def test_t2_later_gate_date_is_true(self) -> None:
        before = _ev("e1", added_at="2026-06-01", source_tier=3, verified_first_date="2026-06-01")
        later = _ev("e2", added_at="2026-06-03", source_tier=3, verified_first_date="2026-06-03")
        s = _story([before, later], last_surfaced="2026-06-01",
                   surfaced_verified_max=date(2026, 6, 1), surfaced_max_tier=3)
        self.assertTrue(novelty.resurface(s))

    def test_t2_new_arxiv_version_is_true(self) -> None:
        before = _ev("e1", added_at="2026-06-01", source_tier=3,
                     verified_first_date="2026-06-01", arxiv_id="2606.00001")
        s = _story([before], last_surfaced="2026-06-01",
                   surfaced_verified_max=date(2026, 6, 1), surfaced_max_tier=3,
                   arxiv_versions={"2606.00001": 3}, surfaced_arxiv_versions={"2606.00001": 2})
        self.assertTrue(novelty.resurface(s))

    # ---- T3: new named entity ----
    def test_t3_new_entity_is_true(self) -> None:
        before = _ev("e1", added_at="2026-06-01", source_tier=3, verified_first_date="2026-06-01")
        s = _story([before], last_surfaced="2026-06-01",
                   surfaced_verified_max=date(2026, 6, 1), surfaced_max_tier=3,
                   entity_names={"openai", "nvidia"},
                   surfaced_entity_names={"openai"})
        self.assertTrue(novelty.resurface(s))


class TestResurge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_helpers()

    def _old_story(self, **kw):
        # gate_date 60 days old → "old" item (exceeds max_age).
        ev = _ev("old", added_at="2026-04-01", source_tier=3,
                 verified_first_date="2026-04-01", arxiv_id="2604.00001")
        s = _story([ev], last_surfaced=None,
                   surfaced_verified_max=date(2026, 4, 1), surfaced_max_tier=3,
                   arxiv_versions=kw.get("arxiv_versions", {"2604.00001": 1}),
                   surfaced_arxiv_versions=kw.get("surfaced_arxiv_versions", {"2604.00001": 1}))
        s.resurged_at = kw.get("resurged_at")
        s.surfaced_resurged_at = kw.get("surfaced_resurged_at")
        s._today_competitors = kw.get("today_competitors", 0)
        return s

    def test_not_old_returns_false(self) -> None:
        # Fresh story (gate_date within max_age) is never a resurge candidate.
        ev = _ev("fresh", added_at="2026-06-02", source_tier=3, verified_first_date="2026-06-02")
        s = _story([ev], last_surfaced=None, surfaced_verified_max=date(2026, 6, 2))
        s.resurged_at = None
        s.surfaced_resurged_at = None
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
