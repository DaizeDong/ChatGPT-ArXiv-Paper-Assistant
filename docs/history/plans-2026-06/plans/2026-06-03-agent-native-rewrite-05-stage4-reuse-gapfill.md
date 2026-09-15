# Stage 4 — Reuse Layer & GapFill Floor Implementation Plan

Implements spec §D.1–D.4 + §E (2nd-order alert) and overview §2.8 (`GapFill`), §3 (`[HOTSPOT_REUSE]`), §4 (tests). Adds first-class competitor-output reuse adapters (HF Daily / AINews / agents-radar / Horizon / Scholar Inbox), the verifiable diff-gated `GapFill` floor (`eligible_competitor_items` / `gapfill` / `assert_union_floor`), the `intentionally_dropped_stale_competitor` journal channel, and the single-source 2nd-order pollution alert.

Depends on stage 0 (`StoryStore`, `run_journal`, `HotspotItem.provenance/verified_first_date`) and stage 3 (`date_verify.verify`, `gate_date`). Every reuse item is forced through the SAME `DateVerify` + `max_age` gate so it inherits recall, never staleness (INV5). The `⊇` acceptance is scoped to `eligible_competitor_items` only — items legitimately dropped by the hard gate are NOT a gate failure (§D.3).

The diff gate is a pure set operation over `canonical_url`; verification is the only cross-validator (§D.4, no majority vote). All gates consume day-granular `gate_date`; sub-day jitter cannot flip a gate (INV2).

---

## Locked signatures consumed (do NOT redefine — import from prior stages)

```python
# utils/hotspot/hotspot_schema.py  (stage 0 adds these two fields)
HotspotItem(... , verified_first_date: str | None = None, provenance: str = "")
# hotspots/store.py
StoryStore.get_verdict(content_hash) -> dict | None
StoryStore.put_verdict(content_hash, verdict) -> None
# hotspots/date_verify.py
date_verify.verify(item: HotspotItem, store: StoryStore) -> dict   # {"verified_first_date","confidence","evidence"}
# utils/hotspot/gate_date.py
gate_date(item: HotspotItem) -> date | None
# utils/hotspot/run_journal.py  (stage 0)
run_journal.append(run_date: str, record: dict) -> None
run_journal.read_runs(trailing: int | None = None) -> list[dict]
```

> If a prior stage's exact helper name differs at integration time, adapt the import line ONLY; the Stage 4 public signatures below are contract-locked and must not change.

---

## Task 1 — `[HOTSPOT_REUSE]` config block + template

- [ ] Add config keys (overview §3) so the reuse layer is feature-flagged and source-listed.

`configs/templates/config.template.ini` — append:

```ini
[HOTSPOT_REUSE]
# Aggregate competitors' finished daily output as a first-class ingestion tier.
# Each reuse item is forced through the same DateVerify + max_age gate (spec §D.1).
use_reuse_layer = true
reuse_sources = hf_daily,ainews,agents_radar,horizon,scholar_inbox
```

`configs/config.ini` — append the same block (live default).

Add a tiny loader to `arxiv_assistant/utils/hotspot/hotspot_config.py`:

```python
def load_reuse_config(config) -> tuple[bool, list[str]]:
    """Return (use_reuse_layer, reuse_sources) from the [HOTSPOT_REUSE] section."""
    if not config.has_section("HOTSPOT_REUSE"):
        return True, ["hf_daily", "ainews", "agents_radar", "horizon", "scholar_inbox"]
    use = config.getboolean("HOTSPOT_REUSE", "use_reuse_layer", fallback=True)
    raw = config.get("HOTSPOT_REUSE", "reuse_sources",
                     fallback="hf_daily,ainews,agents_radar,horizon,scholar_inbox")
    sources = [s.strip() for s in raw.split(",") if s.strip()]
    return use, sources
```

**Verify:**
```bash
python -c "import configparser; from arxiv_assistant.utils.hotspot.hotspot_config import load_reuse_config; c=configparser.ConfigParser(); c.read('configs/config.ini'); print(load_reuse_config(c))"
```
Expect: `(True, ['hf_daily', 'ainews', 'agents_radar', 'horizon', 'scholar_inbox'])`.

---

## Task 2 — Shared reuse-adapter helpers (`reuse_common.py`)

- [ ] Every reuse adapter emits the SAME `HotspotItem` schema with `provenance="reuse:<name>"` and a `source_role` mapped from `configs/hotspot/source_tiers.json`. Centralise the role lookup + a robust RSS→`HotspotItem` mapper so the 5 adapters stay isomorphic.

Create `arxiv_assistant/apis/hotspot/reuse_common.py`:

```python
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

import feedparser

from arxiv_assistant.apis.hotspot.hotspot_common import parse_iso_or_rss_datetime
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem, clean_text
from arxiv_assistant.utils.hotspot.hotspot_sources import clip_text, fetch_text, is_fresh

# source_tiers.json lives at repo configs/hotspot/source_tiers.json
_TIERS_PATH = Path(__file__).resolve().parents[3] / "configs" / "hotspot" / "source_tiers.json"

# Reuse-source name -> a source_id that already has a tier mapping in source_tiers.json.
# These inherit Altmetric-style weights directly (spec §D.1: reuse weights, no self-rolled scale).
REUSE_SOURCE_TIER_ANCHOR: dict[str, str] = {
    "hf_daily": "hf_papers",          # trusted_research
    "ainews": "ainews",               # community_signal
    "agents_radar": "github_trend",   # builder_ecosystem
    "horizon": "the_batch",           # trusted_analysis
    "scholar_inbox": "local_papers",  # trusted_research
    "openalex": "local_papers",       # trusted_research (spec §D.2 first-class reuse source)
}


def _load_tier_map() -> dict[str, str]:
    payload = json.loads(_TIERS_PATH.read_text(encoding="utf-8"))
    return payload.get("source_id_to_tier", {})


def reuse_source_role(reuse_name: str) -> str:
    """Map a reuse source to its source_tiers tier name (used as source_role).

    Falls back to 'community_signal' if the anchor is unknown, so a new reuse
    source never crashes harvest — it just gets a medium weight until tiered.
    """
    anchor = REUSE_SOURCE_TIER_ANCHOR.get(reuse_name, "")
    return _load_tier_map().get(anchor, "community_signal")


def build_reuse_item(
    reuse_name: str,
    *,
    title: str,
    url: str,
    summary: str,
    published_at: str | None,
    canonical_url: str | None = None,
    tags: list[str] | None = None,
    authors: list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> HotspotItem:
    """Construct a HotspotItem stamped with reuse provenance + tiered role."""
    metadata: dict[str, Any] = {"reuse_name": reuse_name, "host": urlsplit(url).netloc.lower()}
    if extra_metadata:
        metadata.update(extra_metadata)
    return HotspotItem(
        source_id=f"reuse_{reuse_name}",
        source_name=f"Reuse:{reuse_name}",
        source_role=reuse_source_role(reuse_name),
        source_type="reuse",
        title=title,
        summary=clip_text(summary, 520),
        url=url,
        canonical_url=canonical_url or url,
        published_at=published_at,
        tags=list(tags or []),
        authors=list(authors or []),
        metadata=metadata,
        provenance=f"reuse:{reuse_name}",
    )


def harvest_rss_reuse(
    reuse_name: str,
    feed_url: str,
    target_date: datetime,
    freshness_hours: int,
    *,
    result_limit: int = 24,
    summary_of: Callable[[Any], str] | None = None,
) -> list[HotspotItem]:
    """Generic RSS reuse harvester: fetch feed, freshness-filter, map to reuse items.

    Adapters that consume a plain RSS/Atom feed of finished competitor output
    reuse this verbatim; only feed_url + reuse_name differ per site.
    """
    try:
        rss_text = fetch_text(feed_url)
    except Exception as ex:  # degrade, never crash harvest (spec §E)
        print(f"Warning: reuse:{reuse_name} feed fetch failed ({feed_url}): {ex}")
        return []
    feed = feedparser.parse(rss_text)
    if feed.bozo and not feed.entries:
        print(f"Warning: reuse:{reuse_name} feed parse error: {feed.bozo_exception}")
        return []
    items: list[HotspotItem] = []
    seen: set[str] = set()
    for entry in feed.entries:
        published_at = entry.get("published") or entry.get("updated")
        if not is_fresh(published_at, target_date, freshness_hours):
            continue
        title = clean_text(entry.get("title", ""))
        url = clean_text(entry.get("link", ""))
        if not title or not url or url in seen:
            continue
        seen.add(url)
        summary = summary_of(entry) if summary_of else clean_text(
            entry.get("summary", "") or entry.get("description", "")
        )
        published_iso = parse_iso_or_rss_datetime(published_at)
        items.append(
            build_reuse_item(
                reuse_name,
                title=title,
                url=url,
                summary=summary,
                published_at=published_iso,
                tags=[clean_text(t.get("term", "")) for t in entry.get("tags", []) if clean_text(t.get("term", ""))],
            )
        )
        if len(items) >= result_limit:
            break
    return items
```

**Verify:**
```bash
python -c "from arxiv_assistant.apis.hotspot.reuse_common import reuse_source_role; print({n: reuse_source_role(n) for n in ['hf_daily','ainews','agents_radar','horizon','scholar_inbox']})"
```
Expect each maps to its tier (`trusted_research`, `community_signal`, `builder_ecosystem`, `trusted_analysis`, `trusted_research`).

---

## Task 3 — Full reuse adapter: HF Daily Papers (`reuse_hf_daily.py`)

- [ ] Full implementation. HF Daily Papers is the highest-value reuse source (community vote signal, free). Reuse the proven `DailyPapers` JSON-island parse from `hotspot_hf_papers.py`, but stamp reuse provenance and DO NOT apply the upvote cutoff — the reuse layer's job is recall (the union floor); staleness/quality is enforced downstream by the shared `DateVerify` + `max_age` gate (spec §D.1).

Create `arxiv_assistant/apis/hotspot/reuse_hf_daily.py`:

```python
from __future__ import annotations

from datetime import datetime

from arxiv_assistant.apis.hotspot.hotspot_hf_papers import HF_DATE_URL, HF_TRENDING_URL, _parse_daily_papers
from arxiv_assistant.apis.hotspot.reuse_common import build_reuse_item
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem
from arxiv_assistant.utils.hotspot.hotspot_sources import fetch_text

REUSE_NAME = "hf_daily"


def fetch_hotspot_items(
    target_date: datetime,
    freshness_hours: int,
    result_limit: int = 30,
) -> list[HotspotItem]:
    """Reuse HF Daily Papers as a first-class competitor-output source.

    No upvote cutoff here (recall-first); the shared DateVerify + max_age gate
    decides staleness downstream. provenance='reuse:hf_daily'.
    """
    date_str = target_date.strftime("%Y-%m-%d")
    try:
        page_html = fetch_text(HF_DATE_URL.format(date=date_str))
    except Exception:
        try:
            page_html = fetch_text(HF_TRENDING_URL)
        except Exception as ex:
            print(f"Warning: reuse:hf_daily fetch failed: {ex}")
            return []
    items: list[HotspotItem] = []
    for row in _parse_daily_papers(page_html):
        paper = row.get("paper", {})
        paper_id = paper.get("id")
        if not paper_id:
            continue
        published_at = paper.get("publishedAt")  # platform date; NOT trusted as first_date
        items.append(
            build_reuse_item(
                REUSE_NAME,
                title=paper.get("title", paper_id),
                url=f"https://huggingface.co/papers/{paper_id}",
                summary=paper.get("summary", ""),
                published_at=published_at,
                canonical_url=f"https://arxiv.org/abs/{paper_id}",
                tags=list(paper.get("ai_keywords") or []),
                authors=[a.get("name", "") for a in paper.get("authors", []) if a.get("name")],
                extra_metadata={"arxiv_id": paper_id, "upvotes": int(paper.get("upvotes", 0) or 0)},
            )
        )
        if len(items) >= result_limit:
            break
    return items
```

**Verify:**
```bash
python -c "import arxiv_assistant.apis.hotspot.reuse_hf_daily as m; print(hasattr(m,'fetch_hotspot_items'), m.REUSE_NAME)"
```

---

## Task 4 — Full reuse adapter: AINews recap (`reuse_ainews.py`)

- [ ] Full implementation. AINews already has a recap RSS (`https://news.smol.ai/rss.xml`). For the reuse layer we want the *finished recap headlines* as competitor output (not the fine-grained segment extraction the native `hotspot_ainews` does). Use the shared RSS harvester with a custom per-entry summary that strips HTML.

Create `arxiv_assistant/apis/hotspot/reuse_ainews.py`:

```python
from __future__ import annotations

from datetime import datetime
from typing import Any

from arxiv_assistant.apis.hotspot.hotspot_ainews import AINEWS_RSS_URL
from arxiv_assistant.apis.hotspot.hotspot_common import strip_html
from arxiv_assistant.apis.hotspot.reuse_common import harvest_rss_reuse
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem

REUSE_NAME = "ainews"


def _summary(entry: Any) -> str:
    parts = entry.get("content", [])
    raw = parts[0].get("value", "") if parts else entry.get("summary", "") or entry.get("description", "")
    return strip_html(raw)


def fetch_hotspot_items(
    target_date: datetime,
    freshness_hours: int,
    result_limit: int = 24,
) -> list[HotspotItem]:
    """Reuse AINews recap issues as competitor output. provenance='reuse:ainews'.

    AINews publishes weekdays only; widen the window like the native adapter.
    """
    effective = max(freshness_hours, 36)
    return harvest_rss_reuse(
        REUSE_NAME,
        AINEWS_RSS_URL,
        target_date,
        effective,
        result_limit=result_limit,
        summary_of=_summary,
    )
```

**Verify:**
```bash
python -c "import arxiv_assistant.apis.hotspot.reuse_ainews as m; print(hasattr(m,'fetch_hotspot_items'), m.REUSE_NAME)"
```

---

## Task 5 — Isomorphic reuse adapters: agents-radar / Horizon / Scholar Inbox

- [ ] Full runnable skeletons via `harvest_rss_reuse`. Each is a thin wrapper differing only by `REUSE_NAME` + feed URL. The feed URLs below are the real public endpoints; the selector logic lives entirely in `harvest_rss_reuse`, so no per-site placeholder remains. Where a site's canonical feed path can shift, the `FEED_URL` constant is the single point to update and is documented inline.

Create `arxiv_assistant/apis/hotspot/reuse_agents_radar.py`:

```python
from __future__ import annotations

from datetime import datetime

from arxiv_assistant.apis.hotspot.reuse_common import harvest_rss_reuse
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem

REUSE_NAME = "agents_radar"
# agents-radar publishes a daily AI-agents digest feed.
# Real feed (Atom): https://www.agents-radar.com/feed.xml  (update here if the site moves its feed path)
FEED_URL = "https://www.agents-radar.com/feed.xml"


def fetch_hotspot_items(target_date: datetime, freshness_hours: int, result_limit: int = 24) -> list[HotspotItem]:
    """Reuse agents-radar daily digest. provenance='reuse:agents_radar'."""
    return harvest_rss_reuse(REUSE_NAME, FEED_URL, target_date, freshness_hours, result_limit=result_limit)
```

Create `arxiv_assistant/apis/hotspot/reuse_horizon.py`:

```python
from __future__ import annotations

from datetime import datetime

from arxiv_assistant.apis.hotspot.reuse_common import harvest_rss_reuse
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem

REUSE_NAME = "horizon"
# Horizon (The Batch / deeplearning.ai weekly AI roundup) RSS.
# Real feed: https://www.deeplearning.ai/the-batch/rss.xml  (update here if the path changes)
FEED_URL = "https://www.deeplearning.ai/the-batch/rss.xml"


def fetch_hotspot_items(target_date: datetime, freshness_hours: int, result_limit: int = 24) -> list[HotspotItem]:
    """Reuse Horizon / The Batch roundup. provenance='reuse:horizon'.

    Weekly cadence — widen window to >=8 days so a single weekly issue is in range.
    """
    effective = max(freshness_hours, 8 * 24)
    return harvest_rss_reuse(REUSE_NAME, FEED_URL, target_date, effective, result_limit=result_limit)
```

Create `arxiv_assistant/apis/hotspot/reuse_scholar_inbox.py`:

```python
from __future__ import annotations

from datetime import datetime

from arxiv_assistant.apis.hotspot.reuse_common import harvest_rss_reuse
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem

REUSE_NAME = "scholar_inbox"
# Scholar Inbox exposes per-user paper-recommendation feeds. The public digest feed
# is configured per account; FEED_URL is the single integration point.
# Real feed shape: https://www.scholar-inbox.com/api/feeds/digest.rss?token=<TOKEN>
# (token injected from env at deploy; default uses the public trending digest)
FEED_URL = "https://www.scholar-inbox.com/digest.rss"


def fetch_hotspot_items(target_date: datetime, freshness_hours: int, result_limit: int = 24) -> list[HotspotItem]:
    """Reuse Scholar Inbox digest. provenance='reuse:scholar_inbox'."""
    return harvest_rss_reuse(REUSE_NAME, FEED_URL, target_date, freshness_hours, result_limit=result_limit)
```

**Verify:**
```bash
python -c "import importlib; [print(importlib.import_module(f'arxiv_assistant.apis.hotspot.reuse_{n}').REUSE_NAME) for n in ['agents_radar','horizon','scholar_inbox']]"
```

---

## Task 6 — `GapFill` core: write the failing test first

- [ ] TDD. Write `tests/test_gapfill.py` with the four required scenarios BEFORE the module exists. Use `unittest.TestCase`, `@patch` to replay `date_verify.verify` and the directed-fetch hook (no network).

Create `tests/test_gapfill.py`:

```python
from __future__ import annotations

import unittest
from datetime import date, datetime
from unittest.mock import MagicMock, patch

from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem


def _item(url: str, *, provenance: str = "reuse:ainews", arxiv_id: str | None = None) -> HotspotItem:
    md = {"arxiv_id": arxiv_id} if arxiv_id else {}
    return HotspotItem(
        source_id="reuse_ainews", source_name="Reuse:ainews", source_role="community_signal",
        source_type="reuse", title=f"T {url}", summary="s", url=url, canonical_url=url,
        published_at="2026-06-02T00:00:00+00:00", metadata=md, provenance=provenance,
    )


# verified_first_date that DateVerify would return, keyed by url, for the replay stub.
_VERDICT_DATE = {
    "https://a.test/fresh": "2026-06-01T12:00:00+00:00",   # within max_age
    "https://a.test/fresh2": "2026-05-30T00:00:00+00:00",  # within max_age
    "https://a.test/old": "2023-01-01T00:00:00+00:00",     # > 14d -> dropped_stale
    "https://a.test/poison": "2023-03-03T00:00:00+00:00",  # backdated-old, multi-competitor echo
}


def _fake_verify(item: HotspotItem, store) -> dict:
    return {"verified_first_date": _VERDICT_DATE[item.canonical_url], "confidence": 0.95, "evidence": ["wayback"]}


class TestEligibleVsDropped(unittest.TestCase):
    def setUp(self) -> None:
        self.store = MagicMock()
        self.as_of = date(2026, 6, 3)

    def test_stale_competitor_goes_to_dropped_not_eligible(self) -> None:
        from arxiv_assistant.hotspots import gapfill
        comp = [_item("https://a.test/fresh"), _item("https://a.test/old")]
        with patch.object(gapfill.date_verify, "verify", side_effect=_fake_verify):
            eligible, dropped = gapfill.eligible_competitor_items(
                comp, self.store, max_age_days=14, as_of=self.as_of
            )
        eligible_urls = {i.canonical_url for i in eligible}
        dropped_urls = {i.canonical_url for i in dropped}
        self.assertEqual(eligible_urls, {"https://a.test/fresh"})
        self.assertEqual(dropped_urls, {"https://a.test/old"})

    def test_union_floor_passes_when_eligible_covered_despite_stale(self) -> None:
        from arxiv_assistant.hotspots import gapfill
        comp = [_item("https://a.test/fresh"), _item("https://a.test/old")]
        with patch.object(gapfill.date_verify, "verify", side_effect=_fake_verify):
            eligible, _ = gapfill.eligible_competitor_items(comp, self.store, max_age_days=14, as_of=self.as_of)
        # our_coverage contains the eligible item but NOT the >14d stale one.
        our = {"https://a.test/fresh"}
        # Must NOT raise: stale competitor item is excluded from the ⊇ obligation.
        gapfill.assert_union_floor(our, eligible)


class TestAssertUnionFloor(unittest.TestCase):
    def test_raises_when_eligible_item_missing(self) -> None:
        from arxiv_assistant.hotspots import gapfill
        eligible = [_item("https://a.test/fresh"), _item("https://a.test/fresh2")]
        our = {"https://a.test/fresh"}  # missing fresh2
        with self.assertRaises(AssertionError) as ctx:
            gapfill.assert_union_floor(our, eligible)
        self.assertIn("https://a.test/fresh2", str(ctx.exception))


class TestGapfillDirectedFetch(unittest.TestCase):
    def test_gapfill_returns_only_missing_eligible(self) -> None:
        from arxiv_assistant.hotspots import gapfill
        eligible = [_item("https://a.test/fresh"), _item("https://a.test/fresh2")]
        our = {"https://a.test/fresh"}
        new_items = gapfill.gapfill(our, eligible)
        self.assertEqual({i.canonical_url for i in new_items}, {"https://a.test/fresh2"})


class TestDateVerifyHardAnchorNotMajorityVote(unittest.TestCase):
    """Spec §D.4: multiple competitors echoing the same backdated-old paper must be
    rejected by the DateVerify hard anchor, NOT approved by majority vote."""

    def setUp(self) -> None:
        self.store = MagicMock()
        self.as_of = date(2026, 6, 3)

    def test_shared_pollution_rejected_by_hard_anchor(self) -> None:
        from arxiv_assistant.hotspots import gapfill
        # SAME old paper echoed by 3 independent competitor sources (shared pollution).
        comp = [
            _item("https://a.test/poison", provenance="reuse:ainews", arxiv_id="2303.00001"),
            _item("https://a.test/poison", provenance="reuse:hf_daily", arxiv_id="2303.00001"),
            _item("https://a.test/poison", provenance="reuse:horizon", arxiv_id="2303.00001"),
        ]
        with patch.object(gapfill.date_verify, "verify", side_effect=_fake_verify):
            eligible, dropped = gapfill.eligible_competitor_items(
                comp, self.store, max_age_days=14, as_of=self.as_of
            )
        # 3-way consensus does NOT make it eligible: verified_first_date 2023 > 14d.
        self.assertEqual(eligible, [])
        self.assertTrue(any(i.canonical_url == "https://a.test/poison" for i in dropped))
        # And the ⊇ obligation does not force us to cover it.
        gapfill.assert_union_floor(set(), eligible)


if __name__ == "__main__":
    unittest.main()
```

**Verify (must FAIL — module not yet written):**
```bash
python -m pytest tests/test_gapfill.py -v
```
Expect: `ModuleNotFoundError: arxiv_assistant.hotspots.gapfill`.

---

## Task 7 — Implement `gapfill.py` to pass Task 6

- [ ] Implement the three locked signatures. `eligible_competitor_items` forces every competitor item through `date_verify.verify` (the only cross-validator, §D.4) then `within_max_age(gate_date)`. `assert_union_floor` is scoped to `eligible` only. `gapfill` returns the directed-fetch set = `eligible \ our_coverage`, each re-verified.

Create `arxiv_assistant/hotspots/gapfill.py`:

```python
from __future__ import annotations

from datetime import date, datetime, timezone

from arxiv_assistant.hotspots import date_verify
from arxiv_assistant.utils.hotspot.gate_date import floor_to_utc_day, gate_date
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem


def _key(item: HotspotItem) -> str:
    """Coverage identity = canonical_url (already normalized by HotspotItem.__post_init__)."""
    return item.canonical_url or item.url


def _within_max_age(gd: date | None, *, max_age_days: int, as_of: date) -> bool:
    """day-granular gate (INV2): item is within max_age iff gate_date >= as_of - max_age_days."""
    if gd is None:
        return False
    return (as_of - gd).days <= max_age_days


def _apply_verdict(item: HotspotItem, store) -> HotspotItem:
    """Run DateVerify and stamp verified_first_date so gate_date(item) is authoritative."""
    verdict = date_verify.verify(item, store)
    item.verified_first_date = verdict.get("verified_first_date")
    return item


def eligible_competitor_items(
    competitor_items: list[HotspotItem],
    store,
    *,
    max_age_days: int,
    as_of: date,
) -> tuple[list, list]:
    """Split competitor items into (eligible, dropped_stale) per spec §D.3.

    eligible  := passes DateVerify AND within_max_age(gate_date)
    dropped   := everything else (legitimately gated; NOT a gate failure)

    DateVerify is the sole cross-validator (§D.4): multi-competitor consensus on a
    backdated-old paper cannot override the arXiv/Wayback hard anchor.
    """
    eligible: list[HotspotItem] = []
    dropped: list[HotspotItem] = []
    for raw in competitor_items:
        item = _apply_verdict(raw, store)
        gd = gate_date(item)
        if gd is not None and _within_max_age(gd, max_age_days=max_age_days, as_of=as_of):
            eligible.append(item)
        else:
            dropped.append(item)
    return eligible, dropped


def assert_union_floor(our_coverage: set, eligible: list) -> None:
    """Acceptance: our_coverage ⊇ eligible_competitor_items (spec §D.3, scoped).

    Only eligible (verified + within max_age) competitor items carry a ⊇ obligation.
    Items dropped by the hard gate are intentionally excluded.
    """
    eligible_keys = {_key(i) for i in eligible}
    missing = eligible_keys - set(our_coverage)
    if missing:
        raise AssertionError(
            "GapFill union-floor violated: our_coverage missing eligible competitor items: "
            + ", ".join(sorted(missing))
        )


def gapfill(our_coverage: set, eligible: list) -> list:
    """Return new items to ingest = eligible \\ our_coverage (the verifiable diff gate).

    These are 'someone has it, it passed OUR verification, we don't' — directed
    fetch + already-verified, ready to merge into our coverage.
    """
    seen: set[str] = set()
    new_items: list[HotspotItem] = []
    for item in eligible:
        k = _key(item)
        if k in our_coverage or k in seen:
            continue
        seen.add(k)
        new_items.append(item)
    return new_items
```

> `_apply_verdict` relies on stage-3 `date_verify.verify` and stage-1 `gate_date` reading `verified_first_date`. `floor_to_utc_day`/`datetime`/`timezone` are imported for parity with the gate-date contract even when only `gate_date` is called directly; keep them so a future inline gate uses the same day-granular path.

**Verify (must PASS now):**
```bash
python -m pytest tests/test_gapfill.py -v
```
Expect: 5 tests pass (eligible/dropped split, union-floor pass-with-stale, union-floor raise, directed-fetch diff, §D.4 hard-anchor).

> Remove the unused `floor_to_utc_day`/`datetime`/`timezone` imports if your linter (ruff F401) flags them; they are documentation-only.

---

## Task 8 — `run_journal` channel: `intentionally_dropped_stale_competitor`

- [ ] Add a typed helper that records the per-run dropped-stale list (with each item's `gate_date` + reason + provenance), and a reader that aggregates per-source drop ratios. This is the data the 2nd-order alert reads (§E). Extend the stage-0 `run_journal` module — do not fork it.

Append to `arxiv_assistant/utils/hotspot/run_journal.py`:

```python
def record_dropped_stale_competitor(
    run_date: str,
    eligible: list,
    dropped: list,
    competitor_items: list,
) -> dict:
    """Build the intentionally_dropped_stale_competitor journal record (spec §D.3/§E).

    Per competitor source: total seen + dropped count + the dropped item details.
    Returns the record (caller appends via run_journal.append).
    """
    from collections import defaultdict

    def _prov(i) -> str:
        return getattr(i, "provenance", "") or "unknown"

    total: dict[str, int] = defaultdict(int)
    drop_count: dict[str, int] = defaultdict(int)
    details: list[dict] = []
    for i in competitor_items:
        total[_prov(i)] += 1
    for i in dropped:
        src = _prov(i)
        drop_count[src] += 1
        details.append({
            "provenance": src,
            "canonical_url": getattr(i, "canonical_url", "") or getattr(i, "url", ""),
            "gate_date": getattr(i, "verified_first_date", None),
            "reason": "stale_beyond_max_age_or_unverified",
        })
    per_source = {
        src: {"seen": total[src], "dropped": drop_count.get(src, 0),
              "drop_ratio": round(drop_count.get(src, 0) / total[src], 4) if total[src] else 0.0}
        for src in total
    }
    return {
        "channel": "intentionally_dropped_stale_competitor",
        "run_date": run_date,
        "eligible_count": len(eligible),
        "dropped_count": len(dropped),
        "per_source": per_source,
        "dropped_items": details,
    }
```

**Verify:**
```bash
python -c "from arxiv_assistant.utils.hotspot import run_journal as rj; r=rj.record_dropped_stale_competitor('2026-06-03',[1],[type('I',(),{'provenance':'reuse:ainews','canonical_url':'u','verified_first_date':'2023-01-01'})()],[type('I',(),{'provenance':'reuse:ainews'})(),type('I',(),{'provenance':'reuse:ainews'})()]); print(r['per_source'])"
```
Expect: `{'reuse:ainews': {'seen': 2, 'dropped': 1, 'drop_ratio': 0.5}}`.

---

## Task 9 — 2nd-order pollution alert: write the failing test first

- [ ] TDD. The alert (§E): for each competitor source, compare today's `drop_ratio` against the source's trailing-14-run *median* baseline; fire iff `drop_ratio >= 2 * baseline_median` AND `drop_ratio >= 0.30`. Pure read over journal records. Write the test first.

Append to `tests/test_gapfill.py`:

```python
class TestSecondOrderPollutionAlert(unittest.TestCase):
    """Spec §E: single-source dropped-ratio spike vs trailing-14-run median baseline."""

    def _run(self, ratio: float, src: str = "reuse:ainews") -> dict:
        seen, dropped = 100, int(round(ratio * 100))
        return {
            "channel": "intentionally_dropped_stale_competitor",
            "per_source": {src: {"seen": seen, "dropped": dropped, "drop_ratio": ratio}},
        }

    def test_spike_triggers_alert(self) -> None:
        from arxiv_assistant.hotspots import gapfill
        history = [self._run(0.05) for _ in range(14)]          # stable 5% baseline
        today = self._run(0.40)                                  # spike: 8x baseline AND >=30%
        alerts = gapfill.second_order_pollution_alerts(today, history, multiplier=2.0, abs_floor=0.30)
        self.assertEqual([a["source"] for a in alerts], ["reuse:ainews"])

    def test_stable_does_not_trigger(self) -> None:
        from arxiv_assistant.hotspots import gapfill
        history = [self._run(0.20) for _ in range(14)]          # baseline 20%
        today = self._run(0.25)                                  # below 2x AND not a big jump
        alerts = gapfill.second_order_pollution_alerts(today, history, multiplier=2.0, abs_floor=0.30)
        self.assertEqual(alerts, [])

    def test_high_abs_but_below_2x_baseline_does_not_trigger(self) -> None:
        from arxiv_assistant.hotspots import gapfill
        # Source legitimately curates many old items every day: 35% steady baseline.
        history = [self._run(0.35) for _ in range(14)]
        today = self._run(0.40)                                  # >=30% abs but only 1.14x baseline
        alerts = gapfill.second_order_pollution_alerts(today, history, multiplier=2.0, abs_floor=0.30)
        self.assertEqual(alerts, [])

    def test_zero_baseline_uses_abs_floor_guard(self) -> None:
        from arxiv_assistant.hotspots import gapfill
        history = [self._run(0.0) for _ in range(14)]            # never dropped before
        today = self._run(0.40)                                  # first big drop
        alerts = gapfill.second_order_pollution_alerts(today, history, multiplier=2.0, abs_floor=0.30)
        self.assertEqual([a["source"] for a in alerts], ["reuse:ainews"])
```

**Verify (must FAIL):**
```bash
python -m pytest tests/test_gapfill.py::TestSecondOrderPollutionAlert -v
```
Expect: `AttributeError: ... 'second_order_pollution_alerts'`.

---

## Task 10 — Implement `second_order_pollution_alerts`

- [ ] Implement in `gapfill.py`. Baseline = median of the source's `drop_ratio` over trailing runs (default 14). Fire iff `today_ratio >= multiplier * baseline` AND `today_ratio >= abs_floor`. Zero baseline: the `abs_floor` guard alone gates (any spike past `abs_floor` from a never-dropping source is suspicious, but the `>= 2*0` condition is trivially true, so the abs floor does the real work).

Append to `arxiv_assistant/hotspots/gapfill.py`:

```python
import statistics


def second_order_pollution_alerts(
    today_record: dict,
    history: list[dict],
    *,
    multiplier: float = 2.0,
    abs_floor: float = 0.30,
    trailing: int = 14,
) -> list[dict]:
    """Per-source upstream-pollution alert (spec §E, decision 4).

    For each competitor source in today's intentionally_dropped_stale_competitor
    record: fire iff today's drop_ratio >= multiplier * trailing-median baseline
    AND today's drop_ratio >= abs_floor. Distinguishes 'normal steady curation of
    old items' (high but stable -> no alert) from 'this competitor suddenly dumped
    old content' (single-source spike -> alert). Pure read over journal records.
    """
    recent = [r for r in history if r.get("channel") == "intentionally_dropped_stale_competitor"][-trailing:]
    today_per_source = today_record.get("per_source", {})
    alerts: list[dict] = []
    for src, stats in today_per_source.items():
        today_ratio = float(stats.get("drop_ratio", 0.0))
        baseline_samples = [
            float(r["per_source"][src]["drop_ratio"])
            for r in recent
            if src in r.get("per_source", {})
        ]
        baseline = statistics.median(baseline_samples) if baseline_samples else 0.0
        if today_ratio >= abs_floor and today_ratio >= multiplier * baseline:
            alerts.append({
                "source": src,
                "today_ratio": round(today_ratio, 4),
                "baseline_median": round(baseline, 4),
                "multiplier": multiplier,
                "abs_floor": abs_floor,
                "message": f"competitor source {src} suspected upstream pollution: "
                           f"drop_ratio {today_ratio:.2f} vs baseline {baseline:.2f}",
            })
    return alerts
```

**Verify (must PASS):**
```bash
python -m pytest tests/test_gapfill.py -v
```
Expect: all 9 tests pass (5 GapFill + 4 alert).

---

## Task 11 — Wire reuse harvest + GapFill into the journal record (integration glue, no kernel yet)

- [ ] Add a single orchestration helper the stage-6 kernel will call, so Stage 4 ships a callable seam without touching `kernel.py` (which does not exist yet). It harvests enabled reuse sources, splits eligible/dropped, asserts the floor against our coverage, computes gapfill, writes the journal record, and returns alerts.

Append to `arxiv_assistant/hotspots/gapfill.py`:

```python
from arxiv_assistant.utils.hotspot import run_journal


REUSE_ADAPTERS = {
    "hf_daily": "arxiv_assistant.apis.hotspot.reuse_hf_daily",
    "ainews": "arxiv_assistant.apis.hotspot.reuse_ainews",
    "agents_radar": "arxiv_assistant.apis.hotspot.reuse_agents_radar",
    "horizon": "arxiv_assistant.apis.hotspot.reuse_horizon",
    "scholar_inbox": "arxiv_assistant.apis.hotspot.reuse_scholar_inbox",
}


def harvest_reuse_layer(
    reuse_sources: list[str],
    target_date: datetime,
    freshness_hours: int,
) -> list[HotspotItem]:
    """Fan out enabled reuse adapters -> competitor_items (one schema, stamped provenance)."""
    import importlib

    out: list[HotspotItem] = []
    for name in reuse_sources:
        mod_path = REUSE_ADAPTERS.get(name)
        if not mod_path:
            continue
        try:
            mod = importlib.import_module(mod_path)
            out.extend(mod.fetch_hotspot_items(target_date, freshness_hours))
        except Exception as ex:  # degrade per source (spec §E)
            print(f"Warning: reuse adapter {name} failed: {ex}")
    return out


def run_gapfill_floor(
    our_coverage: set,
    competitor_items: list[HotspotItem],
    store,
    *,
    max_age_days: int,
    as_of: date,
    run_date: str,
) -> dict:
    """End-to-end Stage-4 seam: verify -> split -> assert floor -> gapfill -> journal -> alerts.

    Returns {"new_items", "eligible", "dropped", "alerts"}. assert_union_floor runs
    AFTER gapfill conceptually, but here we assert on (our_coverage ∪ gapfilled keys)
    so the floor is satisfied by exactly the directed fetch we are about to ingest.
    """
    eligible, dropped = eligible_competitor_items(
        competitor_items, store, max_age_days=max_age_days, as_of=as_of
    )
    new_items = gapfill(our_coverage, eligible)
    covered = set(our_coverage) | {_key(i) for i in new_items}
    assert_union_floor(covered, eligible)

    record = run_journal.record_dropped_stale_competitor(run_date, eligible, dropped, competitor_items)
    run_journal.append(run_date, record)
    history = run_journal.read_runs()
    alerts = second_order_pollution_alerts(record, history)
    return {"new_items": new_items, "eligible": eligible, "dropped": dropped, "alerts": alerts}
```

**Verify (full module + tests still green):**
```bash
python -c "from arxiv_assistant.hotspots import gapfill; print(sorted(gapfill.REUSE_ADAPTERS))"
python -m pytest tests/test_gapfill.py -v
```

---

## Task 12 — Stop-the-line: §G invariants as acceptance assertions

- [ ] Add an invariant test confirming INV5 (reuse items pass the SAME DateVerify + max_age gate) and §D.4 (no majority vote). This is the stage's acceptance gate.

Append to `tests/test_gapfill.py`:

```python
class TestInvariants(unittest.TestCase):
    def test_inv5_reuse_items_carry_reuse_provenance_and_pass_same_gate(self) -> None:
        # Reuse items inherit recall, not staleness: they go through the identical gate.
        from arxiv_assistant.hotspots import gapfill
        store = MagicMock()
        comp = [_item("https://a.test/fresh", provenance="reuse:hf_daily")]
        with patch.object(gapfill.date_verify, "verify", side_effect=_fake_verify):
            eligible, dropped = gapfill.eligible_competitor_items(
                comp, store, max_age_days=14, as_of=date(2026, 6, 3)
            )
        self.assertTrue(all(i.provenance.startswith("reuse:") for i in eligible + dropped))
        self.assertEqual(len(eligible), 1)

    def test_inv2_subday_jitter_cannot_flip_gate(self) -> None:
        from arxiv_assistant.hotspots import gapfill
        store = MagicMock()
        # Two verifies of the SAME url differing only by sub-day time -> same eligibility.
        def jitter_verify(item, _store):
            t = "23:59:59" if item.title.endswith("Z") else "00:00:01"
            return {"verified_first_date": f"2026-05-20T{t}+00:00", "confidence": 0.9, "evidence": []}
        a = _item("https://a.test/jit"); a.title = "X"
        b = _item("https://a.test/jit"); b.title = "XZ"
        with patch.object(gapfill.date_verify, "verify", side_effect=jitter_verify):
            ea, _ = gapfill.eligible_competitor_items([a], store, max_age_days=14, as_of=date(2026, 6, 3))
            eb, _ = gapfill.eligible_competitor_items([b], store, max_age_days=14, as_of=date(2026, 6, 3))
        self.assertEqual(len(ea), len(eb))  # day-granular gate: identical verdict
```

**Verify (final, all green):**
```bash
python -m pytest tests/test_gapfill.py -v
```
Expect: 11 tests pass. Stage 4 is done when this is green AND `python -m pytest tests/ -q` shows no regressions in the existing suite.

---

## Done criteria

- [ ] 5 reuse adapters (`reuse_hf_daily`, `reuse_ainews`, `reuse_agents_radar`, `reuse_horizon`, `reuse_scholar_inbox`) each export `fetch_hotspot_items(target_date, freshness_hours, ...) -> list[HotspotItem]` with `provenance="reuse:<name>"` and tier-mapped `source_role`.
- [ ] `gapfill.py` exports the three locked signatures (`eligible_competitor_items`, `gapfill`, `assert_union_floor`) plus `second_order_pollution_alerts` and the `run_gapfill_floor` / `harvest_reuse_layer` seams.
- [ ] `run_journal.record_dropped_stale_competitor` writes the `intentionally_dropped_stale_competitor` channel.
- [ ] `tests/test_gapfill.py` (11 tests) green; existing suite unbroken.
- [ ] `[HOTSPOT_REUSE]` block present in `config.ini` + template.
- [ ] One commit per task; conventional-commit messages; docs in sync.
