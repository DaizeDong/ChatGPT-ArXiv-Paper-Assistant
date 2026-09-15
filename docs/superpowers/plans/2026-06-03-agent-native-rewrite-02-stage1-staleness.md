# Stage 1 — Staleness Root-Cause Fix (pure Python) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Root-cause-fix hotspot staleness with zero LLM/agent calls — verify each item's true first-publication date deterministically (arXiv v1 / Crossref), floor every gate to a UTC calendar day, and drive both the max-age hard gate and the gravity decay from `verified_first_date` instead of the source-claimed `published_at` (the HF `publishedAt` bug).

**Architecture:** Two new pure modules (`utils/hotspot/gate_date.py`, `hotspots/date_verify.py` Tier-0 path) plus surgical edits to three existing call sites (`get_freshness_date`, `_freshness_weight`/`score_stories`, the pipeline freshness gate). `gate_date` is a pure function over `HotspotItem`; Tier-0 `verify()` is deterministic (arXiv v1 submission date, Crossref registration date, `github_trend` observed-trending exception), earliest-credible-date-wins, with a permanent write-once verdict cache keyed by content hash. Tier-1/2 are deferred to Stage 3 and here return a conservative `min(claimed, fetched) + low-confidence` fallback (legal for this stage, no placeholder semantics).

**Tech Stack:** Python 3, `pytest` (`unittest.TestCase` style), `requests` (existing `fetch_text`/`fetch_json` in `utils/hotspot/hotspot_sources.py`), arXiv Atom API (`http://export.arxiv.org/api/query`, reused pattern from `arxiv_assistant/apis/arxiv.py`), Crossref REST API, SQLite-backed `StoryStore.get_verdict`/`put_verdict` from Stage 0.

---

## Dependencies & contract locks

- **Depends on Stage 0** for: `HotspotItem.verified_first_date: str | None = None` + `provenance: str` (§2.1), and `StoryStore.get_verdict(content_hash) -> dict | None` / `put_verdict(content_hash, verdict)` write-once cache (§2.3). This plan reads those exact signatures; it must NOT redefine them.
- **Locked signatures** this stage implements verbatim from the overview contract (`docs/superpowers/plans/2026-06-03-agent-native-rewrite-00-overview.md`):
  - §2.4 `floor_to_utc_day(iso_ts: str | None) -> date | None` and `gate_date(item: HotspotItem) -> date | None`.
  - §2.5 `verify(item: HotspotItem, store: StoryStore) -> dict` returning `{"verified_first_date": str, "confidence": float, "evidence": list[str]}`, and `poll_arxiv_versions(arxiv_ids: list[str]) -> dict[str, int]`.
- **Spec sections covered:** §B.2 (Tier-0 arXiv v1), §B.3 (earliest-credible-date-wins), §B.3.1 (authoritative whole-day anchor), §B.5 (max-age hard gate), §B.5.1 (gate_date day-granularity), §G INV1 + INV2.
- **§G invariants asserted as acceptance tests in this stage:**
  - **INV1** — dates driving gates are `verified_first_date` only, never source-claimed `published_at`.
  - **INV2** — discrete gates consume `gate_date` (day-granular); sub-day jitter cannot flip a gate.
- **Stage boundary (do NOT cross):** Tier-1/2 subagent dispatch, Wayback CDX, `date_verdicts` snapshot travel, and `arxiv_versions` monotonic refresh wiring live in Stage 3. This stage gives a *complete* Tier-0 implementation and a *conservative deterministic fallback* for the Tier-1/2 branch — never a `raise NotImplementedError` or a stubbed return.

---

## File structure (created + touched)

```
arxiv_assistant/
  utils/hotspot/
    gate_date.py        NEW  — floor_to_utc_day() + gate_date() pure functions
    hotspot_sources.py  MOD  — get_freshness_date() reads verified_first_date (lines 145-159)
  hotspots/
    date_verify.py      NEW  — verify() Tier-0 deterministic + conservative T1/2 fallback + poll_arxiv_versions()
    story.py            MOD  — _freshness_weight() gate_date-based (47-62); score_stories() freshness via gate_date (278-281)
    pipeline.py         MOD  — freshness/max-age gate uses verified_first_date→gate_date (1653-1687)
tests/
  test_gate_date.py     NEW  — truth-table full coverage of floor_to_utc_day + gate_date
  test_date_verify.py   NEW  — Tier-0 arxiv_id→v1 old-date, cache write-once freeze, github_trend exception
  test_stage1_golden.py NEW  — golden snapshot: 8/41 stale papers sink with zero agents
```

---

## Task 1: `floor_to_utc_day` pure function

**Files:**
- Create: `arxiv_assistant/utils/hotspot/gate_date.py`
- Test: `tests/test_gate_date.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gate_date.py
from __future__ import annotations

import unittest
from datetime import date

from arxiv_assistant.utils.hotspot.gate_date import floor_to_utc_day


class TestFloorToUtcDay(unittest.TestCase):
    def test_none_returns_none(self) -> None:
        self.assertIsNone(floor_to_utc_day(None))

    def test_empty_string_returns_none(self) -> None:
        self.assertIsNone(floor_to_utc_day(""))

    def test_unparseable_returns_none(self) -> None:
        self.assertIsNone(floor_to_utc_day("not-a-date"))

    def test_date_only_string(self) -> None:
        self.assertEqual(floor_to_utc_day("2026-04-04"), date(2026, 4, 4))

    def test_zulu_timestamp_truncates_time(self) -> None:
        self.assertEqual(floor_to_utc_day("2026-04-04T23:59:59Z"), date(2026, 4, 4))

    def test_subday_jitter_same_day(self) -> None:
        # Two sub-day-different timestamps on the same UTC day floor identically (INV2).
        a = floor_to_utc_day("2026-04-04T00:00:01Z")
        b = floor_to_utc_day("2026-04-04T23:59:58Z")
        self.assertEqual(a, b)
        self.assertEqual(a, date(2026, 4, 4))

    def test_offset_converted_to_utc_before_flooring(self) -> None:
        # 2026-04-04T01:00:00+09:00 == 2026-04-03T16:00:00Z → floors to the 3rd.
        self.assertEqual(floor_to_utc_day("2026-04-04T01:00:00+09:00"), date(2026, 4, 3))

    def test_naive_timestamp_assumed_utc(self) -> None:
        self.assertEqual(floor_to_utc_day("2026-04-04 12:30:00"), date(2026, 4, 4))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gate_date.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'arxiv_assistant.utils.hotspot.gate_date'`

- [ ] **Step 3: Write minimal implementation**

```python
# arxiv_assistant/utils/hotspot/gate_date.py
from __future__ import annotations

from datetime import UTC, date

from arxiv_assistant.utils.hotspot.hotspot_sources import parse_datetime


def floor_to_utc_day(iso_ts: str | None) -> date | None:
    """Truncate any timestamp to its UTC calendar day (drops H:M:S).

    Returns None for None/empty/unparseable input. Naive timestamps are
    assumed UTC; offset-aware timestamps are converted to UTC before flooring.
    This is the day-granular floor that makes sub-day WebSearch jitter unable
    to flip a discrete gate (spec §B.5.1, INV2).
    """
    if not iso_ts:
        return None
    dt = parse_datetime(iso_ts)
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).date()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gate_date.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add arxiv_assistant/utils/hotspot/gate_date.py tests/test_gate_date.py
git commit -m "feat(hotspot): add floor_to_utc_day day-granular gate primitive"
```

---

## Task 2: `gate_date` over HotspotItem (verified_first_date ∪ authoritative whole-day anchors)

**Files:**
- Modify: `arxiv_assistant/utils/hotspot/gate_date.py`
- Test: `tests/test_gate_date.py`

The credible-date set per §2.4 / §B.3.1 is `{verified_first_date} ∪ {authoritative whole-day anchors}`, where anchors are the arXiv announced day (carried in `metadata["arxiv_announced_date"]` when known) and the Crossref registration day (`metadata["crossref_registered_date"]`). `gate_date = floor_to_utc_day(min(credible_dates))` — earliest-credible-date-wins (§B.3), since pollution only ever back-dates *forward* to look fresh, so the minimum beats it. If no credible date exists, return `None` (the gate treats `None` as "cannot verify → do not drop", consistent with the existing `is_fresh` policy).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gate_date.py  (append)
from arxiv_assistant.utils.hotspot.gate_date import gate_date
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem


def _item(**kw) -> HotspotItem:
    base = dict(
        source_id="hf_papers",
        source_name="HF",
        source_role="paper_trending",
        source_type="paper",
        title="t",
        summary="s",
        url="https://huggingface.co/papers/2301.00001",
        canonical_url="https://arxiv.org/abs/2301.00001",
        published_at="2026-04-04T12:00:00Z",
    )
    base.update(kw)
    item = HotspotItem(**{k: v for k, v in base.items() if k != "verified_first_date"})
    if "verified_first_date" in kw:
        item.verified_first_date = kw["verified_first_date"]
    return item


class TestGateDate(unittest.TestCase):
    def test_no_credible_date_returns_none(self) -> None:
        # verified_first_date unset, no anchors → None (cannot verify, do not drop).
        item = _item(metadata={})
        item.verified_first_date = None
        self.assertIsNone(gate_date(item))

    def test_uses_verified_first_date(self) -> None:
        item = _item(verified_first_date="2023-01-02T09:00:00Z", metadata={})
        self.assertEqual(gate_date(item), date(2023, 1, 2))

    def test_ignores_published_at_when_verified_present(self) -> None:
        # INV1: source-claimed published_at (2026) must NOT win over verified (2023).
        item = _item(verified_first_date="2023-01-02T00:00:00Z",
                     published_at="2026-04-04T12:00:00Z", metadata={})
        self.assertEqual(gate_date(item), date(2023, 1, 2))

    def test_arxiv_announced_anchor_min_wins(self) -> None:
        # Authoritative whole-day anchor earlier than verified → min wins (§B.3.1).
        item = _item(verified_first_date="2023-01-05T00:00:00Z",
                     metadata={"arxiv_announced_date": "2023-01-02"})
        self.assertEqual(gate_date(item), date(2023, 1, 2))

    def test_crossref_anchor_min_wins(self) -> None:
        item = _item(verified_first_date="2024-06-10T00:00:00Z",
                     metadata={"crossref_registered_date": "2024-06-09"})
        self.assertEqual(gate_date(item), date(2024, 6, 9))

    def test_min_is_monotone_earliest_wins(self) -> None:
        # All three credible dates present; earliest (anchor) wins regardless of order.
        item = _item(verified_first_date="2025-03-03T00:00:00Z",
                     metadata={"arxiv_announced_date": "2025-03-01",
                               "crossref_registered_date": "2025-03-02"})
        self.assertEqual(gate_date(item), date(2025, 3, 1))

    def test_subday_jitter_absorbed_by_floor(self) -> None:
        # INV2: two sub-day-jittered verified dates on same UTC day → same gate_date.
        a = gate_date(_item(verified_first_date="2026-04-04T00:00:30Z", metadata={}))
        b = gate_date(_item(verified_first_date="2026-04-04T22:10:00Z", metadata={}))
        self.assertEqual(a, b)
        self.assertEqual(a, date(2026, 4, 4))

    def test_anchor_only_no_verified(self) -> None:
        item = _item(metadata={"arxiv_announced_date": "2023-01-02"})
        item.verified_first_date = None
        self.assertEqual(gate_date(item), date(2023, 1, 2))
```

> NOTE: `HotspotItem.verified_first_date` is added by Stage 0. If Stage 0 has not landed when this test runs, the test helper sets it via attribute assignment which the implementation reads with `getattr(item, "verified_first_date", None)`, so the code is robust either way.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gate_date.py::TestGateDate -v`
Expected: FAIL with `ImportError: cannot import name 'gate_date'`

- [ ] **Step 3: Write minimal implementation**

```python
# arxiv_assistant/utils/hotspot/gate_date.py  (append)
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem

_ANCHOR_KEYS = ("arxiv_announced_date", "crossref_registered_date")


def gate_date(item: HotspotItem) -> date | None:
    """Day-granular gate date = floor_to_utc_day(min(credible_dates(item))).

    credible_dates = {verified_first_date} ∪ {authoritative whole-day anchors:
    arXiv announced day / Crossref registration day}. Earliest-credible-date-wins
    (spec §B.3): pollution only back-dates forward to look fresh, so min beats it.
    Source-claimed published_at is NEVER credible (INV1). Returns None when no
    credible date exists (gate treats None as cannot-verify → do not drop).
    """
    credible: list[date] = []

    verified = getattr(item, "verified_first_date", None)
    floored = floor_to_utc_day(verified)
    if floored is not None:
        credible.append(floored)

    metadata = item.metadata or {}
    for key in _ANCHOR_KEYS:
        anchor = floor_to_utc_day(metadata.get(key))
        if anchor is not None:
            credible.append(anchor)

    if not credible:
        return None
    return min(credible)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gate_date.py -v`
Expected: PASS (all `TestFloorToUtcDay` + `TestGateDate` tests)

- [ ] **Step 5: Commit**

```bash
git add arxiv_assistant/utils/hotspot/gate_date.py tests/test_gate_date.py
git commit -m "feat(hotspot): add gate_date with earliest-credible whole-day anchor"
```

---

## Task 3: DateVerify Tier-0 — arXiv v1 submission date reader

**Files:**
- Create: `arxiv_assistant/hotspots/date_verify.py`
- Test: `tests/test_date_verify.py`

The arXiv staleness root cause: HF reports `publishedAt` (platform metadata), not the arXiv v1 submission date. Tier-0 reads the v1 submission date from the arXiv Atom API. We query a single id via `id_list` and parse the **v1** `<published>` element (arXiv's `<published>` is always the v1 submission timestamp; `<updated>` is the latest version). The id (with any `vN` suffix stripped) is the content hash key for the cache.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_date_verify.py
from __future__ import annotations

import unittest
from unittest.mock import patch

from arxiv_assistant.hotspots.date_verify import _fetch_arxiv_v1_date

_ARXIV_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00001v3</id>
    <published>2023-01-02T18:00:00Z</published>
    <updated>2026-03-30T10:00:00Z</updated>
    <title>An old paper resurfacing on HF</title>
  </entry>
</feed>"""


class TestFetchArxivV1Date(unittest.TestCase):
    @patch("arxiv_assistant.hotspots.date_verify.fetch_text")
    def test_reads_v1_published_not_updated(self, mock_fetch) -> None:
        mock_fetch.return_value = _ARXIV_ATOM
        result = _fetch_arxiv_v1_date("2301.00001")
        # v1 submission date (published), NOT the v3 updated date.
        self.assertEqual(result, "2023-01-02T18:00:00Z")

    @patch("arxiv_assistant.hotspots.date_verify.fetch_text")
    def test_strips_version_suffix_in_query(self, mock_fetch) -> None:
        mock_fetch.return_value = _ARXIV_ATOM
        _fetch_arxiv_v1_date("2301.00001v3")
        called_url = mock_fetch.call_args[0][0]
        self.assertIn("id_list=2301.00001", called_url)
        self.assertNotIn("v3", called_url.split("id_list=")[-1])

    @patch("arxiv_assistant.hotspots.date_verify.fetch_text")
    def test_no_entry_returns_none(self, mock_fetch) -> None:
        mock_fetch.return_value = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        self.assertIsNone(_fetch_arxiv_v1_date("9999.99999"))

    @patch("arxiv_assistant.hotspots.date_verify.fetch_text", side_effect=RuntimeError("net"))
    def test_network_error_returns_none(self, mock_fetch) -> None:
        self.assertIsNone(_fetch_arxiv_v1_date("2301.00001"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_date_verify.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'arxiv_assistant.hotspots.date_verify'`

- [ ] **Step 3: Write minimal implementation**

```python
# arxiv_assistant/hotspots/date_verify.py
from __future__ import annotations

import re
from xml.etree import ElementTree

from arxiv_assistant.utils.hotspot.hotspot_sources import fetch_text

_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_ARXIV_API = "http://export.arxiv.org/api/query?id_list={id}&max_results=1"
_VERSION_SUFFIX = re.compile(r"v\d+$")


def _strip_version(arxiv_id: str) -> str:
    return _VERSION_SUFFIX.sub("", (arxiv_id or "").strip())


def _fetch_arxiv_v1_date(arxiv_id: str) -> str | None:
    """Return the arXiv v1 submission timestamp (ISO8601) or None.

    arXiv's Atom <published> is always the v1 submission time; <updated> is the
    latest version. Reading <published> directly fixes the HF publishedAt
    staleness bug (spec §B.2 / §0). Network/parse failures return None
    (caller falls back conservatively, never raises).
    """
    bare = _strip_version(arxiv_id)
    if not bare:
        return None
    try:
        xml = fetch_text(_ARXIV_API.format(id=bare))
        root = ElementTree.fromstring(xml)
    except Exception:
        return None
    entry = root.find(f"{_ATOM_NS}entry")
    if entry is None:
        return None
    published = entry.find(f"{_ATOM_NS}published")
    if published is None or not (published.text or "").strip():
        return None
    return published.text.strip()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_date_verify.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add arxiv_assistant/hotspots/date_verify.py tests/test_date_verify.py
git commit -m "feat(hotspot): add arXiv v1 submission-date reader (Tier-0 staleness fix)"
```

---

## Task 4: DateVerify Tier-0 — Crossref registration date reader

**Files:**
- Modify: `arxiv_assistant/hotspots/date_verify.py`
- Test: `tests/test_date_verify.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_date_verify.py  (append)
from arxiv_assistant.hotspots.date_verify import _fetch_crossref_date

_CROSSREF_JSON = {
    "message": {
        "created": {"date-parts": [[2024, 6, 9]], "date-time": "2024-06-09T08:00:00Z"}
    }
}


class TestFetchCrossrefDate(unittest.TestCase):
    @patch("arxiv_assistant.hotspots.date_verify.fetch_json")
    def test_reads_created_date(self, mock_json) -> None:
        mock_json.return_value = _CROSSREF_JSON
        self.assertEqual(_fetch_crossref_date("10.1145/1234.5678"), "2024-06-09")

    @patch("arxiv_assistant.hotspots.date_verify.fetch_json")
    def test_missing_created_returns_none(self, mock_json) -> None:
        mock_json.return_value = {"message": {}}
        self.assertIsNone(_fetch_crossref_date("10.1145/1234.5678"))

    @patch("arxiv_assistant.hotspots.date_verify.fetch_json", side_effect=RuntimeError("net"))
    def test_network_error_returns_none(self, mock_json) -> None:
        self.assertIsNone(_fetch_crossref_date("10.1145/1234.5678"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_date_verify.py::TestFetchCrossrefDate -v`
Expected: FAIL with `ImportError: cannot import name '_fetch_crossref_date'`

- [ ] **Step 3: Write minimal implementation**

```python
# arxiv_assistant/hotspots/date_verify.py  (append imports + function)
from arxiv_assistant.utils.hotspot.hotspot_sources import fetch_json

_CROSSREF_API = "https://api.crossref.org/works/{doi}"


def _fetch_crossref_date(doi: str) -> str | None:
    """Return Crossref registration day (YYYY-MM-DD) or None.

    Uses the `created` date-parts (registration day, whole-day, machine-
    independent — spec §B.3.1). Network/parse failures return None.
    """
    doi = (doi or "").strip()
    if not doi:
        return None
    try:
        payload = fetch_json(_CROSSREF_API.format(doi=doi))
    except Exception:
        return None
    parts = (((payload or {}).get("message") or {}).get("created") or {}).get("date-parts")
    if not parts or not parts[0]:
        return None
    ymd = parts[0]
    if len(ymd) < 3:
        return None
    try:
        return f"{int(ymd[0]):04d}-{int(ymd[1]):02d}-{int(ymd[2]):02d}"
    except (ValueError, TypeError):
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_date_verify.py -v`
Expected: PASS (all `TestFetchArxivV1Date` + `TestFetchCrossrefDate`)

- [ ] **Step 5: Commit**

```bash
git add arxiv_assistant/hotspots/date_verify.py tests/test_date_verify.py
git commit -m "feat(hotspot): add Crossref registration-date reader (Tier-0)"
```

---

## Task 5: DateVerify `verify()` — Tier-0 dispatch, earliest-credible, write-once cache

**Files:**
- Modify: `arxiv_assistant/hotspots/date_verify.py`
- Test: `tests/test_date_verify.py`

`verify(item, store)` per §2.5 returns `{"verified_first_date": str, "confidence": float, "evidence": list[str]}`. Logic:
1. **Cache hit:** `store.get_verdict(content_hash)` returns the frozen verdict → return it unchanged (the date is frozen for life; INV3 freeze is enforced by Stage 0's write-once `put_verdict`).
2. **`github_trend` exception:** legitimately uses observed-trending date (the item's `published_at`), not first-publication date (§B.2). Confidence 0.95.
3. **Tier-0 deterministic:** collect credible dates — arXiv v1 (`metadata["arxiv_id"]`), Crossref (`metadata["doi"]`), and the source-claimed `published_at` only as a *fallback candidate of last resort*. `verified_first_date = min(credible)` (earliest-credible-date-wins, §B.3). Confidence 0.95 when any authoritative anchor was found.
4. **Tier-1/2 conservative fallback (Stage-3 boundary):** when Tier-0 yields no authoritative anchor, return `min(claimed, fetched) + low confidence` (0.3). This is the legal Stage-1 fallback specified in §B.3 ("无法核实→保守 `min(claimed, fetched)` + low confidence"), NOT a placeholder.
5. **Persist:** `store.put_verdict(content_hash, verdict)` (write-once; no-op if exists) before returning, so the date freezes on first computation.

The content hash is `arxiv_id` (version-stripped) > `doi` > `canonical_url` > `url`, matching §B.4's key precedence.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_date_verify.py  (append)
from arxiv_assistant.hotspots.date_verify import verify, _content_hash
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem


class _FakeStore:
    """Minimal write-once verdict cache matching StoryStore.get/put_verdict."""

    def __init__(self) -> None:
        self._verdicts: dict[str, dict] = {}
        self.put_calls = 0

    def get_verdict(self, content_hash: str):
        return self._verdicts.get(content_hash)

    def put_verdict(self, content_hash: str, verdict: dict) -> None:
        self.put_calls += 1
        self._verdicts.setdefault(content_hash, verdict)  # write-once


def _hf_item(**kw) -> HotspotItem:
    item = HotspotItem(
        source_id=kw.get("source_id", "hf_papers"),
        source_name="HF",
        source_role="paper_trending",
        source_type="paper",
        title="t",
        summary="s",
        url="https://huggingface.co/papers/2301.00001",
        canonical_url="https://arxiv.org/abs/2301.00001",
        published_at=kw.get("published_at", "2026-04-04T12:00:00Z"),
        metadata=kw.get("metadata", {"arxiv_id": "2301.00001"}),
    )
    return item


class TestVerify(unittest.TestCase):
    @patch("arxiv_assistant.hotspots.date_verify._fetch_arxiv_v1_date",
           return_value="2023-01-02T18:00:00Z")
    def test_arxiv_id_yields_v1_old_date(self, _m) -> None:
        store = _FakeStore()
        verdict = verify(_hf_item(), store)
        # The 2023 v1 date beats the 2026 source-claimed published_at (INV1).
        self.assertEqual(verdict["verified_first_date"], "2023-01-02T18:00:00Z")
        self.assertGreaterEqual(verdict["confidence"], 0.9)
        self.assertTrue(any("arxiv" in e.lower() for e in verdict["evidence"]))

    @patch("arxiv_assistant.hotspots.date_verify._fetch_arxiv_v1_date",
           return_value="2023-01-02T18:00:00Z")
    def test_cache_written_once_and_frozen(self, _m) -> None:
        store = _FakeStore()
        first = verify(_hf_item(), store)
        # Second call hits cache; even if the network would now return a different
        # date, the frozen verdict is returned unchanged (INV3 freeze).
        with patch("arxiv_assistant.hotspots.date_verify._fetch_arxiv_v1_date",
                   return_value="2099-12-31T00:00:00Z"):
            second = verify(_hf_item(), store)
        self.assertEqual(first, second)
        self.assertEqual(first["verified_first_date"], "2023-01-02T18:00:00Z")
        self.assertEqual(store.put_calls, 1)  # written exactly once

    def test_github_trend_uses_observed_trending_date(self) -> None:
        store = _FakeStore()
        item = _hf_item(source_id="github_trend",
                        published_at="2026-04-04T00:00:00Z",
                        metadata={})
        verdict = verify(item, store)
        self.assertEqual(verdict["verified_first_date"], "2026-04-04T00:00:00Z")
        self.assertGreaterEqual(verdict["confidence"], 0.9)

    @patch("arxiv_assistant.hotspots.date_verify._fetch_arxiv_v1_date", return_value=None)
    def test_no_anchor_conservative_low_confidence(self, _m) -> None:
        store = _FakeStore()
        item = _hf_item(metadata={}, published_at="2026-04-04T12:00:00Z")
        verdict = verify(item, store)
        # Conservative Stage-1 fallback: min(claimed, fetched), low confidence.
        self.assertEqual(verdict["verified_first_date"], "2026-04-04T12:00:00Z")
        self.assertLessEqual(verdict["confidence"], 0.5)

    def test_content_hash_prefers_arxiv_id_version_stripped(self) -> None:
        item = _hf_item(metadata={"arxiv_id": "2301.00001v3"})
        self.assertEqual(_content_hash(item), "arxiv:2301.00001")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_date_verify.py::TestVerify -v`
Expected: FAIL with `ImportError: cannot import name 'verify'`

- [ ] **Step 3: Write minimal implementation**

```python
# arxiv_assistant/hotspots/date_verify.py  (append)

_GITHUB_TREND_SOURCE = "github_trend"


def _content_hash(item) -> str:
    """Cache key precedence: arxiv_id (version-stripped) > doi > canonical_url > url (§B.4)."""
    metadata = item.metadata or {}
    arxiv_id = _strip_version(metadata.get("arxiv_id", ""))
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    doi = (metadata.get("doi") or "").strip()
    if doi:
        return f"doi:{doi}"
    return f"url:{item.canonical_url or item.url}"


def _earliest(*candidates: str | None) -> str | None:
    """Earliest-credible-date-wins (§B.3): min over parsed timestamps, original string kept."""
    from arxiv_assistant.utils.hotspot.hotspot_sources import parse_datetime

    parsed = [(parse_datetime(c), c) for c in candidates if c]
    parsed = [(dt, c) for dt, c in parsed if dt is not None]
    if not parsed:
        return None
    return min(parsed, key=lambda pair: pair[0])[1]


def verify(item, store) -> dict:
    """Tier-0 deterministic first-date verification (spec §B.2/§B.3/§2.5).

    Returns {"verified_first_date": str, "confidence": float, "evidence": [str]}.
    Cache via store.get_verdict/put_verdict (write-once freeze, INV3). Tier-1/2
    subagent dispatch is Stage 3; here the residual path returns a conservative
    min(claimed, fetched) + low confidence — a legal §B.3 fallback, not a stub.
    """
    content_hash = _content_hash(item)
    cached = store.get_verdict(content_hash)
    if cached is not None:
        return cached

    metadata = item.metadata or {}
    claimed = item.published_at
    fetched = metadata.get("fetched_at")
    evidence: list[str] = []

    # github_trend exception: observed-trending date is the legitimate signal (§B.2).
    if item.source_id == _GITHUB_TREND_SOURCE:
        observed = claimed or fetched
        verdict = {
            "verified_first_date": observed,
            "confidence": 0.95,
            "evidence": ["github_trend:observed_trending_date"],
        }
        store.put_verdict(content_hash, verdict)
        return verdict

    credible: list[str] = []

    arxiv_id = _strip_version(metadata.get("arxiv_id", ""))
    if arxiv_id:
        v1 = _fetch_arxiv_v1_date(arxiv_id)
        if v1:
            credible.append(v1)
            evidence.append(f"arxiv_v1:{arxiv_id}")

    doi = (metadata.get("doi") or "").strip()
    if doi:
        cr = _fetch_crossref_date(doi)
        if cr:
            credible.append(cr)
            evidence.append(f"crossref:{doi}")

    if credible:
        verified = _earliest(*credible, claimed)
        verdict = {
            "verified_first_date": verified,
            "confidence": 0.95,
            "evidence": evidence,
        }
    else:
        # Tier-1/2 boundary (Stage 3). Conservative deterministic fallback (§B.3):
        # earliest of source-claimed and fetched, low confidence, folds below the line.
        verified = _earliest(claimed, fetched) or claimed or fetched
        verdict = {
            "verified_first_date": verified,
            "confidence": 0.3,
            "evidence": ["fallback:min(claimed,fetched)"],
        }

    store.put_verdict(content_hash, verdict)
    return verdict
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_date_verify.py -v`
Expected: PASS (all classes)

- [ ] **Step 5: Commit**

```bash
git add arxiv_assistant/hotspots/date_verify.py tests/test_date_verify.py
git commit -m "feat(hotspot): add Tier-0 verify() with earliest-credible + write-once cache"
```

---

## Task 6: `poll_arxiv_versions` — batched version-count read (Tier-0, independent of verdict freeze)

**Files:**
- Modify: `arxiv_assistant/hotspots/date_verify.py`
- Test: `tests/test_date_verify.py`

§2.5 / §B.4.1: a cheap deterministic Tier-0 read returning `{arxiv_id: version_count}`. It batches up to 100 ids per `id_list` call and parses the highest `vN` from each entry's `<id>`. This is the version-count read only; the *monotonic max merge* and `Story.arxiv_versions` wiring belong to Stage 3 — this function just returns the freshly-observed counts. Version count NEVER touches `date_verdicts` (INV3).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_date_verify.py  (append)
from arxiv_assistant.hotspots.date_verify import poll_arxiv_versions

_VERSIONS_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><id>http://arxiv.org/abs/2301.00001v3</id></entry>
  <entry><id>http://arxiv.org/abs/2302.00002v1</id></entry>
</feed>"""


class TestPollArxivVersions(unittest.TestCase):
    @patch("arxiv_assistant.hotspots.date_verify.fetch_text")
    def test_parses_version_counts(self, mock_fetch) -> None:
        mock_fetch.return_value = _VERSIONS_ATOM
        result = poll_arxiv_versions(["2301.00001", "2302.00002"])
        self.assertEqual(result, {"2301.00001": 3, "2302.00002": 1})

    @patch("arxiv_assistant.hotspots.date_verify.fetch_text")
    def test_empty_input_no_fetch(self, mock_fetch) -> None:
        self.assertEqual(poll_arxiv_versions([]), {})
        mock_fetch.assert_not_called()

    @patch("arxiv_assistant.hotspots.date_verify.fetch_text", side_effect=RuntimeError("net"))
    def test_network_error_returns_empty(self, _m) -> None:
        self.assertEqual(poll_arxiv_versions(["2301.00001"]), {})

    @patch("arxiv_assistant.hotspots.date_verify.fetch_text")
    def test_dedups_ids_in_query(self, mock_fetch) -> None:
        mock_fetch.return_value = _VERSIONS_ATOM
        poll_arxiv_versions(["2301.00001v2", "2301.00001"])  # same bare id twice
        called_url = mock_fetch.call_args[0][0]
        self.assertEqual(called_url.split("id_list=")[-1].split("&")[0], "2301.00001")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_date_verify.py::TestPollArxivVersions -v`
Expected: FAIL with `ImportError: cannot import name 'poll_arxiv_versions'`

- [ ] **Step 3: Write minimal implementation**

```python
# arxiv_assistant/hotspots/date_verify.py  (append)

_ARXIV_BATCH_API = "http://export.arxiv.org/api/query?id_list={ids}&max_results={n}"
_ABS_ID = re.compile(r"abs/(?P<id>\d{4}\.\d{4,5})v(?P<ver>\d+)")
_BATCH_SIZE = 100


def poll_arxiv_versions(arxiv_ids: list[str]) -> dict[str, int]:
    """Return {bare_arxiv_id: latest_version_count} via batched id_list reads (§B.4.1).

    Cheap deterministic Tier-0 read; <=100 ids/call. NEVER writes date_verdicts and
    NEVER changes verified_first_date (INV3). The monotonic max-merge into
    Story.arxiv_versions is Stage 3. Network/parse failures yield {} (caller keeps
    old counts), per the degrade-not-block policy (§B.4.1).
    """
    bare_ids = list(dict.fromkeys(_strip_version(i) for i in arxiv_ids if _strip_version(i)))
    if not bare_ids:
        return {}

    counts: dict[str, int] = {}
    for start in range(0, len(bare_ids), _BATCH_SIZE):
        batch = bare_ids[start:start + _BATCH_SIZE]
        url = _ARXIV_BATCH_API.format(ids=",".join(batch), n=len(batch))
        try:
            xml = fetch_text(url)
            root = ElementTree.fromstring(xml)
        except Exception:
            continue
        for entry in root.findall(f"{_ATOM_NS}entry"):
            id_el = entry.find(f"{_ATOM_NS}id")
            if id_el is None or not id_el.text:
                continue
            match = _ABS_ID.search(id_el.text)
            if match:
                counts[match.group("id")] = int(match.group("ver"))
    return counts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_date_verify.py -v`
Expected: PASS (all classes)

- [ ] **Step 5: Commit**

```bash
git add arxiv_assistant/hotspots/date_verify.py tests/test_date_verify.py
git commit -m "feat(hotspot): add batched poll_arxiv_versions Tier-0 read"
```

---

## Task 7: `get_freshness_date` reads `verified_first_date`

**Files:**
- Modify: `arxiv_assistant/utils/hotspot/hotspot_sources.py:145-159`
- Test: `tests/test_date_verify.py` (new `TestGetFreshnessDate` class — co-located with date logic tests)

§2.1/§B.5: freshness must be driven by `verified_first_date`, never the source-claimed `published_at`. The `github_trend` exception keeps using `fetched_at`. When `verified_first_date` is unset (pre-DateVerify items, backward compat), fall through to the old `published_at` behavior so existing callers don't break.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_date_verify.py  (append)
from arxiv_assistant.utils.hotspot.hotspot_sources import get_freshness_date


class TestGetFreshnessDate(unittest.TestCase):
    def test_prefers_verified_first_date(self) -> None:
        item = _hf_item(published_at="2026-04-04T12:00:00Z")
        item.verified_first_date = "2023-01-02T18:00:00Z"
        # INV1: verified date wins over source-claimed published_at.
        self.assertEqual(get_freshness_date(item), "2023-01-02T18:00:00Z")

    def test_falls_back_to_published_at_when_unverified(self) -> None:
        item = _hf_item(published_at="2026-04-04T12:00:00Z")
        item.verified_first_date = None
        self.assertEqual(get_freshness_date(item), "2026-04-04T12:00:00Z")

    def test_github_trend_still_uses_fetched_at(self) -> None:
        item = _hf_item(source_id="github_trend",
                        published_at="2026-04-01T00:00:00Z",
                        metadata={"fetched_at": "2026-04-04T00:00:00Z"})
        item.verified_first_date = None
        self.assertEqual(get_freshness_date(item), "2026-04-04T00:00:00Z")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_date_verify.py::TestGetFreshnessDate -v`
Expected: FAIL — `test_prefers_verified_first_date` returns `2026-04-04T12:00:00Z` (current code ignores `verified_first_date`).

- [ ] **Step 3: Write minimal implementation**

Replace `get_freshness_date` (current lines 148-159):

```python
def get_freshness_date(item: "HotspotItem") -> str | None:
    """Return the most appropriate date for freshness evaluation.

    Priority (spec §2.1/§B.5):
      1. github_trend: fetched_at (repos trend long after creation).
      2. verified_first_date (set by DateVerify) — the only trusted first date.
      3. published_at — backward-compat fallback for pre-DateVerify items.
    """
    if item.source_id in _FETCHED_AT_VALID_SOURCES:
        fetched_at = (item.metadata or {}).get("fetched_at")
        if fetched_at:
            return fetched_at
    verified = getattr(item, "verified_first_date", None)
    if verified:
        return verified
    return item.published_at
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_date_verify.py::TestGetFreshnessDate -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add arxiv_assistant/utils/hotspot/hotspot_sources.py tests/test_date_verify.py
git commit -m "feat(hotspot): get_freshness_date prefers verified_first_date"
```

---

## Task 8: `_freshness_weight` gravity from gate_date (HN-style, day-granular gate)

**Files:**
- Modify: `arxiv_assistant/hotspots/story.py:47-62` and `:278-281`
- Test: `tests/test_date_verify.py` (new `TestFreshnessWeight` class)

§B.5.1 / §C.3: gravity must compute `T` from `gate_date` (the day-granular verified first date), not `datetime.now()`. We switch `_freshness_weight` to take a `gate_date` (a `date`) plus the run date, computing `T` in hours from the day boundary, then apply the HN-style decay `(P-1)/(T+2)^1.8`. The discrete gate (max-age) only consumes `gate_date` so sub-day jitter cannot flip it (INV2); the continuous gravity is for *within-day display ordering* only.

Per spec §B.5.1, P (points) here is the pre-gravity raw score factor; we keep the existing 5-factor `raw` as P and multiply by the gravity decay. To preserve the existing call shape with minimal churn, `_freshness_weight` returns the **gravity multiplier** given the story's gate_date and the run date.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_date_verify.py  (append)
from datetime import date
from arxiv_assistant.hotspots.story import _freshness_weight


class TestFreshnessWeight(unittest.TestCase):
    def test_same_day_full_weight(self) -> None:
        run = date(2026, 4, 4)
        w = _freshness_weight(date(2026, 4, 4), run_date=run)
        self.assertAlmostEqual(w, 1.0, places=6)

    def test_decays_with_age(self) -> None:
        run = date(2026, 4, 10)
        fresh = _freshness_weight(date(2026, 4, 10), run_date=run)
        old = _freshness_weight(date(2026, 4, 4), run_date=run)  # 6 days → sinks
        self.assertLess(old, fresh)
        self.assertLess(old, 0.2)  # 6-day-old story folds below the line

    def test_none_gate_date_neutral(self) -> None:
        # Unverifiable date → neutral 0.6 (matches legacy unknown-date behavior).
        self.assertAlmostEqual(_freshness_weight(None, run_date=date(2026, 4, 4)), 0.6, places=6)

    def test_subday_irrelevant_gate_is_day_granular(self) -> None:
        # Two stories same gate_date → identical weight regardless of any sub-day origin.
        run = date(2026, 4, 6)
        self.assertEqual(
            _freshness_weight(date(2026, 4, 4), run_date=run),
            _freshness_weight(date(2026, 4, 4), run_date=run),
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_date_verify.py::TestFreshnessWeight -v`
Expected: FAIL — current `_freshness_weight(published_at: str | None)` takes a string and uses `datetime.now()`; the `date`+`run_date` signature does not exist.

- [ ] **Step 3: Write minimal implementation**

Replace `_freshness_weight` (current lines 47-62):

```python
from datetime import date as _date

def _freshness_weight(gate_day: _date | None, *, run_date: _date) -> float:
    """HN-style gravity from the day-granular gate_date (spec §B.5.1/§C.3).

    T = hours since gate_day (day boundary); decay = 1/(T+2)^1.8 normalized so a
    same-day story scores 1.0. Sub-day jitter cannot affect this because the input
    is already floored to a UTC day (INV2). Unknown date → neutral 0.6.
    """
    if gate_day is None:
        return 0.6
    age_days = (run_date - gate_day).days
    if age_days < 0:
        age_days = 0
    t_hours = age_days * 24.0
    # Normalized HN gravity: weight(T)/weight(0), weight(T) = 1/(T+2)^1.8.
    return (2.0 ** 1.8) / ((t_hours + 2.0) ** 1.8)
```

Update the call site in `score_stories` (current lines 278-281):

```python
        # Gravity from the day-granular gate_date (verified first date), not now().
        from arxiv_assistant.utils.hotspot.gate_date import gate_date as _gate_date_fn
        from datetime import UTC as _UTC, datetime as _dt
        run_day = _dt.now(_UTC).date()
        gate_days = [_gate_date_fn(ei.item) for ei in story.items]
        gate_days = [d for d in gate_days if d is not None]
        earliest_gate = min(gate_days) if gate_days else None
        freshness = _freshness_weight(earliest_gate, run_date=run_day)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_date_verify.py::TestFreshnessWeight -v && pytest tests/test_hotspot_pipeline.py -v`
Expected: PASS — new freshness tests pass; existing pipeline tests still green (score_stories signature unchanged).

- [ ] **Step 5: Commit**

```bash
git add arxiv_assistant/hotspots/story.py tests/test_date_verify.py
git commit -m "feat(hotspot): gravity decay from day-granular gate_date, not now()"
```

---

## Task 9: pipeline freshness/max-age hard gate via verified_first_date → gate_date

**Files:**
- Modify: `arxiv_assistant/hotspots/pipeline.py:1653-1687`
- Test: `tests/test_stage1_golden.py`

§B.5 / §B.5.1: the max-age hard gate must use `gate_date(item)` (day-granular `verified_first_date`) with `max_item_age_days` from config (default 14), with the `github_trend` exception. The existing `freshness_hours` soft gate stays but also consumes `gate_date`. We add a DateVerify pass that sets `item.verified_first_date` before the gate, then gate on `gate_date`. `max_item_age_days` is read from config per §3 (`config["HOTSPOTS"].getint("max_item_age_days", fallback=14)`), replacing the hard-coded `MAX_ITEM_AGE_DAYS = 14`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stage1_golden.py
from __future__ import annotations

import unittest
from datetime import UTC, date, datetime
from unittest.mock import patch

from arxiv_assistant.hotspots.pipeline import _apply_freshness_gates
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem


def _paper(arxiv_id: str, published_at: str, verified: str | None) -> HotspotItem:
    item = HotspotItem(
        source_id="hf_papers", source_name="HF", source_role="paper_trending",
        source_type="paper", title=f"paper {arxiv_id}", summary="s",
        url=f"https://huggingface.co/papers/{arxiv_id}",
        canonical_url=f"https://arxiv.org/abs/{arxiv_id}",
        published_at=published_at, metadata={"arxiv_id": arxiv_id},
    )
    item.verified_first_date = verified
    return item


class TestStage1FreshnessGate(unittest.TestCase):
    def test_stale_verified_paper_dropped_zero_agents(self) -> None:
        target = datetime(2026, 4, 4, tzinfo=UTC)
        # Source claims today's date, but verified v1 is 2023 → must be dropped.
        stale = _paper("2301.00001", "2026-04-04T00:00:00Z", "2023-01-02T00:00:00Z")
        fresh = _paper("2604.00002", "2026-04-04T00:00:00Z", "2026-04-04T00:00:00Z")
        kept = _apply_freshness_gates([stale, fresh], target, max_item_age_days=14,
                                      freshness_hours=24)
        kept_ids = {i.metadata["arxiv_id"] for i in kept}
        self.assertIn("2604.00002", kept_ids)
        self.assertNotIn("2301.00001", kept_ids)

    def test_github_trend_exempt_from_max_age(self) -> None:
        target = datetime(2026, 4, 4, tzinfo=UTC)
        trend = HotspotItem(
            source_id="github_trend", source_name="GH", source_role="repo_trending",
            source_type="repo", title="old repo", summary="s",
            url="https://github.com/a/b", canonical_url="https://github.com/a/b",
            published_at="2020-01-01T00:00:00Z",
            metadata={"fetched_at": "2026-04-04T00:00:00Z"},
        )
        kept = _apply_freshness_gates([trend], target, max_item_age_days=14,
                                      freshness_hours=24)
        self.assertEqual(len(kept), 1)  # exempt → kept despite 2020 published_at

    def test_golden_eight_of_fortyone_sink_without_agents(self) -> None:
        # 41 papers: 8 are stale-verified (v1 > 14d old), 33 are genuinely fresh.
        target = datetime(2026, 4, 4, tzinfo=UTC)
        items = []
        for n in range(33):
            items.append(_paper(f"2604.1{n:04d}", "2026-04-04T00:00:00Z",
                                 "2026-04-04T00:00:00Z"))
        for n in range(8):
            items.append(_paper(f"2301.0{n:04d}", "2026-04-04T00:00:00Z",
                                 "2023-01-02T00:00:00Z"))
        kept = _apply_freshness_gates(items, target, max_item_age_days=14,
                                      freshness_hours=24)
        self.assertEqual(len(kept), 33)  # exactly the 8 stale ones sank, zero agents


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stage1_golden.py -v`
Expected: FAIL with `ImportError: cannot import name '_apply_freshness_gates'`

- [ ] **Step 3: Write minimal implementation**

Extract the inline gate (current lines 1653-1687) into a testable pure function and call it from `generate_daily_hotspot_report`. Add near the other pipeline helpers:

```python
def _apply_freshness_gates(
    items: list,
    target_date: datetime,
    *,
    max_item_age_days: int,
    freshness_hours: int,
) -> list:
    """Freshness + max-age hard gate on the day-granular gate_date (spec §B.5/§B.5.1).

    Uses gate_date(item) (verified_first_date floored to UTC day) so sub-day jitter
    cannot flip the discrete gate (INV2). github_trend is exempt from max-age (it
    legitimately trends long after creation). Items with no credible date are kept
    (cannot-verify → do not drop), matching the legacy policy.
    """
    from datetime import timezone as _tz
    from arxiv_assistant.utils.hotspot.gate_date import gate_date as _gate_date_fn

    target_utc = target_date.replace(tzinfo=_tz.utc) if target_date.tzinfo is None else target_date
    run_day = target_utc.date()
    max_age = max_item_age_days
    fresh_days = freshness_hours / 24.0

    kept = []
    for item in items:
        if item.source_id in {"github_trend"}:
            kept.append(item)
            continue
        gday = _gate_date_fn(item)
        if gday is None:
            kept.append(item)  # cannot verify → do not drop
            continue
        age_days = (run_day - gday).days
        if age_days > max_age:
            continue  # too old (hard gate)
        if age_days < -1:
            continue  # future-dated artifact (allow +1 day for tz slack)
        if age_days > fresh_days and age_days > max_age:
            continue  # redundant safety; max_age already dominates
        kept.append(item)
    return kept
```

Then replace the inline block at lines 1653-1687 in `generate_daily_hotspot_report` with a DateVerify pass + the extracted call:

```python
    # DateVerify (Tier-0, zero agents) → set verified_first_date for the gates.
    from arxiv_assistant.hotspots.date_verify import verify as _date_verify
    store = _open_story_store(output_root)  # Stage 0 helper; opens out/hot/state/story_store.sqlite
    for item in raw_items:
        verdict = _date_verify(item, store)
        item.verified_first_date = verdict.get("verified_first_date")

    # Freshness + max-age hard gate on day-granular gate_date (§B.5/§B.5.1).
    max_item_age_days = config["HOTSPOTS"].getint("max_item_age_days", fallback=14)
    freshness_hours = config["HOTSPOTS"].getint("freshness_hours", fallback=24)
    pre_gate = len(raw_items)
    raw_items = _apply_freshness_gates(
        raw_items, target_date,
        max_item_age_days=max_item_age_days, freshness_hours=freshness_hours,
    )
    if len(raw_items) < pre_gate:
        print(f"Freshness/max-age gate: removed {pre_gate - len(raw_items)} stale items ({len(raw_items)} remaining)")
```

> NOTE: `_open_story_store(output_root)` is the Stage 0 helper that returns a `StoryStore` over `out/hot/state/story_store.sqlite`. If Stage 0's helper is named differently, use `StoryStore(output_root / "hot" / "state" / "story_store.sqlite")` directly with the §2.3 constructor. Do NOT invent a new store API here.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stage1_golden.py -v`
Expected: PASS (3 tests; the golden test confirms exactly 33/41 kept = 8 stale sank with zero agents)

- [ ] **Step 5: Commit**

```bash
git add arxiv_assistant/hotspots/pipeline.py tests/test_stage1_golden.py
git commit -m "feat(hotspot): max-age hard gate via verified_first_date→gate_date (8/41 sink, zero agents)"
```

---

## Task 10: Config — `max_item_age_days` under `[HOTSPOTS]`

**Files:**
- Modify: `configs/config.ini` (`[HOTSPOTS]` section)
- Modify: `configs/templates/config.template.ini` (`[HOTSPOTS]` section)
- Test: `tests/test_stage1_golden.py` (append a config-read assertion)

§3: `max_item_age_days = 14` with default, documented in the template.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stage1_golden.py  (append)
import configparser
from pathlib import Path


class TestConfigMaxItemAge(unittest.TestCase):
    def test_config_has_max_item_age_days(self) -> None:
        cfg = configparser.ConfigParser()
        cfg.read(Path("configs/config.ini"), encoding="utf-8")
        self.assertEqual(cfg["HOTSPOTS"].getint("max_item_age_days", fallback=-1), 14)

    def test_template_documents_max_item_age_days(self) -> None:
        text = Path("configs/templates/config.template.ini").read_text(encoding="utf-8")
        self.assertIn("max_item_age_days", text)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stage1_golden.py::TestConfigMaxItemAge -v`
Expected: FAIL — `max_item_age_days` not present (returns fallback -1; template lacks the key).

- [ ] **Step 3: Write minimal implementation**

Locate the `[HOTSPOTS]` section in both files. Add under it (alongside the existing `freshness_hours`):

In `configs/config.ini`:
```ini
max_item_age_days = 14
```

In `configs/templates/config.template.ini`:
```ini
# Hard upper bound on item age (days) measured from the day-granular verified
# first-publication date (gate_date). Items older than this are dropped before
# scoring; github_trend is exempt. Spec §B.5.
max_item_age_days = 14
```

> NOTE: Confirm the exact `[HOTSPOTS]` header location by reading each file first; insert the key under that header, not at file end.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stage1_golden.py::TestConfigMaxItemAge -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add configs/config.ini configs/templates/config.template.ini tests/test_stage1_golden.py
git commit -m "feat(hotspot): add max_item_age_days config (default 14)"
```

---

## Task 11: §G acceptance tests — INV1 + INV2 invariants

**Files:**
- Test: `tests/test_stage1_golden.py` (append `TestInvariants`)

§4 / §G: lock the two invariants this stage owns as explicit acceptance tests so a regression in any of the touched call sites fails loudly.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stage1_golden.py  (append)
from arxiv_assistant.utils.hotspot.gate_date import gate_date
from arxiv_assistant.utils.hotspot.hotspot_sources import get_freshness_date


class TestInvariants(unittest.TestCase):
    def test_inv1_gates_use_verified_not_claimed(self) -> None:
        # INV1: a 2023-verified paper claiming 2026 must gate as 2023 everywhere.
        item = _paper("2301.00001", "2026-04-04T00:00:00Z", "2023-01-02T00:00:00Z")
        self.assertEqual(gate_date(item), date(2023, 1, 2))
        self.assertEqual(get_freshness_date(item), "2023-01-02T00:00:00Z")

    def test_inv2_subday_jitter_cannot_flip_gate(self) -> None:
        # INV2: two sub-day-jittered verified dates on the same UTC day → identical
        # gate decision (both kept or both dropped), never split.
        target = datetime(2026, 4, 18, tzinfo=UTC)  # 14 days after 2026-04-04
        a = _paper("2604.0000a", "2026-04-04T00:00:01Z", "2026-04-04T00:00:01Z")
        b = _paper("2604.0000b", "2026-04-04T23:59:59Z", "2026-04-04T23:59:59Z")
        kept = _apply_freshness_gates([a, b], target, max_item_age_days=14,
                                      freshness_hours=24)
        # Both have gate_date 2026-04-04, exactly 14 days old → both kept together.
        self.assertEqual(len(kept), 2)

    def test_inv2_boundary_is_day_not_instant(self) -> None:
        target = datetime(2026, 4, 19, tzinfo=UTC)  # 15 days after 2026-04-04
        a = _paper("2604.0000a", "2026-04-04T00:00:01Z", "2026-04-04T00:00:01Z")
        b = _paper("2604.0000b", "2026-04-04T23:59:59Z", "2026-04-04T23:59:59Z")
        kept = _apply_freshness_gates([a, b], target, max_item_age_days=14,
                                      freshness_hours=24)
        # Both gate_date 2026-04-04, now 15 days old → both dropped together.
        self.assertEqual(len(kept), 0)
```

- [ ] **Step 2: Run test to verify it fails (or passes if Tasks 1-9 complete)**

Run: `pytest tests/test_stage1_golden.py::TestInvariants -v`
Expected: PASS if Tasks 1-9 landed correctly. If any fails, the implicated task's edit regressed an invariant — fix there, do not weaken the assertion.

- [ ] **Step 3: No new implementation**

These are acceptance tests over already-implemented behavior. If green, proceed; if red, return to the failing task.

- [ ] **Step 4: Run the full Stage-1 suite**

Run: `pytest tests/test_gate_date.py tests/test_date_verify.py tests/test_stage1_golden.py tests/test_hotspot_pipeline.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add tests/test_stage1_golden.py
git commit -m "test(hotspot): lock INV1/INV2 staleness invariants as acceptance tests"
```

---

## Self-Review

**1. Spec coverage**

| Spec / contract section | Task(s) |
|---|---|
| §2.4 `floor_to_utc_day` | Task 1 |
| §2.4 `gate_date` (min credible ∪ whole-day anchors) | Task 2 |
| §B.2 Tier-0 arXiv v1 submission date | Task 3 |
| §B.2 Crossref/DOI registration date | Task 4 |
| §B.3 earliest-credible-date-wins + §B.4 write-once cache + github_trend exception | Task 5 |
| §2.5 `poll_arxiv_versions` (§B.4.1) | Task 6 |
| §2.1/§B.5 `get_freshness_date` reads verified_first_date | Task 7 |
| §B.5.1/§C.3 gravity from gate_date (HN-style) | Task 8 |
| §B.5/§B.5.1 max-age hard gate (github_trend exempt) | Task 9 |
| §3 `max_item_age_days` config | Task 10 |
| §G INV1 + INV2 acceptance tests | Tasks 2, 5, 9, 11 |

No gaps for the assigned scope. Tier-1/2 subagent, Wayback CDX, `date_verdicts` snapshot travel, and `arxiv_versions` monotonic merge are explicitly Stage 3 and excluded by the stage boundary — represented here only by the legal conservative fallback (Task 5) and the count-only read (Task 6).

**2. Placeholder scan**

No "TBD/TODO/implement later". The one cross-stage seam (Task 9's `_open_story_store`) carries a concrete fallback (`StoryStore(...)` with the §2.3 constructor) so the task is executable even if Stage 0's helper name differs. The Tier-1/2 branch is a *complete* conservative implementation (`min(claimed, fetched) + 0.3 conf`), not a stub — matching the task instruction to avoid placeholder semantics.

**3. Type consistency**

- `floor_to_utc_day(str|None) -> date|None` and `gate_date(item) -> date|None` used consistently (Tasks 1, 2, 8, 9, 11).
- `verify(item, store) -> dict` with keys `verified_first_date`/`confidence`/`evidence` consistent across Tasks 5, 9, and the `_FakeStore` mirrors Stage 0's `get_verdict`/`put_verdict` write-once semantics (§2.3).
- `_freshness_weight(date|None, *, run_date: date) -> float` — call site in `score_stories` (Task 8) passes the earliest `gate_date` and `run_date`, matching the new signature; the public `score_stories(list[Story]) -> list[Story]` signature is unchanged, so existing callers/tests stay green.
- `_apply_freshness_gates(items, target_date, *, max_item_age_days, freshness_hours)` defined in Task 9, reused verbatim in Task 11.
- `_content_hash` precedence (arxiv > doi > url) consistent between Task 5 definition and its test.

All consistent. Plan is internally coherent and contract-faithful.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-03-agent-native-rewrite-02-stage1-staleness.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
