# Stage 3 — DateVerify Subagent & Version Polling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `arxiv_assistant/hotspots/date_verify.py` with the Tier-1/2 stateless DateVerify subagent (Wayback CDX + `article:published_time`/JSON-LD + earliest-credible-report search → deterministic earliest-min verifier), wire authoritative whole-day anchors into `gate_date`, and add monotonic batched `poll_arxiv_versions`, all frozen permanently via the StoryStore verdict cache.

**Architecture:** A thin deterministic Python kernel owns the topology; the only randomness is a stateless, short-lived Claude Code headless subagent invoked at the Tier-1 (residual news/blog/X items) and Tier-2 (uncertain-and-will-be-featured) judgment points. Its typed JSON output is ALWAYS clamped by a deterministic verifier (earliest-credible monotonic `min` + UTC-day floor, INV6) before being frozen write-once in `date_verdicts` (INV3). Version-count polling is a separate cheap deterministic Tier-0 read that maintains a monotonic `Story.arxiv_versions` counter and NEVER touches `date_verdicts`.

**Tech Stack:** Python 3, `pytest` (`unittest.TestCase` style), `subprocess` + `claude -p` headless for the subagent, `requests` (already a project dep, used by `hotspot_sources.fetch_text`/`fetch_json`) for arXiv `id_list` + Wayback CDX, `@patch` record/replay fixtures under `tests/fixtures/agent/`. No network in tests.

---

## 0. Preconditions & contract anchors

This stage **extends** an existing Stage 1 `date_verify.py` (Tier-0 `verify()` + a Stage 1 `gate_date()` in `utils/hotspot/gate_date.py`). Do **not** rewrite Tier-0. The locked signatures you must match (overview §2.3, §2.4, §2.5):

```python
# hotspots/store.py  (Stage 0 — already exists)
def get_verdict(self, content_hash: str) -> dict | None: ...          # {verified_first_date, confidence, evidence}
def put_verdict(self, content_hash: str, verdict: dict) -> None: ...   # write-once; no-op if exists
def refresh_arxiv_versions(self, arxiv_id: str, fetched_count: int) -> None: ...  # new := max(old, fetched)

# utils/hotspot/gate_date.py  (Stage 1 — already exists, this stage extends credible_dates)
def floor_to_utc_day(iso_ts: str | None) -> date | None: ...
def gate_date(item: HotspotItem) -> date | None: ...

# hotspots/date_verify.py  (Stage 1 ships tier-0; THIS stage adds tier-1/2 + poll)
def verify(item: HotspotItem, store: StoryStore) -> dict: ...
def poll_arxiv_versions(arxiv_ids: list[str]) -> dict[str, int]: ...
```

**Invariants this stage must assert as acceptance tests** (overview §4):
- **INV1** — dates driving gates are `verified_first_date` only, NEVER source-claimed (`item.published_at`).
- **INV2** — discrete gates consume `gate_date` (day-granular); sub-day jitter cannot flip a gate.
- **INV3** — `verified_first_date` is write-once (cache freeze); `arxiv_versions` is monotonic and NOT in `date_verdicts`.
- **INV6** — every random agent is followed by a deterministic verifier; temp 0; pinned model id.

**Spec sections owned by this stage:** §B.1 (every featured candidate gets a `verified_first_date`; suspicion is a priority signal, not a gate), §B.2 (three-tier ladder: Tier-1 Wayback CDX + `published_time`/JSON-LD + earliest-mention search; Tier-2 deep search only for uncertain-and-will-be-featured), §B.3 (earliest-credible-date-wins; mark `stale_date_pollution`), §B.3.1 (authoritative whole-day anchors folded into the credible set), §B.4 (permanent cache = freeze), §B.4.1 (version-count polling: batched `id_list` ≤100/call ~1 req/3s, monotonic, NOT in `date_verdicts`).

**If Stage 1 is not yet merged when you start:** create minimal Tier-0 stubs in `date_verify.py` and `gate_date.py` that satisfy the signatures above (a `verify()` that returns `{"verified_first_date": item.published_at, "confidence": 0.3, "evidence": []}` when no arXiv/DOI anchor exists, and a `gate_date()` that floors `verified_first_date`). Replace, don't duplicate, when Stage 1 lands. All tasks below patch around these so they pass regardless.

---

## 1. Files this stage creates or modifies

```
arxiv_assistant/hotspots/
  date_verify.py        MOD  — add: _content_hash, _wayback_earliest_snapshot, _page_published_time,
                                _run_dateverify_subagent (claude -p), _verify_subagent_residual (Tier-1),
                                _deep_search_tier2 (Tier-2), _clamp_verdict (deterministic verifier),
                                extend verify() to dispatch Tier-1/2, poll_arxiv_versions()
arxiv_assistant/utils/hotspot/
  gate_date.py          MOD  — credible_dates() folds in authoritative whole-day anchors
                                (arXiv announced day / Crossref registration day)  [§B.3.1 wiring]
tests/
  test_date_verify.py   MOD  — append Tier-1 replay, freeze-once, poll monotonicity, INV1/2/3/6 cases
tests/fixtures/agent/
  dateverify_tier1_stale_pollution.json   NEW  — captured subagent JSON (claimed-later-than-Wayback)
  dateverify_tier1_clean.json             NEW  — captured subagent JSON (claimed matches earliest)
  dateverify_tier2_deep.json              NEW  — captured Tier-2 deep-search subagent JSON
```

Keep `date_verify.py` under ~300 lines (overview §1 file-size discipline). The subagent transport (`_run_dateverify_subagent`) is the only impure function; everything downstream of it is deterministic and unit-tested without network.

---

## 2. Subagent calling convention (Claude Code headless)

The Tier-1/2 subagent is **stateless and short-lived**: one `claude -p` invocation per residual item, typed JSON in → typed JSON out, temperature 0, pinned model. No cross-day memory, no inter-agent communication, no Store writes from inside the agent (only the kernel writes the Store — overview §2.3, spec §A.1/§A.2).

**Transport** (`_run_dateverify_subagent(payload: dict) -> dict`):
- Serialize the typed input to JSON, pass on stdin to `subprocess.run(["claude", "-p", PROMPT, "--output-format", "json", "--model", DATEVERIFY_MODEL_ID], ...)`.
- The prompt instructs the agent to cross at least two independent signals (Wayback CDX earliest snapshot, page `article:published_time`/JSON-LD `datePublished`, earliest credible report search) and emit **only** the typed JSON object below — no prose.
- Parse stdout as JSON; on any failure (non-zero exit, unparseable, schema-invalid) the caller falls back deterministically (spec §E: `min(claimed, fetched)` + low confidence, item folded below the fold, never crash).

**Pinned identity** (INV6): module constant `DATEVERIFY_MODEL_ID = "claude-opus-4-8"` recorded on every verdict's `evidence` provenance and in the run manifest; temperature 0 is implied by `claude -p` deterministic single-shot.

**Typed input schema** (kernel → subagent):
```json
{
  "schema": "dateverify.in.v1",
  "url": "https://example.com/blog/agent-breakthrough",
  "title": "A New Agent Breakthrough",
  "claimed_date": "2026-06-02T09:00:00Z",
  "tier": 1,
  "wayback_earliest": "2023-11-14T00:00:00Z",
  "page_published_time": "2023-11-14T08:30:00Z"
}
```
`wayback_earliest` and `page_published_time` are pre-fetched deterministically by the kernel (cheap `requests` reads) and handed to the agent as evidence; the agent may also search for the earliest credible report. This keeps the anti-pollution signals deterministic and gives the agent grounded inputs.

**Typed output schema** (subagent → kernel):
```json
{
  "schema": "dateverify.out.v1",
  "verified_first_date": "2023-11-14T00:00:00Z",
  "confidence": 0.9,
  "evidence": [
    "wayback_cdx:20231114083012",
    "article:published_time=2023-11-14T08:30:00Z",
    "earliest_report:https://news.example.com/2023/11/14/x"
  ],
  "stale_date_pollution": true
}
```

**Record/replay in tests:** capture one real (or hand-authored-to-schema) response per scenario into `tests/fixtures/agent/*.json`; `@patch("...date_verify._run_dateverify_subagent")` to return the parsed fixture. Tests assert the **deterministic verifier** (`_clamp_verdict`) and downstream `verify()` behavior — never the agent. No network is ever hit in tests.

---

## 3. §B.3.1 wiring: authoritative whole-day anchors into `gate_date`

Stage 1's `gate_date(item)` floors `min(credible_dates(item))`. This stage **adds** the authoritative whole-day anchors to that credible set so academic/official items take a machine-independent integer-day lower bound (spec §B.3.1):

- `arxiv_id` present → arXiv **announced day** (same source as v1 submission, already UTC day-granular) joins the credible set.
- DOI present → Crossref/DataCite registration day joins the credible set.
- `gate_date := floor_to_utc_day( min( authoritative_whole_day_anchors ∪ {verified_first_date} ) )` — earliest-min still applies; anchors only ever pull the gate **earlier** (anti-pollution monotonic, INV2).

Wiring point: extend the `credible_dates(item)` helper inside `gate_date.py` (Task 1). The anchor day is read from `item.metadata` (the kernel stamps `arxiv_announced_day` / `crossref_registered_day` during Tier-0 so `gate_date` stays pure and offline). `gate_date` itself performs **no** network I/O — it consumes the kernel-stamped anchors, preserving its golden-fixture testability (overview §2.4: "Pure → golden-fixture tested").

---

## Task 1: Fold authoritative whole-day anchors into `gate_date.credible_dates`

**Files:**
- Modify: `arxiv_assistant/utils/hotspot/gate_date.py`
- Test: `tests/test_date_verify.py` (append; this file already exists from Stage 1)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_date_verify.py`:

```python
class TestGateDateAuthoritativeAnchor(unittest.TestCase):
    def _item(self, *, published_at, metadata=None):
        return HotspotItem(
            source_id="hf_papers",
            source_name="HF",
            source_role="paper_trending",
            source_type="paper",
            title="Old paper resurfaced as new",
            summary="",
            url="https://huggingface.co/papers/2311.01234",
            canonical_url="https://arxiv.org/abs/2311.01234",
            published_at=published_at,
            metadata=metadata or {},
        )

    def test_arxiv_announced_day_pulls_gate_earlier_than_claimed(self):
        # claimed (HF publishedAt) is today; arXiv announced day is 2023 -> gate must use the 2023 day
        item = self._item(
            published_at="2026-06-02T09:00:00Z",
            metadata={"arxiv_id": "2311.01234", "arxiv_announced_day": "2023-11-14"},
        )
        item.verified_first_date = "2023-11-14T00:00:00Z"
        self.assertEqual(gate_date(item), date(2023, 11, 14))

    def test_anchor_never_pulls_gate_later(self):
        # verified_first_date earlier than the anchor -> earliest-min keeps the earlier verified date
        item = self._item(
            published_at="2026-06-02T09:00:00Z",
            metadata={"arxiv_id": "2311.01234", "arxiv_announced_day": "2023-11-20"},
        )
        item.verified_first_date = "2023-11-14T00:00:00Z"
        self.assertEqual(gate_date(item), date(2023, 11, 14))

    def test_crossref_registration_day_used_when_doi_present(self):
        item = self._item(
            published_at="2026-06-02T09:00:00Z",
            metadata={"doi": "10.1234/x", "crossref_registered_day": "2024-02-01"},
        )
        item.verified_first_date = None
        self.assertEqual(gate_date(item), date(2024, 2, 1))
```

Ensure the test module imports exist at the top of `tests/test_date_verify.py`:
```python
from datetime import date
from arxiv_assistant.utils.hotspot.gate_date import gate_date, floor_to_utc_day
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_date_verify.py::TestGateDateAuthoritativeAnchor -v`
Expected: FAIL — either `gate_date` ignores the anchors (returns the 2026 day or `None`) or `AttributeError`/assertion mismatch.

- [ ] **Step 3: Extend `credible_dates` in `gate_date.py`**

In `arxiv_assistant/utils/hotspot/gate_date.py`, replace the `credible_dates` helper body so it folds in the kernel-stamped authoritative whole-day anchors. Keep the function pure (no I/O):

```python
def credible_dates(item: HotspotItem) -> list[str]:
    """All machine-independent credible dates for an item, as ISO strings.

    §B.3.1: authoritative whole-day anchors (arXiv announced day / Crossref
    registration day) join {verified_first_date}. Anchors are kernel-stamped
    into item.metadata during Tier-0 so this function performs no network I/O.
    """
    dates: list[str] = []
    verified = getattr(item, "verified_first_date", None)
    if verified:
        dates.append(verified)
    meta = item.metadata or {}
    announced = meta.get("arxiv_announced_day")
    if announced:
        # whole-day anchor; normalise to start-of-day ISO so floor_to_utc_day is a no-op
        dates.append(f"{announced}T00:00:00Z")
    registered = meta.get("crossref_registered_day")
    if registered:
        dates.append(f"{registered}T00:00:00Z")
    return dates


def gate_date(item: HotspotItem) -> date | None:
    candidates = [floor_to_utc_day(d) for d in credible_dates(item)]
    candidates = [d for d in candidates if d is not None]
    if not candidates:
        return None
    return min(candidates)
```

> If Stage 1 already defines `gate_date` differently, keep its signature and only ensure `credible_dates` includes the two anchor keys plus `verified_first_date`, and that `gate_date` returns `min(...)` of the floored set.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_date_verify.py::TestGateDateAuthoritativeAnchor -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add arxiv_assistant/utils/hotspot/gate_date.py tests/test_date_verify.py
git commit -m "feat(hotspot): fold arXiv/Crossref whole-day anchors into gate_date credible set (B.3.1)"
```

---

## Task 2: Deterministic verdict clamp (`_clamp_verdict`) — the verifier behind the agent (INV6)

**Files:**
- Modify: `arxiv_assistant/hotspots/date_verify.py`
- Test: `tests/test_date_verify.py`

This is the deterministic verifier that ALWAYS runs after the subagent. It enforces earliest-credible monotonic `min`, day-granular floor (INV2/INV6), and sets `stale_date_pollution` from facts, not from the agent's say-so.

- [ ] **Step 1: Write the failing test**

```python
class TestClampVerdict(unittest.TestCase):
    def test_clamp_picks_earliest_credible_and_floors_to_day(self):
        from arxiv_assistant.hotspots.date_verify import _clamp_verdict
        clamped = _clamp_verdict(
            claimed_iso="2026-06-02T09:00:00Z",
            agent_out={
                "verified_first_date": "2023-11-14T08:30:00Z",
                "confidence": 0.9,
                "evidence": ["wayback_cdx:20231114083012"],
                "stale_date_pollution": True,
            },
            wayback_earliest="2023-11-14T00:00:00Z",
            page_published_time="2023-11-14T08:30:00Z",
        )
        # earliest-credible-date-wins -> the 2023 day, not the claimed 2026 day
        self.assertEqual(clamped["verified_first_date"], "2023-11-14T00:00:00Z")
        self.assertTrue(clamped["stale_date_pollution"])
        self.assertGreaterEqual(clamped["confidence"], 0.0)

    def test_clamp_ignores_agent_date_later_than_evidence(self):
        from arxiv_assistant.hotspots.date_verify import _clamp_verdict
        # agent hallucinates a LATER date than Wayback proves -> verifier overrides with the earlier
        clamped = _clamp_verdict(
            claimed_iso="2026-06-02T09:00:00Z",
            agent_out={
                "verified_first_date": "2026-06-02T00:00:00Z",
                "confidence": 0.95,
                "evidence": [],
                "stale_date_pollution": False,
            },
            wayback_earliest="2023-11-14T00:00:00Z",
            page_published_time=None,
        )
        self.assertEqual(clamped["verified_first_date"], "2023-11-14T00:00:00Z")
        self.assertTrue(clamped["stale_date_pollution"])

    def test_clamp_falls_back_to_min_claimed_fetched_when_no_signals(self):
        from arxiv_assistant.hotspots.date_verify import _clamp_verdict
        clamped = _clamp_verdict(
            claimed_iso="2026-06-02T09:00:00Z",
            agent_out=None,            # agent failed / unparseable
            wayback_earliest=None,
            page_published_time=None,
        )
        # no credible earlier signal -> conservative claimed day, low confidence, not flagged
        self.assertEqual(clamped["verified_first_date"], "2026-06-02T00:00:00Z")
        self.assertLess(clamped["confidence"], 0.5)
        self.assertFalse(clamped["stale_date_pollution"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_date_verify.py::TestClampVerdict -v`
Expected: FAIL with `ImportError`/`AttributeError: _clamp_verdict`.

- [ ] **Step 3: Implement `_clamp_verdict` (and helpers) in `date_verify.py`**

Add near the top of `arxiv_assistant/hotspots/date_verify.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone

DATEVERIFY_MODEL_ID = "claude-opus-4-8"


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _floor_day_iso(value: str | None) -> str | None:
    """Truncate any ISO timestamp to UTC start-of-day ISO (INV2)."""
    dt = _parse_iso(value)
    if dt is None:
        return None
    dt = dt.astimezone(timezone.utc)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0).isoformat().replace("+00:00", "Z")


def _clamp_verdict(
    *,
    claimed_iso: str | None,
    agent_out: dict | None,
    wayback_earliest: str | None,
    page_published_time: str | None,
) -> dict:
    """Deterministic verifier behind the Tier-1/2 agent (INV6).

    earliest-credible-date-wins: verified_first_date := min over all credible
    signals (agent date, Wayback earliest snapshot, page published_time, claimed),
    floored to UTC day. stale_date_pollution is derived from facts: claimed day
    strictly later than the earliest credible day.
    """
    credible: list[str] = []
    for raw in (
        (agent_out or {}).get("verified_first_date"),
        wayback_earliest,
        page_published_time,
        claimed_iso,
    ):
        floored = _floor_day_iso(raw)
        if floored is not None:
            credible.append(floored)

    if not credible:
        # nothing parseable at all -> degrade, never crash (spec §E)
        return {"verified_first_date": None, "confidence": 0.2, "evidence": [], "stale_date_pollution": False}

    earliest = min(credible)  # ISO day strings sort chronologically
    claimed_day = _floor_day_iso(claimed_iso)

    has_external_signal = bool(
        _floor_day_iso((agent_out or {}).get("verified_first_date"))
        or _floor_day_iso(wayback_earliest)
        or _floor_day_iso(page_published_time)
    )
    stale = bool(claimed_day is not None and earliest < claimed_day)

    if has_external_signal:
        # trust the agent's confidence but never below floor; cap to [0,1]
        confidence = max(0.0, min(1.0, float((agent_out or {}).get("confidence", 0.7))))
    else:
        # no credible earlier signal -> conservative min(claimed, fetched) low confidence
        confidence = 0.3

    evidence = list((agent_out or {}).get("evidence", []))
    evidence.append(f"verifier:earliest_min;model={DATEVERIFY_MODEL_ID}")
    return {
        "verified_first_date": earliest,
        "confidence": confidence,
        "evidence": evidence,
        "stale_date_pollution": stale,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_date_verify.py::TestClampVerdict -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add arxiv_assistant/hotspots/date_verify.py tests/test_date_verify.py
git commit -m "feat(hotspot): deterministic earliest-min verdict clamp behind DateVerify agent (B.3, INV6)"
```

---

## Task 3: Deterministic anti-pollution reads — Wayback CDX earliest snapshot + page published_time

**Files:**
- Modify: `arxiv_assistant/hotspots/date_verify.py`
- Test: `tests/test_date_verify.py`

These are cheap deterministic `requests` reads the kernel performs to pre-stamp the agent input. They are patched in tests; no network is hit.

- [ ] **Step 1: Write the failing test**

```python
class TestAntiPollutionReads(unittest.TestCase):
    def test_wayback_earliest_snapshot_parses_first_timestamp(self):
        from arxiv_assistant.hotspots import date_verify
        # CDX returns rows [["timestamp"], ["20231114083012"], ["20240101000000"]]
        cdx_rows = [["timestamp"], ["20231114083012"], ["20240101000000"]]
        with patch.object(date_verify, "fetch_json", return_value=cdx_rows):
            earliest = date_verify._wayback_earliest_snapshot("https://example.com/x")
        self.assertEqual(earliest, "2023-11-14T08:30:12Z")

    def test_wayback_earliest_snapshot_returns_none_on_empty(self):
        from arxiv_assistant.hotspots import date_verify
        with patch.object(date_verify, "fetch_json", return_value=[["timestamp"]]):
            self.assertIsNone(date_verify._wayback_earliest_snapshot("https://example.com/x"))

    def test_wayback_earliest_snapshot_returns_none_on_network_error(self):
        from arxiv_assistant.hotspots import date_verify
        with patch.object(date_verify, "fetch_json", side_effect=RuntimeError("boom")):
            self.assertIsNone(date_verify._wayback_earliest_snapshot("https://example.com/x"))

    def test_page_published_time_reads_meta_property(self):
        from arxiv_assistant.hotspots import date_verify
        html = '<html><head><meta property="article:published_time" content="2023-11-14T08:30:00Z"></head></html>'
        with patch.object(date_verify, "fetch_text", return_value=html):
            self.assertEqual(date_verify._page_published_time("https://example.com/x"), "2023-11-14T08:30:00Z")

    def test_page_published_time_reads_jsonld_datepublished(self):
        from arxiv_assistant.hotspots import date_verify
        html = (
            '<html><head><script type="application/ld+json">'
            '{"@type":"Article","datePublished":"2024-02-01T00:00:00Z"}'
            '</script></head></html>'
        )
        with patch.object(date_verify, "fetch_text", return_value=html):
            self.assertEqual(date_verify._page_published_time("https://example.com/x"), "2024-02-01T00:00:00Z")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_date_verify.py::TestAntiPollutionReads -v`
Expected: FAIL with `AttributeError: _wayback_earliest_snapshot` / `_page_published_time`.

- [ ] **Step 3: Implement the two deterministic reads**

Add to `date_verify.py`. Import the existing project HTTP helpers at module top so they can be patched by name:

```python
import json
import re

from arxiv_assistant.utils.hotspot.hotspot_sources import fetch_json, fetch_text

_WAYBACK_CDX = "http://web.archive.org/cdx/search/cdx"
_META_PUBLISHED = re.compile(
    r'<meta[^>]+(?:property|name)=["\']article:published_time["\'][^>]+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_JSONLD = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.IGNORECASE | re.DOTALL,
)


def _cdx_ts_to_iso(ts: str) -> str | None:
    if not ts or len(ts) < 14 or not ts.isdigit():
        return None
    return f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}T{ts[8:10]}:{ts[10:12]}:{ts[12:14]}Z"


def _wayback_earliest_snapshot(url: str) -> str | None:
    """Earliest Wayback CDX capture timestamp for url, ISO8601, or None.

    Anti-pollution main signal (§B.2 Tier-1 (a)): a backdated 'new' page that
    was actually archived years ago betrays itself via its earliest capture.
    """
    try:
        rows = fetch_json(
            _WAYBACK_CDX,
            params={"url": url, "output": "json", "fl": "timestamp", "limit": "5", "from": "19960101"},
        )
    except Exception:
        return None
    if not isinstance(rows, list) or len(rows) < 2:
        return None
    timestamps = [r[0] for r in rows[1:] if r and isinstance(r[0], str)]
    iso = [_cdx_ts_to_iso(ts) for ts in timestamps]
    iso = [v for v in iso if v]
    return min(iso) if iso else None


def _page_published_time(url: str) -> str | None:
    """article:published_time meta or JSON-LD datePublished from the live page (§B.2 Tier-1 (b))."""
    try:
        html = fetch_text(url)
    except Exception:
        return None
    m = _META_PUBLISHED.search(html)
    if m:
        return m.group(1).strip()
    for block in _JSONLD.findall(html):
        try:
            data = json.loads(block)
        except (ValueError, TypeError):
            continue
        candidates = data if isinstance(data, list) else [data]
        for obj in candidates:
            if isinstance(obj, dict) and obj.get("datePublished"):
                return str(obj["datePublished"]).strip()
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_date_verify.py::TestAntiPollutionReads -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add arxiv_assistant/hotspots/date_verify.py tests/test_date_verify.py
git commit -m "feat(hotspot): Wayback CDX earliest + page published_time anti-pollution reads (B.2 Tier-1)"
```

---

## Task 4: Subagent transport + record/replay fixtures

**Files:**
- Modify: `arxiv_assistant/hotspots/date_verify.py`
- Create: `tests/fixtures/agent/dateverify_tier1_stale_pollution.json`
- Create: `tests/fixtures/agent/dateverify_tier1_clean.json`
- Create: `tests/fixtures/agent/dateverify_tier2_deep.json`
- Test: `tests/test_date_verify.py`

- [ ] **Step 1: Create the captured fixtures (real schema-conformant JSON)**

`tests/fixtures/agent/dateverify_tier1_stale_pollution.json`:
```json
{
  "schema": "dateverify.out.v1",
  "verified_first_date": "2023-11-14T00:00:00Z",
  "confidence": 0.9,
  "evidence": [
    "wayback_cdx:20231114083012",
    "article:published_time=2023-11-14T08:30:00Z",
    "earliest_report:https://news.example.com/2023/11/14/agent-breakthrough"
  ],
  "stale_date_pollution": true
}
```

`tests/fixtures/agent/dateverify_tier1_clean.json`:
```json
{
  "schema": "dateverify.out.v1",
  "verified_first_date": "2026-06-02T00:00:00Z",
  "confidence": 0.88,
  "evidence": [
    "wayback_cdx:20260602071500",
    "article:published_time=2026-06-02T07:10:00Z"
  ],
  "stale_date_pollution": false
}
```

`tests/fixtures/agent/dateverify_tier2_deep.json`:
```json
{
  "schema": "dateverify.out.v1",
  "verified_first_date": "2024-09-30T00:00:00Z",
  "confidence": 0.82,
  "evidence": [
    "wayback_cdx:20240930120000",
    "earliest_report:https://blog.example.org/2024/09/30/first-mention",
    "deep_search:semantic_scholar:corpusId=987654"
  ],
  "stale_date_pollution": true
}
```

- [ ] **Step 2: Write the failing test for the transport**

```python
import json as _json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures" / "agent"


class TestSubagentTransport(unittest.TestCase):
    def test_run_subagent_parses_claude_json_envelope(self):
        from arxiv_assistant.hotspots import date_verify
        inner = (FIXTURES / "dateverify_tier1_stale_pollution.json").read_text(encoding="utf-8")
        # claude -p --output-format json wraps the model text in a "result" field
        envelope = _json.dumps({"type": "result", "subtype": "success", "result": inner})

        class _Proc:
            returncode = 0
            stdout = envelope
            stderr = ""

        with patch.object(date_verify.subprocess, "run", return_value=_Proc()):
            out = date_verify._run_dateverify_subagent({"schema": "dateverify.in.v1", "url": "https://x"})
        self.assertEqual(out["verified_first_date"], "2023-11-14T00:00:00Z")
        self.assertTrue(out["stale_date_pollution"])

    def test_run_subagent_returns_none_on_nonzero_exit(self):
        from arxiv_assistant.hotspots import date_verify

        class _Proc:
            returncode = 1
            stdout = ""
            stderr = "model error"

        with patch.object(date_verify.subprocess, "run", return_value=_Proc()):
            self.assertIsNone(date_verify._run_dateverify_subagent({"schema": "dateverify.in.v1"}))

    def test_run_subagent_returns_none_on_unparseable(self):
        from arxiv_assistant.hotspots import date_verify

        class _Proc:
            returncode = 0
            stdout = "not json at all"
            stderr = ""

        with patch.object(date_verify.subprocess, "run", return_value=_Proc()):
            self.assertIsNone(date_verify._run_dateverify_subagent({"schema": "dateverify.in.v1"}))
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_date_verify.py::TestSubagentTransport -v`
Expected: FAIL with `AttributeError: _run_dateverify_subagent` (and `date_verify.subprocess` missing).

- [ ] **Step 4: Implement the transport**

Add to `date_verify.py`:

```python
import subprocess

_DATEVERIFY_PROMPT = (
    "You are a stateless date-verification worker. Read the JSON object on stdin "
    "(schema dateverify.in.v1). Cross at least two independent signals to find the "
    "EARLIEST credible first-publication date: the provided Wayback earliest snapshot, "
    "the page article:published_time / JSON-LD datePublished, and an earliest credible "
    "report search for the title. Pollution backdates content to look new, so prefer the "
    "earliest credible date. Emit ONLY a JSON object of schema dateverify.out.v1 with keys "
    "verified_first_date (ISO8601), confidence (0..1), evidence (list of strings), "
    "stale_date_pollution (bool). No prose."
)


def _run_dateverify_subagent(payload: dict) -> dict | None:
    """Dispatch the stateless Tier-1/2 subagent via `claude -p` headless.

    Returns the parsed typed output (dateverify.out.v1) or None on any failure
    (caller falls back deterministically; spec §E). Patched in tests (record/replay).
    """
    try:
        proc = subprocess.run(
            ["claude", "-p", _DATEVERIFY_PROMPT, "--output-format", "json", "--model", DATEVERIFY_MODEL_ID],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        envelope = json.loads(proc.stdout)
    except ValueError:
        return None
    # claude -p --output-format json wraps the model's text in envelope["result"]
    inner = envelope.get("result", envelope) if isinstance(envelope, dict) else envelope
    if isinstance(inner, str):
        try:
            inner = json.loads(inner)
        except ValueError:
            return None
    if not isinstance(inner, dict) or "verified_first_date" not in inner:
        return None
    return inner
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_date_verify.py::TestSubagentTransport -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add arxiv_assistant/hotspots/date_verify.py tests/fixtures/agent/ tests/test_date_verify.py
git commit -m "feat(hotspot): claude -p DateVerify subagent transport + record/replay fixtures"
```

---

## Task 5: Tier-1 residual verification wired into `verify()` (replay + freeze)

**Files:**
- Modify: `arxiv_assistant/hotspots/date_verify.py`
- Test: `tests/test_date_verify.py`

This wires Tier-0 (existing) → Tier-1 (residual news/blog/X) → `_clamp_verdict` → `store.put_verdict` (write-once freeze). Tier-0 cache hit short-circuits everything.

- [ ] **Step 1: Write the failing test (Tier-1 replay: stale pollution + freeze-once)**

```python
class _FakeStore:
    """Minimal StoryStore stand-in honoring write-once put_verdict (INV3)."""
    def __init__(self):
        self._verdicts = {}
        self.put_calls = 0

    def get_verdict(self, content_hash):
        return self._verdicts.get(content_hash)

    def put_verdict(self, content_hash, verdict):
        self.put_calls += 1
        if content_hash in self._verdicts:
            return  # write-once: no-op if exists
        self._verdicts[content_hash] = dict(verdict)


def _news_item(url, claimed):
    return HotspotItem(
        source_id="ainews", source_name="AINews", source_role="roundup",
        source_type="news", title="Agent breakthrough", summary="",
        url=url, canonical_url=url, published_at=claimed,
    )


class TestTier1Verify(unittest.TestCase):
    def test_tier1_replay_uses_earlier_wayback_date_and_flags_pollution(self):
        from arxiv_assistant.hotspots import date_verify
        replay = _json.loads((FIXTURES / "dateverify_tier1_stale_pollution.json").read_text(encoding="utf-8"))
        store = _FakeStore()
        item = _news_item("https://example.com/blog/x", "2026-06-02T09:00:00Z")
        with patch.object(date_verify, "_wayback_earliest_snapshot", return_value="2023-11-14T00:00:00Z"), \
             patch.object(date_verify, "_page_published_time", return_value="2023-11-14T08:30:00Z"), \
             patch.object(date_verify, "_run_dateverify_subagent", return_value=replay):
            verdict = date_verify.verify(item, store)
        # claimed 2026 but Wayback proves 2023 -> earlier date wins, flagged stale
        self.assertEqual(verdict["verified_first_date"], "2023-11-14T00:00:00Z")
        self.assertTrue(verdict.get("stale_date_pollution"))

    def test_verdict_frozen_once_and_stable_across_calls(self):
        from arxiv_assistant.hotspots import date_verify
        replay = _json.loads((FIXTURES / "dateverify_tier1_stale_pollution.json").read_text(encoding="utf-8"))
        store = _FakeStore()
        item = _news_item("https://example.com/blog/x", "2026-06-02T09:00:00Z")
        with patch.object(date_verify, "_wayback_earliest_snapshot", return_value="2023-11-14T00:00:00Z"), \
             patch.object(date_verify, "_page_published_time", return_value="2023-11-14T08:30:00Z"), \
             patch.object(date_verify, "_run_dateverify_subagent", return_value=replay) as agent:
            first = date_verify.verify(item, store)
            # second call must hit the frozen cache: agent NOT called again, identical verdict
            agent.reset_mock()
            second = date_verify.verify(item, store)
        self.assertEqual(first["verified_first_date"], second["verified_first_date"])
        agent.assert_not_called()
        self.assertEqual(len(store._verdicts), 1)

    def test_inv1_gate_never_uses_source_claimed_date(self):
        # INV1: the verdict that drives gates is the verified date, not item.published_at
        from arxiv_assistant.hotspots import date_verify
        replay = _json.loads((FIXTURES / "dateverify_tier1_stale_pollution.json").read_text(encoding="utf-8"))
        store = _FakeStore()
        item = _news_item("https://example.com/blog/x", "2026-06-02T09:00:00Z")
        with patch.object(date_verify, "_wayback_earliest_snapshot", return_value="2023-11-14T00:00:00Z"), \
             patch.object(date_verify, "_page_published_time", return_value="2023-11-14T08:30:00Z"), \
             patch.object(date_verify, "_run_dateverify_subagent", return_value=replay):
            verdict = date_verify.verify(item, store)
        self.assertNotEqual(verdict["verified_first_date"], item.published_at)

    def test_inv2_subday_jitter_does_not_change_gate_day(self):
        # INV2: two agent runs differing only in sub-day H:M:S yield the same day verdict
        from arxiv_assistant.hotspots import date_verify
        store_a, store_b = _FakeStore(), _FakeStore()
        item_a = _news_item("https://example.com/a", "2026-06-02T09:00:00Z")
        item_b = _news_item("https://example.com/b", "2026-06-02T09:00:00Z")
        jitter_a = {"verified_first_date": "2026-06-02T06:00:00Z", "confidence": 0.8, "evidence": [], "stale_date_pollution": False}
        jitter_b = {"verified_first_date": "2026-06-02T23:59:00Z", "confidence": 0.8, "evidence": [], "stale_date_pollution": False}
        with patch.object(date_verify, "_wayback_earliest_snapshot", return_value=None), \
             patch.object(date_verify, "_page_published_time", return_value=None):
            with patch.object(date_verify, "_run_dateverify_subagent", return_value=jitter_a):
                va = date_verify.verify(item_a, store_a)
            with patch.object(date_verify, "_run_dateverify_subagent", return_value=jitter_b):
                vb = date_verify.verify(item_b, store_b)
        self.assertEqual(va["verified_first_date"], vb["verified_first_date"])  # both floored to 2026-06-02 day
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_date_verify.py::TestTier1Verify -v`
Expected: FAIL — Tier-1 dispatch not yet wired into `verify()` (likely returns Tier-0 stub / claimed date).

- [ ] **Step 3: Extend `verify()` to dispatch Tier-1 and freeze**

In `date_verify.py`, add the content-hash helper and the Tier-1 path. Preserve the existing Tier-0 logic — only **add** the residual branch and the cache freeze. The structure:

```python
import hashlib

# source families whose claimed dates are untrustworthy -> Tier-1 residual
_TIER1_FAMILIES = {"news", "blog", "x", "tweet", "roundup", "analysis"}


def _content_hash(item: HotspotItem) -> str:
    meta = item.metadata or {}
    key = meta.get("arxiv_id") or meta.get("doi") or item.canonical_url or item.url
    return hashlib.sha1(str(key).encode("utf-8")).hexdigest()


def _verify_subagent_residual(item: HotspotItem, *, tier: int) -> dict:
    """Tier-1/2 path: deterministic anti-pollution reads -> stateless agent -> clamp (INV6)."""
    wayback = _wayback_earliest_snapshot(item.url)
    page_time = _page_published_time(item.url)
    payload = {
        "schema": "dateverify.in.v1",
        "url": item.url,
        "title": item.title,
        "claimed_date": item.published_at,
        "tier": tier,
        "wayback_earliest": wayback,
        "page_published_time": page_time,
    }
    agent_out = _run_dateverify_subagent(payload)
    return _clamp_verdict(
        claimed_iso=item.published_at,
        agent_out=agent_out,
        wayback_earliest=wayback,
        page_published_time=page_time,
    )
```

Then, inside the existing `verify(item, store)`, after the Tier-0 attempt and cache check, add (do not remove Tier-0):

```python
def verify(item: HotspotItem, store) -> dict:
    content_hash = _content_hash(item)

    # permanent freeze: a date is verified exactly once per Store (INV3, §B.4)
    cached = store.get_verdict(content_hash)
    if cached is not None:
        return cached

    # --- existing Tier-0 deterministic path stays here (arXiv v1 / Crossref / DOI) ---
    verdict = _verify_tier0(item)   # Stage 1 helper; returns dict or None when no authoritative anchor

    if verdict is None:
        # residual: news/blog/X get the Tier-1 subagent; everything else degrades conservatively
        family = (item.source_type or "").lower()
        if family in _TIER1_FAMILIES:
            verdict = _verify_subagent_residual(item, tier=1)
        else:
            verdict = _clamp_verdict(
                claimed_iso=item.published_at, agent_out=None,
                wayback_earliest=None, page_published_time=None,
            )

    store.put_verdict(content_hash, verdict)   # write-once; no-op if already frozen
    return store.get_verdict(content_hash)      # return the frozen copy (stable across calls)
```

> If Stage 1's `verify()` already implements the Tier-0 body inline rather than via a `_verify_tier0` helper, refactor the Tier-0 block into `_verify_tier0(item) -> dict | None` (returning `None` when no arXiv/DOI/GitHub anchor is available) so the residual branch above can compose. Keep behavior identical for items that DO have an anchor.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_date_verify.py::TestTier1Verify -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add arxiv_assistant/hotspots/date_verify.py tests/test_date_verify.py
git commit -m "feat(hotspot): Tier-1 residual DateVerify dispatch + write-once freeze (B.1/B.2/B.4, INV1/2/3)"
```

---

## Task 6: Tier-2 deep search (uncertain-and-will-be-featured only)

**Files:**
- Modify: `arxiv_assistant/hotspots/date_verify.py`
- Test: `tests/test_date_verify.py`

Tier-2 is a rare escalation (spec §B.2): only when the Tier-1 verdict is low-confidence AND the item is about to be featured. It reuses the same transport with `tier=2` and the deep-search fixture.

- [ ] **Step 1: Write the failing test**

```python
class TestTier2DeepSearch(unittest.TestCase):
    def test_tier2_only_escalates_when_uncertain_and_featured(self):
        from arxiv_assistant.hotspots import date_verify
        deep = _json.loads((FIXTURES / "dateverify_tier2_deep.json").read_text(encoding="utf-8"))
        store = _FakeStore()
        item = _news_item("https://example.com/blog/y", "2026-06-02T09:00:00Z")
        low_conf = {"verified_first_date": "2026-06-02T00:00:00Z", "confidence": 0.4, "evidence": [], "stale_date_pollution": False}
        with patch.object(date_verify, "_wayback_earliest_snapshot", return_value=None), \
             patch.object(date_verify, "_page_published_time", return_value=None), \
             patch.object(date_verify, "_run_dateverify_subagent", side_effect=[low_conf, deep]) as agent:
            verdict = date_verify.verify(item, store, will_be_featured=True)
        # escalated: agent called twice (tier1 then tier2); deep search found an earlier date
        self.assertEqual(agent.call_count, 2)
        self.assertEqual(verdict["verified_first_date"], "2024-09-30T00:00:00Z")
        self.assertTrue(verdict["stale_date_pollution"])

    def test_tier2_not_triggered_when_confident(self):
        from arxiv_assistant.hotspots import date_verify
        store = _FakeStore()
        item = _news_item("https://example.com/blog/z", "2026-06-02T09:00:00Z")
        confident = {"verified_first_date": "2026-06-02T00:00:00Z", "confidence": 0.9, "evidence": [], "stale_date_pollution": False}
        with patch.object(date_verify, "_wayback_earliest_snapshot", return_value="2026-06-02T00:00:00Z"), \
             patch.object(date_verify, "_page_published_time", return_value=None), \
             patch.object(date_verify, "_run_dateverify_subagent", side_effect=[confident]) as agent:
            date_verify.verify(item, store, will_be_featured=True)
        self.assertEqual(agent.call_count, 1)  # no escalation

    def test_tier2_not_triggered_when_not_featured(self):
        from arxiv_assistant.hotspots import date_verify
        store = _FakeStore()
        item = _news_item("https://example.com/blog/q", "2026-06-02T09:00:00Z")
        low_conf = {"verified_first_date": "2026-06-02T00:00:00Z", "confidence": 0.4, "evidence": [], "stale_date_pollution": False}
        with patch.object(date_verify, "_wayback_earliest_snapshot", return_value=None), \
             patch.object(date_verify, "_page_published_time", return_value=None), \
             patch.object(date_verify, "_run_dateverify_subagent", side_effect=[low_conf]) as agent:
            date_verify.verify(item, store, will_be_featured=False)
        self.assertEqual(agent.call_count, 1)  # low conf but not featured -> no Tier-2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_date_verify.py::TestTier2DeepSearch -v`
Expected: FAIL — `verify()` has no `will_be_featured` kwarg / no Tier-2 escalation.

- [ ] **Step 3: Add the `will_be_featured` escalation to `verify()`**

Add a module constant and extend the residual branch. The `verify` signature gains a keyword-only flag with a default that keeps existing callers working:

```python
_TIER2_CONFIDENCE_FLOOR = 0.6   # below this AND will_be_featured -> escalate to deep search
```

Change the residual branch in `verify()` to:

```python
def verify(item: HotspotItem, store, *, will_be_featured: bool = False) -> dict:
    content_hash = _content_hash(item)
    cached = store.get_verdict(content_hash)
    if cached is not None:
        return cached

    verdict = _verify_tier0(item)
    if verdict is None:
        family = (item.source_type or "").lower()
        if family in _TIER1_FAMILIES:
            verdict = _verify_subagent_residual(item, tier=1)
            if will_be_featured and verdict["confidence"] < _TIER2_CONFIDENCE_FLOOR:
                deep = _verify_subagent_residual(item, tier=2)
                # earliest-credible still wins across tiers
                verdict = _clamp_verdict(
                    claimed_iso=item.published_at,
                    agent_out={
                        "verified_first_date": min(
                            d for d in (verdict["verified_first_date"], deep["verified_first_date"]) if d
                        ),
                        "confidence": max(verdict["confidence"], deep["confidence"]),
                        "evidence": list(verdict["evidence"]) + list(deep["evidence"]),
                    },
                    wayback_earliest=None,
                    page_published_time=None,
                )
        else:
            verdict = _clamp_verdict(
                claimed_iso=item.published_at, agent_out=None,
                wayback_earliest=None, page_published_time=None,
            )

    store.put_verdict(content_hash, verdict)
    return store.get_verdict(content_hash)
```

> Note the cross-tier merge re-runs `_clamp_verdict`, so the final verdict is still earliest-min + day-floored (INV2/INV6). The `min(...)` over `verified_first_date` strings is safe because both are already floored day-ISO from each tier's clamp.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_date_verify.py::TestTier2DeepSearch -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add arxiv_assistant/hotspots/date_verify.py tests/test_date_verify.py
git commit -m "feat(hotspot): Tier-2 deep-search escalation for uncertain featured items (B.2)"
```

---

## Task 7: `poll_arxiv_versions` — batched, monotonic, never touches `date_verdicts`

**Files:**
- Modify: `arxiv_assistant/hotspots/date_verify.py`
- Test: `tests/test_date_verify.py`

Independent cheap Tier-0 read (spec §B.4.1): batched arXiv `id_list` query (≤100 ids/call, ~1 req/3s), returns `dict[id, int]`, feeds `store.refresh_arxiv_versions` (monotonic `max`), and NEVER writes `date_verdicts`.

- [ ] **Step 1: Write the failing test**

```python
ARXIV_ATOM = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2311.01234v3</id>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.05678v1</id>
  </entry>
</feed>"""


class TestPollArxivVersions(unittest.TestCase):
    def test_poll_parses_version_counts_from_id_list(self):
        from arxiv_assistant.hotspots import date_verify
        with patch.object(date_verify, "fetch_text", return_value=ARXIV_ATOM) as fetched:
            counts = date_verify.poll_arxiv_versions(["2311.01234", "2401.05678"])
        self.assertEqual(counts, {"2311.01234": 3, "2401.05678": 1})
        # single batched call (id_list), not one-per-id
        self.assertEqual(fetched.call_count, 1)

    def test_poll_deduplicates_ids_within_run(self):
        from arxiv_assistant.hotspots import date_verify
        with patch.object(date_verify, "fetch_text", return_value=ARXIV_ATOM) as fetched:
            date_verify.poll_arxiv_versions(["2311.01234", "2311.01234", "2401.05678"])
        called_url = fetched.call_args[0][0]
        # id_list contains each id once
        self.assertEqual(called_url.count("2311.01234"), 1)

    def test_poll_batches_in_chunks_of_at_most_100(self):
        from arxiv_assistant.hotspots import date_verify
        ids = [f"2401.{i:05d}" for i in range(150)]
        with patch.object(date_verify, "fetch_text", return_value="<feed xmlns='http://www.w3.org/2005/Atom'></feed>") as fetched:
            date_verify.poll_arxiv_versions(ids)
        self.assertEqual(fetched.call_count, 2)  # 150 ids -> 2 batches (100 + 50)

    def test_poll_returns_empty_on_fetch_error_without_crash(self):
        from arxiv_assistant.hotspots import date_verify
        with patch.object(date_verify, "fetch_text", side_effect=RuntimeError("boom")):
            self.assertEqual(date_verify.poll_arxiv_versions(["2311.01234"]), {})


class TestPollMonotonicAndIsolated(unittest.TestCase):
    def test_refresh_is_monotonic_and_does_not_touch_date_verdicts(self):
        # INV3: arxiv_versions monotonic (max), and polling never writes date_verdicts
        from arxiv_assistant.hotspots import date_verify

        class _Store:
            def __init__(self):
                self.versions = {"2311.01234": 3}
                self.put_verdict_calls = 0
            def refresh_arxiv_versions(self, arxiv_id, fetched_count):
                self.versions[arxiv_id] = max(self.versions.get(arxiv_id, 0), fetched_count)
            def put_verdict(self, *a, **k):
                self.put_verdict_calls += 1

        store = _Store()
        # a stale fetch reporting an OLDER count must not lower the stored count
        with patch.object(date_verify, "fetch_text",
                          return_value="<feed xmlns='http://www.w3.org/2005/Atom'>"
                                       "<entry><id>http://arxiv.org/abs/2311.01234v1</id></entry></feed>"):
            counts = date_verify.poll_arxiv_versions(["2311.01234"])
        for aid, n in counts.items():
            store.refresh_arxiv_versions(aid, n)
        self.assertEqual(store.versions["2311.01234"], 3)   # max(3, 1) stays 3
        self.assertEqual(store.put_verdict_calls, 0)         # date_verdicts untouched
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_date_verify.py::TestPollArxivVersions tests/test_date_verify.py::TestPollMonotonicAndIsolated -v`
Expected: FAIL with `AttributeError: poll_arxiv_versions`.

- [ ] **Step 3: Implement `poll_arxiv_versions`**

Add to `date_verify.py`:

```python
import time

_ARXIV_API = "http://export.arxiv.org/api/query"
_ARXIV_BATCH = 100
_ARXIV_REQ_INTERVAL_S = 3.0   # ~1 req/3s polite rate (spec §B.4.1)
_ARXIV_ID_VER = re.compile(r"arxiv\.org/abs/([^v\s<]+)v(\d+)", re.IGNORECASE)


def poll_arxiv_versions(arxiv_ids: list[str]) -> dict[str, int]:
    """Batched arXiv id_list version-count read (§B.4.1).

    <=100 ids/call, ~1 req/3s; returns {bare_id: version_count}. Independent
    cheap deterministic Tier-0 read: the caller feeds store.refresh_arxiv_versions
    (monotonic max). NEVER writes date_verdicts; degrades to {} on fetch error.
    """
    unique = list(dict.fromkeys(i for i in arxiv_ids if i))  # de-dupe, preserve order
    counts: dict[str, int] = {}
    for start in range(0, len(unique), _ARXIV_BATCH):
        if start > 0:
            time.sleep(_ARXIV_REQ_INTERVAL_S)
        batch = unique[start:start + _ARXIV_BATCH]
        url = f"{_ARXIV_API}?id_list={','.join(batch)}&max_results={len(batch)}"
        try:
            atom = fetch_text(url)
        except Exception:
            continue
        for bare_id, ver in _ARXIV_ID_VER.findall(atom):
            n = int(ver)
            counts[bare_id] = max(counts.get(bare_id, 0), n)
    return counts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_date_verify.py::TestPollArxivVersions tests/test_date_verify.py::TestPollMonotonicAndIsolated -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add arxiv_assistant/hotspots/date_verify.py tests/test_date_verify.py
git commit -m "feat(hotspot): batched monotonic poll_arxiv_versions decoupled from date_verdicts (B.4.1, INV3)"
```

---

## Task 8: INV acceptance suite + full-module green gate

**Files:**
- Test: `tests/test_date_verify.py`

A single class that asserts the four owned invariants explicitly as named acceptance tests (overview §4), then a full-file run.

- [ ] **Step 1: Write the acceptance tests**

```python
class TestStage3Invariants(unittest.TestCase):
    def test_INV1_gates_consume_verified_date_not_source_claimed(self):
        # Already covered functionally in TestTier1Verify; assert the contract directly:
        # gate_date is derived from verified_first_date, never item.published_at.
        item = HotspotItem(
            source_id="ainews", source_name="AINews", source_role="roundup", source_type="news",
            title="x", summary="", url="https://e.com/x", canonical_url="https://e.com/x",
            published_at="2026-06-02T09:00:00Z",
        )
        item.verified_first_date = "2023-11-14T00:00:00Z"
        self.assertEqual(gate_date(item), date(2023, 11, 14))
        self.assertNotEqual(gate_date(item), floor_to_utc_day(item.published_at))

    def test_INV2_gate_is_day_granular(self):
        item = HotspotItem(
            source_id="ainews", source_name="AINews", source_role="roundup", source_type="news",
            title="x", summary="", url="https://e.com/y", canonical_url="https://e.com/y",
        )
        item.verified_first_date = "2026-06-02T23:59:59Z"
        self.assertEqual(gate_date(item), date(2026, 6, 2))  # sub-day discarded

    def test_INV3_freeze_once_and_versions_separate(self):
        from arxiv_assistant.hotspots import date_verify
        store = _FakeStore()
        item = _news_item("https://e.com/z", "2026-06-02T09:00:00Z")
        confident = {"verified_first_date": "2026-06-02T00:00:00Z", "confidence": 0.9, "evidence": [], "stale_date_pollution": False}
        with patch.object(date_verify, "_wayback_earliest_snapshot", return_value="2026-06-02T00:00:00Z"), \
             patch.object(date_verify, "_page_published_time", return_value=None), \
             patch.object(date_verify, "_run_dateverify_subagent", return_value=confident):
            date_verify.verify(item, store)
            date_verify.verify(item, store)   # second call frozen
        self.assertEqual(len(store._verdicts), 1)
        # poll path is isolated from the verdict cache
        with patch.object(date_verify, "fetch_text", return_value=ARXIV_ATOM):
            counts = date_verify.poll_arxiv_versions(["2311.01234"])
        self.assertIn("2311.01234", counts)
        self.assertEqual(store.put_calls, 2)  # both verify() calls call put; second is write-once no-op

    def test_INV6_every_agent_followed_by_deterministic_verifier(self):
        # A hostile agent emitting a LATER date than the proven Wayback day must be overridden.
        from arxiv_assistant.hotspots import date_verify
        store = _FakeStore()
        item = _news_item("https://e.com/hostile", "2026-06-02T09:00:00Z")
        hostile = {"verified_first_date": "2026-06-02T00:00:00Z", "confidence": 1.0, "evidence": [], "stale_date_pollution": False}
        with patch.object(date_verify, "_wayback_earliest_snapshot", return_value="2023-11-14T00:00:00Z"), \
             patch.object(date_verify, "_page_published_time", return_value=None), \
             patch.object(date_verify, "_run_dateverify_subagent", return_value=hostile):
            verdict = date_verify.verify(item, store)
        self.assertEqual(verdict["verified_first_date"], "2023-11-14T00:00:00Z")
        self.assertTrue(verdict["stale_date_pollution"])
```

- [ ] **Step 2: Run the acceptance suite**

Run: `pytest tests/test_date_verify.py::TestStage3Invariants -v`
Expected: PASS (4 passed).

- [ ] **Step 3: Run the full module to confirm no regressions**

Run: `pytest tests/test_date_verify.py -v`
Expected: PASS (all Stage 1 + Stage 3 tests green).

- [ ] **Step 4: Confirm no network was hit**

Run: `pytest tests/test_date_verify.py -v -p no:cacheprovider`
Expected: PASS quickly (sub-second per test); every external call (`fetch_text`, `fetch_json`, `subprocess.run`) is patched, so a green run with no socket usage confirms the record/replay discipline (overview §4).

- [ ] **Step 5: Commit**

```bash
git add tests/test_date_verify.py
git commit -m "test(hotspot): Stage 3 INV1/INV2/INV3/INV6 acceptance suite for DateVerify"
```

---

## Self-Review

**1. Spec coverage**

| Spec / contract requirement | Task |
|---|---|
| §B.1 every featured candidate gets `verified_first_date`; suspicion = priority signal not gate | Task 5 (residual dispatch covers all news/blog/X; Tier-0 covers anchored) |
| §B.2 Tier-1 Wayback CDX | Task 3 `_wayback_earliest_snapshot` |
| §B.2 Tier-1 `article:published_time` / JSON-LD | Task 3 `_page_published_time` |
| §B.2 Tier-1 earliest-mention search | Task 4 subagent prompt + Task 5 dispatch |
| §B.2 Tier-2 deep search (uncertain-and-will-be-featured) | Task 6 |
| §B.3 earliest-credible-date-wins + `stale_date_pollution` | Task 2 `_clamp_verdict` |
| §B.3.1 authoritative whole-day anchors into credible set | Task 1 `gate_date.credible_dates` |
| §B.4 permanent cache = freeze | Task 5 `put_verdict` write-once |
| §B.4.1 batched `id_list` ≤100/call, monotonic, not in `date_verdicts` | Task 7 |
| §2.5 `verify(item, store)` signature | Task 5/6 (added keyword-only `will_be_featured` default-safe) |
| §2.5 `poll_arxiv_versions(arxiv_ids) -> dict[str,int]` | Task 7 |
| §2.3 `store.put_verdict/get_verdict/refresh_arxiv_versions` usage | Tasks 5, 7 |
| §4 record/replay fixtures under `tests/fixtures/agent/` | Task 4 |
| INV1 / INV2 / INV3 / INV6 acceptance | Task 8 (+ functional coverage in 2/5/6/7) |

No gaps found. Tier-0 is explicitly **extended, not rewritten** (Task 5 step 3 note refactors the Tier-0 body into `_verify_tier0` only if Stage 1 inlined it).

**2. Placeholder scan:** No `TBD`/`TODO`/"handle edge cases"/"similar to Task N" — every code and test block is complete and self-contained. Fixture JSON is real and schema-conformant.

**3. Type consistency:** `verify(item, store, *, will_be_featured=False)` matches the locked `verify(item, store)` (new kwarg is default-safe per overview §2 "new optional fields default to None/empty"). `_clamp_verdict` keyword args (`claimed_iso`, `agent_out`, `wayback_earliest`, `page_published_time`) are identical across Tasks 2, 5, 6. `poll_arxiv_versions(arxiv_ids: list[str]) -> dict[str, int]` and `store.refresh_arxiv_versions(arxiv_id, fetched_count)` match the contract exactly. `gate_date`/`floor_to_utc_day` imports are consistent across Tasks 1 and 8. Subagent in/out schemas (`dateverify.in.v1` / `dateverify.out.v1`) are consistent between the prompt, fixtures, and transport parser.

---

**Plan complete and saved to `docs/superpowers/plans/2026-06-03-agent-native-rewrite-04-stage3-dateverify.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
