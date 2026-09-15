# Stage 0 — Foundation (Story Store + schema) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement each task below task-by-task. Steps use checkbox (`- [ ]`) syntax. Every code step ships **complete real code** — there are no placeholders.

**Goal:** Lay the persistent-state foundation for the agent-native hotspot rewrite: a single-writer SQLite `StoryStore` (spec §A.1 / contract §2.3), the default-safe new `Story` fields (contract §2.2), the `HotspotItem` `verified_first_date`/`provenance` fields (contract §2.1), and a per-run JSONL `run_journal` (spec §E). No gates, no agents, no dedup logic yet — just the schema + Store API every later stage depends on.

**Architecture:** Pure-stdlib persistence layer. `StoryStore` wraps `sqlite3` with four tables (`stories`, `evidence`, `date_verdicts`, `versions`); the Kernel is the only writer (single-writer invariant). `Story`/`HotspotItem` gain default-`None`/empty fields so old `out/hot/reports/*.json` still parse and existing `group_into_stories`/clustering behavior is untouched. `RunJournal` accumulates per-run records in memory and flushes one JSONL line per run.

**Tech Stack:** Python 3, `sqlite3` (stdlib), `json` (stdlib), `dataclasses`, `pytest` with `unittest.TestCase` style, `tempfile.TemporaryDirectory()` for filesystem isolation. No network, no third-party deps in this stage.

---

## Scope & non-goals

In scope (this stage only):
1. `arxiv_assistant/hotspots/store.py` — `StoryStore` (contract §2.3), tables `stories`/`evidence`/`date_verdicts`/`versions`, DB at `out/hot/state/story_store.sqlite`.
2. `arxiv_assistant/hotspots/story.py` — add contract §2.2 default-safe fields to `Story`. **Do not** touch `group_into_stories` id logic, `_story_id`, `score_stories`, `apply_cross_day_penalty`, or `select_and_categorize`.
3. `arxiv_assistant/utils/hotspot/hotspot_schema.py` — add `verified_first_date` + `provenance` to `HotspotItem` (contract §2.1).
4. `arxiv_assistant/utils/hotspot/run_journal.py` — `RunJournal` with `append`/`flush`, per-run JSONL (per-source counts, stage timings, placeholder `intentionally_dropped_stale_competitor` list).
5. `tests/test_store.py` — covers `put_verdict` write-once freeze, `refresh_arxiv_versions` monotonic, `match_or_create` new-vs-hit, `record_surface` snapshot write, dump/load round-trip including `date_verdicts`.

Explicit non-goals (later stages, do not implement here):
- `gate_date`/`floor_to_utc_day` (stage 1), `resurface`/`resurge` (stage 2), embedding model wiring (stage 2 fills `embed_text`; here `match_or_create` consumes a caller-supplied centroid and a stdlib cosine helper local to the Store).
- DateVerify tiers, `poll_arxiv_versions` (stages 1/3).
- `run_journal` 2nd-order alert thresholds (stage 4) — this stage only ships the `intentionally_dropped_stale_competitor` **placeholder list field**, no threshold logic.
- Backfill script, kernel, renderers.

---

## Task 1 — `HotspotItem` gains `verified_first_date` + `provenance`

Contract §2.1: two optional default-safe fields. `verified_first_date` is ISO8601 or `None` (set later by DateVerify, NEVER source-claimed). `provenance` is a string (e.g. `"native:hf_papers"`).

- [ ] **1a. Write the failing test.** Create `tests/test_store.py` with this first test class (the file grows across later tasks; start it here):

```python
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **1b. Run it; confirm it fails.** Command:
  ```
  pytest tests/test_store.py::TestHotspotItemFields -v
  ```
  Expected: both tests **FAIL** with `TypeError: __init__() got an unexpected keyword argument 'provenance'` (and `verified_first_date`).

- [ ] **1c. Minimal implementation.** Edit `arxiv_assistant/utils/hotspot/hotspot_schema.py`. Replace the `HotspotItem` dataclass body (the field block) so the two new fields are added. The full new dataclass:

```python
@dataclass
class HotspotItem:
    source_id: str
    source_name: str
    source_role: str
    source_type: str
    title: str
    summary: str
    url: str
    canonical_url: str
    published_at: str | None = None
    tags: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    verified_first_date: str | None = None  # ISO8601; set by DateVerify. NEVER source-claimed.
    provenance: str = ""                     # e.g. "native:hf_papers" | "reuse:agents-radar"

    def __post_init__(self) -> None:
        self.title = clean_text(self.title)
        self.summary = clean_text(self.summary)
        self.url = normalize_url(self.url)
        self.canonical_url = normalize_url(self.canonical_url or self.url)
        self.tags = [clean_text(tag) for tag in self.tags if clean_text(tag)]
        self.authors = [clean_text(author) for author in self.authors if clean_text(author)]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
```

- [ ] **1d. Run it; confirm it passes.** Command:
  ```
  pytest tests/test_store.py::TestHotspotItemFields -v
  ```
  Expected: both tests **PASS** (`2 passed`). Then run the existing suite to prove no regression:
  ```
  pytest tests/test_hotspot_pipeline.py -q
  ```
  Expected: all existing tests still **PASS** (new fields are default-safe).

- [ ] **1e. Commit.**
  ```
  git checkout -b stage0-foundation
  git add arxiv_assistant/utils/hotspot/hotspot_schema.py tests/test_store.py
  git commit -m "feat(hotspot): add verified_first_date/provenance to HotspotItem (stage 0)"
  ```

---

## Task 2 — `Story` gains persistent-identity / centroid / snapshot fields

Contract §2.2: all default-safe. We add the fields **only**; we do not change `group_into_stories` (it keeps minting SHA1 ids via `_story_id` — retired in stage 2, not here). Because `Story` is constructed in `group_into_stories` with keyword args for the existing fields, adding new fields with defaults is safe.

- [ ] **2a. Write the failing test.** Append this class to `tests/test_store.py`:

```python
from arxiv_assistant.hotspots.enrich import EnrichedItem
from arxiv_assistant.hotspots.story import Story


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
```

- [ ] **2b. Run it; confirm it fails.** Command:
  ```
  pytest tests/test_store.py::TestStoryFields -v
  ```
  Expected: `test_new_persistent_fields_default_safe` **FAILS** with `AttributeError: 'Story' object has no attribute 'first_seen'` (the other two may also fail at attribute access).

- [ ] **2c. Minimal implementation.** Edit `arxiv_assistant/hotspots/story.py`. Replace the `Story` dataclass field block (lines 65–85, the `@dataclass` through `__post_init__`) with the version below. Keep `__post_init__` exactly as it was. The new fields go **after** the existing ones so `group_into_stories(... Story(story_id=..., canonical_item=..., items=..., event_type=..., entity_names=...))` keeps working unchanged:

```python
@dataclass
class Story:
    story_id: str
    canonical_item: EnrichedItem
    items: list[EnrichedItem]
    event_type: str
    entity_names: set[str] = field(default_factory=set)
    category: str = ""
    score: float = 0.0
    headline: str = ""
    summary: str = ""
    why_it_matters: str = ""
    key_takeaways: list[str] = field(default_factory=list)
    # --- stage 0: persistent identity / centroid / status (contract §2.2) ---
    first_seen: str | None = None              # ISO date, immutable once set
    centroid: list[float] | None = None        # embedding; model_id-bound
    centroid_model_id: str = ""
    status: str = "NEW"                         # "NEW" | "ONGOING"
    arxiv_versions: dict[str, int] = field(default_factory=dict)   # id -> version count (monotonic)
    # --- surface snapshots (recorded by StoryStore.record_surface) ---
    last_surfaced: str | None = None
    surfaced_verified_max: str | None = None   # DAY-granular gate_date
    surfaced_entity_names: set[str] = field(default_factory=set)
    surfaced_max_tier: int = 0
    surfaced_arxiv_versions: dict[str, int] = field(default_factory=dict)
    # --- resurgence (§C.4) ---
    resurged_at: str | None = None             # first-ever resurge run-date (immutable)
    surfaced_resurged_at: str | None = None    # last resurgence-lane surface run-date

    def __post_init__(self):
        if not self.category:
            self.category = EVENT_TYPE_TO_CATEGORY.get(self.event_type, "Industry Update")
        if not self.headline:
            self.headline = self.canonical_item.item.title
        if not self.summary:
            self.summary = self.canonical_item.summary
```

> NOTE (contract §2.2): `story_id` is documented as "NOW persistent (assigned by Store)". In this stage it remains a plain `str` field; the legacy `_story_id` SHA1 minting in `group_into_stories` is **retired in stage 2**, not here. We only add fields.

- [ ] **2d. Run it; confirm it passes.** Command:
  ```
  pytest tests/test_store.py::TestStoryFields -v
  ```
  Expected: all three tests **PASS** (`3 passed`). Regression check:
  ```
  pytest tests/test_hotspot_pipeline.py -q
  ```
  Expected: all existing tests still **PASS**.

- [ ] **2e. Commit.**
  ```
  git add arxiv_assistant/hotspots/story.py tests/test_store.py
  git commit -m "feat(hotspot): add persistent identity/centroid/snapshot fields to Story (stage 0)"
  ```

---

## Task 3 — `StoryStore` SQLite schema + `__init__`, `dump_text_snapshot`, `load_text_snapshot`

Contract §2.3: SQLite at `out/hot/state/story_store.sqlite`; tables `stories`, `evidence`, `date_verdicts`, `versions`; text snapshot dir `out/hot/state/snapshot/`. `dump_text_snapshot` MUST include `date_verdicts` (spec §E / §G.4 — the snapshot carries the frozen verdicts to any rebuild machine). We build the file incrementally; this task ships the constructor, DDL, and snapshot round-trip skeleton.

- [ ] **3a. Write the failing test.** Append this class to `tests/test_store.py`:

```python
from arxiv_assistant.hotspots.store import StoryStore


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
```

- [ ] **3b. Run it; confirm it fails.** Command:
  ```
  pytest tests/test_store.py::TestStoreLifecycle -v
  ```
  Expected: collection/import **FAILS** with `ModuleNotFoundError: No module named 'arxiv_assistant.hotspots.store'`.

- [ ] **3c. Minimal implementation.** Create `arxiv_assistant/hotspots/store.py` with the full module below. This task's tests exercise `__init__`, `_table_names`, `put_verdict`, `get_verdict`, `refresh_arxiv_versions`, `dump_text_snapshot`, `load_text_snapshot`, `close`. The remaining contract methods (`active_stories`, `match_or_create`, `upsert_evidence`, `record_surface`) are also implemented now (so the module is complete) and tested in Tasks 4–6.

```python
from __future__ import annotations

import json
import math
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

from arxiv_assistant.hotspots.enrich import EnrichedItem
from arxiv_assistant.hotspots.story import Story

# ---------------------------------------------------------------------------
# DDL — four tables (contract §2.3). One responsibility: persistence.
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS stories (
    story_id                 TEXT PRIMARY KEY,
    first_seen               TEXT,
    centroid                 TEXT,          -- JSON list[float] or NULL
    centroid_model_id        TEXT NOT NULL DEFAULT '',
    status                   TEXT NOT NULL DEFAULT 'NEW',
    event_type               TEXT NOT NULL DEFAULT '',
    headline                 TEXT NOT NULL DEFAULT '',
    entity_names             TEXT NOT NULL DEFAULT '[]',   -- JSON list[str]
    last_surfaced            TEXT,
    surfaced_verified_max    TEXT,
    surfaced_entity_names    TEXT NOT NULL DEFAULT '[]',   -- JSON list[str]
    surfaced_max_tier        INTEGER NOT NULL DEFAULT 0,
    surfaced_arxiv_versions  TEXT NOT NULL DEFAULT '{}',   -- JSON dict[str,int]
    resurged_at              TEXT,
    surfaced_resurged_at     TEXT,
    updated_at               TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS evidence (
    story_id        TEXT NOT NULL,
    canonical_url   TEXT NOT NULL,
    title           TEXT NOT NULL DEFAULT '',
    source_id       TEXT NOT NULL DEFAULT '',
    source_role     TEXT NOT NULL DEFAULT '',
    provenance      TEXT NOT NULL DEFAULT '',
    added_at        TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (story_id, canonical_url)
);

CREATE TABLE IF NOT EXISTS date_verdicts (
    content_hash        TEXT PRIMARY KEY,
    verified_first_date TEXT,
    confidence          REAL NOT NULL DEFAULT 0.0,
    evidence            TEXT NOT NULL DEFAULT '[]'    -- JSON list[str]
);

CREATE TABLE IF NOT EXISTS versions (
    arxiv_id      TEXT PRIMARY KEY,
    version_count INTEGER NOT NULL DEFAULT 0
);
"""

_TABLES = ("stories", "evidence", "date_verdicts", "versions")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity for two equal-length vectors; 0.0 on degenerate input."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class StoryStore:
    """Single-writer SQLite persistence for the agent-native hotspot pipeline.

    Tables: stories, evidence, date_verdicts, versions.
    Only the Kernel constructs and writes a StoryStore (single-writer invariant).
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- low-level helpers ---------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def _table_names(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        return [r["name"] for r in rows]

    @staticmethod
    def _story_entity_names(story: Story) -> list[str]:
        return sorted(story.entity_names)

    @staticmethod
    def _row_to_story(row: sqlite3.Row, evidence_rows: list[sqlite3.Row]) -> Story:
        """Reconstruct a lightweight Story shell from persisted rows.

        Only fields the Store owns are restored; canonical_item/items are left as
        thin placeholders because cross-day matching consumes centroid + snapshot
        fields, not the full enriched payload.
        """
        centroid_raw = row["centroid"]
        centroid = json.loads(centroid_raw) if centroid_raw else None
        story = Story(
            story_id=row["story_id"],
            canonical_item=None,  # type: ignore[arg-type]
            items=[],
            event_type=row["event_type"],
            entity_names=set(json.loads(row["entity_names"])),
            headline=row["headline"],
            summary=" ",  # non-empty so __post_init__ does not dereference canonical_item
            first_seen=row["first_seen"],
            centroid=centroid,
            centroid_model_id=row["centroid_model_id"],
            status=row["status"],
            arxiv_versions={r["arxiv_id"]: r["version_count"] for r in evidence_rows} if False else {},
            last_surfaced=row["last_surfaced"],
            surfaced_verified_max=row["surfaced_verified_max"],
            surfaced_entity_names=set(json.loads(row["surfaced_entity_names"])),
            surfaced_max_tier=row["surfaced_max_tier"],
            surfaced_arxiv_versions=json.loads(row["surfaced_arxiv_versions"]),
            resurged_at=row["resurged_at"],
            surfaced_resurged_at=row["surfaced_resurged_at"],
        )
        return story

    def _persist_story(self, story: Story) -> None:
        self._conn.execute(
            """
            INSERT INTO stories (
                story_id, first_seen, centroid, centroid_model_id, status,
                event_type, headline, entity_names, last_surfaced,
                surfaced_verified_max, surfaced_entity_names, surfaced_max_tier,
                surfaced_arxiv_versions, resurged_at, surfaced_resurged_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(story_id) DO UPDATE SET
                first_seen=excluded.first_seen,
                centroid=excluded.centroid,
                centroid_model_id=excluded.centroid_model_id,
                status=excluded.status,
                event_type=excluded.event_type,
                headline=excluded.headline,
                entity_names=excluded.entity_names,
                last_surfaced=excluded.last_surfaced,
                surfaced_verified_max=excluded.surfaced_verified_max,
                surfaced_entity_names=excluded.surfaced_entity_names,
                surfaced_max_tier=excluded.surfaced_max_tier,
                surfaced_arxiv_versions=excluded.surfaced_arxiv_versions,
                resurged_at=excluded.resurged_at,
                surfaced_resurged_at=excluded.surfaced_resurged_at,
                updated_at=excluded.updated_at
            """,
            (
                story.story_id,
                story.first_seen,
                json.dumps(story.centroid) if story.centroid is not None else None,
                story.centroid_model_id,
                story.status,
                story.event_type,
                story.headline,
                json.dumps(self._story_entity_names(story)),
                story.last_surfaced,
                story.surfaced_verified_max,
                json.dumps(sorted(story.surfaced_entity_names)),
                story.surfaced_max_tier,
                json.dumps(story.surfaced_arxiv_versions),
                story.resurged_at,
                story.surfaced_resurged_at,
                _now_iso(),
            ),
        )
        self._conn.commit()

    # -- identity / dedup ----------------------------------------------------

    def active_stories(self, window_days: int, as_of: date) -> list[Story]:
        """All stories whose first_seen falls within the rolling window ending as_of."""
        rows = self._conn.execute("SELECT * FROM stories").fetchall()
        result: list[Story] = []
        for row in rows:
            fs = row["first_seen"]
            if fs is None:
                continue
            try:
                fs_day = datetime.fromisoformat(fs).date()
            except ValueError:
                fs_day = date.fromisoformat(fs[:10])
            age = (as_of - fs_day).days
            if 0 <= age <= window_days:
                result.append(self._row_to_story(row, []))
        return result

    def match_or_create(
        self,
        cluster_centroid: list[float],
        cluster: Story,
        cosine_threshold: float,
        window_days: int,
        as_of: date,
    ) -> tuple[Story, bool]:
        """Match the day's cluster to an active persistent story by centroid cosine.

        Returns (story, is_new). On match: existing story marked ONGOING, persistent
        story_id + first_seen preserved. On miss: cluster persisted as a new story.
        """
        best: Story | None = None
        best_sim = -1.0
        for candidate in self.active_stories(window_days, as_of):
            if not candidate.centroid:
                continue
            sim = cosine(cluster_centroid, candidate.centroid)
            if sim >= cosine_threshold and sim > best_sim:
                best_sim = sim
                best = candidate

        if best is not None:
            best.status = "ONGOING"
            best.entity_names = best.entity_names | cluster.entity_names
            self._persist_story(best)
            return best, False

        cluster.centroid = cluster_centroid
        cluster.status = "NEW"
        if cluster.first_seen is None:
            cluster.first_seen = as_of.isoformat()
        self._persist_story(cluster)
        return cluster, True

    def upsert_evidence(self, story_id: str, items: list[EnrichedItem], added_at: str) -> None:
        """Insert/refresh evidence rows for a story; (story_id, canonical_url) is the key."""
        for ei in items:
            item = ei.item
            self._conn.execute(
                """
                INSERT INTO evidence (story_id, canonical_url, title, source_id, source_role, provenance, added_at)
                VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(story_id, canonical_url) DO UPDATE SET
                    title=excluded.title,
                    source_id=excluded.source_id,
                    source_role=excluded.source_role,
                    provenance=excluded.provenance
                """,
                (
                    story_id,
                    item.canonical_url,
                    item.title,
                    item.source_id,
                    item.source_role,
                    getattr(item, "provenance", ""),
                    added_at,
                ),
            )
        self._conn.commit()

    def record_surface(self, story: Story, run_date: str, *, lane: str = "featured") -> None:
        """Freeze the story's surface-state snapshot at run_date.

        Writes surfaced_* fields. lane="resurgence" also sets surfaced_resurged_at
        and, on first-ever resurge, the immutable resurged_at.
        """
        story.last_surfaced = run_date
        story.surfaced_verified_max = story.surfaced_verified_max  # set by caller pre-call when known
        story.surfaced_entity_names = set(story.entity_names)
        story.surfaced_arxiv_versions = dict(story.arxiv_versions)
        if lane == "resurgence":
            if story.resurged_at is None:
                story.resurged_at = run_date
            story.surfaced_resurged_at = run_date
        self._persist_story(story)

    # -- date verdict cache (permanent freeze) -------------------------------

    def get_verdict(self, content_hash: str) -> dict | None:
        row = self._conn.execute(
            "SELECT verified_first_date, confidence, evidence FROM date_verdicts WHERE content_hash=?",
            (content_hash,),
        ).fetchone()
        if row is None:
            return None
        return {
            "verified_first_date": row["verified_first_date"],
            "confidence": row["confidence"],
            "evidence": json.loads(row["evidence"]),
        }

    def put_verdict(self, content_hash: str, verdict: dict) -> None:
        """Write-once. No-op if a verdict for content_hash already exists (freeze)."""
        existing = self._conn.execute(
            "SELECT 1 FROM date_verdicts WHERE content_hash=?", (content_hash,)
        ).fetchone()
        if existing is not None:
            return
        self._conn.execute(
            "INSERT INTO date_verdicts (content_hash, verified_first_date, confidence, evidence) VALUES (?,?,?,?)",
            (
                content_hash,
                verdict.get("verified_first_date"),
                float(verdict.get("confidence", 0.0)),
                json.dumps(verdict.get("evidence", [])),
            ),
        )
        self._conn.commit()

    # -- version counts (monotonic, NOT in date_verdicts) --------------------

    def refresh_arxiv_versions(self, arxiv_id: str, fetched_count: int) -> None:
        """new := max(old, fetched_count). Monotonic non-decreasing; never touches date_verdicts."""
        row = self._conn.execute(
            "SELECT version_count FROM versions WHERE arxiv_id=?", (arxiv_id,)
        ).fetchone()
        old = row["version_count"] if row is not None else 0
        new = max(old, int(fetched_count))
        self._conn.execute(
            """
            INSERT INTO versions (arxiv_id, version_count) VALUES (?,?)
            ON CONFLICT(arxiv_id) DO UPDATE SET version_count=excluded.version_count
            """,
            (arxiv_id, new),
        )
        self._conn.commit()

    def get_arxiv_version(self, arxiv_id: str) -> int:
        row = self._conn.execute(
            "SELECT version_count FROM versions WHERE arxiv_id=?", (arxiv_id,)
        ).fetchone()
        return row["version_count"] if row is not None else 0

    # -- audit snapshot ------------------------------------------------------

    def dump_text_snapshot(self, out_dir: Path) -> Path:
        """Dump all four tables to one schema-tagged JSON file. MUST include date_verdicts."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        blob: dict[str, object] = {"schema_version": 1, "dumped_at": _now_iso()}
        for table in _TABLES:
            rows = self._conn.execute(f"SELECT * FROM {table}").fetchall()
            blob[table] = [dict(r) for r in rows]
        snapshot = out_dir / "story_store_snapshot.json"
        snapshot.write_text(json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8")
        return snapshot

    def load_text_snapshot(self, snapshot: Path) -> None:
        """Rebuild all four tables from a text snapshot, inheriting frozen verdicts."""
        blob = json.loads(Path(snapshot).read_text(encoding="utf-8"))
        for table in _TABLES:
            self._conn.execute(f"DELETE FROM {table}")
            for row in blob.get(table, []):
                cols = list(row.keys())
                placeholders = ",".join("?" for _ in cols)
                col_list = ",".join(cols)
                self._conn.execute(
                    f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})",
                    tuple(row[c] for c in cols),
                )
        self._conn.commit()
```

> Implementation note on `_row_to_story`: it passes `canonical_item=None` and `summary=" "`. `Story.__post_init__` only dereferences `canonical_item` when `headline`/`summary` are empty; the persisted `headline` is non-empty for real stories and `summary` is a non-empty sentinel, so `__post_init__` never touches the `None`. The `arxiv_versions` of a reconstructed story is intentionally rebuilt from the `versions` table by the caller (stage 2 populates it); here it defaults to `{}` since cross-day matching only needs the centroid. The `False`-guarded comprehension is removed in the next edit step.

- [ ] **3d. Fix the leftover dead branch.** The `_row_to_story` body above contains `... if False else {}` as a deliberately-inert placeholder so the code is valid in isolation; replace it with the clean form. Edit `arxiv_assistant/hotspots/store.py`:

  Replace:
  ```python
            arxiv_versions={r["arxiv_id"]: r["version_count"] for r in evidence_rows} if False else {},
  ```
  with:
  ```python
            arxiv_versions={},
  ```

- [ ] **3e. Run it; confirm it passes.** Command:
  ```
  pytest tests/test_store.py::TestStoreLifecycle -v
  ```
  Expected: both tests **PASS** (`2 passed`).

- [ ] **3f. Commit.**
  ```
  git add arxiv_assistant/hotspots/store.py tests/test_store.py
  git commit -m "feat(hotspot): add StoryStore SQLite schema + verdict cache + snapshot round-trip (stage 0)"
  ```

---

## Task 4 — `put_verdict` write-once freeze + `refresh_arxiv_versions` monotonic (INV3)

These are the §G.4 / contract §2.3 invariants made executable. `put_verdict` must no-op on a second write for the same `content_hash` (first-date is frozen for life). `refresh_arxiv_versions` must be monotonic non-decreasing and must never write `date_verdicts`.

- [ ] **4a. Write the failing test.** Append to `tests/test_store.py`:

```python
class TestVerdictAndVersions(unittest.TestCase):
    def _store(self, tmp: str) -> StoryStore:
        return StoryStore(Path(tmp) / "state" / "story_store.sqlite")

    def test_put_verdict_writes_once_then_freezes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            store.put_verdict(
                "hash-1",
                {"verified_first_date": "2026-05-20T00:00:00+00:00", "confidence": 0.95, "evidence": ["arxiv"]},
            )
            # second write with a LATER date must be ignored (freeze)
            store.put_verdict(
                "hash-1",
                {"verified_first_date": "2026-06-01T00:00:00+00:00", "confidence": 0.10, "evidence": ["websearch"]},
            )
            v = store.get_verdict("hash-1")
            self.assertEqual(v["verified_first_date"], "2026-05-20T00:00:00+00:00")
            self.assertEqual(v["confidence"], 0.95)
            self.assertEqual(v["evidence"], ["arxiv"])
            store.close()

    def test_get_verdict_missing_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            self.assertIsNone(store.get_verdict("nope"))
            store.close()

    def test_refresh_arxiv_versions_is_monotonic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            store.refresh_arxiv_versions("2606.00001", 1)
            self.assertEqual(store.get_arxiv_version("2606.00001"), 1)
            store.refresh_arxiv_versions("2606.00001", 3)
            self.assertEqual(store.get_arxiv_version("2606.00001"), 3)
            # a lower fetched count must NOT decrease the stored count
            store.refresh_arxiv_versions("2606.00001", 2)
            self.assertEqual(store.get_arxiv_version("2606.00001"), 3)
            store.close()

    def test_version_refresh_never_touches_date_verdicts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            store.refresh_arxiv_versions("2606.00001", 5)
            # no verdict was written by a version refresh
            self.assertIsNone(store.get_verdict("2606.00001"))
            self.assertIsNone(store.get_verdict("hash:2606.00001"))
            store.close()
```

- [ ] **4b. Run it; confirm it passes** (the Task 3 implementation already satisfies these — this task locks the invariants with explicit tests). Command:
  ```
  pytest tests/test_store.py::TestVerdictAndVersions -v
  ```
  Expected: all four tests **PASS** (`4 passed`). If any fails, the regression is in `put_verdict`/`refresh_arxiv_versions` from Task 3 — fix there, do not weaken the test.

- [ ] **4c. Commit.**
  ```
  git add tests/test_store.py
  git commit -m "test(hotspot): lock put_verdict freeze + arxiv_versions monotonicity (stage 0, INV3)"
  ```

---

## Task 5 — `match_or_create` new-vs-hit + `upsert_evidence`

Contract §2.3: `match_or_create` returns `(story, is_new)`. A first cluster with no active match → `is_new=True`, persisted as `NEW` with `first_seen=as_of`. A later cluster whose centroid cosine ≥ threshold against an active story → `is_new=False`, the matched story flipped to `ONGOING` with its original `story_id`/`first_seen` preserved.

- [ ] **5a. Write the failing test.** Append to `tests/test_store.py`:

```python
class TestMatchOrCreate(unittest.TestCase):
    def _store(self, tmp: str) -> StoryStore:
        return StoryStore(Path(tmp) / "state" / "story_store.sqlite")

    def test_first_cluster_creates_new_story(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            story = _make_story("s-new")
            result, is_new = store.match_or_create(
                cluster_centroid=[1.0, 0.0, 0.0],
                cluster=story,
                cosine_threshold=0.90,
                window_days=14,
                as_of=date(2026, 6, 1),
            )
            self.assertTrue(is_new)
            self.assertEqual(result.status, "NEW")
            self.assertEqual(result.first_seen, "2026-06-01")
            self.assertEqual(result.centroid, [1.0, 0.0, 0.0])
            store.close()

    def test_similar_cluster_matches_existing_and_marks_ongoing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            first = _make_story("s-anchor")
            store.match_or_create([1.0, 0.0, 0.0], first, 0.90, 14, date(2026, 6, 1))

            # next day, a near-identical centroid (cosine 1.0) for a fresh cluster
            second = _make_story("s-day2")
            result, is_new = store.match_or_create(
                cluster_centroid=[1.0, 0.0, 0.0],
                cluster=second,
                cosine_threshold=0.90,
                window_days=14,
                as_of=date(2026, 6, 2),
            )
            self.assertFalse(is_new)
            self.assertEqual(result.story_id, "s-anchor")           # persistent id preserved
            self.assertEqual(result.first_seen, "2026-06-01")        # immutable anchor preserved
            self.assertEqual(result.status, "ONGOING")
            store.close()

    def test_dissimilar_cluster_creates_separate_story(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            store.match_or_create([1.0, 0.0, 0.0], _make_story("s-a"), 0.90, 14, date(2026, 6, 1))
            result, is_new = store.match_or_create(
                cluster_centroid=[0.0, 1.0, 0.0],  # orthogonal -> cosine 0
                cluster=_make_story("s-b"),
                cosine_threshold=0.90,
                window_days=14,
                as_of=date(2026, 6, 2),
            )
            self.assertTrue(is_new)
            self.assertEqual(result.story_id, "s-b")
            store.close()

    def test_match_ignores_stories_outside_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            store.match_or_create([1.0, 0.0, 0.0], _make_story("s-old"), 0.90, 14, date(2026, 6, 1))
            # 20 days later, the anchor is outside the 14-day window -> new story
            result, is_new = store.match_or_create(
                cluster_centroid=[1.0, 0.0, 0.0],
                cluster=_make_story("s-fresh"),
                cosine_threshold=0.90,
                window_days=14,
                as_of=date(2026, 6, 21),
            )
            self.assertTrue(is_new)
            self.assertEqual(result.story_id, "s-fresh")
            store.close()

    def test_upsert_evidence_is_idempotent_on_url_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            store.match_or_create([1.0, 0.0, 0.0], _make_story("s-ev"), 0.90, 14, date(2026, 6, 1))
            ei = _make_enriched(_make_item(url="https://arxiv.org/abs/2606.00001"))
            store.upsert_evidence("s-ev", [ei], added_at="2026-06-01")
            store.upsert_evidence("s-ev", [ei], added_at="2026-06-02")  # same url -> no duplicate row
            count = store._conn.execute(
                "SELECT COUNT(*) AS n FROM evidence WHERE story_id=?", ("s-ev",)
            ).fetchone()["n"]
            self.assertEqual(count, 1)
            store.close()
```

- [ ] **5b. Run it; confirm it passes** (Task 3 already implements both methods; this task locks behavior). Command:
  ```
  pytest tests/test_store.py::TestMatchOrCreate -v
  ```
  Expected: all five tests **PASS** (`5 passed`). If `test_similar_cluster_matches_existing_and_marks_ongoing` fails on `result.story_id`, verify `_row_to_story`/`active_stories` reload path; do not relax the assertion.

- [ ] **5c. Commit.**
  ```
  git add tests/test_store.py
  git commit -m "test(hotspot): lock match_or_create new-vs-hit + evidence idempotency (stage 0)"
  ```

---

## Task 6 — `record_surface` writes snapshot fields (featured + resurgence lanes)

Contract §2.3: `record_surface` writes the `surfaced_*` snapshots that the stage-2 `resurface(S)` and stage-4 `resurge(S)` closed-form predicates read. `lane="resurgence"` additionally sets `surfaced_resurged_at` every call and `resurged_at` once (immutable).

- [ ] **6a. Write the failing test.** Append to `tests/test_store.py`:

```python
class TestRecordSurface(unittest.TestCase):
    def _store(self, tmp: str) -> StoryStore:
        return StoryStore(Path(tmp) / "state" / "story_store.sqlite")

    def test_featured_surface_snapshots_entities_and_versions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            story = _make_story("s-surf")
            store.match_or_create([1.0, 0.0, 0.0], story, 0.90, 14, date(2026, 6, 1))
            story.entity_names = {"openai", "anthropic"}
            story.arxiv_versions = {"2606.00001": 2}
            store.record_surface(story, run_date="2026-06-01", lane="featured")

            reloaded = store.active_stories(14, date(2026, 6, 1))[0]
            self.assertEqual(reloaded.last_surfaced, "2026-06-01")
            self.assertEqual(reloaded.surfaced_entity_names, {"openai", "anthropic"})
            self.assertEqual(reloaded.surfaced_arxiv_versions, {"2606.00001": 2})
            self.assertIsNone(reloaded.resurged_at)
            self.assertIsNone(reloaded.surfaced_resurged_at)
            store.close()

    def test_resurgence_lane_sets_resurged_at_once_and_surfaced_each_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(tmp)
            story = _make_story("s-res")
            store.match_or_create([1.0, 0.0, 0.0], story, 0.90, 14, date(2026, 5, 1))

            store.record_surface(story, run_date="2026-06-01", lane="resurgence")
            self.assertEqual(story.resurged_at, "2026-06-01")
            self.assertEqual(story.surfaced_resurged_at, "2026-06-01")

            # second resurgence surface: resurged_at frozen, surfaced_resurged_at advances
            store.record_surface(story, run_date="2026-06-08", lane="resurgence")
            self.assertEqual(story.resurged_at, "2026-06-01")          # immutable
            self.assertEqual(story.surfaced_resurged_at, "2026-06-08")  # advanced

            reloaded = [s for s in store.active_stories(60, date(2026, 6, 8)) if s.story_id == "s-res"][0]
            self.assertEqual(reloaded.resurged_at, "2026-06-01")
            self.assertEqual(reloaded.surfaced_resurged_at, "2026-06-08")
            store.close()
```

- [ ] **6b. Run it; confirm it passes** (Task 3 implements `record_surface`; this task locks the lane semantics). Command:
  ```
  pytest tests/test_store.py::TestRecordSurface -v
  ```
  Expected: both tests **PASS** (`2 passed`).

- [ ] **6c. Commit.**
  ```
  git add tests/test_store.py
  git commit -m "test(hotspot): lock record_surface featured/resurgence snapshot semantics (stage 0)"
  ```

---

## Task 7 — `RunJournal` per-run JSONL (spec §E)

Spec §E: each run writes a `run_journal` JSON with per-source counts, stage timings, and the `intentionally_dropped_stale_competitor` list. This stage ships the data container + append/flush interface only — the 2nd-order alert **thresholds** are stage 4 (we ship the placeholder list field, no threshold logic). One JSONL line is appended per `flush()` so multiple runs accumulate in one file.

- [ ] **7a. Write the failing test.** Create `tests/test_run_journal.py`:

```python
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from arxiv_assistant.utils.hotspot.run_journal import RunJournal


class TestRunJournal(unittest.TestCase):
    def test_append_and_flush_writes_one_jsonl_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            j = RunJournal(run_date="2026-06-03", journal_path=path)
            j.append("source_counts", {"hf_papers": 12, "ainews": 4})
            j.append_stage_timing("harvest", 1.5)
            j.append_stage_timing("embed", 0.25)
            j.record_dropped_stale_competitor(
                {"source_id": "agents_radar", "gate_date": "2023-01-01", "reason": "stale_curated"}
            )
            out = j.flush()
            self.assertEqual(out, path)

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            rec = json.loads(lines[0])
            self.assertEqual(rec["run_date"], "2026-06-03")
            self.assertEqual(rec["source_counts"], {"hf_papers": 12, "ainews": 4})
            self.assertEqual(rec["stage_timings"], {"harvest": 1.5, "embed": 0.25})
            self.assertEqual(len(rec["intentionally_dropped_stale_competitor"]), 1)
            self.assertEqual(
                rec["intentionally_dropped_stale_competitor"][0]["source_id"], "agents_radar"
            )

    def test_second_run_appends_a_new_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            RunJournal(run_date="2026-06-01", journal_path=path).flush()
            RunJournal(run_date="2026-06-02", journal_path=path).flush()
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["run_date"], "2026-06-01")
            self.assertEqual(json.loads(lines[1])["run_date"], "2026-06-02")

    def test_empty_journal_flushes_default_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            RunJournal(run_date="2026-06-03", journal_path=path).flush()
            rec = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(rec["source_counts"], {})
            self.assertEqual(rec["stage_timings"], {})
            self.assertEqual(rec["intentionally_dropped_stale_competitor"], [])
            self.assertEqual(rec["extra"], {})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **7b. Run it; confirm it fails.** Command:
  ```
  pytest tests/test_run_journal.py -v
  ```
  Expected: collection **FAILS** with `ModuleNotFoundError: No module named 'arxiv_assistant.utils.hotspot.run_journal'`.

- [ ] **7c. Minimal implementation.** Create `arxiv_assistant/utils/hotspot/run_journal.py`:

```python
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Default per-run journal location (spec §E). Callers may override.
DEFAULT_JOURNAL_PATH = Path("out/hot/state/run_journal.jsonl")


class RunJournal:
    """Accumulates one run's observability record and flushes it as one JSONL line.

    Stage 0 ships the container + append/flush only. The 2nd-order pollution
    alert thresholds over `intentionally_dropped_stale_competitor` are stage 4;
    here that field is a plain placeholder list with no threshold logic.
    """

    def __init__(self, run_date: str, journal_path: Path | None = None):
        self.run_date = run_date
        self.journal_path = Path(journal_path) if journal_path is not None else DEFAULT_JOURNAL_PATH
        self.source_counts: dict[str, int] = {}
        self.stage_timings: dict[str, float] = {}
        self.intentionally_dropped_stale_competitor: list[dict[str, Any]] = []
        self.extra: dict[str, Any] = {}

    def append(self, key: str, value: Any) -> None:
        """Generic append. `source_counts`/`stage_timings` merge dicts; else goes to `extra`."""
        if key == "source_counts" and isinstance(value, dict):
            self.source_counts.update(value)
        elif key == "stage_timings" and isinstance(value, dict):
            self.stage_timings.update(value)
        else:
            self.extra[key] = value

    def append_stage_timing(self, stage: str, seconds: float) -> None:
        self.stage_timings[stage] = float(seconds)

    def record_dropped_stale_competitor(self, entry: dict[str, Any]) -> None:
        """Append one intentionally-dropped stale competitor item (spec §D.3 / §E)."""
        self.intentionally_dropped_stale_competitor.append(dict(entry))

    def to_record(self) -> dict[str, Any]:
        return {
            "run_date": self.run_date,
            "flushed_at": datetime.now(timezone.utc).isoformat(),
            "source_counts": dict(self.source_counts),
            "stage_timings": dict(self.stage_timings),
            "intentionally_dropped_stale_competitor": list(
                self.intentionally_dropped_stale_competitor
            ),
            "extra": dict(self.extra),
        }

    def flush(self) -> Path:
        """Append this run's record as one JSON line; create parent dirs as needed."""
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(self.to_record(), ensure_ascii=False)
        with self.journal_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return self.journal_path
```

- [ ] **7d. Run it; confirm it passes.** Command:
  ```
  pytest tests/test_run_journal.py -v
  ```
  Expected: all three tests **PASS** (`3 passed`).

- [ ] **7e. Commit.**
  ```
  git add arxiv_assistant/utils/hotspot/run_journal.py tests/test_run_journal.py
  git commit -m "feat(hotspot): add per-run RunJournal JSONL (stage 0, spec E)"
  ```

---

## Task 8 — Full-suite green + stage acceptance

- [ ] **8a. Run the whole new + touched test surface.** Command:
  ```
  pytest tests/test_store.py tests/test_run_journal.py tests/test_hotspot_pipeline.py -v
  ```
  Expected: every test **PASS**. New-field defaults keep the legacy pipeline suite green (backward-compat rail, overview §5).

- [ ] **8b. Assert the stage-0 §G invariant slice.** Confirm by inspection + the tests above:
  - INV3 (verdict freeze): `TestVerdictAndVersions.test_put_verdict_writes_once_then_freezes` green; `refresh_arxiv_versions` monotonic and never writes `date_verdicts` (`test_version_refresh_never_touches_date_verdicts`).
  - Snapshot completeness (spec §E / §G.4): `TestStoreLifecycle.test_dump_and_load_round_trip_includes_date_verdicts` green — `date_verdicts` travels with the snapshot.
  - Single-writer + default-safe schema: `Story`/`HotspotItem` new fields default to `None`/empty; `pytest tests/test_hotspot_pipeline.py` green proves no existing behavior broke.

- [ ] **8c. Final commit (docs/state, if any).** No production-code change is expected here; this is the stage gate. If `git status` is clean, skip. Otherwise:
  ```
  git add -A
  git commit -m "chore(hotspot): stage 0 foundation acceptance gate green"
  ```

---

## Spec ↔ task coverage map (for the reviewer)

| Spec / contract section | Delivered by | Test |
|---|---|---|
| Contract §2.1 (`HotspotItem.verified_first_date`/`provenance`) | Task 1 | `TestHotspotItemFields` |
| Contract §2.2 (`Story` persistent/centroid/snapshot/resurge fields) | Task 2 | `TestStoryFields` |
| Contract §2.3 (`StoryStore` ctor + 4 tables + snapshot) | Task 3 | `TestStoreLifecycle` |
| Spec §C.3.1 store schema fields (`Story.surfaced_*`, `arxiv_versions`, evidence `added_at`) | Tasks 2, 3 (`stories`/`evidence` DDL) | `TestStoryFields`, `TestRecordSurface` |
| Spec §B.4 `date_verdicts` table + freeze | Task 3, Task 4 | `TestVerdictAndVersions` |
| Contract §2.3 `refresh_arxiv_versions` monotonic / `versions` table | Tasks 3, 4 | `TestVerdictAndVersions` |
| Contract §2.3 `match_or_create` / `upsert_evidence` / `active_stories` | Tasks 3, 5 | `TestMatchOrCreate` |
| Contract §2.3 `record_surface` (featured + resurgence lanes) | Tasks 3, 6 | `TestRecordSurface` |
| Contract §2.3 `dump_text_snapshot` MUST include `date_verdicts` (spec §E / §G.4) | Task 3 | `TestStoreLifecycle` |
| Spec §E `run_journal` (per-source counts, stage timings, dropped-stale list) | Task 7 | `TestRunJournal` |
| Overview §4 test conventions (`unittest.TestCase`, `TemporaryDirectory`, no network) | all tasks | — |
| Overview §5 backward-compat (default-safe fields, legacy reports parse) | Tasks 1, 2, 8 | `test_hotspot_pipeline.py` green |
