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
    story_id              TEXT NOT NULL,
    canonical_url         TEXT NOT NULL,
    title                 TEXT NOT NULL DEFAULT '',
    source_id             TEXT NOT NULL DEFAULT '',
    source_role           TEXT NOT NULL DEFAULT '',
    provenance            TEXT NOT NULL DEFAULT '',
    source_tier           INTEGER NOT NULL DEFAULT 0,   -- §6 item 1: tier int from source_tiers.json
    added_at              TEXT NOT NULL DEFAULT '',
    verified_first_date   TEXT,                         -- day-granular gate date for T2 novelty
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

# ---------------------------------------------------------------------------
# Module-level helper (contract §6 item 1 + §2.3)
# ---------------------------------------------------------------------------

def _open_story_store(output_root: Path) -> "StoryStore":
    """Centralise DB-path construction for kernel, backfill, and tests."""
    db_path = Path(output_root) / "hot" / "state" / "story_store.sqlite"
    return StoryStore(db_path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Cosine helper (private; stage2: replace with embed.cosine)
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity for two equal-length vectors; 0.0 on degenerate input.

    stage2: replace with from arxiv_assistant.hotspots.embed import cosine
    """
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
    def _cosine(a: list[float], b: list[float]) -> float:
        """Private cosine similarity wrapper.

        stage2: replace with from arxiv_assistant.hotspots.embed import cosine
        """
        return _cosine(a, b)

    @staticmethod
    def _story_entity_names(story: Story) -> list[str]:
        return sorted(story.entity_names)

    def _load_evidence_ledger(self, story_id: str) -> list[dict]:
        """Load evidence rows for a story, shaped for Story.evidence_ledger.

        Contract §6 item 2: rows have keys canonical_url, source_id, source_role,
        provenance, source_tier (int), added_at (str), verified_first_date (str|None).
        """
        rows = self._conn.execute(
            """
            SELECT canonical_url, source_id, source_role, provenance,
                   source_tier, added_at, verified_first_date
            FROM evidence
            WHERE story_id = ?
            """,
            (story_id,),
        ).fetchall()
        return [
            {
                "canonical_url": r["canonical_url"],
                "source_id": r["source_id"],
                "source_role": r["source_role"],
                "provenance": r["provenance"],
                "source_tier": int(r["source_tier"]),
                "added_at": r["added_at"],
                "verified_first_date": r["verified_first_date"],
            }
            for r in rows
        ]

    def _row_to_story(self, row: sqlite3.Row) -> Story:
        """Reconstruct a lightweight Story shell from persisted rows.

        Only fields the Store owns are restored; canonical_item/items are left as
        thin placeholders because cross-day matching consumes centroid + snapshot
        fields, not the full enriched payload.

        §6 item 2: populates evidence_ledger so NoveltyGate can read it.
        """
        centroid_raw = row["centroid"]
        centroid = json.loads(centroid_raw) if centroid_raw else None

        # Load evidence ledger for this story (§6 item 2)
        evidence_ledger = self._load_evidence_ledger(row["story_id"])

        story = Story(
            story_id=row["story_id"],
            canonical_item=None,  # type: ignore[arg-type]
            items=[],
            event_type=row["event_type"],
            entity_names=set(json.loads(row["entity_names"])),
            headline=row["headline"] or " ",  # non-empty sentinel: __post_init__ must not deref canonical_item=None
            summary=" ",  # non-empty sentinel so __post_init__ does not dereference canonical_item
            first_seen=row["first_seen"],
            centroid=centroid,
            centroid_model_id=row["centroid_model_id"],
            status=row["status"],
            arxiv_versions={},  # rebuilt from versions table by caller (stage 2)
            last_surfaced=row["last_surfaced"],
            surfaced_verified_max=row["surfaced_verified_max"],
            surfaced_entity_names=set(json.loads(row["surfaced_entity_names"])),
            surfaced_max_tier=row["surfaced_max_tier"],
            surfaced_arxiv_versions=json.loads(row["surfaced_arxiv_versions"]),
            resurged_at=row["resurged_at"],
            surfaced_resurged_at=row["surfaced_resurged_at"],
            evidence_ledger=evidence_ledger,
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
        # stage0: full scan is fine at daily volumes; stage2 may add a WHERE on first_seen.
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
                result.append(self._row_to_story(row))
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

        §6 item 2: _cosine is a private helper; stage2 replaces it with embed.cosine.
        """
        best: Story | None = None
        best_sim = -1.0
        for candidate in self.active_stories(window_days, as_of):
            if not candidate.centroid:
                continue
            sim = StoryStore._cosine(cluster_centroid, candidate.centroid)
            if sim >= cosine_threshold and sim > best_sim:
                best_sim = sim
                best = candidate

        if best is not None:
            best.status = "ONGOING"
            best.entity_names = best.entity_names | cluster.entity_names
            # Carry today's enriched items into the persistent-identity shell so that
            # downstream score_stories (which reads story.items) does not see an empty list.
            # Persistent identity fields (story_id, first_seen, centroid, surfaced_*) are
            # preserved from the store; canonical_item and items come from today's cluster.
            best.items = cluster.items
            best.canonical_item = cluster.canonical_item
            self._persist_story(best)
            return best, False

        cluster.centroid = cluster_centroid
        cluster.status = "NEW"
        if cluster.first_seen is None:
            cluster.first_seen = as_of.isoformat()
        self._persist_story(cluster)
        return cluster, True

    def upsert_evidence(self, story_id: str, items: list[EnrichedItem], added_at: str) -> None:
        """Insert/refresh evidence rows for a story; (story_id, canonical_url) is the key.

        §6 item 1: source_tier is stored as int. Derived from item.source_role via
        source_tiers.json if a loader is readily importable; otherwise stored as 0.
        # TODO(stage4): populate source_tier via source_tiers.json
        """
        for ei in items:
            item = ei.item
            # TODO(stage4): populate source_tier via source_tiers.json mapping
            source_tier = 0
            verified_first_date = getattr(item, "verified_first_date", None)
            self._conn.execute(
                """
                INSERT INTO evidence (
                    story_id, canonical_url, title, source_id, source_role,
                    provenance, source_tier, added_at, verified_first_date
                )
                VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(story_id, canonical_url) DO UPDATE SET
                    title=excluded.title,
                    source_id=excluded.source_id,
                    source_role=excluded.source_role,
                    provenance=excluded.provenance,
                    source_tier=excluded.source_tier,
                    verified_first_date=excluded.verified_first_date
                """,
                (
                    story_id,
                    item.canonical_url,
                    item.title,
                    item.source_id,
                    item.source_role,
                    getattr(item, "provenance", ""),
                    source_tier,
                    added_at,
                    verified_first_date,
                ),
            )
        self._conn.commit()

    def record_surface(self, story: Story, run_date: str, *, lane: str = "featured") -> None:
        """Freeze the story's surface-state snapshot at run_date.

        Writes surfaced_* fields. lane="resurgence" also sets surfaced_resurged_at
        and, on first-ever resurge, the immutable resurged_at.
        """
        story.last_surfaced = run_date
        story.surfaced_entity_names = set(story.entity_names)
        story.surfaced_arxiv_versions = dict(story.arxiv_versions)
        if lane == "resurgence":
            if story.resurged_at is None:
                story.resurged_at = run_date
            story.surfaced_resurged_at = run_date
        self._persist_story(story)

    # -- seed backfill (write-once, NOT via match_or_create) -----------------

    def seed_first_seen(self, story: Story, first_seen: str) -> None:
        """Write-once: set first_seen on an existing story row directly.

        Contract §6 item 1 + §2.3: write-once (no-op if first_seen already set);
        MUST NOT route through match_or_create. Writes directly to the stories row.
        """
        row = self._conn.execute(
            "SELECT first_seen FROM stories WHERE story_id = ?",
            (story.story_id,),
        ).fetchone()
        if row is None:
            # Story not yet persisted; persist it with the provided first_seen
            story.first_seen = first_seen
            self._persist_story(story)
            return
        if row["first_seen"] is not None:
            # Already set — write-once: no-op
            return
        # Write first_seen directly without routing through match_or_create
        self._conn.execute(
            "UPDATE stories SET first_seen = ?, updated_at = ? WHERE story_id = ?",
            (first_seen, _now_iso(), story.story_id),
        )
        self._conn.commit()
        story.first_seen = first_seen

    # -- date verdict cache (permanent freeze, INV3) -------------------------

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
        """Write-once (INV3). No-op if a verdict for content_hash already exists (freeze)."""
        existing = self._conn.execute(
            "SELECT 1 FROM date_verdicts WHERE content_hash=?", (content_hash,)
        ).fetchone()
        if existing is not None:
            return
        self._conn.execute(
            "INSERT INTO date_verdicts (content_hash, verified_first_date, confidence, evidence) "
            "VALUES (?,?,?,?)",
            (
                content_hash,
                verdict.get("verified_first_date"),
                float(verdict.get("confidence", 0.0)),
                json.dumps(verdict.get("evidence", [])),
            ),
        )
        self._conn.commit()

    # -- version counts (monotonic, NOT in date_verdicts, INV3) --------------

    def refresh_arxiv_versions(self, arxiv_id: str, fetched_count: int) -> None:
        """new := max(old, fetched_count). Monotonic non-decreasing (INV3);
        never touches date_verdicts."""
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

    # -- audit snapshot (MUST include date_verdicts, contract §2.3 + §E) ----

    def dump_text_snapshot(self, out_dir: Path) -> Path:
        """Dump all four tables to one schema-tagged JSON file.

        MUST include date_verdicts (contract §2.3, spec §E/§G.4).
        Uses atomic write via temp file to avoid partial snapshots.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        blob: dict[str, object] = {"schema_version": 1, "dumped_at": _now_iso()}
        for table in _TABLES:
            rows = self._conn.execute(f"SELECT * FROM {table}").fetchall()  # noqa: S608
            blob[table] = [dict(r) for r in rows]

        snapshot = out_dir / "story_store_snapshot.json"
        tmp = out_dir / "story_store_snapshot.json.tmp"
        tmp.write_text(json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(snapshot)  # atomic rename
        return snapshot

    def load_text_snapshot(self, snapshot: Path) -> None:
        """Rebuild all four tables from a text snapshot, inheriting frozen verdicts."""
        blob = json.loads(Path(snapshot).read_text(encoding="utf-8"))
        if blob.get("schema_version") != 1:
            raise ValueError(
                f"snapshot schema_version {blob.get('schema_version')!r} != 1; refusing to load to avoid silent corruption"
            )
        for table in _TABLES:
            self._conn.execute(f"DELETE FROM {table}")  # noqa: S608
            for row in blob.get(table, []):
                cols = list(row.keys())
                placeholders = ",".join("?" for _ in cols)
                col_list = ",".join(cols)
                self._conn.execute(
                    f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})",  # noqa: S608
                    tuple(row[c] for c in cols),
                )
        self._conn.commit()
