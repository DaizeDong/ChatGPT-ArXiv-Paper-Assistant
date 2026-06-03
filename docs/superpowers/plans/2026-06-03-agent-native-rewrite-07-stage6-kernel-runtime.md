# Stage 6 — Kernel Orchestration & Runtime Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a thin deterministic `Kernel` that drives a fixed-DAG hotspot pipeline with per-`(date,stage)` checkpoints, wire the Synthesize agent touchpoint (temp-0 + structured + deterministic evidence-URL verifier), add an isolated Resurgence section through web_data + renderers, strangle `generate_daily_hotspot_report` into a Kernel call, and migrate runtime to a VPS `claude -p` cron while demoting GitHub Actions to a pure Publisher.

**Architecture:** A single-writer Python Kernel (`hotspots/kernel.py`) owns a hardcoded `STAGES` topology (never an LLM plan). Each stage reads its predecessor's JSON checkpoint, runs deterministic Python (or an agent touchpoint followed by a deterministic verifier), and writes its own checkpoint atomically; re-running a completed `(date,stage)` is a no-op, so same-day re-runs are bit-stable. Stages 0–5 (Store, GateDate, DateVerify, Dedup, NoveltyGate, GapFill, X-harvest) already exist; this stage only adds orchestration, the Synthesize touchpoint, Resurgence rendering, and runtime files.

**Tech Stack:** Python 3.12, `pytest` (`unittest.TestCase` style), `sqlite3` stdlib via `StoryStore`, Claude Code headless (`claude -p`), systemd timer on VPS, GitHub Actions Publisher.

---

## 0. Scope, dependencies & contract locks

This plan implements spec §A.1/§A.2 (thin Kernel + fixed-stage DAG + 5 agent touchpoints + per-`(date,stage)` idempotent checkpoint), §C.4 rendering, §E (VPS cron + headless + idempotent retry + secrets + Actions→Publisher + audit-branch snapshot), §F.2 stage 6 (strangle `generate_daily_hotspot_report`), and §G (anti-nondeterminism constitution — Synthesize agent followed by schema + evidence-URL verifier, temp 0, pinned versions, nightly replay-diff).

**Depends on stages 0–5** (overview doc dependency table). Contract-locked signatures this plan MUST use verbatim:

```python
# hotspots/kernel.py  (overview §2.10)
def run(output_root: Path, target_date: datetime, config, *, stage: str | None = None,
        force: bool = False) -> dict: ...
STAGES = ["harvest","date_verify","gravity_gate","embed","cluster",
          "storystore_match","gapfill","score","synthesize","render"]

# hotspots/store.py (overview §2.3) — used, not defined here:
StoryStore(db_path: Path)
StoryStore.record_surface(story: Story, run_date: str, *, lane: str = "featured") -> None
StoryStore.dump_text_snapshot(out_dir: Path) -> Path     # MUST include date_verdicts
StoryStore.active_stories(window_days: int, as_of: date) -> list[Story]
StoryStore.upsert_evidence(story_id: str, items: list[EnrichedItem], added_at: str) -> None

# hotspots/novelty.py (overview §2.6) — used, not defined here:
def resurge(story, *, max_age_days, run_date, min_competitors, cooldown_days, gate_date_fn=gate_date) -> bool
```

These come from stages 0–5. **Do not redefine them.** If a dependency is absent at execution time, its task notes a deterministic stub fallback so the Kernel still runs degraded (per overview §5 "deterministic Actions fallback").

**Real code this stage strangles / touches:**
- `arxiv_assistant/hotspots/pipeline.py` — `generate_daily_hotspot_report` (1630-1839), `fetch_source_payloads` (887-1027), `apply_digest_synthesis` (1172-1217), `_raw_source_cache_path` (264-265), `_heuristic_takeaways` (1116), `parse_target_datetime` (193), `date_string` (224).
- `scripts/generate_daily_hotspots.py` — CLI entry (already calls `generate_daily_hotspot_report`).
- `arxiv_assistant/utils/hotspot/hotspot_web_data.py` — `build_daily_hotspot_web_payload` (926-1005).
- `arxiv_assistant/renderers/hotspot/render_hot_daily.py` — `render_hot_daily_md` (99+).
- `.github/workflows/cron_runs.yaml`, `.github/workflows/publish_md.yml`.

**Determinism invariants asserted as acceptance tests (overview §4 / spec §G):** INV6 (every agent followed by a deterministic verifier; temp 0; pinned ids) and the replay-diff bit-stability gate.

---

## 1. File structure (created / modified in this stage)

```
arxiv_assistant/hotspots/
  kernel.py                     CREATE — run() + STAGES + checkpoint I/O + stage fns + Synthesize touchpoint
  pipeline.py                   MODIFY — generate_daily_hotspot_report() delegates to kernel.run()
arxiv_assistant/utils/hotspot/
  hotspot_web_data.py           MODIFY — emit "resurgence" section into web payload
arxiv_assistant/renderers/hotspot/
  render_hot_daily.py           MODIFY — render "## Resurgence" markdown section
scripts/
  generate_daily_hotspots.py    MODIFY — add --stage passthrough
deploy/vps/
  hotspot.service               CREATE — systemd oneshot unit (claude -p headless)
  hotspot.timer                 CREATE — systemd daily timer
  hotspot.env.example           CREATE — EnvironmentFile template (NOT real secrets)
  run_hotspot.sh                CREATE — headless entrypoint + snapshot push to audit branch
.github/workflows/
  cron_runs.yaml                MODIFY — gate generation behind runtime==actions (deterministic degrade)
  publish_md.yml                MODIFY — add _zh validation gate (fail if translation merge silently dropped)
tests/
  test_kernel.py                CREATE
  test_replay_diff.py           CREATE
  fixtures/agent/synthesize_ok.json        CREATE — replayed good agent response
  fixtures/agent/synthesize_halluc.json    CREATE — replayed response with a fake evidence URL
  fixtures/replay/raw_2026-05-20.json      CREATE — frozen historical raw input
```

**File-size discipline:** `kernel.py` < ~300 lines; it imports stage logic from stages 0–5 modules rather than re-implementing them. The Synthesize touchpoint and its verifier are the only new judgment logic.

---

## 2. Shared contract recap for executors

Each stage function has the uniform signature `stage_fn(ctx: KernelContext) -> dict` and returns the **checkpoint payload** for that stage. `KernelContext` is a frozen dataclass carrying `output_root`, `target_date`, `config`, `store`, `run_date` (ISO day string), `journal` (list for run_journal rows), and a `read(stage)` helper to load an upstream checkpoint. The Kernel — and only the Kernel — calls `store.record_surface` / `store.upsert_evidence`, satisfying the single-writer rule (spec §A.1).

Checkpoint files live at `out/hot/state/checkpoint/<date>/<stage>.json`. A stage is "done" iff its checkpoint file exists and parses; `force=True` deletes checkpoints for the target date before running.

---

## Task 1: KernelContext + checkpoint path helpers

**Files:**
- Create: `arxiv_assistant/hotspots/kernel.py`
- Test: `tests/test_kernel.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kernel.py
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from arxiv_assistant.hotspots import kernel


class TestCheckpointIO(unittest.TestCase):
    def test_checkpoint_roundtrip_and_done_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            td = datetime(2026, 5, 20, tzinfo=timezone.utc)
            self.assertFalse(kernel._checkpoint_done(root, td, "harvest"))
            path = kernel._write_checkpoint(root, td, "harvest", {"items": [1, 2, 3]})
            self.assertTrue(path.exists())
            self.assertTrue(kernel._checkpoint_done(root, td, "harvest"))
            self.assertEqual(kernel._read_checkpoint(root, td, "harvest"), {"items": [1, 2, 3]})

    def test_checkpoint_path_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            td = datetime(2026, 5, 20, tzinfo=timezone.utc)
            p = kernel._checkpoint_path(root, td, "score")
            self.assertEqual(p, root / "hot" / "state" / "checkpoint" / "2026-05-20" / "score.json")

    def test_clear_checkpoints_removes_all_stages_for_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            td = datetime(2026, 5, 20, tzinfo=timezone.utc)
            kernel._write_checkpoint(root, td, "harvest", {})
            kernel._write_checkpoint(root, td, "score", {})
            kernel._clear_checkpoints(root, td)
            self.assertFalse(kernel._checkpoint_done(root, td, "harvest"))
            self.assertFalse(kernel._checkpoint_done(root, td, "score"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_kernel.py::TestCheckpointIO -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'arxiv_assistant.hotspots.kernel'`.

- [ ] **Step 3: Write minimal implementation**

```python
# arxiv_assistant/hotspots/kernel.py
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from arxiv_assistant.hotspots.pipeline import date_string

STAGES: list[str] = [
    "harvest", "date_verify", "gravity_gate", "embed", "cluster",
    "storystore_match", "gapfill", "score", "synthesize", "render",
]


def _checkpoint_dir(output_root: Path, target_date: datetime) -> Path:
    return Path(output_root) / "hot" / "state" / "checkpoint" / date_string(target_date)


def _checkpoint_path(output_root: Path, target_date: datetime, stage: str) -> Path:
    return _checkpoint_dir(output_root, target_date) / f"{stage}.json"


def _checkpoint_done(output_root: Path, target_date: datetime, stage: str) -> bool:
    path = _checkpoint_path(output_root, target_date, stage)
    if not path.exists():
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except (json.JSONDecodeError, OSError):
        return False


def _read_checkpoint(output_root: Path, target_date: datetime, stage: str) -> dict[str, Any]:
    path = _checkpoint_path(output_root, target_date, stage)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_checkpoint(output_root: Path, target_date: datetime, stage: str, payload: dict[str, Any]) -> Path:
    path = _checkpoint_path(output_root, target_date, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    tmp.replace(path)  # atomic on same filesystem
    return path


def _clear_checkpoints(output_root: Path, target_date: datetime) -> None:
    d = _checkpoint_dir(output_root, target_date)
    if d.exists():
        shutil.rmtree(d)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_kernel.py::TestCheckpointIO -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add arxiv_assistant/hotspots/kernel.py tests/test_kernel.py
git commit -m "feat(kernel): checkpoint I/O + atomic write + clear helpers"
```

---

## Task 2: KernelContext dataclass + bounded retry/degrade helper

**Files:**
- Modify: `arxiv_assistant/hotspots/kernel.py`
- Test: `tests/test_kernel.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kernel.py  (append)
import configparser


class TestContextAndRetry(unittest.TestCase):
    def _config(self) -> configparser.ConfigParser:
        cfg = configparser.ConfigParser()
        cfg["HOTSPOTS"] = {"enabled": "true"}
        return cfg

    def test_context_run_date_is_target_utc_day(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = kernel.KernelContext(
                output_root=Path(tmp),
                target_date=datetime(2026, 5, 20, 13, 30, tzinfo=timezone.utc),
                config=self._config(),
                store=None,
                journal=[],
            )
            self.assertEqual(ctx.run_date, "2026-05-20")

    def test_retry_succeeds_after_transient_failures(self) -> None:
        calls = {"n": 0}

        def flaky() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("transient")
            return "ok"

        out = kernel._with_retry(flaky, attempts=3, base_delay=0.0)
        self.assertEqual(out, "ok")
        self.assertEqual(calls["n"], 3)

    def test_retry_degrades_to_fallback_after_exhaustion(self) -> None:
        def always_fail() -> str:
            raise RuntimeError("boom")

        out = kernel._with_retry(always_fail, attempts=3, base_delay=0.0, fallback=lambda: "degraded")
        self.assertEqual(out, "degraded")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_kernel.py::TestContextAndRetry -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'KernelContext'`.

- [ ] **Step 3: Write minimal implementation**

```python
# arxiv_assistant/hotspots/kernel.py  (append after _clear_checkpoints)
import time


@dataclass(frozen=True)
class KernelContext:
    output_root: Path
    target_date: datetime
    config: Any
    store: Any
    journal: list = field(default_factory=list)

    @property
    def run_date(self) -> str:
        return date_string(self.target_date)

    def read(self, stage: str) -> dict[str, Any]:
        return _read_checkpoint(self.output_root, self.target_date, stage)


def _with_retry(
    fn: Callable[[], Any],
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    fallback: Callable[[], Any] | None = None,
) -> Any:
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — bounded retry boundary
            last_exc = exc
            if attempt < attempts - 1 and base_delay > 0:
                time.sleep(base_delay * (2 ** attempt))
    if fallback is not None:
        return fallback()
    raise last_exc if last_exc else RuntimeError("retry failed")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_kernel.py::TestContextAndRetry -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add arxiv_assistant/hotspots/kernel.py tests/test_kernel.py
git commit -m "feat(kernel): KernelContext + bounded retry with deterministic degrade"
```

---

## Task 3: Fixed-DAG driver — `run()` with resume + force + single-stage

**Files:**
- Modify: `arxiv_assistant/hotspots/kernel.py`
- Test: `tests/test_kernel.py`

The driver iterates `STAGES` in order, calling the registered stage function only when its checkpoint is absent (resume); a stage whose checkpoint exists is a no-op. Stage functions are looked up in a hardcoded `_STAGE_FNS` dict — the topology is **in code, never an LLM plan** (spec §G.2). For this task we register trivial echo stage functions; real stage bodies arrive in Tasks 4–7.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kernel.py  (append)
class TestDagDriver(unittest.TestCase):
    def _config(self) -> configparser.ConfigParser:
        cfg = configparser.ConfigParser()
        cfg["HOTSPOTS"] = {"enabled": "true"}
        return cfg

    def test_stage_order_is_fixed(self) -> None:
        self.assertEqual(
            kernel.STAGES,
            ["harvest", "date_verify", "gravity_gate", "embed", "cluster",
             "storystore_match", "gapfill", "score", "synthesize", "render"],
        )

    def test_run_executes_stages_in_order_and_records_each(self) -> None:
        order: list[str] = []

        def make(stage_name: str):
            def _fn(ctx: kernel.KernelContext) -> dict:
                order.append(stage_name)
                return {"stage": stage_name}
            return _fn

        fns = {s: make(s) for s in kernel.STAGES}
        with tempfile.TemporaryDirectory() as tmp, \
                unittest.mock.patch.object(kernel, "_STAGE_FNS", fns):
            manifest = kernel.run(
                Path(tmp), datetime(2026, 5, 20, tzinfo=timezone.utc), self._config(),
            )
        self.assertEqual(order, kernel.STAGES)
        self.assertEqual(manifest["stages_run"], kernel.STAGES)
        self.assertEqual(manifest["date"], "2026-05-20")

    def test_resume_is_noop_on_completed_stages(self) -> None:
        order: list[str] = []

        def make(stage_name: str):
            def _fn(ctx: kernel.KernelContext) -> dict:
                order.append(stage_name)
                return {"stage": stage_name}
            return _fn

        fns = {s: make(s) for s in kernel.STAGES}
        with tempfile.TemporaryDirectory() as tmp, \
                unittest.mock.patch.object(kernel, "_STAGE_FNS", fns):
            root = Path(tmp)
            td = datetime(2026, 5, 20, tzinfo=timezone.utc)
            kernel.run(root, td, self._config())
            order.clear()
            manifest = kernel.run(root, td, self._config())  # second run, all cached
        self.assertEqual(order, [])  # nothing re-executed
        self.assertEqual(manifest["stages_run"], [])
        self.assertEqual(manifest["stages_skipped"], kernel.STAGES)

    def test_single_stage_run_executes_only_that_stage(self) -> None:
        order: list[str] = []
        fns = {s: (lambda ctx, n=s: (order.append(n), {"stage": n})[1]) for s in kernel.STAGES}
        with tempfile.TemporaryDirectory() as tmp, \
                unittest.mock.patch.object(kernel, "_STAGE_FNS", fns):
            kernel.run(Path(tmp), datetime(2026, 5, 20, tzinfo=timezone.utc),
                       self._config(), stage="score")
        self.assertEqual(order, ["score"])

    def test_force_clears_then_reruns(self) -> None:
        order: list[str] = []
        fns = {s: (lambda ctx, n=s: (order.append(n), {"stage": n})[1]) for s in kernel.STAGES}
        with tempfile.TemporaryDirectory() as tmp, \
                unittest.mock.patch.object(kernel, "_STAGE_FNS", fns):
            root = Path(tmp)
            td = datetime(2026, 5, 20, tzinfo=timezone.utc)
            kernel.run(root, td, self._config())
            order.clear()
            kernel.run(root, td, self._config(), force=True)
        self.assertEqual(order, kernel.STAGES)  # all re-run after force
```

(Add `import unittest.mock` at top of test module.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_kernel.py::TestDagDriver -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'run'` / `_STAGE_FNS`.

- [ ] **Step 3: Write minimal implementation**

```python
# arxiv_assistant/hotspots/kernel.py  (append)

# Topology is hardcoded here (spec §G.2): never produced by an LLM.
# Real stage bodies are bound in Tasks 4-7; tests patch _STAGE_FNS.
_STAGE_FNS: dict[str, Callable[["KernelContext"], dict[str, Any]]] = {}


def _open_store(output_root: Path, config: Any):
    try:
        from arxiv_assistant.hotspots.store import StoryStore
    except Exception:
        return None  # degraded deterministic run (overview §5)
    db_path = Path(output_root) / "hot" / "state" / "story_store.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return StoryStore(db_path)


def run(output_root: Path, target_date: datetime, config, *, stage: str | None = None,
        force: bool = False) -> dict:
    output_root = Path(output_root)
    if force:
        _clear_checkpoints(output_root, target_date)

    if stage is not None and stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; valid: {STAGES}")
    target_stages = [stage] if stage is not None else list(STAGES)

    store = _open_store(output_root, config)
    journal: list = []
    ctx = KernelContext(
        output_root=output_root, target_date=target_date, config=config,
        store=store, journal=journal,
    )

    stages_run: list[str] = []
    stages_skipped: list[str] = []
    for name in target_stages:
        if stage is None and _checkpoint_done(output_root, target_date, name):
            stages_skipped.append(name)
            continue
        fn = _STAGE_FNS[name]
        payload = fn(ctx)
        _write_checkpoint(output_root, target_date, name, payload)
        stages_run.append(name)

    return {
        "date": ctx.run_date,
        "stages_run": stages_run,
        "stages_skipped": stages_skipped,
        "journal": journal,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_kernel.py::TestDagDriver -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add arxiv_assistant/hotspots/kernel.py tests/test_kernel.py
git commit -m "feat(kernel): fixed-DAG run() with resume/force/single-stage idempotence"
```

---

## Task 4: Harvest → gravity_gate stage bodies (reuse stage 0–1 logic)

**Files:**
- Modify: `arxiv_assistant/hotspots/kernel.py`
- Test: `tests/test_kernel.py`

These stages wrap existing deterministic pipeline functions. `harvest` calls `fetch_source_payloads`; `date_verify` annotates `verified_first_date` via stage-3 `DateVerify` (with a deterministic stub fallback when absent); `gravity_gate` applies the max-age hard gate using `gate_date` (stage 1). Per-source adapter failure inside `fetch_source_payloads` already degrades (it catches exceptions and records `items=[]`), so harvest needs no extra retry.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kernel.py  (append)
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem


def _item(title: str, url: str, published_at: str) -> HotspotItem:
    return HotspotItem(
        source_id="hf_papers", source_name="HF", source_role="papers",
        source_type="papers", title=title, summary="s", url=url,
        canonical_url=url, published_at=published_at, tags=[], authors=[], metadata={},
    )


class TestHarvestStages(unittest.TestCase):
    def _config(self) -> configparser.ConfigParser:
        cfg = configparser.ConfigParser()
        cfg["HOTSPOTS"] = {"enabled": "true", "max_raw_items": "120", "max_item_age_days": "14"}
        return cfg

    def test_harvest_stage_serializes_items(self) -> None:
        items = [_item("A", "https://x/a", "2026-05-20T00:00:00Z")]
        with tempfile.TemporaryDirectory() as tmp, \
                unittest.mock.patch.object(
                    kernel, "_fetch_source_payloads",
                    return_value=(items, {"hf_papers": 1}, {})):
            ctx = kernel.KernelContext(
                output_root=Path(tmp),
                target_date=datetime(2026, 5, 20, tzinfo=timezone.utc),
                config=self._config(), store=None, journal=[],
            )
            payload = kernel._stage_harvest(ctx)
        self.assertEqual(payload["source_stats"], {"hf_papers": 1})
        self.assertEqual(len(payload["items"]), 1)
        self.assertEqual(payload["items"][0]["title"], "A")

    def test_gravity_gate_drops_items_older_than_max_age(self) -> None:
        fresh = _item("fresh", "https://x/fresh", "2026-05-20T00:00:00Z")
        stale = _item("stale", "https://x/stale", "2026-04-01T00:00:00Z")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            td = datetime(2026, 5, 20, tzinfo=timezone.utc)
            kernel._write_checkpoint(root, td, "date_verify", {
                "items": [kernel._serialize_item(fresh), kernel._serialize_item(stale)],
            })
            ctx = kernel.KernelContext(output_root=root, target_date=td,
                                       config=self._config(), store=None, journal=[])
            payload = kernel._stage_gravity_gate(ctx)
        titles = {it["title"] for it in payload["items"]}
        self.assertIn("fresh", titles)
        self.assertNotIn("stale", titles)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_kernel.py::TestHarvestStages -v`
Expected: FAIL — `_stage_harvest` / `_serialize_item` undefined.

- [ ] **Step 3: Write minimal implementation**

```python
# arxiv_assistant/hotspots/kernel.py  (append)
from datetime import timedelta, timezone as _tz

from arxiv_assistant.hotspots.pipeline import (
    fetch_source_payloads as _fetch_source_payloads,
    _serialize_items,
)
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem
from arxiv_assistant.utils.hotspot.hotspot_sources import parse_datetime


def _serialize_item(item: HotspotItem) -> dict[str, Any]:
    return _serialize_items([item])[0]


def _deserialize_item(row: dict[str, Any]) -> HotspotItem:
    return HotspotItem(
        source_id=row.get("source_id", ""), source_name=row.get("source_name", ""),
        source_role=row.get("source_role", ""), source_type=row.get("source_type", ""),
        title=row.get("title", ""), summary=row.get("summary", ""),
        url=row.get("url", ""), canonical_url=row.get("canonical_url", ""),
        published_at=row.get("published_at"), tags=row.get("tags", []) or [],
        authors=row.get("authors", []) or [], metadata=row.get("metadata", {}) or {},
    )


def _gate_date_for(item: HotspotItem):
    """Day-granular credible date (spec §B.5.1). Uses stage-1 gate_date; falls back
    to floored published_at when the GateDate module is absent (degraded run)."""
    try:
        from arxiv_assistant.utils.hotspot.gate_date import gate_date
        return gate_date(item)
    except Exception:
        dt = parse_datetime(getattr(item, "verified_first_date", None) or item.published_at)
        return dt.astimezone(_tz.utc).date() if dt else None


def _stage_harvest(ctx: KernelContext) -> dict[str, Any]:
    items, source_stats, api_usage = _fetch_source_payloads(
        ctx.target_date, ctx.output_root, ctx.config, force=False)
    cap = ctx.config["HOTSPOTS"].getint("max_raw_items", fallback=120)
    items = items[:cap]
    return {
        "items": [_serialize_item(it) for it in items],
        "source_stats": source_stats,
        "api_usage": api_usage,
    }


def _stage_date_verify(ctx: KernelContext) -> dict[str, Any]:
    rows = ctx.read("harvest")["items"]
    items = [_deserialize_item(r) for r in rows]
    try:
        from arxiv_assistant.hotspots.date_verify import verify
        for it in items:
            verdict = _with_retry(
                lambda it=it: verify(it, ctx.store),
                attempts=3, base_delay=0.0,
                fallback=lambda it=it: {
                    "verified_first_date": it.published_at, "confidence": 0.3, "evidence": [],
                },
            )
            it.verified_first_date = verdict.get("verified_first_date") or it.published_at
    except Exception:
        for it in items:  # degraded: trust floored published_at, low confidence
            it.verified_first_date = it.published_at
    return {"items": [_serialize_item(it) for it in items]}


def _stage_gravity_gate(ctx: KernelContext) -> dict[str, Any]:
    rows = ctx.read("date_verify")["items"]
    items = [_deserialize_item(r) for r in rows]
    max_age = ctx.config["HOTSPOTS"].getint("max_item_age_days", fallback=14)
    as_of = ctx.target_date.astimezone(_tz.utc).date()
    cutoff = as_of - timedelta(days=max_age)
    kept: list[HotspotItem] = []
    dropped = 0
    for it in items:
        gd = _gate_date_for(it)
        if gd is not None and gd < cutoff:
            dropped += 1
            continue
        kept.append(it)
    ctx.journal.append({"stage": "gravity_gate", "dropped_stale": dropped, "kept": len(kept)})
    return {"items": [_serialize_item(it) for it in kept]}
```

Now bind these three into the registry (replace the empty `_STAGE_FNS` dict at the bottom of the module — keep it the last statement so all functions are defined):

```python
# arxiv_assistant/hotspots/kernel.py  (REPLACE the `_STAGE_FNS = {}` line with:)
_STAGE_FNS = {
    "harvest": _stage_harvest,
    "date_verify": _stage_date_verify,
    "gravity_gate": _stage_gravity_gate,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_kernel.py::TestHarvestStages -v`
Expected: PASS (2 tests). Re-run `tests/test_kernel.py::TestDagDriver` — still green because those tests patch `_STAGE_FNS` wholesale.

- [ ] **Step 5: Commit**

```bash
git add arxiv_assistant/hotspots/kernel.py tests/test_kernel.py
git commit -m "feat(kernel): harvest/date_verify/gravity_gate stage bodies with degrade"
```

---

## Task 5: embed → cluster → storystore_match → gapfill → score stage bodies

**Files:**
- Modify: `arxiv_assistant/hotspots/kernel.py`
- Test: `tests/test_kernel.py`

These wrap stage 2/4 deterministic logic. Where a dependency module is missing, the body degrades to the legacy in-place pipeline path (`enrich_items_heuristic` → `group_into_stories` → `score_stories`) so the Kernel always produces stories. To keep this task focused and testable without the SQLite Store, we test the **degraded** path (store=None), which is also the Actions fallback path (overview §5).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kernel.py  (append)
class TestStoryStages(unittest.TestCase):
    def _config(self) -> configparser.ConfigParser:
        cfg = configparser.ConfigParser()
        cfg["HOTSPOTS"] = {
            "enabled": "true", "mode": "heuristic", "target_topics": "5",
            "target_watchlist_topics": "3", "max_topics_per_category": "4",
            "cross_day_window_days": "14", "crossday_cosine_threshold": "0.90",
        }
        return cfg

    def test_score_stage_emits_featured_and_watchlist(self) -> None:
        items = [
            _item("Big Model release", "https://x/a", "2026-05-20T00:00:00Z"),
            _item("Big Model release", "https://x/b", "2026-05-20T00:00:00Z"),
            _item("Other thing", "https://x/c", "2026-05-20T00:00:00Z"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            td = datetime(2026, 5, 20, tzinfo=timezone.utc)
            kernel._write_checkpoint(root, td, "gravity_gate",
                                     {"items": [kernel._serialize_item(i) for i in items]})
            ctx = kernel.KernelContext(output_root=root, target_date=td,
                                       config=self._config(), store=None, journal=[])
            # embed/cluster/storystore_match/gapfill pass-through in degraded mode
            kernel._write_checkpoint(root, td, "embed", kernel._stage_embed(ctx))
            kernel._write_checkpoint(root, td, "cluster", kernel._stage_cluster(ctx))
            kernel._write_checkpoint(root, td, "storystore_match", kernel._stage_storystore_match(ctx))
            kernel._write_checkpoint(root, td, "gapfill", kernel._stage_gapfill(ctx))
            payload = kernel._stage_score(ctx)
        self.assertIn("featured", payload)
        self.assertIn("watchlist", payload)
        self.assertIsInstance(payload["featured"], list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_kernel.py::TestStoryStages -v`
Expected: FAIL — `_stage_embed` undefined.

- [ ] **Step 3: Write minimal implementation**

```python
# arxiv_assistant/hotspots/kernel.py  (append)
from arxiv_assistant.hotspots.pipeline import (
    enrich_items_batch, enrich_items_heuristic, group_into_stories, score_stories,
    select_and_categorize, _story_to_topic_dict,
)


def _items_from(ctx: KernelContext, stage: str) -> list[HotspotItem]:
    return [_deserialize_item(r) for r in ctx.read(stage)["items"]]


def _stage_embed(ctx: KernelContext) -> dict[str, Any]:
    # Embedding centroids are computed inside stage-2 dedup when present; in degraded
    # mode this is a structural pass-through carrying items forward unchanged.
    return {"items": ctx.read("gravity_gate")["items"]}


def _stage_cluster(ctx: KernelContext) -> dict[str, Any]:
    return {"items": ctx.read("embed")["items"]}


def _stage_storystore_match(ctx: KernelContext) -> dict[str, Any]:
    # Persistent-id assignment happens here when a Store is present (stage 2).
    # The single-writer rule is honoured: only the Kernel touches the Store.
    return {"items": ctx.read("cluster")["items"]}


def _stage_gapfill(ctx: KernelContext) -> dict[str, Any]:
    return {"items": ctx.read("storystore_match")["items"]}


def _enrich(ctx: KernelContext, items: list[HotspotItem]):
    cfg = ctx.config["HOTSPOTS"]
    mode = cfg.get("mode", "heuristic")
    if mode == "openai":
        model = cfg.get("model_enrich", cfg.get("model_screen"))
        return enrich_items_batch(items, model, cfg.getint("enrich_batch_size", fallback=20),
                                  cfg.getint("retry", fallback=3))
    return enrich_items_heuristic(items)


def _stage_score(ctx: KernelContext) -> dict[str, Any]:
    cfg = ctx.config["HOTSPOTS"]
    items = _items_from(ctx, "gapfill")
    enriched = _enrich(ctx, items)
    stories = score_stories(group_into_stories(enriched))
    featured, watchlist, _ = select_and_categorize(
        stories,
        target_featured=cfg.getint("target_topics", fallback=5),
        target_watchlist=cfg.getint("target_watchlist_topics", fallback=3),
        max_per_category=cfg.getint("max_topics_per_category", fallback=4),
    )
    return {
        "featured": [_story_to_topic_dict(s, keep=True) for s in featured],
        "watchlist": [_story_to_topic_dict(s, watchlist=True) for s in watchlist],
        "all_topics": [_story_to_topic_dict(s) for s in stories],
    }
```

Extend the registry:

```python
# arxiv_assistant/hotspots/kernel.py  (REPLACE the _STAGE_FNS dict again:)
_STAGE_FNS = {
    "harvest": _stage_harvest,
    "date_verify": _stage_date_verify,
    "gravity_gate": _stage_gravity_gate,
    "embed": _stage_embed,
    "cluster": _stage_cluster,
    "storystore_match": _stage_storystore_match,
    "gapfill": _stage_gapfill,
    "score": _stage_score,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_kernel.py::TestStoryStages -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add arxiv_assistant/hotspots/kernel.py tests/test_kernel.py
git commit -m "feat(kernel): embed/cluster/match/gapfill/score stage bodies (degrade-safe)"
```

---

## Task 6: Synthesize touchpoint — temp-0 agent + deterministic schema + evidence-URL verifier

**Files:**
- Modify: `arxiv_assistant/hotspots/kernel.py`
- Create: `tests/fixtures/agent/synthesize_ok.json`
- Create: `tests/fixtures/agent/synthesize_halluc.json`
- Test: `tests/test_kernel.py`

This is spec §G.3 / §A.2 touchpoint ④⑤. The agent produces bilingual `headline`/`summary` per featured topic; we then run a **deterministic verifier**: (a) schema check (required keys, en+zh both present and non-empty), and (b) every cited evidence URL must already exist in that story's real evidence URL set. A topic that fails verification is rejected and falls back to `_heuristic_takeaways` + existing title (spec §E "Synthesize failure → `_heuristic_takeaways`"). The agent call itself is `temperature=0` and the model id is pinned and recorded in the manifest (spec §G.6); tests replay a captured JSON fixture instead of hitting the network (overview §4).

- [ ] **Step 1: Write the failing test + fixtures**

Create `tests/fixtures/agent/synthesize_ok.json`:

```json
{
  "topics": [
    {
      "TOPIC_ID": "t1",
      "headline_en": "Frontier lab ships agentic coding model",
      "headline_zh": "前沿实验室发布智能体编码模型",
      "summary_en": "A new release improves long-horizon coding tasks.",
      "summary_zh": "新版本提升了长程编码任务表现。",
      "evidence": ["https://x/a"]
    }
  ]
}
```

Create `tests/fixtures/agent/synthesize_halluc.json` (cites a URL not in the story):

```json
{
  "topics": [
    {
      "TOPIC_ID": "t1",
      "headline_en": "Frontier lab ships agentic coding model",
      "headline_zh": "前沿实验室发布智能体编码模型",
      "summary_en": "A new release improves long-horizon coding tasks.",
      "summary_zh": "新版本提升了长程编码任务表现。",
      "evidence": ["https://evil/fabricated"]
    }
  ]
}
```

```python
# tests/test_kernel.py  (append)
FIXT = Path(__file__).resolve().parent / "fixtures" / "agent"


class TestSynthesizeVerifier(unittest.TestCase):
    def _topic(self) -> dict:
        return {
            "TOPIC_ID": "t1", "title": "Old title",
            "WHY_IT_MATTERS": "", "KEY_TAKEAWAYS": [],
            "EVIDENCE_URLS": ["https://x/a", "https://x/b"],
        }

    def test_schema_check_rejects_missing_zh(self) -> None:
        row = {"TOPIC_ID": "t1", "headline_en": "h", "summary_en": "s",
               "headline_zh": "", "summary_zh": "", "evidence": ["https://x/a"]}
        self.assertFalse(kernel._synthesis_row_valid(row, {"https://x/a"}))

    def test_evidence_url_must_exist_in_story(self) -> None:
        row = {"TOPIC_ID": "t1", "headline_en": "h", "headline_zh": "标题",
               "summary_en": "s", "summary_zh": "摘要", "evidence": ["https://evil/x"]}
        self.assertFalse(kernel._synthesis_row_valid(row, {"https://x/a"}))

    def test_valid_row_passes(self) -> None:
        row = {"TOPIC_ID": "t1", "headline_en": "h", "headline_zh": "标题",
               "summary_en": "s", "summary_zh": "摘要", "evidence": ["https://x/a"]}
        self.assertTrue(kernel._synthesis_row_valid(row, {"https://x/a"}))

    def test_stage_applies_good_agent_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            td = datetime(2026, 5, 20, tzinfo=timezone.utc)
            kernel._write_checkpoint(root, td, "score",
                                     {"featured": [self._topic()], "watchlist": [], "all_topics": []})
            cfg = configparser.ConfigParser()
            cfg["HOTSPOTS"] = {"enabled": "true", "mode": "openai",
                               "model_synthesize": "pinned-model-v1"}
            ctx = kernel.KernelContext(output_root=root, target_date=td, config=cfg,
                                       store=None, journal=[])
            replay = json.loads((FIXT / "synthesize_ok.json").read_text(encoding="utf-8"))
            with unittest.mock.patch.object(kernel, "_call_synthesize_agent", return_value=replay):
                payload = kernel._stage_synthesize(ctx)
        topic = payload["featured"][0]
        self.assertEqual(topic["HEADLINE"], "Frontier lab ships agentic coding model")
        self.assertEqual(topic["HEADLINE_ZH"], "前沿实验室发布智能体编码模型")
        self.assertEqual(payload["manifest"]["synthesize_model"], "pinned-model-v1")
        self.assertEqual(payload["manifest"]["synthesize_temperature"], 0)

    def test_stage_rejects_hallucinated_url_and_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            td = datetime(2026, 5, 20, tzinfo=timezone.utc)
            topic = self._topic()
            kernel._write_checkpoint(root, td, "score",
                                     {"featured": [topic], "watchlist": [], "all_topics": []})
            cfg = configparser.ConfigParser()
            cfg["HOTSPOTS"] = {"enabled": "true", "mode": "openai",
                               "model_synthesize": "pinned-model-v1"}
            ctx = kernel.KernelContext(output_root=root, target_date=td, config=cfg,
                                       store=None, journal=[])
            replay = json.loads((FIXT / "synthesize_halluc.json").read_text(encoding="utf-8"))
            with unittest.mock.patch.object(kernel, "_call_synthesize_agent", return_value=replay):
                payload = kernel._stage_synthesize(ctx)
        out = payload["featured"][0]
        self.assertEqual(out["HEADLINE"], "Old title")          # original title kept
        self.assertTrue(out["KEY_TAKEAWAYS"])                    # heuristic fallback filled
        self.assertIn("t1", payload["manifest"]["synthesize_rejected"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_kernel.py::TestSynthesizeVerifier -v`
Expected: FAIL — `_synthesis_row_valid` / `_stage_synthesize` / `_call_synthesize_agent` undefined.

- [ ] **Step 3: Write minimal implementation**

```python
# arxiv_assistant/hotspots/kernel.py  (append)
from arxiv_assistant.hotspots.pipeline import _heuristic_takeaways

_SYNTH_REQUIRED = ("headline_en", "headline_zh", "summary_en", "summary_zh")


def _story_evidence_urls(topic: dict[str, Any]) -> set[str]:
    urls: set[str] = set()
    for key in ("EVIDENCE_URLS", "evidence_urls", "SOURCE_URLS"):
        for u in topic.get(key, []) or []:
            if u:
                urls.add(str(u))
    if topic.get("URL"):
        urls.add(str(topic["URL"]))
    return urls


def _synthesis_row_valid(row: dict[str, Any], story_urls: set[str]) -> bool:
    # (a) schema: all required bilingual fields present & non-empty
    for key in _SYNTH_REQUIRED:
        if not str(row.get(key, "")).strip():
            return False
    # (b) every cited evidence URL must really exist in the story (anti-hallucination)
    cited = [str(u) for u in row.get("evidence", []) or []]
    if not cited:
        return False
    return all(u in story_urls for u in cited)


def _call_synthesize_agent(topics: list[dict[str, Any]], model: str) -> dict[str, Any]:
    """Dispatch the stateless bilingual Synthesize subagent at temperature 0 with a
    forced JSON schema. Real impl shells `claude -p` headless; tests replay a fixture.
    Pinned `model` is recorded by the caller into the manifest (spec §G.6)."""
    from arxiv_assistant.hotspots.synthesize import synthesize_bilingual  # stage-deferred impl
    return synthesize_bilingual(topics, model=model, temperature=0)


def _stage_synthesize(ctx: KernelContext) -> dict[str, Any]:
    score = ctx.read("score")
    featured = [dict(t) for t in score.get("featured", [])]
    cfg = ctx.config["HOTSPOTS"]
    model = cfg.get("model_synthesize", cfg.get("model_summarize", cfg.get("model_screen", "")))

    rejected: list[str] = []
    if cfg.get("mode", "heuristic") == "openai" and featured:
        payload = _with_retry(
            lambda: _call_synthesize_agent(featured, model),
            attempts=2, base_delay=0.0, fallback=lambda: {"topics": []},
        )
        by_id = {str(r.get("TOPIC_ID")): r for r in payload.get("topics", [])}
        for topic in featured:
            tid = str(topic.get("TOPIC_ID"))
            row = by_id.get(tid)
            if row is not None and _synthesis_row_valid(row, _story_evidence_urls(topic)):
                topic["HEADLINE"] = row["headline_en"]
                topic["HEADLINE_ZH"] = row["headline_zh"]
                topic["WHY_IT_MATTERS"] = row["summary_en"]
                topic["WHY_IT_MATTERS_ZH"] = row["summary_zh"]
            else:
                rejected.append(tid)
                topic["HEADLINE"] = topic.get("title", topic.get("HEADLINE", ""))
                if not topic.get("KEY_TAKEAWAYS"):
                    topic["KEY_TAKEAWAYS"] = _heuristic_takeaways(topic)
    else:
        for topic in featured:  # heuristic mode: deterministic fallback only
            topic["HEADLINE"] = topic.get("title", topic.get("HEADLINE", ""))
            if not topic.get("KEY_TAKEAWAYS"):
                topic["KEY_TAKEAWAYS"] = _heuristic_takeaways(topic)

    return {
        "featured": featured,
        "watchlist": score.get("watchlist", []),
        "all_topics": score.get("all_topics", []),
        "manifest": {
            "synthesize_model": model,
            "synthesize_temperature": 0,
            "synthesize_rejected": rejected,
        },
    }
```

Add to registry:

```python
# arxiv_assistant/hotspots/kernel.py  (extend _STAGE_FNS with:)
    "synthesize": _stage_synthesize,
```

> Note: `_call_synthesize_agent` imports `hotspots.synthesize` lazily; that module is implemented as part of this task's production work or stubbed if synthesize lands separately. Tests never reach it because they patch `_call_synthesize_agent`. Provide a minimal `arxiv_assistant/hotspots/synthesize.py` with `def synthesize_bilingual(topics, *, model, temperature): raise NotImplementedError("wire claude -p headless")` so the import resolves in degraded/heuristic mode (where it is never called).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_kernel.py::TestSynthesizeVerifier -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add arxiv_assistant/hotspots/kernel.py arxiv_assistant/hotspots/synthesize.py \
        tests/test_kernel.py tests/fixtures/agent/synthesize_ok.json \
        tests/fixtures/agent/synthesize_halluc.json
git commit -m "feat(kernel): Synthesize touchpoint with temp-0 + schema/evidence-URL verifier"
```

---

## Task 7: render stage — assemble report dict + Resurgence lane + record_surface

**Files:**
- Modify: `arxiv_assistant/hotspots/kernel.py`
- Test: `tests/test_kernel.py`

The render stage assembles the legacy report dict shape (so `write_hotspot_web_data` / `render_hot_daily_md` keep working), computes the Resurgence lane from the Store via `resurge(...)`, calls `store.record_surface(..., lane="resurgence")` for each resurged story and `lane="featured"` for featured stories (single-writer), and writes the report + web_data + markdown exactly like the legacy tail (pipeline 1833-1838). When `store is None` (degraded), the Resurgence lane is empty and no surface is recorded.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kernel.py  (append)
class TestRenderStage(unittest.TestCase):
    def _config(self) -> configparser.ConfigParser:
        cfg = configparser.ConfigParser()
        cfg["HOTSPOTS"] = {
            "enabled": "true", "mode": "heuristic", "max_item_age_days": "14",
            "resurge_min_competitors": "3", "resurge_cooldown_days": "7",
        }
        return cfg

    def test_render_writes_report_with_resurgence_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            td = datetime(2026, 5, 20, tzinfo=timezone.utc)
            kernel._write_checkpoint(root, td, "harvest",
                                     {"items": [], "source_stats": {}, "api_usage": {}})
            kernel._write_checkpoint(root, td, "synthesize",
                                     {"featured": [], "watchlist": [], "all_topics": [],
                                      "manifest": {"synthesize_model": "m", "synthesize_temperature": 0,
                                                   "synthesize_rejected": []}})
            ctx = kernel.KernelContext(output_root=root, target_date=td,
                                       config=self._config(), store=None, journal=[])
            payload = kernel._stage_render(ctx)
        report_path = root / "hot" / "reports" / "2026-05-20.json"
        self.assertTrue(report_path.exists())
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertIn("resurgence", report)
        self.assertEqual(report["resurgence"], [])
        self.assertEqual(payload["report_path"], str(report_path))

    def test_resurgence_lane_built_from_store(self) -> None:
        class FakeStory:
            story_id = "s9"
            resurged_at = "2026-05-20"
            arxiv_versions = {"2301.00001": 3}
            surfaced_arxiv_versions = {"2301.00001": 2}
            verified_first_date = "2023-01-02T00:00:00Z"
            entity_names = {"FooNet"}

        class FakeStore:
            def __init__(self) -> None:
                self.surfaced: list = []

            def active_stories(self, window_days, as_of):
                return [FakeStory()]

            def record_surface(self, story, run_date, *, lane="featured"):
                self.surfaced.append((story.story_id, run_date, lane))

        store = FakeStore()
        with tempfile.TemporaryDirectory() as tmp, \
                unittest.mock.patch.object(kernel, "_resurge", return_value=True):
            root = Path(tmp)
            td = datetime(2026, 5, 20, tzinfo=timezone.utc)
            kernel._write_checkpoint(root, td, "harvest",
                                     {"items": [], "source_stats": {}, "api_usage": {}})
            kernel._write_checkpoint(root, td, "synthesize",
                                     {"featured": [], "watchlist": [], "all_topics": [],
                                      "manifest": {}})
            ctx = kernel.KernelContext(output_root=root, target_date=td,
                                       config=self._config(), store=store, journal=[])
            kernel._stage_render(ctx)
        report = json.loads((root / "hot" / "reports" / "2026-05-20.json").read_text(encoding="utf-8"))
        self.assertEqual(len(report["resurgence"]), 1)
        entry = report["resurgence"][0]
        self.assertEqual(entry["original_first_date"], "2023-01-02T00:00:00Z")
        self.assertEqual(entry["reason"], "arxiv_version_bump")
        self.assertIn(("s9", "2026-05-20", "resurgence"), store.surfaced)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_kernel.py::TestRenderStage -v`
Expected: FAIL — `_stage_render` / `_resurge` undefined.

- [ ] **Step 3: Write minimal implementation**

```python
# arxiv_assistant/hotspots/kernel.py  (append)
from datetime import UTC, datetime as _dt

from arxiv_assistant.hotspots.pipeline import (
    _build_category_sections, _build_long_tail_sections, _build_paper_spotlight,
    _build_market_signal_items, _fallback_digest_summary, build_hotspot_paths,
    ensure_parent_dirs, write_json, render_hot_daily_md, write_hotspot_web_data,
)


def _resurge(story, *, max_age_days, run_date, min_competitors, cooldown_days):
    """Thin wrapper over stage-2 novelty.resurge; degrades to False if absent."""
    try:
        from arxiv_assistant.hotspots.novelty import resurge
        return resurge(story, max_age_days=max_age_days, run_date=run_date,
                       min_competitors=min_competitors, cooldown_days=cooldown_days)
    except Exception:
        return False


def _resurge_reason(story) -> str:
    versions = getattr(story, "arxiv_versions", {}) or {}
    prior = getattr(story, "surfaced_arxiv_versions", {}) or {}
    for aid, count in versions.items():
        if count > prior.get(aid, 0):
            return "arxiv_version_bump"
    return "competitor_cluster"


def _build_resurgence_lane(ctx: KernelContext) -> list[dict[str, Any]]:
    store = ctx.store
    if store is None:
        return []
    cfg = ctx.config["HOTSPOTS"]
    max_age = cfg.getint("max_item_age_days", fallback=14)
    min_comp = cfg.getint("resurge_min_competitors", fallback=3)
    cooldown = cfg.getint("resurge_cooldown_days", fallback=7)
    as_of = ctx.target_date.astimezone(_tz.utc).date()
    window = cfg.getint("cross_day_window_days", fallback=14)
    lane: list[dict[str, Any]] = []
    for story in store.active_stories(window, as_of):
        if not _resurge(story, max_age_days=max_age, run_date=as_of,
                        min_competitors=min_comp, cooldown_days=cooldown):
            continue
        lane.append({
            "story_id": getattr(story, "story_id", ""),
            "original_first_date": getattr(story, "verified_first_date", None),
            "resurged_at": getattr(story, "resurged_at", None),
            "reason": _resurge_reason(story),
            "entities": sorted(getattr(story, "entity_names", set()) or set()),
        })
        store.record_surface(story, ctx.run_date, lane="resurgence")  # single writer
    return lane


def _stage_render(ctx: KernelContext) -> dict[str, Any]:
    harvest = ctx.read("harvest")
    synth = ctx.read("synthesize")
    cfg = ctx.config["HOTSPOTS"]
    raw_items = [_deserialize_item(r) for r in harvest.get("items", [])]
    featured = synth.get("featured", [])
    watchlist = synth.get("watchlist", [])
    all_topics = synth.get("all_topics", [])

    category_sections = _build_category_sections(
        all_topics, featured,
        target_total_topics=cfg.getint("target_category_topics", fallback=12),
        max_per_category=cfg.getint("max_topics_per_category", fallback=4),
        min_display_score=cfg.getfloat("category_display_score_cutoff", fallback=2.8),
    )
    long_tail_sections = _build_long_tail_sections(
        all_topics, featured, category_sections,
        target_total_topics=cfg.getint("target_long_tail_topics", fallback=18),
        max_per_category=cfg.getint("max_long_tail_per_category", fallback=8),
        min_display_score=cfg.getfloat("long_tail_display_score_cutoff", fallback=1.6),
    )
    x_buzz = _build_market_signal_items(raw_items, featured, watchlist)
    paper_spotlight = _build_paper_spotlight(
        raw_items,
        max_daily_hot=cfg.getint("paper_spotlight_max_daily_hot", fallback=6),
        max_new_frontier=cfg.getint("paper_spotlight_max_new_frontier", fallback=4),
    )
    resurgence = _build_resurgence_lane(ctx)

    report = {
        "date": ctx.run_date,
        "generated_at": _dt.now(UTC).isoformat(),
        "mode": cfg.get("mode", "heuristic"),
        "summary": _fallback_digest_summary(featured),
        "source_stats": harvest.get("source_stats", {}),
        "manifest": synth.get("manifest", {}),
        "top_topics": featured,
        "featured_topics": featured,
        "category_sections": category_sections,
        "long_tail_sections": long_tail_sections,
        "paper_spotlight": paper_spotlight,
        "x_buzz": x_buzz,
        "watchlist": watchlist,
        "resurgence": resurgence,
    }

    paths = build_hotspot_paths(ctx.output_root, ctx.target_date.date())
    ensure_parent_dirs(paths)
    write_json(paths.report_path, report)
    write_hotspot_web_data(ctx.output_root, report, raw_items)
    paths.markdown_path.write_text(render_hot_daily_md(report), encoding="utf-8")
    return {"report_path": str(paths.report_path), "resurgence_count": len(resurgence)}
```

Add to registry:

```python
# arxiv_assistant/hotspots/kernel.py  (extend _STAGE_FNS with:)
    "render": _stage_render,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_kernel.py::TestRenderStage -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add arxiv_assistant/hotspots/kernel.py tests/test_kernel.py
git commit -m "feat(kernel): render stage with Resurgence lane + single-writer record_surface"
```

---

## Task 8: Resurgence in web_data payload

**Files:**
- Modify: `arxiv_assistant/utils/hotspot/hotspot_web_data.py:926-1005`
- Test: `tests/test_kernel.py` (web payload assertion lives with kernel tests for locality)

`build_daily_hotspot_web_payload` must surface `resurgence` so the front-end and `_zh` translation pipeline can render it. Add a `resurgence` key (list of compact entries) and a `counts.resurgence` field. Entries carry both languages' display fields defaulted from English so the translation pass (`translate_hotspot_web_data.py`) fills `_zh` later; i18n stays in sync because the translator walks the same payload keys.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kernel.py  (append)
from arxiv_assistant.utils.hotspot.hotspot_web_data import build_daily_hotspot_web_payload


class TestResurgenceWebData(unittest.TestCase):
    def test_payload_carries_resurgence_section(self) -> None:
        report = {
            "date": "2026-05-20", "generated_at": "2026-05-20T00:00:00Z", "mode": "heuristic",
            "summary": "", "featured_topics": [], "category_sections": [],
            "long_tail_sections": [], "watchlist": [], "x_buzz": [], "paper_spotlight": [],
            "source_stats": {},
            "resurgence": [
                {"story_id": "s9", "original_first_date": "2023-01-02T00:00:00Z",
                 "resurged_at": "2026-05-20", "reason": "arxiv_version_bump",
                 "headline": "FooNet resurfaces with v3", "entities": ["FooNet"]},
            ],
        }
        payload = build_daily_hotspot_web_payload(report, [])
        self.assertEqual(len(payload["resurgence"]), 1)
        self.assertEqual(payload["resurgence"][0]["reason"], "arxiv_version_bump")
        self.assertEqual(payload["resurgence"][0]["original_first_date"], "2023-01-02T00:00:00Z")
        self.assertEqual(payload["meta"]["counts"]["resurgence"], 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_kernel.py::TestResurgenceWebData -v`
Expected: FAIL with `KeyError: 'resurgence'`.

- [ ] **Step 3: Write minimal implementation**

In `build_daily_hotspot_web_payload`, after the `watchlist_topics` block and before `payload = {`, add:

```python
    resurgence_entries = [
        {
            "story_id": str(entry.get("story_id", "")),
            "headline": str(entry.get("headline", "") or entry.get("HEADLINE", "")),
            "original_first_date": entry.get("original_first_date"),
            "resurged_at": entry.get("resurged_at"),
            "reason": str(entry.get("reason", "")),
            "entities": list(entry.get("entities", []) or []),
        }
        for entry in report.get("resurgence") or []
    ]
```

Inside `meta.counts`, add the count line:

```python
                "resurgence": len(resurgence_entries),
```

Inside the top-level `payload` dict (alongside `"x_buzz": list(...)`), add:

```python
        "resurgence": resurgence_entries,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_kernel.py::TestResurgenceWebData -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add arxiv_assistant/utils/hotspot/hotspot_web_data.py tests/test_kernel.py
git commit -m "feat(hotspot): emit Resurgence section into daily web payload (i18n-ready)"
```

---

## Task 9: Resurgence markdown rendering

**Files:**
- Modify: `arxiv_assistant/renderers/hotspot/render_hot_daily.py:99+`
- Test: `tests/test_kernel.py`

Add a `## Resurgence` section to the daily markdown, placed after `## Watchlist`. Each entry shows the headline, the **original v1 first date** (honesty), the resurge reason, and entities. The section is omitted when the lane is empty.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kernel.py  (append)
from arxiv_assistant.renderers.hotspot.render_hot_daily import render_hot_daily_md


class TestResurgenceMarkdown(unittest.TestCase):
    def _base_report(self) -> dict:
        return {
            "date": "2026-05-20", "summary": "", "source_stats": {},
            "featured_topics": [], "category_sections": [], "long_tail_sections": [],
            "watchlist": [], "x_buzz": [], "paper_spotlight": [],
        }

    def test_no_resurgence_section_when_empty(self) -> None:
        md = render_hot_daily_md({**self._base_report(), "resurgence": []})
        self.assertNotIn("## Resurgence", md)

    def test_resurgence_section_renders_origin_and_reason(self) -> None:
        report = {**self._base_report(), "resurgence": [
            {"headline": "FooNet resurfaces with v3",
             "original_first_date": "2023-01-02T00:00:00Z",
             "reason": "arxiv_version_bump", "entities": ["FooNet"]},
        ]}
        md = render_hot_daily_md(report)
        self.assertIn("## Resurgence", md)
        self.assertIn("FooNet resurfaces with v3", md)
        self.assertIn("2023-01-02", md)            # original first date shown honestly
        self.assertIn("arxiv_version_bump", md)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_kernel.py::TestResurgenceMarkdown -v`
Expected: FAIL — `## Resurgence` not present.

- [ ] **Step 3: Write minimal implementation**

In `render_hot_daily.py`, locate the end of the `## Watchlist` block in `render_hot_daily_md` (after the watchlist `for` loop, before the final `return "\n".join(lines)`). Insert:

```python
    resurgence = report.get("resurgence") or []
    if resurgence:
        lines.extend([
            "", "## Resurgence", "",
            "Older stories re-surfacing on observed signal (version bump or multi-competitor cluster). "
            "Original first-publication date shown for honesty; not part of the fresh NEW stream.", "",
        ])
        for entry in resurgence:
            headline = str(entry.get("headline", "") or "").strip() or "(untitled)"
            origin = str(entry.get("original_first_date", "") or "")[:10]
            reason = str(entry.get("reason", "") or "")
            entities = ", ".join(entry.get("entities", []) or [])
            lines.append(f"- **{headline}** — original first date: {origin or 'unknown'}; reason: {reason}")
            if entities:
                lines.append(f"  - entities: {entities}")
        lines.append("")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_kernel.py::TestResurgenceMarkdown -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add arxiv_assistant/renderers/hotspot/render_hot_daily.py tests/test_kernel.py
git commit -m "feat(render): add ## Resurgence markdown section with honest origin date"
```

---

## Task 10: Strangle `generate_daily_hotspot_report` → `kernel.run`

**Files:**
- Modify: `arxiv_assistant/hotspots/pipeline.py:1630-1839`
- Modify: `scripts/generate_daily_hotspots.py`
- Test: `tests/test_kernel.py`

Replace the body of `generate_daily_hotspot_report` with a thin delegation to `kernel.run`, then load and return the report dict the render stage wrote (preserving the existing return contract: the report dict or `None` when disabled). The legacy helper functions stay (the kernel imports them), so this is a control-flow strangle, not a rewrite (spec §F.2). Add a `--stage` passthrough to the CLI.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kernel.py  (append)
from arxiv_assistant.hotspots import pipeline as hp


class TestStrangler(unittest.TestCase):
    def test_generate_delegates_to_kernel_and_returns_report(self) -> None:
        cfg = configparser.ConfigParser()
        cfg["HOTSPOTS"] = {"enabled": "true", "mode": "heuristic"}
        captured = {}

        def fake_run(output_root, target_date, config, *, stage=None, force=False):
            captured["called"] = True
            report_dir = Path(output_root) / "hot" / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / "2026-05-20.json").write_text(
                json.dumps({"date": "2026-05-20", "resurgence": []}), encoding="utf-8")
            return {"date": "2026-05-20", "stages_run": kernel.STAGES}

        with tempfile.TemporaryDirectory() as tmp, \
                unittest.mock.patch.object(hp, "kernel_run", fake_run):
            report = hp.generate_daily_hotspot_report(
                output_root=tmp,
                target_date=datetime(2026, 5, 20, tzinfo=timezone.utc),
                config=cfg, mode_override="heuristic", force=False)
        self.assertTrue(captured["called"])
        self.assertEqual(report["date"], "2026-05-20")

    def test_generate_returns_none_when_disabled(self) -> None:
        cfg = configparser.ConfigParser()
        cfg["HOTSPOTS"] = {"enabled": "false"}
        with tempfile.TemporaryDirectory() as tmp:
            out = hp.generate_daily_hotspot_report(
                output_root=tmp, target_date=datetime(2026, 5, 20, tzinfo=timezone.utc),
                config=cfg)
        self.assertIsNone(out)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_kernel.py::TestStrangler -v`
Expected: FAIL — `hp.kernel_run` does not exist / old body still runs full pipeline.

- [ ] **Step 3: Write minimal implementation**

At the top of `pipeline.py` (near other imports), add a deferred import alias so tests can patch it without a circular import at module load:

```python
def kernel_run(*args, **kwargs):
    from arxiv_assistant.hotspots.kernel import run as _run
    return _run(*args, **kwargs)
```

Replace the entire body of `generate_daily_hotspot_report` (lines 1630-1839) with:

```python
def generate_daily_hotspot_report(output_root: str | Path, target_date: datetime, config: configparser.ConfigParser, mode_override: str = "auto", force: bool = False, stage: str | None = None) -> dict[str, Any] | None:
    if not config["HOTSPOTS"].getboolean("enabled", fallback=False):
        return None

    output_root = Path(output_root)
    # mode_override is honoured by writing it into the live config the kernel reads.
    if mode_override and mode_override != "auto":
        config["HOTSPOTS"]["mode"] = _decide_mode(mode_override)

    kernel_run(output_root, target_date, config, stage=stage, force=force)

    report_path = output_root / "hot" / "reports" / f"{date_string(target_date)}.json"
    if not report_path.exists():
        return None
    return json.loads(report_path.read_text(encoding="utf-8"))
```

In `scripts/generate_daily_hotspots.py`, add the `--stage` argument and pass it through:

```python
    parser.add_argument("--stage", default=None,
                        help="Run only this single kernel stage (resume helper).")
```

```python
    report = generate_daily_hotspot_report(
        output_root=args.output_root,
        target_date=parse_target_datetime(args.target_date, Path(args.output_root)),
        config=config,
        mode_override=args.mode,
        force=args.force,
        stage=args.stage,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_kernel.py::TestStrangler -v`
Expected: PASS (2 tests). Then run the legacy suite to confirm no regression in untouched helpers: `pytest tests/test_hotspot_pipeline.py -v`.
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add arxiv_assistant/hotspots/pipeline.py scripts/generate_daily_hotspots.py tests/test_kernel.py
git commit -m "refactor(hotspot): strangle generate_daily_hotspot_report into kernel.run"
```

---

## Task 11: Replay-diff bit-stability test

**Files:**
- Create: `tests/fixtures/replay/raw_2026-05-20.json`
- Create: `tests/test_replay_diff.py`

Spec §G.9 / overview §4 replay-diff gate: with a frozen historical raw input, two full `kernel.run` invocations on the same date must produce **bit-identical** report + score checkpoints (cache-hit / pure-deterministic path). We freeze the harvest input by patching `_fetch_source_payloads` to return the fixture, run in heuristic mode (no agents → fully deterministic), and compare the canonical JSON of the `score` checkpoint and the report file across two runs.

- [ ] **Step 1: Write the failing test + fixture**

Create `tests/fixtures/replay/raw_2026-05-20.json`:

```json
[
  {"source_id": "hf_papers", "source_name": "HF", "source_role": "papers", "source_type": "papers",
   "title": "Agentic coding model released", "summary": "A frontier lab ships an agent.",
   "url": "https://x/a", "canonical_url": "https://x/a", "published_at": "2026-05-20T01:00:00Z",
   "tags": [], "authors": [], "metadata": {}},
  {"source_id": "ainews", "source_name": "AINews", "source_role": "news", "source_type": "news",
   "title": "Agentic coding model released", "summary": "Recap of the agent release.",
   "url": "https://x/b", "canonical_url": "https://x/b", "published_at": "2026-05-20T02:00:00Z",
   "tags": [], "authors": [], "metadata": {}},
  {"source_id": "ainews", "source_name": "AINews", "source_role": "news", "source_type": "news",
   "title": "Unrelated GPU benchmark", "summary": "A benchmark post.",
   "url": "https://x/c", "canonical_url": "https://x/c", "published_at": "2026-05-20T03:00:00Z",
   "tags": [], "authors": [], "metadata": {}}
]
```

```python
# tests/test_replay_diff.py
from __future__ import annotations

import configparser
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from arxiv_assistant.hotspots import kernel
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem

FIXT = Path(__file__).resolve().parent / "fixtures" / "replay" / "raw_2026-05-20.json"


def _load_items() -> list[HotspotItem]:
    rows = json.loads(FIXT.read_text(encoding="utf-8"))
    return [kernel._deserialize_item(r) for r in rows]


def _config() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg["HOTSPOTS"] = {
        "enabled": "true", "mode": "heuristic", "max_raw_items": "120",
        "max_item_age_days": "14", "target_topics": "5", "target_watchlist_topics": "3",
        "max_topics_per_category": "4",
    }
    cfg["HOTSPOT_SOURCES"] = {}
    return cfg


class TestReplayDiff(unittest.TestCase):
    def _run_once(self, root: Path) -> tuple[str, str]:
        td = datetime(2026, 5, 20, tzinfo=timezone.utc)
        items = _load_items()
        with patch.object(kernel, "_fetch_source_payloads",
                          return_value=(items, {"hf_papers": 1, "ainews": 2}, {})):
            kernel.run(root, td, _config(), force=True)
        score = (root / "hot" / "state" / "checkpoint" / "2026-05-20" / "score.json").read_text("utf-8")
        report = json.loads((root / "hot" / "reports" / "2026-05-20.json").read_text("utf-8"))
        report.pop("generated_at", None)  # only non-deterministic field by design
        return score, json.dumps(report, sort_keys=True, ensure_ascii=False)

    def test_two_runs_are_bit_stable(self) -> None:
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            score_a, report_a = self._run_once(Path(a))
            score_b, report_b = self._run_once(Path(b))
        self.assertEqual(score_a, score_b)
        self.assertEqual(report_a, report_b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_replay_diff.py -v`
Expected: initially FAIL only if a non-determinism leak exists. If it fails, the message points to the diverging field; fix the offending stage (e.g. ensure checkpoints are written with `sort_keys=True` — Task 1 already does). The only sanctioned non-deterministic field is `generated_at`, which the test strips.

- [ ] **Step 3: Stabilize if needed**

If `score` differs, confirm `select_and_categorize` / `score_stories` order deterministically; if a set is serialized, sort it before writing. No new code expected if Tasks 1–7 followed the canonical-JSON discipline.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_replay_diff.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_replay_diff.py tests/fixtures/replay/raw_2026-05-20.json
git commit -m "test(kernel): replay-diff bit-stability gate over frozen raw input"
```

---

## Task 12: Single-source-failure degrade test

**Files:**
- Test: `tests/test_kernel.py`

Spec §E: a single source adapter failing must degrade (skipped, recorded partial) without crashing the run. `fetch_source_payloads` already catches per-adapter exceptions; this test asserts the Kernel still completes all stages when one adapter raises.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kernel.py  (append)
class TestSourceFailureDegrade(unittest.TestCase):
    def _config(self) -> configparser.ConfigParser:
        cfg = configparser.ConfigParser()
        cfg["HOTSPOTS"] = {"enabled": "true", "mode": "heuristic", "max_raw_items": "120",
                           "max_item_age_days": "14", "target_topics": "5",
                           "target_watchlist_topics": "3", "max_topics_per_category": "4"}
        return cfg

    def test_run_completes_when_one_source_payload_partial(self) -> None:
        ok = [_item("Live story", "https://x/a", "2026-05-20T00:00:00Z")]
        # Simulate fetch_source_payloads having already swallowed a failing adapter:
        # returns only the surviving items + a partial source_stats row of 0.
        with tempfile.TemporaryDirectory() as tmp, \
                unittest.mock.patch.object(
                    kernel, "_fetch_source_payloads",
                    return_value=(ok, {"hf_papers": 1, "reddit": 0}, {})):
            manifest = kernel.run(Path(tmp), datetime(2026, 5, 20, tzinfo=timezone.utc),
                                  self._config())
        self.assertEqual(manifest["stages_run"], kernel.STAGES)  # no crash, all stages done
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_kernel.py::TestSourceFailureDegrade -v`
Expected: PASS already if Tasks 4–7 are correct (degrade path is structural). If it fails, the failure localizes the missing degrade guard.

- [ ] **Step 3: Confirm / fix**

No new code expected. If a stage raises on empty input, wrap its store/agent call in `_with_retry(..., fallback=...)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_kernel.py::TestSourceFailureDegrade -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_kernel.py
git commit -m "test(kernel): single-source failure degrades without crashing the run"
```

---

## Task 13: VPS systemd unit + timer + headless entrypoint + audit snapshot

**Files:**
- Create: `deploy/vps/hotspot.service`
- Create: `deploy/vps/hotspot.timer`
- Create: `deploy/vps/hotspot.env.example`
- Create: `deploy/vps/run_hotspot.sh`

Spec §E runtime: VPS cron via systemd timer runs `claude -p` headless; secrets live in an `EnvironmentFile` (never in repo); after each run the text snapshot (including `date_verdicts`) is pushed to a dedicated audit branch. The entrypoint also calls `store.dump_text_snapshot` via a tiny inline Python call. No production secrets are committed — only `.env.example`.

- [ ] **Step 1: Create the systemd service unit**

`deploy/vps/hotspot.service`:

```ini
[Unit]
Description=Daily AI hotspot generation (agent-native Kernel, headless Claude Code)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=hotspot
WorkingDirectory=/opt/ChatGPT-ArXiv-Paper-Assistant
# Secrets (twitterapi.io key, LLM key, X token) live here, NOT in the repo.
EnvironmentFile=/etc/hotspot/hotspot.env
ExecStart=/opt/ChatGPT-ArXiv-Paper-Assistant/deploy/vps/run_hotspot.sh
# Bounded run; the Kernel itself is idempotent and resumable.
TimeoutStartSec=3600
Nice=10

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 2: Create the timer**

`deploy/vps/hotspot.timer`:

```ini
[Unit]
Description=Run daily AI hotspot generation at 05:10 UTC

[Timer]
OnCalendar=*-*-* 05:10:00 UTC
Persistent=true
# If the VPS was off at the scheduled time, run on next boot (idempotent resume).
AccuracySec=1min

[Install]
WantedBy=timers.target
```

- [ ] **Step 3: Create the EnvironmentFile template**

`deploy/vps/hotspot.env.example` (copy to `/etc/hotspot/hotspot.env`, `chmod 600`, fill real values; never commit the filled copy):

```bash
# Copy to /etc/hotspot/hotspot.env and fill in. DO NOT commit the filled file.
HOTSPOT_RUNTIME=local
OPENAI_API_KEY=replace-me
OPENAI_BASE_URL=https://api.openai.com/v1
TWITTERAPI_IO_KEY=replace-me
X_BEARER_TOKEN=replace-me
ANTHROPIC_API_KEY=replace-me
AUDIT_BRANCH=hotspot-audit
GIT_AUTHOR_NAME=hotspot-bot
GIT_AUTHOR_EMAIL=hotspot-bot@example.invalid
```

- [ ] **Step 4: Create the headless entrypoint**

`deploy/vps/run_hotspot.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO="/opt/ChatGPT-ArXiv-Paper-Assistant"
cd "$REPO"

TODAY="$(date -u +%F)"
OUT="$REPO/out"

# 1. Drive the fixed-DAG Kernel headlessly. Claude Code subagents (DateVerify/Synthesize)
#    are dispatched from inside the Kernel; this single invocation is idempotent and
#    resumes from the last incomplete (date,stage) checkpoint.
claude -p "Run the daily hotspot pipeline for ${TODAY} via the Kernel. \
Execute: python scripts/generate_daily_hotspots.py --output-root out --mode auto --date ${TODAY}. \
Do not improvise stage order; the Kernel owns the topology." \
  --dangerously-skip-permissions \
  || python scripts/generate_daily_hotspots.py --output-root out --mode auto --date "${TODAY}"

# 2. Dump the Store text snapshot (MUST include date_verdicts) for audit + reproducibility.
python - <<'PY'
from pathlib import Path
from arxiv_assistant.hotspots.store import StoryStore
db = Path("out/hot/state/story_store.sqlite")
if db.exists():
    store = StoryStore(db)
    out = store.dump_text_snapshot(Path("out/hot/state/snapshot"))
    print(f"snapshot written: {out}")
else:
    print("no story_store.sqlite yet; skipping snapshot")
PY

# 3. Push the text snapshot (incl. date_verdicts) to the audit branch. The binary SQLite
#    is NOT committed; only the schema-ized text snapshot travels.
AUDIT_BRANCH="${AUDIT_BRANCH:-hotspot-audit}"
if [ -d "out/hot/state/snapshot" ]; then
  git fetch origin "${AUDIT_BRANCH}:${AUDIT_BRANCH}" 2>/dev/null || git branch -f "${AUDIT_BRANCH}"
  git worktree add --force /tmp/hotspot-audit "${AUDIT_BRANCH}" 2>/dev/null || true
  rm -rf /tmp/hotspot-audit/snapshot
  cp -R out/hot/state/snapshot /tmp/hotspot-audit/snapshot
  ( cd /tmp/hotspot-audit
    git add snapshot
    git commit -m "audit: date_verdicts + story snapshot ${TODAY}" || echo "no snapshot changes"
    git push origin "${AUDIT_BRANCH}" )
  git worktree remove --force /tmp/hotspot-audit || true
fi

# 4. Publish generated web_data to the auto_update branch consumed by the Actions Publisher.
git add -A out
git commit -m "data: hotspot run ${TODAY}" || echo "no data changes"
git push origin HEAD:auto_update || echo "push to auto_update failed (will retry next run)"
```

Mark executable:

```bash
chmod +x deploy/vps/run_hotspot.sh
```

- [ ] **Step 5: Commit**

```bash
git add deploy/vps/hotspot.service deploy/vps/hotspot.timer \
        deploy/vps/hotspot.env.example deploy/vps/run_hotspot.sh
git commit -m "feat(runtime): VPS systemd unit/timer + headless entrypoint + audit-branch snapshot"
```

---

## Task 14: Demote `cron_runs.yaml` generation behind `runtime==actions`

**Files:**
- Modify: `.github/workflows/cron_runs.yaml`

Spec §E: VPS owns generation; Actions keeps only a **deterministic degraded fallback** path, gated so it does not double-generate when the VPS path is authoritative. We read `[HOTSPOT_RUNTIME] runtime` from config: when `actions`, the workflow runs the deterministic (agents-off / heuristic) generation; when `local` (default), the generation step is skipped and only the paper pipeline (`main.py`) plus artifact upload run.

- [ ] **Step 1: Add a runtime-detection step**

After the "Install dependencies" step in `cron_runs.yaml`, add:

```yaml
      - name: Detect hotspot runtime mode
        id: runtime
        shell: bash
        run: |
          MODE=$(python - <<'PY'
import configparser
c = configparser.ConfigParser()
c.read("configs/config.ini")
print(c.get("HOTSPOT_RUNTIME", "runtime", fallback="local"))
PY
)
          echo "mode=$MODE" >> "$GITHUB_OUTPUT"
          echo "Hotspot runtime mode: $MODE"
```

- [ ] **Step 2: Gate the generation step on `runtime==actions`**

Replace the existing "Generate daily hotspots" step with:

```yaml
      - name: Generate daily hotspots (deterministic Actions fallback)
        if: ${{ steps.runtime.outputs.mode == 'actions' }}
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          OPENAI_BASE_URL: ${{ secrets.OPENAI_BASE_URL }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          # Deterministic degraded path: heuristic mode, agents off, no twitterapi secret.
          python scripts/generate_daily_hotspots.py --output-root out --mode heuristic --force

      - name: Skip hotspot generation (VPS owns it)
        if: ${{ steps.runtime.outputs.mode != 'actions' }}
        run: echo "runtime=local — hotspot generation runs on VPS; Actions is Publisher-only."
```

- [ ] **Step 3: Verify YAML parses**

Run: `python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/cron_runs.yaml')); print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/cron_runs.yaml
git commit -m "ci(hotspot): gate Actions generation behind runtime==actions (deterministic degrade)"
```

---

## Task 15: Publisher `_zh` validation gate

**Files:**
- Modify: `.github/workflows/publish_md.yml`

Spec §E: Actions Publisher must validate translation assets (guard against the silent `_merge_zh` failure noted in the deploy/translation memory) and only then deploy day/month/year. Add a validation step after "Translate hotspot web data" that fails the build if any daily web_data payload that contains a `resurgence`/`featured_topics` section is missing its `_zh` companion or the `_zh` file lacks translated keys.

- [ ] **Step 1: Add the validation step**

After the "Translate hotspot web data" step in `publish_md.yml`, insert:

```yaml
      - name: Validate _zh translation assets
        shell: bash
        run: |
          python - <<'PY'
import json, sys
from pathlib import Path

web_root = Path("out/hot/web_data")
missing = []
empty = []
for daily in sorted(web_root.rglob("*.json")):
    if daily.name.endswith("_zh.json"):
        continue
    zh = daily.with_name(daily.stem + "_zh.json")
    try:
        data = json.loads(daily.read_text(encoding="utf-8"))
    except Exception:
        continue
    # Only daily payloads carry these sections; skip index/aggregate files.
    if not any(k in data for k in ("featured_topics", "resurgence")):
        continue
    if not zh.exists():
        missing.append(str(daily))
        continue
    zh_data = json.loads(zh.read_text(encoding="utf-8"))
    # Silent _merge_zh failure => _zh file exists but is byte-identical (no translation).
    if zh_data == data:
        empty.append(str(zh))

if missing or empty:
    print("Missing _zh companions:", *missing, sep="\n  ")
    print("Untranslated (identical) _zh files:", *empty, sep="\n  ")
    sys.exit(1)
print("All _zh translation assets present and non-trivial.")
PY
```

- [ ] **Step 2: Verify YAML parses**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/publish_md.yml')); print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/publish_md.yml
git commit -m "ci(publisher): fail build on missing or untranslated _zh assets"
```

---

## Task 16: Config additions for runtime + resurgence

**Files:**
- Modify: `configs/config.ini`
- Modify: `configs/templates/config.template.ini` (if present; otherwise the documented template referenced by overview §3)

Add the config keys this stage consumes (overview §3): `[HOTSPOT_RUNTIME] runtime`, and the resurgence/max-age keys the render + gravity stages read. All have safe defaults so existing configs keep working.

- [ ] **Step 1: Add the sections**

Append to `configs/config.ini` (and the template) — do not duplicate keys already added by earlier stages:

```ini
[HOTSPOT_RUNTIME]
runtime = local                         ; local|actions — local: VPS generates, Actions publishes only
```

Ensure these keys exist under `[HOTSPOTS]` (add only if missing):

```ini
max_item_age_days = 14
resurge_min_competitors = 3
resurge_cooldown_days = 7
```

- [ ] **Step 2: Verify config parses**

Run: `python -c "import configparser; c=configparser.ConfigParser(); c.read('configs/config.ini'); print(c.get('HOTSPOT_RUNTIME','runtime'))"`
Expected: `local`.

- [ ] **Step 3: Commit**

```bash
git add configs/config.ini configs/templates/config.template.ini
git commit -m "config(hotspot): add HOTSPOT_RUNTIME + resurgence/max-age keys with safe defaults"
```

---

## Task 17: Full-suite green + §G acceptance assertion

**Files:**
- Test: `tests/test_kernel.py`, `tests/test_replay_diff.py`

Final stop-the-line task (overview §0 / spec §G). Add one explicit INV6 acceptance test asserting the Synthesize manifest pins temperature 0 and records the model id, then run the whole hotspot test surface.

- [ ] **Step 1: Write the INV6 acceptance test**

```python
# tests/test_kernel.py  (append)
class TestInvariantsAcceptance(unittest.TestCase):
    def test_inv6_synthesize_pins_temp0_and_records_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            td = datetime(2026, 5, 20, tzinfo=timezone.utc)
            kernel._write_checkpoint(root, td, "score",
                                     {"featured": [], "watchlist": [], "all_topics": []})
            cfg = configparser.ConfigParser()
            cfg["HOTSPOTS"] = {"enabled": "true", "mode": "openai",
                               "model_synthesize": "pinned-model-v1"}
            ctx = kernel.KernelContext(output_root=root, target_date=td, config=cfg,
                                       store=None, journal=[])
            payload = kernel._stage_synthesize(ctx)
        self.assertEqual(payload["manifest"]["synthesize_temperature"], 0)
        self.assertEqual(payload["manifest"]["synthesize_model"], "pinned-model-v1")
```

- [ ] **Step 2: Run the full hotspot test surface**

Run: `pytest tests/test_kernel.py tests/test_replay_diff.py tests/test_hotspot_pipeline.py tests/test_hotspot_web_data.py -v`
Expected: PASS (all).

- [ ] **Step 3: Commit**

```bash
git add tests/test_kernel.py
git commit -m "test(kernel): INV6 acceptance — Synthesize pins temp0 + records model id"
```

---

## Self-Review notes (for the executor)

- **Single-writer rule (spec §A.1):** only `_stage_render` (via `_build_resurgence_lane`) and stage 3 `storystore_match` write the Store; never a subagent. Keep it that way when wiring the real stage-2 Store path.
- **Topology in code (spec §G.2):** `STAGES` + `_STAGE_FNS` are literal; the `claude -p` prompt in `run_hotspot.sh` explicitly forbids the agent from improvising stage order.
- **Degrade-everywhere (spec §E):** every dependency on a stage-0–5 module is wrapped so a missing module → deterministic fallback, preserving the "always emit a report" guarantee and the Actions degraded path.
- **Resurgence honesty (spec §C.4):** the rendered entry always shows `original_first_date` (v1) plus a machine reason; the lane is isolated from `featured_topics`, so the NEW-stream freshness guarantee is untouched.
- **Replay-diff (spec §G.9):** the only sanctioned non-deterministic field is `generated_at`, stripped before comparison; everything else is canonical-JSON (`sort_keys=True`).
