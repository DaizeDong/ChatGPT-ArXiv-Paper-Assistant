from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from arxiv_assistant.hotspots.pipeline import (
    _apply_freshness_gates,
    _serialize_items,
    date_string,
    fetch_source_payloads as _fetch_source_payloads,
)
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem

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


# ---------------------------------------------------------------------------
# Serialisation helpers (Task 4)
# ---------------------------------------------------------------------------

def _serialize_item(item: HotspotItem) -> dict[str, Any]:
    return _serialize_items([item])[0]


def _deserialize_item(row: dict[str, Any]) -> HotspotItem:
    # FIX 1: preserve verified_first_date + provenance so the date_verify →
    # gravity_gate handoff retains the verified date (INV1 round-trip integrity).
    return HotspotItem(
        source_id=row.get("source_id", ""),
        source_name=row.get("source_name", ""),
        source_role=row.get("source_role", ""),
        source_type=row.get("source_type", ""),
        title=row.get("title", ""),
        summary=row.get("summary", ""),
        url=row.get("url", ""),
        canonical_url=row.get("canonical_url", ""),
        published_at=row.get("published_at"),
        tags=row.get("tags", []) or [],
        authors=row.get("authors", []) or [],
        metadata=row.get("metadata", {}) or {},
        verified_first_date=row.get("verified_first_date"),
        provenance=row.get("provenance", "") or "",
    )


# ---------------------------------------------------------------------------
# Stage bodies — Task 4: harvest → date_verify → gravity_gate
# ---------------------------------------------------------------------------

def _stage_harvest(ctx: KernelContext) -> dict[str, Any]:
    items, source_stats, api_usage = _fetch_source_payloads(
        ctx.target_date, ctx.output_root, ctx.config, force=False
    )
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
        from arxiv_assistant.hotspots.date_verify import verify  # type: ignore[import]
        for it in items:
            verdict = _with_retry(
                lambda it=it: verify(it, ctx.store),
                attempts=3,
                base_delay=0.0,
                fallback=lambda it=it: {
                    "verified_first_date": it.published_at,
                    "confidence": 0.3,
                    "evidence": [],
                },
            )
            it.verified_first_date = verdict.get("verified_first_date") or it.published_at
    except Exception:
        for it in items:  # degraded: trust published_at as low-confidence fallback
            it.verified_first_date = it.published_at
    return {"items": [_serialize_item(it) for it in items]}


def _stage_gravity_gate(ctx: KernelContext) -> dict[str, Any]:
    # FIX 2: delegate to canonical _apply_freshness_gates (DRY, consistent with
    # prod Stage 1). Removes hand-rolled duplicate logic and preserves the
    # canonical None=keep policy (cannot-verify → do not drop).
    rows = ctx.read("date_verify")["items"]
    items = [_deserialize_item(r) for r in rows]
    max_age = ctx.config["HOTSPOTS"].getint("max_item_age_days", fallback=14)
    kept = _apply_freshness_gates(items, ctx.target_date, max_item_age_days=max_age)
    dropped = len(items) - len(kept)
    ctx.journal.append({"stage": "gravity_gate", "dropped_stale": dropped, "kept": len(kept)})
    return {"items": [_serialize_item(it) for it in kept]}


# ---------------------------------------------------------------------------
# Topology is hardcoded here (spec §G.2): never produced by an LLM.
# Real stage bodies are bound in Tasks 4-7; tests patch _STAGE_FNS.
# ---------------------------------------------------------------------------
_STAGE_FNS: dict[str, Callable[["KernelContext"], dict[str, Any]]] = {
    "harvest": _stage_harvest,
    "date_verify": _stage_date_verify,
    "gravity_gate": _stage_gravity_gate,
}


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
    try:
        for name in target_stages:
            if stage is None and _checkpoint_done(output_root, target_date, name):
                stages_skipped.append(name)
                continue
            fn = _STAGE_FNS[name]
            payload = fn(ctx)
            _write_checkpoint(output_root, target_date, name, payload)
            stages_run.append(name)
    finally:
        if store is not None and hasattr(store, "close"):
            store.close()

    return {
        "date": ctx.run_date,
        "stages_run": stages_run,
        "stages_skipped": stages_skipped,
        "journal": journal,
    }
