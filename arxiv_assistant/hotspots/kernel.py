from __future__ import annotations

import json
import shutil
import time
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
