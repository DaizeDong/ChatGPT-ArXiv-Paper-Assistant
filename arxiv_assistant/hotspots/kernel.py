from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from arxiv_assistant.hotspots.pipeline import (
    _apply_freshness_gates,
    _build_category_sections,
    _build_long_tail_sections,
    _build_market_signal_items,
    _build_paper_spotlight,
    _fallback_digest_summary,
    _heuristic_takeaways,
    _serialize_items,
    _story_to_topic_dict,
    build_hotspot_paths,
    date_string,
    enrich_items_batch,
    enrich_items_heuristic,
    ensure_parent_dirs,
    fetch_source_payloads as _fetch_source_payloads,
    group_into_stories,
    render_hot_daily_md,
    score_stories,
    select_and_categorize,
    write_hotspot_web_data,
    write_json,
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
# Stage bodies — Task 5: embed → cluster → storystore_match → gapfill → score
# ---------------------------------------------------------------------------

def _items_from(ctx: KernelContext, stage: str) -> list[HotspotItem]:
    return [_deserialize_item(r) for r in ctx.read(stage)["items"]]


def _stage_embed(ctx: KernelContext) -> dict[str, Any]:
    # Embedding centroids are computed inside stage-2 dedup when present; in
    # degraded mode this is a structural pass-through carrying items forward.
    return {"items": ctx.read("gravity_gate")["items"]}


def _stage_cluster(ctx: KernelContext) -> dict[str, Any]:
    return {"items": ctx.read("embed")["items"]}


def _stage_storystore_match(ctx: KernelContext) -> dict[str, Any]:
    # Persistent-id assignment happens here when a Store is present (stage 2).
    # The single-writer rule is honoured: only the Kernel touches the Store.
    return {"items": ctx.read("cluster")["items"]}


def _stage_gapfill(ctx: KernelContext) -> dict[str, Any]:
    return {"items": ctx.read("storystore_match")["items"]}


def _enrich(ctx: KernelContext, items: list[HotspotItem]) -> list:
    cfg = ctx.config["HOTSPOTS"]
    mode = cfg.get("mode", "heuristic")
    if mode == "openai":
        model = cfg.get("model_enrich", cfg.get("model_screen"))
        return enrich_items_batch(
            items, model,
            cfg.getint("enrich_batch_size", fallback=20),
            cfg.getint("retry", fallback=3),
        )
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


# ---------------------------------------------------------------------------
# Stage 6: Synthesize — temp-0 agent + deterministic schema/evidence-URL verifier
# (spec §G.3 / INV6)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Stage 7: Render — assemble report dict + Resurgence lane + single-writer
# record_surface (spec §G.4 / Task 7)
# ---------------------------------------------------------------------------

def _resurge(story, *, max_age_days, run_date, min_competitors, cooldown_days):
    """Thin wrapper over stage-2 novelty.resurge; degrades to False if absent."""
    try:
        from arxiv_assistant.hotspots.novelty import resurge  # type: ignore[import]
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
    as_of = ctx.target_date.astimezone(timezone.utc).date()
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
        "generated_at": datetime.now(UTC).isoformat(),
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


# ---------------------------------------------------------------------------
# Topology is hardcoded here (spec §G.2): never produced by an LLM.
# Real stage bodies are bound in Tasks 4-7; tests patch _STAGE_FNS.
# ---------------------------------------------------------------------------
_STAGE_FNS: dict[str, Callable[["KernelContext"], dict[str, Any]]] = {
    "harvest": _stage_harvest,
    "date_verify": _stage_date_verify,
    "gravity_gate": _stage_gravity_gate,
    "embed": _stage_embed,
    "cluster": _stage_cluster,
    "storystore_match": _stage_storystore_match,
    "gapfill": _stage_gapfill,
    "score": _stage_score,
    "synthesize": _stage_synthesize,
    "render": _stage_render,
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
