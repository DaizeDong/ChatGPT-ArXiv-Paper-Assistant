from __future__ import annotations

import json
import os as _os
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

    # Pre-filter 1 (port pipeline 1700-1704): drop title/URL mismatches.
    try:
        from arxiv_assistant.utils.hotspot.hotspot_sources import url_title_consistent
        items = [it for it in items if url_title_consistent(it.title, it.url)]
    except Exception:
        pass  # degrade-safe: skip filter if import fails

    # Pre-filter 2 (port pipeline 1707-1710): drop items with no published_at.
    items = [it for it in items if it.published_at]

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
    """Structural item-carrying stage (pass-through).

    The real Stage-2 cluster/match/gapfill work is consolidated in _stage_score
    (Stories held in-memory there to avoid Story JSON serialization across
    checkpoints); this stage remains a structural item-carrying stage that
    forwards gravity-gated HotspotItem dicts to the cluster stage.
    """
    return {"items": ctx.read("gravity_gate")["items"]}


def _stage_cluster(ctx: KernelContext) -> dict[str, Any]:
    """Structural item-carrying stage (pass-through).

    The real Stage-2 cluster/match/gapfill work is consolidated in _stage_score
    (Stories held in-memory there to avoid Story JSON serialization across
    checkpoints); this stage remains a structural item-carrying stage that
    forwards HotspotItem dicts to the storystore_match stage.
    """
    return {"items": ctx.read("embed")["items"]}


def _stage_storystore_match(ctx: KernelContext) -> dict[str, Any]:
    """Structural item-carrying stage (pass-through).

    The real Stage-2 cluster/match/gapfill work is consolidated in _stage_score
    (Stories held in-memory there to avoid Story JSON serialization across
    checkpoints); persistent-id assignment and cross-day matching happen inside
    _stage_score where Story objects are kept in-memory. The single-writer rule
    is honoured: only the Kernel touches the Store (inside _stage_score).
    """
    return {"items": ctx.read("cluster")["items"]}


def _stage_gapfill(ctx: KernelContext) -> dict[str, Any]:
    """Structural item-carrying stage (pass-through).

    The real Stage-2 cluster/match/gapfill work is consolidated in _stage_score
    (Stories held in-memory there to avoid Story JSON serialization across
    checkpoints); this stage carries gravity-gated HotspotItem dicts forward
    unchanged so _stage_score can read them from the gapfill checkpoint.
    """
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
    """Real Stage-2 cross-day dedup path (port of pipeline.py:1728-1825).

    Reads gravity-gated HotspotItems from the gapfill checkpoint, runs the
    full cross-day dedup pipeline (intraday cluster → cross-day match →
    ONGOING suppression → quality filter → select), and writes
    record_surface for featured stories (single-writer rule).

    Story objects are held in-memory here to avoid Story JSON serialization
    across checkpoints (the embed/cluster/storystore_match/gapfill stages
    remain structural pass-throughs carrying HotspotItem dicts only).

    Degrade boundary is NARROW (FIX 3): the naive group_into_stories +
    score_stories fallback fires ONLY when the dedup stack cannot be imported
    (ImportError). A genuine bug inside match_crossday/classify/etc. PROPAGATES
    rather than silently degrading cross-day dedup to naive — the very
    regression this stage exists to prevent must never become invisible.
    """
    cfg = ctx.config["HOTSPOTS"]
    items = _items_from(ctx, "gapfill")

    # Deterministic run day (spec §G.9 / INV2): score_stories' freshness/gravity
    # term must be measured against the frozen target date, NOT wall-clock now().
    # Otherwise two replays of the same raw input that straddle a UTC day boundary
    # produce different scores → different select_and_categorize output → the
    # replay-diff bit-stability gate flakes. Thread the target run day through.
    run_day = ctx.target_date.astimezone(timezone.utc).date()

    try:
        from arxiv_assistant.hotspots.dedup import (  # type: ignore[import]
            classify_cross_day,
            cluster_intraday,
            match_crossday,
        )
        from arxiv_assistant.hotspots.pipeline import (
            _is_low_quality_story,
            _load_recent_featured_urls,
        )
        from arxiv_assistant.hotspots.story import apply_cross_day_penalty
    except ImportError:
        # Naive fallback ONLY when the dedup stack is unavailable (FIX 3).
        enriched = _enrich(ctx, items)
        stories = score_stories(group_into_stories(enriched), run_date=run_day)
        featured_stories, watchlist_stories, _ = select_and_categorize(
            stories,
            target_featured=cfg.getint("target_topics", fallback=5),
            target_watchlist=cfg.getint("target_watchlist_topics", fallback=3),
            max_per_category=cfg.getint("max_topics_per_category", fallback=4),
        )
        return {
            "featured": [_story_to_topic_dict(s, keep=True) for s in featured_stories],
            "watchlist": [_story_to_topic_dict(s, watchlist=True) for s in watchlist_stories],
            "all_topics": [_story_to_topic_dict(s) for s in stories],
        }

    # --- Real Stage-2 path (errors here PROPAGATE; no silent degrade) ---

    # Step 1: Cross-day URL dedup (pipeline 1728-1738)
    recent_urls, recent_headlines = _load_recent_featured_urls(
        ctx.output_root, ctx.target_date
    )
    if recent_urls:
        items = [
            it for it in items
            if (it.canonical_url or "") not in recent_urls
            and (it.url or "") not in recent_urls
        ]

    # Step 2: Enrich — SINGLE enrich, post-dedup (FIX 2; pipeline 1746-1750)
    enriched = _enrich(ctx, items)

    # Step 3: Intraday cluster
    l1_threshold = cfg.getfloat("cross_day_cosine_threshold", fallback=0.72)
    intraday = cluster_intraday(enriched, threshold=l1_threshold)

    store = ctx.store
    if store is not None:
        # Step 4 (store present): cross-day match + classify + suppress
        crossday_threshold = cfg.getfloat("cross_day_cosine_threshold", fallback=0.90)
        cross_day_window = cfg.getint("cross_day_window_days", fallback=14)
        as_of = ctx.target_date.astimezone(timezone.utc).date()
        matched = match_crossday(
            intraday, store,
            cosine_threshold=crossday_threshold,
            window_days=cross_day_window,
            as_of=as_of,
        )
        for s in matched:
            s.cross_day_status = classify_cross_day(s)
        stories = score_stories(matched, run_date=run_day)
        featured_eligible_scored = [s for s in stories if s.cross_day_status != "ONGOING"]
    else:
        # Step 4 (degraded, store=None): headline penalty fallback
        stories = score_stories(intraday, run_date=run_day)
        if recent_headlines:
            stories = apply_cross_day_penalty(stories, recent_headlines)
        featured_eligible_scored = stories

    # Step 5: Quality filter (pipeline 1794-1799)
    stories = [s for s in stories if not _is_low_quality_story(s)]
    featured_eligible_scored = [
        s for s in featured_eligible_scored if not _is_low_quality_story(s)
    ]

    # Step 6: Select & categorize (pipeline 1806-1811)
    featured_stories, watchlist_stories, _ = select_and_categorize(
        featured_eligible_scored,
        target_featured=cfg.getint("target_topics", fallback=5),
        target_watchlist=cfg.getint("target_watchlist_topics", fallback=3),
        max_per_category=cfg.getint("max_topics_per_category", fallback=4),
    )

    # Step 7: record_surface — SINGLE-WRITER (pipeline 1817-1820)
    if store is not None:
        for s in featured_stories:
            store.record_surface(s, ctx.run_date, lane="featured")

    # Step 8: Return topic dicts
    return {
        "featured": [_story_to_topic_dict(s, keep=True) for s in featured_stories],
        "watchlist": [_story_to_topic_dict(s, watchlist=True) for s in watchlist_stories],
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
    watchlist = list(synth.get("watchlist", []))  # mutable copy for FIX C
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

    # FIX C: Waterfall watchlist dedup (legacy pipeline.py:1868-1873).
    # Collect all TOPIC_IDs already claimed by featured + category + long-tail
    # sections, then filter watchlist to drop any entry whose TOPIC_ID is
    # already present in that set (a topic can't appear in both featured and
    # watchlist).
    claimed_ids: set[str] = {t["TOPIC_ID"] for t in featured}
    for section in category_sections:
        claimed_ids.update(t["TOPIC_ID"] for t in section.get("topics", []))
    for section in long_tail_sections:
        claimed_ids.update(t["TOPIC_ID"] for t in section.get("topics", []))
    watchlist = [t for t in watchlist if t["TOPIC_ID"] not in claimed_ids]

    x_buzz = _build_market_signal_items(raw_items, featured, watchlist)
    paper_spotlight = _build_paper_spotlight(
        raw_items,
        max_daily_hot=cfg.getint("paper_spotlight_max_daily_hot", fallback=6),
        max_new_frontier=cfg.getint("paper_spotlight_max_new_frontier", fallback=4),
        use_s2_signal=cfg.getboolean("use_semantic_scholar_signal", fallback=True),
        s2_api_key=(cfg.get("semantic_scholar_api_key", fallback="") or _os.getenv("S2_API_KEY") or None),
    )
    resurgence = _build_resurgence_lane(ctx)

    # FIX B: Assemble totals / usage / costs (legacy pipeline.py:1885-1921).
    # totals: real counts from available kernel data; sub-counts not available
    # from the render stage (enriched_items, stories) are omitted / set to 0
    # as they are not tracked across checkpoints.
    totals = {
        "raw_items": len(raw_items),
        "enriched_items": 0,   # not tracked across checkpoints; consumers use raw_items
        "stories": 0,           # not tracked; use all_topics count as proxy
        "featured": len(featured),
        "watchlist": len(watchlist),
        "category_topics": sum(len(s.get("topics", [])) for s in category_sections),
        "long_tail_topics": sum(len(s.get("topics", [])) for s in long_tail_sections),
        "paper_spotlight_items": sum(len(s.get("items", [])) for s in paper_spotlight),
    }
    # costs: zero because LLM token/cost tracking is not threaded through kernel
    # checkpoints (do NOT fabricate values).
    costs = {"prompt": 0.0, "completion": 0.0, "total": 0.0}
    # usage: replicate legacy _build_usage_payload structure (pipeline.py:1279-1330).
    # LLM sub-dict uses real mode; all token/cost fields are zero (not tracked).
    # External sub-dict is built from harvest's api_usage (the real source data).
    api_usage: dict[str, Any] = harvest.get("api_usage") or {}
    mode = cfg.get("mode", "heuristic")
    llm_row: dict[str, Any] = {
        "provider": "OpenAI",
        "billing_model": "quota" if mode == "openai" else "disabled",
        "screen_model": cfg.get("model_screen", None),
        "summary_model": cfg.get("model_summarize", cfg.get("model_screen", None)),
        "requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "prompt_cost": 0.0,
        "completion_cost": 0.0,
        "total_cost": 0.0,
    }
    external_rows: dict[str, Any] = {}
    external_request_total = 0
    x_request_total = 0
    estimated_external_cost = 0.0
    for source_id, row in api_usage.items():
        requests_count = int(row.get("requests", 0) or 0)
        estimated_cost = round(float(row.get("estimated_cost", 0.0) or 0.0), 6)
        external_rows[source_id] = {
            "provider": str(row.get("provider", source_id)),
            "billing_model": str(row.get("billing_model", "unknown")),
            "requests": requests_count,
            "items": int(row.get("items", 0) or 0),
            "estimated_cost": estimated_cost,
            "cache_hit": bool(row.get("cache_hit", False)),
        }
        external_request_total += requests_count
        estimated_external_cost += estimated_cost
        if source_id.startswith("x_"):
            x_request_total += requests_count
    usage: dict[str, Any] = {
        "llm": llm_row,
        "external": external_rows,
        "summary": {
            "external_requests": external_request_total,
            "x_requests": x_request_total,
            "estimated_external_cost": round(estimated_external_cost, 6),
        },
    }

    report = {
        "date": ctx.run_date,
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": mode,
        "summary": _fallback_digest_summary(featured),
        "source_stats": harvest.get("source_stats", {}),
        "manifest": synth.get("manifest", {}),
        "totals": totals,    # FIX B
        "costs": costs,      # FIX B
        "usage": usage,      # FIX B
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
    # FIX A: write normalized items file (legacy pipeline.py:1917).
    # scripts/rebuild_hotspot_web_data.py:120-123 requires out/hot/normalized/<date>.json.
    write_json(paths.normalized_path, _serialize_items(raw_items))
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
