from __future__ import annotations

from datetime import date

from arxiv_assistant.utils.hotspot.gate_date import gate_date


def _max_tier(evidence: list) -> int:
    return max((int(getattr(e, "source_tier", 0)) for e in evidence), default=0)


def _max_gate_date(evidence: list, gate_date_fn) -> date | None:
    days = [gate_date_fn(e.item) for e in evidence]
    days = [d for d in days if d is not None]
    return max(days) if days else None


def resurface(story, *, gate_date_fn=gate_date) -> bool:
    """§C.3.1 closed-form resurface predicate: T1 ∨ T2 ∨ T3. Zero LLM.

    Reads ONLY Store-resident structured facts (source_tier ints, day-granular
    gate_date, arxiv version counts, entity_names). URL-set churn, sub-day jitter,
    same-tier evidence, and any free-text judgment are constructively excluded.
    """
    last = story.last_surfaced
    added = story.evidence_added_since(last)

    # T1: strictly higher source_tier than any evidence before last surface.
    # Requires new evidence items (no new items → no tier jump possible).
    if added:
        before_tier = max(_max_tier(story.evidence_before(last)), int(story.surfaced_max_tier or 0))
        if _max_tier(added) > before_tier:
            return True

        # T2a: a strictly later day-granular gate_date among newly-added evidence.
        new_gate = _max_gate_date(added, gate_date_fn)
        if new_gate is not None and story.surfaced_verified_max is not None:
            if new_gate > story.surfaced_verified_max:
                return True
        elif new_gate is not None and story.surfaced_verified_max is None:
            return True

    # T2b: a strictly increased arXiv version count vs the last-surface snapshot.
    # Story-level monotonic counter — independent of new evidence rows.
    for arxiv_id, count in (story.arxiv_versions or {}).items():
        prev = (story.surfaced_arxiv_versions or {}).get(arxiv_id, 0)
        if count > prev:
            return True

    # T3: a named entity not present at last surface.
    # Story-level set comparison — independent of new evidence rows.
    if (set(story.entity_names) - set(story.surfaced_entity_names or set())):
        return True

    return False


def _today_competitor_count(story) -> int:
    """Default: count distinct competitor (reuse-layer) sources among today's evidence.

    Reads the Stage-0 ledger fields: an evidence item added today whose provenance is
    a reuse-layer competitor source. Kernel may inject a precise fn; this default is a
    safe fallback for unit/offline use.
    """
    seen: set[str] = set()
    for e in getattr(story, "_today_evidence", []):
        prov = getattr(e.item, "provenance", "") or ""
        if prov.startswith("reuse:"):
            seen.add(prov)
    return len(seen)


def resurge(
    story,
    *,
    max_age_days: int,
    run_date: date,
    min_competitors: int,
    cooldown_days: int,
    gate_date_fn=gate_date,
    competitor_count_fn=_today_competitor_count,
) -> bool:
    """§C.4 resurgence predicate: R1 ∨ R2, evaluated ONLY for OLD stories. Zero LLM.

    Old := gate_date(story.canonical_item) is older than max_age_days vs run_date.
    R1: an arXiv version count strictly exceeds the last resurge-surface snapshot
        (each new version fires at most once).
    R2: >= min_competitors distinct competitor sources re-raise the item today AND
        the cooldown gate is open (surfaced_resurged_at is None or run_date is at
        least cooldown_days past it) — so the same competitor cluster re-raising the
        same old item day-after-day fires at most once per cooldown.
    """
    gd = gate_date_fn(story.canonical_item.item)
    if gd is None:
        return False
    if (run_date - gd).days <= max_age_days:
        return False  # not old → not a resurge candidate

    # R1: version jump strictly above the last resurge-surface snapshot.
    for arxiv_id, count in (story.arxiv_versions or {}).items():
        prev = (story.surfaced_arxiv_versions or {}).get(arxiv_id, 0)
        if count > prev:
            return True

    # R2: cross-competitor same-day cluster + cooldown gate.
    if competitor_count_fn(story) >= min_competitors:
        last_resurge = story.surfaced_resurged_at
        if last_resurge is None or (run_date - last_resurge).days >= cooldown_days:
            return True

    return False
