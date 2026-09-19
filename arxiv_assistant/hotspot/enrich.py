from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from arxiv_assistant.utils import llm_gateway
from arxiv_assistant.hotspot.support.cluster import (
    SOURCE_ROLE_WEIGHTS,
    _extract_entities,
)
from arxiv_assistant.hotspot.support.schema import HotspotItem
from arxiv_assistant.utils.prompt_loader import read_prompt

EVENT_TYPES = {
    "product_release",
    "funding",
    "acquisition",
    "research_paper",
    "tooling",
    "industry_move",
    "opinion",
    "tutorial",
    "recap",
    "other",
}

_EVENT_TYPE_PATTERNS: dict[str, list[re.Pattern]] = {
    "product_release": [
        re.compile(r"\b(?:launch(?:es|ed)?|releas(?:es|ed)?|ship(?:s|ped)?|roll(?:s|ed)?\s*out)\b", re.I),
        re.compile(r"\b(?:v\d+(?:\.\d+)*|version\s+\d+|beta|preview|GA|general\s+availability)\b", re.I),
        re.compile(r"\b(?:new\s+(?:model|api|platform|sdk|feature|tool|service))\b", re.I),
        re.compile(r"\b(?:open[- ]?sourc(?:es|ed|ing))\b", re.I),
        re.compile(r"\b(?:announc(?:es|ed)?|introducing|unveil(?:s|ed)?)\b", re.I),
    ],
    "funding": [
        re.compile(r"\$\s*\d+(?:\.\d+)?\s*(?:million|billion|[mb])\b", re.I),
        re.compile(r"\b(?:series\s+[a-f]|seed\s+round|funding|fundraise|raise[ds]?)\b", re.I),
        re.compile(r"\b(?:valuation|valued\s+at|worth)\b", re.I),
    ],
    "acquisition": [
        re.compile(r"\b(?:acqui(?:re[ds]?|sition)|merg(?:er|es|ed)|buyout|buys?|bought)\b", re.I),
    ],
    "research_paper": [
        re.compile(r"\b(?:paper|arxiv|preprint)\b", re.I),
        re.compile(r"\b(?:state[- ]of[- ]the[- ]art|SOTA|outperforms?|surpass(?:es|ed)?)\b", re.I),
        re.compile(r"\b(?:novel\s+(?:architecture|method|approach|technique))\b", re.I),
    ],
    "tooling": [
        re.compile(r"\b(?:tool|sdk|framework|cli|workflow|library|package)\b", re.I),
        re.compile(r"\b(?:github|repository|repo)\b", re.I),
    ],
    "industry_move": [
        re.compile(r"\b(?:partnership|policy|regulation|infrastructure|chip|datacenter)\b", re.I),
        re.compile(r"\b(?:hire[ds]?|depart(?:s|ed)?|join(?:s|ed)?|appointment|CEO|CTO)\b", re.I),
    ],
    "opinion": [
        re.compile(r"\b(?:opinion|take|think|believe|argue|debate|controversial)\b", re.I),
        re.compile(r"\b(?:overrated|underrated|overhyped|is dead|is dying)\b", re.I),
    ],
    "tutorial": [
        re.compile(r"\b(?:tutorial|guide|how[- ]?to|walkthrough|step[- ]by[- ]step)\b", re.I),
        re.compile(r"\b(?:getting started|cookbook|recipe)\b", re.I),
    ],
    "recap": [
        re.compile(r"\b(?:weekly|roundup|digest|wrap|newsletter|top \d+)\b", re.I),
        re.compile(r"\b(?:summary|recap|highlights|overview)\b", re.I),
    ],
}

_ROLE_EVENT_HINTS = {
    "official_news": "product_release",
    "research_backbone": "research_paper",
    "paper_trending": "research_paper",
    "github_trend": "tooling",
}

EVENT_TYPE_TO_CATEGORY = {
    "product_release": "Product Release",
    "funding": "Market Signal",
    "acquisition": "Market Signal",
    "research_paper": "Research",
    "tooling": "Tooling",
    "industry_move": "Industry Update",
    "opinion": "Industry Update",
    "tutorial": "Industry Update",
    "recap": "Industry Update",
    "other": "Industry Update",
}


@dataclass
class EnrichedItem:
    item: HotspotItem
    event_type: str
    entities: list[dict[str, str]]
    summary: str
    importance: int  # 1-10
    same_event_as: int | None = None
    batch_index: int = 0
    #: WHICH PATH PRODUCED THIS ROW. "llm" only when a model actually answered
    #: for this item. Heuristic output is labelled by the reason it happened, so
    #: a run where every model call failed can never be mistaken for a run where
    #: the model ran and had little to say.
    enrich_source: str = "heuristic"

    @property
    def category(self) -> str:
        return EVENT_TYPE_TO_CATEGORY.get(self.event_type, "Industry Update")


def _heuristic_event_type(item: HotspotItem) -> str:
    text = f"{item.title} {item.summary or ''}"
    scores: dict[str, float] = {}
    for event_type, patterns in _EVENT_TYPE_PATTERNS.items():
        hits = sum(1 for p in patterns if p.search(text))
        scores[event_type] = hits / max(len(patterns) * 0.4, 1.0)

    role_hint = _ROLE_EVENT_HINTS.get(item.source_role)
    if role_hint:
        scores[role_hint] = scores.get(role_hint, 0) + 0.5

    if item.source_type == "paper" or item.metadata.get("arxiv_id"):
        scores["research_paper"] = scores.get("research_paper", 0) + 1.0

    best = max(scores, key=lambda k: scores[k])
    if scores[best] >= 0.3:
        return best
    return role_hint or "other"


def _heuristic_importance(item: HotspotItem) -> int:
    base = SOURCE_ROLE_WEIGHTS.get(item.source_role, 2.5)
    score = base * 1.2

    metadata = item.metadata or {}
    if metadata.get("is_official"):
        score += 1.5
    if metadata.get("github_url") or metadata.get("github_stars"):
        score += 0.8
    if int(metadata.get("upvotes", 0) or 0) > 50:
        score += 0.5
    if int(metadata.get("daily_score", 0) or 0) > 10:
        score += 0.5
    if int(metadata.get("hn_score", 0) or 0) > 50:
        score += 0.5

    return max(1, min(10, round(score)))


def _heuristic_entities(item: HotspotItem) -> list[dict[str, str]]:
    text = f"{item.title} {item.summary or ''}"
    raw_entities = _extract_entities(text)
    return [{"name": name, "type": "organization"} for name in sorted(raw_entities)]


def enrich_items_heuristic(items: list[HotspotItem]) -> list[EnrichedItem]:
    enriched: list[EnrichedItem] = []
    for i, item in enumerate(items):
        enriched.append(
            EnrichedItem(
                item=item,
                event_type=_heuristic_event_type(item),
                entities=_heuristic_entities(item),
                summary=item.summary or item.title,
                importance=_heuristic_importance(item),
                batch_index=i,
            )
        )
    return enriched


# ---------------------------------------------------------------------------
# LLM batch enrichment
# ---------------------------------------------------------------------------

#: Delimiter used to fold a system+user message pair into the ONE prompt string
#: the gateway takes. Explicit section banners, not a bare newline join, so the
#: model still sees where the standing instructions end and the payload begins.
_ROLE_BANNERS = {
    "system": "SYSTEM INSTRUCTIONS",
    "user": "USER",
    "assistant": "ASSISTANT",
}


def messages_to_prompt(messages: list[dict[str, str]]) -> str:
    """Flatten an OpenAI-style message list into one delimited prompt string.

    Instruction semantics are preserved: content is copied verbatim, in order,
    under a banner naming the role it came from.
    """
    parts: list[str] = []
    for message in messages or []:
        role = str(message.get("role", "user") or "user").strip().lower()
        content = str(message.get("content", "") or "")
        if not content.strip():
            continue
        banner = _ROLE_BANNERS.get(role, role.upper() or "USER")
        parts.append(f"===== {banner} =====\n{content}")
    return "\n\n".join(parts)


def _chat_completion(
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.1,
    *,
    config: Any = None,
    gateway_call: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """One model call, routed through :mod:`arxiv_assistant.utils.llm_gateway`."""
    caller = gateway_call or llm_gateway.call
    result = caller(
        messages_to_prompt(messages),
        config=config,
        timeout_s=llm_gateway.DEFAULT_TIMEOUT_S,
    )

    text = str(getattr(result, "text", "") or "")
    data = getattr(result, "data", None)
    if not text.strip() and data is not None:
        # A schema-capable backend may answer with parsed data and no raw text.
        text = json.dumps(data, ensure_ascii=False)

    return {
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "model": model,
        "_gateway": {
            "backend": str(getattr(result, "backend", "") or ""),
            "provider": str(getattr(result, "provider", "") or ""),
        },
    }


def _format_items_for_prompt(items: list[HotspotItem], offset: int = 0) -> str:
    lines: list[str] = []
    for i, item in enumerate(items):
        summary_snippet = (item.summary or "")[:200]
        lines.append(
            f"[{offset + i}] title: {item.title}\n"
            f"    summary: {summary_snippet}\n"
            f"    source: {item.source_name} ({item.source_role})"
        )
    return "\n\n".join(lines)


def _parse_enrichment_response(raw_text: str) -> list[dict[str, Any]]:
    cleaned = re.sub(r"```json\s*", "", raw_text or "")
    cleaned = re.sub(r"```", "", cleaned).strip()
    return json.loads(cleaned)


@dataclass
class EnrichmentStatus:
    """Which path actually produced the enrichment, in numbers.

    Exists because a total model outage and a quiet news day used to look
    identical downstream: both ended as heuristic rows with no marker on them.
    ``path`` is the one-word answer; the counts are the evidence.
    """

    mode_requested: str = "heuristic"
    batches: int = 0
    batches_llm: int = 0
    batches_failed: int = 0
    items: int = 0
    items_llm: int = 0
    items_heuristic: int = 0
    backend: str = ""
    provider: str = ""
    errors: list[str] = field(default_factory=list)

    MAX_ERRORS = 10

    def record_error(self, message: str) -> None:
        if len(self.errors) < self.MAX_ERRORS:
            self.errors.append(str(message)[:300])

    @property
    def path(self) -> str:
        if self.mode_requested == "heuristic":
            return "heuristic"
        if self.items == 0:
            return "empty"
        if self.items_llm == 0:
            return "heuristic_fallback"  # LLM was asked for and answered nothing
        if self.items_heuristic:
            return "mixed"
        return "llm"

    @property
    def llm_ok(self) -> bool:
        return self.mode_requested != "heuristic" and self.items_llm > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "mode_requested": self.mode_requested,
            "llm_ok": self.llm_ok,
            "batches": self.batches,
            "batches_llm": self.batches_llm,
            "batches_failed": self.batches_failed,
            "items": self.items,
            "items_llm": self.items_llm,
            "items_heuristic": self.items_heuristic,
            "backend": self.backend,
            "provider": self.provider,
            "errors": list(self.errors),
        }


def heuristic_status(items: list[EnrichedItem]) -> EnrichmentStatus:
    """Status for a run that never intended to call a model."""
    return EnrichmentStatus(
        mode_requested="heuristic",
        items=len(items),
        items_heuristic=len(items),
    )


def enrich_items_batch(
    items: list[HotspotItem],
    model: str,
    batch_size: int = 20,
    retry_count: int = 3,
    *,
    config: Any = None,
    gateway_call: Callable[..., Any] | None = None,
) -> list[EnrichedItem]:
    """Enrich items using LLM batch processing with heuristic fallback.

    Unchanged return contract (a plain list). Callers that need to know WHICH
    path produced the list call :func:`enrich_items_batch_with_status`.
    """
    enriched, _status = enrich_items_batch_with_status(
        items, model, batch_size, retry_count,
        config=config, gateway_call=gateway_call,
    )
    return enriched


def enrich_items_batch_with_status(
    items: list[HotspotItem],
    model: str,
    batch_size: int = 20,
    retry_count: int = 3,
    *,
    config: Any = None,
    gateway_call: Callable[..., Any] | None = None,
) -> tuple[list[EnrichedItem], EnrichmentStatus]:
    """Same work as :func:`enrich_items_batch`, plus the honesty record.

    The heuristic fallback is kept exactly as it was -- it is the honest degrade
    -- but every fallback is now counted and labelled, so "the model produced
    this" and "the model never answered" are different observable outputs.
    """
    prompt_template = read_prompt("hotspot.enrich")
    enriched: list[EnrichedItem] = []
    status = EnrichmentStatus(mode_requested="llm")

    # Sort items by primary entity so related items land in the same batch,
    # maximizing LLM same_event_as cross-reference coverage.
    def _entity_sort_key(item: HotspotItem) -> str:
        entities = _heuristic_entities(item)
        return entities[0]["name"].lower() if entities else item.source_name.lower()

    sorted_items = sorted(items, key=_entity_sort_key)

    for batch_start in range(0, len(sorted_items), batch_size):
        batch = sorted_items[batch_start : batch_start + batch_size]
        items_text = _format_items_for_prompt(batch, offset=batch_start)
        user_prompt = prompt_template.replace("{items}", items_text)
        status.batches += 1

        llm_results: list[dict[str, Any]] | None = None
        for _ in range(max(retry_count, 1)):
            try:
                data = _chat_completion(
                    model=model,
                    messages=[{"role": "user", "content": user_prompt}],
                    config=config,
                    gateway_call=gateway_call,
                )
                raw_content = data["choices"][0]["message"]["content"] or "[]"
                llm_results = _parse_enrichment_response(raw_content)
                gateway_meta = data.get("_gateway") or {}
                status.backend = status.backend or str(gateway_meta.get("backend", "") or "")
                status.provider = status.provider or str(gateway_meta.get("provider", "") or "")
                break
            except Exception as ex:
                # Not swallowed: the reason is recorded on the status object the
                # caller writes into the report.
                status.record_error(f"batch {batch_start}: {ex}")
                print(f"Warning: enrichment batch failed, retrying: {ex}")

        if llm_results is None:
            status.batches_failed += 1
            print(f"Warning: enrichment batch {batch_start}-{batch_start + len(batch)} failed, using heuristic fallback")
            enriched.extend(enrich_items_heuristic(batch))
            # Fix batch_index to be global
            for ei in enriched[-len(batch):]:
                ei.batch_index += batch_start
                ei.enrich_source = "heuristic_batch_failed"
                status.items += 1
                status.items_heuristic += 1
            continue

        status.batches_llm += 1

        # Build index lookup from LLM response
        result_by_index: dict[int, dict[str, Any]] = {}
        for row in llm_results:
            idx = row.get("index")
            if idx is not None:
                result_by_index[int(idx)] = row

        for i, item in enumerate(batch):
            global_idx = batch_start + i
            row = result_by_index.get(global_idx)
            if row is None:
                # Fallback for missing items
                status.items += 1
                status.items_heuristic += 1
                enriched.append(
                    EnrichedItem(
                        item=item,
                        event_type=_heuristic_event_type(item),
                        entities=_heuristic_entities(item),
                        summary=item.summary or item.title,
                        importance=_heuristic_importance(item),
                        batch_index=global_idx,
                        enrich_source="heuristic_row_missing",
                    )
                )
                continue

            event_type = str(row.get("event_type", "other")).strip()
            if event_type not in EVENT_TYPES:
                event_type = _heuristic_event_type(item)

            raw_entities = row.get("entities", [])
            if not isinstance(raw_entities, list):
                raw_entities = []
            # Supplement with heuristic entities (regex-based) for robust grouping
            heuristic_names = {e["name"].lower() for e in _heuristic_entities(item)}
            llm_names = {e["name"].lower() for e in raw_entities if isinstance(e, dict)}
            for name in heuristic_names - llm_names:
                raw_entities.append({"name": name, "type": "organization"})

            importance = int(row.get("importance", 5) or 5)
            importance = max(1, min(10, importance))

            same_event_as = row.get("same_event_as")
            if same_event_as is not None:
                same_event_as = int(same_event_as)

            status.items += 1
            status.items_llm += 1
            enriched.append(
                EnrichedItem(
                    item=item,
                    event_type=event_type,
                    entities=raw_entities,
                    summary=str(row.get("summary", "")).strip() or item.summary or item.title,
                    importance=importance,
                    same_event_as=same_event_as,
                    batch_index=global_idx,
                    enrich_source="llm",
                )
            )

    return enriched, status
