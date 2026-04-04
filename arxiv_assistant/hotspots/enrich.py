from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import requests

from arxiv_assistant.utils.hotspot.hotspot_cluster import (
    SOURCE_ROLE_WEIGHTS,
    _extract_entities,
)
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem
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

def _chat_completion(model: str, messages: list[dict[str, str]], temperature: float = 0.1) -> dict[str, Any]:
    from arxiv_assistant.utils.local_env import load_local_env
    load_local_env()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "temperature": temperature, "messages": messages},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


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


def enrich_items_batch(
    items: list[HotspotItem],
    model: str,
    batch_size: int = 20,
    retry_count: int = 3,
) -> list[EnrichedItem]:
    """Enrich items using LLM batch processing with heuristic fallback."""
    prompt_template = read_prompt("hotspot.enrich")
    enriched: list[EnrichedItem] = []

    for batch_start in range(0, len(items), batch_size):
        batch = items[batch_start : batch_start + batch_size]
        items_text = _format_items_for_prompt(batch, offset=batch_start)
        user_prompt = prompt_template.replace("{items}", items_text)

        llm_results: list[dict[str, Any]] | None = None
        for _ in range(max(retry_count, 1)):
            try:
                data = _chat_completion(
                    model=model,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                raw_content = data["choices"][0]["message"]["content"] or "[]"
                llm_results = _parse_enrichment_response(raw_content)
                break
            except Exception as ex:
                print(f"Warning: enrichment batch failed, retrying: {ex}")

        if llm_results is None:
            print(f"Warning: enrichment batch {batch_start}-{batch_start + len(batch)} failed, using heuristic fallback")
            enriched.extend(enrich_items_heuristic(batch))
            # Fix batch_index to be global
            for ei in enriched[-len(batch):]:
                ei.batch_index += batch_start
            continue

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
                enriched.append(
                    EnrichedItem(
                        item=item,
                        event_type=_heuristic_event_type(item),
                        entities=_heuristic_entities(item),
                        summary=item.summary or item.title,
                        importance=_heuristic_importance(item),
                        batch_index=global_idx,
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

            enriched.append(
                EnrichedItem(
                    item=item,
                    event_type=event_type,
                    entities=raw_entities,
                    summary=str(row.get("summary", "")).strip() or item.summary or item.title,
                    importance=importance,
                    same_event_as=same_event_as,
                    batch_index=global_idx,
                )
            )

    return enriched
