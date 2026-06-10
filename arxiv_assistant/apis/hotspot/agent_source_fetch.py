"""Per-source agent fetcher -- agent-only gathering proof-of-concept.

Instead of a brittle Python scraper per source, point a ``claude -p`` agent at the
SAME curated source URL (a lab blog index, an RSS/listing page, a subreddit, the
HF papers page, a GitHub releases page, ...) and have it WebFetch the page and
extract the recent AI/ML item permalinks. Same coverage as the deterministic
scraper, no source-specific parsing code, resilient to site redesigns.

Anti-hallucination reuses the scout's DETERMINISTIC URL verifier (INV6): every
emitted item must have a syntactically valid, non-blocklisted, live http(s) URL,
and must be an item PERMALINK (not the index page itself). Degrades to ``[]`` on
any agent/parse/mapping failure -- it never raises out of ``fetch_source_via_agent``.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Optional

from arxiv_assistant.apis.hotspot.hotspot_agent_scout import (
    _SCHEMA,
    _default_url_alive,
    _url_is_acceptable,
)
from arxiv_assistant.utils.agent_runner import AgentError, run_agent
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem, clean_text, normalize_url
from arxiv_assistant.utils.hotspot.hotspot_sources import clip_text


def _build_prompt(name: str, url: str, kind: str, target_date: datetime, freshness_hours: int, result_limit: int) -> str:
    """Prompt the agent to FETCH one specific source page and extract recent item permalinks."""
    as_of = target_date.strftime("%Y-%m-%d")
    body = (
        "You are an AI/ML source extractor. Use the WebFetch tool to fetch this exact page:\n"
        "    {url}\n"
        "It is the '{name}' source (a {kind} listing / index / feed page). From it, extract the "
        "AI/ML items that were POSTED in the LAST {hours} HOURS as of {as_of} (UTC). Cover only "
        "ARTIFICIAL-INTELLIGENCE / MACHINE-LEARNING items; ignore unrelated entries.\n\n"
        "For EACH item return an object with keys:\n"
        '  - "title": the item headline.\n'
        '  - "url": the CANONICAL ITEM PERMALINK -- the specific post/paper/release URL on this '
        "source (e.g. the individual blog-post page, the arXiv abstract page, the GitHub release "
        "tag). It MUST be a specific item page, NOT the index/listing page above, and NOT a "
        "search-results or aggregator link.\n"
        '  - "summary": one short sentence on what it is.\n'
        '  - "published_at": the post date/time if shown (ISO 8601), else omit.\n\n'
        "If the page is JavaScript-walled, empty, or unreachable, you MAY use the WebSearch tool "
        "to find this same source's recent item permalinks instead. If you cannot confirm an item "
        "URL is real and reachable, OMIT that item.\n\n"
        "Write all titles and summaries in ENGLISH; do not narrate or reason in another language.\n\n"
        "OUTPUT CONTRACT (critical): Respond with ONLY a single raw JSON object of the form "
        '{{"items": [ {{...}}, {{...}} ]}}, starting with \'{{\' and ending with \'}}\', at most '
        "{limit} items. Do NOT wrap it in a markdown code fence (no triple backticks), and do NOT "
        "write any prose, preamble, reasoning, or commentary before or after the JSON."
    )
    return body.format(
        url=url, name=name, kind=(kind or "source"), hours=int(freshness_hours),
        as_of=as_of, limit=int(result_limit),
    )


def _item_to_hotspot(raw: dict[str, Any], *, name: str, kind: str, origin_url: str) -> HotspotItem:
    """Map one agent item dict to a HotspotItem. Raises on malformed input so the
    caller skips just this item."""
    title = clean_text(raw.get("title"))
    url = clean_text(raw.get("url"))
    if not title or not url:
        raise ValueError("agent-source item missing title/url")

    source_id = "agent_src:{0}".format(name)
    provenance = "agent:source:{0}".format(name)
    item = HotspotItem(
        source_id=source_id,
        source_name=name,
        source_role="agent_source",
        source_type=kind or "news",
        title=title,
        summary=clip_text(raw.get("summary"), 420),
        url=url,
        canonical_url=url,
        published_at=clean_text(raw.get("published_at")) or None,
        tags=["agent-source", name],
        authors=[],
        metadata={
            "source_kind": kind or "news",
            "origin_url": origin_url,
        },
    )
    if hasattr(item, "provenance"):
        item.provenance = provenance
    item.metadata.setdefault("provenance", provenance)
    item.metadata.setdefault("source_id", source_id)
    return item


def fetch_source_via_agent(
    name: str,
    url: str,
    kind: str,
    target_date: datetime,
    freshness_hours: int,
    *,
    result_limit: int = 20,
    model: str = "claude-sonnet-4-6",
    timeout_s: int = 240,
    agent_fn: Callable[..., dict[str, Any]] = run_agent,
    url_check_fn: Optional[Callable[[str], bool]] = None,
) -> list[HotspotItem]:
    """Fetch one source page via an agent and return verifier-passed item permalinks.

    Args:
        name:            Short source name (e.g. ``"openai_blog"``); ids/provenance derive from it.
        url:             The source INDEX/listing page the agent fetches.
        kind:            Item kind for ``source_type`` (e.g. ``"blog"``/``"papers"``/``"news"``).
        target_date:     "As of" date the freshness window anchors to.
        freshness_hours: Look-back window (hours).
        result_limit:    Max items to request AND max survivors to emit.
        agent_fn:        Injectable transport (defaults to ``run_agent``); tests pass a mock.
        url_check_fn:    Injectable liveness check (defaults to ``_default_url_alive``).

    Returns:
        Up to ``result_limit`` HotspotItems, each a verifier-passed item PERMALINK
        (never the index ``url``), deduped by canonical_url. ``[]`` on any failure.
    """
    if url_check_fn is None:
        url_check_fn = _default_url_alive

    index_norm = normalize_url(url)
    prompt = _build_prompt(name, url, kind, target_date, freshness_hours, result_limit)

    try:
        payload = agent_fn(
            prompt,
            schema=_SCHEMA,
            model=model,
            tools=["WebFetch", "WebSearch"],
            timeout_s=timeout_s,
        )
    except AgentError as ex:  # transport/parse/schema failure -> degrade to []
        print("Warning: agent source fetch '{0}' failed (AgentError): {1}".format(name, ex))
        return []
    except Exception as ex:  # any unexpected transport failure must not crash the run
        print("Warning: agent source fetch '{0}' failed (unexpected): {1}".format(name, ex))
        return []

    if not isinstance(payload, dict):
        print("Warning: agent source fetch '{0}' returned a non-dict payload.".format(name))
        return []
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        print("Warning: agent source fetch '{0}' missing a valid 'items' list.".format(name))
        return []

    items: list[HotspotItem] = []
    seen: set[str] = set()
    for raw in raw_items:
        if len(items) >= result_limit:
            break
        if not isinstance(raw, dict):
            continue
        item_url = clean_text(raw.get("url"))
        # INV6 deterministic verifier: scheme/host gate FIRST, then liveness.
        if not _url_is_acceptable(item_url, url_check_fn=url_check_fn):
            continue
        # Must be an item PERMALINK, not the index page we fetched.
        if normalize_url(item_url) == index_norm:
            continue
        try:
            item = _item_to_hotspot(raw, name=name, kind=kind, origin_url=url)
        except Exception as ex:  # per-item mapping failure skips this item only
            print("Warning: agent source fetch '{0}' skipped a malformed item: {1}".format(name, ex))
            continue
        if item.canonical_url in seen:
            continue
        seen.add(item.canonical_url)
        items.append(item)

    return items[:result_limit]
