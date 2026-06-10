"""Browser-capable subagent source fetcher -- for sources that DEFINITELY fail
with plain scrapers / WebFetch (reddit/X bot-walls, JS-heavy sites).

Plain ``WebFetch`` (see ``agent_source_fetch``) times out or returns nothing on
bot-walled / JavaScript-rendered pages. Those sources are STATICALLY routed (see
``utils.hotspot.source_routes``) to this tool, which drives a ``claude -p``
subagent that uses the **playwright MCP** to act like a human browser: navigate
to the page, wait for the feed to render, snapshot/evaluate the DOM, scroll, and
extract the recent AI/ML item permalinks.

Anti-hallucination reuses the scout's DETERMINISTIC URL verifier (INV6): every
emitted item must have a syntactically valid, non-blocklisted, live http(s) URL,
and be an item PERMALINK (not the index page itself). Degrades to ``[]`` on any
agent/parse/mapping failure -- never raises out of ``fetch_source_via_browser``.

Note: real headless playwright-MCP availability is an environment dependency
validated separately; the unit tests fully mock the transport.
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

# playwright MCP browser tools (act-like-human) + WebFetch as a weak fallback.
PLAYWRIGHT_TOOLS = [
    "mcp__plugin_playwright_playwright__browser_navigate",
    "mcp__plugin_playwright_playwright__browser_snapshot",
    "mcp__plugin_playwright_playwright__browser_evaluate",
    "mcp__plugin_playwright_playwright__browser_wait_for",
    "mcp__plugin_playwright_playwright__browser_click",
    "WebFetch",
]


def _build_prompt(name: str, url: str, kind: str, target_date: datetime, freshness_hours: int, result_limit: int) -> str:
    """Prompt the subagent to BROWSE one bot-walled/JS source and extract item permalinks."""
    as_of = target_date.strftime("%Y-%m-%d")
    body = (
        "You are an AI/ML source extractor for a bot-walled or JavaScript-heavy page that plain "
        "HTTP fetch CANNOT read. USE THE BROWSER (playwright): navigate to this exact page, wait "
        "for the feed/list to finish rendering, take a snapshot or evaluate the DOM, and scroll if "
        "the items load lazily:\n"
        "    {url}\n"
        "It is the '{name}' source (a {kind} page). Act like a human browsing it. From the "
        "rendered page, extract the AI/ML items POSTED in the LAST {hours} HOURS as of {as_of} "
        "(UTC). Cover only ARTIFICIAL-INTELLIGENCE / MACHINE-LEARNING items; ignore unrelated "
        "entries.\n\n"
        "For EACH item return an object with keys:\n"
        '  - "title": the item headline.\n'
        '  - "url": the CANONICAL ITEM PERMALINK -- the specific post/thread/article/release URL '
        "on this site (the individual item page). It MUST be a specific item page, NOT the "
        "index/listing page above, and NOT a search-results or aggregator link.\n"
        '  - "summary": one short sentence on what it is.\n'
        '  - "published_at": the post date/time if shown (ISO 8601), else omit.\n\n'
        "If, after browsing, you cannot confirm an item URL is real and reachable, OMIT that item.\n\n"
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
        raise ValueError("browser-source item missing title/url")

    source_id = "browser_src:{0}".format(name)
    provenance = "browser:source:{0}".format(name)
    item = HotspotItem(
        source_id=source_id,
        source_name=name,
        source_role="browser_source",
        source_type=kind or "social",
        title=title,
        summary=clip_text(raw.get("summary"), 420),
        url=url,
        canonical_url=url,
        published_at=clean_text(raw.get("published_at")) or None,
        tags=["browser-source", name],
        authors=[],
        metadata={
            "source_kind": kind or "social",
            "origin_url": origin_url,
            "fetch_route": "browser",
        },
    )
    if hasattr(item, "provenance"):
        item.provenance = provenance
    item.metadata.setdefault("provenance", provenance)
    item.metadata.setdefault("source_id", source_id)
    return item


def fetch_source_via_browser(
    name: str,
    url: str,
    kind: str,
    target_date: datetime,
    freshness_hours: int,
    *,
    result_limit: int = 15,
    model: str = "claude-sonnet-4-6",
    timeout_s: int = 300,
    agent_fn: Callable[..., dict[str, Any]] = run_agent,
    url_check_fn: Optional[Callable[[str], bool]] = None,
    tools: Optional[list[str]] = None,
) -> list[HotspotItem]:
    """Browse one bot-walled/JS source via a playwright subagent; return item permalinks.

    Args:
        name:            Short source name (e.g. ``"reddit_localllama"``); ids/provenance derive from it.
        url:             The source INDEX/feed page the agent browses.
        kind:            Item kind for ``source_type`` (e.g. ``"social"``/``"news"``).
        target_date:     "As of" date the freshness window anchors to.
        freshness_hours: Look-back window (hours).
        result_limit:    Max items to request AND max survivors to emit.
        agent_fn:        Injectable transport (defaults to ``run_agent``); tests pass a mock.
        url_check_fn:    Injectable liveness check (defaults to ``_default_url_alive``).
        tools:           Allowed tools for the subagent (defaults to ``PLAYWRIGHT_TOOLS``).

    Returns:
        Up to ``result_limit`` HotspotItems, each a verifier-passed item PERMALINK
        (never the index ``url``), deduped by canonical_url. ``[]`` on any failure.
    """
    if url_check_fn is None:
        url_check_fn = _default_url_alive
    if tools is None:
        tools = PLAYWRIGHT_TOOLS

    index_norm = normalize_url(url)
    prompt = _build_prompt(name, url, kind, target_date, freshness_hours, result_limit)

    try:
        payload = agent_fn(
            prompt,
            schema=_SCHEMA,
            model=model,
            tools=tools,
            timeout_s=timeout_s,
        )
    except AgentError as ex:  # transport/parse/schema failure -> degrade to []
        print("Warning: browser source fetch '{0}' failed (AgentError): {1}".format(name, ex))
        return []
    except Exception as ex:  # any unexpected transport failure must not crash the run
        print("Warning: browser source fetch '{0}' failed (unexpected): {1}".format(name, ex))
        return []

    if not isinstance(payload, dict):
        print("Warning: browser source fetch '{0}' returned a non-dict payload.".format(name))
        return []
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        print("Warning: browser source fetch '{0}' missing a valid 'items' list.".format(name))
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
        # Must be an item PERMALINK, not the index page we browsed.
        if normalize_url(item_url) == index_norm:
            continue
        try:
            item = _item_to_hotspot(raw, name=name, kind=kind, origin_url=url)
        except Exception as ex:  # per-item mapping failure skips this item only
            print("Warning: browser source fetch '{0}' skipped a malformed item: {1}".format(name, ex))
            continue
        if item.canonical_url in seen:
            continue
        seen.add(item.canonical_url)
        items.append(item)

    return items[:result_limit]
