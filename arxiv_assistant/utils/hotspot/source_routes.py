"""Static subagent-route registry for KNOWN-FAIL sources.

Design decision (static routing, NOT runtime escalation): most sources have a
working direct Python scraper and stay on it. A small set of sources DEFINITELY
fails with plain scrapers / WebFetch -- bot-walled (reddit/X) or JS-heavy +
LLM-extract-keyed (some CN lab blogs). Those are routed here, statically, to a
subagent tool:

  - ``"browser"`` -> ``apis.hotspot.browser_source_fetch.fetch_source_via_browser``
    (playwright act-like-human; the only thing that gets past bot-walls / JS).
  - ``"agent"``   -> ``apis.hotspot.agent_source_fetch.fetch_source_via_agent``
    (plain WebFetch; for pages that are fetchable but whose scraper needed a key
    we don't have, e.g. LLM-extract 401).

This is plain data + two helpers; no protection detection, no try-direct-first.
A source either has a direct scraper (not listed here) or a static subagent route
(listed here). URLs marked "verify" should be confirmed before production use.
"""
from __future__ import annotations

from typing import Optional

# name -> {url, kind, route, reason}
SUBAGENT_ROUTES: dict[str, dict[str, str]] = {
    "reddit_localllama": {
        "url": "https://www.reddit.com/r/LocalLLaMA/",
        "kind": "social",
        "route": "browser",
        "reason": "reddit bot-wall (403 to scraper, timeout to plain WebFetch)",
    },
    "reddit_machinelearning": {
        "url": "https://www.reddit.com/r/MachineLearning/",
        "kind": "social",
        "route": "browser",
        "reason": "reddit bot-wall (403 to scraper, timeout to plain WebFetch)",
    },
    "x_ai_news": {
        "url": "https://x.com/search?q=AI%20OR%20LLM&f=live",
        "kind": "social",
        "route": "browser",
        "reason": "X login-wall; needs a real browser session (verify)",
    },
    "jiqizhixin": {
        "url": "https://www.jiqizhixin.com/",
        "kind": "news",
        "route": "browser",
        "reason": "JS-heavy render + LLM-extract 401; plain fetch returns an empty shell",
    },
    "zhipu_blog": {
        "url": "https://www.zhipuai.cn/news",
        "kind": "news",
        "route": "agent",
        "reason": "LLM-extract 401 only; page is plain-fetchable via the agent (verify url)",
    },
    "bytedance_seed_blog": {
        "url": "https://seed.bytedance.com/en/blog",
        "kind": "news",
        "route": "agent",
        "reason": "LLM-extract 401 only; agent WebFetch reads it without a key (verify url)",
    },
}


def route_for(name: str) -> Optional[str]:
    """Return the static subagent route (``"browser"``/``"agent"``) for ``name``,
    or ``None`` if the source is not subagent-routed (i.e. uses its direct scraper)."""
    return (SUBAGENT_ROUTES.get(name) or {}).get("route")


def iter_subagent_sources() -> list[tuple[str, dict[str, str]]]:
    """All statically subagent-routed (name, spec) pairs. Each spec has url/kind/route/reason."""
    return list(SUBAGENT_ROUTES.items())
