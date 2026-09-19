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
    # CN AI-lab blogs: JS SPAs whose in-repo `playwright_llm` mode 401s on the
    # OpenAI extraction step. Routed to the browser subagent (claude renders +
    # extracts, ZERO key) and DISABLED in configs/hotspot/official_blogs.json so
    # they are not double-fetched. The browser also handles the soft login/consent
    # walls these CN sites often show. URLs mirror official_blogs.json source_ids.
    "zhipu_blog": {
        "url": "https://www.zhipuai.cn/",
        "kind": "news",
        "route": "browser",
        "reason": "JS SPA + playwright_llm 401; zero-key browser render+extract",
    },
    "bytedance_seed_blog": {
        "url": "https://seed.bytedance.com/en/blog",
        "kind": "news",
        "route": "browser",
        "reason": "JS SPA + playwright_llm 401; zero-key browser render+extract",
    },
    "baichuan_blog": {
        "url": "https://www.baichuan-ai.com/home",
        "kind": "news",
        "route": "browser",
        "reason": "JS SPA + playwright_llm 401; zero-key browser render+extract",
    },
    "01ai_blog": {
        "url": "https://www.01.ai/",
        "kind": "news",
        "route": "browser",
        "reason": "JS SPA + playwright_llm 401; zero-key browser render+extract",
    },
    "stepfun_blog": {
        "url": "https://www.stepfun.com/",
        "kind": "news",
        "route": "browser",
        "reason": "JS SPA + playwright_llm 401; zero-key browser render+extract (root URL)",
    },
}


def route_for(name: str) -> Optional[str]:
    """Return the static subagent route (``"browser"``/``"agent"``) for ``name``,
    or ``None`` if the source is not subagent-routed (i.e. uses its direct scraper)."""
    return (SUBAGENT_ROUTES.get(name) or {}).get("route")


def iter_subagent_sources() -> list[tuple[str, dict[str, str]]]:
    """All statically subagent-routed (name, spec) pairs. Each spec has url/kind/route/reason."""
    return list(SUBAGENT_ROUTES.items())
