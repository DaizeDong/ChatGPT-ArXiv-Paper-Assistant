"""Agent Scout source (spec section A / plan Task 1).

A kernel harvest source (peer of ``hotspot_twitterapi`` and the RSS feeds) that
runs a single ``claude -p`` web-research call to discover recent notable AI/ML
developments the deterministic feeds miss -- especially X/social buzz (replacing
the metered ``twitterapi.io`` key) and breaking releases/news.

Anti-hallucination is enforced by a DETERMINISTIC verifier (spec INV6): every
scout item is dropped unless its ``url`` is (a) a syntactically valid http(s)
URL on a non-blocklisted host AND (b) survives a lightweight liveness check
(``url_check_fn``). A fabricated/dead link therefore cannot enter the report.

The source degrades to ``[]`` on agent failure/timeout/empty/malformed output and
on any per-item mapping failure -- it NEVER raises out of ``fetch_hotspot_items``,
so the free deterministic feeds remain the coverage floor.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Optional
from urllib.parse import urlsplit

import requests

from arxiv_assistant.utils.agent_runner import AgentError, run_agent
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem, clean_text
from arxiv_assistant.utils.hotspot.hotspot_sources import clip_text

SOURCE_ID = "agent_scout"
PROVENANCE = "agent:scout"

# Hosts a canonical source URL must NOT be -- search/aggregator landing pages are
# not canonical sources and are a common shape for a hallucinated "I found it"
# link. The host check runs BEFORE the liveness check so these never resolve.
_BLOCKLISTED_HOSTS = frozenset(
    {
        "google.com",
        "www.google.com",
        "google.co.uk",
        "bing.com",
        "www.bing.com",
        "duckduckgo.com",
        "www.duckduckgo.com",
        "search.brave.com",
        "search.yahoo.com",
        "yahoo.com",
        "www.yahoo.com",
        "baidu.com",
        "www.baidu.com",
        "example.com",
        "www.example.com",
        "example.org",
        "example.net",
        "localhost",
    }
)

_SCHEMA: dict[str, Any] = {"required": ["items"], "properties": {"items": {"type": "array"}}}


def _build_prompt(target_date: datetime, freshness_hours: int, result_limit: int) -> str:
    """Build the web-research prompt for the scout agent.

    Scopes the search to the last ``freshness_hours`` window as of ``target_date``
    and to AI/ML topicality, and demands the CANONICAL source URL (not a
    search-results link) so the deterministic verifier has a real target.
    """
    as_of = target_date.strftime("%Y-%m-%d")
    return (
        "You are an AI/ML news scout. Using web search and fetch, find the most notable "
        "AI/ML developments from the LAST {hours} HOURS as of {as_of} (UTC). Include: "
        "new research papers, model/product releases, significant X/Twitter or other social "
        "discussion, and breaking news. Cover only ARTIFICIAL-INTELLIGENCE / MACHINE-LEARNING "
        "topics; ignore unrelated tech or general news.\n\n"
        "For EACH development return an object with keys:\n"
        '  - "title": a concise headline.\n'
        '  - "url": the CANONICAL source URL (the arXiv abstract page, the official blog post, '
        "the GitHub repo, or the exact tweet permalink). Do NOT return a Google/Bing/DuckDuckGo "
        "search-results link, an aggregator landing page, or a made-up URL -- only a real, "
        "directly-reachable source page you actually found.\n"
        '  - "summary": one sentence on why it matters.\n'
        '  - "source_kind": one of "paper" | "release" | "social" | "news".\n'
        '  - "published_at": the publication date/time if known (ISO 8601), else omit.\n\n'
        "Return STRICT JSON ONLY of the form "
        '{{"items": [ {{...}}, {{...}} ]}} with at most {limit} items. '
        "If you are unsure a URL is real and reachable, omit that item."
    ).format(hours=int(freshness_hours), as_of=as_of, limit=int(result_limit))


def _is_arxiv_or_doi(host: str, url: str) -> bool:
    """arXiv abstract/pdf ids and DOI urls are treated as alive WITHOUT a fetch:
    they are stable, canonical identifiers, and arXiv rate-limits HEAD probes."""
    if host in {"arxiv.org", "www.arxiv.org"} and ("/abs/" in url or "/pdf/" in url):
        return True
    if host in {"doi.org", "dx.doi.org", "www.doi.org"}:
        return True
    return False


def _default_url_alive(url: str) -> bool:
    """Lightweight liveness check used when no ``url_check_fn`` is injected.

    arXiv abs/pdf ids and DOI urls count as alive without any network call.
    Otherwise a HEAD (falling back to a ranged GET) must return 200-399. Any
    exception -> False, so a fabricated/dead link is discarded.
    """
    split = urlsplit(url)
    host = split.netloc.lower()
    if _is_arxiv_or_doi(host, url):
        return True
    try:
        resp = requests.head(url, allow_redirects=True, timeout=8)
        if 200 <= resp.status_code < 400:
            return True
        # Some hosts reject HEAD (405) -- retry with a tiny ranged GET.
        resp = requests.get(url, allow_redirects=True, timeout=8, headers={"Range": "bytes=0-0"})
        return 200 <= resp.status_code < 400
    except Exception:  # network/DNS/TLS/timeout -> treat as dead (anti-hallucination)
        return False


def _url_is_acceptable(url: str, *, url_check_fn: Callable[[str], bool]) -> bool:
    """INV6 gate: (1) syntactic http(s) URL with a non-blocklisted host, then
    (2) liveness via ``url_check_fn``. The scheme/host gate runs FIRST so a
    garbage or blocklisted url is dropped even if ``url_check_fn`` would pass it.
    """
    if not url:
        return False
    split = urlsplit(url)
    if split.scheme not in ("http", "https"):
        return False
    host = split.netloc.lower()
    if not host or host in _BLOCKLISTED_HOSTS:
        return False
    # Defend against obvious aggregator/search hosts beyond the explicit blocklist.
    if host.endswith(".google.com") or host.endswith(".bing.com"):
        return False
    try:
        return bool(url_check_fn(url))
    except Exception:  # a broken url_check_fn must not crash the harvest
        return False


def _item_to_hotspot(raw: dict[str, Any]) -> HotspotItem:
    """Map one agent item dict to a HotspotItem. Raises on malformed input so the
    caller can skip just this item (never the whole batch)."""
    title = clean_text(raw.get("title"))
    url = clean_text(raw.get("url"))
    if not title or not url:
        raise ValueError("scout item missing title/url")

    source_kind = clean_text(raw.get("source_kind")) or "news"
    published_at = clean_text(raw.get("published_at")) or None

    item = HotspotItem(
        source_id=SOURCE_ID,
        source_name="Agent Scout",
        source_role="agent_discovery",
        source_type="news",
        title=title,
        summary=clip_text(raw.get("summary"), 420),
        url=url,
        canonical_url=url,
        published_at=published_at,
        tags=["agent-scout", source_kind],
        authors=[],
        metadata={
            "source_kind": source_kind,
            "discovered_url": url,
        },
    )
    # Set the Stage-0 provenance field if the dataclass exposes it, and ALWAYS
    # mirror provenance + source_id into metadata (defensive, mirrors twitterapi).
    if hasattr(item, "provenance"):
        item.provenance = PROVENANCE
    item.metadata.setdefault("provenance", PROVENANCE)
    item.metadata.setdefault("source_id", SOURCE_ID)
    return item


def fetch_hotspot_items(
    target_date: datetime,
    freshness_hours: int,
    *,
    result_limit: int = 40,
    model: str = "claude-sonnet-4-6",
    timeout_s: int = 300,
    agent_fn: Callable[..., dict[str, Any]] = run_agent,
    url_check_fn: Optional[Callable[[str], bool]] = None,
) -> list[HotspotItem]:
    """Run the scout agent and return verifier-passed HotspotItems.

    Args:
        target_date:     "As of" date the freshness window is anchored to.
        freshness_hours: Look-back window (hours) the agent is asked to cover.
        result_limit:    Max items to request AND max survivors to emit.
        model:           Model id passed to ``agent_fn``.
        timeout_s:       Agent subprocess timeout (seconds).
        agent_fn:        Injectable transport (defaults to ``run_agent``); tests
                         pass a mock so the suite never spawns ``claude``.
        url_check_fn:    Injectable liveness check (defaults to
                         ``_default_url_alive``); tests pass a fake so the suite
                         never hits the network.

    Returns:
        Up to ``result_limit`` HotspotItems, each with a verifier-passed,
        resolvable, non-blocklisted http(s) URL, deduped by canonical_url.
        ``[]`` on any agent/parse/mapping failure (degrade, never raise).
    """
    if url_check_fn is None:
        url_check_fn = _default_url_alive

    prompt = _build_prompt(target_date, freshness_hours, result_limit)

    try:
        payload = agent_fn(
            prompt,
            schema=_SCHEMA,
            model=model,
            tools=["WebSearch", "WebFetch"],
            timeout_s=timeout_s,
        )
    except AgentError as ex:  # transport/parse/schema failure -> degrade to []
        print(f"Warning: agent scout failed (AgentError): {ex}")
        return []
    except Exception as ex:  # any unexpected transport failure must not crash the run
        print(f"Warning: agent scout failed (unexpected): {ex}")
        return []

    if not isinstance(payload, dict):
        print("Warning: agent scout returned a non-dict payload. Skipping.")
        return []
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        print("Warning: agent scout payload missing a valid 'items' list. Skipping.")
        return []

    items: list[HotspotItem] = []
    seen: set[str] = set()
    for raw in raw_items:
        if len(items) >= result_limit:
            break
        if not isinstance(raw, dict):
            continue
        url = clean_text(raw.get("url"))
        # INV6 deterministic verifier: scheme/host gate FIRST, then liveness.
        if not _url_is_acceptable(url, url_check_fn=url_check_fn):
            continue
        try:
            item = _item_to_hotspot(raw)
        except Exception as ex:  # per-item mapping failure skips this item only
            print(f"Warning: agent scout skipped a malformed item: {ex}")
            continue
        if item.canonical_url in seen:
            continue  # dedupe by canonical_url within the batch
        seen.add(item.canonical_url)
        items.append(item)

    return items[:result_limit]
