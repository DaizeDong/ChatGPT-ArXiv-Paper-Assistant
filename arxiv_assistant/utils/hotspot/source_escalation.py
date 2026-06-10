"""Route-aware source escalation: cheap direct scrape -> agent WebFetch -> browser bypass.

A source is first attempted with its cheap deterministic scraper (DIRECT). A deterministic
``detect_protection`` classifier inspects the outcome (item count + exception / HTTP status /
body size) and a fixed policy decides whether — and to which tier — to escalate:

    DIRECT  (Python scraper)            cheap/fast; the floor.
    AGENT   (claude -p WebFetch)        rescues a source whose scraper broke but the page is
                                        plainly fetchable (layout change, 401-LLM-extract, empty).
    BROWSER (act-like-human playwright) rescues bot-walled / JS-rendered sources (reddit/X/JS CN);
                                        a pluggable ``browser_fn`` (real playwright wiring lands later).

Every tier is fail-fast and degrade-safe: a tier raising is recorded and the chain continues;
the result records which route produced items (``route_used``) plus a per-tier ``attempts`` trail
for observability (which sources are degrading, which tier rescued them). NEVER raises.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Optional

# Route + status constants
ROUTE_DIRECT = "direct"
ROUTE_AGENT = "agent"
ROUTE_BROWSER = "browser"

OK = "ok"
PROTECTED = "protected"      # 403/429/captcha/login-wall/timeout -> needs the human-like browser
JS_WALLED = "js_walled"      # 200 but empty/JS-shell body -> needs the browser to render
NEEDS_AGENT = "needs_agent"  # 401 / LLM-extraction-key / auth -> the agent WebFetch can read it
EMPTY = "empty"              # 0 items, real page -> try the cheaper agent before the browser
ERROR = "error"              # unexpected scraper failure -> try the agent

_PROTECTED_KEYS = ("403", "429", "captcha", "blocked", "forbidden", "too many requests", "rate limit")
_TIMEOUT_KEYS = ("timeout", "timed out")
_NEEDS_AGENT_KEYS = ("401", "unauthorized", "api key", "llm extraction")
_JS_SHELL_MAX_BODY = 1500  # a 200 response shorter than this with 0 items is almost surely a JS shell


def detect_protection(
    *,
    item_count: int,
    exception: BaseException | None = None,
    http_status: int | None = None,
    body_len: int | None = None,
) -> str:
    """Classify a DIRECT scrape outcome deterministically (see module docstring for the policy)."""
    if exception is not None:
        s = str(exception).lower()
        if isinstance(exception, TimeoutError) or any(k in s for k in _PROTECTED_KEYS) or any(k in s for k in _TIMEOUT_KEYS):
            return PROTECTED
        if any(k in s for k in _NEEDS_AGENT_KEYS):
            return NEEDS_AGENT
        return ERROR
    if http_status in (403, 429):
        return PROTECTED
    if http_status == 401:
        return NEEDS_AGENT
    if item_count >= 1:
        return OK
    if body_len is not None and body_len < _JS_SHELL_MAX_BODY:
        return JS_WALLED
    return EMPTY


def escalation_route(status: str) -> Optional[str]:
    """Map a non-OK status to the tier to escalate to (None == no escalation)."""
    if status == OK:
        return None
    if status in (PROTECTED, JS_WALLED):
        return ROUTE_BROWSER  # protected/JS pages need the act-like-human browser
    return ROUTE_AGENT        # NEEDS_AGENT / EMPTY / ERROR: try the cheaper agent WebFetch first


def fetch_source_resilient(
    name: str,
    url: str,
    kind: str,
    target_date: datetime,
    freshness_hours: int,
    *,
    direct_fn: Callable[[], list],
    agent_fn: Optional[Callable[..., list]] = None,
    browser_fn: Optional[Callable[..., list]] = None,
    result_limit: int = 20,
    agent_timeout_s: int = 120,
    prefer: str = "direct",
) -> dict[str, Any]:
    """Fetch one source with tiered escalation; returns items + the route that produced them.

    ``direct_fn()`` is the existing scraper (a thunk). ``agent_fn`` and ``browser_fn`` follow the
    ``fetch_source_via_agent``-style signature ``(name, url, kind, target_date, freshness_hours, *,
    result_limit, [timeout_s])``. ``prefer`` in {"direct","agent","browser"} can skip DIRECT for a
    known-protected source. Returns ``{items, route_used, status, attempts}``; never raises.
    """
    attempts: list[dict[str, Any]] = []

    def _run_agent() -> Optional[list]:
        if agent_fn is None:
            return None
        try:
            items = agent_fn(name, url, kind, target_date, freshness_hours,
                             result_limit=result_limit, timeout_s=agent_timeout_s) or []
        except Exception as exc:  # tier failure is recorded, the chain continues
            attempts.append({"route": ROUTE_AGENT, "status": ERROR, "n_items": 0, "error": str(exc)[:200]})
            return None
        attempts.append({"route": ROUTE_AGENT, "status": OK if items else EMPTY, "n_items": len(items)})
        return items or None

    def _run_browser() -> Optional[list]:
        if browser_fn is None:
            return None
        try:
            items = browser_fn(name, url, kind, target_date, freshness_hours,
                               result_limit=result_limit) or []
        except Exception as exc:  # tier failure is recorded, the chain continues
            attempts.append({"route": ROUTE_BROWSER, "status": ERROR, "n_items": 0, "error": str(exc)[:200]})
            return None
        attempts.append({"route": ROUTE_BROWSER, "status": OK if items else EMPTY, "n_items": len(items)})
        return items or None

    # --- Tier DIRECT (unless prefer skips it) ---
    if prefer == "direct":
        exc: Optional[BaseException] = None
        try:
            items = direct_fn() or []
        except Exception as e:  # scraper failure -> classify + escalate, never propagate
            items, exc = [], e
        status = detect_protection(item_count=len(items), exception=exc)
        rec = {"route": ROUTE_DIRECT, "status": status, "n_items": len(items)}
        if exc is not None:
            rec["error"] = str(exc)[:200]
        attempts.append(rec)
        if status == OK:
            return {"items": items, "route_used": ROUTE_DIRECT, "status": status, "attempts": attempts}
        primary = escalation_route(status)
    else:
        status = NEEDS_AGENT if prefer == ROUTE_AGENT else PROTECTED
        primary = ROUTE_AGENT if prefer == ROUTE_AGENT else ROUTE_BROWSER

    # --- Escalation: primary tier, then the other as a weaker fallback ---
    order = [primary, ROUTE_AGENT if primary == ROUTE_BROWSER else ROUTE_BROWSER]
    for route in order:
        got = _run_browser() if route == ROUTE_BROWSER else _run_agent()
        if got:
            return {"items": got, "route_used": route, "status": status, "attempts": attempts}

    return {"items": [], "route_used": "none", "status": status, "attempts": attempts}
