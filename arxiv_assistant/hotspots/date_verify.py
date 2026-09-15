"""DateVerify — Tier-0 deterministic first-date verification (Stage 1)."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from xml.etree import ElementTree

from arxiv_assistant.utils.models import DEFAULT_DEEP_MODEL
from arxiv_assistant.utils.agent_runner import AgentError, run_agent
from arxiv_assistant.utils.hotspot.hotspot_sources import fetch_json, fetch_text, parse_datetime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_ARXIV_API = "http://export.arxiv.org/api/query?id_list={id}&max_results=1"
_CROSSREF_API = "https://api.crossref.org/works/{doi}"
_VERSION_SUFFIX = re.compile(r"v\d+$")

# Task 6: batched version-count read (§B.4.1)
_ARXIV_BATCH_API = "http://export.arxiv.org/api/query?id_list={ids}&max_results={n}"
_ABS_ID = re.compile(r"abs/(?P<id>\d{4}\.\d{4,5})v(?P<ver>\d+)")
_BATCH_SIZE = 100

_GITHUB_TREND_SOURCE = "github_trend"

# Tier-0 confidence when at least one authoritative anchor was found.
_CONFIDENCE_HIGH = 0.95
# Conservative Stage-1 fallback confidence when no authoritative anchor exists.
_CONFIDENCE_LOW = 0.3

# source families whose claimed dates are untrustworthy -> Tier-1 residual (§B.2)
_TIER1_FAMILIES = {"news", "blog", "x", "tweet", "roundup", "analysis"}

# Tier-2 deep-search escalation threshold: below this confidence AND will_be_featured -> escalate (§B.2)
_TIER2_CONFIDENCE_FLOOR = 0.6


# ---------------------------------------------------------------------------
# Stage 3: deterministic verdict helpers (INV6, §B.3)
# ---------------------------------------------------------------------------

DATEVERIFY_MODEL_ID = DEFAULT_DEEP_MODEL


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _floor_day_iso(value: str | None) -> str | None:
    """Truncate any ISO timestamp to UTC start-of-day ISO (INV2)."""
    dt = _parse_iso(value)
    if dt is None:
        return None
    dt = dt.astimezone(timezone.utc)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0).isoformat().replace("+00:00", "Z")


def _clamp_verdict(
    *,
    claimed_iso: str | None,
    agent_out: dict | None,
    wayback_earliest: str | None,
    page_published_time: str | None,
) -> dict:
    """Deterministic verifier behind the Tier-1/2 agent (INV6)."""
    credible: list[str] = []
    for raw in (
        (agent_out or {}).get("verified_first_date"),
        wayback_earliest,
        page_published_time,
        claimed_iso,
    ):
        floored = _floor_day_iso(raw)
        if floored is not None:
            credible.append(floored)

    if not credible:
        # nothing parseable at all -> degrade, never crash (spec §E)
        return {"verified_first_date": None, "confidence": 0.2, "evidence": [], "stale_date_pollution": False}

    earliest = min(credible)  # ISO day strings sort chronologically
    claimed_day = _floor_day_iso(claimed_iso)

    # has_external_signal is ONLY from non-agent external signals (Wayback / page
    # meta).  The agent's own date must NOT count here — an agent that hallucinated
    # a future date and was overridden by min() would otherwise still leak its
    # high confidence (INV6 anti-hallucination hardening).
    has_external_signal = bool(
        _floor_day_iso(wayback_earliest)
        or _floor_day_iso(page_published_time)
    )
    stale = bool(claimed_day is not None and earliest < claimed_day)

    agent_floored = _floor_day_iso((agent_out or {}).get("verified_first_date"))
    agent_agreed = bool(agent_floored is not None and agent_floored == earliest)
    agent_overridden = bool(agent_floored is not None and agent_floored > earliest)

    if agent_overridden:
        # Agent proposed a date LATER than the earliest credible date (overridden by
        # min()).  Cap to LOW REGARDLESS of external signal — a wrong-direction agent
        # must not lend high confidence even when Wayback/page rescued the date.
        # The date is still correct (min wins); only the trust is capped (INV6).
        confidence = _CONFIDENCE_LOW
    elif has_external_signal or agent_agreed:
        # Real external proof, or agent's date agreed with the earliest credible
        # date — carry the (clamped-to-[0,1]) agent confidence.
        confidence = max(0.0, min(1.0, float((agent_out or {}).get("confidence", 0.7))))
    else:
        # No external signal and no usable agent agreement -> conservative low.
        confidence = _CONFIDENCE_LOW

    evidence = list((agent_out or {}).get("evidence", []))
    evidence.append(f"verifier:earliest_min;model={DATEVERIFY_MODEL_ID}")
    return {
        "verified_first_date": earliest,
        "confidence": confidence,
        "evidence": evidence,
        "stale_date_pollution": stale,
    }


# ---------------------------------------------------------------------------
# Stage 3: Tier-1 subagent transport (§B.2, §2.11)
# ---------------------------------------------------------------------------

_DATEVERIFY_PROMPT = (
    "You are a stateless date-verification worker. Read the JSON object on stdin "
    "(schema dateverify.in.v1). Cross at least two independent signals to find the "
    "EARLIEST credible first-publication date: the provided Wayback earliest snapshot, "
    "the page article:published_time / JSON-LD datePublished, and an earliest credible "
    "report search for the title. Pollution backdates content to look new, so prefer the "
    "earliest credible date. Emit ONLY a JSON object of schema dateverify.out.v1 with keys "
    "verified_first_date (ISO8601), confidence (0..1), evidence (list of strings), "
    "stale_date_pollution (bool). No prose."
)

# Minimal JSON Schema for the dateverify.out.v1 output (used by run_agent validator).
_DATEVERIFY_OUT_SCHEMA = {
    "required": ["verified_first_date", "confidence", "evidence"],
    "properties": {
        "verified_first_date": {"type": "string"},
        "confidence": {"type": "number"},
        "evidence": {"type": "array"},
    },
}


def _run_dateverify_subagent(payload: dict) -> dict | None:
    """Dispatch the stateless Tier-1/2 DateVerify subagent via ``claude -p`` headless."""
    prompt = _DATEVERIFY_PROMPT + "\n\nInput:\n" + json.dumps(payload)
    return run_agent(
        prompt,
        schema=_DATEVERIFY_OUT_SCHEMA,
        model=DATEVERIFY_MODEL_ID,
    )


# ---------------------------------------------------------------------------
# Stage 3: anti-pollution deterministic reads (§B.2 Tier-1)
# ---------------------------------------------------------------------------

_WAYBACK_CDX = "http://web.archive.org/cdx/search/cdx"
# Two patterns to handle both attribute orders:
#   1. property/name first, content second: <meta property="article:published_time" content="...">
#   2. content first, property/name second: <meta content="..." property="article:published_time">
_META_PUBLISHED_PROP_FIRST = re.compile(
    r'<meta[^>]+(?:property|name)=["\']article:published_time["\'][^>]+content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_META_PUBLISHED_CONTENT_FIRST = re.compile(
    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name)=["\']article:published_time["\']',
    re.IGNORECASE,
)
_JSONLD = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.IGNORECASE | re.DOTALL,
)


def _cdx_ts_to_iso(ts: str) -> str | None:
    if not ts or len(ts) < 14 or not ts.isdigit():
        return None
    return f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}T{ts[8:10]}:{ts[10:12]}:{ts[12:14]}Z"


def _wayback_earliest_snapshot(url: str) -> str | None:
    """Earliest Wayback CDX capture timestamp for url, ISO8601, or None.

    Anti-pollution main signal (§B.2 Tier-1 (a)): a backdated 'new' page that
    was actually archived years ago betrays itself via its earliest capture.
    """
    try:
        rows = fetch_json(
            _WAYBACK_CDX,
            params={"url": url, "output": "json", "fl": "timestamp", "limit": "5", "from": "19960101"},
        )
    except Exception:
        return None
    if not isinstance(rows, list) or len(rows) < 2:
        return None
    timestamps = [r[0] for r in rows[1:] if r and isinstance(r[0], str)]
    iso = [_cdx_ts_to_iso(ts) for ts in timestamps]
    iso = [v for v in iso if v]
    return min(iso) if iso else None


def _page_published_time(url: str) -> str | None:
    """article:published_time meta or JSON-LD datePublished from the live page (§B.2 Tier-1 (b))."""
    try:
        html = fetch_text(url)
    except Exception:
        return None
    # Try both meta attribute orders.
    for pattern in (_META_PUBLISHED_PROP_FIRST, _META_PUBLISHED_CONTENT_FIRST):
        m = pattern.search(html)
        if m:
            return m.group(1).strip()
    for block in _JSONLD.findall(html):
        try:
            data = json.loads(block)
        except (ValueError, TypeError):
            continue
        # Normalise to a flat list of candidate objects.
        candidates: list = data if isinstance(data, list) else [data]
        # Also expand one level of @graph if present.
        expanded: list = []
        for obj in candidates:
            if isinstance(obj, dict):
                graph = obj.get("@graph")
                if isinstance(graph, list):
                    expanded.extend(graph)
                else:
                    expanded.append(obj)
            else:
                expanded.append(obj)
        for obj in expanded:
            if isinstance(obj, dict) and obj.get("datePublished"):
                return str(obj["datePublished"]).strip()
    return None


# ---------------------------------------------------------------------------
# Stage 3: Tier-1 residual verification helper (§B.2 Tier-1, §B.4)
# ---------------------------------------------------------------------------


def _verify_subagent_residual(item, *, tier: int) -> dict:
    """Tier-1 path: deterministic anti-pollution reads → stateless agent → clamp (INV6)."""
    wayback = _wayback_earliest_snapshot(item.url)
    page_time = _page_published_time(item.url)
    payload = {
        "schema": "dateverify.in.v1",
        "url": item.url,
        "title": item.title,
        "claimed_date": item.published_at,
        "tier": tier,
        "wayback_earliest": wayback,
        "page_published_time": page_time,
    }
    agent_out = _run_dateverify_subagent(payload)
    return _clamp_verdict(
        claimed_iso=item.published_at,
        agent_out=agent_out,
        wayback_earliest=wayback,
        page_published_time=page_time,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _strip_version(arxiv_id: str) -> str:
    """Remove trailing version suffix (e.g. 'v3') from an arXiv id."""
    return _VERSION_SUFFIX.sub("", (arxiv_id or "").strip())


# ---------------------------------------------------------------------------
# Task 3: arXiv v1 submission-date reader
# ---------------------------------------------------------------------------


def _fetch_arxiv_v1_date(arxiv_id: str) -> str | None:
    """Return the arXiv v1 submission timestamp (ISO8601) or None."""
    bare = _strip_version(arxiv_id)
    if not bare:
        return None
    try:
        xml = fetch_text(_ARXIV_API.format(id=bare))
        root = ElementTree.fromstring(xml)
    except Exception:
        return None
    entry = root.find(f"{_ATOM_NS}entry")
    if entry is None:
        return None
    published = entry.find(f"{_ATOM_NS}published")
    if published is None or not (published.text or "").strip():
        return None
    return published.text.strip()


# ---------------------------------------------------------------------------
# Task 4: Crossref registration-date reader
# ---------------------------------------------------------------------------


def _fetch_crossref_date(doi: str) -> str | None:
    """Return Crossref registration day (YYYY-MM-DD) or None.

    Uses the `created` date-parts (registration day, whole-day, machine-
    independent — spec §B.3.1).  Network/parse failures return None.
    """
    doi = (doi or "").strip()
    if not doi:
        return None
    try:
        payload = fetch_json(_CROSSREF_API.format(doi=doi))
    except Exception:
        return None
    parts = (((payload or {}).get("message") or {}).get("created") or {}).get("date-parts")
    if not parts or not parts[0]:
        return None
    ymd = parts[0]
    if len(ymd) < 3:
        return None
    try:
        return f"{int(ymd[0]):04d}-{int(ymd[1]):02d}-{int(ymd[2]):02d}"
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Task 5: content_hash + verify() dispatch
# ---------------------------------------------------------------------------


def _content_hash(item) -> str:
    """Cache key precedence: arxiv_id (version-stripped) > doi > canonical_url > url (§B.4)."""
    metadata = item.metadata or {}
    arxiv_id = _strip_version(metadata.get("arxiv_id", ""))
    if arxiv_id:
        return f"arxiv:{arxiv_id}"
    doi = (metadata.get("doi") or "").strip()
    if doi:
        return f"doi:{doi}"
    return f"url:{item.canonical_url or item.url}"


def _earliest(*candidates: str | None) -> str | None:
    """Earliest-credible-date-wins (§B.3): return the minimum timestamp string.

    Parses each candidate with parse_datetime; keeps the original string form
    of the winner so callers preserve the source precision (full ISO8601 or
    YYYY-MM-DD).  Returns None when no candidate is parseable.
    """
    parsed = [(parse_datetime(c), c) for c in candidates if c]
    parsed = [(dt, c) for dt, c in parsed if dt is not None]
    if not parsed:
        return None
    return min(parsed, key=lambda pair: pair[0])[1]


def verify(item, store, *, will_be_featured: bool = False) -> dict:
    """Tier-0 deterministic first-date verification (spec §B.2/§B.3/§2.5)."""
    content_hash = _content_hash(item)

    # --- 1. Cache hit (INV3 permanent freeze) ---
    cached = store.get_verdict(content_hash)
    if cached is not None:
        return cached

    metadata = item.metadata or {}
    claimed = item.published_at
    fetched = metadata.get("fetched_at")
    evidence: list[str] = []

    # --- 2. github_trend exception: observed-trending date is the signal (§B.2) ---
    if item.source_id == _GITHUB_TREND_SOURCE:
        observed = claimed or fetched
        verdict = {
            "verified_first_date": observed,
            "confidence": _CONFIDENCE_HIGH,
            "evidence": ["github_trend:observed_trending_date"],
        }
        store.put_verdict(content_hash, verdict)
        return verdict

    # --- 3. Tier-0 deterministic readers ---
    credible: list[str] = []

    arxiv_id = _strip_version(metadata.get("arxiv_id", ""))
    if arxiv_id:
        v1 = _fetch_arxiv_v1_date(arxiv_id)
        if v1:
            credible.append(v1)
            evidence.append(f"arxiv_v1:{arxiv_id}")

    doi = (metadata.get("doi") or "").strip()
    if doi:
        cr = _fetch_crossref_date(doi)
        if cr:
            credible.append(cr)
            evidence.append(f"crossref:{doi}")

    if credible:
        # Earliest-credible-date-wins; include source-claimed as an additional
        # candidate so that if the source date is earlier (e.g. preprint posted
        # before DOI registration) we still honour it.
        verified = _earliest(*credible, claimed)
        verdict = {
            "verified_first_date": verified,
            "confidence": _CONFIDENCE_HIGH,
            "evidence": evidence,
        }
    else:
        # --- 4. Stage-3 Tier-1 dispatch for residual items (§B.2 Tier-1) ---
        # Residual news/blog/X items get the Tier-1 subagent (Wayback CDX +
        # published_time + earliest-mention).  AgentError degrades conservatively
        # to min(claimed,fetched)+LOW (spec §E — never crash).  Non-residual items
        # fall through to the conservative Stage-1 fallback immediately.
        family = (item.source_type or "").lower()
        if family in _TIER1_FAMILIES:
            try:
                verdict = _verify_subagent_residual(item, tier=1)
            except AgentError:
                # Deterministic degrade: conservative min(claimed,fetched)+LOW (§E)
                verified = _earliest(claimed, fetched) or claimed or fetched
                verdict = {
                    "verified_first_date": verified,
                    "confidence": _CONFIDENCE_LOW,
                    "evidence": ["fallback:min(claimed,fetched);agent_error"],
                }
            # Tier-2 deep-search escalation: rare, cost-controlled (§B.2).
            # Only when Tier-1 confidence is low AND item will be featured.
            if will_be_featured and verdict["confidence"] < _TIER2_CONFIDENCE_FLOOR:
                try:
                    deep = _verify_subagent_residual(item, tier=2)
                    # Cross-tier merge: earliest-credible-date-wins (INV6) over both tiers.
                    # Re-run _clamp_verdict with the merged agent_out so INV2/INV6 still hold.
                    verdict = _clamp_verdict(
                        claimed_iso=item.published_at,
                        agent_out={
                            "verified_first_date": min(
                                (d for d in (verdict["verified_first_date"], deep["verified_first_date"]) if d),
                                default=None,  # both None -> let _clamp_verdict degrade (no ValueError)
                            ),
                            "confidence": max(verdict["confidence"], deep["confidence"]),
                            "evidence": list(verdict.get("evidence", [])) + list(deep.get("evidence", [])),
                        },
                        wayback_earliest=None,
                        page_published_time=None,
                    )
                except AgentError:
                    # Degrade silently: keep Tier-1 result, no crash (§E)
                    pass
        else:
            # Non-residual, no Tier-0 anchor: conservative Stage-1 fallback (§B.3)
            verified = _earliest(claimed, fetched) or claimed or fetched
            verdict = {
                "verified_first_date": verified,
                "confidence": _CONFIDENCE_LOW,
                "evidence": ["fallback:min(claimed,fetched)"],
            }

    # --- 5. Persist (write-once; no-op if already frozen) ---
    store.put_verdict(content_hash, verdict)
    return store.get_verdict(content_hash)  # return the frozen copy (stable across calls)


# ---------------------------------------------------------------------------
# Task 6: poll_arxiv_versions — batched version-count read (§B.4.1 / §2.5)
# ---------------------------------------------------------------------------


def poll_arxiv_versions(arxiv_ids: list[str]) -> dict[str, int]:
    """Return {bare_arxiv_id: latest_version_count} via batched id_list reads (§B.4.1)."""
    bare_ids = list(dict.fromkeys(s for i in arxiv_ids if (s := _strip_version(i))))
    if not bare_ids:
        return {}

    counts: dict[str, int] = {}
    for start in range(0, len(bare_ids), _BATCH_SIZE):
        batch = bare_ids[start:start + _BATCH_SIZE]
        url = _ARXIV_BATCH_API.format(ids=",".join(batch), n=len(batch))
        try:
            xml = fetch_text(url)
            root = ElementTree.fromstring(xml)
        except Exception:
            continue
        for entry in root.findall(f"{_ATOM_NS}entry"):
            id_el = entry.find(f"{_ATOM_NS}id")
            if id_el is None or not id_el.text:
                continue
            match = _ABS_ID.search(id_el.text)
            if match:
                counts[match.group("id")] = int(match.group("ver"))
    return counts
