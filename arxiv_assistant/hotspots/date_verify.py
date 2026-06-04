"""DateVerify — Tier-0 deterministic first-date verification (Stage 1).

Tier-0 reads authoritative, machine-readable publication dates from external
registries (arXiv Atom API, Crossref) without any LLM or subagent.  Results
are frozen in the StoryStore via write-once put_verdict (INV3).

Stage-3 will extend this module with Tier-1/2 subagent dispatch; a clear seam
is marked below.  Until then, items with no authoritative anchor receive a
conservative min(claimed, fetched) verdict at LOW confidence (§B.3 legal
fallback — not a stub, not a placeholder).

Spec §2.5 verify() signature:
    verify(item, store, *, will_be_featured=False) -> dict
        {"verified_first_date": str, "confidence": float, "evidence": list[str]}
"""
from __future__ import annotations

import re
from xml.etree import ElementTree

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
    """Return the arXiv v1 submission timestamp (ISO8601) or None.

    arXiv's Atom <published> is ALWAYS the v1 submission time; <updated> is the
    latest version.  Reading <published> directly fixes the HF publishedAt
    staleness bug (spec §B.2 / §0).

    The id is stripped of any vN suffix before querying so the Atom API returns
    the canonical entry regardless of which version the caller saw.

    Network/parse failures return None — the caller degrades conservatively and
    never raises.
    """
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
    """Tier-0 deterministic first-date verification (spec §B.2/§B.3/§2.5).

    Returns {"verified_first_date": str, "confidence": float, "evidence": [str]}.

    Steps:
    1. Cache hit  → return frozen verdict (INV3 write-once freeze).
    2. github_trend exception → observed-trending date is the legitimate signal (§B.2).
    3. Tier-0 deterministic → arXiv v1, Crossref; earliest-credible-date-wins (§B.3).
    4. Conservative Stage-1 fallback → min(claimed, fetched) + LOW confidence (§B.3).
       This is a real, fully-specified Stage-1 behaviour, not a stub.
    5. Write verdict via store.put_verdict (write-once; no-op if exists).

    `will_be_featured` (default False, added per §2.5 / addendum 3) is reserved
    for Stage 3 to gate the Tier-2 deep-search escalation.  It has no effect in
    Stage 1 Tier-0 — do not remove.

    # stage3: Tier-1/2 subagent extends here (Wayback CDX + published_time +
    #         earliest-mention search).  Consult spec §B.1/§B.4 and stage-3
    #         plan task 5 for the exact dispatch contract.  The seam is the
    #         `else` branch below where `credible` is empty.
    """
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
        # --- 4. Stage-1 conservative fallback (§B.3 "无法核实→保守 min(claimed,fetched)") ---
        # Tier-1/2 subagent dispatch is NOT implemented until Stage 3.
        # This is the explicitly-specified legal Stage-1 behaviour: return the
        # earliest of source-claimed and metadata fetched_at, mark confidence LOW
        # so downstream gates fold it below the authoritative-anchor path.
        # stage3: Tier-1/2 subagent extends here — replace this else-branch with
        #         a run_agent(Wayback/published_time) call; keep the conservative
        #         path as the network-failure/timeout degrade.
        verified = _earliest(claimed, fetched) or claimed or fetched
        verdict = {
            "verified_first_date": verified,
            "confidence": _CONFIDENCE_LOW,
            "evidence": ["fallback:min(claimed,fetched)"],
        }

    # --- 5. Persist (write-once; no-op if already frozen) ---
    store.put_verdict(content_hash, verdict)
    return verdict


# ---------------------------------------------------------------------------
# Task 6: poll_arxiv_versions — batched version-count read (§B.4.1 / §2.5)
# ---------------------------------------------------------------------------


def poll_arxiv_versions(arxiv_ids: list[str]) -> dict[str, int]:
    """Return {bare_arxiv_id: latest_version_count} via batched id_list reads (§B.4.1).

    Cheap deterministic Tier-0 read; <=100 ids/call. NEVER writes date_verdicts and
    NEVER changes verified_first_date (INV3). The monotonic max-merge into
    Story.arxiv_versions is Stage 3. Network/parse failures yield {} (caller keeps
    old counts), per the degrade-not-block policy (§B.4.1).
    """
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
