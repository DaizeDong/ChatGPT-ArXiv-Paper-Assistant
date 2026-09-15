# Hybrid Agent-Native Daily Research — Design Spec

**Date:** 2026-06-05
**Builds on:** the agent-native hotspot rewrite (PR #11). Does NOT replace it — adds a
gathering layer + a zero-config profile + the daily VPS cron, reusing all the anti-pollution
scaffolding already built.

## Goal

A **daily, fully-automated Claude Code research run on a VPS** that needs **no API keys**:
Claude Code subagents do the broad multi-source gathering + relevance filtering, the existing
deterministic kernel keeps the output trustworthy (anti-stale / anti-duplicate / anti-hallucination),
and the result (daily arXiv paper digest + AI hotspots) is rendered and pushed to GitHub.

User decisions (2026-06-05):
- **Hybrid gathering** — keep the free, key-free, reliable feeds as a *floor*; use an agent
  fleet for the parts that needed keys or brittle scrapers (X/social, changing blogs) and add
  one broad "scout" web-search pass for discovery beyond the fixed source list.
- **Runs on VPS** via the `systemd` timer + headless `claude -p` entrypoint already in `deploy/vps/`.

## What is REUSED unchanged (the value of PR #11)

- **Kernel DAG** (`hotspots/kernel.py`) — fixed 10-stage flow, per-`(date,stage)` checkpoints.
- **StoryStore** — persistent cross-day identity -> deduplication (the #1 risk of "agent broad
  search" is repeats; the Store already solves it).
- **DateVerify** — per-item earliest-credible-date (arXiv v1 / Crossref / Wayback) -> anti-stale
  (the #2 risk of broad scraping is old content dressed as new).
- **Deterministic verifiers (INV6)** after every agent — anti-hallucination.
- **Paper AgentFilter** (`mode=agent_only`) — already does relevance/novelty judging via
  `claude -p`, **no OpenAI**. (Verified working: ReAct->keep, off-topic->drop.)
- **Synthesize agent** — bilingual headlines via `claude -p`.
- **Render** -> report.json + web_data + markdown (+ `_zh` translation walks resurgence).

## What is NEW (three pieces only)

### A. Agent Scout source — `arxiv_assistant/apis/hotspot/hotspot_agent_scout.py`
A kernel harvest source (peer of `hotspot_twitterapi`, the RSS feeds, etc.) that runs **one (or a
few) `claude -p` calls with web tools** to discover recent notable AI developments the
deterministic feeds miss — especially **X/social buzz** (replaces the `twitterapi.io` key) and
breaking releases/news.

- Transport: the existing `utils/agent_runner.run_agent(prompt, *, schema, model, tools, timeout_s)`,
  invoked with `tools=["WebSearch","WebFetch"]` and a research prompt scoped to the last
  `freshness_hours` window + AI/ML topicality.
- Returns a `{"items": [...]}` JSON list; each mapped to a `HotspotItem` with
  `source_id="agent_scout"`, `provenance="agent:scout"`, carrying the discovered `url`, title,
  one-line why-it-matters, and (when the agent found one) a date.
- **Deterministic verifier (INV6, anti-hallucination):** every scout item is dropped unless its
  `url` is (a) syntactically a real http(s) URL on a non-blocklisted host AND (b) survives a
  lightweight liveness/resolution check (HEAD/GET 2xx-3xx, or arXiv/DOI id resolves). Items
  without a resolvable URL are discarded — a fabricated link cannot enter the report.
- Degrades to `[]` on agent failure/timeout/empty (the free feeds remain the coverage floor).
- Gated by `[HOTSPOT_SOURCES] use_agent_scout` (default OFF in the committed config; ON in the
  agent-native profile).

Downstream: scout items flow through the SAME DateVerify -> gravity_gate -> dedup ->
score -> render, so they are date-clamped, de-duplicated against the Store, and quality-filtered
exactly like every other source. The scout never writes the Store directly (single-writer rule).

### B. Zero-config "agent-native" profile — `configs/profiles/agent-native.ini` (+ docs)
A config overlay a VPS user copies over `config.ini` for a **no-key** deployment:
- `[PAPER_FILTER] mode = agent_only` (paper relevance via agent, no OpenAI).
- `[HOTSPOT_SOURCES] use_twitterapi = false`, `use_agent_scout = true` (X via scout, no key).
- `[HOTSPOTS] mode = heuristic` for the deterministic enrich path (the **Synthesize** agent still
  runs for bilingual headlines), so no OpenAI is needed anywhere.
- `[HOTSPOT_RUNTIME] runtime = local` (VPS owns generation — already the default).
- LLM-extraction blog sources that require OpenAI degrade silently to static fetch (already do).
The committed `config.ini` defaults are unchanged (existing users unaffected); the profile is opt-in.

### C. VPS daily cron wiring — extend `deploy/vps/run_hotspot.sh`
The headless entrypoint already drives the hotspot kernel. Extend it to ALSO run the agent-native
**paper digest** (`python main.py` with the agent_native profile -> agent_only filtering) before
hotspots, so one `systemd` timer produces both the daily paper digest and the hotspots, fully
agent-driven, then pushes web_data + audit snapshot to GitHub. No secrets in the unit beyond the
git push token; no OpenAI/twitterapi keys required.

## Reliability, determinism, cost (the honest tradeoffs)

- **Determinism:** the scout source is *intentionally* non-deterministic (it discovers today's
  news). It is therefore **excluded from the replay-diff bit-stability gate**; everything
  downstream of harvest stays deterministic and bit-stable. The Store + DateVerify + verifier
  bound the pollution the scout could introduce.
- **Reliability:** the free deterministic feeds (arXiv API / HN / RSS / GitHub trending) remain
  the coverage **floor** — if the scout returns nothing, the day still produces a full report.
- **Cost:** a broad daily web-research run is token-heavy but runs on the **subscription** via
  `claude -p` (the reason for the VPS-headless design), not metered API. Bound it with a
  per-run agent count + timeout in config.
- **Web tools in headless:** the scout needs `WebSearch`/`WebFetch` available to `claude -p` on
  the VPS. If unavailable, the scout degrades to `[]` and logs it (free feeds carry the run).

## Acceptance

- `use_agent_scout=true` -> a `claude -p` scout runs, returns items, **every** emitted item has a
  verifier-passed resolvable URL (hallucinated links dropped), and they merge into the report
  through the normal dedup/date/score path.
- A no-key run with the agent-native profile produces a full paper digest (agent_only) + hotspots
  (with scout) end-to-end (the deterministic-feed floor + scout breadth).
- Existing committed `config.ini` behavior unchanged; full test suite stays green (the scout source
  is unit-tested with `run_agent` mocked — never hits the network in tests).
- `deploy/vps/run_hotspot.sh` runs paper digest + hotspots in one timer; no API keys in the unit.

## Out of scope (this increment)
- Full-agent gathering (removing all deterministic scrapers) — the hybrid keeps them as the floor.
- A front-end consumer for the resurgence section.
- The 5 designated-known-debt `test_hotspot_web_data.py` cases.
