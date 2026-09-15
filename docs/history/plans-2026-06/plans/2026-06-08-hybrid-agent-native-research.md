# Hybrid Agent-Native Daily Research — Implementation Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. One commit per task, conventional
> commits + `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer.

**Goal:** Add the agent gathering layer + a zero-config profile + the VPS daily cron, per
`docs/superpowers/specs/2026-06-05-hybrid-agent-native-research-design.md`. Build on PR #11; reuse
the kernel/Store/verifiers/DateVerify/render/AgentFilter unchanged. No API keys required at runtime.

**Ground truth to match (read before coding):**
- `arxiv_assistant/apis/hotspot/hotspot_twitterapi.py` — the peer source adapter shape
  (`fetch_hotspot_items(target_date, freshness_hours, seed_path, *, ...) -> list[HotspotItem]`,
  degrade-to-`[]`, `SOURCE_ID`/`PROVENANCE`, map provider JSON -> `HotspotItem`).
- `arxiv_assistant/utils/agent_runner.py` — `run_agent(prompt, *, schema, model, tools, timeout_s) -> dict`,
  raises `AgentError`. Pass `tools=["WebSearch","WebFetch"]` for the scout.
- `arxiv_assistant/utils/hotspot/hotspot_schema.py` — `HotspotItem` fields incl. `provenance`,
  `clean_text`.
- `arxiv_assistant/hotspots/pipeline.py` `fetch_source_payloads` — how sources are registered behind
  `[HOTSPOT_SOURCES]` flags (see the `x_twitterapi` registration added in Stage 5).
- Tests: `unittest.TestCase`, `@patch`, never hit the network (mock `run_agent` AND the URL check).

---

## Task 1: Agent Scout source `arxiv_assistant/apis/hotspot/hotspot_agent_scout.py`

**Create** the adapter:
- `SOURCE_ID = "agent_scout"`, `PROVENANCE = "agent:scout"`.
- `fetch_hotspot_items(target_date, freshness_hours, *, result_limit=40, model="claude-sonnet-4-6", timeout_s=300, agent_fn=run_agent, url_check_fn=_default_url_alive) -> list[HotspotItem]`
  (inject `agent_fn`/`url_check_fn` so tests never touch network/claude).
- Build a research prompt: find notable AI/ML developments (papers, model/product releases,
  significant X/social discussion, news) from the last `freshness_hours` hours; for each return
  `{title, url, summary, source_kind, published_at?}`. Scope to AI/ML; ask for the canonical source
  URL (arXiv/blog/repo/tweet permalink), not a search-results link.
- `schema = {"required":["items"], "properties":{"items":{"type":"array"}}}`.
- Call `agent_fn(prompt, schema=schema, model=model, tools=["WebSearch","WebFetch"], timeout_s=timeout_s)`.
- Map each item -> `HotspotItem(source_id="agent_scout", source_name="Agent Scout", source_role="agent_discovery", source_type="news", title, summary, url, canonical_url=url, published_at, tags=["agent-scout", source_kind], metadata={"provenance":PROVENANCE,"source_id":SOURCE_ID,...})`; set `.provenance` if the field exists.
- **Deterministic verifier (INV6):** drop any item whose `url` is not a syntactically valid
  http(s) URL on a non-blocklisted host, OR fails `url_check_fn(url)` (liveness). `_default_url_alive(url)`
  does a `requests.head` (fallback `get`) with a short timeout and accepts 2xx/3xx; arXiv/DOI ids
  count as alive without a fetch. A fabricated/dead link is discarded.
- Cap at `result_limit`; dedupe by canonical_url within the batch.
- Degrade to `[]` (with a `print` warning) on `AgentError`, empty, or any mapping failure — never raise.

**Tests** (`tests/test_hotspot_agent_scout.py`, mock `agent_fn` + `url_check_fn`):
- valid agent dict (2 live URLs) -> 2 `HotspotItem`s with `provenance=="agent:scout"`, correct fields.
- one item with a non-resolvable URL (`url_check_fn` returns False) -> dropped (anti-hallucination).
- one item with a non-http/garbage url -> dropped.
- `AgentError` -> `[]` (degrade-not-crash).
- `result_limit` honored; dedupe of duplicate canonical_url.

Commit: `feat(hotspot): agent scout source (claude -p web research + URL-liveness verifier)`

## Task 2: Register `agent_scout` + config flag

- In `pipeline.py` `fetch_source_payloads`, register `agent_scout` behind
  `hotspot_sources.getboolean("use_agent_scout", fallback=False)` (DEFAULT OFF — committed config
  unchanged), calling `fetch_agent_scout_items(target_date, freshness_hours, result_limit=..., timeout_s=...)`.
  Add a `SOURCE_USAGE_META` entry `{"provider":"claude-code","billing_model":"subscription"}`.
- Add `use_agent_scout = false` under `[HOTSPOT_SOURCES]` in BOTH `configs/config.ini` and
  `configs/templates/config.template.ini` (pure ASCII comments). Add `[HOTSPOTS]` keys
  `agent_scout_result_limit = 40`, `agent_scout_timeout_s = 300` (read by the registration).
- Tests: extend the pipeline source-registration test to assert `agent_scout` is registered when the
  flag is on and absent when off; key-gate -> no network. Keep existing pipeline tests green.

Commit: `feat(hotspot): register agent_scout source behind use_agent_scout flag`

## Task 3: Zero-config agent-native profile

- Create `configs/profiles/agent-native.ini` (a full config a user copies over `config.ini` for a
  no-key VPS deploy) with: `[PAPER_FILTER] mode = agent_only`; `[HOTSPOT_SOURCES] use_twitterapi = false`,
  `use_agent_scout = true`; `[HOTSPOTS] mode = heuristic` (Synthesize agent still runs);
  `[HOTSPOT_RUNTIME] runtime = local`; everything else mirroring `config.template.ini`. Pure ASCII.
- Document it in `docs/UPGRADE-agent-native-hotspot.md`: "Zero-key deployment: `cp configs/profiles/agent-native.ini configs/config.ini`; needs only the `claude` CLI (logged in) + a git push token; no OpenAI/twitterapi keys."
- Test: `configparser` parses the profile; assert the three key flags resolve as above.

Commit: `feat(config): zero-key agent-native profile (agent_only papers + agent scout, no API keys)`

## Task 4: VPS daily cron runs paper digest + hotspots

- Extend `deploy/vps/run_hotspot.sh`: BEFORE the hotspot generation, run the agent-native **paper
  digest** (`python main.py` — which with the agent-native profile filters papers via `agent_only`
  through `claude -p`, no OpenAI). Keep the existing hotspot kernel run, snapshot dump, audit-branch
  push, and `auto_update` publish. No API keys added to the unit (only the existing git push token).
  Add a short comment block explaining the no-key agent-native flow.
- Add a one-line note to `deploy/vps/hotspot.env.example` that OPENAI/TWITTERAPI keys are OPTIONAL
  (only needed if not using the agent-native profile).
- Test: `bash -n deploy/vps/run_hotspot.sh` (syntax) — no execution.

Commit: `feat(runtime): VPS cron runs agent-native paper digest + hotspots in one timer`

## Task 5: Full-suite regression

- `python -m pytest tests/ -q --ignore=tests/test_replay_diff.py` -> exactly the 5 known-debt
  `test_hotspot_web_data.py` failures, ZERO new (the new scout source is mocked in tests, no network).
- Confirm `import` of the new module + `main.py` + pipeline still clean; config files pure ASCII
  (`open(...).encode('ascii')`).
- No commit unless a fix was needed.

Commit (if needed): `test(hotspot): confirm agent-scout integration leaves suite green`
