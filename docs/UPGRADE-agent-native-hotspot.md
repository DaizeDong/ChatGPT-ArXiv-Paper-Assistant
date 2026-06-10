# Upgrade Notes — Agent-Native Hotspot Rewrite

This branch rewrites the hotspot pipeline around a thin deterministic **Kernel** (fixed
10-stage DAG with per-`(date, stage)` checkpoints) plus short-lived, verifier-gated
**Claude Code subagents** at the judgment points, and adds an agent filtering modality to
the paper pipeline. It is designed so an existing user with an **unmodified config** keeps
working, but two behaviors changed by default and are worth knowing before you upgrade.

## TL;DR for existing users

| Area | Default after upgrade | Action needed |
|---|---|---|
| Paper filtering | `[PAPER_FILTER] mode = api_only` = **byte-identical** to before | none |
| Hotspot in-process call (`generate_daily_hotspot_report`) | still returns a report dict | none |
| **CI hotspot generation** | `[HOTSPOT_RUNTIME] runtime = local` -> GitHub **Actions no longer generates** hotspots (a VPS owns it) | **Actions-only users: set `runtime = actions`** (see below) |
| Hotspot Synthesize headlines | claude -p subagent in `openai` mode, heuristic otherwise | optional: set `model_synthesize` |

## 1. Runtime ownership moved to a VPS (the one behavior change to know)

The design moves daily hotspot **generation** to a local/VPS host running Claude Code
headless (`claude -p`) on a `systemd` timer (see `deploy/vps/`), and demotes GitHub Actions
to a pure **Publisher**. The new `[HOTSPOT_RUNTIME] runtime` key controls this:

- `runtime = local` (the new default): the VPS generates; the `cron_runs.yaml` workflow
  **skips** hotspot generation and only publishes.
- `runtime = actions`: the workflow runs a **deterministic, agents-off heuristic**
  generation in CI, as before.

**If you run only on GitHub Actions (no VPS), set this to keep CI generating hotspots:**

```ini
[HOTSPOT_RUNTIME]
runtime = actions
```

The paper pipeline (`main.py`) is unaffected by this key and continues to run in CI.

## 2. Paper filtering gained an agent modality (opt-in, zero default change)

`[PAPER_FILTER] mode` selects how surviving papers are scored:

- `api_only` (default): the historical `filter_by_gpt` path, unchanged.
- `cascade`: cheap Rule (h-index) -> Api scoring -> escalate only the **borderline** band
  `[agent_borderline_low, agent_borderline_high)` to a Claude Code subagent.
- `agent_only`: every survivor judged by the subagent.

Agent verdicts are validated by a deterministic verifier (schema + evidence must reference
the paper's own arXiv id) before they are trusted; a failed/hallucinated verdict
conservatively drops the paper. The agent transport is `claude -p` headless
(`utils/agent_runner.run_agent`); set the model with `[PAPER_FILTER] agent_model`
(default `claude-sonnet-4-6`).

## 3. Agent touchpoints (all verifier-gated — INV6)

Every subagent is followed by a deterministic verifier; no agent output is used unverified:

- **DateVerify** — per-item first-publication date (arXiv v1 / Crossref / Wayback,
  earliest-credible-wins), clamped by a deterministic verifier. Fixes stale-item leakage.
- **Synthesize** — bilingual (en/zh) headline + summary per featured topic, in `openai`
  mode, via `claude -p`; the verifier rejects any row missing a bilingual field or citing
  evidence not already in the story (anti-hallucination). In `heuristic` mode (or on agent
  failure) it degrades to deterministic heuristic headlines. Set `[HOTSPOTS] model_synthesize`
  to pin the model.
- **Paper AgentFilter** — see section 2.

## 4. New config sections (all have safe fallbacks)

All new keys exist in both `configs/config.ini` and `configs/templates/config.template.ini`
with defaults that preserve current behavior: `[HOTSPOTS]` (date/dedup/resurge keys incl.
`cross_day_cosine_threshold`, `embed_model_id`, `max_item_age_days`, `resurge_*`, plus
`use_semantic_scholar_signal`, `agent_scout_*`, `subagent_source_*`), `[HOTSPOT_SOURCES]`
(`use_twitterapi` -- the managed X channel; `use_agent_scout`, `use_market_intel_sources`,
`use_subagent_routes`), `[HOTSPOT_REUSE]`, `[HOTSPOT_RUNTIME]`, `[PAPER_FILTER]`. The retired
`use_x_official`/`use_x_paperpulse` keys were **removed** (see section 7). Config files are pure
ASCII and all readers use UTF-8.

## Known limitations (conscious, documented)

- The kernel report's `usage`/`costs` LLM **token/cost are zero** (not threaded through the
  per-stage checkpoints); the **external** API usage (twitterapi.io etc.) is real.
- The **Resurgence** section is carried through the web payload, markdown, and `_zh`
  translator, but no front-end component renders it yet (i18n-ready; UI pending).
- A set of `tests/test_hotspot_web_data.py` assertions about a richer `source_section_totals`
  web-payload contract are **pre-existing known debt** (a separate web-UI concern), left red
  by deliberate decision (5 failing tests).
- The Semantic Scholar citation signal is **~0 for brand-new papers** (too recent to be cited);
  it differentiates older/resurfaced/already-cited papers, and is harmless (0 bonus) otherwise.
- The browser subagent needs the **playwright MCP** exposed to `claude -p` (verified available
  when spawned from a Claude context; confirm it on the headless VPS cron). **Hard-login** sites
  need a pre-seeded cookie profile; the prompt only handles soft walls / consent banners.
- **Bright Data** is connected at the CLI but its tools are **not exposed to a `claude -p`
  subagent**, so the browser route uses playwright (not Bright Data) for protected sources.

## Zero-key (fully agent-native) deployment

For a VPS deploy that needs **no API keys at all**, copy the ready-made profile over your config:

```bash
cp configs/profiles/agent-native.ini configs/config.ini
```

This profile needs only the `claude` CLI (logged in) plus a git push token. With it:

- **Papers** are filtered by a Claude subagent (`[PAPER_FILTER] mode = agent_only`). Note the
  profile keeps `[SELECTION] run_openai = true`: that flag is the outer gate that *enables* the
  paper-filter step in `main.py`; `mode = agent_only` then routes it to the subagent (claude -p)
  instead of OpenAI. The `agent_only` path never calls the API scorer, so no OpenAI key is used
  by the digest (`python main.py`).
- **X/social + breadth** come from the **agent scout** (`[HOTSPOT_SOURCES] use_agent_scout = true`,
  `use_twitterapi = false`), which uses Claude's `WebSearch`/`WebFetch` instead of the metered
  twitterapi.io key.
- **Protected/JS sources** (Reddit, the xAI blog, the Chinese-lab SPA blogs) come from the
  **browser subagent** (`use_subagent_routes = true`); see section 5.
- **Headlines** are deterministic-heuristic in this profile. The hotspots run in `mode = heuristic`,
  so neither the OpenAI screening/enrichment nor the bilingual **Synthesize agent** runs (the
  Synthesize agent is gated on `mode = openai`). To get agent bilingual headlines you would set
  `mode = openai`, which then also needs an OpenAI key for enrichment -- so the zero-key profile
  deliberately uses heuristic headlines.

No OpenAI or twitterapi keys are required. Everything runs on the Claude subscription.

## 5. Tiered source gathering (most sources free; protected ones via subagent)

Gathering is statically routed by reliability -- there is no per-source "try direct then fall back"
machinery; each source is assigned its best tool once:

- **Direct scrapers (free, default):** arXiv/HF papers, lab-blog RSS, analysis feeds, roundups,
  GitHub trending, Hacker News, AINews, local papers.
- **Browser subagent** (`apis/hotspot/browser_source_fetch.py`, playwright `claude -p`, zero-key):
  the known-fail sources listed in `arxiv_assistant/utils/hotspot/source_routes.py`
  (`SUBAGENT_ROUTES`) -- Reddit (bot-wall 403), the Cloudflare-walled xAI blog, and the
  Chinese-lab SPA blogs (Zhipu / ByteDance Seed / Baichuan / 01.AI / StepFun / jiqizhixin).
  Gated by `[HOTSPOT_SOURCES] use_subagent_routes` (default `false`; `true` in the agent-native
  profile). When on, Reddit comes only from the browser subagent (the direct scraper is
  suppressed -- no double-fetch). The browser prompt renders JS, scrolls, and dismisses
  cookie/consent and soft login-walls; the same URL-liveness verifier drops fabricated links.
- **X:** `twitterapi.io` (metered, default) or the zero-key agent scout.
- **market-intel reuse:** `arxiv_assistant/utils/market_intel_bridge.py` injects the `market-intel`
  skill's curated `frontier-research` + `x-twitter` source matrix into the scout prompt at runtime
  (skill refresh auto-broadens the scout; falls back to a built-in venue list if the skill is absent).

The `WebFetch` agent fetcher `apis/hotspot/agent_source_fetch.py` exists as a building block for
standard-but-brittle sources; plain `WebFetch` cannot bypass bot-walls/JS, so protected sources use
the browser route.

## 6. Free quality signal: Semantic Scholar citations in paper spotlight

`[HOTSPOTS] use_semantic_scholar_signal = true` (default) adds a free, no-key Semantic Scholar
citation-significance signal (`influentialCitationCount` / `citationCount`, one batched `paper/batch`
request) to the paper-spotlight ranking, distinguishing genuinely-cited work from upvote-only items.
Degrade-safe: when S2 is unavailable the bonus is 0 for all papers and ranking is identical to the
baseline. An optional `semantic_scholar_api_key` (or `S2_API_KEY` env) lifts the rate limit.

## 7. Deprecations removed in this branch

- **`x_official` and `x_paperpulse` sources** -- removed entirely (modules, registrations, config
  keys); superseded by `use_twitterapi` (the official-X path needed X API Pro ~$5k/mo; the
  PaperPulse upstream feed is dead).
- **The OpenAI `playwright_llm` blog-extraction mode** -- removed; the Chinese-lab SPA blogs that
  used it now route to the zero-key browser subagent, so **hotspot gathering no longer needs an
  OpenAI key** at all.
- **`source_escalation.py`** (a dynamic detect-and-escalate framework) -- dropped in favor of the
  simpler static `source_routes.py`.
- **`semantic_scholar.get_author_batch`** -- removed (unused).
