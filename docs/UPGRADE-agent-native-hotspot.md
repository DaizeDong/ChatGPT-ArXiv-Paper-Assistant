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
`cross_day_cosine_threshold`, `embed_model_id`, `max_item_age_days`, `resurge_*`),
`[HOTSPOT_SOURCES]` (`use_twitterapi` — the managed X channel that replaces the dead
`use_x_official`/`use_x_paperpulse`), `[HOTSPOT_REUSE]`, `[HOTSPOT_RUNTIME]`, `[PAPER_FILTER]`.
Config files are pure ASCII and all readers use UTF-8.

## Known limitations (conscious, documented)

- The kernel report's `usage`/`costs` LLM **token/cost are zero** (not threaded through the
  per-stage checkpoints); the **external** API usage (twitterapi.io etc.) is real.
- The **Resurgence** section is carried through the web payload, markdown, and `_zh`
  translator, but no front-end component renders it yet (i18n-ready; UI pending).
- A set of `tests/test_hotspot_web_data.py` assertions about a richer `source_section_totals`
  web-payload contract are **pre-existing known debt** (a separate web-UI concern), left red
  by deliberate decision.

## Zero-key (fully agent-native) deployment

For a VPS deploy that needs **no API keys at all**, copy the ready-made profile over your config:

```bash
cp configs/profiles/agent-native.ini configs/config.ini
```

This profile needs only the `claude` CLI (logged in) plus a git push token. With it:

- **Papers** are filtered by a Claude subagent (`[PAPER_FILTER] mode = agent_only`), so no
  OpenAI key is used by the digest (`python main.py`).
- **X/social + breadth** come from the **agent scout** (`[HOTSPOT_SOURCES] use_agent_scout = true`,
  `use_twitterapi = false`), which uses Claude's `WebSearch`/`WebFetch` instead of the metered
  twitterapi.io key.
- **Bilingual headlines** come from the Synthesize agent (hotspots run in `mode = heuristic`,
  so no OpenAI screening/enrichment is invoked).

No OpenAI or twitterapi keys are required. Everything runs on the Claude subscription.
