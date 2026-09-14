# ChatGPT ArXiv Paper Assistant

> *Last update: 2026-06-09*
> An enhanced version of the [GPT paper assistant](https://github.com/tatsu-lab/gpt_paper_assistant).
> Two daily pipelines -- a personalized arXiv paper digest and an AI-hotspots digest -- published as a multi-page static site.
> Now **agent-native**: a thin deterministic kernel orchestrates verifier-gated Claude Code subagents, and the whole system runs with **zero model API keys** through the local claude CLI.

See the [changelog](CHANGELOG.md) and the [upgrade notes](docs/UPGRADE-agent-native-hotspot.md) for the agent-native architecture and how to deploy it.

## Overview

Two complementary pipelines:

- **Personalized Daily Arxiv Paper** (`main.py`): fetch new arXiv papers, gate by author h-index, filter for relevance/novelty, render daily/monthly/yearly archives.
- **Daily AI Hotspots** (`arxiv_assistant/hotspots/kernel.py`): gather many AI sources (papers, lab blogs, roundups/news, GitHub, Hacker News, X, Reddit), verify dates, cluster into persistent stories, de-duplicate across days, score, and produce a concise daily "what matters today" digest.

Generated results are pushed to the `auto_update` branch; `main` stays code-only.

### What "agent-native" means here

- **Thin deterministic Kernel** drives a fixed 10-stage DAG (`harvest -> date_verify -> gravity_gate -> embed -> cluster -> storystore_match -> gapfill -> score -> synthesize -> render`) with per-`(date, stage)` JSON checkpoints (resumable, bit-stable), a single-writer story store, and clustering for cross-day de-duplication. Topology is in code -- never produced by an LLM.
- **Verifier-gated Claude Code subagents** sit only at judgment points; every agent is followed by a deterministic verifier so no agent output is trusted unverified (transport: `claude -p` via `arxiv_assistant/utils/agent_runner.run_agent`):
  - **DateVerify** -- per-item first-publication date (arXiv v1 / Crossref / Wayback, earliest-credible-wins), clamped by a verifier (fixes stale-item leakage).
  - **Synthesize** -- bilingual (en/zh) headline + summary per featured topic; the verifier rejects any row missing a bilingual field or citing evidence not in the story. (Gated by `[HOTSPOTS] use_synthesize_agent`, which defaults on only when `[HOTSPOTS] mode` is an LLM-enrich mode -- `llm`, or its archival alias `openai`; the agent itself runs on `claude -p` and needs no key either way. Otherwise headlines are heuristic.)
  - **Paper AgentFilter** -- relevance/novelty judged by a subagent; the verifier requires the evidence URL to reference the paper's own arXiv id.
  - **Gathering subagents** (see source routing) -- the URL-liveness verifier drops any fabricated/dead link.
- **Tiered, mostly-free source gathering**: most sources use cheap direct scrapers; the few protected/JS/login-walled ones are served by a Claude Code subagent (Web or browser) -- no per-source scraper rot.

## Run modes

**No model API key is needed in any mode.** Every model call in this repo goes through one gateway (`arxiv_assistant/utils/llm_gateway.py`) whose `[LLM] backend = auto` default prefers the keyless [`llmcall`](https://github.com/DaizeDong/llmcall) primitive (chain `codexg -> codex -> cc -> claude`) and falls back to this repo's own `claude -p` transport on a bare clone. OpenAI is a **legacy opt-in**, never automatic.

| Mode | Config | Keys needed | What runs |
|---|---|---|---|
| **Default** | `configs/config.ini` | **no model key**: `llmcall` if installed, else the `claude` CLI (logged in). Optional: twitterapi.io (X), S2/Slack | `api_only` paper filtering (batched relevance/novelty scoring, now carried by the gateway); twitterapi X; direct scrapers |
| **Zero-key agent-native** | `cp configs/profiles/agent-native.ini configs/config.ini` | **only the `claude` CLI (logged in) + a git push token** | `agent_only` paper filtering (per-paper `claude -p` verdicts); agent scout for X/breadth; subagent source routes (playwright) for reddit/CN-lab blogs; heuristic hotspot headlines |
| **Legacy OpenAI** (opt-in) | set `[LLM] backend = openai` and export `OPENAI_API_KEY` | OpenAI | the historical HTTP path. `auto` will never select this, by design: a dead key has to surface as an outage instead of quietly becoming the default again |

See [docs/UPGRADE-agent-native-hotspot.md](docs/UPGRADE-agent-native-hotspot.md) for the full agent-native story and deployment notes.

## Quickstart

### Run on GitHub Actions (default mode)

1. Fork/copy this repo and [enable scheduled workflows](https://docs.github.com/en/actions/using-workflows/disabling-and-enabling-a-workflow).
2. Edit the paper prompts under `prompts/paper/` (especially `prompts/paper/paper_topics.txt`) to match what you want to follow.
3. Copy `configs/templates/config.template.ini` to `configs/config.ini` and set your arXiv categories (`arxiv_category`).
4. **Register a self-hosted runner.** Every workflow here is `runs-on: self-hosted`, because the keyless backends are local programs: `llmcall` is not a PyPI dependency of this repo, and the repo-local fallback shells out to the `claude` CLI. A GitHub-hosted runner has neither and cannot be given either, so on `ubuntu-latest` the default `[LLM] backend = auto` resolves to the agent transport, finds no CLI, and every scoring call fails -- loudly, by design, but the digest for that day is empty.

   Register the runner on a machine where `llmcall` or the `claude` CLI is already logged in (Settings -> Actions -> Runners -> New self-hosted runner, or `gh api repos/OWNER/REPO/actions/runners/registration-token` for the token). Either OS works: every `run:` step declares `shell: bash`, because a Windows runner defaults `run:` to PowerShell and every script in these files is bash. A test enforces both properties.

   Two consequences worth knowing. **Scheduled runs queue while the machine is off** rather than failing, and fire when it comes back. And **publishing now depends on that machine too**, since all six workflows moved together; if you would rather keep GitHub Pages publishing alive independently, put `publish_md`, `push_results` and `refresh_x_authority_inventory` back on `ubuntu-latest` -- none of them calls a model.

   The alternative, if you ever want CI to run without that machine: set `[LLM] backend = openai` and provide `OPENAI_API_KEY` (+ `OPENAI_BASE_URL`) as [GitHub Secrets](https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions); see [GUIDE_GITHUB_API.md](GUIDE_GITHUB_API.md) for a free option. It costs money and reintroduces a key to rotate. Do **not** instead try to make the keyless chain work on a hosted runner by putting CLI credentials into Actions secrets: those are personal subscription credentials, and a hosted runner is not a place to send them.
5. Set GitHub Pages build source to [GitHub Actions](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site#publishing-with-a-custom-github-actions-workflow).
6. Copy `configs/templates/authors.template.txt` to `configs/authors.txt` and list authors (with their Semantic Scholar IDs).
7. **X/Twitter source** for hotspots (optional -- skip it for the zero-key mode): set `TWITTERAPI_IO_KEY` ([twitterapi.io](https://twitterapi.io), ~$0.15/1k tweets, no X dev account). Legacy `X_BEARER_TOKEN` is no longer used (the official-X and PaperPulse sources were retired).
8. Optional: `S2_KEY` (Semantic Scholar, speeds author lookup + lifts citation-signal rate limits) and `SLACK_KEY` + `SLACK_CHANNEL_ID` (Slack notifications).
9. Keep the repo private so Actions stay [active past 60 days](https://docs.github.com/en/actions/using-workflows/disabling-and-enabling-a-workflow).

> **Runtime note:** the default `[HOTSPOT_RUNTIME] runtime = local` means GitHub Actions **publishes** but does not **generate** hotspots (a VPS owns generation). If you run only on Actions with no VPS, set `runtime = actions` to keep CI generating hotspots. See the upgrade notes.

### Run on a VPS (zero-key agent-native)

For a no-API-key daily run through the local claude CLI:

```bash
cp configs/profiles/agent-native.ini configs/config.ini   # agent_only papers + agent scout + subagent routes
# install deploy/vps/ systemd unit + timer (headless `claude -p` cron); see deploy/vps/
```

Requirements: the `claude` CLI logged in, a git push token, the **playwright MCP** available to `claude -p` (for the browser-subagent sources), and -- for hard-login sites -- a pre-seeded cookie profile. No OpenAI or twitterapi keys are required.

### Running locally

Install `requirements.txt`, then copy `.env.example` to `.env` and set any keys you use.

```bash
# Personalized Daily Arxiv Paper
python main.py --output-root out --mode auto
python scripts/generate_monthly_summaries.py --output-root out --mode auto

# Daily AI Hotspots
python scripts/generate_daily_hotspots.py --output-root out --mode auto --force
python -m arxiv_assistant.renderers.build_multipage_site
```

`generate_daily_hotspots.py` also accepts `--stage <name>` to run a single kernel stage (resume helper) and `--date YYYY-MM-DD`.

## Backfilling missed days

`scripts/remedy_missed_dates.py` rebuilds any past day. It pulls that day's papers from the arXiv **API** with an explicit date window (the RSS feed only carries the current announcement), so a day is reproducible long after the fact, and it writes the same `out/json` and `out/md` artifacts a live run would.

```bash
# what would run, no API calls
python scripts/remedy_missed_dates.py --dates 2026-09-02,2026-09-03 --output-root ../archive/out --print-plan

# rebuild, six dates at a time
python scripts/remedy_missed_dates.py --dates "$(cat dates.txt)" --output-root ../archive/out --jobs 6
```

Point `--output-root` at a checkout of `auto_update`: that is where the archive lives, and the plan infers each day's search window from the days already recorded there.

`--jobs N` runs N dates concurrently in separate processes. Separate processes rather than threads because the pipeline keeps module-level state (the config singleton, the gateway's call ledger, the filter's rate-limit counters) that is not safe to share; each child is handed an explicit window so it never re-infers one from a tree its siblings are writing into. Children also skip the root `out/output.md` refresh, which is a "most recent run" convenience that means nothing for a backfill and would otherwise be a race.

**A backfill is where a silent failure costs the most**: it writes an authoritative-looking archive for a day nobody will look at again, once per date, unattended. So each date resets the call ledger, stamps `filter_health` and the providers that actually answered into its bundle, and marks itself `remedied`. The run ends with one list of the dates whose scoring never ran, and exits non-zero if there were any. An empty day in the archive should always be checkable against that record rather than assumed to mean nothing was relevant.

## Weekly digest / reader model

The daily pipelines still run daily and still write the full archive. What changed is what gets *read*: a weekly digest that keeps only material which would **change your mind**, not material that is merely on topic.

- **Reader model** -- `configs/reader/questions/q1..q5.md`, one open research question each, five fixed sections (`当前看法` / `支持证据` / `反对证据` / `什么会让我改看法` / `想做的实验`). These are edited **by hand only**; no agent and no pipeline stage ever writes to them, and a test enforces that.
- **Delta score** -- `prompts/reader/delta_scoring.txt` asks one question per item: would this edit one of those five fields? Not "is it relevant". Verdicts pass a deterministic verifier (known question id, known field, integer 0..10, a reason that names something specific) before they are trusted.
- **Cadence** -- `scripts/generate_weekly_digest.py` reads the last 7 days of `out/`, scores, keeps `delta_score >= [READER] delta_score_cutoff`, and writes `out/weekly/<YYYY-MM>/<date>-weekly.{md,json}`. `.github/workflows/weekly_digest.yaml` runs it Monday 08:00 America/New_York (two crons plus a timezone guard, because GitHub cron is UTC-only and DST makes one expression wrong for half the year). Daily Slack stays off; the weekly has its own `[OUTPUT] push_weekly_to_slack`.
- **Fixed length** -- at most 5 deep-read and 15 skim, ranked by delta score then by the pipeline's own score. No source table and no category expansion; those stay on the daily archive pages.
- **The archive is on another branch.** `out/` is gitignored on code branches, so pass `--archive-root` pointing at an `auto_update` checkout. Locally: `git worktree add ../archive auto_update`.

```bash
python scripts/generate_weekly_digest.py --archive-root ../archive --end-date 2026-09-09
python scripts/query_archive.py "MoE routing stability" --since 2026-08-01 --archive-root ../archive
```

**An empty digest always says why.** It prints the single line `本周没有改变看法的内容` in exactly one case: scoring ran, over a real window, against a populated reader model, and nothing cleared the bar. An empty reader model, an unreachable scorer, a window with no archive days, or an upstream filter that scanned papers while burning zero tokens each render a loud block naming the cause instead. This is not decoration: between 2026-06-05 and 2026-09-09 the paper pipeline published an empty `{}` archive every single day because an expired API key made every call raise into a bare `except`, and every run still exited 0. A digest that renders that as a quiet week is worse than no digest.

`scripts/query_archive.py` searches both archive trees (BM25, no new dependency, CJK bigrams so Chinese queries work), then cites a link for every conclusion and drops any URL the model invented. `--no-llm` gives ranked hits with no model access at all.

## How paper filtering works

`[PAPER_FILTER] mode` selects how surviving papers are scored (after the arXiv fetch + author h-index gate):

- `api_only` (default): the historical batched `filter_by_gpt` relevance/novelty scoring. The prompts, batching and score parsing are unchanged; what changed is the transport underneath -- `call_chatgpt` now dispatches through `utils/llm_gateway`, so the same batches run keyless over `llmcall` (or `claude -p`) instead of over an OpenAI key.
- `cascade`: cheap rule (h-index) -> API scoring -> escalate only the borderline band `[agent_borderline_low, agent_borderline_high)` to a Claude Code subagent.
- `agent_only`: every surviving paper judged one at a time by the subagent (`claude -p`), with the per-paper evidence verifier. Used by the zero-key profile.

Agent verdicts pass a deterministic verifier (schema + evidence must reference the paper's own arXiv id) before they are trusted. Paper-spotlight ranking can add a **free Semantic Scholar citation-significance** signal (`[HOTSPOTS] use_semantic_scholar_signal = true`, degrade-safe -- no behavior change when S2 is unavailable; note that brand-new papers have ~0 citations).

Whichever mode is active, **every model call is recorded in a per-process call ledger** (`llm_gateway.LEDGER`: attempted / succeeded / by backend / by provider / errors), and `main.py` feeds `LEDGER.attempted` and `LEDGER.succeeded` to `utils/pipeline_health`. When the ledger is populated it is authoritative and token counts are ignored -- the keyless backends report no OpenAI tokens, so the old token-based detector would have called every healthy run an outage. "We attempted N calls and none succeeded" is the backend-agnostic statement of an outage, and it is what makes "nothing matched" and "the model never ran" print differently.

The same reasoning governs the **usage table** at the top of each digest. A keyless provider answers without reporting tokens, so the counters stay at zero -- and a table printing `0` and `$0.00` reads as "this run was free" rather than "nobody counted it". Unreported usage now renders as `not reported`, and the model cell names the providers that actually answered (the same string as `meta.usage.model`) rather than the nominal `[SELECTION] model`, which only the legacy OpenAI path ever calls. Per-model prices come from the LiteLLM table, refreshed by `scripts/refresh_model_pricing.py` (cache in `.cache/`, snapshot fallback in `arxiv_assistant/utils/pricing.py`); the daily workflow refreshes it when the cache is over 24h old. Default model ids for the direct-agent paths live in one module, `arxiv_assistant/utils/models.py`, so a generation turnover is one edit rather than nine.

## How hotspots gather sources

Sources are routed by reliability, mostly for free:

- **Direct scrapers (free):** arXiv/HF papers, AI-lab blog RSS, analysis feeds, roundups, GitHub trending, Hacker News, AINews, local papers.
- **Browser subagent (`apis/hotspot/browser_source_fetch.py`, playwright, zero-key):** the known-protected/JS sources -- Reddit, the Cloudflare-walled xAI blog, and the Chinese-lab SPA blogs -- listed in the static `arxiv_assistant/utils/hotspot/source_routes.py` table and activated by `[HOTSPOT_SOURCES] use_subagent_routes` (on in the agent-native profile). This handles JS rendering, cookie/consent banners, and bot-walls a plain scraper or `WebFetch` cannot.
- **X:** `twitterapi.io` (`use_twitterapi`, metered) by default, or the zero-key **agent scout** (`use_agent_scout`) which web-searches across a curated venue matrix.
- **market-intel reuse:** when present, `arxiv_assistant/utils/market_intel_bridge.py` injects the [market-intel](https://github.com/DaizeDong/market-intel) skill's curated `frontier-research` + `x-twitter` source matrix into the scout prompt at runtime, so refreshing that skill automatically broadens the scout.

All gathered items flow through the same DateVerify -> de-dup -> score -> render path, so a fabricated or stale item cannot reach the report.

## Prompting

- `prompts/paper/paper_topics.txt` defines which papers the paper pipeline keeps; `prompts/paper/score_criteria.txt` controls relevance/novelty judging.
- Hotspot and monthly-summary prompts live under `prompts/hotspot/` and `prompts/monthly/`. See [prompts/README.md](prompts/README.md).

Be specific: describe the primary contribution types you want, and rule out downstream-application papers if precision matters more than recall.

## Acknowledgement

Originally built by Tatsunori Hashimoto, licensed under Apache 2.0.
Thanks to Chenglei Si for testing and benchmarking the GPT filter.
