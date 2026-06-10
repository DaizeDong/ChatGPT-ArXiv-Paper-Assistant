# ChatGPT ArXiv Paper Assistant

> *Last update: 2026-06-09*
> An enhanced version of the [GPT paper assistant](https://github.com/tatsu-lab/gpt_paper_assistant).
> Two daily pipelines -- a personalized arXiv paper digest and an AI-hotspots digest -- published as a multi-page static site.
> Now **agent-native**: a thin deterministic kernel orchestrates verifier-gated Claude Code subagents, and the whole system can run with **zero API keys** on the Claude subscription.

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
  - **Synthesize** -- bilingual (en/zh) headline + summary per featured topic; the verifier rejects any row missing a bilingual field or citing evidence not in the story. (Runs only when `[HOTSPOTS] mode = openai`; otherwise headlines are heuristic.)
  - **Paper AgentFilter** -- relevance/novelty judged by a subagent; the verifier requires the evidence URL to reference the paper's own arXiv id.
  - **Gathering subagents** (see source routing) -- the URL-liveness verifier drops any fabricated/dead link.
- **Tiered, mostly-free source gathering**: most sources use cheap direct scrapers; the few protected/JS/login-walled ones are served by a Claude Code subagent (Web or browser) -- no per-source scraper rot.

## Run modes

| Mode | Config | Keys needed | What runs |
|---|---|---|---|
| **Default** (unchanged from before) | `configs/config.ini` | OpenAI (paper filter + hotspot LLM screening), twitterapi.io (X), optional S2/Slack | `api_only` paper filtering; twitterapi X; direct scrapers |
| **Zero-key agent-native** | `cp configs/profiles/agent-native.ini configs/config.ini` | **only the `claude` CLI (logged in) + a git push token** | `agent_only` paper filtering (claude -p, no OpenAI); agent scout for X/breadth; subagent source routes (playwright) for reddit/CN-lab blogs; heuristic hotspot headlines |

The committed default is **byte-compatible with the previous behavior** -- existing users are unaffected. See [docs/UPGRADE-agent-native-hotspot.md](docs/UPGRADE-agent-native-hotspot.md) for the full agent-native story and deployment notes.

## Quickstart

### Run on GitHub Actions (default mode)

1. Fork/copy this repo and [enable scheduled workflows](https://docs.github.com/en/actions/using-workflows/disabling-and-enabling-a-workflow).
2. Edit the paper prompts under `prompts/paper/` (especially `prompts/paper/paper_topics.txt`) to match what you want to follow.
3. Copy `configs/templates/config.template.ini` to `configs/config.ini` and set your arXiv categories (`arxiv_category`).
4. Set `OPENAI_API_KEY` (+ `OPENAI_BASE_URL` if needed) as [GitHub Secrets](https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions). See [GUIDE_GITHUB_API.md](GUIDE_GITHUB_API.md) for a free option.
5. Set GitHub Pages build source to [GitHub Actions](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site#publishing-with-a-custom-github-actions-workflow).
6. Copy `configs/templates/authors.template.txt` to `configs/authors.txt` and list authors (with their Semantic Scholar IDs).
7. **X/Twitter source** for hotspots (optional -- skip it for the zero-key mode): set `TWITTERAPI_IO_KEY` ([twitterapi.io](https://twitterapi.io), ~$0.15/1k tweets, no X dev account). Legacy `X_BEARER_TOKEN` is no longer used (the official-X and PaperPulse sources were retired).
8. Optional: `S2_KEY` (Semantic Scholar, speeds author lookup + lifts citation-signal rate limits) and `SLACK_KEY` + `SLACK_CHANNEL_ID` (Slack notifications).
9. Keep the repo private so Actions stay [active past 60 days](https://docs.github.com/en/actions/using-workflows/disabling-and-enabling-a-workflow).

> **Runtime note:** the default `[HOTSPOT_RUNTIME] runtime = local` means GitHub Actions **publishes** but does not **generate** hotspots (a VPS owns generation). If you run only on Actions with no VPS, set `runtime = actions` to keep CI generating hotspots. See the upgrade notes.

### Run on a VPS (zero-key agent-native)

For a no-API-key daily run on the Claude subscription:

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

## How paper filtering works

`[PAPER_FILTER] mode` selects how surviving papers are scored (after the arXiv fetch + author h-index gate):

- `api_only` (default): the historical OpenAI `filter_by_gpt` relevance/novelty scoring -- byte-identical to before.
- `cascade`: cheap rule (h-index) -> API scoring -> escalate only the borderline band `[agent_borderline_low, agent_borderline_high)` to a Claude Code subagent.
- `agent_only`: every surviving paper judged by the subagent (`claude -p`, no OpenAI). Used by the zero-key profile.

Agent verdicts pass a deterministic verifier (schema + evidence must reference the paper's own arXiv id) before they are trusted. Paper-spotlight ranking can add a **free Semantic Scholar citation-significance** signal (`[HOTSPOTS] use_semantic_scholar_signal = true`, degrade-safe -- no behavior change when S2 is unavailable; note that brand-new papers have ~0 citations).

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
