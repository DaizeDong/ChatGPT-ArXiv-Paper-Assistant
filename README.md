# ChatGPT ArXiv Paper Assistant

> *Last update: 2026-03-22*  
> An enhanced version of the [GPT paper assistant](https://github.com/tatsu-lab/gpt_paper_assistant).  
> This repo now supports both a personalized daily arXiv digest and a daily AI hotspots digest, then publishes them as a multi-page static site.

See the [changelog](CHANGELOG.md) for recent changes and [the hotspot implementation plan](docs/DAILY_AI_HOTSPOTS_PLAN.md) for the new multi-source news pipeline.

## Overview

This repository has two complementary pipelines:

- `Paper digest`: fetch new arXiv papers, filter them with author rules and LLM scoring, then render daily, monthly, and yearly archives.
- `AI hotspots`: aggregate papers, official blogs, roundup/news sites, GitHub, and Hacker News, then use clustering plus LLM screening to produce a concise daily "what matters today" summary.

The generated results are pushed to the `auto_update` branch. The `main` branch should stay code-only.

## Main Features

- Personalized daily arXiv filtering with configurable prompts and score thresholds
- Daily AI hotspots digest built from multiple external signals
- Monthly paper summaries and monthly/yearly hotspot archives
- Multi-page static website with day/month/year navigation
- Automatic model pricing refresh from LiteLLM
- GitHub Actions workflows for daily runs, missed-date remediation, result sync, and Pages publishing

## Output Structure

Generated artifacts live under `out/` on the results branch:

- `out/json/`: daily selected paper results
- `out/md/`: daily paper markdown
- `out/monthly/`: structured monthly paper summary JSON
- `out/hot/reports/`: daily hotspot structured JSON reports
- `out/hot/md/`: hotspot markdown pages

Published site pages are rebuilt during CI and are not tracked in `main`.

## Quickstart

### Run on GitHub Actions

1. Fork or copy this repo.
2. Copy `prompts/templates/paper_topics.template.txt` to `prompts/paper_topics.txt` and describe the paper topics you care about.
3. Copy `configs/templates/config.template.ini` to `configs/config.ini` and adjust categories, thresholds, and source toggles.
4. Set `OPENAI_API_KEY` as a GitHub secret. If you need a custom-compatible endpoint, also set `OPENAI_BASE_URL`.
5. If you want to use GitHub Models, see [GUIDE_GITHUB_API.md](GUIDE_GITHUB_API.md).
6. Set the GitHub Pages publishing source to GitHub Actions.

Recommended extras:

7. Copy `prompts/templates/score_criteria.template.txt` to `prompts/score_criteria.txt` and tailor the scoring instructions.
8. Copy `configs/templates/authors.template.txt` to `configs/authors.txt` if you want author-based boosting.
9. Set `S2_KEY` if you want faster Semantic Scholar author lookup.
10. Set `GITHUB_TOKEN` permissions as required by the workflows. The default repository token is enough for Actions and Pages in this repo setup.
11. Optionally configure Slack secrets if you still want Slack delivery.

The main workflows are:

- `cron_runs`: generate the daily paper digest and daily hotspot digest
- `push_results`: sync generated `out/` artifacts to `auto_update`
- `publish_md`: rebuild the static site from `auto_update` data and deploy to Pages
- `remedy_missed_dates`: repair missing or incomplete dates and regenerate derived outputs

## Running Locally

Install dependencies from `requirements.txt`, then set environment variables such as `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `S2_KEY`, `SLACK_KEY`, and `SLACK_CHANNEL_ID` as needed.

Common commands:

```bash
python main.py
python scripts/generate_monthly_summaries.py --output-root out --mode auto
python scripts/generate_daily_hotspots.py --output-root out --mode auto --force
python -m arxiv_assistant.renderers.build_multipage_site
```

Useful notes:

- `main.py` generates the paper digest only. Derived site files are rebuilt separately.
- `scripts/generate_daily_hotspots.py` can also be run with `--mode heuristic` for local smoke tests without live LLM calls.
- `out/` is treated as generated data and should not be committed to `main`.

## Key Configuration Areas

The most important config sections are in `configs/config.ini`:

- `[FILTERING]`: arXiv categories, thresholds, author search, and paper filtering behavior
- `[OUTPUT]`: JSON/Markdown/Slack toggles
- `[HOTSPOTS]`: hotspot digest enablement, limits, and scoring controls
- `[HOTSPOT_SOURCES]`: source-level toggles for papers, roundups, blogs, GitHub, and Hacker News
- `[HOTSPOT_GITHUB]`: GitHub trend source settings
- `[HOTSPOT_HN]`: Hacker News source settings

The roundup site registry lives in `configs/hotspot_roundup_sites.json`.

## Daily AI Hotspots

The hotspot pipeline is designed to complement the paper digest rather than replace it.

Current source groups include:

- local selected arXiv papers
- Hugging Face trending papers
- AINews / roundup sites
- official research and vendor blogs
- GitHub trend signals
- Hacker News discussion signals

These sources are normalized, clustered into topics, screened with an LLM, and rendered into:

- daily hotspot pages
- monthly hotspot archive pages
- yearly hotspot archive pages

Implementation details and rollout phases are documented in [docs/DAILY_AI_HOTSPOTS_PLAN.md](docs/DAILY_AI_HOTSPOTS_PLAN.md).

## Prompting

`prompts/paper_topics.txt` defines what kinds of papers you want the paper pipeline to keep.  
`prompts/score_criteria.txt` controls how relevance and novelty are judged.  
Monthly summaries and daily hotspots also have their own prompt files under `prompts/`.

Being specific helps. Prefer describing the primary contribution types you want, and explicitly rule out downstream application papers if precision matters more than recall.

## How Paper Filtering Works

For the paper digest, the current pipeline is:

1. Fetch candidate arXiv papers for the target day.
2. Optionally resolve authors via Semantic Scholar.
3. Apply author-based matching and h-index gating.
4. Run title and abstract filtering with the configured OpenAI-compatible model.
5. Keep papers that pass relevance and novelty thresholds.
6. Render daily outputs and derive monthly/yearly views.

The hotspot pipeline is separate: it consumes multiple sources, clusters them into candidate topics, then uses LLM screening to decide which topics become the daily "frontier AI hotspots" report.

## Acknowledgement

This repo and code were originally built by Tatsunori Hashimoto and are licensed under the Apache 2.0 license.  
Thanks to Chenglei Si for testing and benchmarking the GPT filter.
