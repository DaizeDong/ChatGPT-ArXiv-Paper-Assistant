# ChatGPT ArXiv Paper Assistant

> *Last update: 2026-03-22*  
> An enhanced version of the [GPT paper assistant](https://github.com/tatsu-lab/gpt_paper_assistant).  
> This repo now supports both Personalized Daily Arxiv Paper and a daily AI hotspots digest, then publishes them as a multi-page static site.

See the [changelog](CHANGELOG.md) for recent changes and [the hotspot implementation plan](docs/DAILY_AI_HOTSPOTS.md) for the new multi-source news pipeline.

## Overview

This repository has two complementary pipelines:

- **Personalized Daily Arxiv Paper**: fetch new arXiv papers, filter them with author rules and LLM scoring, then render daily, monthly, and yearly archives.
- **Daily AI Hotspots**: aggregate papers, official blogs, roundup/news sites, GitHub, and Hacker News, then use clustering plus LLM screening to produce a concise daily "what matters today" summary.

The generated results are pushed to the `auto_update` branch. The `main` branch should stay code-only.

### Main Features

- Personalized daily arXiv filtering with configurable prompts and score thresholds
- Daily AI hotspots digest built from multiple external signals
- Monthly paper summaries and monthly/yearly hotspot archives
- Multi-page static website with day/month/year navigation
- Automatic model pricing refresh from LiteLLM
- GitHub Actions workflows for daily runs, missed-date remediation, result sync, and Pages publishing

## Quickstart

### Run on GitHub Actions

1. Copy/fork this repo to a new GitHub repo and [enable scheduled workflows](https://docs.github.com/en/actions/using-workflows/disabling-and-enabling-a-workflow) if you fork it.
2. Review the paper prompts under `prompts/paper/`, especially `prompts/paper/paper_topics.txt`, and edit them to match the kinds of papers you want to follow. If you want a clean starting point, use files in `templates` as references.
3. Copy `configs/templates/config.template.ini` to `configs/config.ini` and set your desired ArXiv categories `arxiv_category`.
4. Set your openai key `OPENAI_API_KEY` and base url `OPENAI_BASE_URL` (if you need one) as [GitHub Secrets](https://docs.github.com/en/actions/security-guides/using-secrets-in-github-actions#creating-secrets-for-a-repository). To can get a free one from GitHub, please reference [GUIDE_GITHUB_API.md](GUIDE_GITHUB_API.md).
5. In your repo settings, set GitHub page build sources to be [GitHub Actions](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site#publishing-with-a-custom-github-actions-workflow).

At this point, your bot should run daily and publish a static website. The results will be pushed to the `auto_update` branch automatically. You can test this by running the GitHub action workflow manually.

6. Copy `configs/templates/authors.template.txt` to `configs/authors.txt` and list the authors you actually want to follow. The numbers behind the author are important. They are semantic scholar author IDs which you can find by looking up the authors on semantic scholar and taking the numbers at the end of the URL.
7. Take a look at `configs/config.ini` to tweak how things are filtered.
8. Get and set up a `X_BEARER_TOKEN` as a GitHub secret, you can get one from X [Developer Console](https://console.x.com). This is for the hotspot pipeline to grab daily tweets.
9. Get and set up a semantic scholar API key (`S2_KEY`) as a GitHub secret. Otherwise the author search step will be very slow. (For now the keys are tight, so you may not be able to get one.)
10. Set up a [slack bot](https://api.slack.com/start/quickstart), get the OAuth key, set it to `SLACK_KEY` as a GitHub secret.
11. Make a channel for the bot (and invite it to the channel), get its [Slack Channel ID](https://stackoverflow.com/questions/40940327/what-is-the-simplest-way-to-find-a-slack-team-id-and-a-channel-id), set it as `SLACK_CHANNEL_ID` in a GitHub secret.
12. Set the GitHub repo private to avoid GitHub actions being [set to inactive after 60 days](https://docs.github.com/en/actions/using-workflows/disabling-and-enabling-a-workflow).

### Running Locally

Install dependencies from `requirements.txt`, then copy `.env.example` to `.env` and set environment variables as needed.

To generate **Personalized Daily Arxiv Paper**:

```bash
python main.py --output-root out --mode auto
python scripts/generate_monthly_summaries.py --output-root out --mode auto
```

To generate **Daily AI Hotspots**, run `scripts/generate_daily_hotspots.py`:

```bash
python scripts/generate_daily_hotspots.py --output-root out --mode auto --force
python -m arxiv_assistant.renderers.build_multipage_site
```

## Prompting

- `prompts/paper/paper_topics.txt` defines what kinds of papers you want the paper pipeline to keep.  
- `prompts/paper/score_criteria.txt` controls how relevance and novelty are judged.  
- Daily hotspots and monthly summaries live under `prompts/hotspot/` and `prompts/monthly/`. See [prompts/README.md](prompts/README.md) for the full layout, and `prompts/paper/example_prompt_structure.md` for a simple paper-prompt reference.

Being specific helps. Prefer describing the primary contribution types you want, and explicitly rule out downstream application papers if precision matters more than recall.

## How Paper Filtering Works

For **Personalized Daily Arxiv Paper**, the current pipeline is:

1. Fetch candidate arXiv papers for the target day.
2. Optionally resolve authors via Semantic Scholar.
3. Apply author-based matching and h-index gating.
4. Run title filtering through LLM API calls to remove obviously irrelevant papers.
5. Run abstract filtering through LLM API calls to score relevance and novelty, then rank papers by a weighted combination of these scores.
6. Keep papers that pass relevance and novelty thresholds.
7. Render daily outputs and derive monthly/yearly views.

The **Daily AI Hotspots** pipeline is separate:

1. Fetch daily signals from local selected papers, official blogs, roundup/news sites, GitHub, Hacker News, and configured X-related sources.
2. Normalize all fetched items into a shared hotspot schema and deduplicate obviously repeated links.
3. Cluster related items into candidate topics and compute deterministic quality, heat, importance, evidence, and confidence signals.
4. Apply confidence-aware routing so strong/weak topics are handled heuristically and only the ambiguous middle band is sent to the LLM.
5. Use the LLM to review borderline candidate topics, then synthesize a compact daily summary from the final featured set.
6. Keep only high-confidence featured topics for the top section, then expand the rest into source-first tables for broad coverage.
7. Render daily, monthly, and yearly hotspot archives, then publish them together with Personalized Daily Arxiv Paper.

## Acknowledgement

This repo and code were originally built by Tatsunori Hashimoto and are licensed under the Apache 2.0 license.  
Thanks to Chenglei Si for testing and benchmarking the GPT filter.
