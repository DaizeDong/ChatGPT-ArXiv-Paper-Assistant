# Changelog

### 2026-09-10

- **Retired the OpenAI API key path.** Every model call in the repo now goes through a single gateway, `arxiv_assistant/utils/llm_gateway.py`, which prefers the keyless [`llmcall`](https://github.com/DaizeDong/llmcall) primitive (chain `codexg -> codex -> cc -> claude`, ) and falls back to this repo's own `claude -p` transport (`utils/agent_runner`) so a bare clone with only the `claude` CLI still works. `llmcall` is detected by a guarded import, not by an env var and not by a vendored copy, so `requirements.txt` gains nothing.
- The new `[LLM]` section (`backend` / `effort` / `model` / `timeout_s`) selects the transport; `ARXIV_ASSISTANT_LLM_BACKEND` overrides it for a deploy host or a test. The default `backend = auto` picks `llmcall` when it imports and `agent` otherwise, and **never** picks `openai`: a dead key has to surface as an outage rather than quietly become the default again. OpenAI survives only as an explicit opt-in (`backend = openai` plus `OPENAI_API_KEY`), and the client is no longer constructed unless that backend is actually selected.
- `filter_by_gpt`'s paper scoring (`api_only`, still the default mode) now runs over the gateway with unchanged prompts, batching and score parsing, so the digest needs no API key.
- **Outage detection no longer reads token counts.** The gateway keeps a per-process call ledger (attempted / succeeded / by backend / by provider / capped errors) and `main.py` hands it to the new `utils/pipeline_health.assess_paper_filter_health`, where it is authoritative when present. This inversion is deliberate: the keyless backends report no OpenAI tokens, so the old token-based detector would have flagged every healthy run forever, while the failure it was written for (an expired key turning every call into a swallowed exception on a run that still exits 0) reads as `attempted > 0, succeeded == 0`.
- **Hotspot enrichment and screening left the HTTP path too.** `hotspots/enrich.py::_chat_completion` and `filters/filter_hotspots.py::_chat_completion` no longer `requests.post` at `/chat/completions`; both go through the gateway and keep their OpenAI-shaped return value so every caller is untouched. The system+user message pair is folded into the gateway's single prompt under explicit `===== SYSTEM INSTRUCTIONS =====` / `===== USER =====` banners, content verbatim, so the instruction semantics are unchanged. The configured `model` is echoed back but deliberately NOT forwarded: it is an OpenAI catalogue name that means nothing to the llmcall chain, and forwarding it would ask those backends for a model that does not exist.
- **`mode = openai` is now an alias, not a vendor.** `kernel.LLM_ENRICH_MODES = {"llm", "openai"}`; `llm` is the name for new configs and `openai` is kept because it is DATA, sitting inside 173 archived reports as `"mode": "openai"`. Renaming it would make those archives unreadable by their own reader.
- **The enrichment record now reaches the published report.** `_enrich` returns an `EnrichmentStatus` (`path`, `llm_ok`, batch/item counts, backend, provider, capped errors) and each row carries `enrich_source`. It was already written to the score checkpoint, but `_stage_synthesize` dropped it and `_stage_render` never wrote it, so the only artifact a later reader opens still said nothing about whether a model ran. Both stages now carry it, and `usage.llm` reports the provider that actually answered (`none` + `billing_model = disabled` when none did) instead of a hardcoded `"OpenAI"`. A total outage and a quiet news day still produce identical enriched rows -- that is the honest degrade -- so this record is the only thing that tells them apart.
- Added `tests/test_llm_gateway.py` (45 tests, no network): backend resolution precedence including a negative control that `auto` never resolves to `openai` while a key is present, both injected-backend call paths, schema passthrough, and ledger arithmetic under a thread pool.

### 2026-06-02

- X/Twitter hotspot source now supports **twitterapi.io** as a managed provider, selected automatically when `TWITTERAPI_IO_KEY` is set. Far cheaper than the official X API v2 (~$0.15/1k tweets vs ~$200/mo Basic), needs no X developer account, and works for new accounts. The official `X_BEARER_TOKEN` path remains as an automatic fallback — no config change required, fully backward compatible. Only the two network calls in `hotspot_x_official.py` were swapped behind a provider switch; all filtering/scoring logic is unchanged. No new dependency (reuses `requests`).

### 2026-04-04

- Fixed date semantics for HF Papers and GitHub sources: introduced `fetched_at` metadata to distinguish trending date from original publish date, ensuring correct freshness evaluation.
- Improved clustering quality: entity-sorted batching for better cross-batch `same_event_as` coverage, and added `SequenceMatcher` as a fallback similarity measure to catch synonym substitutions.
- Replaced hardcoded scoring normalization with P50/P95/max dynamic 3-segment mapping for better score differentiation across varying data distributions.
- Merged Source Feed into "Other Updates" in the frontend.

### 2026-04-02

- Repositioned Daily AI Hotspots as an artifact-centric executive brief: added artifact detection, substance penalty, rebalanced source weights and scoring toward official/research sources, and tightened screening to drop community-only noise.

### 2026-04-01

- Reformated the structure for daily arXiv papers by grouping them into multiple topics.
- Routed daily hotspot-worthy papers into the Daily AI Hotspots paper spotlight section.

### 2026-03-23

- Added daily hotspot usage reporting for OpenAI tokens/cost and external API request counts.
- Tightened hotspot filtering and X authority inventory handling to improve signal quality and quota stability.

### 2026-02-23

- Major updates in repository structure and pipeline features.
- Simplified the Daily AI Hotspots UI for better readability.

### 2026-03-22

- Added the Daily AI Hotspots pipeline, combining local selected papers, Hugging Face trending papers, roundup/news sites, official blogs, GitHub trends, and Hacker News discussions.

### 2026-03-21

- Support multi-page rendering!
- Support automatically integrating the latest models and prices from LiteLLM!

### 2025-10-16

- Added config template for minimal setup without author searching.

### 2025-9-23

- Fixed the connection error when `OPENAI_BASE_URL` isn't set.
- Updated the versions of some packages in `requirements.txt`.

### 2025-9-8

- Fixed a bug that caused gpt-5 failed to work.
- GPT-related exceptions will be printed all the time.

### 2025-9-5

- Added GPT-5 and updated prices.
- Updated guide for Github Copilot Free plan

### 2025-5-27

- Added system prompts for GPT filtering.

### 2025-4-3

- Added retries for ArXiv API calls.
- Rearranged date formats from `MM/DD/YYYY` to `YYYY-MM-DD`.

### 2025-3-25

- Supported the identification of title filtering prompts.
- Moved changelogs and prompt examples out of the readme file.

### 2025-3-20

- Supported getting papers on any specified date through arXiv API.
- Enhanced the logic of getting papers from RSS feeds.
- Added a script for remedying papers for missed dates.

### 2025-3-13

- Rearranged the file structure and cleaned some unused code snippets.

### 2025-2-19

- Added retrying for failed completion calls.
- Fixed the output file name, which will first follow ArXiv update time instead of local time.

### 2025-2-18

- Fixed a paper formatting bug which destroyed the performance of title filtering.
- Added retrying logic for GPT filtering so that there will be no paper missed.
- Added toggles that control the title/abstract filtering.
- Enhanced the debugging information by recording more logs and dumping more debug files.

### 2025-2-11

- Added a rate limit to API calls.

### 2025-2-3

- Fixed a bug that mistakenly filters all papers with high h-index.

### 2025-1-31

- Updated all github actions to the latest version.

### 2025-1-29

- Supported price calculation for cache tokens.
- Updated the price for `deepseek-chat` and `deepseek-reasoner`.

### 2025-1-28

- Fixed adaptive batch size when `paper_num <= adaptive_threshold`.
- Fixed the rename when `output.md` already exists.
- Added details in the return information for selected/filtered papers.

### 2025-1-25

- Fixed the exception when no paper is available.

### 2025-1-22

- Added a function that adaptively scales the `batch_size` by the number of papers.
- Supported detailed logging the cost of prompt and completion tokens.
- Adjusted the format of prompts to better utilize ChatGPT cache.

### 2025-1-21

- Fixed the auto-push workflow.
- Supported setting prompts for scoring.

### 2025-1-18

- Fixed the invalid retry logic for author searching.

### 2025-1-17

- Added a workflow that automatically pushes outputs to the `auto_update` branch.
- Added a toggle that decides whether to search authors before paper filtering.
- Rearranged the output directory, separating the formal outputs and debug logs.
- Enhanced the logging logic. Now it prints out more information about preserved papers and costs.

### 2025-1-10

- Set the version of `httpx` package to `0.27.2` for compatibility.
- Supported setting the `base_url` for OpenAI API.
- Supported counting costs for the latest GPT-4o series models.

### 2024-2-15

- Fixed a bug with author parsing in the RSS format.
- Cost estimates for title filtering being off.
- Crash when 0 papers are on the feed.

### 2024-2-7

- Fixed a critical issue from ArXiv changing their RSS format.
- Added and enabled a title filtering to reduce costs.
