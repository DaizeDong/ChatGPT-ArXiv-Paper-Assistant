# Daily AI Hotspots

Technical Report for the current hotspot-ranking algorithm in this repository.  
Last updated: 2026-03-22

## Abstract

This document describes the algorithm behind the `Daily AI Hotspots` system implemented in this repository.

The original repository is strong at selecting specialized arXiv papers, but that alone is not sufficient to answer a different product question: *what is the AI field broadly reacting to today?* A daily hotspot system needs to combine high-quality technical sources, social/editorial resonance, open-source movement, and official announcements, then compress them into a small number of distinct topics.

The implemented system therefore operates as a separate pipeline. It aggregates heterogeneous signals, normalizes them into a shared schema, clusters overlapping evidence into candidate topics, scores those topics with a hybrid deterministic plus LLM procedure, and renders a compact daily report together with monthly and yearly archives.

The design goal is not exhaustive news coverage. The goal is a selective, technically serious front page of daily frontier AI topics.

## 1. Problem Definition

The hotspot task differs from the paper-digest task in three important ways:

1. The target unit is a **topic**, not a paper.
2. The ranking objective includes **quality and heat together**, not just technical relevance.
3. Evidence may come from different source classes: papers, blogs, roundup sites, repositories, and discussion threads.

In practical terms, the system should answer:

- What are the most important AI topics today?
- Which concrete items are driving each topic?
- Why does each topic matter?
- Which topics deserve the limited space of a daily front page?

## 2. Design Principles

The implemented algorithm follows these principles:

- The main list should be selective. Publishing fewer topics is preferable to filling the page with weak material.
- Heat matters, but heat alone is not enough. A topic should ideally have technical substance, concrete artifacts, or credible supporting evidence.
- Different source families should contribute different kinds of evidence.
- The system should degrade gracefully when some sources fail or when an OpenAI key is unavailable.
- Generated outputs should remain reconstructible from `out/` on the results branch, while `main` stays code-only.

## 3. System Overview

The hotspot pipeline runs after the daily paper pipeline and is implemented in [generate_daily_hotspots.py](C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/scripts/generate_daily_hotspots.py).

At a high level, the pipeline is:

1. Fetch raw items from enabled source adapters.
2. Normalize all items into a shared `HotspotItem` schema.
3. Remove exact duplicates and stale items.
4. Cluster overlapping items into candidate topics.
5. Compute deterministic cluster scores.
6. Trim to a manageable candidate set.
7. Screen candidates either:
   - with OpenAI, or
   - with a heuristic fallback when OpenAI is unavailable.
8. Apply diversity-aware final topic selection.
9. Synthesize the final daily summary.
10. Write structured outputs and render markdown pages.

The pipeline supports three execution modes:

- `openai`: force LLM screening and synthesis
- `heuristic`: force deterministic fallback behavior
- `auto`: use OpenAI when `OPENAI_API_KEY` is present, otherwise fall back to heuristic mode

## 4. Source Families

The system deliberately combines multiple source classes. Each source contributes a different kind of signal.

| Source family | Primary role | Typical evidence |
| --- | --- | --- |
| Local selected arXiv papers | research backbone | papers already selected by the personalized paper pipeline |
| Hugging Face trending papers | paper popularity | externally trending papers, paper upvotes, linked GitHub repos |
| AINews | community heat | topics repeated across X, Reddit, Discord, and linked posts |
| Roundup/news sites | headline consensus / builder momentum / editorial depth | repeated newsletter coverage of launches, tools, and discussions |
| Official blogs | official release signal | OpenAI, Anthropic, Google, Meta announcements and research posts |
| GitHub trend search | open-source adoption | fast-rising repos, demos, frameworks, and model releases |
| Hacker News | technical discussion | AI-relevant discussions with visible score and comment depth |

The currently enabled roundup-site registry is stored in [hotspot_roundup_sites.json](C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/configs/hotspot_roundup_sites.json).

## 5. Unified Data Model

All source adapters emit the same normalized object type, defined in [hotspot_schema.py](C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/arxiv_assistant/utils/hotspot_schema.py).

### 5.1 HotspotItem

Each raw item is converted into a `HotspotItem` with the following logical fields:

- `source_id`: stable adapter identifier
- `source_name`: human-readable source name
- `source_role`: semantic role such as `research_backbone`, `community_heat`, or `official_news`
- `source_type`: broad source type such as `paper`, `roundup`, `official_blog`, `repo`, or `discussion`
- `title`
- `summary`
- `url`
- `canonical_url`
- `published_at`
- `tags`
- `authors`
- `metadata`

Normalization cleans whitespace, normalizes URLs, and strips tracking parameters from query strings.

### 5.2 HotspotCluster

After clustering, each candidate topic is represented as a `HotspotCluster` containing:

- `cluster_id`
- `title`
- `canonical_url`
- `summary`
- `items`
- `source_ids`
- `source_names`
- `source_roles`
- `source_types`
- `tags`
- `published_at`
- `deterministic_score`

The hotspot system therefore reasons about *clusters* rather than isolated raw items.

## 6. Candidate Collection and Pre-filtering

Raw fetching is adapter-specific, but several constraints are applied consistently.

### 6.1 Freshness control

The pipeline only admits items within a configurable freshness window. The current default is controlled in [config.ini](C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/configs/config.ini):

- `freshness_hours = 36`
- `local_papers_max_staleness_days = 2`

This prevents the daily report from drifting into stale commentary or backfilled paper results.

### 6.2 Raw-item caps

To control cost and keep downstream ranking stable, the pipeline caps the total number of raw items before clustering. The current default is:

- `max_raw_items = 120`

This cap matters because the hotspot system is quality-sensitive: more raw items do not necessarily produce better topics.

### 6.3 Exact deduplication

Before clustering, the pipeline performs strict deduplication on:

- `source_id`
- canonical URL
- normalized title

When multiple copies of the same item exist, the longer summary is preferred.

## 7. Topic Clustering

Clustering is implemented in [hotspot_cluster.py](C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/arxiv_assistant/utils/hotspot_cluster.py).

This stage is central to the algorithm. The system does not rank raw links directly; it ranks merged topics.

### 7.1 Matching logic

Two items are merged when they are clearly the same underlying topic. The cluster match function uses the following priority:

1. Exact canonical URL match
2. Exact arXiv ID match
3. Exact GitHub repo URL match
4. Title-based similarity

Title similarity is computed from normalized tokens after removing stopwords. A separate overlap boost is used when the shared overlap is unusually specific.

### 7.2 Generic-token suppression

A common failure mode in AI-news aggregation is false merging due to generic shared words like `open`, `agent`, `model`, or `benchmark`.

To reduce that failure mode, the clusterer excludes a blacklist of generic overlap tokens before applying overlap boosts. This means two unrelated headlines will not be merged just because they both mention generic AI vocabulary.

### 7.3 Similarity threshold

The current clusterer uses a merge threshold of `0.68`. Items are processed in descending recency and source-role order, then assigned to the first bucket whose seed item exceeds the threshold.

This greedy strategy is intentionally simple and inspectable. It is sufficient because upstream item quality is already heavily filtered.

## 8. Deterministic Cluster Scoring

Each cluster receives a deterministic score before any LLM call. This score is used for candidate trimming.

### 8.1 Source-role weights

The current source-role prior is:

| Source role | Weight |
| --- | ---: |
| `research_backbone` | 5.4 |
| `official_news` | 5.0 |
| `paper_trending` | 4.8 |
| `community_heat` | 4.5 |
| `github_trend` | 4.2 |
| `headline_consensus` | 4.0 |
| `builder_momentum` | 3.8 |
| `editorial_depth` | 3.4 |
| `hn_discussion` | 3.0 |

These weights do not directly determine the final ranking. They are only priors used to decide which clusters are worth deeper inspection.

### 8.2 Deterministic score structure

The deterministic score is a weighted sum of:

- source-role priors
- number of distinct source IDs
- number of distinct source types
- signal metrics such as:
  - paper `daily_score`
  - Hugging Face upvotes
  - community activity counts
  - GitHub stars
  - Hacker News score
- bonuses for:
  - official-source evidence
  - GitHub evidence

This score is stored as `cluster.deterministic_score` and is used only for candidate selection, not for the final front-page order.

## 9. Candidate Trimming Before LLM

The system intentionally does not send every cluster to the model.

Candidate trimming is implemented in `deterministic_trim()` in [generate_daily_hotspots.py](C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/scripts/generate_daily_hotspots.py).

The procedure:

1. Sort clusters by deterministic score.
2. Reserve budget by source role.
3. Fill the remaining capacity with the highest-scoring leftovers.

The current role budgets are:

- `research_backbone`: 3
- `paper_trending`: 5
- `official_news`: 3
- `headline_consensus`: 3
- `editorial_depth`: 2
- `community_heat`: 2
- `builder_momentum`: 1
- `github_trend`: 2
- `hn_discussion`: 2

The current default LLM candidate cap is:

- `max_clusters_for_llm = 18`

This stage is important because it preserves source diversity before the expensive screening step.

## 10. Screening and Scoring

Screening logic lives in [filter_hotspots.py](C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/arxiv_assistant/filters/filter_hotspots.py).

The pipeline supports two screening modes:

- `heuristic_screen_clusters()`
- `screen_clusters_with_openai()`

### 10.1 Heuristic signal model

For each cluster, the heuristic model computes:

- `FRONTIERNESS`
- `TECHNICAL_DEPTH`
- `CROSS_SOURCE_RESONANCE`
- `IMPORTANCE`
- `EVIDENCE_STRENGTH`
- `ACTIONABILITY`
- `HYPE_PENALTY`

These signals are derived from the cluster text, source roles, source types, and metadata such as stars, upvotes, HN score, and community activity.

The current heuristic categories are:

- `Research`
- `Product Release`
- `Tooling`
- `Industry Update`
- `Community Signal`

### 10.2 First-stage keep / watchlist decision

The heuristic screener first computes a cluster-level score:

```text
stage1_score =
  0.28 * quality
  + 0.28 * heat
  + 0.24 * importance
  + 0.10 * actionability
  + 0.10 * evidence_strength
  - 0.05 * hype_penalty
```

This stage is used to decide:

- main-list keep
- watchlist keep
- reject

Current default thresholds:

- `screening_score_cutoff = 4.0`
- `watchlist_score_cutoff = 3.3`

### 10.3 Final topic score

After category assignment and summary construction, each topic is assigned a second score used for downstream ranking:

```text
final_topic_score =
  0.34 * quality
  + 0.30 * heat
  + 0.24 * importance
  + 0.08 * actionability
  + 0.04 * evidence_strength
  - 0.08 * hype_penalty
```

This second score is intentionally more selective against hype and slightly more aggressive about heat.

## 11. OpenAI Screening

When `OPENAI_API_KEY` is available and mode permits it, the same clusters are screened by an OpenAI-compatible model instead of pure heuristics.

The current model defaults are:

- `model_screen = gpt-5.4`
- `model_summarize = gpt-5.4`

The LLM receives:

- cluster title
- deterministic score
- source names
- source roles
- source types
- tags
- representative evidence items

It returns structured JSONL containing:

- `KEEP`
- `WATCHLIST`
- `CATEGORY`
- `QUALITY`
- `HEAT`
- `IMPORTANCE`
- `SUMMARY`
- `WHY_IT_MATTERS`

If LLM screening fails for a batch, the pipeline automatically falls back to heuristic screening for that batch rather than aborting the whole day.

## 12. Diversity-aware Final Selection

After screening, the system still does not simply take the top `N` topics. A separate trimming stage enforces diversity and reduces collapse into paper-only front pages.

This logic is implemented in `_trim_topics()` in [generate_daily_hotspots.py](C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/scripts/generate_daily_hotspots.py).

### 12.1 Output targets

The current default targets are:

- `target_topics = 5`
- `min_topics = 3`
- `target_watchlist_topics = 3`

### 12.2 Bucket-aware selection

Topics are mapped into broad buckets:

- `official`
- `tooling`
- `research`
- `community`

The selection logic:

1. Reserves room for at least one `official`, one `community`, and one `research` slot when available.
2. Applies per-category and per-source caps.
3. Caps paper-heavy topics so the page does not degrade into a paper-only digest.
4. Forces at least one real heat/community slot when available.
5. Falls back to watchlist promotion if the main list is too thin.

This is the main mechanism that keeps the product balanced between quality and heat.

## 13. Final Digest Synthesis

Once the final topics are selected, the system runs one final synthesis step.

This stage:

- assigns a polished `HEADLINE`
- produces `WHY_IT_MATTERS`
- emits `KEY_TAKEAWAYS`
- writes the short page-level executive summary

When OpenAI is unavailable, the system falls back to a deterministic sentence built from the top three topic headlines.

## 14. Output Artifacts

The hotspot system writes several layers of outputs under `out/hot/`:

```text
out/
  hot/
    raw/
    normalized/
    clusters/
    reports/
    md/
```

Concretely:

- `out/hot/raw/YYYY-MM-DD/*.json`: per-source cached raw fetches
- `out/hot/normalized/YYYY-MM-DD.json`: normalized items
- `out/hot/clusters/YYYY-MM-DD.json`: clustered candidates
- `out/hot/reports/YYYY-MM-DD.json`: final structured daily report
- `out/hot/md/YYYY-MM/YYYY-MM-DD-hotspots.md`: markdown source for rendered pages

This layered output is important for debugging and replaying historical days.

## 15. Rendering and Publishing

Hotspot rendering is integrated into the same multi-page site used by the paper digest.

Published page families include:

- `hot/`
- `hot/YYYY-MM-DD/`
- `hot/YYYY-MM/`
- `hot/YYYY/`

The build path is:

1. raw outputs are generated under `out/`
2. markdown pages are assembled into `out/site`
3. static HTML is rendered from `out/site`
4. CI publishes the generated HTML to GitHub Pages

The publish stage rebuilds from generated outputs and does not require committing site HTML to `main`.

## 16. Workflow Integration

The hotspot pipeline is integrated into two production workflows.

### 16.1 Daily cron generation

The daily workflow:

1. runs the paper pipeline
2. runs the hotspot pipeline
3. syncs generated outputs to `auto_update`
4. rebuilds and republishes the multi-page site

### 16.2 Missed-date remediation

When a historical date is repaired, the remedy workflow now:

1. regenerates the repaired paper outputs
2. regenerates the hotspot report for the same date
3. syncs the repaired results back to `auto_update`

This keeps the hotspot archive consistent with repaired paper archives.

## 17. Failure Handling

The system is designed for partial failure, not all-or-nothing failure.

Current failure policy:

- each source writes raw output independently
- a failed source adapter yields an empty list rather than a global crash
- LLM screening falls back to heuristic screening batch-by-batch
- digest synthesis falls back to a deterministic summary if necessary

This behavior is critical because some source sites are noisy or structurally unstable.

## 18. Cost Control

The hotspot pipeline is intentionally designed to keep model cost bounded.

Key controls:

- raw-item cap before clustering
- cluster-level rather than item-level screening
- candidate trimming before LLM
- batch screening
- heuristic fallback mode for local testing or quota exhaustion

This keeps the hotspot system much cheaper than a naive "score everything" design.

## 19. Current Limitations

The current implementation is strong enough for production use, but it still has clear limitations:

- clustering is greedy and title-driven rather than embedding-driven
- direct X ingestion is not implemented; social heat is proxied through roundup sources
- some roundup sites are editorially inconsistent and can still inject noise
- the category taxonomy is intentionally compact and may be too coarse for some months
- deterministic priors are hand-tuned rather than learned from labeled data

These are acceptable tradeoffs for a daily report system that must remain stable, inspectable, and cheap to run.

## 20. Why This Algorithm Works

The core reason this system works is that it separates three jobs that are often conflated:

1. **candidate generation**: broad but bounded multi-source collection
2. **topic formation**: merge duplicate evidence into coherent clusters
3. **front-page selection**: aggressively trim using quality, heat, and diversity constraints

Without clustering, the output becomes a noisy link list.  
Without heat signals, the output becomes a personalized paper digest in disguise.  
Without quality controls, the output becomes a hype tracker.

The implemented algorithm is effective because it explicitly models all three.

## Appendix A. Current Roundup-Site Registry

The current roundup registry is maintained in [hotspot_roundup_sites.json](C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/configs/hotspot_roundup_sites.json).

| Site | Role | Cadence | Enabled |
| --- | --- | --- | --- |
| AINews | `community_heat` | daily | yes |
| The Rundown AI | `headline_consensus` | daily | yes |
| Superhuman AI | `headline_consensus` | daily | yes |
| The Neuron | `headline_consensus` | daily | yes |
| TLDR AI | `headline_consensus` | daily | yes |
| Ben's Bites | `builder_momentum` | twice weekly | yes |
| The Batch | `editorial_depth` | weekly | yes |
| Import AI | `editorial_depth` | weekly | yes |
| Last Week in AI | `editorial_depth` | weekly | yes |
| AlphaSignal | `builder_momentum` | regular | no |

These sources are treated as evidence sources, not ground truth.

## Appendix B. Main Implementation Files

Core implementation files:

- [generate_daily_hotspots.py](C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/scripts/generate_daily_hotspots.py)
- [filter_hotspots.py](C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/arxiv_assistant/filters/filter_hotspots.py)
- [hotspot_cluster.py](C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/arxiv_assistant/utils/hotspot_cluster.py)
- [hotspot_schema.py](C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/arxiv_assistant/utils/hotspot_schema.py)
- [hotspot_local_papers.py](C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/arxiv_assistant/apis/hotspot_local_papers.py)
- [hotspot_hf_papers.py](C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/arxiv_assistant/apis/hotspot_hf_papers.py)
- [hotspot_ainews.py](C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/arxiv_assistant/apis/hotspot_ainews.py)
- [hotspot_roundups.py](C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/arxiv_assistant/apis/hotspot_roundups.py)
- [hotspot_official_blogs.py](C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/arxiv_assistant/apis/hotspot_official_blogs.py)
- [hotspot_github.py](C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/arxiv_assistant/apis/hotspot_github.py)
- [hotspot_hn.py](C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/arxiv_assistant/apis/hotspot_hn.py)
- [render_hot_daily.py](C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/arxiv_assistant/renderers/render_hot_daily.py)
- [build_multipage_site.py](C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/arxiv_assistant/renderers/build_multipage_site.py)

Prompt files:

- [hotspot_system_prompt.txt](C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/prompts/hotspot_system_prompt.txt)
- [hotspot_screening_criteria.txt](C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/prompts/hotspot_screening_criteria.txt)
- [postfix_prompt_hotspot_screening.txt](C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/prompts/postfix_prompt_hotspot_screening.txt)
- [hotspot_digest_writer.txt](C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/prompts/hotspot_digest_writer.txt)
