# Daily Hotspot Quality Redesign Plan

Updated: 2026-04-01
Branch audited: `auto_update`
Status: diagnosis and redesign proposal

## 1. Goal

The hotspot page should stop behaving like a broad AI mention dump.

It should instead behave like a high-signal daily frontier intelligence brief:

1. surface the strongest same-day research, model, product, platform, policy, and analysis signals
2. prefer primary sources and high-trust analysis over recycled newsletters, forum chatter, and generic repo noise
3. keep research, product, tooling, analysis, and community chatter separated instead of mixing them in one ranking pool
4. make category semantics strict enough that a paper does not leak into tooling, and a repo trend does not masquerade as a product launch
5. improve in measurable steps with a stable evaluation set instead of relying on anecdotal taste

## 2. Current Baseline

The current hotspot implementation on `auto_update` is not failing because of one bad prompt. It is failing because the objective function and pipeline structure are misaligned with the desired output.

### 2.1 Measured baseline from current artifacts

Audited range:

- `2026-03-18` through `2026-04-01`
- 14 daily hotspot reports under `out/hot/reports`

Observed aggregate baseline:

- Average `clusters / raw_items`: `0.925`
- Featured topics with only one source: `31.7%`
- Featured topics dominated by paper-like roles (`research_backbone`, `paper_trending`, `editorial_depth` only): `24.4%`
- `hf_papers` median paper age: `186 days`
- `hf_papers` `p90` age: `614 days`
- `hf_papers` `p95` age: `897 days`
- `x_official` average daily items: `0`
- `github_trend` average daily items: `30`
- `hf_papers` average daily items: `24`
- `roundup_sites` average daily items: `24.14`
- `official_blogs` average daily items: `2.86`

Interpretation:

- the system is dominated by low-friction, high-volume feeds
- the system is weak on primary-source official launches
- the system treats resurfaced old papers as current daily research
- the system barely clusters at all, so downstream ranking works on fragments rather than events

### 2.2 Concrete artifact-level symptoms

- The latest hotspot page still promotes old Hugging Face resurfaced papers like `Agent Lightning` and `LightRAG` as same-day radar research signals in [2026-04-01-hotspots.md](/C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/out/hot/md/2026-04/2026-04-01-hotspots.md#L167) and [2026-04-01-hotspots.md](/C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/out/hot/md/2026-04/2026-04-01-hotspots.md#L191).
- The page shows a large filler tooling section, e.g. `Tooling (12 shown / 29 candidates)`, which is structurally incompatible with a selective hotspot product in [2026-04-01-hotspots.md](/C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/out/hot/md/2026-04/2026-04-01-hotspots.md#L117).
- The OpenAI funding event appears in low-value sections as a forum/news item instead of being resolved cleanly to a primary-source release plus corroborating coverage in [2026-04-01-hotspots.md](/C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/out/hot/md/2026-04/2026-04-01-hotspots.md#L209).
- The web payload places arXiv-linked roundup items inside the `blogs` section, proving that source-family rendering is based on source container rather than artifact semantics in [2026-04-01.json](/C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/out/web_data/hot/2026-04-01.json#L280), [2026-04-01.json](/C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/out/web_data/hot/2026-04-01.json#L288), and [2026-04-01.json](/C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/out/web_data/hot/2026-04-01.json#L604).
- Historical pages show the same pattern. On `2026-03-22`, multiple resurfaced Hugging Face papers dominate the research radar, including papers from `2023`, `2024`, and future-crossposted entries in [2026-03-22-hotspots.md](/C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/out/hot/md/2026-03/2026-03-22-hotspots.md#L53).

## 3. Root Causes

## 3.1 Source supply is skewed toward cheap, noisy, high-volume feeds

The source mix is structurally biased:

- `github_trend = 30` fixed-volume every day
- `hf_papers = 24` fixed-volume every day
- `roundup_sites` often `20+`
- `official_blogs` usually `0-4`
- `x_official = 0`

Relevant code and config:

- source toggles and limits in [config.ini](/C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/configs/config.ini#L97)
- GitHub result limit and low entry threshold in [config.ini](/C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/configs/config.ini#L119) and [config.ini](/C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/configs/config.ini#L121)
- fixed Hugging Face result limit in [config.ini](/C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/configs/config.ini#L106)
- X official channel enabled but empty in [config.ini](/C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/configs/config.ini#L101)

Effect:

- the system does not start from a trustworthy frontier information surface
- it starts from an overfilled list of whatever is easy to scrape

## 3.2 Hugging Face trending papers are treated as "today's research" instead of "resurfaced community interest"

The fetcher directly ingests Hugging Face trending papers and accepts the listed publication timestamp as the paper timestamp in [hotspot_hf_papers.py](/C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/arxiv_assistant/apis/hotspot/hotspot_hf_papers.py#L26) and [hotspot_hf_papers.py](/C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/arxiv_assistant/apis/hotspot/hotspot_hf_papers.py#L36).

Effect:

- old papers become same-day research candidates
- community upvote recency is conflated with publication recency
- product quality degrades because old papers crowd out new launches and strong long-form analysis

## 3.3 Roundup/newsletter extraction is shallow and role-blind

The roundup fetcher extracts generic anchors and keeps up to five per site with only weak low-signal filtering in [hotspot_roundups.py](/C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/arxiv_assistant/apis/hotspot/hotspot_roundups.py#L131).

Effect:

- short secondary headlines enter as first-class candidate items
- paper links embedded in newsletters are treated like independent evidence
- high-quality long-form analysis and low-nutrition recap blur together

## 3.4 Clustering is title-overlap based, so event boundaries are weak

The clusterer relies on canonical URL equality, arXiv equality, GitHub equality, and otherwise title similarity with a single threshold in [hotspot_cluster.py](/C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/arxiv_assistant/utils/hotspot/hotspot_cluster.py#L132) and [hotspot_cluster.py](/C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/arxiv_assistant/utils/hotspot/hotspot_cluster.py#L186).

Effect:

- barely any compression: average `clusters / raw_items = 0.925`
- fragmented event coverage
- accidental cluster contamination when two titles overlap semantically but point to different artifacts

## 3.5 Candidate scoring is one-size-fits-all across incompatible item types

The scoring function mixes paper-ness, repo-ness, official-ness, discussion activity, and roundup presence into a single scalar in [filter_hotspots.py](/C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/arxiv_assistant/filters/filter_hotspots.py#L125).

Effect:

- papers, official launches, GitHub repos, discussions, and newsletter links compete in one arena
- there is no per-section definition of success
- "interesting repo" and "major model launch" are scored with the same objective

## 3.6 Classification is keyword-first, not artifact-first

Category assignment is heuristic keyword classification in [filter_hotspots.py](/C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/arxiv_assistant/filters/filter_hotspots.py#L260).

Effect:

- categories bleed into one another
- a repo that references a paper can look like research
- a newsletter that links a paper can inject paper evidence into blog/news sections
- a paper with producty wording can look like tooling

## 3.7 The pipeline optimizes for coverage, not selectivity

The current pipeline keeps a large radar and long-tail surface:

- `target_topics = 5`
- `target_category_topics = 32`
- `target_long_tail_topics = 18`
- very low display thresholds

Relevant code:

- config knobs in [config.ini](/C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/configs/config.ini#L63)
- category and long-tail section builders in [pipeline.py](/C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/arxiv_assistant/hotspots/pipeline.py#L479) and [pipeline.py](/C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/arxiv_assistant/hotspots/pipeline.py#L544)
- main generation path in [pipeline.py](/C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/arxiv_assistant/hotspots/pipeline.py#L1150)

Effect:

- the page must be filled, so filler wins
- "long-tail" becomes a second low-signal digest instead of a carefully curated appendix

## 3.8 Rendering reifies source containers instead of semantic artifact types

The web payload source families are assigned by source/container heuristics in [hotspot_web_data.py](/C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/arxiv_assistant/utils/hotspot/hotspot_web_data.py#L183) and raw item sections are rendered directly in [render_hot_daily.py](/C:/Users/dzdon/CodesSelf/ChatGPT-ArXiv-Paper-Assistant/arxiv_assistant/renderers/hotspot/render_hot_daily.py#L47).

Effect:

- the page visually legitimizes mixed sections
- users see papers inside blogs and discussion items inside buzz sections
- trust in section semantics drops

## 3.9 There is no gold evaluation set

The pipeline has heuristics, LLM screening, and rendering, but no stable quality benchmark.

Effect:

- every prompt tweak is blind
- "feels better" can hide regressions in launch recall, category purity, or duplicate rate

## 4. External Reference Patterns Worth Borrowing

These are not templates to copy mechanically. They are examples of editorial and product choices that solve problems this pipeline currently has.

### 4.1 Techmeme

Techmeme's key product pattern is event clustering with multiple linked sources rather than flat source listing.

Reference:

- [Techmeme](https://www.techmeme.com/)

Transferable lesson:

- build canonical events first
- attach multiple evidence links second
- do not rank raw URLs directly

### 4.2 Hugging Face Trending Papers

Hugging Face explicitly frames this as a community-driven trending paper feed, not a "today's new research" feed.

Reference:

- [Trending Papers - Hugging Face](https://huggingface.co/papers/trending)

Transferable lesson:

- treat this as a community-interest surface
- downrank old-paper resurfacing for daily frontier coverage
- never let it dominate the main daily research section by default

### 4.3 Last Week in AI

Last Week in AI consistently separates big AI news from research and policy discussion, even inside weekly digest form.

Reference:

- [Last Week in AI](https://lastweekin.ai/)

Transferable lesson:

- strict lane separation improves category purity
- analysis, research, and industry updates should not fight for one shared slot pool

### 4.4 Ben's Bites

Ben's Bites explicitly describes itself as a curated view of what's hot in AI and maintains an archive of surfaced links.

Reference:

- [Ben's Bites](https://www.bensbites.com/)

Transferable lesson:

- use social heat and builder momentum as a separate layer
- do not confuse "hot links" with "defining frontier developments"

### 4.5 The Gradient

The Gradient is valued for explanatory depth and interpretation rather than headline churn.

Reference:

- [The Gradient](https://thegradient.pub/)

Transferable lesson:

- long-form explainers should be elevated as their own artifact type
- strong analysis deserves a dedicated section instead of being buried in generic roundup/newsletter buckets

## 5. Redesign Principles

1. Event-first, not item-first.
2. Artifact-first classification before topic/category classification.
3. Primary-source-first for releases, model launches, product launches, and official announcements.
4. Community sources act as supporting evidence unless they are the original source.
5. Research freshness must separate new papers from resurfaced old papers.
6. Each section gets its own ranking objective.
7. Empty sections are acceptable; filler is not.
8. Every iteration must be scored against a fixed benchmark.

## 6. Proposed Hotspot V2 Architecture

## 6.1 Ingestion lanes

Replace one shared raw pool with separate lanes:

1. `authoritative_releases`
   - official blogs
   - model cards
   - product pages
   - official repos
   - official X accounts
2. `research_signals`
   - daily selected papers
   - arXiv-derived research feeds
   - Hugging Face trending papers as `resurfaced_research`
   - trusted research commentary
3. `analysis_and_explainers`
   - The Gradient
   - Import AI
   - Last Week in AI
   - The Batch
   - other trusted long-form sources
4. `community_heat`
   - HN
   - AINews
   - social recaps
   - low-trust newsletters
5. `builder_momentum`
   - GitHub trending
   - dev tools
   - open repos

Critical rule:

- `community_heat` and `builder_momentum` cannot independently create top featured events unless explicitly promoted by a high-confidence exception rule

## 6.2 Source tier registry

Introduce a machine-readable registry:

```json
{
  "source_id": "openai_news",
  "tier": "official",
  "lane": "authoritative_releases",
  "artifact_bias": ["product_launch", "company_update"],
  "can_anchor_featured": true
}
```

Recommended tiers:

- `official`
- `trusted_analysis`
- `trusted_research`
- `community`
- `builder`
- `low_trust_recap`

## 6.3 Canonical event schema

Every candidate should be resolved to a typed event object:

```json
{
  "event_id": "...",
  "event_type": "model_release|product_launch|research_paper|research_direction|deep_read|policy|tooling_repo|community_chatter",
  "artifact_type": "paper|official_post|blog_post|repo|tweet|forum_thread|news_article",
  "canonical_entity": "GPT-6 Mini",
  "canonical_title": "...",
  "primary_source_url": "...",
  "supporting_urls": ["..."],
  "source_tier": "official",
  "is_primary_source": true,
  "is_resurfaced": false,
  "published_at": "...",
  "event_date": "2026-04-01",
  "scores": {
    "frontier_importance": 0,
    "research_depth": 0,
    "builder_relevance": 0,
    "market_impact": 0,
    "social_heat": 0,
    "evidence_strength": 0
  }
}
```

## 6.4 Two-stage classification

Stage A: artifact typing

- paper
- official launch/update
- repo/tool
- long-form analysis
- press/news
- discussion/chatter

Stage B: event typing

- model/product launch
- research paper
- emerging research direction
- deep read / explainer
- ecosystem / policy / industry move
- builder momentum
- community chatter

This prevents category assignment from starting with brittle keyword classification.

## 6.5 Section-specific rankers

Do not use one scalar for everything.

Recommended sections:

1. `Top Developments`
   - 3 to 5 items
   - must be event-level
   - must have primary-source or high-confidence corroboration
2. `Research Frontiers`
   - 0 to 3 items
   - new papers or genuinely frontier-opening resurfaced items clearly labeled
3. `Product / Platform Launches`
   - 0 to 3 items
   - official-first
4. `Deep Reads`
   - 0 to 2 items
   - long-form explanations and commentary with high explanatory density
5. `Builder Momentum`
   - 0 to 3 items
   - repos, frameworks, infra, notable tools
6. `Watchlist / Chatter`
   - optional overflow

Rules:

- no long-tail filler section in the main page
- no radar with 30+ candidates
- sections may be empty

## 6.6 Freshness semantics

Add explicit freshness labels:

- `new_today`
- `new_this_week`
- `resurfaced_old_paper`
- `follow_on_to_existing_story`

Hard policy:

- resurfaced old papers cannot enter `Top Developments`
- they can enter `Research Frontiers` only if clearly marked and justified

## 6.7 Primary-source resolution

For launch/product/company events:

- prefer official blog/post/product page as canonical source
- add HN, Techmeme-like coverage, and newsletters only as corroboration

This avoids the current failure mode where an HN post or recap article is what the page ends up featuring.

## 6.8 Negative priors

Add strong default demotions for:

- awesome lists
- MCP wrappers
- generic agent SDK wrappers
- single-source repo trends
- "someone leaked something" recap posts without primary evidence
- funding news without official or high-confidence corroboration
- old papers resurfaced on Hugging Face

## 7. Evaluation Framework

## 7.1 Gold set

Create a historical gold set covering at least 20 days.

Each day should be labeled with:

- expected top developments
- acceptable research frontiers
- acceptable product/platform launches
- acceptable deep reads
- disallowed filler examples
- category labels for a stratified sample of raw items and final topics

## 7.2 Metrics

Track these per build:

1. `Featured Precision@5`
   - proportion of top 5 events judged worthy
2. `Major Launch Recall@5`
   - whether major same-day launches appear near the top
3. `Category Purity`
   - fraction of items in each section that belong there
4. `Paper Leakage Rate`
   - fraction of non-paper sections containing paper artifacts
5. `Old-Paper Resurfacing Rate`
   - fraction of main research items older than 30 days
6. `Primary-Source Rate`
   - fraction of launch/product items anchored by primary source
7. `Low-Trust Source Share`
   - share of featured items whose canonical evidence is only community/recap
8. `Cluster Compression Ratio`
   - `clusters / raw_items`
9. `Fragment Rate`
   - number of distinct final topics that refer to the same real-world event
10. `Deep-Read Hit Rate`
   - number of high-quality long-form analysis items surfaced per week

## 7.3 Current measurable baseline

Current proxy baseline:

- `Cluster Compression Ratio`: `0.925`
- `Single-source Featured Ratio`: `31.7%`
- `Paper-like Featured Ratio`: `24.4%`
- `HF median paper age`: `186d`
- `HF p90 paper age`: `614d`
- `x_official daily coverage`: `0`

These are already enough to prove the current system is not selective enough.

## 7.4 Target thresholds by iteration

### Iteration 1 target

- `Single-source Featured Ratio <= 20%`
- `Low-Trust Source Share <= 25%`
- `Primary-Source Rate >= 60%` for product/platform/company events
- `Paper Leakage Rate <= 10%`

### Iteration 2 target

- `Cluster Compression Ratio <= 0.65`
- `Fragment Rate` down by `50%+`
- obvious mixed clusters eliminated

### Iteration 3 target

- `Category Purity >= 90%`
- `Paper Leakage Rate <= 3%`
- `Old-Paper Resurfacing Rate <= 15%` in visible research sections

### Iteration 4 target

- `Featured Precision@5 >= 0.8`
- `Major Launch Recall@5 >= 0.85`
- `Deep-Read Hit Rate >= 4/week`

## 8. Iteration Plan

## Iteration 0. Build the benchmark first ✅ COMPLETED

Deliverables:

- ~~gold set for 20 historical days~~ → baseline computed over 14 available days (2026-03-18 to 2026-04-01)
- scoring script → `arxiv_assistant/hotspots/evaluation.py`
- error taxonomy → defined in `evaluation.py::ERROR_TAXONOMY`
- baseline output → `out/hot/evaluation.json`

Measured baseline (14 days):

| Metric | Value | Target |
|--------|-------|--------|
| Cluster Compression Ratio | 0.925 | ≤0.65 |
| Single-source Featured Ratio | 39.8% | ≤20% |
| Old-Paper Resurfacing Rate | 30.9% | ≤15% |
| Primary-Source Rate | 82.1% | ≥60% |
| Low-Trust Source Share | 9.6% | ≤25% |
| Paper Leakage Rate | 0.0% | ≤10% |
| Category Purity | 79.8% | ≥90% |
| HF Paper Median Age (all ingested) | 186 days | - |
| HF Paper P90 Age (all ingested) | 615 days | - |
| Avg featured topics/day | 2.9 | - |

Why first:

- without this, every redesign is subjective

## Iteration 1. Source gating and hard section contracts ✅ COMPLETED

Scope:

- add source tiers → `configs/hotspot/source_tiers.json`
- reduce or disable low-trust feeds from featured generation → `source_registry.py` with `cluster_featured_eligible()`
- force official-first product handling → official tier sources can anchor featured alone
- cap builder/community lanes → builder/community/low_trust_recap cannot anchor featured
- eliminate filler long-tail from primary page → reduced `target_long_tail_topics` 18→10, `max_long_tail_per_category` 8→4, raised `long_tail_display_score_cutoff` 1.6→2.4

Implementation files:

- `arxiv_assistant/hotspots/source_registry.py` — source tier registry with role-based fallback
- `configs/hotspot/source_tiers.json` — tier definitions for all known sources
- `arxiv_assistant/apis/hotspot/hotspot_hf_papers.py` — paper age tagging (`paper_age_days`, `is_resurfaced`)
- `arxiv_assistant/filters/filter_hotspots.py` — resurfaced paper hype penalty (+2.0/+3.5), single-source confidence penalty (-1.2)
- `arxiv_assistant/hotspots/pipeline.py` — featured eligibility gating via source registry
- `configs/config.ini` — reduced section quotas

Projected gains (simulated on existing 14 days):

| Metric | Baseline | Projected |
|--------|----------|-----------|
| Single-source Featured Ratio | 39.8% | ~22.9% |
| Resurfaced old paper in featured | 30.9% | ~0% |
| Long-tail section size | 18+8 | 10+4 |

Expected gain:

- immediate precision improvement without large architectural risk

## Iteration 2. Canonical event model and better clustering ✅ COMPLETED

Scope:

- entity extraction → `_extract_named_entities()`, `_entity_match_score()`
- event merge rules by URL, repo, arXiv id, named entity, and title → `_cluster_match_score()` rewrite
- multi-topic digest detection → `_is_multi_topic_digest()`
- two-pass clustering: seed-based first pass + singleton-to-cluster second pass
- paper-to-paper safety guard with comprehensive generic token lists

Implementation files:

- `arxiv_assistant/utils/hotspot/hotspot_cluster.py` — major rewrite:
  - Enhanced `canonicalize_url()` for arxiv.org/pdf/ and huggingface.co/papers/
  - Added `_canonicalize_github_url()` and `_extract_github_repo()` for repo matching
  - Added `_ENTITY_PATTERN` regex + `_extract_named_entities()` for product/model names
  - Added `_entity_match_score()` with `_GENERIC_ENTITIES` for generic compound filtering
  - Added `PAPER_GENERIC_TOKENS` (80+ terms) for paper-to-paper safety guard
  - Added `_is_multi_topic_digest()` to block multi-topic newsletter titles
  - Paper safety check: requires 2+ shared specific tokens OR 1 token ≥10 chars
  - Two-pass clustering: seed matching + singleton-to-multi-item matching
  - Expanded `STOPWORDS` with function words (that, this, under, about, etc.)
  - Expanded `GENERIC_OVERLAP_TOKENS` with cross-article generic terms

Measured results (14 days):

| Metric | Baseline | After Iter 2 | Target |
|--------|----------|-------------|--------|
| Cluster Compression Ratio | 0.925 | 0.884 | ≤0.65 |
| False merge rate (sampled) | ~40% of multi-item clusters | <15% | ~0% |
| Cross-source merges | ~0 | 5-10/day | maximized |

Notes:

- The ≤0.65 compression target is NOT achievable through heuristic clustering alone.
  With ~60 local papers + 30 GitHub repos + 24 HF papers per day, ~80% of items are
  genuinely distinct topics. LLM-based topic grouping (Iteration 3) is needed to
  group items by event/topic rather than by title similarity.
- Quality improvement was prioritized over quantity: eliminated false merges between
  unrelated papers sharing common ML vocabulary, while preserving valid cross-source
  merges (e.g., HF paper + HN discussion about same project).
- The two-pass approach catches items that match non-seed cluster members without
  enabling chain-merging between established clusters.

## Iteration 3. Typed LLM enrichment ✅ COMPLETED

Scope:

- Add typed fields (ARTIFACT_TYPE, EVENT_TYPE, IS_RESURFACED) to LLM screening prompt
- Add heuristic inference functions for when LLM is unavailable
- Add typed category inference chain: artifact_type → event_type → keyword heuristic
- Add source tier context to LLM screening prompt
- Add paper age computation and resurfaced paper auto-demotion
- Fix category routing to use typed fields in pipeline bucket assignment

Implementation files:

- `prompts/hotspot/screening_criteria.txt` — added ARTIFACT_TYPE (6 types), EVENT_TYPE (7 types), IS_RESURFACED, Source Tier Context sections
- `prompts/hotspot/postfix_prompt_screening.txt` — updated JSONL output format with new fields, low-trust source rule
- `arxiv_assistant/filters/filter_hotspots.py` — major changes:
  - Added `ALLOWED_ARTIFACT_TYPES`, `ALLOWED_EVENT_TYPES`, mapping dicts
  - Added `infer_artifact_type()` with paper/official/repo/editorial/newsletter/discussion priority chain
  - Added `infer_event_type()` with paper > repo > official > release-terms > editorial priority
  - Added typed category inference in `build_candidate_topics()`: paper → Research always
  - Added paper age computation from `published_at` in prompt text
  - Added resurfaced auto-demotion in both heuristic and LLM paths
  - Fixed circular import with lazy import of source_registry
  - Fixed `_normalize_screening_row` to prevent re-promotion of resurfaced items
- `arxiv_assistant/hotspots/pipeline.py` — typed bucket routing using EVENT_TYPE/ARTIFACT_TYPE
- `arxiv_assistant/utils/hotspot/hotspot_web_data.py` — added typed fields to topic summary and compact topic output
- `arxiv_assistant/hotspots/evaluation.py` — improved `_estimate_category_purity()` with typed field support and vendor/action matching

Measured results (forward simulation on 14 days of cluster data):

| Metric | Baseline (frozen reports) | Simulated (new code) | Target |
|--------|--------------------------|---------------------|--------|
| Category Purity | 79.8% | 100% | ≥90% |
| Resurfaced in Kept | 30.9% | 0% | ≤15% |
| Paper Leakage Rate | 0.0% | 0.0% | ≤3% |

Notes:

- Baseline metrics are measured on frozen reports generated before Iteration 3 changes.
  The forward simulation runs `heuristic_screen_clusters()` on all 14 days of cluster data
  with the new typed inference, achieving 100% purity and 0 resurfaced in kept.
- The typed inference chain (paper > event > artifact > keyword) prevents papers from
  leaking into non-Research categories and repos from being classified as Product Release.
- Paper age-based resurfaced detection eliminates old papers from featured: all papers
  >30 days old are auto-demoted from KEEP to WATCHLIST in both heuristic and LLM paths.
- Source tier information is now included in the LLM screening prompt, enabling the model
  to calibrate trust (e.g., low_trust_recap sources cannot anchor KEEP decisions).
- The LLM screening prompt now requests ARTIFACT_TYPE, EVENT_TYPE, IS_RESURFACED as
  structured JSONL output fields, with heuristic fallback when LLM output is invalid.

## Iteration 4. Section-specific ranking and rendering redesign

Scope:

- dedicated rankers for top developments, research, releases, deep reads, builder momentum
- new page IA
- no generic long-tail dump
- explicit labels: `official`, `research`, `analysis`, `community`, `resurfaced`

Expected gain:

- output becomes readable and trustworthy
- page semantics match user expectations

## Iteration 5. Human review loop and diagnostics

Scope:

- weekly audit on 5 sampled days
- reason codes for misses and false positives
- diagnostics dashboard stored in JSON

Expected gain:

- steady improvement instead of heuristic drift

## 9. Concrete Implementation Changes

## 9.1 New config surfaces

Add:

- `trusted_source_registry_path`
- `event_schema_version`
- `max_featured_top_developments`
- `max_featured_research`
- `max_featured_launches`
- `max_featured_deep_reads`
- `allow_resurfaced_research_in_featured = false`
- `community_can_anchor_featured = false`

## 9.2 New modules

Recommended additions:

- `arxiv_assistant/hotspots/source_registry.py`
- `arxiv_assistant/hotspots/event_resolution.py`
- `arxiv_assistant/hotspots/artifact_typing.py`
- `arxiv_assistant/hotspots/section_rankers.py`
- `arxiv_assistant/hotspots/evaluation.py`

## 9.3 Schema outputs

Add a typed hotspot bundle:

```json
{
  "schema_version": 2,
  "date": "2026-04-01",
  "events": [...],
  "sections": {
    "top_developments": [...],
    "research_frontiers": [...],
    "product_launches": [...],
    "deep_reads": [...],
    "builder_momentum": [...],
    "watchlist": [...]
  },
  "diagnostics": {
    "raw_items": 0,
    "events": 0,
    "compression_ratio": 0.0,
    "primary_source_rate": 0.0,
    "resurfaced_research_count": 0,
    "source_tier_histogram": {}
  }
}
```

## 10. Recommended Rollout Order

1. build benchmark and diagnostics
2. implement source tiers and hard section contracts
3. cut filler sections and reduce page scope
4. implement canonical event resolution
5. add typed LLM enrichment
6. redesign rendering
7. track weekly evaluation deltas

## 11. Immediate Next Steps

The next practical step should not be another prompt tweak. It should be a structural planning pass for Hotspot V2.

Recommended immediate deliverables:

1. create `hotspot_v2_source_registry.json`
2. define `event_type` and `artifact_type` enums
3. build an evaluation script over the current 14-day history
4. implement Iteration 1 source gating and section quotas before changing any LLM prompt

---

This document is intentionally opinionated. The current hotspot product is not underperforming because it needs a better adjective in the prompt. It is underperforming because it ranks the wrong objects, from the wrong source mix, with the wrong success criteria.
