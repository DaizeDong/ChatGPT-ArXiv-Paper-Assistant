# Hotspot V2 Quality Execution Plan

Updated: 2026-04-02
Status: In progress — Phase 1 COMPLETE
Estimated execution time: 8 hours continuous
Target quality: Industry-grade daily AI intelligence brief (top-conference / paid-tool level)

---

## 0. Quality Vision

The final product must read like a Techmeme-grade or The Information-grade daily AI intelligence brief:

1. **Top 3-5 featured topics** are genuinely the most important AI events of the day — model launches, major product releases, breakthrough research, significant policy moves
2. **Research Frontiers** shows only fresh (<7 days), high-impact papers — not resurfaced community favorites from 2023-2024
3. **Product & Platform** shows only official, primary-source announcements — not recycled newsletter headlines about funding rounds
4. **Deep Reads** shows substantive long-form analysis with real intellectual depth — not shallow roundup summaries
5. **Builder Momentum** shows genuinely notable repos with high stars and real utility — not awesome-lists and wrapper SDKs
6. **Zero filler** — empty sections are acceptable; padding is not

---

## 1. Current State Diagnosis

### 1.1 Source supply (2026-04-01 snapshot)

| Source | Items/day | Quality | Status |
|--------|-----------|---------|--------|
| `local_papers` | 26 | Medium | Working — keyword-filtered arXiv, no importance ranking |
| `hf_papers` | 24 | **Low** | Working — 63% are >30-day resurfaced papers |
| `github_trend` | 30 | **Low** | Working — stars_cutoff=20 too low, awesome-lists dominate |
| `roundup_sites` | 23 | **Low** | Working — shallow title extraction, 4 low-trust sources |
| `official_blogs` | 3 | **High** | Working — only 4 vendors (OpenAI/Anthropic/Google/Meta) |
| `hn_discussion` | 5 | Medium | Working — good as corroboration signal |
| `x_official` | **0** | N/A | **Broken** — no X API bearer token configured |
| `ainews` | **0** | N/A | **Broken** — RSS feed appears non-functional |
| `x_ainews_twitter` | **0** | N/A | **Broken** — depends on ainews RSS having Twitter section |
| `x_paperpulse` | **0** | N/A | **Broken** — PaperPulse API returning empty |

**Total: 111 items, ~10 genuinely high-quality.**

### 1.2 Measured baseline metrics (14-day average, Iter 0-4 code)

| Metric | Baseline | Target |
|--------|----------|--------|
| Cluster Compression Ratio | 0.925 | ≤0.70 |
| Single-source Featured Ratio | 39.8% | ≤15% |
| Old-Paper Resurfacing Rate | 30.9% | 0% in featured, ≤10% in research section |
| Primary-Source Rate (product topics) | 82.1% | ≥90% |
| Low-Trust Source Share (featured) | 9.6% | ≤5% |
| Category Purity | 79.8% | ≥95% |
| Deep-Read Hit Rate | 0/week | ≥5/week |
| Multi-source Featured Ratio | 60.2% | ≥85% |
| Section Count (avg/day) | 2.9 | ≥4 |
| Official blogs items/day | 3 | ≥8 |
| High-quality source ratio | ~3% | ≥40% |

### 1.3 Root cause priority ranking

| Priority | Root Cause | Impact on Output |
|----------|-----------|-----------------|
| **P0** | 4 sources producing 0 items (x_official, ainews, x_ainews_twitter, x_paperpulse) | Missing real-time product launches, social heat, research buzz |
| **P1** | Only 4 official blog sources | Missing Microsoft, Mistral, DeepSeek, Stability, Cohere, NVIDIA, etc. |
| **P2** | HF papers includes old papers with no age filter at ingestion | 63% resurfaced papers pollute research section |
| **P3** | GitHub stars_cutoff=20 too low | Awesome-lists, wrapper SDKs, low-effort repos dominate |
| **P4** | No long-form analysis sources | Deep Reads section permanently empty |
| **P5** | Roundup extraction is title-only, no body text | Cannot distinguish 3-sentence news from 5000-word analysis |
| **P6** | Scoring weights over-reward paper presence | Papers inflate quality/heat even when low-impact |
| **P7** | LLM screening sees too few clusters (18 max) | Most topics never get LLM quality judgment |

---

## 2. Evaluation Framework

### 2.1 Metrics to track at every checkpoint

Each phase ends with a full evaluation run. The evaluation script (`evaluation.py`) will be extended to cover these dimensions:

**Source Health Metrics (NEW):**
- `active_source_count`: Number of sources producing >0 items
- `high_quality_item_ratio`: Items from official/trusted_analysis/trusted_research / total items
- `source_diversity_score`: Entropy of source distribution (avoid single-source dominance)
- `official_items_per_day`: Average items from official-tier sources

**Content Quality Metrics (EXISTING + ENHANCED):**
- `cluster_compression_ratio`: clusters / raw_items (target ≤0.70)
- `single_source_featured_ratio`: featured topics with 1 source (target ≤15%)
- `multi_source_featured_ratio`: featured topics with ≥2 sources (target ≥85%)
- `old_paper_resurfacing_rate`: papers >30d in research section (target ≤10%)
- `primary_source_rate`: product/launch topics with official source (target ≥90%)
- `low_trust_source_share`: featured topics anchored only by low-trust sources (target ≤5%)
- `category_purity`: topics correctly classified (target ≥95%)
- `paper_leakage_rate`: papers in non-research sections (target ≤2%)

**Output Quality Metrics (NEW):**
- `featured_precision_proxy`: featured topics that are multi-source AND (official OR trusted_analysis) (target ≥80%)
- `deep_read_hit_rate_per_week`: items in Deep Reads section per week (target ≥5)
- `section_count_per_day`: average non-empty semantic sections (target ≥4)
- `builder_quality_score`: avg GitHub stars of Builder Momentum items (target ≥200)
- `filler_ratio`: topics with final_score < 3.0 shown in any section (target 0%)

**Negative Signal Metrics (NEW):**
- `awesome_list_count`: repos matching `awesome-*` pattern in output (target 0)
- `wrapper_sdk_count`: repos matching `*-wrapper`, `*-agent-sdk` in output (target 0)
- `resurfaced_in_featured`: old papers in featured topics (target 0)
- `duplicate_event_count`: same real-world event appearing in multiple sections (target 0)

### 2.2 Evaluation checkpoints

```
Phase 1 complete → run eval → record baseline with fixed sources
Phase 2 complete → run eval → compare to Phase 1
Phase 3 complete → run eval → compare to Phase 2
Phase 4 complete → run eval → compare to Phase 3
Phase 5 complete → run eval → compare to Phase 4
Phase 6 complete → run eval → final quality gate
```

### 2.3 Quality gate definition

The system is "V2 complete" when ALL of these hold simultaneously:

- active_source_count ≥ 8
- high_quality_item_ratio ≥ 35%
- single_source_featured_ratio ≤ 15%
- old_paper_resurfacing_rate = 0% in featured, ≤10% in research section
- category_purity ≥ 95%
- deep_read_hit_rate ≥ 5/week
- section_count ≥ 4/day
- awesome_list_count = 0
- filler_ratio = 0%

---

## 3. Execution Phases

---

### Phase 1: Source Infrastructure Repair (Est. 1.5h)

**Goal:** Fix all broken sources so the system receives data from every configured channel.

#### 1A. Diagnose and fix `ainews` (0 items)

**Files:** `arxiv_assistant/apis/hotspot/hotspot_ainews.py`

The AINews RSS feed (`https://news.smol.ai/rss.xml`) is returning 0 items. Diagnose:
1. Fetch the RSS URL manually and inspect the response
2. Check if the feed format changed (feedparser may be failing silently)
3. Check if all entries are filtered out by the freshness window or signal filters
4. If the feed URL changed, update it
5. If the feed is truly dead, mark it as non-functional and add a fallback (e.g., AINews HTML page scrape)

**Acceptance:** `ainews` produces ≥1 item on a test run.

#### 1B. Diagnose and fix `x_ainews_twitter` (0 items)

**Files:** `arxiv_assistant/apis/hotspot/hotspot_x_ainews.py`

This depends on the AINews RSS having an "AI Twitter Recap" section. If 1A fixes the RSS feed, this may self-resolve. Otherwise:
1. Check if the RSS feed still has a Twitter section
2. If not, this source should be deprecated or replaced with direct X data

**Acceptance:** `x_ainews_twitter` produces ≥1 item OR is explicitly disabled with a documented reason.

#### 1C. Diagnose and fix `x_paperpulse` (0 items)

**Files:** `arxiv_assistant/apis/hotspot/hotspot_x_paperpulse.py`

PaperPulse API (`https://www.paperpulse.ai/api/researcher-feed`) is returning empty. Diagnose:
1. Fetch the API endpoint manually
2. Check if the API requires authentication or has changed
3. If the API is dead, mark as non-functional and add alternative research tweet tracking

**Acceptance:** `x_paperpulse` produces ≥1 item OR is explicitly disabled with documented fallback.

#### 1D. Handle `x_official` (requires X API bearer token)

**Files:** `arxiv_assistant/apis/hotspot/hotspot_x_official.py`, docs

The X API requires a bearer token from paid API access. This is a configuration issue, not a code bug.
1. Document the requirement clearly in a setup guide
2. Add a graceful warning log when bearer token is missing (not just silent empty return)
3. If bearer token is available in env, verify the fetcher works end-to-end
4. If not available, add a **fallback pathway**: extract official X content from alternative sources (e.g., Nitter mirrors, social media aggregators, or existing roundup sites that embed tweets)

**Acceptance:** Either `x_official` produces items with a configured token, OR a documented fallback exists and the system degrades gracefully with a clear log message.

#### 1E. Update evaluation framework for source health

**Files:** `arxiv_assistant/hotspots/evaluation.py`

Add `active_source_count`, `high_quality_item_ratio`, `source_diversity_score`, `official_items_per_day` to daily and aggregate metrics.

**Acceptance:** Evaluation output includes source health metrics.

#### Phase 1 checkpoint

Run full evaluation on all available report days. Record:
- How many sources now produce >0 items
- Total items/day change
- Source distribution entropy

**Commit:** `feat(hotspot): Phase 1 — fix broken source infrastructure`

#### Phase 1 Results (2026-04-02)

| Source | Before | After | Fix |
|--------|--------|-------|-----|
| `ainews` | 0 items | 7 items | Extended freshness window to 96h for weekend coverage |
| `x_ainews_twitter` | 0 items | 6 items | Same freshness fix; Twitter Recap section now extracted |
| `x_paperpulse` | 0 items | 0 items (stale) | Added staleness detection; API data from Sep 2025, frozen |
| `x_official` | 0 items | 0 items (no token) | Added warning log; graceful degradation documented |

New files created:
- `configs/hotspot/source_tiers.json` — source tier classification
- `arxiv_assistant/hotspots/evaluation.py` — evaluation framework with source health metrics

Evaluation baseline (Phase 1):
- Active sources: 4.6/day (target >= 8)
- High-quality item ratio: 33.2% (target >= 35%)
- Old paper resurfacing: 66.7% (target <= 10%)
- Quality gate: 4/9 passed

---

### Phase 2: Source Quality Gates (Est. 1.5h)

**Goal:** Dramatically reduce noise from existing sources by tightening ingestion-level quality filters.

#### 2A. HF Papers: Age filtering at ingestion

**Files:** `arxiv_assistant/apis/hotspot/hotspot_hf_papers.py`

Current state: All 24 trending papers are ingested regardless of age. Papers from 2023 (PagedAttention) and 2024 (OpenDevin, LightRAG) enter the pipeline.

Changes:
1. Add ingestion-time age filter: skip papers with `paper_age_days > 14`
2. Add upvote minimum: skip papers with `upvotes < 5` (basic quality signal)
3. Reduce `hf_result_limit` from 24 to 12 (quality over quantity)
4. Log skipped papers with reason for diagnostics

Expected impact: HF papers drops from 24 → ~8-12 fresh papers. Resurfaced paper pollution eliminated at source.

**Acceptance:** No paper >14 days old in HF papers output. Simulated resurfacing rate drops to ≤5%.

#### 2B. GitHub Trending: Raise quality floor

**Files:** `arxiv_assistant/apis/hotspot/hotspot_github.py`, `configs/config.ini`

Current state: `stars_cutoff=20`, `created_within_days=10`, `result_limit=30`. This admits awesome-lists, WeChat wrappers, and trivial agent SDKs.

Changes:
1. Raise `stars_cutoff` from 20 to 80
2. Reduce `created_within_days` from 10 to 5
3. Add title/description blacklist patterns at ingestion:
   - `awesome-*` prefix → skip
   - Contains "wrapper" or "wechat" or "weixin" or "微信" → skip unless stars > 500
   - Contains only "mcp" or "agent-sdk" with no other distinguishing content → skip unless stars > 300
4. Reduce `result_limit` from 30 to 15
5. Improve search queries to be more specific (add "language model", "neural network", "transformer" to increase precision)

Expected impact: GitHub items drops from 30 → ~10-15 higher-quality repos. Awesome-lists and wrappers eliminated.

**Acceptance:** Zero awesome-list repos in output. Average stars of GitHub items ≥ 100.

#### 2C. Roundup sites: Reduce low-trust noise

**Files:** `arxiv_assistant/apis/hotspot/hotspot_roundups.py`, `configs/hotspot/roundup_sites.json`

Current state: 4 low-trust sources (The Rundown AI, Superhuman AI, The Neuron, TLDR AI) contribute ~12 shallow headlines. Max 5 items per site.

Changes:
1. Reduce per-site limit for `low_trust_recap` tier sources from 5 to 2
2. Keep per-site limit at 5 for `trusted_analysis` tier sources (Import AI, The Batch, Last Week in AI)
3. Add minimum title length filter from 18 to 30 characters
4. Add signal-word requirement: at least one keyword match (model names, company names, technical terms) required for low-trust sources
5. Explicitly skip roundup items that look like paper links (contain "arxiv.org" or "huggingface.co/papers" in URL) — these should come from dedicated paper sources, not roundup recycling

Expected impact: Low-trust roundup items drop from ~12 to ~4-6. Paper link duplication from roundups eliminated.

**Acceptance:** Low-trust source items ≤ 8/day. Zero roundup items with arxiv/HF paper URLs.

#### 2D. Update scoring to reflect new source quality landscape

**Files:** `arxiv_assistant/filters/filter_hotspots.py`

With fewer but higher-quality items, the scoring thresholds may need recalibration:
1. Increase `HYPE_PENALTY` for single-source items from community/builder tiers
2. Increase bonus for multi-source confirmation (RESONANCE base from 1.6 to 2.0 when source_count ≥ 2)
3. Add explicit penalty for items matching blacklist patterns (awesome-list, wrapper) that survived ingestion

**Acceptance:** Forward simulation shows single_source_featured_ratio ≤ 25%.

#### Phase 2 checkpoint

Run full evaluation. Compare to Phase 1 baseline:
- Expect: raw_items/day drops by 30-40%, but high_quality_item_ratio increases by 3-5x
- Expect: resurfacing_rate near 0%, awesome-list count = 0

**Commit:** `feat(hotspot): Phase 2 — tighten source quality gates`

---

### Phase 3: Expand High-Quality Sources (Est. 2h)

**Goal:** Fill the quality gap by adding sources that actually produce the content we want in each semantic section.

#### 3A. Expand official blogs (authoritative releases)

**Files:** `arxiv_assistant/apis/hotspot/hotspot_official_blogs.py`

Current: 4 sources (OpenAI, Anthropic, Google, Meta). Missing major AI companies.

Add these official sources:

| Source | URL | Mode | Notes |
|--------|-----|------|-------|
| Microsoft AI Blog | `https://blogs.microsoft.com/ai/feed/` | RSS | Azure AI, Copilot |
| NVIDIA AI Blog | `https://blogs.nvidia.com/blog/category/deep-learning/feed/` | RSS | GPU/AI infra |
| DeepMind Blog | `https://deepmind.google/blog/rss.xml` | RSS | Research + products |
| Mistral AI Blog | `https://mistral.ai/news/` | HTML | Open-weight models |
| Stability AI Blog | `https://stability.ai/news` | HTML | Image/video gen |
| Cohere Blog | `https://cohere.com/blog` | HTML | Enterprise LLM |
| Hugging Face Blog | `https://huggingface.co/blog` | HTML | Open-source ecosystem |
| xAI Blog | `https://x.ai/blog` | HTML | Grok updates |

Implementation:
1. Make `OFFICIAL_SOURCES` registry-driven instead of hard-coded (load from `configs/hotspot/official_blogs.json`)
2. Add RSS and HTML extraction modes for each new source
3. Test each source's freshness detection and title extraction
4. Update `source_tiers.json` with tier assignments for new sources (all `official` tier)

Expected impact: `official_blogs` items/day increases from 3 to 8-15.

**Acceptance:** ≥ 6 official blog sources producing items on a test run. All new sources have correct tier assignment.

#### 3B. Add long-form analysis feeds (Deep Reads supply)

**Files:** New `configs/hotspot/analysis_feeds.json`, modified `arxiv_assistant/apis/hotspot/hotspot_roundups.py` or new dedicated fetcher

The Deep Reads section is permanently empty because no ingested source provides long-form analysis content. Add dedicated high-quality analysis sources:

| Source | URL | Content Type | Notes |
|--------|-----|-------------|-------|
| The Gradient | `https://thegradient.pub/rss/` | RSS | Long-form ML analysis |
| Lilian Weng's Blog | `https://lilianweng.github.io/index.xml` | RSS | Research explainers |
| Simon Willison's Blog | `https://simonwillison.net/atom/everything` | Atom | AI tooling analysis |
| Jay Alammar's Blog | `https://jalammar.github.io/feed.xml` | RSS | Visual ML explanations |
| Sebastian Raschka | `https://magazine.sebastianraschka.com/feed` | RSS | LLM research analysis |
| Chip Huyen's Blog | `https://huyenchip.com/feed.xml` | RSS | ML systems |
| Distill.pub | `https://distill.pub/rss.xml` | RSS | Interactive research |

Implementation:
1. Create `configs/hotspot/analysis_feeds.json` registry
2. Either extend roundup fetcher with a dedicated `analysis_feed` mode or create a new `hotspot_analysis_feeds.py` fetcher
3. These sources should use RSS/Atom parsing (not HTML scraping) for reliability
4. Tag items with `source_role: "editorial_depth"` and `source_type: "analysis_blog"`
5. Add to `source_tiers.json` as `trusted_analysis` tier
6. Each item should capture: title, full summary (up to 800 chars), URL, published_at
7. Freshness window: 7 days (analysis content is less time-sensitive than news)

Expected impact: Deep Reads section starts receiving 2-5 items/day.

**Acceptance:** ≥ 3 analysis feed sources producing items. Deep Read candidates ≥ 2/day.

#### 3C. Improve research paper signals

**Files:** `arxiv_assistant/apis/hotspot/hotspot_hf_papers.py`, potentially new fetcher

Current research signal is poor: local_papers are keyword-filtered arXiv with no quality ranking, HF papers are community trending (biased toward old popular papers).

Improvements:
1. **Papers With Code trending**: Add `https://paperswithcode.com/latest` as a research signal source (HTML scrape or API). These papers have code, benchmarks, and recency guarantees.
2. **Semantic Scholar trending**: Add `https://api.semanticscholar.org/graph/v1/paper/search` with `sort=citationCount:desc` and date filter. Free API, good quality signal.
3. **HF Daily Papers** (not trending): The HF "daily papers" section (`https://huggingface.co/papers`) shows recently submitted papers (not trending), which is a better freshness signal.
4. **arXiv category feeds**: Consider adding RSS feeds for specific arXiv categories (cs.AI, cs.CL, cs.LG, cs.CV) as additional research signal.

Implementation priority: Papers With Code first (reliable, fresh, code-attached papers). Semantic Scholar second (API-based, flexible). Others as time allows.

**Acceptance:** Research section candidates include ≥ 3 fresh (<7 day) papers with code/impact signals.

#### 3D. Add alternative social signal sources

**Files:** New or modified fetchers

Since `x_official` requires a paid API token, add alternative social signal pathways:
1. **Reddit AI subreddits**: Add `r/MachineLearning`, `r/artificial` hot posts via Reddit JSON API (no auth needed: `https://www.reddit.com/r/MachineLearning/hot.json`)
2. **Lobsters**: Add `https://lobste.rs/t/ai.rss` for high-quality technical discussion
3. **ProductHunt AI**: Add trending AI products as a product launch signal

Implementation: Prioritize Reddit (free API, high volume, good signal-to-noise with score filtering). Lobsters second (free RSS, very high quality but low volume). ProductHunt third (if time allows).

**Acceptance:** ≥ 1 new social signal source producing items.

#### Phase 3 checkpoint

Run full evaluation. Compare to Phase 2:
- Expect: total items/day stays similar or increases slightly, but high_quality_item_ratio ≥ 30%
- Expect: official_items_per_day ≥ 8
- Expect: deep_read_count ≥ 2/day
- Expect: source_diversity_score significantly improved

**Commit:** `feat(hotspot): Phase 3 — expand high-quality source coverage`

---

### Phase 4: Clustering & Scoring Refinement (Est. 1h)

**Goal:** With better source data flowing in, tune the clustering and scoring to maximize signal extraction.

#### 4A. Improve cross-source clustering

**Files:** `arxiv_assistant/utils/hotspot/hotspot_cluster.py`

With more sources, cross-source merging becomes more important:
1. Add URL domain canonicalization for new sources (e.g., `paperswithcode.com` → paper URLs, reddit post → external link)
2. Add entity matching for company/product names from official blog titles to roundup/social mentions
3. Lower cross-type matching threshold from 0.70 to 0.60 for (official + roundup) pairs — these are very likely about the same event
4. Add explicit link-following: if a roundup item's URL points to an official blog post already in the pipeline, merge them

Expected impact: Cluster compression ratio improves from 0.884 to ~0.75. Cross-source merges increase.

**Acceptance:** Compression ratio ≤ 0.80. False merge rate (sampled) < 5%.

#### 4B. Recalibrate heuristic scoring

**Files:** `arxiv_assistant/filters/filter_hotspots.py`

With better sources, the scoring formula needs rebalancing:
1. **Increase official source weight**: IMPORTANCE bonus for `has_official` from +2.3 to +3.0
2. **Increase multi-source weight**: CONFIDENCE bonus for `source_count > 1` from +0.9 to +1.4
3. **Decrease paper-only inflation**: QUALITY bonus for `has_paper` from +2.1 to +1.5 (papers are important but shouldn't dominate by default)
4. **Add analysis source bonus**: New bonus for `has_editorial_depth` source role: QUALITY +1.5, IMPORTANCE +0.8
5. **Strengthen HYPE_PENALTY**: For single-source community items, increase penalty from +3.4 to +4.0
6. **Add GitHub stars to scoring**: For repo items, factor in actual stars count more heavily (currently log-based, increase coefficient)

Expected impact: Official launches and multi-source events score significantly higher. Single-source papers and repos score lower.

**Acceptance:** Forward simulation shows featured topics dominated by multi-source and official-anchored items.

#### 4C. Increase LLM screening coverage

**Files:** `configs/config.ini`

Currently only 18 clusters go to LLM, and only 3-10 get actual review. With higher-quality input, we should screen more:
1. Increase `max_clusters_for_llm` from 18 to 30
2. Increase `max_review_clusters` from 10 to 15
3. Adjust auto-keep thresholds to be slightly stricter (let more items through to LLM review)
4. Increase `screening_score_cutoff` from 4.0 to 4.5 (be more selective)

Expected impact: More topics get LLM quality judgment. Fewer topics are auto-promoted by heuristics alone.

**Acceptance:** ≥ 50% of final featured topics have been LLM-reviewed.

#### Phase 4 checkpoint

Run full evaluation. Compare to Phase 3:
- Expect: multi_source_featured_ratio ≥ 70%
- Expect: single_source_featured_ratio ≤ 20%
- Expect: cluster_compression_ratio ≤ 0.80
- Expect: featured_precision_proxy ≥ 70%

**Commit:** `feat(hotspot): Phase 4 — refine clustering and scoring`

---

### Phase 5: LLM Screening & Prompt Refinement (Est. 1h)

**Goal:** Upgrade the LLM screening to make better keep/watchlist/drop decisions with new source context.

#### 5A. Enhance screening prompt with source quality context

**Files:** `prompts/hotspot/screening_criteria.txt`, `prompts/hotspot/postfix_prompt_screening.txt`

The LLM screening prompt should convey:
1. **Explicit source tier hierarchy** with concrete examples: "An official_post from OpenAI News about a model release is inherently more significant than a newsletter_recap from The Rundown AI about the same topic"
2. **Strict keep criteria**: KEEP requires one of:
   - Official source + genuine product/model/policy announcement
   - Multi-source (≥2) convergence on the same topic
   - High-impact research from trusted_research source with supporting evidence
3. **Deep Reads identification**: Tell LLM to flag `blog_analysis` items with high explanatory density as ARTIFACT_TYPE=blog_analysis, EVENT_TYPE=deep_analysis
4. **Negative examples**: Explicitly tell LLM to DROP:
   - Awesome-lists and curated link collections
   - Generic agent/MCP wrapper repos
   - Resurfaced papers (>14 days) unless there's a specific new development
   - Funding news without primary source
   - Conference/workshop announcements

#### 5B. Add KEY_TAKEAWAYS quality gate

**Files:** `arxiv_assistant/filters/filter_hotspots.py`

The LLM generates KEY_TAKEAWAYS for featured topics. Currently these are unconstrained. Add:
1. Require at least 2 non-trivial takeaways for KEEP status
2. Parse takeaway quality: reject if takeaway merely restates the headline
3. If LLM fails to produce meaningful takeaways, demote from KEEP to WATCHLIST

#### 5C. Improve summary synthesis quality

**Files:** `arxiv_assistant/hotspots/pipeline.py` (digest synthesis section)

The daily summary paragraph should be more opinionated and less bland:
1. Update summary prompt to focus on "what a senior AI engineer should know about today"
2. Require the summary to name specific companies, models, or papers
3. Cap at 3 sentences
4. Lead with the single most important development, not a generic "today's topics included..."

#### Phase 5 checkpoint

Run with LLM screening on latest available data. Compare:
- Expect: featured topics are more selective and defensible
- Expect: KEY_TAKEAWAYS are substantive
- Expect: daily summary is specific and informative

**Commit:** `feat(hotspot): Phase 5 — improve LLM screening and synthesis`

---

### Phase 6: Rendering & Output Polish (Est. 0.5h)

**Goal:** Make the final output clean, professional, and information-dense.

#### 6A. Markdown renderer improvements

**Files:** `arxiv_assistant/renderers/hotspot/render_hot_daily.py`

1. **Featured topics format**: Add source tier badges (e.g., `[Official]`, `[Multi-Source]`, `[Analysis]`)
2. **Remove redundant metadata**: Don't show raw numeric scores (final=5.923) in the rendered output — these are internal pipeline signals, not user-facing information
3. **Semantic section headers**: Use descriptive subheadings (e.g., "Research Frontiers — Fresh papers with frontier impact" instead of just "Research Frontiers")
4. **Evidence links**: Show primary source first, corroboration second, with clear labels
5. **Empty section handling**: Don't render section header if section is empty

#### 6B. Web data output improvements

**Files:** `arxiv_assistant/utils/hotspot/hotspot_web_data.py`

1. Ensure new source types (analysis blogs, Reddit, etc.) are correctly mapped to source families
2. Add `source_tier` and `artifact_type` to web payload for frontend rendering
3. Update `_item_signal_score()` to properly weight new source types

#### 6C. Coverage snapshot improvements

Add to the Coverage Snapshot section:
1. Source health status (which sources produced 0 items)
2. Quality distribution (how many items per tier)
3. Remove internal-only metrics from user-facing output

#### Phase 6 checkpoint

Visual inspection of rendered output. Must look professional and information-dense.

**Commit:** `feat(hotspot): Phase 6 — polish rendering and output format`

---

### Phase 7: End-to-End Validation & Tuning (Est. 0.5h)

**Goal:** Run the complete pipeline end-to-end, validate all metrics, fix any integration issues.

#### 7A. Full pipeline dry run

Run `generate_daily_hotspot_report()` with all new sources and logic. Verify:
1. All sources fetch without errors
2. Clustering produces reasonable groups
3. LLM screening makes good decisions
4. Semantic sections are populated correctly
5. Rendered output is clean

#### 7B. Forward simulation on historical data

Re-run evaluation on the 14-day historical period to measure improvement:
1. Source health improvements (can only be measured where cached raw data exists)
2. Scoring/classification improvements (measurable via forward simulation)
3. Rendering improvements (visual inspection of regenerated reports)

#### 7C. Final metric gate

All metrics must pass the quality gate from Section 2.3. If any metric fails, diagnose and fix before declaring complete.

#### 7D. Documentation update

**Files:** `docs/HOTSPOT_QUALITY_REDESIGN_PLAN.md`, `docs/HOTSPOT_V2_EXECUTION_PLAN.md`

Update both documents with:
1. Final measured metrics
2. Comparison to baseline
3. Known remaining limitations
4. Suggestions for further improvement (Phase 8+)

**Commit:** `feat(hotspot): Phase 7 — end-to-end validation and documentation`

---

## 4. File Impact Summary

### Files to create

| File | Purpose |
|------|---------|
| `configs/hotspot/official_blogs.json` | Registry-driven official blog sources |
| `configs/hotspot/analysis_feeds.json` | Long-form analysis feed registry |
| `configs/hotspot/github_blacklist.json` | GitHub repo blacklist patterns |
| `arxiv_assistant/apis/hotspot/hotspot_analysis_feeds.py` | Analysis feed fetcher |
| `arxiv_assistant/apis/hotspot/hotspot_reddit.py` | Reddit AI subreddit fetcher (if time allows) |
| `arxiv_assistant/apis/hotspot/hotspot_pwc.py` | Papers With Code fetcher (if time allows) |

### Files to modify (major changes)

| File | Changes |
|------|---------|
| `arxiv_assistant/apis/hotspot/hotspot_official_blogs.py` | Registry-driven instead of hard-coded; 8+ sources |
| `arxiv_assistant/apis/hotspot/hotspot_hf_papers.py` | Age filter, upvote minimum at ingestion |
| `arxiv_assistant/apis/hotspot/hotspot_github.py` | Higher stars threshold, blacklist patterns |
| `arxiv_assistant/apis/hotspot/hotspot_roundups.py` | Tier-aware per-site limits, paper URL dedup |
| `arxiv_assistant/apis/hotspot/hotspot_ainews.py` | Fix/diagnose RSS failure |
| `arxiv_assistant/apis/hotspot/hotspot_x_official.py` | Graceful degradation with logging |
| `arxiv_assistant/filters/filter_hotspots.py` | Rebalanced scoring weights |
| `arxiv_assistant/utils/hotspot/hotspot_cluster.py` | Better cross-source merging |
| `arxiv_assistant/hotspots/pipeline.py` | Integration of new sources, config updates |
| `arxiv_assistant/hotspots/evaluation.py` | New source health and output quality metrics |
| `arxiv_assistant/renderers/hotspot/render_hot_daily.py` | Cleaner rendering |
| `arxiv_assistant/utils/hotspot/hotspot_web_data.py` | New source family mappings |
| `prompts/hotspot/screening_criteria.txt` | Stricter keep criteria, negative examples |
| `prompts/hotspot/postfix_prompt_screening.txt` | Updated rules |
| `configs/config.ini` | Updated thresholds and limits |
| `configs/hotspot/source_tiers.json` | New source tier entries |

### Files to modify (minor changes)

| File | Changes |
|------|---------|
| `configs/hotspot/roundup_sites.json` | Adjusted per-site limits |
| `docs/HOTSPOT_QUALITY_REDESIGN_PLAN.md` | Status updates |

---

## 5. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| New source URLs change or go down | Use try/except with clear error logging; fallback to cached data |
| RSS feeds have different formats | Use feedparser (handles most formats); add per-source parsing if needed |
| Too many sources → LLM cost increase | Keep max_clusters_for_llm at 30; aggressive pre-filtering reduces volume |
| Over-aggressive filtering removes good content | Always compare to previous checkpoint; keep blacklists narrow |
| Cross-source clustering creates false merges | Keep merge threshold conservative; monitor false merge rate |
| New sources produce irrelevant content | Apply same keyword/freshness filters; test each source individually |

---

## 6. Time Budget

| Phase | Estimated Time | Cumulative |
|-------|---------------|-----------|
| Phase 1: Source Infrastructure Repair | 1.5h | 1.5h |
| Phase 2: Source Quality Gates | 1.5h | 3.0h |
| Phase 3: Expand High-Quality Sources | 2.0h | 5.0h |
| Phase 4: Clustering & Scoring Refinement | 1.0h | 6.0h |
| Phase 5: LLM Screening & Prompt Refinement | 1.0h | 7.0h |
| Phase 6: Rendering & Output Polish | 0.5h | 7.5h |
| Phase 7: End-to-End Validation & Tuning | 0.5h | 8.0h |

**Buffer strategy:** If Phase 3 (source expansion) runs long, reduce scope by prioritizing official blogs and analysis feeds; defer Reddit and Papers With Code to a follow-up iteration.

---

## 7. Success Criteria Summary

After 8 hours of execution, the hotspot system must:

1. **Read like a professional daily AI brief** — not a link dump
2. **Feature only genuinely important events** — multi-source confirmed, primary-source anchored
3. **Keep research fresh and impactful** — no 2023/2024 resurfaced papers
4. **Surface real product launches** — from official sources, not newsletter retellings
5. **Include substantive analysis** — Deep Reads with actual intellectual depth
6. **Show high-quality repos only** — notable projects, not awesome-lists
7. **Pass all quantitative quality gates** — see Section 2.3
8. **Have ≥8 active, healthy data sources** — diverse, reliable, high-quality

The target is a system that a senior AI engineer would pay $20/month to read every morning.
