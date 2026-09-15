# Investigation: X/Twitter Coverage Gap in AI Hotspot System

Date: 2026-04-07

## 1. Current State Analysis

### 1.1 Existing X/Twitter Source Adapters

The system already has **three** X/Twitter source adapters:

| Source | Module | Config Key | Default | Actual |
|--------|--------|-----------|---------|--------|
| `x_official` | `hotspot_x_official.py` | `use_x_official` | `false` (template) | `true` (config.ini) |
| `x_ainews_twitter` | `hotspot_x_ainews.py` | `use_x_ainews_twitter` | `true` | `true` |
| `x_paperpulse` | `hotspot_x_paperpulse.py` | `use_x_paperpulse` | `true` | `true` |

Supporting infrastructure:
- `hotspot_x_common.py` -- shared newsworthiness filters, authority checking, regex patterns
- `x_authority_registry.py` -- X account discovery, following-graph expansion, registry management
- `configs/hotspot/x_authority_seeds.json` -- 52 manually curated seed accounts (29 company/official + 23 researchers)
- `configs/hotspot/x_authority_inventory.json` -- auto-expanded registry with **546 accounts** (310 active)
- `.github/workflows/refresh_x_authority_inventory.yml` -- monthly auto-refresh of the registry

### 1.2 Actual Output: X Items Per Day

Analysis of raw source data across 14 tracked dates (2026-03-18 to 2026-04-01):

| Source | Items Returned | Days with >0 items | Avg items/day |
|--------|---------------|--------------------|--------------:|
| `x_official` | **0 every day** | 0/14 | 0.0 |
| `x_paperpulse` | **0 every day** | 0/14 | 0.0 |
| `x_ainews_twitter` | 0-6 items | 6/14 | ~2.6 |
| (for comparison) `hf_papers` | 24 every day | 14/14 | 24.0 |
| (for comparison) `roundup_sites` | 21-23 | 14/14 | 22.0 |
| (for comparison) `github_trend` | 30 every day | 14/14 | 30.0 |

**Key finding**: Of the 3 X sources, only `x_ainews_twitter` produces any items, and only on ~43% of days (when AINews publishes a weekday issue with an "AI Twitter Recap" section). The other two sources consistently return zero items.

### 1.3 Root Causes

#### `x_official` -- Always Returns 0

This source calls the X API v2 Recent Search endpoint directly. It returns zero items because:

1. **Bearer Token likely not configured in GitHub Actions secret** -- while `X_BEARER_TOKEN` is referenced in the cron workflow, the actual secret may not be set, or the Free/Basic tier token lacks Recent Search access.
2. **Aggressive newsworthiness filters** -- even if data is fetched, `is_newsworthy_x_text()` applies multiple filter layers:
   - `SELF_WORK_PATTERNS` blocks tweets containing "we released", "new paper", "open source", "weights", "arxiv" -- which are exactly the phrases official AI accounts use for product launches.
   - `GENERAL_DROP_PATTERNS` blocks "challenge", "conference", "summit", "workshop" -- potentially catching legitimate AI conference announcements.
   - For official accounts without external links and activity <60, tweets are dropped.
   - For researcher accounts: requires external link + activity >= 40 + commentary pattern match -- very strict.
3. **Rate limit sensitivity** -- the `_query_accounts` function silently gives up on 429 errors and simply returns what it has (often nothing).

#### `x_paperpulse` -- Always Returns 0

This source fetches from `https://www.paperpulse.ai/api/researcher-feed`. It returns 0 because:

1. **Staleness detection** -- the adapter checks if the newest tweet in the feed is older than 7 days, and returns 0 if so. PaperPulse's researcher feed API may be frozen/stale.
2. **Restrictive filtering** -- only accepts tweets from accounts in the authority registry with `kind=researcher`, combined with the same strict `is_newsworthy_x_text` filters.

#### `x_ainews_twitter` -- Sporadic, Indirect

This is the only working X source, but it's severely limited:

1. **Indirect source**: Parses the "AI Twitter Recap" HTML section from the AINews RSS feed, not from X directly.
2. **Weekday-only**: AINews publishes only on weekdays, so weekends and some days have 0 items.
3. **Editorial curation**: Only captures tweets that AINews editors chose to highlight, missing many important X posts.
4. **Small yield**: When items exist, typically produces only 4-6 items per day.
5. **Authority filter**: Further drops items where the tweeter is not in the authority registry.

#### `ainews` source actively excludes X links

The general `ainews` adapter lists `x.com`, `twitter.com`, `mobile.twitter.com` in `INTERNAL_AINEWS_DOMAINS`, meaning X links are treated as "internal" and never selected as the best anchor URL for ainews items. This is by design to avoid duplication with `x_ainews_twitter`, but means the main ainews adapter can never surface X content.

### 1.4 Coverage Gap Summary

The system has a well-architected X integration with 546 tracked accounts, but **in practice produces essentially zero direct X/Twitter items per day**. The indirect `x_ainews_twitter` source provides sporadic, low-volume, editorially filtered X content. Major AI announcements that break first on X (product launches from OpenAI, Anthropic, Google, etc.) are only captured if secondary news sources pick them up.

## 2. Technical Feasibility Analysis

### 2.1 X API v2 Tiers and Capabilities

| Tier | Cost | Tweet Read Cap | Recent Search | User Lookup |
|------|------|---------------|---------------|-------------|
| Free | $0 | 10K reads/month (app-level) | No | Yes (limited) |
| Basic | $200/month | 50K reads/month | No | Yes |
| Pro | $5,000/month | 1M reads/month | Full, 10K results | Yes |
| Enterprise | Custom | Custom | Full | Full |

**Critical issue**: The Recent Search endpoint (used by `x_official`) requires **Pro tier ($5,000/month)** or higher. The Free and Basic tiers do not include search. This explains why `x_official` always returns 0 -- the bearer token likely belongs to a Free or Basic tier app.

With Free tier (10K reads/month):
- User Timeline lookup: ~333 tweets/day budget
- If monitoring 42 active accounts (24 official + 18 researcher), that's ~8 tweets per account per day -- tight but feasible for timeline-based fetching.

With Basic tier ($200/month, 50K reads/month):
- ~1,667 tweets/day budget
- ~40 tweets per account per day -- comfortable.

### 2.2 Alternative: Timeline-Based Fetching (User Tweets Endpoint)

The current `x_official` adapter uses Recent Search, which requires Pro tier. An alternative approach uses the **User Tweets endpoint** (`GET /2/users/:id/tweets`), available on **Basic tier ($200/month)**:

- Fetch latest tweets from each monitored account individually
- Rate limit: 900 requests per 15 min (app auth) or 900/15min (user auth)
- Can request up to 100 tweets per request
- Supports tweet.fields for metrics, entities, etc.

This would be a feasible redesign: instead of searching for tweets matching a query, iterate through priority accounts and fetch their recent tweets.

### 2.3 Existing Infrastructure Readiness

The codebase is well-prepared for expanded X integration:

- `X_BEARER_TOKEN` secret already configured in GitHub Actions workflows
- `x_authority_registry.py` already handles User Lookup and Following endpoints
- `x_authority_inventory.json` has 310 active accounts ready to monitor
- `is_newsworthy_x_text()` filter infrastructure exists (though needs tuning)
- Tier-based prioritization (`_authority_priority`) already ranks accounts
- Monthly registry refresh workflow already runs

## 3. Filter Analysis: Why Legitimate Tweets Get Dropped

### 3.1 The `SELF_WORK_PATTERNS` Problem

The most significant filter issue: `SELF_WORK_PATTERNS` drops tweets containing:
- "we released", "new paper", "open source", "weights", "arxiv", "preprint", "code for"

These patterns are applied to **all accounts including official company accounts**. But official AI company tweets like "We released GPT-5 today" or "New paper on reasoning" are exactly the high-value content the system should capture. The filter was designed to suppress researcher self-promotion but incorrectly applies to official accounts too.

**Impact**: Any OpenAI tweet saying "We released Claude Opus 5" would be filtered out.

### 3.2 The External Link Requirement

For official accounts without external links and activity < 60, tweets are dropped. Many important announcements are standalone tweets without external links (e.g., "Introducing GPT-5: our most capable model yet" with an image but no link). The activity threshold of 60 is low enough for major accounts, but the external link requirement is overly restrictive.

### 3.3 Researcher Filter Strictness

For researchers:
- Requires external link (drops commentary-only tweets)
- Requires activity >= 40
- Must match RESEARCHER_COMMENTARY_PATTERNS or OFFICIAL_NEWS_PATTERNS
- SELF_WORK_PATTERNS blocks any research announcements

A researcher sharing their reaction to a major AI release (high-signal commentary) would be blocked if they don't include an external link.

## 4. Recommended Solutions (Priority-Ordered)

### Priority 1: Fix the `x_official` Adapter to Use User Timeline Endpoint (Medium Effort)

**Problem**: Current adapter uses Recent Search (requires Pro tier at $5,000/month).

**Solution**: Rewrite `hotspot_x_official.py` to use the User Tweets timeline endpoint instead:

```
GET https://api.x.com/2/users/:id/tweets
```

This endpoint is available on Basic tier ($200/month) and even partially on Free tier.

**Implementation**:
1. Add a `_fetch_user_timeline()` function using `GET /2/users/:id/tweets`
2. Use the existing `iter_active_authority_accounts()` to get the priority account list
3. Fetch recent tweets for each account, applying rate limit awareness
4. Keep existing `is_newsworthy_x_text()` filtering (with fixes from Priority 2)

**Files to modify**:
- `arxiv_assistant/apis/hotspot/hotspot_x_official.py` -- replace `_iter_recent_search` with `_fetch_user_timeline`
- Config: Existing `[HOTSPOT_X]` section already has all needed parameters

**Cost**: Free tier for low volume (10K reads/month) or Basic tier ($200/month) for comfortable coverage.

### Priority 2: Fix the Newsworthiness Filters (Low Effort, High Impact)

**Problem**: `SELF_WORK_PATTERNS` blocks legitimate product launches from official accounts.

**Solution**: Exempt official/company accounts from `SELF_WORK_PATTERNS`:

```python
def is_newsworthy_x_text(text, *, authority_kind="official", ...):
    ...
    # Only apply self-work filters to researchers, not official accounts
    if authority_kind == "researcher":
        if any(pattern.search(normalized) for pattern in SELF_WORK_PATTERNS):
            return False
    # For official accounts, these are exactly the tweets we want
    ...
```

Additionally, for official accounts, lower the activity threshold when no external link is present (major announcements often have images, not links).

**Files to modify**:
- `arxiv_assistant/apis/hotspot/hotspot_x_common.py` -- adjust `is_newsworthy_x_text()` logic

### Priority 3: Fix PaperPulse Staleness Detection (Low Effort)

**Problem**: PaperPulse API may be returning stale data, triggering the staleness guard.

**Solution**:
1. Log the actual staleness date for diagnostics
2. Consider adding a fallback: if PaperPulse is stale, still return items but with a degraded quality score
3. Consider adding alternative researcher tweet feeds

**Files to modify**:
- `arxiv_assistant/apis/hotspot/hotspot_x_paperpulse.py`

### Priority 4: Add Missing Authority Accounts (Low Effort)

**Problem**: Some important AI accounts may be missing from the seed list.

**Recommended additions to `x_authority_seeds.json`**:

**Official accounts to add**:
- `@GoogleAI` -- Google's main AI account (already present)
- `@MSFTResearch` -- Microsoft Research
- `@AIatMeta` -- Meta's AI research division
- `@Sora` -- OpenAI's video generation
- `@CohereForAI` -- Cohere's research arm
- `@MistralAILabs` -- Mistral research
- `@reaborneAI` or `@Reka_AI` -- Reka AI
- `@adaborneAI` or `@AI2Inc` -- Allen Institute for AI
- `@Apple_ML` or Apple's ML account (if active)
- `@SambaNova` -- SambaNova Systems
- `@ModularAI` -- Modular (Mojo/MAX)
- `@datababorneAI` or `@datababorneAI` -- Databricks AI
- `@AmazonScience` -- Amazon Science
- `@Scale_AI` -- Scale AI

**Key researcher accounts to verify presence**:
- `@sama` -- Sam Altman (OpenAI CEO)
- `@aabornemodei` -- Daniela Amodei (Anthropic President)
- `@emaborneadwu` -- Emad Mostaque (ex-Stability AI)
- `@ylecun` -- Yann LeCun (Meta AI)
- `@hardmaru` -- David Ha (Sakana AI)
- `@kabornearpathy` -- Confirmed present as `andrejkarpathy`
- `@arthurmensch` -- Arthur Mensch (Mistral CEO)

**Files to modify**:
- `configs/hotspot/x_authority_seeds.json`

### Priority 5: Improve AINews Twitter Recap Extraction (Low Effort)

**Problem**: The existing `x_ainews_twitter` adapter only works on weekdays and yields few items.

**Solution**: The current implementation is correct but limited by AINews's publishing schedule. No code change needed -- this will always be a supplementary source.

### Priority 6: Add RSS/Nitter Proxy Sources (Medium Effort, Free)

**Problem**: Direct X API access is expensive.

**Alternative free sources for X content**:

1. **Nitter Instances** -- Nitter is a free, open-source Twitter frontend. Some instances provide RSS feeds for user timelines:
   - `https://nitter.net/<username>/rss` (availability varies, many instances are unstable)
   - **Risk**: Nitter instances are frequently blocked by X and go offline. Not reliable for production.

2. **RSS Bridge** -- Self-hosted service that creates RSS feeds from various websites including X:
   - Can be hosted on the same server or a VPS
   - `https://rss-bridge.org/bridge01/?action=display&bridge=TwitterBridge&u=openai&format=Atom`
   - **Risk**: Also subject to X blocking. Requires self-hosting.

3. **Third-Party Aggregators**:
   - **Paved/Newspipe/Feedbin** -- some services aggregate X content into RSS
   - **Feedly AI feeds** -- Feedly tracks X accounts and provides AI-curated feeds
   - **ThreadReaderApp** -- archives X threads, has RSS support

4. **Existing News Sources as X Proxies** -- The current `analysis_feeds` and `roundup_sites` sources already capture many X announcements indirectly:
   - TechCrunch AI, The Verge AI, Ars Technica AI -- cover major X announcements within hours
   - The Rundown AI, Superhuman AI -- newsletter roundups that include X highlights
   - AINews -- already captured via `x_ainews_twitter`

5. **Mastodon/Bluesky Cross-Posts** -- Many AI researchers cross-post to Mastodon and Bluesky, which have free, open APIs with no rate limits. Could add adapters for:
   - Bluesky API (`https://bsky.app/`) -- many AI researchers are active here
   - Mastodon API -- some researchers cross-post

**Recommendation**: Nitter/RSS Bridge approaches are fragile. The most reliable free approach is to ensure maximum coverage from existing news sources that echo X content, and supplement with Bluesky API.

## 5. Implementation Roadmap

### Phase 1: Quick Wins (1-2 days)

1. **Fix `is_newsworthy_x_text()` filters** -- exempt official accounts from SELF_WORK_PATTERNS
2. **Verify X_BEARER_TOKEN** is set in GitHub Actions secrets and determine its tier
3. **Add missing seed accounts** to `x_authority_seeds.json`

Files: `hotspot_x_common.py`, `x_authority_seeds.json`

### Phase 2: Timeline-Based X Source (3-5 days)

1. **Rewrite `x_official` to use User Timeline endpoint** instead of Recent Search
2. **Add rate-limit-aware batching** for timeline fetches
3. **Add diagnostic logging** to track filter pass/reject rates
4. **Update tests** in `test_hotspot_x_sources.py`

Files: `hotspot_x_official.py`, `test_hotspot_x_sources.py`, `config.ini`

### Phase 3: Supplementary Sources (5-7 days, optional)

1. **Add Bluesky adapter** for AI researchers who cross-post
2. **Evaluate PaperPulse API status** and fix or remove if permanently stale
3. **Consider a lightweight scraping approach** for critical accounts as a backup

Files: New `hotspot_bluesky.py`, updates to `pipeline.py` and `config.ini`

## 6. Cost Summary

| Approach | Monthly Cost | Expected Daily X Items | Reliability |
|----------|-------------|----------------------|-------------|
| Current state | $0 | 0-6 (indirect) | Low |
| Fix filters + Free tier timeline | $0 | 10-20 | Medium |
| Basic tier timeline | $200/month | 30-60 | High |
| Pro tier (current design) | $5,000/month | 60-100+ | High |
| Nitter/RSS Bridge | $0-5 (hosting) | 20-40 | Low (unstable) |
| Bluesky supplement | $0 | 5-15 | Medium |

**Recommended**: Fix filters (Phase 1, free) + Basic tier timeline rewrite (Phase 2, $200/month). This would increase X coverage from ~0 to 30-60 items/day at reasonable cost.

## 7. Key Files Reference

| File | Role |
|------|------|
| `arxiv_assistant/apis/hotspot/hotspot_x_official.py` | Direct X API adapter (needs redesign) |
| `arxiv_assistant/apis/hotspot/hotspot_x_ainews.py` | AINews Twitter recap parser (working, limited) |
| `arxiv_assistant/apis/hotspot/hotspot_x_paperpulse.py` | PaperPulse researcher feed (broken/stale) |
| `arxiv_assistant/apis/hotspot/hotspot_x_common.py` | Shared filters and authority checking |
| `arxiv_assistant/utils/hotspot/x_authority_registry.py` | Account discovery and graph expansion |
| `configs/hotspot/x_authority_seeds.json` | Manual seed accounts (52 entries) |
| `configs/hotspot/x_authority_inventory.json` | Auto-expanded registry (546 accounts) |
| `configs/hotspot/source_tiers.json` | Source trust tier mappings |
| `configs/config.ini` | Source enable/disable flags and parameters |
| `.github/workflows/cron_runs.yaml` | Daily pipeline with X_BEARER_TOKEN |
| `.github/workflows/refresh_x_authority_inventory.yml` | Monthly registry refresh |
| `arxiv_assistant/hotspots/pipeline.py` | Source orchestration (lines 857-885) |
| `tests/test_hotspot_x_sources.py` | X source unit tests |
