# Investigation: Cross-Day Content Duplication in AI Hotspot System

**Date**: 2026-04-07
**Status**: Root cause identified, solution designed

---

## 1. Problem Verification

### 1.1 Exact-Duplicate Headlines Across Days

Analysis of web data files from `web/public/web_data/hot/2026-03-28.json` through `2026-04-04.json` reveals severe cross-day content duplication:

| Headline | Days Present | Count |
|----------|-------------|-------|
| "Anthropic essentially bans OpenClaw from Claude by making subscribers pay extra" | 03-28, 03-29, 03-30, 03-31, 04-01, 04-04 | **6** |
| "Anthropic ramps up its political activities with a new PAC" | 03-28, 03-29, 03-30, 03-31, 04-01, 04-04 | **6** |
| "Google releases Gemma 4 models." | 03-28, 03-29, 03-30, 03-31, 04-01 | **5** |
| "Netflix just dropped their first public model..." | 03-28, 03-30, 03-31, 04-01 | **4** |
| "We gave 12 LLMs a startup to run for a year..." | 03-29, 03-30, 03-31, 04-01 | **4** |
| "OpenAI Leadership Reshuffle..." | 03-28, 03-29, 03-30 | **3** |
| "Preview tool helps makers visualize 3D-printed objects" | 03-28, 03-29, 03-30 | **3** |
| "Evaluating the ethics of autonomous systems" | 03-28, 03-29, 04-03 | **3** |

### 1.2 Near-Duplicate Headlines (Same Event, Different Wording)

The same underlying event appears with slightly different headlines across days:

- **"Anthropic acquires Coefficient Bio"** — appears in 7 different days with 4 headline variants
- **"OpenAI Acquires TBPN"** — appears in 8 different days with 6 headline variants
- **"Gemma 4 release"** — appears in 8 different days with 5 headline variants
- **"OpenAI Leadership Reshuffle"** — appears in 5 different days with 3 variants

### 1.3 Evidence URL Overlap

At the raw evidence level, the overlap is even more stark:

| Day Pair | Shared Evidence URLs | Overlap Rate |
|----------|---------------------|-------------|
| 03-28 vs 03-29 | 19 shared out of 32,29 | ~59-66% |
| 03-29 vs 03-30 | 19 shared out of 29,27 | ~66-70% |
| 03-30 vs 03-31 | 19 shared out of 27,32 | ~59-70% |
| 03-31 vs 04-01 | 19 shared out of 32,32 | ~59% |
| 04-01 vs 04-02 | 1 shared out of 32,26 | ~3% (improvement) |
| 04-02 vs 04-03 | 4 shared out of 26,31 | ~13-15% |
| 04-03 vs 04-04 | 5 shared out of 31,34 | ~15-16% |

Multiple evidence URLs (like specific Reddit posts, TechCrunch articles, Ars Technica articles) appear in **6 consecutive days**.

---

## 2. Root Cause Analysis

### 2.1 Root Cause #1: Overly Wide Source Freshness Windows

**Location**: Individual source fetchers in `arxiv_assistant/apis/hotspot/`

The `is_fresh()` function in `hotspot_sources.py` (line 159) uses a **symmetric** freshness window: it accepts items within `+/- freshness_hours` of the target date. This means even with `freshness_hours=36`, the total acceptance window is **72 hours (3 days)**.

But multiple sources override this to be even wider:

| Source | Effective Window | Code Location |
|--------|-----------------|---------------|
| `ainews` | `max(36, 96)` = **192h total (8 days)** | `hotspot_ainews.py:191` |
| `x_ainews_twitter` | `max(36, 96)` = **192h total (8 days)** | `hotspot_x_ainews.py:112` |
| `analysis_feeds` | `max(36, 168)` = **336h total (14 days)** | `hotspot_analysis_feeds.py:42` |
| `reddit` | `max(36, 48)` = **96h total (4 days)** | `hotspot_reddit.py:26` |
| All others | Config `freshness_hours=36` = **72h total (3 days)** | `config.ini:64` |

### 2.2 Root Cause #2: Pipeline Freshness Gate Is Also Too Loose

**Location**: `arxiv_assistant/hotspots/pipeline.py`, lines 1578-1589

The pipeline applies a secondary freshness gate after fetching:
```python
freshness_cutoff = target_utc - timedelta(hours=36)
```
This filters items older than 36h **before** the target date. But items published within 36h can still span 2 consecutive days (e.g., published at noon on April 1 passes the gate for both April 1 and April 2 target dates).

### 2.3 Root Cause #3: No Cross-Day Deduplication Mechanism

**Location**: `arxiv_assistant/hotspots/pipeline.py`, function `generate_daily_hotspot_report()`

The pipeline processes each day in **complete isolation**:
1. `fetch_source_payloads()` fetches raw items with the wide freshness windows
2. `group_into_stories()` clusters items into stories
3. `select_and_categorize()` selects featured topics
4. `write_hotspot_web_data()` writes the output

**At no point does the pipeline read or reference previous days' reports.** There is:
- No loading of prior day reports to check for already-featured topics
- No exclusion list of URLs/headlines that appeared recently
- No penalty for stories that were already covered in previous days
- No "staleness" tracking across pipeline runs

### 2.4 Root Cause #4: Raw Source Caching Compounds the Problem

**Location**: `pipeline.py`, lines 920-923

When `reuse_cached_raw = true` (default in `config.ini`), the pipeline caches raw source data per-day:
```python
cache_path = _raw_source_cache_path(output_root, target_date, source_id)
```
This is at `out/hot/raw/{date}/{source_id}.json`. Since each day creates a separate cache directory, the same items get cached independently for each day. However, the caching is per-day per-source, so it correctly avoids re-fetching within the same day — the issue is that the same external items are freshly fetched on different days due to the wide windows.

### 2.5 Root Cause #5: Story Grouping Only Considers Same-Day Items

**Location**: `arxiv_assistant/hotspots/story.py`, function `group_into_stories()`

The Union-Find story grouping (4 merge passes) only operates on items within the current day's pipeline run. It correctly merges duplicate items within a single day but has no awareness of stories from previous days.

### 2.6 Root Cause #6: Freshness Weight Decay Is Insufficient

**Location**: `arxiv_assistant/hotspots/story.py`, function `_freshness_weight()`

The freshness weight decay is:
- < 12h: 1.0
- < 24h: 0.8
- < 36h: 0.6
- ≥ 36h: 0.4

Even at 36+ hours old, an item still contributes 40% weight — enough to remain in the final output, especially if it has strong source signals (official news, high community activity).

---

## 3. Solution Design

### Phase 1: Cross-Day Exclusion List (Primary Fix)

**Goal**: Prevent topics that were already featured from appearing again on subsequent days.

#### 3.1 Build Historical Topic Index

**File**: `arxiv_assistant/hotspots/pipeline.py`
**New function**: `_load_recent_featured_topics()`

```python
def _load_recent_featured_topics(
    output_root: Path,
    target_date: datetime,
    lookback_days: int = 5,
) -> dict[str, dict[str, Any]]:
    """Load featured topics from the last N days' reports."""
    recent_topics: dict[str, dict[str, Any]] = {}  # URL -> topic info
    for days_back in range(1, lookback_days + 1):
        past_date = target_date - timedelta(days=days_back)
        report_path = output_root / "hot" / "reports" / f"{date_string(past_date)}.json"
        if not report_path.exists():
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for topic in report.get("featured_topics", []) + report.get("top_topics", []):
            for item in topic.get("items", []):
                url = item.get("canonical_url") or item.get("url", "")
                if url:
                    recent_topics[url] = {
                        "headline": topic.get("HEADLINE") or topic.get("title", ""),
                        "date": date_string(past_date),
                        "topic_id": topic.get("TOPIC_ID", ""),
                    }
        # Also collect category section topics
        for section in report.get("category_sections", []):
            for topic in section.get("topics", []):
                for item in topic.get("items", []):
                    url = item.get("canonical_url") or item.get("url", "")
                    if url:
                        recent_topics[url] = {
                            "headline": topic.get("HEADLINE") or topic.get("title", ""),
                            "date": date_string(past_date),
                            "topic_id": topic.get("TOPIC_ID", ""),
                        }
    return recent_topics
```

#### 3.2 Filter Raw Items Against History

**File**: `arxiv_assistant/hotspots/pipeline.py`
**Modify**: `generate_daily_hotspot_report()` — add a new filtering step after the freshness gate (around line 1589):

```python
# Cross-day dedup: remove items that were featured in recent days
recent_topics = _load_recent_featured_topics(output_root, target_date, lookback_days=5)
pre_dedup = len(raw_items)
raw_items = [
    item for item in raw_items
    if (item.canonical_url or item.url) not in recent_topics
]
if len(raw_items) < pre_dedup:
    print(f"Cross-day dedup: removed {pre_dedup - len(raw_items)} previously featured items")
```

### Phase 2: Story-Level Cross-Day Dedup (Stronger Catch)

**Goal**: Even if individual items slip through, detect and penalize stories that are semantically similar to previous days' featured topics.

#### 3.3 Story Staleness Penalty

**File**: `arxiv_assistant/hotspots/story.py`
**New function**: `apply_cross_day_penalty()`

After `score_stories()`, apply a penalty to stories that are similar to previously featured topics:

```python
def apply_cross_day_penalty(
    stories: list[Story],
    recent_headlines: list[str],
    penalty_factor: float = 0.3,
) -> list[Story]:
    """Penalize stories whose headlines are similar to recently featured topics."""
    from difflib import SequenceMatcher
    from arxiv_assistant.utils.hotspot.hotspot_cluster import significant_title_tokens

    recent_tokens = [significant_title_tokens(h) for h in recent_headlines]

    for story in stories:
        story_tokens = significant_title_tokens(story.headline)
        max_sim = 0.0
        for r_headline, r_tokens in zip(recent_headlines, recent_tokens):
            # Token Jaccard
            if story_tokens and r_tokens:
                union_size = len(story_tokens | r_tokens)
                if union_size > 0:
                    jaccard = len(story_tokens & r_tokens) / union_size
                    max_sim = max(max_sim, jaccard)
            # Sequence similarity
            seq_sim = SequenceMatcher(None, story.headline.lower(), r_headline.lower()).ratio()
            max_sim = max(max_sim, seq_sim)

        if max_sim >= 0.5:
            story.score *= (1.0 - penalty_factor * min(max_sim, 1.0))
            story.score = round(max(0.1, story.score), 3)

    stories.sort(key=lambda s: s.score, reverse=True)
    return stories
```

**Integrate in pipeline** (after `score_stories` call, around line 1604):

```python
stories = score_stories(group_into_stories(enriched_items))

# Apply cross-day staleness penalty
recent_headlines = [
    info["headline"]
    for info in _load_recent_featured_topics(output_root, target_date, lookback_days=5).values()
]
recent_headlines = list(set(recent_headlines))  # deduplicate
if recent_headlines:
    stories = apply_cross_day_penalty(stories, recent_headlines)
```

### Phase 3: Tighten Source Freshness Windows

**Goal**: Reduce the amount of stale content entering the pipeline at the source level.

#### 3.4 Source Freshness Adjustments

| File | Change | New Value |
|------|--------|-----------|
| `hotspot_ainews.py:191` | `max(freshness_hours, 96)` → `max(freshness_hours, 48)` | 48h (was 96h) |
| `hotspot_x_ainews.py:112` | `max(freshness_hours, 96)` → `max(freshness_hours, 48)` | 48h (was 96h) |
| `hotspot_analysis_feeds.py:42` | `max(freshness_hours, 168)` → `max(freshness_hours, 72)` | 72h (was 168h) |
| `hotspot_reddit.py:26` | `max(freshness_hours, 48)` → use `freshness_hours` directly | 36h (was 48h) |

#### 3.5 Make `is_fresh()` Asymmetric

**File**: `arxiv_assistant/utils/hotspot/hotspot_sources.py`, function `is_fresh()` (line 159)

The current symmetric window (`target_date +/- freshness_hours`) allows future-dated items and excessively old items. Change to asymmetric:

```python
def is_fresh(published_at: str | None, target_date: datetime, freshness_hours: int) -> bool:
    if published_at is None:
        return True
    published_dt = parse_datetime(published_at)
    if published_dt is None:
        return True
    if target_date.tzinfo is None:
        target_date = target_date.replace(tzinfo=UTC)
    window_start = target_date - timedelta(hours=freshness_hours)
    window_end = target_date + timedelta(hours=6)  # Only 6h future tolerance
    return window_start <= published_dt <= window_end
```

### Phase 4: Configuration Additions

**File**: `configs/config.ini`, section `[HOTSPOTS]`

Add new configuration options:

```ini
# Cross-day deduplication
cross_day_dedup_enabled = true
cross_day_lookback_days = 5
cross_day_penalty_factor = 0.3
cross_day_headline_similarity_threshold = 0.5
```

---

## 4. Implementation Priority

| Priority | Change | Impact | Effort |
|----------|--------|--------|--------|
| **P0** | Phase 1: Cross-day URL exclusion list | Eliminates exact same-item repeats | Low |
| **P0** | Phase 3.5: Asymmetric `is_fresh()` | Prevents future-dated item leakage | Low |
| **P1** | Phase 3.4: Tighten source freshness | Reduces stale input volume | Low |
| **P1** | Phase 2: Story-level staleness penalty | Catches semantically similar repeats | Medium |
| **P2** | Phase 4: Configuration | Makes thresholds tunable | Low |

---

## 5. Files to Modify

| File | Changes |
|------|---------|
| `arxiv_assistant/hotspots/pipeline.py` | Add `_load_recent_featured_topics()`, integrate cross-day dedup in `generate_daily_hotspot_report()` |
| `arxiv_assistant/hotspots/story.py` | Add `apply_cross_day_penalty()` |
| `arxiv_assistant/utils/hotspot/hotspot_sources.py` | Make `is_fresh()` asymmetric |
| `arxiv_assistant/apis/hotspot/hotspot_ainews.py` | Reduce effective freshness to 48h |
| `arxiv_assistant/apis/hotspot/hotspot_x_ainews.py` | Reduce effective freshness to 48h |
| `arxiv_assistant/apis/hotspot/hotspot_analysis_feeds.py` | Reduce effective freshness to 72h |
| `arxiv_assistant/apis/hotspot/hotspot_reddit.py` | Remove freshness override |
| `configs/config.ini` | Add cross-day dedup configuration |

---

## 6. Testing Plan

1. **Regression test**: Run the pipeline for 2026-04-03 and 2026-04-04 with the fix, verify that topics present on 04-03 do not appear on 04-04.
2. **Edge case**: Ensure genuinely evolving stories (e.g., a funding round announced on day 1 with new details on day 2) can still appear if they have sufficiently new evidence items.
3. **Minimum coverage**: Verify that the pipeline still produces at least `min_topics` (3) featured topics even after cross-day dedup removes candidates.
4. **Lookback window**: Test with `lookback_days=5` to ensure the system doesn't over-filter on slow news days.
