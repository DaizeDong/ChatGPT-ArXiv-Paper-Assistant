# Investigation: Stale Content in Daily Hotspots

**Date**: 2026-04-07
**Reporter**: Claude (automated investigation)
**Scope**: Why content from 2023-2025 appears in 2026-04-04 daily hotspot page

---

## 1. Problem Verification

### 1.1 Stale Items Found in `web/public/web_data/hot/2026-04-04.json`

Analysis of all 41 items in `source_sections` against the target date 2026-04-04:

| Age Category | Count | Source Family |
|---|---|---|
| Within 36 hours | 24 | official (7), market-signals (2), analysis (8), github (1), industry (6) |
| 2-7 days old | 9 | papers (4), github (5) |
| 1-6 months old | 1 | papers |
| 6-12 months old | 3 | papers |
| Over 1 year old | 4 | papers |
| **Total stale (>7 days)** | **8** | **All from papers** |

### 1.2 Specific Stale Items

All 8 severely stale items come from the **papers** section (source: `hf_papers` / Hugging Face Trending Papers):

| published_at | Age | Title |
|---|---|---|
| 2023-10-14 | ~2.5 years | A decoder-only foundation model for time-series forecasting |
| 2024-07-25 | ~1.7 years | Very Large-Scale Multi-Agent Simulation in AgentScope |
| 2024-10-08 | ~1.5 years | LightRAG: Simple and Fast Retrieval-Augmented Generation |
| 2024-12-28 | ~1.3 years | TradingAgents: Multi-Agents LLM Financial Trading Framework |
| 2025-04-10 | ~1 year | The AI Scientist-v2: Workshop-Level Automated Scientific Discovery |
| 2025-08-22 | ~7 months | AgentScope 1.0: A Developer-Centric Framework |
| 2025-08-26 | ~7 months | VibeVoice Technical Report |
| 2025-10-16 | ~5.5 months | PaddleOCR-VL: Boosting Multilingual Document Parsing |

### 1.3 Stale Content in Featured Topics

Featured topics also reference stale papers as evidence. For example:
- **"AgentScope 1.0"** (published 2025-08-22) appears as a featured topic
- **"VibeVoice Technical Report"** (published 2025-08-26) appears as a featured topic

These topics have no `published_at` in the `featured_topics` evidence array, so the web UI cannot show dates to users.

### 1.4 Other Source Families

All non-paper sources produce correctly dated, recent items:
- **official**: All 7 items within 36h
- **market-signals**: Both items within 36h
- **analysis**: All 8 items within 36h
- **github**: 1 item within 36h, 5 items within 2-7 days (these are repo creation dates, which is acceptable since they are *trending* repos)
- **industry**: All 6 items within 36h

---

## 2. Root Cause Analysis

### 2.1 The `fetched_at` Bypass (Primary Root Cause)

The core problem is a design flaw in the `get_freshness_date()` function that allows HF trending papers to bypass all freshness checks.

**File**: `arxiv_assistant/utils/hotspot/hotspot_sources.py`, lines 145-156

```python
def get_freshness_date(item: "HotspotItem") -> str | None:
    fetched_at = (item.metadata or {}).get("fetched_at")
    if fetched_at:
        return fetched_at       # <-- ALWAYS returns target_date for HF papers
    return item.published_at
```

**File**: `arxiv_assistant/apis/hotspot/hotspot_hf_papers.py`, line 79

```python
metadata={
    "fetched_at": target_date.isoformat(),  # <-- Always set to today
    ...
}
```

**The bypass chain**:
1. HF adapter scrapes trending papers from `huggingface.co/papers/trending`
2. Each paper's `published_at` is the arxiv publication date (can be years old)
3. The adapter sets `metadata["fetched_at"] = target_date` for every paper
4. `get_freshness_date()` returns `fetched_at` (today) instead of `published_at` (the real date)
5. The pipeline freshness gate (pipeline.py line 1584) calls `get_freshness_date()`, gets today's date, and passes the item through
6. `_is_item_on_date()` in web_data.py also calls `get_freshness_date()`, same bypass

**Result**: A paper published in 2023 that is trending on HF today bypasses all freshness filters.

### 2.2 Why `fetched_at` Was Introduced

The `fetched_at` mechanism was designed for **GitHub trending repos**:
- A GitHub repo's `created_at` date is when it was first created
- A repo created months ago can become *trending* today
- Using `created_at` for freshness would incorrectly reject trending repos
- `fetched_at` correctly represents "when we observed it trending"

**File**: `arxiv_assistant/apis/hotspot/hotspot_github.py`, lines 93/108

```python
created_at = row.get("created_at") or row.get("updated_at")  # published_at
metadata={"fetched_at": target_date.isoformat(), ...}         # freshness override
```

For GitHub repos, this makes sense because `created_at` is 2-10 days before (configured via `created_within_days`), and the `fetched_at` override prevents them from being filtered out.

### 2.3 Why It Fails for HF Papers

For HF papers, the situation is fundamentally different:
- `published_at` is the arxiv submission date (the actual publication date)
- Papers from 2023 can trend on HF in 2026 for various reasons (new interest, viral tweets, etc.)
- A 2-year-old paper trending on HF is NOT breaking news
- `fetched_at` should NOT override `published_at` for papers

### 2.4 The HF Adapter Does Not Filter by Publication Date

**File**: `arxiv_assistant/apis/hotspot/hotspot_hf_papers.py`

The HF adapter:
1. Accepts a `freshness_hours` parameter but **never uses it**
2. Does not call `is_fresh()` at all
3. Only filters by upvote count (`MIN_UPVOTES = 5`)
4. Accepts any paper regardless of age

Compare to other adapters:
- `hotspot_ainews.py`: calls `is_fresh()` (line 208)
- `hotspot_official_blogs.py`: calls `is_fresh()` (line 134)
- `hotspot_roundups.py`: calls `is_fresh()` (line 199)
- `hotspot_analysis_feeds.py`: calls `is_fresh()` (line 62)
- `hotspot_reddit.py`: has timestamp cutoff (line 26)
- `hotspot_hn.py`: calls `is_fresh()` (line 60)

### 2.5 Per-Source Freshness Window Analysis

| Source | Freshness Mechanism | Effective Window | Risk |
|---|---|---|---|
| `official_blogs` | `is_fresh()` | 36h (default) | Low |
| `roundup_sites` | `is_fresh()` | 36h (default) | Low |
| `ainews` | `is_fresh()` | 96h (4 days) | Low |
| `analysis_feeds` | `is_fresh()` | 168h (7 days) | Low |
| `reddit` | timestamp cutoff | 48h | Low |
| `hn_discussion` | `is_fresh()` | 36h | Low |
| `x_ainews_twitter` | `is_fresh()` | 96h | Low |
| `x_paperpulse` | `is_fresh()` | 36h | Low |
| **`hf_papers`** | **None** | **Unlimited** | **Critical** |
| `github_trend` | `fetched_at` bypass | `created_within_days` config (10 days) | Low (controlled) |
| `local_papers` | `max_staleness_days` | 2 days | Low |

### 2.6 Story Aggregation Amplification

When a stale paper gets through the freshness gate, it can be grouped into a Story with fresh items from other sources. The story-centric pipeline (`story.py`) uses the *maximum* `freshness_date` across all items in a story for scoring (line 278):

```python
freshness_dates = [get_freshness_date(ei.item) for ei in story.items]
freshness = _freshness_weight(max(freshness_dates) if freshness_dates else None)
```

This means a fresh Reddit post about a 2023 paper can pull that stale paper into a high-scoring story.

---

## 3. Solution

### 3.1 Fix 1: Remove `fetched_at` from HF Papers Adapter (Critical)

**File**: `arxiv_assistant/apis/hotspot/hotspot_hf_papers.py`

Remove the `fetched_at` key from metadata. The HF papers adapter should rely on `published_at` (the actual arxiv publication date) for freshness evaluation:

```python
# BEFORE (line 79):
metadata={
    "fetched_at": target_date.isoformat(),
    ...
}

# AFTER:
metadata={
    # No fetched_at — use published_at for freshness
    ...
}
```

### 3.2 Fix 2: Add `is_fresh()` Check to HF Papers Adapter (Critical)

**File**: `arxiv_assistant/apis/hotspot/hotspot_hf_papers.py`

Add explicit freshness filtering using the `published_at` date. Use a wider window (e.g., 30 days) since papers can take time to gain attention on HF:

```python
from arxiv_assistant.utils.hotspot.hotspot_sources import is_fresh

# In fetch_hotspot_items(), after getting published_at:
MAX_PAPER_AGE_HOURS = 30 * 24  # 30 days

published_at = paper.get("publishedAt") or target_date.isoformat()

# Filter: skip papers that are too old
if not is_fresh(published_at, target_date, MAX_PAPER_AGE_HOURS):
    continue
```

### 3.3 Fix 3: Make `get_freshness_date()` Source-Aware (Recommended)

**File**: `arxiv_assistant/utils/hotspot/hotspot_sources.py`

The current `get_freshness_date()` blindly prefers `fetched_at` for all sources. It should only use `fetched_at` for sources where the concept makes sense (e.g., GitHub repos):

```python
# Sources where fetched_at should override published_at
_FETCHED_AT_VALID_SOURCES = {"github_trend"}

def get_freshness_date(item: "HotspotItem") -> str | None:
    if item.source_id in _FETCHED_AT_VALID_SOURCES:
        fetched_at = (item.metadata or {}).get("fetched_at")
        if fetched_at:
            return fetched_at
    return item.published_at
```

### 3.4 Fix 4: Add Maximum Age Guard to Pipeline Freshness Gate (Defense in Depth)

**File**: `arxiv_assistant/hotspots/pipeline.py`, around line 1577

Add a hard maximum age check that uses `published_at` directly, regardless of `fetched_at`. No item should appear in daily hotspots if its actual publication date is more than N days ago:

```python
MAX_ITEM_AGE_DAYS = 14  # Hard ceiling: no item older than 14 days

# After the existing freshness gate:
max_age_cutoff = target_utc - timedelta(days=MAX_ITEM_AGE_DAYS)
pre_age = len(raw_items)
age_filtered = []
for item in raw_items:
    if item.published_at:
        dt = parse_datetime(item.published_at)
        if dt is not None and dt < max_age_cutoff:
            continue  # Too old regardless of fetched_at
    age_filtered.append(item)
raw_items = age_filtered
if len(raw_items) < pre_age:
    print(f"Max-age filter: removed {pre_age - len(raw_items)} items older than {MAX_ITEM_AGE_DAYS}d")
```

### 3.5 Fix 5: Add Published Date to Featured Topic Evidence (UI Improvement)

**File**: `arxiv_assistant/utils/hotspot/hotspot_web_data.py`, `_build_compact_topic()` function

Currently, evidence items in `featured_topics` do not include `published_at`:

```python
# BEFORE (line 551):
evidence.append({
    "title": ...,
    "url": ...,
    "source_name": ...,
})

# AFTER:
evidence.append({
    "title": ...,
    "url": ...,
    "source_name": ...,
    "published_at": item.get("published_at"),
})
```

This allows the web UI to display dates and lets users identify stale content.

---

## 4. Implementation Priority

| Priority | Fix | Impact | Effort |
|---|---|---|---|
| P0 | Fix 1: Remove `fetched_at` from HF adapter | Eliminates the bypass | 1 line |
| P0 | Fix 2: Add `is_fresh()` to HF adapter | Source-level filtering | ~5 lines |
| P1 | Fix 3: Source-aware `get_freshness_date()` | Prevents future similar bugs | ~10 lines |
| P1 | Fix 4: Pipeline max-age guard | Defense in depth | ~10 lines |
| P2 | Fix 5: Published date in evidence | UI transparency | 1 line |

Fixes 1 and 2 alone would resolve the current issue. Fixes 3 and 4 provide systemic protection against similar issues from future source additions.

---

## 5. Files Requiring Changes

| File | Changes |
|---|---|
| `arxiv_assistant/apis/hotspot/hotspot_hf_papers.py` | Remove `fetched_at`, add `is_fresh()` call |
| `arxiv_assistant/utils/hotspot/hotspot_sources.py` | Make `get_freshness_date()` source-aware |
| `arxiv_assistant/hotspots/pipeline.py` | Add max-age guard using `published_at` directly |
| `arxiv_assistant/utils/hotspot/hotspot_web_data.py` | Add `published_at` to topic evidence |
