# Daily AI Hotspots Web Redesign Plan

Last updated: 2026-03-22

## Progress Status

Current branch: `test_news`

| Phase | Scope | Status |
| --- | --- | --- |
| Baseline | current hotspot coverage expansion before frontend migration | completed |
| Phase 1 | `web_data` schema and JSON generation | completed |
| Phase 2 | frontend scaffold and build chain | completed |
| Phase 3 | source-first daily page | completed |
| Phase 4 | source/topic detail pages | completed |
| Phase 5 | month/year archive pages | completed |
| Phase 6 | publish workflow migration | completed |

## 1. Purpose

This document describes the next-stage redesign of the `Daily AI Hotspots` product in this repository.

The current hotspot pipeline already has a workable ranking algorithm and output archive, but the presentation layer is still constrained by a markdown-first workflow. That approach is now the bottleneck.

The product goal has changed:

- the daily hotspot page should maximize **same-day coverage**
- information should be organized primarily by **source family**
- each item should stay **compact**
- topic aggregation should remain important, but it should become a **secondary navigation layer**, not the only main view

The redesign therefore moves the hotspot experience from a markdown-rendered report into a static web application that reads structured JSON at runtime.

## 2. Product Goals

The redesigned hotspot system should answer the following questions well:

- What happened in AI today across papers, blogs, official announcements, GitHub, and social/community channels?
- Which source families are driving the day?
- What topics appear repeatedly across multiple sources?
- How can a user quickly scan many signals without reading long cards?

The page should behave more like a dense AI news terminal or research radar than a blog post.

## 3. Core UX Principles

The redesigned daily page will follow these principles:

- Coverage first: the page should expose many same-day signals, not just five polished topics.
- Source-first organization: users should be able to scan the day by source family.
- Short item format: most items should use one to three lines only.
- Topic-aware navigation: topic clustering should remain available as a summary and drill-down path.
- Static deployment: the final product must still build to static assets for GitHub Pages.
- Code/data separation: Python generates data; the frontend renders it.

## 4. Target Information Architecture

The hotspot product should expose these page families:

### 4.1 Daily hotspot page

Route:

- `/hot/YYYY-MM-DD/`

Primary purpose:

- high-density scan of the day across all major source families

### 4.2 Source detail page

Routes:

- `/hot/YYYY-MM-DD/source/x-buzz/`
- `/hot/YYYY-MM-DD/source/blogs/`
- `/hot/YYYY-MM-DD/source/official/`
- `/hot/YYYY-MM-DD/source/papers/`
- `/hot/YYYY-MM-DD/source/github/`
- `/hot/YYYY-MM-DD/source/discussions/`

Primary purpose:

- preserve more same-day candidates without overloading the main page

### 4.3 Topic detail page

Routes:

- `/hot/YYYY-MM-DD/topic/<slug>/`
- optionally `/hot/YYYY-MM-DD/category/<slug>/`

Primary purpose:

- merge repeated evidence into one view for deeper reading

### 4.4 Monthly and yearly archive pages

Routes:

- `/hot/YYYY-MM/`
- `/hot/YYYY/`

Primary purpose:

- navigation, trend inspection, and archive recovery

## 5. Daily Page Layout

The daily page should no longer be a linear markdown document. It should be a structured web UI.

Recommended layout:

1. Top navigation
2. Day summary strip
3. Topic summary strip
4. Source-first content sections
5. Long-tail coverage area

### 5.1 Top navigation

Include:

- previous / next day
- month archive
- year archive
- switch between `Paper Digest` and `AI Hotspots`

### 5.2 Day summary strip

This is a compact bar, not a large hero section.

It should show:

- date
- total signals collected
- number of featured topics
- number of source families represented
- total links shown on page

### 5.3 Topic summary strip

This replaces the old idea that the main page should itself be topic-centric.

The strip should show:

- 3 to 6 highest-priority themes
- each theme represented by:
  - title
  - occurrence count
  - number of contributing sources

The strip acts as a fast map of the day rather than a large editorial block.

### 5.4 Source-first sections

These are the core of the page. Each section should be dense and compact.

Required sections:

- `X / Buzz`
- `Blogs / Newsletters`
- `Official Updates`
- `Papers`
- `GitHub / Tools`
- `Discussions`

Each section should:

- display many compact items
- support `show more`
- expose source chips and tags
- let the user drill into source-specific detail pages

### 5.5 Long-tail signals

This section exists to preserve breadth.

It should:

- show lower-priority but still relevant same-day items
- remain one-line or two-line per item
- favor density over explanation

The long-tail area is essential because it prevents the pipeline from discarding too much information after aggregation.

## 6. Data Architecture

The redesigned frontend should be **item-first**, not topic-first.

### 6.1 Primary data unit: HotspotFeedItem

The frontend should consume source items directly.

Each item should expose:

- `id`
- `date`
- `title`
- `summary_short`
- `url`
- `source_family`
- `source_name`
- `source_role`
- `published_at`
- `heat`
- `evidence_score`
- `occurrence_count`
- `topic_ids`
- `tags`

### 6.2 Secondary data unit: HotspotTopic

Topics remain important, but as an index over items.

Each topic should expose:

- `id`
- `title`
- `summary`
- `category`
- `featured`
- `source_count`
- `item_count`
- `top_source_families`
- `rank`
- `item_ids`

### 6.3 Daily page payload

Daily JSON should include:

- `meta`
- `featured_topics`
- `topic_summary_strip`
- `source_sections`
- `long_tail_sections`
- `source_stats`
- `totals`

Suggested output path:

```text
out/web_data/
  hot/
    2026-03-21.json
    2026-03/
      index.json
    2026/
      index.json
    index.json
```

## 7. Ranking Model

The UI redesign does not remove the hotspot ranking algorithm. It reassigns the ranking outputs to different presentation roles.

### 7.1 Featured layer

Use the strongest hybrid scoring:

- quality
- heat
- evidence
- multi-source resonance
- LLM judgement

Keep this layer small.

### 7.2 Source section layer

Rank within each source family primarily by:

- same-day relevance
- source-local heat
- occurrence in other source families
- topic linkage

This layer should be broader and less selective than the featured layer.

### 7.3 Long-tail layer

Keep this layer wide but compact.

Rules:

- keep only AI-relevant items
- aggressively strip verbosity
- allow lower-priority candidates to remain visible

## 8. Frontend Architecture

The hotspot site should become a real frontend application.

Recommended stack:

- `Vite`
- `React`
- `TypeScript`
- lightweight router
- CSS modules or plain CSS with clear tokens

Why this stack:

- builds to static assets
- fast local development
- good fit for GitHub Pages
- enough structure for multiple archive/detail routes

Suggested source tree:

```text
web/
  src/
    app/
    components/
    pages/
    sections/
    lib/
    styles/
  public/
```

## 9. Visual Direction

The hotspot UI should feel modern, dense, and intentional.

Recommended design direction:

- dashboard-like layout rather than article-like layout
- strong section headers
- compact cards or list rows
- source chips
- topic chips
- density toggle
- sticky filter bar
- restrained but clear accent color system

The page should prioritize scan speed and structure over decorative hero blocks.

## 10. Build and Deployment Flow

The build should remain static and CI-friendly.

Target pipeline:

1. Python pipeline generates:
   - raw hotspot data
   - normalized hotspot data
   - web-facing JSON payloads
2. Frontend build reads `out/web_data/`
3. Frontend builds static assets into `dist/`
4. GitHub Actions deploys `dist/` to Pages

Important constraint:

- `dist/` must not be committed
- `main` remains code-only
- `auto_update` remains data-only

## 11. Migration Plan

The redesign should be implemented in stages.

### Phase 1: Define the new JSON contract

Deliverables:

- item-first daily payload
- monthly/yearly index payloads
- schema documentation

Success condition:

- existing hotspot pipeline can generate frontend-ready JSON without depending on markdown

### Phase 2: Scaffold the frontend app

Deliverables:

- `web/` project
- routing skeleton
- global layout
- theme support

Success condition:

- static shell builds locally and deploys to Pages

### Phase 3: Build the source-first daily page

Deliverables:

- day summary strip
- topic summary strip
- source-family sections
- long-tail section
- compact item UI

Success condition:

- one daily page is fully usable without markdown

### Phase 4: Build source detail and topic detail pages

Deliverables:

- per-source subpages
- per-topic subpages
- links from the daily page

Success condition:

- main page can stay dense while deeper pages preserve the rest of the day

### Phase 5: Build monthly and yearly archive pages

Deliverables:

- hotspot month page
- hotspot year page
- source and topic distribution summaries

Success condition:

- archive navigation works without the old markdown layout

### Phase 6: Replace the old markdown hotspot frontend

Deliverables:

- Pages entry switched to the new app
- markdown hotspot pages kept only as debug/fallback output

Success condition:

- the new frontend is the primary user-facing hotspot interface
- publish CI rebuilds hotspot `web_data` from saved reports and normalized items before the web build
- final Pages output serves the React hotspot app under `/hot/...` routes without committing generated HTML
- manual `workflow_dispatch` can validate non-`main` site branches by overriding the site-code ref

## 12. Acceptance Criteria

The redesign is successful when:

- the daily page shows a wide same-day signal surface
- the page is organized by source family rather than only topic
- most items are readable in one quick scan
- topic navigation still exists as a secondary layer
- archives remain static and reproducible
- no generated HTML is committed to `main`

## 13. Non-goals

The redesign is not trying to:

- build a server-backed realtime product
- replace the paper digest
- make every item deeply editorialized
- depend on client-side APIs that require secrets at runtime

## 14. Immediate Next Step

The next implementation slice should be:

1. define the new `web_data` schema
2. generate one JSON payload for `/hot/YYYY-MM-DD/`
3. scaffold the frontend shell
4. render a single source-first daily hotspot page from JSON

That slice is the smallest step that proves the new architecture and unlocks the rest of the redesign.
