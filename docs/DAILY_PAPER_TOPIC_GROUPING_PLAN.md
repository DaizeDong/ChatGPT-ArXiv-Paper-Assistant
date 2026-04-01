# Personalized Daily Paper Topic Grouping Plan

Updated: 2026-04-01
Status: Implemented, verified, and extended

## 1. Goal

The daily paper page should no longer render all selected papers as one flat list. It should:

1. Render papers in blocks that follow the maintained `paper topics` taxonomy.
2. Keep the existing ranking logic inside each block:
   - sort by `SCORE` descending
   - then by `RELEVANCE` descending
3. Hide empty topic blocks.
4. Use structured topic attribution instead of inferring sections from free-text `COMMENT`.
5. Keep daily, monthly, archive, hotspot local paper input, and historical rebuilds aligned to one taxonomy.

## 2. Progress Snapshot

### 2026-04-01 follow-up

Completed:

- daily paper abstract scoring now also produces `HOTSPOT_PAPER_TAGS` and `HOTSPOT_PAPER_COMMENT`
- hotspot-worthy papers are split out of the personalized daily paper list into a dedicated `hotspot-papers.json`
- daily hotspot report and web payload now expose a curated `paper_spotlight` section
- daily hotspot web page now renders spotlight paper panels ahead of the broader topic and source sections
- local hotspot paper ingestion now reads both:
  - `YYYY-MM-DD-output.json`
  - `YYYY-MM-DD-hotspot-papers.json`

Result:

- personalized daily papers remain focused on the user's foundational topic preferences
- same-day broad-interest papers and genuinely frontier-opening papers are still surfaced, but in the hotspot experience instead of the personalized paper page

### Phase status

- Phase 0. Taxonomy and display rules: completed
- Phase 1. Canonical topic registry: completed
- Phase 2. Abstract filter topic output contract: completed
- Phase 3. Parser validation and fallback: completed
- Phase 4. Daily schema upgrade: completed
- Phase 5. Daily markdown topic-grouped renderer: completed
- Phase 6. Monthly summary taxonomy alignment: completed
- Phase 7. Historical rebuild and backfill script: completed
- Phase 8. React paper pages: intentionally deferred

### Follow-up iteration status

- Iteration A. Diagnostics and observability: initial implementation completed
- Iteration B. Shared daily payload I/O: completed
- Iteration C. Better low-confidence topic handling: not started
- Iteration D. Optional front-end unification: deferred

### Delivered files

- `configs/paper_topics_schema.json`
- `arxiv_assistant/paper_topics.py`
- `arxiv_assistant/paper_daily_io.py`
- `arxiv_assistant/filters/filter_gpt.py`
- `main.py`
- `arxiv_assistant/renderers/paper/render_daily.py`
- `arxiv_assistant/renderers/paper/monthly_summary.py`
- `arxiv_assistant/renderers/paper/render_monthly_summary_with_link.py`
- `scripts/generate_monthly_summaries.py`
- `scripts/rebuild_paper_markdown.py`
- `arxiv_assistant/apis/hotspot/hotspot_local_papers.py`
- `prompts/paper/postfix_prompt_abstract.txt`
- `prompts/monthly/criteria.txt`
- `prompts/monthly/postfix_prompt.txt`
- `tests/test_paper_topics.py`
- `tests/test_paper_daily_io.py`
- `tests/test_paper_daily_renderer.py`
- `tests/test_monthly_summary_topics.py`
- `tests/test_rebuild_paper_markdown.py`

## 3. Final Architecture

### 3.1 Canonical topic registry

Implemented:

- `configs/paper_topics_schema.json`
- `arxiv_assistant/paper_topics.py`

The registry is now the single source of truth for:

- stable topic IDs
- topic labels
- display order
- prompt-visible topic descriptions
- fallback alias and keyword rules

The current stable topic IDs are:

1. `architecture_training`
2. `efficiency_scaling`
3. `representation_structure`
4. `memory_systems`
5. `world_models_open_ended_rl`

### 3.2 Shared topic normalization and fallback

Implemented in `arxiv_assistant/paper_topics.py`:

- registry loading and validation
- label or alias normalization back to stable topic IDs
- heuristic topic assignment for:
  - invalid LLM output
  - missing topic output
  - non-OpenAI runs
  - historical backfill
- stable daily sorting helper
- stable grouping helper
- companion bundle builder
- stable anchor helpers for topic sections and paper anchors

### 3.3 Daily output model

Implemented behavior:

- flat `YYYY-MM-DD-output.json` is preserved
- each flat paper entry now also carries topic metadata
- companion `YYYY-MM-DD-daily-papers.json` is also written

The daily companion bundle contains:

- `schema_version`
- `meta`
- `topic_order`
- `papers`
- `topic_sections`

### 3.4 Rendering model

Daily page rendering is now:

1. sort selected papers once with the shared daily sort helper
2. group sorted papers by `PRIMARY_TOPIC_ID`
3. render only non-empty sections in registry order
4. keep each topic block as the stable projection of the global ranking

Rendered page structure now includes:

- summary table
- `Topic Coverage` table
- topic-first table of contents
- topic sections with stable paper anchors based on arXiv IDs

### 3.5 Monthly summary model

Monthly summary now reuses the same taxonomy.

Implemented:

- generated monthly reports accept either old `PRIMARY_CATEGORY` labels or new `PRIMARY_TOPIC_ID`
- heuristic monthly fallback also uses shared topic attribution
- monthly summary sections are keyed by topic ID and rendered by topic label
- monthly report prompt now asks for `PRIMARY_TOPIC_ID` and `SECONDARY_TOPIC_IDS`

### 3.6 Historical rebuild model

Implemented:

- `scripts/rebuild_paper_markdown.py`

The rebuild script can:

- discover existing daily JSON payloads
- backfill topic fields into flat `output.json`
- generate companion `daily-papers.json`
- regenerate grouped daily markdown
- refresh top-level `out/latest.md`

### 3.7 Shared daily payload I/O and diagnostics

Implemented:

- `arxiv_assistant/paper_daily_io.py`

This helper now centralizes:

- daily JSON discovery
- flat output vs bundle payload extraction
- standardized daily JSON writing

Initial diagnostics are now included in the companion bundle and debug outputs:

- per-topic paper counts
- topic assignment source counts
- invalid primary topic fallback count
- missing primary topic fallback count
- heuristic vs default vs LLM assignment totals

## 4. Phase-by-Phase Delivery

## Phase 0. Freeze taxonomy and grouping rule

Completed result:

- daily grouping uses only `PRIMARY_TOPIC_ID`
- `MATCHED_TOPIC_IDS` is metadata only
- empty topic blocks are hidden

## Phase 1. Introduce canonical topic registry

Completed result:

- shared topic IDs, labels, and order now live in `configs/paper_topics_schema.json`
- code reads this registry through `arxiv_assistant/paper_topics.py`

## Phase 2. Extend abstract filter output contract

Completed result:

- `prompts/paper/postfix_prompt_abstract.txt` now asks for:
  - `PRIMARY_TOPIC_ID`
  - `MATCHED_TOPIC_IDS`
  - `TOPIC_MATCH_COMMENT`
- the abstract user prompt now injects a generated topic registry block, so allowed IDs stay in sync with the schema

## Phase 3. Parser validation and fallback

Completed result:

- `filter_gpt.py` now normalizes and validates topic output
- invalid or missing topic fields fall back through the shared heuristic classifier
- numeric score parsing was hardened so string-like model output does not leak through as unstable state

## Phase 4. Upgrade daily result schema

Completed result:

- `main.py` now writes:
  - enriched flat `output.json`
  - companion `daily-papers.json`
- shared sorting and grouping logic now comes from `arxiv_assistant/paper_topics.py`

## Phase 5. Refactor daily markdown renderer

Completed result:

- `render_daily.py` now renders:
  - topic coverage
  - topic-first table of contents
  - grouped paper sections
- anchors are stable and arXiv-based:
  - paper anchor: `paper-2501-12345`
  - topic anchor: `topic-memory-systems`
- renderer now sorts through the shared helper, so grouping correctness does not depend on the caller

## Phase 6. Align monthly summary to the same taxonomy

Completed result:

- `monthly_summary.py` no longer owns an old hard-coded category taxonomy
- `render_monthly_summary_with_link.py` now renders topic labels from the registry
- `scripts/generate_monthly_summaries.py` now uses:
  - shared topic registry prompt block
  - shared topic normalization
  - shared heuristic fallback

## Phase 7. Historical rebuild and backfill

Completed result:

- `scripts/rebuild_paper_markdown.py` added
- old daily JSON can be upgraded in place without rerunning the full acquisition pipeline

## Phase 8. React paper pages

Not implemented by design.

Reason:

- current personalized paper pages are still produced by the Python markdown/static-site pipeline
- no user-facing need required a React migration in this iteration

## 5. Acceptance Checklist

### Implemented acceptance criteria

- Daily page renders by topic order: yes
- Empty topics are hidden: yes
- Paper order inside each topic matches the global ranking projection: yes
- Papers are not duplicated across sections: yes
- Flat `output.json` remains readable by downstream consumers: yes
- Monthly summary taxonomy aligns with daily taxonomy: yes
- Historical daily pages can be rebuilt from saved JSON: yes
- Hotspot local paper import remains compatible: yes

### Explicit compatibility decisions

- `out/json/YYYY-MM/YYYY-MM-DD-output.json` remains the compatibility file for existing readers
- new structured data is additive via `YYYY-MM-DD-daily-papers.json`
- monthly loaders can still consume older `PRIMARY_CATEGORY` payloads during migration

## 6. Test Record

### Targeted test scope added

- `tests/test_paper_topics.py`
- `tests/test_paper_daily_io.py`
- `tests/test_paper_daily_renderer.py`
- `tests/test_monthly_summary_topics.py`
- `tests/test_rebuild_paper_markdown.py`

### Existing regression suites re-run

- `tests/test_build_multipage_site.py`
- `tests/test_render_static_site.py`
- full repository test suite

### Latest results

- `python -m py_compile ...`: passed
- `python -m pytest tests/test_paper_topics.py tests/test_paper_daily_renderer.py tests/test_rebuild_paper_markdown.py tests/test_monthly_summary_topics.py tests/test_build_multipage_site.py tests/test_render_static_site.py -q`: passed
- `python -m pytest -q`: passed

Final result:

- `61 passed`

## 7. Post-Implementation Review

### What improved beyond the original minimum plan

- The daily renderer now self-sorts through the shared helper instead of trusting the caller.
- GPT score parsing is now more defensive against string-typed numeric output.
- Hotspot local paper ingestion now preserves topic metadata in tags and metadata fields.
- Monthly loaders accept both old label-based reports and new ID-based reports, which reduces migration risk.
- Daily payload discovery and flat-vs-bundle extraction are now centralized instead of duplicated.
- Topic attribution diagnostics are now emitted as structured data for bundles and debug output.

### Remaining tradeoffs

- Fallback topic assignment is heuristic, not model-based, for historical backfill and non-OpenAI runs.
- The topic registry is still maintained as a JSON file plus prose topic prompt, even though the schema is now the machine truth.
- Diagnostics are currently written to JSON artifacts, not surfaced in the rendered paper page yet.

## 8. Optimization Space

### Good next targets

1. Add topic attribution diagnostics to a visible UI surface or markdown appendix when useful.
2. Add a low-cost optional second-pass classifier for low-confidence topic assignments.
3. Surface topic metadata in more downstream UIs, not only the paper page and monthly summary.
4. Consider generating `web_data/paper` only if the static markdown pipeline starts limiting iteration speed.

## 9. Iteration Plan From Here

## Iteration A. Diagnostics and observability

Goal:

- measure how often fallback topic assignment is used
- detect topic drift early

Deliverables:

- debug report for topic assignment source counts
- per-day topic histogram

Current status:

- initial structured diagnostics are implemented in the companion bundle and debug output
- histogram surfacing is not yet rendered in UI

## Iteration B. Shared daily payload I/O

Goal:

- remove duplicate daily payload parsing logic

Deliverables:

- one shared helper for flat output vs companion bundle
- monthly script and rebuild script migrated to it

Current status:

- completed

## Iteration C. Better low-confidence topic handling

Goal:

- improve classification quality on boundary papers

Deliverables:

- low-confidence heuristic threshold
- optional second-pass classifier or curated alias expansion

## Iteration D. Optional front-end unification

Goal:

- move paper day/month/year pages to the structured web pipeline only if it becomes necessary

Deliverables:

- `web_data/paper`
- paper routes in `web/`

## 10. Recommended Operational Use

1. Run the normal daily pipeline to produce enriched flat JSON, bundle JSON, and grouped markdown.
2. Run `scripts/generate_monthly_summaries.py` as before; it now uses the unified topic taxonomy.
3. Use `scripts/rebuild_paper_markdown.py` whenever historical daily pages need to be migrated or refreshed from saved JSON.

---

This document now serves as both the implementation plan and the progress ledger for the completed topic-grouping rollout.
