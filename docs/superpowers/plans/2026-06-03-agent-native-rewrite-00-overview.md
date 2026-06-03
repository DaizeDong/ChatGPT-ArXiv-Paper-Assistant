# Agent-Native Hotspot Rewrite — Plan Index & Shared Contract

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement each stage plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Migrate the AI-hotspot pipeline to an agent-native architecture (thin deterministic Python Kernel + stateless subagents at judgment points + persistent Story Store) that root-cause-fixes cross-day duplication, staleness, and X coverage — without rewriting the renderers or the paper-filtering pipeline.

**Architecture:** Strangler-fig migration in 7 stages (0–6) plus a paper-pipeline agent-filter add-on. Each stage ships working, independently testable software. Deterministic Python does the bulk; LLM subagents appear only at DateVerify / Dedup-adjudication / GapFill / X-newsworthiness / Synthesize, each followed by a deterministic verifier.

**Tech Stack:** Python 3, `pytest` (`unittest.TestCase` style), SQLite (`sqlite3` stdlib), `fastembed`/`sentence-transformers` (local multilingual Matryoshka embeddings), Claude Code headless (`claude -p`) on VPS/local cron, existing `configparser` config, GitHub Actions (demoted to Publisher).

**Spec:** `docs/superpowers/specs/2026-06-02-agent-native-hotspot-rewrite-design.md` (v1.3). This index doc locks the cross-cutting contract; each stage plan implements a slice of it.

---

## 0. Plan documents & dependency order

| # | File | Spec § | Depends on | Ships |
|---|---|---|---|---|
| 00 | `…-00-overview.md` (this file) | all | — | the locked contract below |
| 01 | `…-01-stage0-foundation.md` | A.1, C.3.1, B.4, E | 00 | `StoryStore` (SQLite) + schema fields + `run_journal` |
| 02 | `…-02-stage1-staleness.md` | B.2 Tier-0, B.3.1, B.5, B.5.1, G | 01 | `GateDate` + max-age hard gate + gravity-from-first_date + arXiv v1 read (NO agents) |
| 03 | `…-03-stage2-dedup.md` | C.1–C.4 | 01, 02 | `Embed` + `Dedup` (L0/L1/L2) + `NoveltyGate.resurface/resurge` replacing `apply_cross_day_penalty` + backfill job |
| 04 | `…-04-stage3-dateverify.md` | B.1–B.4.1 | 01, 02 | `DateVerify` Tier-1/2 subagent + permanent verdict cache + version-count polling |
| 05 | `…-05-stage4-reuse-gapfill.md` | D.1–D.4, E | 01, 04 | reuse-layer adapters + `GapFill` ⊇ assertion + `intentionally_dropped_stale_competitor` + 2nd-order alert |
| 06 | `…-06-stage5-x-channel.md` | A.3, X-coverage doc | 01 | twitterapi.io harvest adapter + `XNewsworthy` |
| 07 | `…-07-stage6-kernel-runtime.md` | A.1, A.2, E, F.2 | 01–06 | `Kernel` DAG + checkpoints + Synthesize agent + Resurgence rendering + VPS cron + Actions→Publisher |
| 08 | `…-08-paper-agent-filter.md` | H.1–H.3 | 01 (Store optional) | `PaperFilter` interface + `ApiScoreFilter`/`RuleFilter` wrap + `AgentFilter` + cascade |

**Execution order:** 01 → 02 → 03 ‖ 04 (03 and 04 are independent after 01/02, can run in parallel) → 05 → 06 → 07. 08 is independent of the hotspot chain and may run any time after 01 (it reuses Store only for optional reuse-signal lookups; if 01 not done, 08's reuse-signal step is a no-op stub).

**Stop-the-line rule:** Every stage's final task asserts the relevant §G invariants (listed per stage). A stage is not "done" until its golden-fixture/replay tests are green.

---

## 1. Module layout (new + touched files)

```
arxiv_assistant/
  hotspots/
    store.py            NEW  — StoryStore (SQLite single-writer)            [stage 0]
    date_verify.py      NEW  — Tier-0 pure + Tier-1/2 subagent + version poll [stage 1 tier-0, stage 3 rest]
    novelty.py          NEW  — resurface(S) / resurge(S) closed-form predicates [stage 2]
    embed.py            NEW  — multilingual Matryoshka embedding + cosine     [stage 2]
    dedup.py            NEW  — L0/L1 deterministic + L2 cross-day match        [stage 2]
    gapfill.py          NEW  — competitor diff gate + GapFill dispatch         [stage 4]
    kernel.py           NEW  — fixed-DAG orchestrator, single Store writer     [stage 6]
    pipeline.py         MOD  — strangler: stages extracted/called by kernel    [stages 1,2,6]
    story.py            MOD  — Story gets persistent-id/centroid/snapshot fields [stage 0]
    enrich.py           (unchanged)
  utils/hotspot/
    gate_date.py        NEW  — gate_date()/floor_to_utc_day() pure functions   [stage 1]
    hotspot_schema.py   MOD  — HotspotItem gets verified_first_date/provenance  [stage 0]
    hotspot_sources.py  MOD  — get_freshness_date reads verified_first_date     [stage 1]
    run_journal.py      NEW  — per-run JSONL journal + 2nd-order alert thresholds [stage 0, stage 4]
  apis/hotspot/
    hotspot_twitterapi.py NEW — twitterapi.io adapter                          [stage 5]
    reuse_*.py          NEW  — competitor-output adapters (HF/AINews/…)         [stage 4]
  filters/
    paper_filter.py     NEW  — PaperFilter protocol + ApiScoreFilter/RuleFilter/AgentFilter [stage 8]
tests/
  test_store.py, test_gate_date.py, test_novelty.py, test_dedup.py,
  test_date_verify.py, test_gapfill.py, test_kernel.py, test_paper_filter.py   NEW
scripts/
  backfill_story_store.py NEW — one-off historical seed w/ dedup-first        [stage 2]
```

**File-size discipline:** keep each new module < ~300 lines, one responsibility. `pipeline.py` (1839 lines) is strangled, not rewritten in place — stages extract pure functions the kernel calls.

---

## 2. LOCKED shared types & signatures

> Stage drafters MUST use these exact names/signatures. Do not invent variants. New optional fields default to `None`/empty for backward compatibility.

### 2.1 Existing types (do not break)
`HotspotItem` (`utils/hotspot/hotspot_schema.py`): `source_id, source_name, source_role, source_type, title, summary, url, canonical_url, published_at, tags, authors, metadata`. **Stage 0 adds** two optional fields:
```python
verified_first_date: str | None = None   # ISO8601; set by DateVerify. NEVER source-claimed.
provenance: str = ""                      # e.g. "native:hf_papers" | "reuse:agents-radar"
```
`EnrichedItem` (`hotspots/enrich.py`): wraps `.item: HotspotItem`, plus `.summary, .importance, .entities, .event_type, .same_event_as` (unchanged).

### 2.2 `Story` new persistent fields (stage 0, in `hotspots/story.py`)
```python
# added to @dataclass Story (all default-safe):
story_id: str                              # NOW persistent (assigned by Store, not recomputed SHA1)
first_seen: str | None = None              # ISO date, immutable once set
centroid: list[float] | None = None        # embedding; model_id-bound
centroid_model_id: str = ""
status: str = "NEW"                         # "NEW" | "ONGOING"
arxiv_versions: dict[str, int] = field(default_factory=dict)          # id -> version count (monotonic)
# surface snapshots (recorded by Store.record_surface):
last_surfaced: str | None = None
surfaced_verified_max: str | None = None   # DAY-granular gate_date
surfaced_entity_names: set[str] = field(default_factory=set)
surfaced_max_tier: int = 0
surfaced_arxiv_versions: dict[str, int] = field(default_factory=dict)
# resurgence (§C.4):
resurged_at: str | None = None             # first-ever resurge run-date (immutable)
surfaced_resurged_at: str | None = None    # last resurgence-lane surface run-date
```
> NOTE: the legacy module-level `_story_id(items)` SHA1 helper is RETIRED in stage 2; `group_into_stories` stops minting ids and the Store assigns persistent ids via `match_or_create`.

### 2.3 `StoryStore` API (stage 0, `hotspots/store.py`)
```python
class StoryStore:
    def __init__(self, db_path: Path): ...                      # opens/creates SQLite
    # identity / dedup
    def active_stories(self, window_days: int, as_of: date) -> list[Story]: ...
    def match_or_create(self, cluster_centroid: list[float], cluster: Story,
                        cosine_threshold: float, window_days: int, as_of: date
                        ) -> tuple[Story, bool]: ...            # returns (story, is_new)
    def upsert_evidence(self, story_id: str, items: list[EnrichedItem], added_at: str) -> None: ...
    def record_surface(self, story: Story, run_date: str, *, lane: str = "featured") -> None: ...
                                                                # writes surfaced_* snapshots; lane="resurgence" also sets surfaced_resurged_at
    # date verdict cache (permanent freeze)
    def get_verdict(self, content_hash: str) -> dict | None: ...   # {verified_first_date, confidence, evidence}
    def put_verdict(self, content_hash: str, verdict: dict) -> None: ...  # write-once; no-op if exists
    # version counts (monotonic, NOT in date_verdicts)
    def refresh_arxiv_versions(self, arxiv_id: str, fetched_count: int) -> None: ...  # new := max(old, fetched)
    # backfill (stage 2 dependency)
    def seed_first_seen(self, story: Story, first_seen: str) -> None: ...  # write-once; MUST NOT route through live match_or_create
    # audit snapshot
    def dump_text_snapshot(self, out_dir: Path) -> Path: ...    # MUST include date_verdicts table
    def load_text_snapshot(self, snapshot: Path) -> None: ...
```
**Evidence ledger (table `evidence`):** each row carries `story_id, canonical_url, source_id, source_tier:int, added_at:str(run-date)`. `Story` exposes two read helpers over its loaded ledger (used by `NoveltyGate`): `evidence_added_since(snapshot_date) -> list` and `evidence_before(snapshot_date) -> list`. A module-level `_open_story_store(output_root) -> StoryStore` helper centralizes DB-path construction for the kernel, backfill, and tests.

SQLite path: `out/hot/state/story_store.sqlite`. Text snapshot dir: `out/hot/state/snapshot/`. Tables: `stories`, `evidence`, `date_verdicts`, `versions`.

### 2.4 `GateDate` pure functions (stage 1, `utils/hotspot/gate_date.py`)
```python
def floor_to_utc_day(iso_ts: str | None) -> date | None: ...   # truncate to UTC calendar day
def gate_date(item: HotspotItem) -> date | None: ...           # floor_to_utc_day(min(credible_dates(item)))
# credible_dates = {verified_first_date} ∪ {authoritative whole-day anchors: arXiv announced / Crossref reg}
```
Used by every discrete gate (max-age, gravity, NoveltyGate T2, GapFill within_max_age). Pure → golden-fixture tested.

### 2.5 `DateVerify` (stage 1 tier-0; stage 3 tier-1/2 & polling, `hotspots/date_verify.py`)
```python
def verify(item: HotspotItem, store: StoryStore, *, will_be_featured: bool = False) -> dict: ...
   # returns {"verified_first_date": str, "confidence": float, "evidence": list[str]}
   # will_be_featured (default-safe, added stage 3) gates the Tier-2 deep-search escalation only.
   # tier-0 deterministic (arXiv v1 / Crossref / DOI), cache via store.get/put_verdict;
   # tier-1/2 dispatch a stateless subagent (Wayback CDX + published_time + earliest mention),
   # earliest-credible-date-wins; ALWAYS followed by deterministic verifier.
def poll_arxiv_versions(arxiv_ids: list[str]) -> dict[str, int]: ...
   # batched arXiv id_list query (<=100 ids/call, ~1 req/3s); independent cheap Tier-0 read.
```

### 2.6 `NoveltyGate` (stage 2, `hotspots/novelty.py`)
```python
def resurface(story: Story, *, gate_date_fn=gate_date) -> bool: ...   # §C.3.1 closed-form T1∨T2∨T3, zero LLM
def resurge(story: Story, *, max_age_days: int, run_date: date,
            min_competitors: int, cooldown_days: int,
            gate_date_fn=gate_date) -> bool: ...                       # §C.4 R1∨R2, zero LLM
```

### 2.7 `Embed`/`Dedup` (stage 2)
```python
# embed.py
EMBED_MODEL_ID = "..."                                   # pinned, stored on each centroid
def embed_text(text: str) -> list[float]: ...
def cosine(a: list[float], b: list[float]) -> float: ...
# dedup.py
def cluster_intraday(items: list[EnrichedItem]) -> list[Story]: ...     # L0 exact + L1 semantic (cosine>0.90)
def match_crossday(today: list[Story], store: StoryStore, *, cosine_threshold: float,
                   window_days: int, as_of: date) -> list[Story]: ...    # L2: assigns persistent ids, NEW/ONGOING
```
> **`cosine` canonical location:** `embed.cosine` (stage 2) is the single source of truth. Stage 0's `StoryStore` needs cosine for centroid matching before stage 2 lands, so it keeps a private `_cosine` helper; **stage 2's first task replaces that private helper with an import of `embed.cosine`** (tracked in §6 addenda). Centroids are unit-normalized and stored with `centroid_model_id`.

### 2.8 `GapFill` (stage 4, `hotspots/gapfill.py`)
```python
def eligible_competitor_items(competitor_items: list[HotspotItem], store: StoryStore,
                              *, max_age_days: int, as_of: date) -> tuple[list, list]: ...
   # returns (eligible, dropped_stale); eligible passes DateVerify AND within_max_age(gate_date)
def gapfill(our_coverage: set[str], eligible: list[HotspotItem]) -> list[HotspotItem]: ...
def assert_union_floor(our_coverage: set[str], eligible: list[HotspotItem]) -> None: ...  # ⊇ acceptance
```

### 2.9 `PaperFilter` modality (stage 8, `filters/paper_filter.py`)
```python
@dataclass
class FilterVerdict:
    keep: bool
    relevance: float        # same scale as existing GPT score
    novelty: float
    rationale: str
    evidence: list[str]
class PaperFilter(Protocol):
    def judge(self, paper, criteria: str) -> FilterVerdict: ...
class RuleFilter:      ...  # wraps filter_author h-index gate
class ApiScoreFilter:  ...  # wraps existing filter_gpt single-call scoring — ZERO behavior change
class AgentFilter:     ...  # Claude Code subagent; temp0 + structured + deterministic verifier
def cascade_filter(papers, criteria, config) -> list[FilterVerdict]: ...  # Rule→Api→(Agent on borderline)
```

### 2.10 `Kernel` (stage 6, `hotspots/kernel.py`)
```python
def run(output_root: Path, target_date: datetime, config, *, stage: str | None = None,
        force: bool = False) -> dict: ...   # fixed DAG; single Store writer; per-(date,stage) checkpoints
STAGES = ["harvest","date_verify","gravity_gate","embed","cluster",
          "storystore_match","gapfill","score","synthesize","render"]
```

### 2.11 Shared agent transport (`utils/agent_runner.py`) — used by stages 3, 6, 8
All three LLM-subagent touchpoints (DateVerify Tier-1/2, Synthesize, AgentFilter) share ONE headless transport so there is a single place to pin model/temperature and apply the deterministic-verifier discipline:
```python
def run_agent(prompt: str, *, schema: dict, model: str, tools: list[str] | None = None,
              timeout_s: int = 120) -> dict: ...
   # subprocess.run(["claude","-p",prompt,"--output-format","json","--model",model, ...]);
   # parses the result envelope; raises AgentError on non-zero/timeout (callers degrade deterministically).
   # temp 0 always; caller validates the returned dict against `schema` + its own verifier before use.
```
Stage 3 ships `run_agent` (first consumer); stages 6 and 8 import it. Tests `@patch("arxiv_assistant.utils.agent_runner.run_agent")` to replay fixtures — no network.

---

## 3. Config additions (`configs/config.ini` + template)

Add under `[HOTSPOTS]` (all with defaults; document in `configs/templates/config.template.ini`):
```ini
max_item_age_days = 14                  # B.5 hard gate (per-source-family override allowed)
cross_day_window_days = 14              # C.1 L2 rolling window
crossday_cosine_threshold = 0.90        # C.1/C.2 merge threshold
resurge_min_competitors = 3             # C.4 R2 default
resurge_cooldown_days = 7               # C.4 R2 cooldown
embed_model_id = <pinned id>            # G.7 centroid binding
[HOTSPOT_REUSE]
use_reuse_layer = true
reuse_sources = hf_daily,ainews,agents_radar,horizon,scholar_inbox
[HOTSPOT_RUNTIME]
runtime = local                         # local|actions
[PAPER_FILTER]
mode = cascade                          # api_only|agent_only|cascade
agent_borderline_low = 6.0
agent_borderline_high = 8.0
```
`use_x_official`/`use_x_paperpulse` are superseded by `use_twitterapi` (stage 5).

---

## 4. Test & determinism conventions (every stage obeys)

- **Runner:** `pytest tests/test_<module>.py -v`. Style: `unittest.TestCase` subclasses, `test_` methods, `tempfile.TemporaryDirectory()` for fs, `@patch` for network. Match existing `tests/test_hotspot_pipeline.py`.
- **Pure functions** (`gate_date`, `floor_to_utc_day`, `resurface`, `resurge`, `cosine`): **golden-fixture full coverage** — enumerate the truth table.
- **Agent steps** (`DateVerify` tier-1/2, `AgentFilter`, `Synthesize`): unit-tested via **record/replay** — store a captured JSON response fixture under `tests/fixtures/agent/`, patch the subagent call to replay it, assert the deterministic verifier + downstream behavior. Never hit the network in tests.
- **§G invariants as acceptance tests** (referenced per stage):
  - INV1 dates driving gates are `verified_first_date` only (never source-claimed).
  - INV2 discrete gates consume `gate_date` (day-granular); sub-day jitter cannot flip a gate.
  - INV3 `verified_first_date` write-once (cache freeze); `arxiv_versions` monotonic, not in `date_verdicts`.
  - INV4 RESURFACE/resurge are closed-form boolean over Store fields; zero LLM, zero URL-churn, zero sub-day.
  - INV5 reuse items pass the same DateVerify + max_age gate (inherit recall, not staleness).
  - INV6 every random agent is followed by a deterministic verifier; temp 0; pinned model/embedding ids.
- **Replay-diff gate:** stage 1+ adds `tests/test_replay_diff.py` — fix a historical raw-input fixture, assert stage outputs (story set + scores) are bit-stable across two runs (cache hit path).

---

## 5. Migration safety rails (apply across stages)

- One commit per task; conventional-commit messages; keep docs in sync (user preference).
- Backward-compat: new schema fields default-safe; old `out/hot/reports/*.json` still parse.
- **Backfill before dedup (stage 2 critical):** the 30-day history already contains the 6-day-duplicate bug; a naive backfill would mint 6 polluted `first_seen` anchors. `scripts/backfill_story_store.py` runs the SAME dedup over history first, then seeds one `first_seen` per real story.
- Renderers (`renderers/hotspot/*`, `build_multipage_site.py`) and i18n are NOT touched until stage 6 (only the Resurgence section is added there).
- Keep a deterministic Actions fallback path (agents off → reduced deterministic run) until stage 6 proves the VPS path.

---

## 6. Cross-stage reconciliation addenda (RESOLVED in consistency review)

These were surfaced while drafting the stage plans and are now BINDING — implementers follow these over any conflicting wording inside an individual stage plan:

1. **`StoryStore` surface (stage 0 must include):** beyond §2.3, stage 0 ships `seed_first_seen(story, first_seen)` (write-once, NOT via `match_or_create`), the `evidence` table columns `source_tier:int` + `added_at:str`, the `Story.evidence_added_since/evidence_before` ledger helpers, and the module-level `_open_story_store(output_root)`. Stage 2's backfill and `NoveltyGate` depend on all five.
2. **`cosine` ownership:** `embed.cosine` (stage 2) is canonical. Stage 0 keeps a private `StoryStore._cosine`; **stage 2 Task 1 replaces it with `from arxiv_assistant.hotspots.embed import cosine`.** Do not maintain two implementations past stage 2.
3. **`DateVerify.verify` signature:** `verify(item, store, *, will_be_featured=False)` — the keyword is default-safe; stage-1 callers pass nothing, stage-3 wires `will_be_featured` from the featured-candidate set to gate Tier-2 only.
4. **Shared agent transport:** stages 3/6/8 MUST call `utils/agent_runner.run_agent` (§2.11), not hand-rolled `subprocess` blocks. Stage 3 creates it; 6 and 8 import it. This is the single pin-point for model id + temperature 0 + the INV6 verifier handoff.
5. **Adapter calling convention:** registry-backed adapters (X, reuse) use the real positional-`seed_path` + keyword-`result_limit` shape, e.g. `fetch_hotspot_items(target_date, freshness_hours, seed_path, *, result_limit=80, ...)`, matching the existing pipeline fan-out — this is the concretization of §2's `fetch_hotspot_items(...)` shorthand, not a deviation.
6. **`run_journal` surface:** stage 0 ships `RunJournal` (append/flush + per-source counts + stage timings + empty `intentionally_dropped_stale_competitor`); stage 4 adds `record_dropped_stale_competitor` + `second_order_pollution_alerts` (reads the journal only).
7. **Backward-compat field access:** until stage 0 merges, later-stage code reads new `HotspotItem`/`Story` fields via `getattr(..., default)`; once stage 0 lands they are plain dataclass fields. Tests set them as attributes either way.
8. **TDD note (stage 0):** Task 3 ships the complete `StoryStore` module, so its later tasks (verdict-freeze, monotonic versions, record_surface) are "confirm-pass + lock-invariant" tests rather than red→green — acceptable because the invariants, not the line coverage, are the acceptance criteria.

**Spec→stage coverage (final check):** A→{01,07} · B→{02,04} · C→{03} · D→{05} · E→{01,05,07} · F→{all + this index} · G→{all, as INV acceptance tests} · H→{08}. No spec section is unassigned.
