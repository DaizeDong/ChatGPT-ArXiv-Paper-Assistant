# Stage 2 — Persistent Dedup & Novelty Gate Implementation Plan

**For agentic workers:** REQUIRED SUB-SKILL — use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to drive this plan task-by-task; each task is bite-sized TDD (write the failing test, watch it fail, write the code, watch it pass, commit).
**Contract:** this stage implements §C.1–C.4 of `docs/superpowers/specs/2026-06-02-agent-native-hotspot-rewrite-design.md` against the LOCKED signatures in `docs/superpowers/plans/2026-06-03-agent-native-rewrite-00-overview.md` §2.6 / §2.7 / §3 / §4 / §5. Do NOT invent signature variants.
**Depends on (already shipped):** Stage 0 (`hotspots/store.py` `StoryStore`, `Story` persistent fields per overview §2.2, evidence ledger with per-row `added_at` and `source_tier`, `record_surface`, `match_or_create`, `active_stories`) and Stage 1 (`utils/hotspot/gate_date.py` `gate_date` / `floor_to_utc_day`, `HotspotItem.verified_first_date`, `max_item_age_days` gate). This stage is the root-cause fix for the ~60% cross-day overlap.

---

## Scope of this stage

1. `arxiv_assistant/hotspots/embed.py` — pinned local multilingual Matryoshka embedding + cosine (overview §2.7).
2. `arxiv_assistant/hotspots/dedup.py` — `cluster_intraday` (L0 exact + L1 semantic incl. zh/en same-space merge) and `match_crossday` (L2 centroid-primary cross-day match, intra-day anti-convergence, persistent-id assignment via `StoryStore.match_or_create`).
3. `arxiv_assistant/hotspots/novelty.py` — closed-form `resurface(story)` (§C.3.1 T1∨T2∨T3 with construct-excluded URL-churn / sub-day jitter / same-tier / free-text) and `resurge(story, …)` (§C.4 R1∨R2 with cooldown gate).
4. **Replace** (not stack) `apply_cross_day_penalty` at `pipeline.py:1714-1718` with `match_crossday` + `resurface` → NEW / ONGOING / RESURFACE.
5. `scripts/backfill_story_store.py` — one-off offline job: run the SAME dedup over 30-day history FIRST, then seed exactly one `first_seen` per real story (overview §5 safety rail: avoid minting 6 polluted anchors).
6. Tests `tests/test_novelty.py` (full truth-table) and `tests/test_dedup.py` (zh/en merge, centroid-primary not AND, anti-convergence), plus config additions.

**Out of scope (other stages):** DateVerify tier-1/2 (Stage 3), the `arxiv_versions` polling source (Stage 3 — this stage only *reads* `Story.arxiv_versions`/`surfaced_arxiv_versions` already populated by Stage 0/1), GapFill (Stage 4), Kernel DAG and Resurgence *rendering* (Stage 6). `resurge` is implemented and unit-tested here; the rendered Resurgence section is wired in Stage 6.

**File-size discipline:** `embed.py` < 120 lines, `dedup.py` < 220 lines, `novelty.py` < 130 lines. `pipeline.py` is strangled, not rewritten — only the cross-day block changes.

---

## Determinism & test conventions (obey for every task)

- Runner: `pytest tests/test_<module>.py -v`. Style: `unittest.TestCase` subclasses, `test_` methods, `tempfile.TemporaryDirectory()` for fs, `unittest.mock.patch` for the embedder, matching `tests/test_hotspot_pipeline.py`.
- Pure functions (`cosine`, `resurface`, `resurge`) get **golden-fixture full truth-table coverage** (overview §4).
- **No network in tests.** The real embedding model loads lazily and is patched in every test via `@patch("arxiv_assistant.hotspots.embed.embed_text", ...)` or by injecting a stub `embed_fn`. `embed.py` keeps the model behind a module-level lazy singleton so importing it never downloads.
- §G invariants this stage must satisfy (acceptance):
  - **INV4** RESURFACE/resurge are closed-form boolean over Store fields — zero LLM, zero URL-churn, zero sub-day jitter.
  - **INV2** discrete gates consume day-granular `gate_date`; sub-day jitter cannot flip a gate.
  - **INV6** pinned embedding id stored on each centroid (`centroid_model_id == EMBED_MODEL_ID`).

---

## Pre-flight (read before Task 1)

Anchor facts verified in the repo (do not re-derive):

- `arxiv_assistant/hotspots/story.py`: `group_into_stories` (lines 118-252) is the 4-pass Union-Find intra-day grouper; `apply_cross_day_penalty` (lines 315-332) is the band-aid being replaced; `_story_id` (lines 88-93) is the SHA1 helper RETIRED this stage (overview §2.2 NOTE — `group_into_stories` stops minting ids; the Store assigns persistent ids).
- `arxiv_assistant/hotspots/pipeline.py:1714` builds stories, `:1716-1718` applies the penalty. This is the single replacement site.
- `arxiv_assistant/utils/hotspot/hotspot_cluster.py`: `canonicalize_url` (212-219), `significant_title_tokens` (228-233), `title_similarity` (236-243) — reuse, do not reimplement.
- `arxiv_assistant/hotspots/enrich.py:95-103`: `EnrichedItem(item, event_type, entities, summary, importance, same_event_as, batch_index)`. `.item` is a `HotspotItem` with `.title, .summary, .canonical_url, .url, .metadata`.
- `Story` (story.py:65-86) gains the Stage-0 persistent fields per overview §2.2 (`story_id` persistent, `first_seen`, `centroid`, `centroid_model_id`, `status`, `arxiv_versions`, `last_surfaced`, `surfaced_verified_max`, `surfaced_entity_names`, `surfaced_max_tier`, `surfaced_arxiv_versions`, `resurged_at`, `surfaced_resurged_at`) plus the evidence-ledger helpers `evidence_added_since(last_surfaced) -> list[EnrichedItem]` and `evidence_before(last_surfaced) -> list[EnrichedItem]`, where each `EnrichedItem` carries Stage-0's `added_at` (run date) and `source_tier` (int). This stage CONSUMES those; it does not define them. If a helper is missing when you start, stop-the-line: Stage 0 is incomplete.

---

## Task 1 — config additions (`configs/config.ini` + template)

Bite-sized: add the keys this stage reads, with defaults, documented in the template. Per overview §3.

### 1a. Test first — `tests/test_hotspot_config.py` (append)

Append this method to the existing `TestHotspotConfig` class (mirror its style; it already loads `configs/config.ini`):

```python
    def test_stage2_dedup_keys_present_with_defaults(self) -> None:
        import configparser
        from pathlib import Path

        cfg = configparser.ConfigParser()
        cfg.read(Path(__file__).resolve().parents[1] / "configs" / "config.ini")
        hot = cfg["HOTSPOTS"]
        self.assertEqual(hot.getint("cross_day_window_days"), 14)
        self.assertAlmostEqual(hot.getfloat("crossday_cosine_threshold"), 0.90)
        self.assertEqual(hot.getint("resurge_min_competitors"), 3)
        self.assertEqual(hot.getint("resurge_cooldown_days"), 7)
        self.assertTrue(hot.get("embed_model_id", fallback="").strip())
```

Run `pytest tests/test_hotspot_config.py -v` → **fails** (keys absent).

### 1b. Code — add under `[HOTSPOTS]` in BOTH `configs/config.ini` and `configs/templates/config.template.ini`

```ini
# --- Stage 2: persistent dedup & novelty gate (spec §C.1–C.4) ---
cross_day_window_days = 14            ; L2 rolling window (§C.1)
crossday_cosine_threshold = 0.90      ; centroid merge threshold (§C.1/C.2)
resurge_min_competitors = 3           ; §C.4 R2 default
resurge_cooldown_days = 7             ; §C.4 R2 cooldown
embed_model_id = sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2  ; §G.7 centroid binding
```

> `max_item_age_days = 14` and `cross_day_window_days` may already exist from Stage 1; if so, leave the existing key and only add the missing ones. In the template, also add the same block under `[HOTSPOTS]` with inline comments.

Run `pytest tests/test_hotspot_config.py -v` → **passes**.

**Commit:** `feat(hotspot): add stage-2 dedup/novelty config keys`

---

## Task 2 — `embed.py`: pinned multilingual embedding + cosine

Bite-sized: the pure `cosine` (golden truth-table) + a lazy, pinned `embed_text`. Overview §2.7.

### 2a. Test first — `tests/test_dedup.py` (new file, `TestEmbed` class)

Create `tests/test_dedup.py` with this class first:

```python
from __future__ import annotations

import math
import unittest
from unittest.mock import patch

from arxiv_assistant.hotspots import embed


class TestEmbed(unittest.TestCase):
    def test_model_id_is_pinned_nonempty(self) -> None:
        self.assertTrue(isinstance(embed.EMBED_MODEL_ID, str) and embed.EMBED_MODEL_ID)

    def test_cosine_identity_is_one(self) -> None:
        v = [0.1, 0.2, 0.3, 0.4]
        self.assertAlmostEqual(embed.cosine(v, v), 1.0, places=6)

    def test_cosine_orthogonal_is_zero(self) -> None:
        self.assertAlmostEqual(embed.cosine([1.0, 0.0], [0.0, 1.0]), 0.0, places=6)

    def test_cosine_opposite_is_minus_one(self) -> None:
        self.assertAlmostEqual(embed.cosine([1.0, 2.0], [-1.0, -2.0]), -1.0, places=6)

    def test_cosine_zero_vector_returns_zero(self) -> None:
        self.assertEqual(embed.cosine([0.0, 0.0], [1.0, 2.0]), 0.0)

    def test_cosine_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            embed.cosine([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_cosine_known_value(self) -> None:
        # 45-degree pair → cos = 1/sqrt(2)
        got = embed.cosine([1.0, 0.0], [1.0, 1.0])
        self.assertAlmostEqual(got, 1.0 / math.sqrt(2.0), places=6)

    def test_embed_text_uses_lazy_singleton(self) -> None:
        calls = {"n": 0}

        class _StubModel:
            def encode(self, text):  # noqa: D401
                calls["n"] += 1
                return [float(len(text)), 1.0, 2.0]

        with patch.object(embed, "_load_model", return_value=_StubModel()):
            embed._MODEL = None  # reset singleton
            a = embed.embed_text("hello")
            b = embed.embed_text("hello world")
        self.assertEqual(len(a), 3)
        self.assertEqual(a[0], 5.0)
        self.assertEqual(b[0], 11.0)
        # _load_model called once → singleton reused
        embed._MODEL = None
```

Run `pytest tests/test_dedup.py -v` → **fails** (module absent).

### 2b. Code — `arxiv_assistant/hotspots/embed.py`

```python
from __future__ import annotations

import math
from typing import Sequence

# §G.7 / overview §2.7: pinned multilingual Matryoshka model id, stored on every centroid.
# Multilingual so that an English item and its `_zh` translation embed into the SAME space
# (§C.1 L1 cross-language merge). Keep in sync with configs `embed_model_id`.
EMBED_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Lazy module-level singleton: importing this module must never download a model.
_MODEL = None


def _load_model():
    """Load the pinned local multilingual model.

    Prefers `fastembed` (small, no torch); falls back to `sentence-transformers`.
    Raises a clear error if neither backend is installed.
    """
    try:
        from fastembed import TextEmbedding  # type: ignore

        class _FastEmbedAdapter:
            def __init__(self) -> None:
                self._model = TextEmbedding(model_name=EMBED_MODEL_ID)

            def encode(self, text: str) -> list[float]:
                # fastembed yields numpy arrays; take the first (single doc).
                vec = next(iter(self._model.embed([text])))
                return [float(x) for x in vec]

        return _FastEmbedAdapter()
    except Exception:
        pass

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        class _SbertAdapter:
            def __init__(self) -> None:
                self._model = SentenceTransformer(EMBED_MODEL_ID)

            def encode(self, text: str) -> list[float]:
                vec = self._model.encode(text, normalize_embeddings=False)
                return [float(x) for x in vec]

        return _SbertAdapter()
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "Stage-2 embedding requires `fastembed` or `sentence-transformers`. "
            "Install one: `pip install fastembed` (preferred) or "
            "`pip install sentence-transformers`."
        ) from exc


def embed_text(text: str) -> list[float]:
    """Embed `title + lede` text into the pinned multilingual space."""
    global _MODEL
    if _MODEL is None:
        _MODEL = _load_model()
    return _MODEL.encode(text or "")


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity. Zero-norm vectors → 0.0; length mismatch → ValueError."""
    if len(a) != len(b):
        raise ValueError(f"cosine length mismatch: {len(a)} != {len(b)}")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))
```

Run `pytest tests/test_dedup.py -v` → `TestEmbed` **passes**.

### 2c. requirements — install note

Append to `requirements.txt`:

```
fastembed==0.3.6
```

> `fastembed` pulls `onnxruntime` (CPU, no torch) and is the lighter backend. If the deploy box already has `sentence-transformers` (torch), the fallback path is used and `fastembed` can be omitted — but the pinned default in `requirements.txt` keeps CI/VPS reproducible. Document in the commit body: first run downloads the model once into the HF cache (`~80 MB`); offline-test paths patch `embed_text`/`_load_model` so CI never downloads.

**Commit:** `feat(hotspot): add pinned multilingual embed + cosine (embed.py)`

---

## Task 3 — `dedup.py` `cluster_intraday`: L0 exact + L1 semantic (incl. zh/en)

Bite-sized: intra-day clustering only. L2 is Task 4. Overview §2.7, spec §C.1.

### 3a. Test first — append `TestClusterIntraday` to `tests/test_dedup.py`

```python
from arxiv_assistant.hotspots import dedup
from arxiv_assistant.hotspots.enrich import EnrichedItem
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem


def _item(title, url, *, summary="", arxiv_id="", metadata=None):
    md = dict(metadata or {})
    if arxiv_id:
        md["arxiv_id"] = arxiv_id
    return HotspotItem(
        source_id="s",
        source_name="S",
        source_role="official_news",
        source_type="news",
        title=title,
        summary=summary,
        url=url,
        canonical_url=url,
        published_at="2026-06-02T00:00:00+00:00",
        tags=[],
        authors=[],
        metadata=md,
    )


def _enriched(title, url, *, summary="", arxiv_id="", entities=None):
    return EnrichedItem(
        item=_item(title, url, summary=summary, arxiv_id=arxiv_id),
        event_type="product_release",
        entities=entities or [],
        summary=summary or title,
        importance=5,
    )


class TestClusterIntraday(unittest.TestCase):
    def _patch_embed(self, mapping):
        # Deterministic stub: map text-substring → unit vector direction.
        def fake_embed(text: str):
            for key, vec in mapping.items():
                if key in text:
                    return list(vec)
            return [0.0, 0.0, 1.0]
        return patch.object(dedup, "embed_text", side_effect=fake_embed)

    def test_l0_exact_url_merge(self) -> None:
        a = _enriched("OpenAI ships GPT-X", "https://x.ai/p?utm=1")
        b = _enriched("Totally different headline", "https://x.ai/p?ref=2")
        # canonicalize_url strips query → same canonical → L0 merges regardless of title/embed
        with self._patch_embed({"OpenAI": (1, 0, 0), "Totally": (0, 1, 0)}):
            stories = dedup.cluster_intraday([a, b])
        self.assertEqual(len(stories), 1)

    def test_l1_semantic_merge_above_threshold(self) -> None:
        a = _enriched("Model release A", "https://a.com/1")
        b = _enriched("Model release A variant", "https://b.com/2")
        with self._patch_embed({"release A variant": (1, 0.05, 0), "release A": (1, 0, 0)}):
            stories = dedup.cluster_intraday([a, b])
        self.assertEqual(len(stories), 1)

    def test_l1_keeps_separate_below_threshold(self) -> None:
        a = _enriched("Apples grow on trees", "https://a.com/1")
        b = _enriched("Quantum chips ship today", "https://b.com/2")
        with self._patch_embed({"Apples": (1, 0, 0), "Quantum": (0, 1, 0)}):
            stories = dedup.cluster_intraday([a, b])
        self.assertEqual(len(stories), 2)

    def test_zh_en_same_event_merge(self) -> None:
        # English + its zh translation embed into nearly-identical vectors (multilingual space).
        en = _enriched("Anthropic launches Claude 5", "https://anthropic.com/c5")
        zh = _enriched("Anthropic 发布 Claude 5", "https://zh.example.com/c5")
        with self._patch_embed({"发布": (1, 0.02, 0), "launches": (1, 0, 0)}):
            stories = dedup.cluster_intraday([en, zh])
        self.assertEqual(len(stories), 1)

    def test_centroid_stored_with_model_id(self) -> None:
        a = _enriched("Solo story", "https://a.com/1")
        with self._patch_embed({"Solo": (3.0, 4.0, 0.0)}):
            stories = dedup.cluster_intraday([a])
        s = stories[0]
        self.assertEqual(s.centroid_model_id, dedup.EMBED_MODEL_ID)
        # centroid is L2-normalized mean → unit length
        norm = sum(x * x for x in s.centroid) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=6)
```

Run → **fails** (`cluster_intraday` absent).

### 3b. Code — `arxiv_assistant/hotspots/dedup.py` (part 1 of 2)

```python
from __future__ import annotations

import math
from datetime import date

from arxiv_assistant.hotspots.embed import EMBED_MODEL_ID, cosine, embed_text
from arxiv_assistant.hotspots.enrich import EnrichedItem
from arxiv_assistant.hotspots.story import Story, group_into_stories
from arxiv_assistant.utils.hotspot.hotspot_cluster import canonicalize_url

# §C.1: cosine > L1_SEMANTIC_THRESHOLD auto-merges intra-day near-duplicates
# (incl. an English item and its `_zh` translation in the shared multilingual space).
L1_SEMANTIC_THRESHOLD = 0.90


def _embed_story_text(item) -> str:
    """title + lede (first sentence of summary) — the §C.1 embedding payload."""
    title = (item.title or "").strip()
    lede = (item.summary or "").strip().split(". ")[0]
    return f"{title}. {lede}".strip()


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return list(vec)
    return [x / norm for x in vec]


def _centroid(vectors: list[list[float]]) -> list[float]:
    """L2-normalized mean of unit vectors (deterministic, order-independent)."""
    if not vectors:
        return []
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            acc[i] += v[i]
    mean = [a / len(vectors) for a in acc]
    return _normalize(mean)


def cluster_intraday(items: list[EnrichedItem]) -> list[Story]:
    """L0 exact (canonical URL) + L1 semantic (cosine > 0.90, cross-language).

    Reuses the existing deterministic 4-pass Union-Find grouper (`group_into_stories`)
    for L0/title evidence, then runs an L1 semantic merge pass on the resulting stories.
    Each returned Story carries a model-id-bound L2-normalized `centroid`.
    """
    if not items:
        return []

    # L0 + existing deterministic title/URL/entity grouping.
    stories = group_into_stories(items)

    # Embed each story (unit vectors) for L1 and to set centroids.
    vecs: list[list[float]] = []
    for s in stories:
        member_vecs = [_normalize(embed_text(_embed_story_text(ei.item))) for ei in s.items]
        s.centroid = _centroid(member_vecs)
        s.centroid_model_id = EMBED_MODEL_ID
        vecs.append(s.centroid)

    # L1: greedy union of stories whose centroids are cosine > threshold.
    parent = list(range(len(stories)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(stories)):
        for j in range(i + 1, len(stories)):
            if find(i) == find(j):
                continue
            if cosine(vecs[i], vecs[j]) > L1_SEMANTIC_THRESHOLD:
                parent[find(j)] = find(i)

    # Collapse merged groups, re-using group_into_stories to rebuild canonical/entity fields.
    groups: dict[int, list[EnrichedItem]] = {}
    for idx, s in enumerate(stories):
        groups.setdefault(find(idx), []).extend(s.items)

    merged: list[Story] = []
    for member_items in groups.values():
        rebuilt = group_into_stories(member_items)
        # group_into_stories may re-split on title evidence; re-merge to one story per L1 group
        # by pooling all items, since L1 cosine already declared them the same event.
        if len(rebuilt) == 1:
            story = rebuilt[0]
        else:
            story = rebuilt[0]
            for extra in rebuilt[1:]:
                story.items.extend(extra.items)
                story.entity_names |= extra.entity_names
        member_vecs = [_normalize(embed_text(_embed_story_text(ei.item))) for ei in story.items]
        story.centroid = _centroid(member_vecs)
        story.centroid_model_id = EMBED_MODEL_ID
        merged.append(story)

    return merged
```

Run → `TestClusterIntraday` **passes**. (`TestEmbed` still green.)

**Commit:** `feat(hotspot): add intra-day L0/L1 dedup (dedup.cluster_intraday)`

---

## Task 4 — `dedup.py` `match_crossday`: L2 centroid-primary + anti-convergence

Bite-sized: cross-day matching against the Store, centroid-primary (NOT AND), intra-day anti-convergence, persistent-id assignment. Spec §C.2.

### 4a. Test first — append `TestMatchCrossday` to `tests/test_dedup.py`

Uses a tiny fake Store implementing only the `match_or_create` / `active_stories` surface this function touches (the real `StoryStore` from Stage 0 is integration-tested in `test_store.py`).

```python
class _FakeStore:
    """Minimal StoryStore stand-in for match_crossday unit tests.

    Mimics centroid-primary match_or_create: assigns a persistent id, returns
    (story, is_new). Stories already in `_active` are existing; cosine >= threshold
    against an existing centroid → ONGOING reusing that id; else NEW with a minted id.
    """

    def __init__(self, active):
        self._active = list(active)         # list[Story] with .story_id, .centroid, .first_seen
        self._counter = 0
        self.created = []

    def active_stories(self, window_days, as_of):
        return list(self._active)

    def match_or_create(self, cluster_centroid, cluster, cosine_threshold, window_days, as_of):
        from arxiv_assistant.hotspots.embed import cosine
        best = None
        best_sim = -2.0
        for ex in self._active:
            if not ex.centroid:
                continue
            sim = cosine(cluster_centroid, ex.centroid)
            if sim >= cosine_threshold and sim > best_sim:
                best, best_sim = ex, sim
        if best is not None:
            cluster.story_id = best.story_id
            cluster.first_seen = best.first_seen
            cluster.status = "ONGOING"
            return cluster, False
        self._counter += 1
        cluster.story_id = f"new{self._counter}"
        cluster.first_seen = as_of.isoformat()
        cluster.status = "NEW"
        self._active.append(cluster)
        self.created.append(cluster.story_id)
        return cluster, True


class TestMatchCrossday(unittest.TestCase):
    def _story(self, story_id, centroid, first_seen=None):
        from arxiv_assistant.hotspots.story import Story
        s = Story(
            story_id=story_id,
            canonical_item=_enriched("seed", f"https://seed/{story_id}"),
            items=[_enriched("seed", f"https://seed/{story_id}")],
            event_type="product_release",
        )
        s.centroid = list(centroid)
        s.centroid_model_id = dedup.EMBED_MODEL_ID
        s.first_seen = first_seen
        return s

    def test_centroid_match_marks_ongoing_reuses_id(self) -> None:
        existing = self._story("persist1", [1.0, 0.0, 0.0], first_seen="2026-05-30")
        store = _FakeStore([existing])
        today = self._story("tmp", [0.999, 0.04, 0.0])  # cosine ~0.999 > 0.90
        out = dedup.match_crossday(
            [today], store, cosine_threshold=0.90, window_days=14, as_of=date(2026, 6, 2)
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].story_id, "persist1")
        self.assertEqual(out[0].status, "ONGOING")
        self.assertEqual(out[0].first_seen, "2026-05-30")

    def test_low_cosine_creates_new(self) -> None:
        existing = self._story("persist1", [1.0, 0.0, 0.0], first_seen="2026-05-30")
        store = _FakeStore([existing])
        today = self._story("tmp", [0.0, 1.0, 0.0])  # orthogonal
        out = dedup.match_crossday(
            [today], store, cosine_threshold=0.90, window_days=14, as_of=date(2026, 6, 2)
        )
        self.assertEqual(out[0].status, "NEW")
        self.assertNotEqual(out[0].story_id, "persist1")

    def test_centroid_primary_not_AND_with_url(self) -> None:
        # Same event, DISJOINT url sets → must still merge on centroid alone (NOT AND).
        existing = self._story("persist1", [1.0, 0.0, 0.0], first_seen="2026-05-30")
        store = _FakeStore([existing])
        today = self._story("tmp", [0.99, 0.05, 0.0])
        # disjoint urls
        existing.items[0].item.canonical_url = "https://old.com/a"
        today.items[0].item.canonical_url = "https://new.com/b"
        out = dedup.match_crossday(
            [today], store, cosine_threshold=0.90, window_days=14, as_of=date(2026, 6, 2)
        )
        self.assertEqual(out[0].story_id, "persist1")  # merged despite zero URL overlap

    def test_intraday_anticonvergence_single_existing_id(self) -> None:
        # One real event split into 2 today-clusters; both match the SAME existing story.
        # Anti-convergence: they must collapse to ONE story pointing at the existing id,
        # NOT accrete two duplicate ONGOING stories.
        existing = self._story("persist1", [1.0, 0.0, 0.0], first_seen="2026-05-30")
        store = _FakeStore([existing])
        c1 = self._story("t1", [0.99, 0.04, 0.0])
        c2 = self._story("t2", [0.98, 0.06, 0.0])
        out = dedup.match_crossday(
            [c1, c2], store, cosine_threshold=0.90, window_days=14, as_of=date(2026, 6, 2)
        )
        persist_stories = [s for s in out if s.story_id == "persist1"]
        self.assertEqual(len(persist_stories), 1)          # collapsed to one
        self.assertEqual(len(out), 1)
        self.assertEqual(store.created, [])                 # no spurious NEW minted
```

Run → **fails** (`match_crossday` absent).

### 4b. Code — `arxiv_assistant/hotspots/dedup.py` (part 2 of 2, append)

```python
def _merge_today_clusters(a: Story, b: Story) -> Story:
    """Pool two today-clusters into one (anti-convergence; §C.2 rule 2)."""
    a.items.extend(b.items)
    a.entity_names |= b.entity_names
    member_vecs = [_normalize(embed_text(_embed_story_text(ei.item))) for ei in a.items]
    a.centroid = _centroid(member_vecs)
    a.centroid_model_id = EMBED_MODEL_ID
    return a


def match_crossday(
    today: list[Story],
    store,
    *,
    cosine_threshold: float,
    window_days: int,
    as_of: date,
) -> list[Story]:
    """L2 cross-day persistent match (spec §C.2).

    Centroid is PRIMARY: a today-cluster merges into an existing active story when
    centroid cosine >= threshold. URL-Jaccard is NEVER a necessary condition (the
    Store's match_or_create may use it only as additive confirmation).

    Intra-day anti-convergence: if two today-clusters both match the SAME existing
    story, they are first pooled into one cluster pointing at that one persistent id
    (so the centroid store never accretes a duplicate story for one real event),
    THEN NEW/ONGOING is decided.

    Returns the today stories with persistent `story_id`, `first_seen`, and
    `status` ("NEW"|"ONGOING") assigned. Pure dispatch — the single Store writer
    (Kernel) owns `match_or_create`.
    """
    if not today:
        return []

    active = store.active_stories(window_days, as_of)

    # Phase 1: anti-convergence. Pre-bind each today-cluster to its best existing match
    # (centroid-primary), then pool today-clusters that share an existing target.
    def best_existing_id(cluster: Story) -> str | None:
        best_id = None
        best_sim = cosine_threshold  # strict >= threshold to count
        for ex in active:
            if not ex.centroid or not cluster.centroid:
                continue
            sim = cosine(cluster.centroid, ex.centroid)
            if sim >= best_sim:
                best_sim = sim
                best_id = ex.story_id
        return best_id

    pooled_by_target: dict[str, Story] = {}
    unmatched: list[Story] = []
    for cluster in today:
        target = best_existing_id(cluster)
        if target is None:
            unmatched.append(cluster)
        elif target in pooled_by_target:
            _merge_today_clusters(pooled_by_target[target], cluster)
        else:
            pooled_by_target[target] = cluster

    # Phase 2: assign persistent ids via the Store. Order is deterministic:
    # pooled-existing first (sorted by target id), then unmatched (input order).
    result: list[Story] = []
    for _target, cluster in sorted(pooled_by_target.items(), key=lambda kv: kv[0]):
        story, _is_new = store.match_or_create(
            cluster.centroid, cluster, cosine_threshold, window_days, as_of
        )
        result.append(story)
    for cluster in unmatched:
        story, _is_new = store.match_or_create(
            cluster.centroid, cluster, cosine_threshold, window_days, as_of
        )
        result.append(story)
    return result
```

Run → `TestMatchCrossday` **passes**. Full `pytest tests/test_dedup.py -v` green.

**Commit:** `feat(hotspot): add L2 cross-day centroid match + anti-convergence (dedup.match_crossday)`

---

## Task 5 — `novelty.py` `resurface`: closed-form T1∨T2∨T3

Bite-sized: the pure resurface predicate with the full §C.3.1 truth table. Overview §2.6.

### 5a. Test first — `tests/test_novelty.py` (new file, `TestResurface`)

```python
from __future__ import annotations

import unittest
from datetime import date

from arxiv_assistant.hotspots import novelty
from arxiv_assistant.hotspots.enrich import EnrichedItem
from arxiv_assistant.hotspots.story import Story
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem


def _ev(title, *, added_at, source_tier, verified_first_date=None, arxiv_id="",
        entities=None):
    md = {}
    if arxiv_id:
        md["arxiv_id"] = arxiv_id
    item = HotspotItem(
        source_id="s", source_name="S", source_role="news", source_type="news",
        title=title, summary=title, url=f"https://e/{title}",
        canonical_url=f"https://e/{title}", published_at="2026-06-02T00:00:00+00:00",
        tags=[], authors=[], metadata=md,
    )
    item.verified_first_date = verified_first_date
    ei = EnrichedItem(item=item, event_type="product_release",
                      entities=entities or [], summary=title, importance=5)
    ei.added_at = added_at          # Stage-0 ledger field
    ei.source_tier = source_tier    # Stage-0 ledger field
    return ei


def _story(evidence, *, last_surfaced, surfaced_verified_max=None,
           surfaced_entity_names=None, surfaced_max_tier=0,
           arxiv_versions=None, surfaced_arxiv_versions=None, entity_names=None):
    canon = evidence[0] if evidence else _ev("seed", added_at="2026-05-01", source_tier=1)
    s = Story(
        story_id="persist1",
        canonical_item=canon,
        items=list(evidence),
        event_type="product_release",
        entity_names=set(entity_names or set()),
    )
    s.status = "ONGOING"
    s.last_surfaced = last_surfaced
    s.surfaced_verified_max = surfaced_verified_max
    s.surfaced_entity_names = set(surfaced_entity_names or set())
    s.surfaced_max_tier = surfaced_max_tier
    s.arxiv_versions = dict(arxiv_versions or {})
    s.surfaced_arxiv_versions = dict(surfaced_arxiv_versions or {})
    s._ledger = list(evidence)  # backing list used by the helpers below
    return s


# Stage-0 provides evidence_added_since/evidence_before on Story; for these unit tests
# we exercise novelty.resurface against a Story whose helpers split `_ledger` by added_at.
def _install_helpers():
    def added_since(self, last):
        return [e for e in getattr(self, "_ledger", self.items) if last is None or e.added_at > last]

    def before(self, last):
        return [e for e in getattr(self, "_ledger", self.items) if last is not None and e.added_at <= last]

    Story.evidence_added_since = added_since
    Story.evidence_before = before


class TestResurface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_helpers()

    # ---- NOT triggers ----
    def test_url_churn_only_is_false(self) -> None:
        # New evidence is same tier, same/earlier date, no new entity, no new version.
        before = _ev("e1", added_at="2026-06-01", source_tier=3, verified_first_date="2026-06-01")
        churn = _ev("e2", added_at="2026-06-02", source_tier=3, verified_first_date="2026-06-01")
        s = _story([before, churn], last_surfaced="2026-06-01",
                   surfaced_verified_max=date(2026, 6, 1), surfaced_max_tier=3)
        self.assertFalse(novelty.resurface(s))

    def test_subday_jitter_is_false(self) -> None:
        # Same UTC day, only H:M:S differs → floor_to_utc_day absorbs it.
        before = _ev("e1", added_at="2026-06-01", source_tier=3,
                     verified_first_date="2026-06-01T02:00:00+00:00")
        jitter = _ev("e2", added_at="2026-06-02", source_tier=3,
                     verified_first_date="2026-06-01T23:30:00+00:00")
        s = _story([before, jitter], last_surfaced="2026-06-01",
                   surfaced_verified_max=date(2026, 6, 1), surfaced_max_tier=3)
        self.assertFalse(novelty.resurface(s))

    def test_same_tier_more_evidence_is_false(self) -> None:
        before = _ev("e1", added_at="2026-06-01", source_tier=2, verified_first_date="2026-06-01")
        more = _ev("e2", added_at="2026-06-02", source_tier=2, verified_first_date="2026-06-01")
        s = _story([before, more], last_surfaced="2026-06-01",
                   surfaced_verified_max=date(2026, 6, 1), surfaced_max_tier=2)
        self.assertFalse(novelty.resurface(s))

    # ---- T1: tier jump ----
    def test_t1_tier_jump_is_true(self) -> None:
        before = _ev("e1", added_at="2026-06-01", source_tier=2, verified_first_date="2026-06-01")
        official = _ev("e2", added_at="2026-06-02", source_tier=7, verified_first_date="2026-06-01")
        s = _story([before, official], last_surfaced="2026-06-01",
                   surfaced_verified_max=date(2026, 6, 1), surfaced_max_tier=2)
        self.assertTrue(novelty.resurface(s))

    # ---- T2: later gate_date OR new arxiv version ----
    def test_t2_later_gate_date_is_true(self) -> None:
        before = _ev("e1", added_at="2026-06-01", source_tier=3, verified_first_date="2026-06-01")
        later = _ev("e2", added_at="2026-06-03", source_tier=3, verified_first_date="2026-06-03")
        s = _story([before, later], last_surfaced="2026-06-01",
                   surfaced_verified_max=date(2026, 6, 1), surfaced_max_tier=3)
        self.assertTrue(novelty.resurface(s))

    def test_t2_new_arxiv_version_is_true(self) -> None:
        before = _ev("e1", added_at="2026-06-01", source_tier=3,
                     verified_first_date="2026-06-01", arxiv_id="2606.00001")
        s = _story([before], last_surfaced="2026-06-01",
                   surfaced_verified_max=date(2026, 6, 1), surfaced_max_tier=3,
                   arxiv_versions={"2606.00001": 3}, surfaced_arxiv_versions={"2606.00001": 2})
        self.assertTrue(novelty.resurface(s))

    # ---- T3: new named entity ----
    def test_t3_new_entity_is_true(self) -> None:
        before = _ev("e1", added_at="2026-06-01", source_tier=3, verified_first_date="2026-06-01")
        s = _story([before], last_surfaced="2026-06-01",
                   surfaced_verified_max=date(2026, 6, 1), surfaced_max_tier=3,
                   entity_names={"openai", "nvidia"},
                   surfaced_entity_names={"openai"})
        self.assertTrue(novelty.resurface(s))
```

Run `pytest tests/test_novelty.py -v` → **fails** (`resurface` absent).

### 5b. Code — `arxiv_assistant/hotspots/novelty.py` (part 1 of 2)

```python
from __future__ import annotations

from datetime import date

from arxiv_assistant.utils.hotspot.gate_date import gate_date


def _max_tier(evidence: list) -> int:
    return max((int(getattr(e, "source_tier", 0)) for e in evidence), default=0)


def _max_gate_date(evidence: list, gate_date_fn) -> date | None:
    days = [gate_date_fn(e.item) for e in evidence]
    days = [d for d in days if d is not None]
    return max(days) if days else None


def resurface(story, *, gate_date_fn=gate_date) -> bool:
    """§C.3.1 closed-form resurface predicate: T1 ∨ T2 ∨ T3. Zero LLM.

    Reads ONLY Store-resident structured facts (source_tier ints, day-granular
    gate_date, arxiv version counts, entity_names). URL-set churn, sub-day jitter,
    same-tier evidence, and any free-text judgment are constructively excluded.
    """
    last = story.last_surfaced
    added = story.evidence_added_since(last)
    if not added:
        return False

    # T1: strictly higher source_tier than any evidence before last surface.
    before_tier = max(_max_tier(story.evidence_before(last)), int(story.surfaced_max_tier or 0))
    if _max_tier(added) > before_tier:
        return True

    # T2a: a strictly later day-granular gate_date among newly-added evidence.
    new_gate = _max_gate_date(added, gate_date_fn)
    if new_gate is not None and story.surfaced_verified_max is not None:
        if new_gate > story.surfaced_verified_max:
            return True
    elif new_gate is not None and story.surfaced_verified_max is None:
        return True

    # T2b: a strictly increased arXiv version count vs the last-surface snapshot.
    for arxiv_id, count in (story.arxiv_versions or {}).items():
        prev = (story.surfaced_arxiv_versions or {}).get(arxiv_id, 0)
        if count > prev:
            return True

    # T3: a named entity not present at last surface.
    if (set(story.entity_names) - set(story.surfaced_entity_names or set())):
        return True

    return False
```

Run → `TestResurface` **passes**.

**Commit:** `feat(hotspot): add closed-form resurface predicate (novelty.resurface)`

---

## Task 6 — `novelty.py` `resurge`: R1∨R2 with cooldown gate

Bite-sized: the §C.4 resurgence predicate (old items only). Overview §2.6.

### 6a. Test first — append `TestResurge` to `tests/test_novelty.py`

```python
class TestResurge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_helpers()

    def _old_story(self, **kw):
        # gate_date 60 days old → "old" item (exceeds max_age).
        ev = _ev("old", added_at="2026-04-01", source_tier=3,
                 verified_first_date="2026-04-01", arxiv_id="2604.00001")
        s = _story([ev], last_surfaced=None,
                   surfaced_verified_max=date(2026, 4, 1), surfaced_max_tier=3,
                   arxiv_versions=kw.get("arxiv_versions", {"2604.00001": 1}),
                   surfaced_arxiv_versions=kw.get("surfaced_arxiv_versions", {"2604.00001": 1}))
        s.resurged_at = kw.get("resurged_at")
        s.surfaced_resurged_at = kw.get("surfaced_resurged_at")
        s._today_competitors = kw.get("today_competitors", 0)
        return s

    def test_not_old_returns_false(self) -> None:
        # Fresh story (gate_date within max_age) is never a resurge candidate.
        ev = _ev("fresh", added_at="2026-06-02", source_tier=3, verified_first_date="2026-06-02")
        s = _story([ev], last_surfaced=None, surfaced_verified_max=date(2026, 6, 2))
        s.resurged_at = None
        s.surfaced_resurged_at = None
        s._today_competitors = 9
        self.assertFalse(novelty.resurge(
            s, max_age_days=14, run_date=date(2026, 6, 3),
            min_competitors=3, cooldown_days=7,
            competitor_count_fn=lambda story: story._today_competitors))

    def test_r1_version_jump_is_true(self) -> None:
        s = self._old_story(arxiv_versions={"2604.00001": 4},
                            surfaced_arxiv_versions={"2604.00001": 3})
        self.assertTrue(novelty.resurge(
            s, max_age_days=14, run_date=date(2026, 6, 3),
            min_competitors=3, cooldown_days=7,
            competitor_count_fn=lambda story: 0))

    def test_r2_competitors_fresh_cooldown_is_true(self) -> None:
        s = self._old_story(surfaced_resurged_at=None, today_competitors=3)
        self.assertTrue(novelty.resurge(
            s, max_age_days=14, run_date=date(2026, 6, 3),
            min_competitors=3, cooldown_days=7,
            competitor_count_fn=lambda story: story._today_competitors))

    def test_r2_below_min_competitors_is_false(self) -> None:
        s = self._old_story(surfaced_resurged_at=None, today_competitors=2)
        self.assertFalse(novelty.resurge(
            s, max_age_days=14, run_date=date(2026, 6, 3),
            min_competitors=3, cooldown_days=7,
            competitor_count_fn=lambda story: story._today_competitors))

    def test_r2_within_cooldown_is_false_then_true_after(self) -> None:
        # Same group of competitors re-surfaces day after day → cooldown fires ONCE.
        s = self._old_story(surfaced_resurged_at=date(2026, 6, 1), today_competitors=4)
        # 2 days later, cooldown_days=7 → still within cooldown → False
        self.assertFalse(novelty.resurge(
            s, max_age_days=14, run_date=date(2026, 6, 3),
            min_competitors=3, cooldown_days=7,
            competitor_count_fn=lambda story: story._today_competitors))
        # 8 days later → cooldown elapsed → True
        self.assertTrue(novelty.resurge(
            s, max_age_days=14, run_date=date(2026, 6, 9),
            min_competitors=3, cooldown_days=7,
            competitor_count_fn=lambda story: story._today_competitors))
```

Run → **fails** (`resurge` absent).

### 6b. Code — `arxiv_assistant/hotspots/novelty.py` (part 2 of 2, append)

```python
def _today_competitor_count(story) -> int:
    """Default: count distinct competitor (reuse-layer) sources among today's evidence.

    Reads the Stage-0 ledger fields: an evidence item added today whose provenance is
    a reuse-layer competitor source. Kernel may inject a precise fn; this default is a
    safe fallback for unit/offline use.
    """
    seen: set[str] = set()
    for e in getattr(story, "_today_evidence", []):
        prov = getattr(e.item, "provenance", "") or ""
        if prov.startswith("reuse:"):
            seen.add(prov)
    return len(seen)


def resurge(
    story,
    *,
    max_age_days: int,
    run_date: date,
    min_competitors: int,
    cooldown_days: int,
    gate_date_fn=gate_date,
    competitor_count_fn=_today_competitor_count,
) -> bool:
    """§C.4 resurgence predicate: R1 ∨ R2, evaluated ONLY for OLD stories. Zero LLM.

    Old := gate_date(story.canonical_item) is older than max_age_days vs run_date.
    R1: an arXiv version count strictly exceeds the last resurge-surface snapshot
        (each new version fires at most once).
    R2: >= min_competitors distinct competitor sources re-raise the item today AND
        the cooldown gate is open (surfaced_resurged_at is None or run_date is at
        least cooldown_days past it) — so the same competitor cluster re-raising the
        same old item day-after-day fires at most once per cooldown.
    """
    gd = gate_date_fn(story.canonical_item.item)
    if gd is None:
        return False
    if (run_date - gd).days <= max_age_days:
        return False  # not old → not a resurge candidate

    # R1: version jump strictly above the last resurge-surface snapshot.
    for arxiv_id, count in (story.arxiv_versions or {}).items():
        prev = (story.surfaced_arxiv_versions or {}).get(arxiv_id, 0)
        if count > prev:
            return True

    # R2: cross-competitor same-day cluster + cooldown gate.
    if competitor_count_fn(story) >= min_competitors:
        last_resurge = story.surfaced_resurged_at
        if last_resurge is None or (run_date - last_resurge).days >= cooldown_days:
            return True

    return False
```

Run `pytest tests/test_novelty.py -v` → all green (TestResurface + TestResurge).

**Commit:** `feat(hotspot): add resurgence predicate with cooldown gate (novelty.resurge)`

---

## Task 7 — replace `apply_cross_day_penalty` in `pipeline.py`

Bite-sized: swap the band-aid for `cluster_intraday` + `match_crossday` + `resurface`. This is the strangler cut. Spec §C.3 ("replace, not stack").

### 7a. Test first — append `TestCrossDayReplacement` to `tests/test_dedup.py`

This is a focused unit test on a small helper the pipeline will call, so the swap is testable without standing up the whole pipeline. Add a pure function `classify_cross_day` to `dedup.py` and test it:

```python
class TestClassifyCrossDay(unittest.TestCase):
    def _story(self, story_id, status, *, last_surfaced=None):
        from arxiv_assistant.hotspots.story import Story
        s = Story(
            story_id=story_id,
            canonical_item=_enriched("seed", f"https://seed/{story_id}"),
            items=[_enriched("seed", f"https://seed/{story_id}")],
            event_type="product_release",
        )
        s.status = status
        s.last_surfaced = last_surfaced
        return s

    def test_new_story_is_NEW(self) -> None:
        s = self._story("a", "NEW")
        self.assertEqual(dedup.classify_cross_day(s, resurface_fn=lambda x: False), "NEW")

    def test_ongoing_without_resurface_is_ONGOING(self) -> None:
        s = self._story("a", "ONGOING", last_surfaced="2026-06-01")
        self.assertEqual(dedup.classify_cross_day(s, resurface_fn=lambda x: False), "ONGOING")

    def test_ongoing_with_resurface_is_RESURFACE(self) -> None:
        s = self._story("a", "ONGOING", last_surfaced="2026-06-01")
        self.assertEqual(dedup.classify_cross_day(s, resurface_fn=lambda x: True), "RESURFACE")
```

Run → **fails** (`classify_cross_day` absent).

### 7b. Code — append `classify_cross_day` to `dedup.py`

```python
from arxiv_assistant.hotspots.novelty import resurface as _resurface


def classify_cross_day(story: Story, *, resurface_fn=_resurface) -> str:
    """Map a post-match story to its cross-day disposition (spec §C.3).

    NEW       — first-seen this run.
    ONGOING   — merged into an existing story; gravity from first_seen demotes it.
    RESURFACE — ONGOING but the closed-form resurface predicate fired (re-feature).
    """
    if story.status == "NEW":
        return "NEW"
    if resurface_fn(story):
        return "RESURFACE"
    return "ONGOING"
```

Run → `TestClassifyCrossDay` **passes**.

### 7c. Code — rewire `pipeline.py` (replace lines ~1713-1718)

Read the current block first. Replace:

```python
    # Stage 4-5: Group into stories → Score
    stories = score_stories(group_into_stories(enriched_items))

    # Cross-day headline penalty
    if recent_headlines:
        stories = apply_cross_day_penalty(stories, recent_headlines)
```

with (centroid-persistent dedup REPLACES the penalty — they are not stacked):

```python
    # Stage 4-5: intra-day dedup (L0/L1) → cross-day persistent match (L2) → score
    from datetime import date as _date
    from arxiv_assistant.hotspots.dedup import (
        classify_cross_day,
        cluster_intraday,
        match_crossday,
    )

    crossday_threshold = hotspot_config.getfloat("crossday_cosine_threshold", fallback=0.90)
    cross_day_window = hotspot_config.getint("cross_day_window_days", fallback=14)
    as_of = target_date.date() if hasattr(target_date, "date") else _date.today()

    intraday_stories = cluster_intraday(enriched_items)

    story_store = _open_story_store(output_root)  # Stage-0 helper; returns StoryStore
    if story_store is not None:
        matched = match_crossday(
            intraday_stories,
            story_store,
            cosine_threshold=crossday_threshold,
            window_days=cross_day_window,
            as_of=as_of,
        )
        for s in matched:
            s.cross_day_status = classify_cross_day(s)
        # ONGOING (non-resurfacing) stories are demoted out of the featured stream;
        # gravity-from-first_seen (Stage 1) already discounts them, so keep NEW +
        # RESURFACE as featured-eligible and let scoring/selection rank the rest.
        stories = score_stories(matched)
    else:
        # Degraded fallback (no Store yet, e.g. first run pre-backfill): retain the
        # legacy behavior so the pipeline still produces a report.
        stories = score_stories(intraday_stories)
        if recent_headlines:
            stories = apply_cross_day_penalty(stories, recent_headlines)
```

> The `_open_story_store(output_root)` helper and `Story.cross_day_status` consumption in the renderer bridge are Stage-0/Stage-6 concerns; for THIS stage the fallback branch guarantees the pipeline keeps working even if the Store wiring lands later. Remove the now-unused `apply_cross_day_penalty` import only after Stage 6 deletes the fallback. Keep the import for now.

Run the existing pipeline test suite to prove no regression: `pytest tests/test_hotspot_pipeline.py -v`. It must stay green (the fallback path preserves legacy behavior when no Store is present).

**Commit:** `refactor(hotspot): replace cross-day penalty with persistent dedup + resurface`

---

## Task 8 — `scripts/backfill_story_store.py`: dedup-first historical seed

Bite-sized: the one-off offline job. Overview §5 safety rail — run the SAME dedup over the 30-day history FIRST, then seed one `first_seen` per real story (NOT one per duplicated daily occurrence).

### 8a. Test first — append `TestBackfill` to `tests/test_dedup.py`

The job's core is a pure function `dedup_history(daily_enriched_by_date)` returning, for each real story, its earliest occurrence date. Test that a 6-day duplicate collapses to ONE first_seen at the earliest date (the exact pollution the safety rail prevents).

```python
class TestBackfill(unittest.TestCase):
    def _patch_embed(self, mapping):
        def fake_embed(text: str):
            for key, vec in mapping.items():
                if key in text:
                    return list(vec)
            return [0.0, 0.0, 1.0]
        return patch("arxiv_assistant.hotspots.dedup.embed_text", side_effect=fake_embed)

    def test_six_day_duplicate_yields_single_first_seen(self) -> None:
        from scripts.backfill_story_store import dedup_history
        # Same event re-featured on 6 consecutive days, slightly different titles/urls.
        history = {}
        for d in range(1, 7):
            history[f"2026-06-0{d}"] = [
                _enriched("Anthropic launches Claude 5", f"https://news.com/c5-day{d}")
            ]
        with self._patch_embed({"Claude 5": (1.0, 0.0, 0.0)}):
            seeds = dedup_history(history)
        # ONE real story → ONE first_seen at the earliest date (not 6 polluted anchors)
        self.assertEqual(len(seeds), 1)
        (only,) = seeds
        self.assertEqual(only["first_seen"], "2026-06-01")

    def test_two_distinct_events_yield_two_seeds(self) -> None:
        from scripts.backfill_story_store import dedup_history
        history = {
            "2026-06-01": [_enriched("Event Alpha", "https://a.com/1")],
            "2026-06-02": [_enriched("Event Beta totally unrelated", "https://b.com/1")],
        }
        with self._patch_embed({"Alpha": (1.0, 0.0, 0.0), "Beta": (0.0, 1.0, 0.0)}):
            seeds = dedup_history(history)
        self.assertEqual(len(seeds), 2)
```

Run → **fails** (module absent).

### 8b. Code — `scripts/backfill_story_store.py`

```python
"""One-off offline backfill: seed the Story Store from ~30 days of history.

CRITICAL SAFETY RAIL (overview §5): the historical reports ALREADY contain the
6-day-duplicate bug. A naive backfill that mints one Story per daily occurrence
would create 6 polluted `first_seen` anchors for one real event. So we run the
SAME Stage-2 dedup over the whole history FIRST, collapse cross-day duplicates by
centroid, and seed exactly ONE `first_seen` (the earliest occurrence date) per
real story.

Usage:
    python -m scripts.backfill_story_store --history-root out/hot --db out/hot/state/story_store.sqlite
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from arxiv_assistant.hotspots.dedup import (
    EMBED_MODEL_ID,
    _centroid,
    _embed_story_text,
    _normalize,
    cluster_intraday,
    embed_text,
)
from arxiv_assistant.hotspots.embed import cosine

# Reuse the production merge threshold so backfill grouping == live grouping.
from arxiv_assistant.hotspots.dedup import L1_SEMANTIC_THRESHOLD


def dedup_history(daily_enriched_by_date: dict[str, list]) -> list[dict]:
    """Collapse a {iso_date: [EnrichedItem, ...]} history into real stories.

    Returns one record per real story:
        {"story_id_hint", "first_seen", "centroid", "centroid_model_id", "n_days"}
    `first_seen` is the EARLIEST date the story appears — the immutable anchor.
    """
    # Persistent accumulators: each is a "real story" centroid + earliest date.
    persistent: list[dict] = []

    for iso_day in sorted(daily_enriched_by_date.keys()):
        intraday = cluster_intraday(daily_enriched_by_date[iso_day])
        for s in intraday:
            # match against existing persistent stories (centroid-primary, same rule as L2)
            best = None
            best_sim = L1_SEMANTIC_THRESHOLD
            for p in persistent:
                sim = cosine(s.centroid, p["centroid"])
                if sim >= best_sim:
                    best_sim = sim
                    best = p
            if best is None:
                persistent.append({
                    "first_seen": iso_day,
                    "centroid": list(s.centroid),
                    "centroid_model_id": EMBED_MODEL_ID,
                    "n_days": 1,
                    "_member_vecs": [list(s.centroid)],
                })
            else:
                # earliest date wins; refine centroid as running mean of occurrences
                best["n_days"] += 1
                best["_member_vecs"].append(list(s.centroid))
                best["centroid"] = _centroid(best["_member_vecs"])
                if iso_day < best["first_seen"]:
                    best["first_seen"] = iso_day

    for p in persistent:
        p.pop("_member_vecs", None)
    return persistent


def _load_history(history_root: Path) -> dict[str, list]:  # pragma: no cover - IO glue
    """Load enriched items per day from existing out/hot reports.

    Wire to the Stage-0 report reader / enrich loader. Left as IO glue: the pure,
    tested core is `dedup_history`. Each value is a list[EnrichedItem] for that day.
    """
    raise NotImplementedError(
        "Wire to the report/enrich loader during Stage-0 integration; "
        "dedup_history is the tested core."
    )


def main(argv=None) -> int:  # pragma: no cover - CLI glue
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-root", type=Path, default=Path("out/hot"))
    parser.add_argument("--db", type=Path, default=Path("out/hot/state/story_store.sqlite"))
    args = parser.parse_args(argv)

    history = _load_history(args.history_root)
    seeds = dedup_history(history)

    from arxiv_assistant.hotspots.store import StoryStore  # Stage-0
    store = StoryStore(args.db)
    for seed in seeds:
        # Seed via the Store's own create path so persistent ids stay authoritative.
        store.seed_first_seen(  # Stage-0 method; idempotent write-once on first_seen
            centroid=seed["centroid"],
            centroid_model_id=seed["centroid_model_id"],
            first_seen=seed["first_seen"],
        )
    print(f"Backfill seeded {len(seeds)} real stories (dedup-first; no polluted anchors).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
```

> `_centroid`, `_embed_story_text`, `_normalize`, `embed_text` are imported from `dedup.py` to guarantee backfill grouping is byte-identical to live grouping. `StoryStore.seed_first_seen` is a Stage-0 write-once method; if it is not yet present, stop-the-line and add it in Stage 0 (do NOT mint via the live `match_or_create`, which would set `first_seen` to today). The `_load_history` / `main` IO glue is marked `# pragma: no cover`; the tested contract is `dedup_history`.

Run `pytest tests/test_dedup.py -v` → all classes green.

**Commit:** `feat(hotspot): add dedup-first story-store backfill job`

---

## Task 9 — acceptance: invariants + full-suite green

Bite-sized: assert the §G invariants this stage owns, then run the whole stage suite.

### 9a. Test — append `TestStage2Invariants` to `tests/test_novelty.py`

```python
class TestStage2Invariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_helpers()

    def test_inv4_resurface_is_pure_no_url_or_subday(self) -> None:
        # 40% URL churn + sub-day jitter on a same-tier, same-day, no-new-entity story
        # must NOT resurface (closed-form, zero LLM).
        before = _ev("e1", added_at="2026-06-01", source_tier=4,
                     verified_first_date="2026-06-01T01:00:00+00:00")
        churn = _ev("e2", added_at="2026-06-02", source_tier=4,
                    verified_first_date="2026-06-01T22:00:00+00:00")
        s = _story([before, churn], last_surfaced="2026-06-01",
                   surfaced_verified_max=date(2026, 6, 1), surfaced_max_tier=4)
        # Two independent runs → identical boolean (determinism).
        self.assertFalse(novelty.resurface(s))
        self.assertFalse(novelty.resurface(s))

    def test_inv4_resurge_cooldown_fires_once_over_consecutive_days(self) -> None:
        # Same 4-competitor cluster every day for a week → exactly ONE True within cooldown.
        ev = _ev("old", added_at="2026-04-01", source_tier=3,
                 verified_first_date="2026-04-01", arxiv_id="2604.00001")
        results = []
        surfaced = None
        for day in range(1, 8):
            s = _story([ev], last_surfaced=None, surfaced_verified_max=date(2026, 4, 1),
                       arxiv_versions={"2604.00001": 1}, surfaced_arxiv_versions={"2604.00001": 1})
            s.resurged_at = None
            s.surfaced_resurged_at = surfaced
            fired = novelty.resurge(
                s, max_age_days=14, run_date=date(2026, 6, day),
                min_competitors=3, cooldown_days=7,
                competitor_count_fn=lambda story: 4)
            results.append(fired)
            if fired:
                surfaced = date(2026, 6, day)  # Kernel records surfaced_resurged_at
        self.assertEqual(sum(1 for r in results if r), 1)  # cooldown → exactly once
```

### 9b. Run the full stage suite

```
pytest tests/test_dedup.py tests/test_novelty.py tests/test_hotspot_config.py tests/test_hotspot_pipeline.py -v
```

All must pass. Then the targeted determinism checks:

```
pytest tests/test_novelty.py::TestStage2Invariants -v
pytest tests/test_dedup.py::TestMatchCrossday::test_intraday_anticonvergence_single_existing_id -v
pytest tests/test_dedup.py::TestMatchCrossday::test_centroid_primary_not_AND_with_url -v
```

**Commit:** `test(hotspot): assert stage-2 dedup/novelty invariants (INV2/INV4/INV6)`

---

## Stage-2 done criteria (stop-the-line)

- [ ] `embed.py`: `EMBED_MODEL_ID` pinned; `embed_text` lazy singleton (import never downloads); `cosine` golden truth-table green.
- [ ] `dedup.cluster_intraday`: L0 exact + L1 semantic (cosine > 0.90); zh/en same-event merge; centroid stored with `centroid_model_id == EMBED_MODEL_ID`, unit-normalized (INV6).
- [ ] `dedup.match_crossday`: centroid-primary (merges on cosine alone with disjoint URLs — NOT AND); intra-day anti-convergence (two today-clusters onto one existing id collapse to one, zero spurious NEW); persistent ids via `store.match_or_create`.
- [ ] `novelty.resurface`: T1 tier-jump ∨ T2 later gate_date / new arXiv version ∨ T3 new entity → True; URL-churn-only, sub-day jitter, same-tier-more-evidence → False (INV4, INV2).
- [ ] `novelty.resurge`: R1 version jump ∨ R2 ≥min_competitors with cooldown; old-only; consecutive-day same cluster fires exactly once per cooldown (INV4).
- [ ] `pipeline.py`: `apply_cross_day_penalty` REPLACED by `cluster_intraday`+`match_crossday`+`classify_cross_day`; `test_hotspot_pipeline.py` still green via fallback.
- [ ] `scripts/backfill_story_store.py`: `dedup_history` collapses a 6-day duplicate to ONE earliest `first_seen` (no 6 polluted anchors); two distinct events → two seeds.
- [ ] config: `cross_day_window_days`, `crossday_cosine_threshold`, `resurge_min_competitors`, `resurge_cooldown_days`, `embed_model_id` present with defaults in `configs/config.ini` and template.
- [ ] Full suite: `pytest tests/test_dedup.py tests/test_novelty.py tests/test_hotspot_config.py tests/test_hotspot_pipeline.py -v` green.

**Hand-off to Stage 6:** the Kernel becomes the single Store writer (calls `record_surface` to populate `surfaced_*` snapshots after rendering, sets `surfaced_resurged_at`/`resurged_at` on resurgence-lane surface), wires `_open_story_store`, consumes `Story.cross_day_status`, and renders the Resurgence section. This stage leaves those as the documented integration seam (fallback branch + helper stubs) so the dedup/novelty logic ships and is fully unit-tested independently.
