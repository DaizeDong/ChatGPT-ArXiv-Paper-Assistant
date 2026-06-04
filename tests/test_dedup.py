from __future__ import annotations

import math
import unittest
from unittest.mock import patch

from arxiv_assistant.hotspots import dedup, embed
from arxiv_assistant.hotspots.enrich import EnrichedItem
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem


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
        class _StubModel:
            def encode(self, text):  # noqa: D401
                return [float(len(text)), 1.0, 2.0]

        import arxiv_assistant.hotspots.embed as embed_mod
        embed_mod._MODEL = None
        with patch.object(embed_mod, "_load_model", return_value=_StubModel()) as mock_loader:
            embed_mod.embed_text("hello")
            embed_mod.embed_text("hello world")
            mock_loader.assert_called_once()
        embed_mod._MODEL = None  # reset global so we don't leak the stub to other tests


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

    def test_embed_dim_is_768(self) -> None:
        """mpnet-base-v2 produces 768-dimensional embeddings."""
        import arxiv_assistant.hotspots.embed as embed_mod
        # Reset any stub left from other tests.
        embed_mod._MODEL = None
        vec = embed.embed_text("hello world")
        self.assertEqual(len(vec), 768, f"Expected dim 768, got {len(vec)}")

    def test_l1_crosslingual_merge_beyond_l0(self) -> None:
        """Real-embedding POSITIVE: L1 semantic merge isolates cross-lingual zh/en same-event.

        Pair: 'Google unveils Gemini 3' / '谷歌发布 Gemini 3'.
        Empirically verified 2026-06: group_into_stories returns 2 separate stories
        (L0 SequenceMatcher does NOT fire — no shared Latin tokens beyond 'Gemini 3'
        which is only a 2-token proper noun and Jaccard/containment thresholds not met),
        while mpnet cosine ~0.979 >> 0.72 threshold so L1 merges them.

        The fixture uses EMPTY entities so Pass 3 entity-merge cannot fire,
        and distinct canonical_urls with no shared arxiv_id so Pass 2 cannot fire.
        Asserts BOTH:
          (a) group_into_stories([en, zh]) == 2  (L0 keeps them apart)
          (b) cluster_intraday([en, zh]) == 1    (L1 merges them)
        This proves the merge is done by L1, not L0.

        Uses paraphrase-multilingual-mpnet-base-v2 (cached). If cosine unexpectedly
        drops below 0.72, or L0 unexpectedly fires, we STOP and report.
        """
        en = _enriched(
            "Google unveils Gemini 3", "https://google.com/gemini3",
            entities=[],  # empty — disable Pass 3 entity-merge
        )
        zh = _enriched(
            "谷歌发布 Gemini 3", "https://zh.example.com/gemini3",
            entities=[],  # empty — disable Pass 3 entity-merge
        )

        # (a) Verify L0 keeps them apart.
        from arxiv_assistant.hotspots.story import group_into_stories as _l0
        l0_count = len(_l0([en, zh]))
        if l0_count != 2:
            self.fail(
                f"STOP: group_into_stories merged the Gemini-3 zh/en pair at L0 "
                f"(got {l0_count} stories). L0 fired unexpectedly; pick a different pair."
            )

        # Measure cosine for transparency.
        vec_en = dedup._normalize(embed.embed_text("Google unveils Gemini 3. "))
        vec_zh = dedup._normalize(embed.embed_text("谷歌发布 Gemini 3. "))
        cosine_val = embed.cosine(vec_en, vec_zh)
        print(f"\n[test_l1_crosslingual_merge_beyond_l0] cosine(en, zh)={cosine_val:.4f}  threshold=0.72")

        if cosine_val < 0.72:
            self.fail(
                f"STOP: zh/en same-event cosine ({cosine_val:.4f}) < 0.72 threshold; "
                f"the probe assumption does not hold with the loaded model. "
                f"Do not fake — investigate the model or pair."
            )

        # (b) Verify L1 merges them.
        stories = dedup.cluster_intraday([en, zh])
        self.assertEqual(
            l0_count, 2,
            "L0 must keep the pair separate (this is the isolation guarantee).",
        )
        self.assertEqual(
            len(stories), 1,
            f"Expected 1 merged story: L1 semantic (cosine={cosine_val:.4f} > 0.72) "
            f"should merge what L0 kept apart; got {len(stories)}.",
        )

    def test_gpt5_sora2_no_merge(self) -> None:
        """Real-embedding NEGATIVE: related-but-different events must NOT merge.

        'OpenAI GPT-5 launch' vs 'OpenAI Sora 2 video generation' scores ~0.46 < 0.72.
        Titles are chosen so that L0 title-token passes do not fire either
        (containment 0.50 < 0.80; SequenceMatcher 0.41 < 0.65).
        If the measured cosine unexpectedly crosses 0.72 we STOP and report.
        """
        a = _enriched("OpenAI GPT-5 launch", "https://openai.com/gpt5")
        b = _enriched("OpenAI Sora 2 video generation", "https://openai.com/sora2")

        stories = dedup.cluster_intraday([a, b])

        vec_a = dedup._normalize(embed.embed_text("OpenAI GPT-5 launch. "))
        vec_b = dedup._normalize(embed.embed_text("OpenAI Sora 2 video generation. "))
        cosine_val = embed.cosine(vec_a, vec_b)
        print(f"\n[test_gpt5_sora2_no_merge] cosine(GPT-5-launch, Sora-2-video)={cosine_val:.4f}  threshold=0.72")

        if cosine_val >= 0.72:
            self.fail(
                f"STOP: GPT-5-launch/Sora-2-video cosine ({cosine_val:.4f}) >= 0.72 threshold; "
                f"separation probe assumption violated. "
                f"Do not fake — investigate the model or threshold."
            )

        self.assertEqual(
            len(stories), 2,
            f"Expected 2 separate stories for GPT-5/Sora-2 pair "
            f"(cosine={cosine_val:.4f}), got {len(stories)}",
        )


if __name__ == "__main__":
    unittest.main()
