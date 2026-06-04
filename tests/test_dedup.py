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

    def test_zh_en_real_embeddings(self) -> None:
        """Real multilingual embedding test: same event zh/en should have cosine > unrelated pair.

        Uses the actual paraphrase-multilingual-MiniLM-L12-v2 model (cached locally).
        Demonstrates that zh/en same-event pair cosine >> unrelated pair cosine.
        """
        en_text = "Anthropic launches Claude 5"
        zh_text = "Anthropic 发布 Claude 5"
        unrelated_text = "Stock market falls on interest rate fears"

        vec_en = dedup._normalize(embed.embed_text(en_text))
        vec_zh = dedup._normalize(embed.embed_text(zh_text))
        vec_unrelated = dedup._normalize(embed.embed_text(unrelated_text))

        cosine_same = embed.cosine(vec_en, vec_zh)
        cosine_diff = embed.cosine(vec_en, vec_unrelated)

        # Log the measured values for transparency.
        print(f"\n[test_zh_en_real_embeddings] cosine(en, zh)={cosine_same:.4f}  cosine(en, unrelated)={cosine_diff:.4f}")

        # The zh/en same-event pair must score substantially higher than the unrelated pair.
        self.assertGreater(cosine_same, cosine_diff + 0.05,
            f"Expected zh/en same-event cosine ({cosine_same:.4f}) >> unrelated ({cosine_diff:.4f})")

        # If the model achieves ≥ 0.90, the L1 threshold is met for real data.
        # If not, we report the measured value rather than fake the test.
        if cosine_same >= 0.90:
            print(f"  → zh/en cosine {cosine_same:.4f} ≥ 0.90: L1 threshold met.")
        else:
            print(f"  → CONCERN: zh/en cosine {cosine_same:.4f} < 0.90 threshold; "
                  f"real zh/en merge would NOT fire with default threshold. "
                  f"Consider lowering cross_day_cosine_threshold or using stronger pairs.")


if __name__ == "__main__":
    unittest.main()
