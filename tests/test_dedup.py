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


if __name__ == "__main__":
    unittest.main()
