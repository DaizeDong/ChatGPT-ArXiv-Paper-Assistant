from __future__ import annotations

import math
from typing import Sequence

# §G.7 / overview §2.7: pinned multilingual Matryoshka model id, stored on every centroid.
# Multilingual so that an English item and its `_zh` translation embed into the SAME space
# (§C.1 L1 cross-language merge). Keep in sync with configs `embed_model_id`.
EMBED_MODEL_ID = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

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
    except (ImportError, ModuleNotFoundError):
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
    except (ImportError, ModuleNotFoundError):  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "Stage-2 embedding requires `fastembed` or `sentence-transformers`. "
            "Install one: `pip install fastembed` (preferred) or "
            "`pip install sentence-transformers`."
        )


def embed_text(text: str | None) -> list[float]:
    """Embed `title + lede` text into the pinned multilingual space."""
    global _MODEL
    if _MODEL is None:
        _MODEL = _load_model()
    return list(_MODEL.encode(text or ""))


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
