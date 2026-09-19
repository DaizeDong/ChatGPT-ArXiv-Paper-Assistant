"""Paper-pipeline sources: where a day's candidate papers come from.

This package used to re-export three hotspot fetchers as well, so importing the
paper pipeline's sources pulled in the hotspot feed's, and `apis` looked like it
owned both. The hotspot sources live in arxiv_assistant.hotspot.sources, with
the rest of that subsystem.
"""

from . import arxiv, semantic_scholar

__all__ = [
    "arxiv",
    "semantic_scholar",
]
