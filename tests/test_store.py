from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem


def _make_item(
    *,
    source_id: str = "hf_papers",
    title: str = "A paper",
    url: str = "https://arxiv.org/abs/2606.00001",
    provenance: str = "",
    verified_first_date: str | None = None,
) -> HotspotItem:
    return HotspotItem(
        source_id=source_id,
        source_name="HF",
        source_role="paper_trending",
        source_type="paper",
        title=title,
        summary="A summary.",
        url=url,
        canonical_url=url,
        published_at="2026-06-01T12:00:00+00:00",
        metadata={"arxiv_id": "2606.00001"},
        provenance=provenance,
        verified_first_date=verified_first_date,
    )


class TestHotspotItemFields(unittest.TestCase):
    def test_new_fields_default_safe(self) -> None:
        item = HotspotItem(
            source_id="hf_papers",
            source_name="HF",
            source_role="paper_trending",
            source_type="paper",
            title="A paper",
            summary="A summary.",
            url="https://arxiv.org/abs/2606.00001",
            canonical_url="https://arxiv.org/abs/2606.00001",
        )
        self.assertIsNone(item.verified_first_date)
        self.assertEqual(item.provenance, "")

    def test_new_fields_round_trip_through_to_dict(self) -> None:
        item = _make_item(provenance="native:hf_papers", verified_first_date="2026-05-20T00:00:00+00:00")
        payload = item.to_dict()
        self.assertEqual(payload["provenance"], "native:hf_papers")
        self.assertEqual(payload["verified_first_date"], "2026-05-20T00:00:00+00:00")
        # round-trip back through the dataclass
        restored = HotspotItem(**payload)
        self.assertEqual(restored.provenance, "native:hf_papers")
        self.assertEqual(restored.verified_first_date, "2026-05-20T00:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
