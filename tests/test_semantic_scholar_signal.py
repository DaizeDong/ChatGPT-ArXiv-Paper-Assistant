from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from arxiv_assistant.apis.semantic_scholar import get_paper_citations
from arxiv_assistant.hotspots.pipeline import (
    _S2_BONUS_CAP,
    _build_paper_spotlight,
    _s2_significance_bonus,
)
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem


def _paper(arxiv_id: str, kind: str = "daily_hot", daily_score: int = 5) -> HotspotItem:
    return HotspotItem(
        source_id="hf_papers", source_name="HF", source_role="papers",
        source_type="papers", title=f"Paper {arxiv_id}", summary="s",
        url=f"https://arxiv.org/abs/{arxiv_id}", canonical_url=f"https://arxiv.org/abs/{arxiv_id}",
        published_at="2026-06-01T00:00:00Z", tags=[], authors=[],
        metadata={"spotlight_primary_kind": kind, "arxiv_id": arxiv_id,
                  "daily_score": daily_score, "relevance": 6, "novelty": 6},
    )


class TestGetPaperCitations(unittest.TestCase):
    def test_parses_batch_response(self) -> None:
        resp = MagicMock(status_code=200)
        resp.json.return_value = [
            {"citationCount": 120, "influentialCitationCount": 14, "year": 2024},
            None,  # S2 doesn't know the second paper
        ]
        with patch("arxiv_assistant.apis.semantic_scholar.requests.post", return_value=resp) as p:
            out = get_paper_citations(["2401.00001", "2606.99999"])
        self.assertEqual(out["2401.00001"]["influentialCitationCount"], 14)
        self.assertEqual(out["2401.00001"]["citationCount"], 120)
        self.assertNotIn("2606.99999", out)  # null entry omitted
        # one batch request, ARXIV: prefixed ids
        body = p.call_args.kwargs["json"]
        self.assertEqual(body["ids"], ["ARXIV:2401.00001", "ARXIV:2606.99999"])

    def test_empty_input_no_request(self) -> None:
        with patch("arxiv_assistant.apis.semantic_scholar.requests.post") as p:
            self.assertEqual(get_paper_citations([]), {})
            p.assert_not_called()

    def test_non_200_degrades_to_empty(self) -> None:
        resp = MagicMock(status_code=429)
        with patch("arxiv_assistant.apis.semantic_scholar.requests.post", return_value=resp):
            self.assertEqual(get_paper_citations(["2401.00001"]), {})

    def test_exception_degrades_to_empty(self) -> None:
        with patch("arxiv_assistant.apis.semantic_scholar.requests.post", side_effect=RuntimeError("net")):
            self.assertEqual(get_paper_citations(["2401.00001"]), {})


class TestS2Bonus(unittest.TestCase):
    def test_monotonic_and_bounded(self) -> None:
        self.assertEqual(_s2_significance_bonus(None), 0.0)
        self.assertEqual(_s2_significance_bonus({}), 0.0)
        low = _s2_significance_bonus({"influentialCitationCount": 1, "citationCount": 3})
        high = _s2_significance_bonus({"influentialCitationCount": 50, "citationCount": 500})
        self.assertGreater(high, low)
        self.assertGreater(low, 0.0)
        huge = _s2_significance_bonus({"influentialCitationCount": 100000, "citationCount": 999999})
        self.assertLessEqual(huge, _S2_BONUS_CAP)


def _items_of(spotlight: list, kind: str) -> list:
    # _build_paper_spotlight returns sections [{kind,label,description,items:[...]}, ...]
    for sec in spotlight:
        if sec.get("kind") == kind:
            return sec.get("items", [])
    return []


class TestSpotlightIntegration(unittest.TestCase):
    def test_cited_paper_outranks_uncited(self) -> None:
        items = [_paper("2401.AAAAA"), _paper("2606.BBBBB")]  # identical base scores
        cites = {"2401.AAAAA": {"influentialCitationCount": 30, "citationCount": 300}}
        spotlight = _build_paper_spotlight(
            items, max_daily_hot=6, max_new_frontier=4,
            use_s2_signal=True, citations_fn=lambda ids, api_key=None: cites,
        )
        order = [p["arxiv_id"] for p in _items_of(spotlight, "daily_hot")]
        self.assertEqual(order[0], "2401.AAAAA")  # the cited paper ranks first
        cited = next(p for p in _items_of(spotlight, "daily_hot") if p["arxiv_id"] == "2401.AAAAA")
        self.assertEqual(cited["s2_influential_citations"], 30)  # observability stashed

    def test_degraded_equals_baseline(self) -> None:
        items = [_paper("2401.AAAAA", daily_score=4), _paper("2606.BBBBB", daily_score=9)]
        baseline = _build_paper_spotlight(items, max_daily_hot=6, max_new_frontier=4, use_s2_signal=False)
        degraded = _build_paper_spotlight(
            items, max_daily_hot=6, max_new_frontier=4,
            use_s2_signal=True, citations_fn=lambda ids, api_key=None: {},  # S2 unavailable
        )
        base_order = [p["arxiv_id"] for p in _items_of(baseline, "daily_hot")]
        degr_order = [p["arxiv_id"] for p in _items_of(degraded, "daily_hot")]
        self.assertEqual(base_order, degr_order)  # zero behavior change when S2 yields nothing

    def test_flag_off_skips_s2_call(self) -> None:
        called = {"n": 0}
        def fn(ids, api_key=None):
            called["n"] += 1
            return {}
        _build_paper_spotlight([_paper("2401.AAAAA")], max_daily_hot=6, max_new_frontier=4,
                               use_s2_signal=False, citations_fn=fn)
        self.assertEqual(called["n"], 0)  # no network when flag is off


if __name__ == "__main__":
    unittest.main()
