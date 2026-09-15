from __future__ import annotations

import unittest
from unittest.mock import patch

from arxiv_assistant.apis import arxiv as arxiv_api


def _feed(count: int, offset: int = 0, area: str = "cs.LG") -> str:
    """One Atom feed carrying `count` synthetic entries, ids offset by `offset`."""
    entries = []
    for i in range(count):
        n = offset + i
        entries.append(
            "<entry>"
            f"<id>http://arxiv.org/abs/2601.{n:05d}v1</id>"
            f"<title>Synthetic paper {n}</title>"
            f"<summary>Synthetic abstract {n}</summary>"
            "<author><name>Ada Example</name></author>"
            f'<arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="{area}"/>'
            "</entry>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom">' + "".join(entries) + "</feed>"
    )


class _Response:
    def __init__(self, text: str) -> None:
        self.text = text


class ArxivPagingTest(unittest.TestCase):
    """The API query must page the way arXiv's manual asks, not demand everything.

    Asking for max_results=10000 in one call is what earned this host a
    multi-hour 429 on the API endpoint. These tests fail if the page size
    regresses past the documented slice, or if paging silently drops results.
    """

    def _run(self, pages, page_size=None):
        calls = []

        def fake_get(url):
            calls.append(url)
            return _Response(pages[len(calls) - 1])

        page_size = page_size or arxiv_api.ARXIV_PAGE_SIZE
        with patch.object(arxiv_api, "_arxiv_get", side_effect=fake_get), \
             patch.object(arxiv_api, "ARXIV_PAGE_SIZE", page_size):
            entries, papers = arxiv_api.get_papers_from_arxiv_api(
                "cs.LG", (2026, 7, 6), (2026, 7, 6)
            )
        return calls, entries, papers

    def test_page_size_stays_within_the_documented_slice(self):
        self.assertLessEqual(arxiv_api.ARXIV_PAGE_SIZE, 2000)
        calls, _, _ = self._run([_feed(3)])
        self.assertIn(f"max_results={arxiv_api.ARXIV_PAGE_SIZE}", calls[0])
        self.assertNotIn("max_results=10000", calls[0])

    def test_a_short_page_ends_the_loop_without_an_extra_request(self):
        calls, entries, papers = self._run([_feed(3)], page_size=5)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(entries), 3)
        self.assertEqual(len(papers), 3)

    def test_a_full_page_is_followed_and_every_result_is_kept(self):
        pages = [_feed(5, offset=0), _feed(5, offset=5), _feed(2, offset=10)]
        calls, entries, papers = self._run(pages, page_size=5)
        self.assertEqual(len(calls), 3)
        self.assertIn("start=0", calls[0])
        self.assertIn("start=5", calls[1])
        self.assertIn("start=10", calls[2])
        self.assertEqual(len(entries), 12)
        self.assertEqual(len(papers), 12)
        # Distinct ids: a paging bug that re-requests page 0 would still count 12.
        self.assertEqual(len({p.arxiv_id for p in papers}), 12)

    def test_an_empty_first_page_returns_nothing(self):
        calls, entries, papers = self._run([_feed(0)], page_size=5)
        self.assertEqual(len(calls), 1)
        self.assertEqual(entries, [])
        self.assertEqual(papers, [])


if __name__ == "__main__":
    unittest.main()
