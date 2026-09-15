import unittest
from datetime import date, timedelta
from unittest.mock import patch
from xml.etree import ElementTree

from arxiv_assistant.apis import arxiv_oai

RECORD = """
  <record xmlns="http://www.openarchives.org/OAI/2.0/">
    <header><identifier>oai:arXiv.org:{id}</identifier><datestamp>{stamp}</datestamp></header>
    <metadata>
      <arXiv xmlns="http://arxiv.org/OAI/arXiv/">
        <id>{id}</id>
        <created>{created}</created>
        <authors><author><keyname>Example</keyname><forenames>Ada</forenames></author></authors>
        <title>Paper {id}</title>
        <categories>{cats}</categories>
        <abstract>Abstract {id}</abstract>
      </arXiv>
    </metadata>
  </record>"""


def feed(records, token=""):
    token_xml = f"<resumptionToken>{token}</resumptionToken>" if token else ""
    return ('<?xml version="1.0"?><OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">'
            f"<ListRecords>{''.join(records)}{token_xml}</ListRecords></OAI-PMH>")


def record(id_, created, cats="cs.LG cs.AI", stamp=None):
    return RECORD.format(id=id_, created=created, cats=cats, stamp=stamp or created)


def error(code):
    return ('<?xml version="1.0"?><OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">'
            f'<error code="{code}">no</error></OAI-PMH>')


class Response:
    def __init__(self, text):
        self.text = text


class RecordParsingTest(unittest.TestCase):
    def parse(self, xml):
        return arxiv_oai.record_to_paper(ElementTree.fromstring(xml.strip()))

    def test_fields_come_from_the_metadata(self):
        paper, created, cats = self.parse(record("2607.00001", "2026-07-20"))
        self.assertEqual(paper.arxiv_id, "2607.00001")
        self.assertEqual(paper.title, "Paper 2607.00001")
        self.assertEqual(paper.authors, ["Ada Example"])
        self.assertEqual(created, "2026-07-20")
        self.assertEqual(cats, ["cs.LG", "cs.AI"])

    def test_created_is_not_the_datestamp(self):
        # The whole reason this source needs a wider harvest window.
        _, created, _ = self.parse(record("1205.1277", "2012-05-06", stamp="2026-07-22"))
        self.assertEqual(created, "2012-05-06")

    def test_a_record_without_metadata_is_skipped_not_crashed(self):
        xml = ('<record xmlns="http://www.openarchives.org/OAI/2.0/">'
               "<header><identifier>x</identifier></header></record>")
        self.assertEqual(self.parse(xml), (None, "", []))


class HarvestTest(unittest.TestCase):
    def run_harvest(self, pages, area="cs.LG", window=((2026, 7, 20), (2026, 7, 20)), **kw):
        self.calls = []

        def fake_get(params):
            self.calls.append(params)
            # Clamp: the retry decorator re-enters on failure, and running off
            # the end of the list would report an IndexError instead of the
            # error under test.
            return Response(pages[min(len(self.calls) - 1, len(pages) - 1)])

        with patch.object(arxiv_oai, "oai_get", side_effect=fake_get):
            return arxiv_oai.get_papers_from_arxiv_oai(area, window[0], window[1], **kw)

    def test_only_papers_created_inside_the_window_are_kept(self):
        pages = [feed([record("2607.00001", "2026-07-20"),
                       record("2607.00002", "2026-07-25"),
                       record("1205.1277", "2012-05-06")])]
        _, papers = self.run_harvest(pages)
        self.assertEqual([p.arxiv_id for p in papers], ["2607.00001"])

    def test_the_harvest_window_runs_past_the_wanted_window(self):
        self.run_harvest([feed([])])
        self.assertEqual(self.calls[0]["from"], "2026-07-20")
        self.assertGreater(self.calls[0]["until"], "2026-07-20")

    def test_the_until_date_is_never_in_the_future(self):
        # arXiv rejects that with badArgument, which would break every daily run.
        today = date.today()
        self.run_harvest([feed([])], window=(arxiv_oai.date_to_tuple(today),
                                             arxiv_oai.date_to_tuple(today)))
        self.assertLessEqual(self.calls[0]["until"], today.isoformat())

    def test_a_resumption_token_is_followed_and_carries_nothing_else(self):
        pages = [feed([record("2607.00001", "2026-07-20")], token="tok1"),
                 feed([record("2607.00002", "2026-07-20")])]
        _, papers = self.run_harvest(pages)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.calls[1], {"verb": "ListRecords", "resumptionToken": "tok1"})
        self.assertEqual(len(papers), 2)

    def test_a_repeated_record_is_returned_once(self):
        pages = [feed([record("2607.00001", "2026-07-20")], token="tok1"),
                 feed([record("2607.00001", "2026-07-20")])]
        _, papers = self.run_harvest(pages)
        self.assertEqual(len(papers), 1)

    def test_area_and_primary_filters(self):
        pages = [feed([record("2607.00001", "2026-07-20", cats="cs.AI cs.LG"),
                       record("2607.00003", "2026-07-20", cats="cs.CV")])]
        _, any_cat = self.run_harvest(pages)
        self.assertEqual([p.arxiv_id for p in any_cat], ["2607.00001"])

        _, primary = self.run_harvest(pages, force_primary=True)
        self.assertEqual(primary, [])

    def test_no_records_matched_is_an_answer_not_an_error(self):
        _, papers = self.run_harvest([error("noRecordsMatch")])
        self.assertEqual(papers, [])

    def test_any_other_protocol_error_is_raised(self):
        # patched sleep: the retry decorator would otherwise wait 30s a try.
        with patch("time.sleep"), self.assertRaises(RuntimeError):
            self.run_harvest([error("badArgument")])


class FlowControlTest(unittest.TestCase):
    def test_503_is_waited_out_then_retried(self):
        class R:
            def __init__(self, code, text=""):
                self.status_code, self.text, self.headers = code, text, {"Retry-After": "12"}

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise AssertionError("should not raise on the retry")

        answers = [R(503), R(200, "ok")]
        with patch.object(arxiv_oai.requests, "get", side_effect=answers), \
             patch.object(arxiv_oai, "pace_arxiv_request"), \
             patch.object(arxiv_oai.time, "sleep") as sleep:
            out = arxiv_oai.oai_get({"verb": "ListRecords"})
        self.assertEqual(out.text, "ok")
        self.assertEqual(sleep.call_args[0][0], 12.0)


if __name__ == "__main__":
    unittest.main()
