"""Tests for ``arxiv_assistant.reader.questions``."""
from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from arxiv_assistant.reader.questions import (
    FIELD_KEYS,
    ReaderQuestion,
    ReaderQuestionError,
    load_questions,
    parse_question_document,
    render_questions_block,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_QUESTIONS_DIR = REPO_ROOT / "configs" / "reader" / "questions"
READER_PKG = REPO_ROOT / "arxiv_assistant" / "reader"

# A fully written document, in the exact shape a researcher would produce.
POPULATED_DOC = """# Q2: Do continuous bit-widths beat integer quantization?

## 当前看法

Integer bit-widths are a deployment artifact, not an accuracy limit.

## 支持证据

- LiftQuant reports 3.4 bits at 2-bit memory cost.

## 反对证据

## 什么会让我改看法

A kernel benchmark showing the decode overhead eats the memory win.

## 想做的实验
"""


class ParseQuestionDocumentTest(unittest.TestCase):
    def test_extracts_id_and_title_from_ascii_colon(self) -> None:
        question = parse_question_document(
            "# Q3: Does RL on verifiers generalize?\n", fallback_id="ignored"
        )
        self.assertEqual(question.question_id, "q3")
        self.assertEqual(question.title, "Does RL on verifiers generalize?")

    def test_extracts_id_and_title_from_fullwidth_colon(self) -> None:
        """The templates are authored in Chinese; a full-width colon is normal."""
        question = parse_question_document(
            "# Q4：稀疏注意力还有没有上限\n", fallback_id="ignored"
        )
        self.assertEqual(question.question_id, "q4")
        self.assertEqual(question.title, "稀疏注意力还有没有上限")

    def test_lowercase_q_and_fallback_id(self) -> None:
        self.assertEqual(
            parse_question_document("# q7: whatever\n", fallback_id="zz").question_id,
            "q7",
        )
        # No title line at all -> the filename stem is the identity of record.
        headless = parse_question_document("## 当前看法\n\nsomething\n", fallback_id="Q9")
        self.assertEqual(headless.question_id, "q9")
        self.assertEqual(headless.title, "")

    def test_maps_all_five_headings_to_field_keys(self) -> None:
        text = (
            "# Q1: t\n"
            "## 当前看法\nview\n"
            "## 支持证据\nfor\n"
            "## 反对证据\nagainst\n"
            "## 什么会让我改看法\nchange\n"
            "## 想做的实验\nexp\n"
        )
        question = parse_question_document(text, fallback_id="q1")
        self.assertEqual(
            {key: question.fields[key] for key in FIELD_KEYS},
            {
                "current_view": "view",
                "supporting_evidence": "for",
                "opposing_evidence": "against",
                "would_change_my_mind": "change",
                "experiments": "exp",
            },
        )

    def test_tolerates_a_missing_heading(self) -> None:
        """A half-written document is normal input, never an error."""
        text = "# Q5: t\n## 当前看法\nview\n## 想做的实验\nexp\n"
        question = parse_question_document(text, fallback_id="q5")
        self.assertEqual(set(question.fields), set(FIELD_KEYS))
        self.assertEqual(question.fields["current_view"], "view")
        self.assertEqual(question.fields["opposing_evidence"], "")
        self.assertEqual(question.populated_fields, ["current_view", "experiments"])

    def test_unknown_heading_is_dropped_not_misfiled(self) -> None:
        text = "# Q6: t\n## 随便写的标题\nstray\n## 当前看法\nview\n"
        question = parse_question_document(text, fallback_id="q6")
        self.assertEqual(question.populated_fields, ["current_view"])
        self.assertNotIn("stray", "".join(question.fields.values()))

    def test_populated_document_reports_populated_fields(self) -> None:
        question = parse_question_document(POPULATED_DOC, fallback_id="q2")
        self.assertTrue(question.is_populated)
        self.assertEqual(
            question.populated_fields,
            ["current_view", "supporting_evidence", "would_change_my_mind"],
        )
        self.assertIn("LiftQuant", question.fields["supporting_evidence"])
        self.assertEqual(question.fields["opposing_evidence"], "")

    def test_html_comment_only_field_is_not_content(self) -> None:
        question = parse_question_document(
            "# Q1: t\n## 当前看法\n<!-- write your belief here -->\n", fallback_id="q1"
        )
        self.assertFalse(question.is_populated)

    def test_multiline_html_comment_only_field_is_not_content(self) -> None:
        """Only comment-stripping can handle this one."""
        question = parse_question_document(
            "# Q1: t\n## 当前看法\n<!-- write your belief here,\n"
            "     as a falsifiable statement,\n     not a question -->\n",
            fallback_id="q1",
        )
        self.assertFalse(question.is_populated)
        self.assertEqual(question.fields["current_view"], "")

    def test_placeholder_line_only_field_is_not_content(self) -> None:
        question = parse_question_document(
            "# Q1: t\n## 当前看法\n<在这里写下你的看法>\n", fallback_id="q1"
        )
        self.assertFalse(question.is_populated)


class ShippedTemplateGuardTest(unittest.TestCase):
    """The shipped templates must never be mistaken for filled-in documents."""

    def setUp(self) -> None:
        self.questions = load_questions(SHIPPED_QUESTIONS_DIR)

    def test_all_shipped_templates_parse(self) -> None:
        self.assertGreaterEqual(len(self.questions), 1)
        for question in self.questions:
            with self.subTest(path=question.path):
                self.assertRegex(question.question_id, r"^q\d+$")
                self.assertEqual(set(question.fields), set(FIELD_KEYS))

    def test_templates_still_contain_the_noise_this_guard_is_about(self) -> None:
        """Premise check: a template stripped of its scaffolding proves nothing.

        Without this, someone could delete every comment and placeholder from the
        templates and the guard below would keep passing for the wrong reason.
        """
        blobs = [p.read_text(encoding="utf-8") for p in SHIPPED_QUESTIONS_DIR.glob("q*.md")]
        self.assertTrue(any("<!--" in b for b in blobs), "no HTML comments left in templates")
        self.assertTrue(
            any(re.search(r"^#\s*[Qq]\d+\s*[:：]\s*<[^!].*>\s*$", b, re.M) for b in blobs),
            "no <placeholder> titles left in templates",
        )

    def test_shipped_template_titles_strip_to_nothing(self) -> None:
        """The other half of the guard: the title is a ``<placeholder>`` too.

        If placeholder stripping regresses, every template acquires a title that
        reads like a real research question, and the digest starts quoting it.
        """
        for question in self.questions:
            with self.subTest(path=question.path):
                self.assertEqual(question.title, "")

    def test_every_shipped_template_is_unpopulated(self) -> None:
        for question in self.questions:
            with self.subTest(path=question.path):
                self.assertFalse(
                    question.is_populated,
                    f"{question.path} looks populated; template noise is no longer "
                    f"being stripped (populated fields: {question.populated_fields})",
                )

    def test_shipped_templates_render_to_nothing(self) -> None:
        self.assertEqual(render_questions_block(self.questions), "")


class RenderQuestionsBlockTest(unittest.TestCase):
    def test_renders_only_populated_questions(self) -> None:
        populated = parse_question_document(POPULATED_DOC, fallback_id="q2")
        empty = parse_question_document("# Q1: untouched\n## 当前看法\n", fallback_id="q1")
        block = render_questions_block([empty, populated])
        self.assertIn("q2", block)
        self.assertNotIn("q1", block)
        self.assertIn("- current_view:", block)
        # Empty fields of a populated question are marked, not omitted: the model
        # must see that "opposing_evidence" is blank rather than guess it.
        self.assertIn("- opposing_evidence: (empty)", block)

    def test_untitled_populated_question_still_renders(self) -> None:
        question = ReaderQuestion(question_id="q8", title="", fields={"current_view": "x"})
        self.assertIn("q8: (untitled)", render_questions_block([question]))

    def test_returns_empty_string_when_nothing_is_populated(self) -> None:
        empty = parse_question_document("# Q1: untouched\n", fallback_id="q1")
        self.assertEqual(render_questions_block([empty, empty]), "")
        self.assertEqual(render_questions_block([]), "")


class LoadQuestionsTest(unittest.TestCase):
    def test_missing_directory_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope"
            with self.assertRaises(ReaderQuestionError) as ctx:
                load_questions(missing)
            self.assertIn("not found", str(ctx.exception))

    def test_directory_without_question_documents_raises(self) -> None:
        """A README-only directory is a deployment fault, not an empty model."""
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "README.md").write_text("no questions here\n", encoding="utf-8")
            with self.assertRaises(ReaderQuestionError) as ctx:
                load_questions(Path(tmp))
            self.assertIn("q*.md", str(ctx.exception))

    def test_loads_sorted_by_question_id_with_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "q2.md").write_text("# Q2: second\n", encoding="utf-8")
            (root / "q1.md").write_text("# Q1: first\n", encoding="utf-8")
            questions = load_questions(root)
        self.assertEqual([q.question_id for q in questions], ["q1", "q2"])
        self.assertEqual([q.path.name for q in questions], ["q1.md", "q2.md"])


# ---------------------------------------------------------------------------
# The human-only invariant
# ---------------------------------------------------------------------------

# (regex, human name, a snippet that MUST match it). The samples are the reason
# this test can fail: they prove the scanner still recognises a writer. Without
# them a typo in any regex would silently downgrade this test to a no-op.
_WRITE_PATTERNS = (
    (r"\.write_text\s*\(", "Path.write_text", 'p.write_text("x", encoding="utf-8")'),
    (r"\.write_bytes\s*\(", "Path.write_bytes", "p.write_bytes(b'x')"),
    (r"\.write\s*\(", "file.write", "handle.write('x')"),
    (r"\.writelines\s*\(", "file.writelines", "handle.writelines(rows)"),
    (
        r"\bopen\s*\([^)]*mode\s*=\s*[\"'][^\"']*[wax+]",
        "open(mode=...)",
        'open(path, mode="w", encoding="utf-8")',
    ),
    (
        r"\bopen\s*\([^)]*,\s*[\"'][^\"']*[wax+][^\"']*[\"']",
        "open(..., 'w')",
        'open(path, "a", encoding="utf-8")',
    ),
    (r"\bos\.replace\s*\(", "os.replace", "os.replace(tmp, dst)"),
    (r"\bos\.rename\s*\(", "os.rename", "os.rename(tmp, dst)"),
    (r"\bos\.remove\s*\(|\bos\.unlink\s*\(|\.unlink\s*\(", "unlink", "path.unlink()"),
    (r"\bos\.makedirs\s*\(|\.mkdir\s*\(", "mkdir", "path.mkdir(parents=True)"),
    (r"\bshutil\b", "shutil", "import shutil"),
    (r"\btempfile\b", "tempfile", "import tempfile"),
)

_READER_SOURCES = ("questions.py", "delta.py")


class HumanOnlyInvariantTest(unittest.TestCase):
    """Neither reader module may contain any file-writing call, ever.

    The question documents are hand-maintained. The moment any code in this
    package can write, "the agent edited my open questions and I did not notice"
    becomes possible, and there is no way to notice it after the fact.
    """

    def test_the_scanner_can_actually_detect_a_writer(self) -> None:
        """Positive control. A detector that matches nothing prints the same green
        as a package that writes nothing."""
        for pattern, name, sample in _WRITE_PATTERNS:
            with self.subTest(pattern=name):
                self.assertRegex(sample, pattern)

    def test_reader_sources_contain_no_write_call(self) -> None:
        offences = []
        for filename in _READER_SOURCES:
            path = READER_PKG / filename
            self.assertTrue(path.is_file(), f"missing reader source: {path}")
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                code = line.split("#", 1)[0]
                for pattern, name, _sample in _WRITE_PATTERNS:
                    if re.search(pattern, code):
                        offences.append(f"{filename}:{lineno}: {name}: {line.strip()}")
        self.assertEqual(
            offences,
            [],
            "the reader package must never write to disk; found:\n" + "\n".join(offences),
        )


if __name__ == "__main__":
    unittest.main()
