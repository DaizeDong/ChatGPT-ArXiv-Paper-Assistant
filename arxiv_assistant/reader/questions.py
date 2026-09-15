"""Reader-model question documents: load and parse ``configs/reader/questions/*.md``.

These documents are the ONLY human input to the weekly digest. Nothing in this
package, and no pipeline stage, ever writes to them (``tests/test_reader_questions.py``
guards that this module exposes no writer).

Each document has a fixed five-field shape. The headings are Chinese; the stable
machine keys are the ones in :data:`FIELD_KEYS`, and those are what the scoring
prompt and the verifier speak.

An EMPTY reader model is a first-class, reportable state -- not silently "nothing
matched". :func:`load_questions` records ``is_populated`` per question so callers
can tell "the researcher has no open questions written down" apart from "nothing
this week moved them". Conflating those two is the failure this whole package
exists to avoid.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

# Stable machine keys, in document order. The scoring prompt lists exactly these.
FIELD_KEYS: tuple[str, ...] = (
    "current_view",
    "supporting_evidence",
    "opposing_evidence",
    "would_change_my_mind",
    "experiments",
)

# Heading text -> machine key. The headings are fixed by contract; a document that
# renames one loses that field rather than silently mapping it somewhere wrong.
_HEADING_TO_KEY: Dict[str, str] = {
    "当前看法": "current_view",
    "支持证据": "supporting_evidence",
    "反对证据": "opposing_evidence",
    "什么会让我改看法": "would_change_my_mind",
    "想做的实验": "experiments",
}

_TITLE_RE = re.compile(r"^#\s*(?P<qid>[Qq]\d+)\s*[:：]\s*(?P<title>.*?)\s*$", re.M)
_HEADING_RE = re.compile(r"^##\s+(?P<heading>.+?)\s*$", re.M)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


class ReaderQuestionError(Exception):
    """Raised when the questions directory cannot be read at all.

    Deliberately NOT raised for an empty or half-filled document: a researcher who
    has not written a question yet is a normal state that callers must report,
    while a missing directory is a deployment fault.
    """


@dataclass(frozen=True)
class ReaderQuestion:
    question_id: str
    title: str
    fields: Dict[str, str] = field(default_factory=dict)
    path: Path | None = None

    @property
    def is_populated(self) -> bool:
        """True when the researcher has written ANY substantive field."""
        return any(self.fields.get(key, "").strip() for key in FIELD_KEYS)

    @property
    def populated_fields(self) -> List[str]:
        return [key for key in FIELD_KEYS if self.fields.get(key, "").strip()]


def _strip_template_noise(text: str) -> str:
    """Drop HTML comments and placeholder angle-bracket lines.

    The shipped templates carry instructional ``<!-- ... -->`` comments and
    ``<在这里写...>`` placeholders. Counting those as content would make an
    untouched template look populated, which is exactly the confusion this
    module refuses to allow.
    """
    text = _HTML_COMMENT_RE.sub("", text)
    kept = [
        line
        for line in text.splitlines()
        if not (line.strip().startswith("<") and line.strip().endswith(">"))
    ]
    return "\n".join(kept).strip()


def parse_question_document(text: str, *, fallback_id: str, path: Path | None = None) -> ReaderQuestion:
    """Parse one question markdown document into a :class:`ReaderQuestion`."""
    title_match = _TITLE_RE.search(text)
    if title_match:
        question_id = title_match.group("qid").lower()
        title = _strip_template_noise(title_match.group("title")).strip()
    else:
        question_id = fallback_id.lower()
        title = ""

    fields: Dict[str, str] = {key: "" for key in FIELD_KEYS}
    headings = list(_HEADING_RE.finditer(text))
    for index, match in enumerate(headings):
        key = _HEADING_TO_KEY.get(match.group("heading").strip())
        if key is None:
            continue
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        fields[key] = _strip_template_noise(text[match.end():end])

    return ReaderQuestion(question_id=question_id, title=title, fields=fields, path=path)


def load_questions(questions_dir: Path) -> List[ReaderQuestion]:
    """Load every ``q*.md`` under *questions_dir*, sorted by question id.

    Raises :class:`ReaderQuestionError` when the directory is missing or holds no
    question documents. A directory full of untouched templates loads fine and
    yields questions whose ``is_populated`` is False -- callers must check that.
    """
    questions_dir = Path(questions_dir)
    if not questions_dir.is_dir():
        raise ReaderQuestionError(
            f"Reader questions directory not found: {questions_dir}. "
            "Create it (see configs/reader/questions/README.md) or set "
            "[READER] enabled = false."
        )

    paths = sorted(p for p in questions_dir.glob("q*.md") if p.is_file())
    if not paths:
        raise ReaderQuestionError(
            f"No question documents (q*.md) under {questions_dir}."
        )

    questions = [
        parse_question_document(
            p.read_text(encoding="utf-8"), fallback_id=p.stem, path=p
        )
        for p in paths
    ]
    return sorted(questions, key=lambda q: q.question_id)


def render_questions_block(questions: Sequence[ReaderQuestion]) -> str:
    """Render the question documents for the ``{questions}`` prompt placeholder.

    Only populated questions are rendered: handing the model five empty documents
    invites it to invent a rationale for whichever id it saw first.
    """
    chunks: List[str] = []
    for question in questions:
        if not question.is_populated:
            continue
        lines = [f"### {question.question_id}: {question.title or '(untitled)'}"]
        for key in FIELD_KEYS:
            value = question.fields.get(key, "").strip()
            lines.append(f"- {key}: {value if value else '(empty)'}")
        chunks.append("\n".join(lines))
    return "\n\n".join(chunks)
