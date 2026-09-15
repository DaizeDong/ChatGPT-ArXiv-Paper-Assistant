"""Ask a question of the archive: BM25 retrieval over papers + hotspots, then a cited answer."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arxiv_assistant.reader.retrieval import BM25Index, Document, ScoredDocument
from arxiv_assistant.utils import llm_gateway
from arxiv_assistant.utils.agent_runner import AgentError
from arxiv_assistant.utils.hotspot.hotspot_config import load_repo_config, repo_root
from arxiv_assistant.utils.llm_client import resolve_agent_model
from arxiv_assistant.utils.console import force_utf8_stdio
from arxiv_assistant.utils.local_env import load_local_env
from arxiv_assistant.utils.prompt_loader import read_prompt

load_local_env()

REPO_ROOT = repo_root()

DEFAULT_TOP_K = 10
DEFAULT_TIMEOUT_S = 240
MAX_DOC_CHARS = 1600  # per document, in the synthesis prompt

# Structural envelope only. The real contract is enforced by _verify_agent_answer.
AGENT_SCHEMA: Dict[str, Any] = {
    "required": ["conclusions"],
    "properties": {"conclusions": {"type": "array"}},
}


class AnswerStatus:
    """See the module docstring. These are strings, not an Enum, because they land
    verbatim in ``--json`` output that other tools parse."""

    ANSWERED = "answered"
    NO_HITS = "no_hits"
    NO_DATA = "no_data"
    UNAVAILABLE = "unavailable"
    REJECTED = "rejected"
    SKIPPED = "skipped"


FAILURE_STATUSES = frozenset({AnswerStatus.NO_DATA, AnswerStatus.UNAVAILABLE, AnswerStatus.REJECTED})


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


@dataclass
class Coverage:
    """What the window actually contained. Reported on EVERY answer, including the
    good ones, so a thin answer can always be traced to a thin window."""

    since: str = ""
    until: str = ""
    days_in_window: int = 0
    paper_days_present: int = 0
    paper_days_missing: int = 0
    paper_days_empty: int = 0
    hotspot_days_present: int = 0
    hotspot_days_missing: int = 0
    paper_docs: int = 0
    hotspot_docs: int = 0
    unreadable_files: List[str] = field(default_factory=list)

    @property
    def total_docs(self) -> int:
        return self.paper_docs + self.hotspot_docs

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["total_docs"] = self.total_docs
        return data

    def summary_line(self) -> str:
        return (
            f"window {self.since}..{self.until}: {self.days_in_window} days, "
            f"{self.total_docs} docs ({self.paper_docs} paper / {self.hotspot_docs} hotspot); "
            f"papers {self.paper_days_present} present, {self.paper_days_missing} missing, "
            f"{self.paper_days_empty} present-but-empty; "
            f"hotspots {self.hotspot_days_present} present, {self.hotspot_days_missing} missing"
        )


def _read_json(path: Path, coverage: Coverage) -> Optional[Any]:
    """Read one archive file. A corrupt file is RECORDED, never swallowed: it shows up
    in ``coverage.unreadable_files`` so a shrinking corpus cannot pass as a quiet week."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        coverage.unreadable_files.append(f"{path.name}: {exc}")
        return None


def _clean(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(_clean(item) for item in value if item)
    return " ".join(str(value or "").split())


def _paper_documents(mapping: Mapping[str, Any], day: str) -> List[Document]:
    docs: List[Document] = []
    for arxiv_id, entry in (mapping or {}).items():
        if not isinstance(entry, Mapping):
            continue
        title = _clean(entry.get("title"))
        text = " ".join(
            part
            for part in (title, _clean(entry.get("abstract")), _clean(entry.get("COMMENT")))
            if part
        )
        if not text:
            continue
        docs.append(
            Document(
                doc_id=f"paper:{arxiv_id}",
                kind="paper",
                date=day,
                title=title or str(arxiv_id),
                text=text,
                url=f"https://arxiv.org/abs/{arxiv_id}",
            )
        )
    return docs


def _iter_report_topics(report: Mapping[str, Any]):
    """Every topic in the day, not just the ones that survived selection.

    ``featured_topics`` is the digest's shortlist; the interesting long tail lives in
    ``category_sections`` / ``long_tail_sections`` / ``watchlist``. Older archived
    reports lack some of these keys entirely, hence .get() everywhere.
    """
    for key in ("featured_topics", "watchlist"):
        for topic in report.get(key) or []:
            if isinstance(topic, Mapping):
                yield topic
    for key in ("category_sections", "long_tail_sections"):
        for section in report.get(key) or []:
            if not isinstance(section, Mapping):
                continue
            for topic in section.get("topics") or []:
                if isinstance(topic, Mapping):
                    yield topic


def _hotspot_documents(report: Mapping[str, Any], day: str) -> List[Document]:
    docs: List[Document] = []
    seen: set = set()
    for topic in _iter_report_topics(report):
        topic_id = str(topic.get("TOPIC_ID") or topic.get("cluster_id") or "").strip()
        if not topic_id or topic_id in seen:
            continue  # dedupe within the day: featured topics repeat inside sections
        seen.add(topic_id)
        title = _clean(topic.get("HEADLINE") or topic.get("title"))
        text = " ".join(
            part
            for part in (
                title,
                _clean(topic.get("summary")),
                _clean(topic.get("WHY_IT_MATTERS")),
                _clean(topic.get("KEY_TAKEAWAYS")),
            )
            if part
        )
        if not text:
            continue
        items = topic.get("items") or []
        url = ""
        if isinstance(items, list) and items and isinstance(items[0], Mapping):
            url = _clean(items[0].get("url") or items[0].get("canonical_url"))
        docs.append(
            Document(
                doc_id=f"hotspot:{day}:{topic_id}",
                kind="hotspot",
                date=day,
                title=title or topic_id,
                text=text,
                url=url,
            )
        )
    return docs


def load_corpus(archive_root: Path, since: date, until: date) -> tuple[List[Document], Coverage]:
    """Walk every calendar day in [since, until] across BOTH archive trees."""
    coverage = Coverage(since=since.isoformat(), until=until.isoformat())
    documents: List[Document] = []
    json_root = archive_root / "out" / "json"
    hot_root = archive_root / "out" / "hot" / "reports"

    day = since
    while day <= until:
        coverage.days_in_window += 1
        iso = day.isoformat()

        paper_path = json_root / day.strftime("%Y-%m") / f"{iso}-output.json"
        if paper_path.is_file():
            mapping = _read_json(paper_path, coverage)
            if isinstance(mapping, Mapping) and mapping:
                coverage.paper_days_present += 1
                docs = _paper_documents(mapping, iso)
                coverage.paper_docs += len(docs)
                documents.extend(docs)
            elif isinstance(mapping, Mapping):
                coverage.paper_days_empty += 1  # the literal "{}" -- an outage fingerprint
        else:
            coverage.paper_days_missing += 1

        hot_path = hot_root / f"{iso}.json"
        if hot_path.is_file():
            report = _read_json(hot_path, coverage)
            if isinstance(report, Mapping):
                coverage.hotspot_days_present += 1
                docs = _hotspot_documents(report, iso)
                coverage.hotspot_docs += len(docs)
                documents.extend(docs)
        else:
            coverage.hotspot_days_missing += 1

        day += timedelta(days=1)

    return documents, coverage


# ---------------------------------------------------------------------------
# Synthesis: agent proposes, deterministic verifier disposes (INV6)
# ---------------------------------------------------------------------------


def _normalize_url(url: str) -> str:
    return str(url or "").strip().rstrip("/").casefold()


def build_prompt(question: str, hits: Sequence[ScoredDocument]) -> str:
    template = read_prompt("reader.archive_query")
    blocks: List[str] = []
    for hit in hits:
        doc = hit.document
        text = doc.text if len(doc.text) <= MAX_DOC_CHARS else doc.text[: MAX_DOC_CHARS - 3] + "..."
        blocks.append(
            "\n".join(
                [
                    f"### {doc.doc_id}",
                    f"- date: {doc.date}",
                    f"- kind: {doc.kind}",
                    f"- title: {doc.title}",
                    f"- url: {doc.url or '(no url -- this document CANNOT be cited)'}",
                    f"- text: {text}",
                    "",
                ]
            )
        )
    return template.replace("{question}", question.strip()).replace(
        "{documents}", "\n".join(blocks).strip()
    )


def _verify_agent_answer(
    payload: Any, hits: Sequence[ScoredDocument]
) -> Optional[List[Dict[str, Any]]]:
    """Deterministic verifier (INV6), same shape as ``paper_filter._verify_agent_response``."""
    if not isinstance(payload, Mapping):
        return None
    raw_conclusions = payload.get("conclusions")
    if not isinstance(raw_conclusions, list):
        return None

    allowed = {_normalize_url(hit.document.url): hit.document for hit in hits if hit.document.url}
    clean: List[Dict[str, Any]] = []
    claimed = 0
    for row in raw_conclusions:
        if not isinstance(row, Mapping):
            continue
        text = _clean(row.get("text"))
        if not text:
            continue
        claimed += 1
        raw_citations = row.get("citations")
        if not isinstance(raw_citations, list):
            raw_citations = []
        kept: List[Dict[str, str]] = []
        seen: set = set()
        for url in raw_citations:
            if not isinstance(url, str):
                continue
            key = _normalize_url(url)
            doc = allowed.get(key)
            if doc is None or key in seen:
                continue
            seen.add(key)
            kept.append({"url": doc.url, "doc_id": doc.doc_id, "date": doc.date, "title": doc.title})
        if not kept:
            continue  # unsourced claim -> deleted
        clean.append({"text": text, "citations": kept})

    if claimed and not clean:
        return None
    return clean


def _result_payload(result: Any) -> Any:
    """The dict a gateway result carries, from ``.data`` or from JSON in ``.text``.

    The agent transport parses into ``.data``; llmcall may leave the JSON in
    ``.text``. Anything that is neither comes back as-is and the verifier rejects
    it, which is the correct outcome: an unparseable answer is not an empty one.
    """
    if isinstance(result.data, Mapping):
        return result.data
    text = str(getattr(result, "text", "") or "").strip()
    if text:
        try:
            parsed = json.loads(text)
        except ValueError:
            return result.data
        return parsed
    return result.data


def synthesize(
    question: str,
    hits: Sequence[ScoredDocument],
    *,
    model: Optional[str] = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    config: Any = None,
    backend: Optional[str] = None,
    agent_fn: Optional[Callable[..., Any]] = None,
    llmcall_fn: Optional[Callable[..., Any]] = None,
    call_fn: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Run the model over the retrieved slice and verify what comes back."""
    if not hits:
        return {"status": AnswerStatus.NO_HITS, "conclusions": [], "note": "retrieval returned no hits"}

    chosen = backend
    if chosen is None and agent_fn is not None:
        chosen = llm_gateway.BACKEND_AGENT
    if chosen is None and llmcall_fn is not None:
        chosen = llm_gateway.BACKEND_LLMCALL

    gateway_call = call_fn or llm_gateway.call
    try:
        result = gateway_call(
            build_prompt(question, hits),
            schema=AGENT_SCHEMA,
            config=config,
            backend=chosen,
            model=model or None,
            timeout_s=timeout_s,
            agent_fn=agent_fn,
            llmcall_fn=llmcall_fn,
        )
    except AgentError as exc:
        return {
            "status": AnswerStatus.UNAVAILABLE,
            "conclusions": [],
            "note": f"synthesis agent failed, so NO synthesis ran: {exc}",
            "backend": chosen or "",
            "provider": "",
        }
    payload = _result_payload(result)
    answered_by = {
        "backend": str(getattr(result, "backend", "") or chosen or ""),
        "provider": str(getattr(result, "provider", "") or ""),
    }
    conclusions = _verify_agent_answer(payload, hits)
    if conclusions is None:
        return {
            "status": AnswerStatus.REJECTED,
            "conclusions": [],
            "note": "verifier rejected the agent answer: malformed, or every citation was "
            "fabricated and no conclusion survived",
            **answered_by,
        }
    note = _clean(payload.get("note")) if isinstance(payload, Mapping) else ""
    if not conclusions:
        return {
            "status": AnswerStatus.NO_HITS,
            "conclusions": [],
            "note": note or "the agent read the slice and found nothing that answers the question",
            **answered_by,
        }
    return {
        "status": AnswerStatus.ANSWERED,
        "conclusions": conclusions,
        "note": note,
        **answered_by,
    }


# ---------------------------------------------------------------------------
# Callable entry point
# ---------------------------------------------------------------------------


def query_archive(
    question: str,
    *,
    since: str,
    until: Optional[str] = None,
    archive_root: str | Path,
    top_k: int = DEFAULT_TOP_K,
    use_llm: bool = True,
    model: Optional[str] = None,
    config: Any = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    backend: Optional[str] = None,
    agent_fn: Optional[Callable[..., Any]] = None,
    llmcall_fn: Optional[Callable[..., Any]] = None,
    call_fn: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Answer *question* from the archive at *archive_root*. Never raises for missing days."""
    since_date = _parse_date(since, "--since")
    until_date = _parse_date(until, "--until") if until else date.today()
    if until_date < since_date:
        raise ValueError(f"empty window: --until {until_date} is before --since {since_date}")

    root = Path(archive_root).expanduser().resolve()
    if not (root / "out").is_dir():
        raise ValueError(
            f"archive root {root} has no out/ directory. On a code branch the repo's own "
            "out/ is gitignored and empty; point --archive-root at a data-branch checkout."
        )

    documents, coverage = load_corpus(root, since_date, until_date)
    index = BM25Index(documents)
    hits = index.search(question, top_k=top_k)

    if index.is_empty:
        answer = {
            "status": AnswerStatus.NO_DATA,
            "conclusions": [],
            "note": (
                "no documents in this window, so NOTHING was searched: "
                f"{coverage.paper_days_missing} paper days missing, "
                f"{coverage.paper_days_empty} paper days present but empty, "
                f"{coverage.hotspot_days_missing} hotspot days missing"
            ),
        }
    elif not use_llm:
        answer = {
            "status": AnswerStatus.SKIPPED,
            "conclusions": [],
            "note": "--no-llm: retrieval only, no synthesis was attempted",
        }
    else:
        chosen = backend
        if chosen is None and agent_fn is not None:
            chosen = llm_gateway.BACKEND_AGENT
        if chosen is None and llmcall_fn is not None:
            chosen = llm_gateway.BACKEND_LLMCALL
        if chosen is None:
            chosen = llm_gateway.resolve_backend(config)

        # A model id is only resolved for the backend that needs one. The llmcall
        # chain picks its own provider and model; handing it a claude model id
        # from [LLM_AGENT] would pin the wrong thing on the wrong transport.
        resolved_model = model or ""
        if chosen == llm_gateway.BACKEND_AGENT and not resolved_model:
            resolved_model = resolve_agent_model(
                config if config is not None
                else load_repo_config(REPO_ROOT / "configs" / "config.ini")
            )
        answer = synthesize(
            question,
            hits,
            model=resolved_model or None,
            timeout_s=timeout_s,
            config=config,
            backend=chosen,
            agent_fn=agent_fn,
            llmcall_fn=llmcall_fn,
            call_fn=call_fn,
        )
        answer["model"] = resolved_model
        answer.setdefault("backend", chosen)

    return {
        "question": question,
        "archive_root": str(root),
        "since": coverage.since,
        "until": coverage.until,
        "top_k": top_k,
        "coverage": coverage.to_dict(),
        "hits": [
            {
                "rank": hit.rank,
                "score": hit.score,
                "doc_id": hit.document.doc_id,
                "kind": hit.document.kind,
                "date": hit.document.date,
                "title": hit.document.title,
                "url": hit.document.url,
                "matched_terms": list(hit.matched_terms),
                "snippet": hit.document.text[:280],
            }
            for hit in hits
        ],
        "answer": answer,
    }


def _parse_date(value: str, flag: str) -> date:
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{flag} must be YYYY-MM-DD, got {value!r}") from exc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def render_text(result: Mapping[str, Any]) -> str:
    lines: List[str] = []
    coverage = result["coverage"]
    lines.append(f"question: {result['question']}")
    lines.append(
        "coverage: window {since}..{until}: {days} days, {docs} docs "
        "({papers} paper / {hot} hotspot)".format(
            since=coverage["since"],
            until=coverage["until"],
            days=coverage["days_in_window"],
            docs=coverage["total_docs"],
            papers=coverage["paper_docs"],
            hot=coverage["hotspot_docs"],
        )
    )
    lines.append(
        "          paper days: {p} present, {m} missing, {e} present-but-empty".format(
            p=coverage["paper_days_present"],
            m=coverage["paper_days_missing"],
            e=coverage["paper_days_empty"],
        )
    )
    lines.append(
        "          hotspot days: {p} present, {m} missing".format(
            p=coverage["hotspot_days_present"], m=coverage["hotspot_days_missing"]
        )
    )
    if coverage["unreadable_files"]:
        lines.append(f"          UNREADABLE FILES: {len(coverage['unreadable_files'])}")
        for entry in coverage["unreadable_files"]:
            lines.append(f"            - {entry}")

    lines.append("")
    lines.append(f"ranked hits ({len(result['hits'])}):")
    if not result["hits"]:
        lines.append("  (none)")
    for hit in result["hits"]:
        lines.append(
            f"  {hit['rank']:>2}. [{hit['score']:.3f}] {hit['date']} {hit['kind']:<7} {hit['title']}"
        )
        lines.append(f"      {hit['url'] or '(no url)'}")
        lines.append(f"      matched: {', '.join(hit['matched_terms'][:12])}")

    answer = result["answer"]
    lines.append("")
    lines.append(f"answer status: {answer['status']}")
    if answer.get("backend"):
        provider = answer.get("provider") or ""
        lines.append(
            "  answered by: backend={backend}{provider}{model}".format(
                backend=answer["backend"],
                provider=f" provider={provider}" if provider else "",
                model=f" model={answer['model']}" if answer.get("model") else "",
            )
        )
    if answer.get("note"):
        lines.append(f"  note: {answer['note']}")
    for index, conclusion in enumerate(answer.get("conclusions") or [], start=1):
        lines.append(f"  {index}. {conclusion['text']}")
        for citation in conclusion["citations"]:
            lines.append(f"     - {citation['date']} {citation['url']}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask a question of the paper + hotspot archive.")
    parser.add_argument("question", help="The question, in English or Chinese.")
    parser.add_argument("--since", required=True, help="Window start, YYYY-MM-DD (inclusive).")
    parser.add_argument("--until", default=None, help="Window end, YYYY-MM-DD (inclusive). Default: today.")
    parser.add_argument(
        "--archive-root",
        default=str(REPO_ROOT),
        help="Checkout that contains out/. Default: this repo (whose out/ is empty on code branches).",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="How many documents to retrieve.")
    parser.add_argument("--json", action="store_true", help="Emit the full result as JSON.")
    parser.add_argument("--no-llm", action="store_true", help="Retrieval only; never call the model.")
    parser.add_argument("--timeout-s", type=int, default=DEFAULT_TIMEOUT_S, help="Synthesis agent timeout.")
    return parser.parse_args()


def main() -> None:
    force_utf8_stdio()  # the archive is full of Chinese titles; see utils/console.py
    args = parse_args()
    config = load_repo_config(REPO_ROOT / "configs" / "config.ini")
    try:
        result = query_archive(
            args.question,
            since=args.since,
            until=args.until,
            archive_root=args.archive_root,
            top_k=args.top_k,
            use_llm=not args.no_llm,
            config=config,
            timeout_s=args.timeout_s,
        )
    except ValueError as exc:
        print(f"query_archive: {exc}", file=sys.stderr)
        raise SystemExit(2)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(render_text(result))

    status = result["answer"]["status"]
    if status in FAILURE_STATUSES:
        # Loud on purpose: an outage, a missing archive or a hallucinating model must
        # not exit 0 looking like a tidy "nothing to report".
        print(f"query_archive: FAILED status={status}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
