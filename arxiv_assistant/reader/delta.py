"""Delta scoring: does this item CHANGE one of the researcher's question documents?

This is the reader model's judgment point, and it follows the repo's INV6 shape:
an LLM proposes, a deterministic verifier disposes. Nothing an agent returns is
trusted unverified.

THREE OUTCOMES, DELIBERATELY DISTINCT (see :class:`DeltaStatus`):

* ``scored``      the model returned a row and the verifier accepted it.
* ``rejected``    the model returned a row and the verifier threw it out.
* ``unavailable`` scoring never ran (no transport, no questions, agent error).

Collapsing ``unavailable`` into "score 0" is the specific bug this module refuses
to have. A weekly digest that prints "nothing changed my mind" because the model
was unreachable looks exactly like a quiet week, and the paper pipeline in this
repo silently produced empty output for three months for precisely that reason.
Callers MUST branch on status before rendering an empty digest.

The daily pipelines only ANNOTATE with these verdicts; they never drop an item.
Filtering to the ``delta_score_cutoff`` happens in the weekly digest alone, so a
story that fails the gate today is not burned out of tomorrow's candidate pool.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

from arxiv_assistant.reader.questions import FIELD_KEYS, ReaderQuestion, render_questions_block
from arxiv_assistant.utils import llm_gateway
from arxiv_assistant.utils.agent_runner import AgentError
from arxiv_assistant.utils.prompt_loader import read_prompt

DEFAULT_BATCH_SIZE = 10
DEFAULT_TIMEOUT_S = 180
MAX_REASON_WORDS = 40

# Schema handed to the gateway's structural validator. The real contract is enforced
# row by row in `verify_verdict_row`; this only rejects a malformed envelope.
AGENT_SCHEMA: Dict[str, Any] = {
    "required": ["verdicts"],
    "properties": {"verdicts": {"type": "array"}},
}


class DeltaStatus:
    SCORED = "scored"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class DeltaCandidate:
    """One thing that could move a question document, from either pipeline."""

    candidate_id: str
    kind: str  # "paper" | "hotspot"
    title: str
    text: str
    url: str = ""
    # Secondary sort key, used to break delta_score ties: NOVELTY for papers,
    # FINAL_SCORE for hotspot topics. There is no novelty field on hotspot topics.
    tiebreak: float = 0.0
    date: str = ""
    payload: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class DeltaVerdict:
    candidate_id: str
    status: str
    delta_score: int = 0
    question_id: str = ""
    field: str = ""
    one_line_reason: str = ""
    note: str = ""

    @property
    def passes(self) -> bool:
        return self.status == DeltaStatus.SCORED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "status": self.status,
            "delta_score": self.delta_score,
            "question_id": self.question_id,
            "field": self.field,
            "one_line_reason": self.one_line_reason,
            "note": self.note,
        }


@dataclass(frozen=True)
class ReaderSettings:
    enabled: bool = True
    questions_dir: str = "configs/reader/questions"
    delta_score_cutoff: int = 7
    max_deep_read: int = 5
    max_skim: int = 15
    model: str = ""
    timeout_s: int = DEFAULT_TIMEOUT_S
    annotate_in_daily_run: bool = True


def resolve_reader_settings(config: Any) -> ReaderSettings:
    """Read ``[READER]`` with the kernel's getter-with-fallback convention.

    A config that predates the section yields defaults rather than a KeyError, so
    an old config.ini (or the zero-key profile before it was updated) still runs.
    """
    defaults = ReaderSettings()
    try:
        has_section = config.has_section("READER")
    except AttributeError:
        has_section = False
    if not has_section:
        return defaults
    cfg = config["READER"]
    return ReaderSettings(
        enabled=cfg.getboolean("enabled", fallback=defaults.enabled),
        questions_dir=cfg.get("questions_dir", defaults.questions_dir) or defaults.questions_dir,
        delta_score_cutoff=cfg.getint("delta_score_cutoff", fallback=defaults.delta_score_cutoff),
        max_deep_read=cfg.getint("max_deep_read", fallback=defaults.max_deep_read),
        max_skim=cfg.getint("max_skim", fallback=defaults.max_skim),
        model=(cfg.get("model", "") or "").strip(),
        timeout_s=cfg.getint("timeout_s", fallback=defaults.timeout_s),
        annotate_in_daily_run=cfg.getboolean(
            "annotate_in_daily_run", fallback=defaults.annotate_in_daily_run
        ),
    )


# ---------------------------------------------------------------------------
# Deterministic verifier (INV6)
# ---------------------------------------------------------------------------


def verify_verdict_row(
    row: Any,
    *,
    valid_question_ids: Sequence[str],
    candidates: Sequence[DeltaCandidate],
) -> DeltaVerdict | None:
    """Return a clean verdict, or None when the row must be thrown out.

    Rejection rules, all deterministic:
      - row is not a dict, or `index` is not an in-range int
      - `question_id` is not one of the loaded documents
      - `field` is not one of the five canonical field keys
      - `delta_score` is not an int in 0..10 (bools are not ints here)
      - `one_line_reason` is empty, or is longer than MAX_REASON_WORDS
      - a high score with a reason that names nothing specific
    """
    if not isinstance(row, dict):
        return None

    index = row.get("index")
    if isinstance(index, bool) or not isinstance(index, int):
        return None
    if not (0 <= index < len(candidates)):
        return None
    candidate = candidates[index]

    question_id = str(row.get("question_id", "")).strip().lower()
    if question_id not in set(valid_question_ids):
        return _rejected(candidate, f"unknown question_id {question_id!r}")

    field_key = str(row.get("field", "")).strip()
    if field_key not in FIELD_KEYS:
        return _rejected(candidate, f"unknown field {field_key!r}")

    score = row.get("delta_score")
    if isinstance(score, bool) or not isinstance(score, int):
        return _rejected(candidate, "delta_score is not an integer")
    if not (0 <= score <= 10):
        return _rejected(candidate, f"delta_score {score} out of range")

    reason = str(row.get("one_line_reason", "")).strip()
    if not reason:
        return _rejected(candidate, "empty one_line_reason")
    if len(reason.split()) > MAX_REASON_WORDS:
        return _rejected(candidate, "one_line_reason too long")

    # A claim to have moved a field must name something. The prompt says a reason
    # that cannot name what changes caps the score at 3; enforce it rather than
    # trusting the model to have obeyed.
    if score >= 4 and _reason_is_generic(reason):
        return _rejected(candidate, "high score with a generic reason")

    return DeltaVerdict(
        candidate_id=candidate.candidate_id,
        status=DeltaStatus.SCORED,
        delta_score=score,
        question_id=question_id,
        field=field_key,
        one_line_reason=reason,
    )


_GENERIC_REASON_MARKERS = (
    "relevant to",
    "useful background",
    "related to the topic",
    "generally interesting",
    "may be of interest",
    "aligns with",
)


def _reason_is_generic(reason: str) -> bool:
    lowered = reason.lower()
    if any(marker in lowered for marker in _GENERIC_REASON_MARKERS):
        return True
    # Fewer than four words cannot name a specific result.
    return len(lowered.split()) < 4


def _rejected(candidate: DeltaCandidate, note: str) -> DeltaVerdict:
    return DeltaVerdict(
        candidate_id=candidate.candidate_id,
        status=DeltaStatus.REJECTED,
        note=note,
    )


def _unavailable(candidates: Iterable[DeltaCandidate], note: str) -> Dict[str, DeltaVerdict]:
    return {
        c.candidate_id: DeltaVerdict(
            candidate_id=c.candidate_id, status=DeltaStatus.UNAVAILABLE, note=note
        )
        for c in candidates
    }


# ---------------------------------------------------------------------------
# Prompt assembly and scoring
# ---------------------------------------------------------------------------


def build_prompt(candidates: Sequence[DeltaCandidate], questions: Sequence[ReaderQuestion]) -> str:
    template = read_prompt("reader.delta_scoring")
    item_lines: List[str] = []
    for index, candidate in enumerate(candidates):
        item_lines.append(f"### index {index}")
        item_lines.append(f"- kind: {candidate.kind}")
        item_lines.append(f"- title: {candidate.title}")
        if candidate.url:
            item_lines.append(f"- url: {candidate.url}")
        item_lines.append(f"- text: {_truncate(candidate.text, 1800)}")
        item_lines.append("")
    return template.replace("{questions}", render_questions_block(questions)).replace(
        "{items}", "\n".join(item_lines).strip()
    )


def _truncate(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _verdict_rows(result: Any) -> List[Any] | None:
    """Pull the ``verdicts`` array out of a gateway result, or None.

    The two backends hand back the payload differently: the agent transport
    returns the parsed dict as ``.data``, while llmcall may put a JSON string in
    ``.text`` when its own schema pass did not populate ``.data``. Both shapes
    have to reach the same verifier, and anything else has to come back as None
    so the caller can say "scoring produced nothing usable" rather than
    "nothing scored".
    """
    data = result.data if isinstance(result.data, Mapping) else None
    if data is None:
        text = str(getattr(result, "text", "") or "").strip()
        if text:
            try:
                parsed = json.loads(text)
            except ValueError:
                parsed = None
            if isinstance(parsed, Mapping):
                data = parsed
    if data is None:
        return None
    rows = data.get("verdicts")
    return rows if isinstance(rows, list) else None


def score_candidates(
    candidates: Sequence[DeltaCandidate],
    questions: Sequence[ReaderQuestion],
    *,
    model: str,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    batch_size: int = DEFAULT_BATCH_SIZE,
    config: Any = None,
    backend: str | None = None,
    agent_fn: Callable[..., Dict[str, Any]] | None = None,
    llmcall_fn: Callable[..., Any] | None = None,
    call_fn: Callable[..., Any] | None = None,
) -> Dict[str, DeltaVerdict]:
    """Score every candidate. Returns candidate_id -> verdict, one per candidate.

    Never raises for a transport failure: the affected candidates come back with
    status ``unavailable`` so the caller can say so out loud.

    THE TRANSPORT SEAM. Calls go through :func:`arxiv_assistant.utils.llm_gateway.call`,
    which resolves the backend (llmcall chain, or this repo's ``claude -p``
    runner) and records every attempt in the module-level ledger. Three injection
    points survive for tests, and they are NOT interchangeable:

    ``agent_fn``   a stand-in for the repo's agent runner. Passing it PINS the
                   backend to ``agent``, because a test that hands over a fake
                   transport must not have its fake bypassed by whatever the host
                   machine happens to have installed. A suite that silently
                   resolved to llmcall would hit the network.
    ``llmcall_fn`` the same, for the llmcall side.
    ``call_fn``    replaces the gateway entrypoint wholesale.

    Production passes none of them and lets ``[LLM] backend`` decide.
    """
    if not candidates:
        return {}

    populated = [q for q in questions if q.is_populated]
    if not populated:
        return _unavailable(
            candidates,
            "reader model is empty: no question document under [READER] questions_dir "
            "has any content, so nothing can be scored against it",
        )

    valid_ids = [q.question_id for q in populated]
    verdicts: Dict[str, DeltaVerdict] = {}
    gateway_call = call_fn or llm_gateway.call
    chosen_backend = backend
    if chosen_backend is None and agent_fn is not None:
        chosen_backend = llm_gateway.BACKEND_AGENT
    if chosen_backend is None and llmcall_fn is not None:
        chosen_backend = llm_gateway.BACKEND_LLMCALL

    for start in range(0, len(candidates), max(1, batch_size)):
        batch = list(candidates[start : start + max(1, batch_size)])
        try:
            result = gateway_call(
                build_prompt(batch, populated),
                schema=AGENT_SCHEMA,
                config=config,
                backend=chosen_backend,
                model=model or None,
                timeout_s=timeout_s,
                agent_fn=agent_fn,
                llmcall_fn=llmcall_fn,
            )
        except AgentError as exc:
            verdicts.update(_unavailable(batch, f"delta agent failed: {exc}"))
            continue

        rows = _verdict_rows(result)
        if rows is None:
            verdicts.update(_unavailable(batch, "agent returned no verdicts array"))
            continue

        seen_ids: set[str] = set()
        for row in rows:
            verdict = verify_verdict_row(
                row, valid_question_ids=valid_ids, candidates=batch
            )
            if verdict is None:
                continue
            # Last write wins is wrong here: a duplicate index means the model
            # scored one item twice and skipped another. Keep the first.
            if verdict.candidate_id in seen_ids:
                continue
            seen_ids.add(verdict.candidate_id)
            verdicts[verdict.candidate_id] = verdict

        for candidate in batch:
            if candidate.candidate_id not in verdicts:
                verdicts[candidate.candidate_id] = DeltaVerdict(
                    candidate_id=candidate.candidate_id,
                    status=DeltaStatus.REJECTED,
                    note="no verdict row returned for this item",
                )

    return verdicts


# ---------------------------------------------------------------------------
# Candidate extraction from the two archive shapes
# ---------------------------------------------------------------------------


def candidates_from_paper_mapping(
    mapping: Mapping[str, Mapping[str, Any]], *, date: str = ""
) -> List[DeltaCandidate]:
    """Build candidates from a ``<date>-output.json`` style arxiv_id -> entry map."""
    candidates: List[DeltaCandidate] = []
    for arxiv_id, entry in (mapping or {}).items():
        if not isinstance(entry, Mapping):
            continue
        title = str(entry.get("title", "") or "")
        abstract = str(entry.get("abstract", "") or "")
        comment = str(entry.get("COMMENT", "") or "")
        text = abstract if abstract else comment
        if comment and abstract:
            text = f"{abstract}\n\nFilter comment: {comment}"
        candidates.append(
            DeltaCandidate(
                candidate_id=f"paper:{arxiv_id}",
                kind="paper",
                title=title,
                text=text,
                url=f"https://arxiv.org/abs/{arxiv_id}",
                tiebreak=float(entry.get("NOVELTY", 0) or 0),
                date=date,
                payload=entry,
            )
        )
    return candidates


def candidates_from_hotspot_report(report: Mapping[str, Any]) -> List[DeltaCandidate]:
    """Build candidates from one ``out/hot/reports/<date>.json``.

    Reads the SELECT survivors (``featured_topics``) plus ``watchlist``. It does
    not walk category_sections/long_tail_sections: those are the archive tail, and
    pulling them in would make the weekly scoring cost scale with the whole day.
    """
    date = str(report.get("date", "") or "")
    seen: set[str] = set()
    candidates: List[DeltaCandidate] = []
    for bucket in ("featured_topics", "watchlist"):
        for topic in report.get(bucket) or []:
            if not isinstance(topic, Mapping):
                continue
            topic_id = str(topic.get("TOPIC_ID") or topic.get("cluster_id") or "")
            if not topic_id or topic_id in seen:
                continue
            seen.add(topic_id)
            title = str(topic.get("HEADLINE") or topic.get("title") or "")
            parts = [
                str(topic.get("summary", "") or ""),
                str(topic.get("WHY_IT_MATTERS", "") or ""),
                " ".join(str(t) for t in (topic.get("KEY_TAKEAWAYS") or [])),
            ]
            items = topic.get("items") or []
            url = ""
            if items and isinstance(items[0], Mapping):
                url = str(items[0].get("url") or items[0].get("canonical_url") or "")
            candidates.append(
                DeltaCandidate(
                    candidate_id=f"hotspot:{date}:{topic_id}",
                    kind="hotspot",
                    title=title,
                    text="\n".join(p for p in parts if p.strip()),
                    url=url,
                    tiebreak=float(topic.get("FINAL_SCORE", 0) or 0),
                    date=date,
                    payload=topic,
                )
            )
    return candidates


def annotate(candidate: DeltaCandidate, verdict: DeltaVerdict) -> DeltaCandidate:
    """Attach a verdict to a candidate's payload copy without mutating the source."""
    payload = dict(candidate.payload or {})
    payload["DELTA"] = verdict.to_dict()
    return replace(candidate, payload=payload)


def rank(
    candidates: Sequence[DeltaCandidate], verdicts: Mapping[str, DeltaVerdict]
) -> List[DeltaCandidate]:
    """Sort by delta_score, then by the pipeline's own score. Highest first."""

    def key(candidate: DeltaCandidate) -> tuple:
        verdict = verdicts.get(candidate.candidate_id)
        score = verdict.delta_score if verdict and verdict.passes else -1
        return (score, candidate.tiebreak)

    return sorted(candidates, key=key, reverse=True)


def dumps_verdicts(verdicts: Mapping[str, DeltaVerdict]) -> str:
    return json.dumps(
        {cid: v.to_dict() for cid, v in sorted(verdicts.items())},
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
