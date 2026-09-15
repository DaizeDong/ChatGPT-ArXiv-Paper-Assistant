"""Render the weekly delta digest to markdown.

WHY THIS IS NOT A DAILY RENDERER
--------------------------------
The daily pages answer "what happened". They are source-first (``## Source
Stats``) and category-first (``## Topic Radar By Category``, ``## Long-tail
Signals``, the paper side's topic-coverage table and per-topic table of
contents), because an archive page has to be browsable by anyone, on any axis,
forever. This file answers a different question -- "what should I read, and why
does it move *my* open questions" -- so it carries none of those blocks. It is
two ranked flat lists and nothing else. The daily renderers are untouched by
this module; it does not import them and does not share their helpers, so their
output stays byte-identical.

WHY THE EMPTY CASE IS MOST OF THIS FILE
---------------------------------------
An empty digest has six possible causes and only ONE of them is good news:

1. the researcher has not written any question document yet;
2. delta scoring could not run (no transport / agent error) -- ``DeltaStatus.UNAVAILABLE``;
3. no archive day in the window exists at all (wrong ``--archive-root``, or a dead pipeline);
4. the paper pipeline scanned papers but emitted zero of them with zero LLM
   tokens spent -- the live three-month outage this repo actually suffered, where
   every OpenAI call raised into a bare ``except``, every run exited 0, and every
   day's archive was the literal two bytes ``{}``;
5. every model call failed -- the gateway ledger says N attempted, 0 succeeded,
   which is the backend-agnostic replacement for the token-count test in case 4
   (the llmcall chain reports no OpenAI tokens at all, so that test alone would
   go blind the moment the backend changed);
6. scoring genuinely ran against a real reader model and nothing cleared the cutoff.

Only case 6 may print the calm one-liner :data:`QUIET_WEEK_LINE`. Cases 1 to 5
print what is wrong, loudly, at the top of the page. A digest that renders a
tidy "nothing to report" while the machinery underneath is dead is the exact
failure mode this whole feature was built to make impossible.

The states are NOT re-derived here. The script hands them over in
``digest["diagnostics"]``; this renderer only reads and reports them, so there is
one place where "why is it empty" is computed and one place where it is shown.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

# The single sentence that is allowed to stand for an empty week, and only when
# every diagnostic says the machinery ran and found nothing worth surfacing.
QUIET_WEEK_LINE = "本周没有改变看法的内容"

# Section headings. Deep read is the short list you are expected to actually
# open; skim is the tail that cleared the cutoff but not the deep-read cap.
DEEP_READ_HEADING = "深读"
SKIM_HEADING = "略读"

_KIND_LABELS = {"paper": "论文", "hotspot": "热点"}


def _diagnostics(digest: Mapping[str, Any]) -> Mapping[str, Any]:
    diagnostics = digest.get("diagnostics")
    return diagnostics if isinstance(diagnostics, Mapping) else {}


def _int(value: Any, default: int = 0) -> int:
    """Coerce a diagnostics number without turning a missing key into a zero.

    Callers that care about "missing" pass ``default=-1``: a diagnostics block
    that lost a key must not be readable as a confident zero, because zero is
    itself one of the alarm conditions.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return int(value)


def _status_counts(diagnostics: Mapping[str, Any]) -> Dict[str, int]:
    raw = diagnostics.get("verdict_status_counts")
    raw = raw if isinstance(raw, Mapping) else {}
    return {
        "scored": _int(raw.get("scored")),
        "rejected": _int(raw.get("rejected")),
        "unavailable": _int(raw.get("unavailable")),
    }


def _escape_link_text(text: str) -> str:
    """Keep a title with brackets from breaking out of its markdown link."""
    return str(text or "").replace("[", "\\[").replace("]", "\\]").strip()


def _entry_lines(index: int, item: Mapping[str, Any]) -> List[str]:
    """One ranked entry: score, the question it moves, the reason, the link.

    Everything a reader needs to decide whether to open it, and nothing about
    which category or which source it came from -- those axes belong to the
    daily archive pages.
    """
    title = _escape_link_text(item.get("title")) or "(无标题)"
    url = str(item.get("url") or "").strip()
    headline = f"[{title}]({url})" if url else title

    score = _int(item.get("delta_score"))
    question_id = str(item.get("question_id") or "?")
    field_key = str(item.get("field") or "?")
    badges = f"`delta {score}` `{question_id}` `{field_key}`"

    kind = str(item.get("kind") or "")
    kind_label = _KIND_LABELS.get(kind, kind or "未知来源")
    date = str(item.get("date") or "").strip() or "日期缺失"

    lines = [f"{index}. **{headline}** {badges}", f"   - {date} · {kind_label}"]
    reason = str(item.get("one_line_reason") or "").strip()
    if reason:
        lines.append(f"   - {reason}")
    return lines


def _section(heading: str, items: Sequence[Mapping[str, Any]]) -> List[str]:
    if not items:
        return []
    lines = [f"## {heading}（{len(items)}）", ""]
    for index, item in enumerate(items, start=1):
        lines.extend(_entry_lines(index, item))
    lines.append("")
    return lines


def _outage_block(diagnostics: Mapping[str, Any]) -> List[str] | None:
    """Return the loud block explaining an empty digest, or None if it is honest.

    Order matters and is from most upstream to least: a missing reader model
    makes every downstream count meaningless, and an unreachable agent makes the
    archive counts meaningless in turn. Reporting the first cause in that chain
    is the one that is actually actionable.
    """
    populated = _int(diagnostics.get("reader_questions_populated"), default=-1)
    statuses = _status_counts(diagnostics)
    days_found = _int(diagnostics.get("days_found"), default=-1)
    days_requested = _int(diagnostics.get("days_requested"), default=-1)

    if populated <= 0:
        return [
            "> **本周摘要无法出具：阅读模型是空的。**",
            ">",
            "> `[READER] questions_dir` 下的问题文档一个都没有填写内容，"
            "所以没有任何“看法”可以被改变，打分根本无从谈起。",
            ">",
            "> 这不等于本周没有值得看的东西，只等于还没人写下自己在想什么。"
            "先填 `configs/reader/questions/q*.md` 中至少一份的正文，再重跑本脚本。",
        ]

    # NEW CAUSE, added when every model call started going through
    # utils/llm_gateway: the ledger can prove that no provider in the chain
    # answered. It is checked BEFORE the unavailable count because it names WHY
    # those verdicts are unavailable, which is the actionable half. An ABSENT
    # ledger is not an alarm: it only means the digest was built before this
    # block existed, and the unavailable count below still covers that case.
    ledger = diagnostics.get("llm_ledger")
    if isinstance(ledger, Mapping):
        attempted = _int(ledger.get("attempted"))
        succeeded = _int(ledger.get("succeeded"))
        if attempted > 0 and succeeded == 0:
            errors = ledger.get("errors")
            errors = [str(e) for e in errors] if isinstance(errors, (list, tuple)) else []
            backend = str(diagnostics.get("llm_backend") or "").strip() or "未知"
            return [
                "> **本周摘要不可信：模型调用链一次都没成功。**",
                ">",
                f"> 本次共发起 {attempted} 次模型调用，成功 0 次。"
                f"当前后端：`{backend}`。",
                ">",
                "> 也就是说，下面的内容不是“评估过之后没有值得看的”，"
                "而是**根本没有被评估过**。",
                ">",
                f"> 首个错误：{errors[0] if errors else '(未记录)'}",
            ]

    if statuses["unavailable"] > 0:
        return [
            "> **本周摘要不可信：delta 打分没能跑完。**",
            ">",
            f"> {statuses['unavailable']} 个候选的判定状态是 "
            "`unavailable`（打分器不可达、超时或返回了无法解析的结果），"
            "它们既没有通过也没有被否决，而是从未被评估。",
            ">",
            "> 这些候选里可能正有改变看法的内容。**不要把这一页读成“本周平静”。**"
            "查 `[READER] model` / agent 传输链路，修好后重跑。",
        ]

    if days_found == 0:
        return [
            "> **本周摘要无法出具：窗口内一天归档都没找到。**",
            ">",
            f"> 请求了 {days_requested} 天，实际读到 0 天。"
            "常见原因是 `--archive-root` 指错了目录（代码分支上的 `out/` 是空的，"
            "真实归档在数据分支），或者上游流水线已经停了。",
            ">",
            "> 空窗口和平静的一周不是一回事，所以这里不打印“没有改变看法的内容”。",
        ]

    scanned = _int(diagnostics.get("paper_scanned_seen"))
    papers = _int(diagnostics.get("paper_papers_seen"))
    tokens = _int(diagnostics.get("paper_llm_tokens_seen"))
    empty_days = diagnostics.get("paper_days_empty")
    empty_days = list(empty_days) if isinstance(empty_days, (list, tuple)) else []
    if scanned > 0 and papers == 0 and tokens == 0:
        return [
            "> **论文侧本周疑似整体故障，不是“本周没有好论文”。**",
            ">",
            f"> 窗口内共扫描 {scanned} 篇 arXiv 论文，最终产出 0 篇，"
            f"而 LLM 用量是 0 token。扫了却一个 token 都没花，"
            "说明每次模型调用都失败并被吞掉了，退出码依然是 0。",
            ">",
            f"> 空归档日（`output.json` 内容为 `{{}}`）：{', '.join(empty_days) or '无'}。",
            ">",
            "> 热点侧的结果（如有）仍然列在下面，但论文侧这一周等于没有候选。",
        ]

    return None


def _header_lines(digest: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> List[str]:
    week_start = str(digest.get("week_start") or "").strip() or "?"
    week_end = str(digest.get("week_end") or "").strip() or "?"
    lines = [f"# 每周 Delta 摘要 {week_start} ~ {week_end}", ""]

    days_found = _int(diagnostics.get("days_found"), default=-1)
    days_requested = _int(diagnostics.get("days_requested"), default=-1)
    cutoff = _int(diagnostics.get("delta_score_cutoff"), default=-1)
    scored = _int(diagnostics.get("candidates_scored"), default=-1)
    meta_bits = []
    if days_requested >= 0:
        meta_bits.append(f"归档 {days_found}/{days_requested} 天")
    if scored >= 0:
        meta_bits.append(f"候选 {scored} 条")
    if cutoff >= 0:
        meta_bits.append(f"cutoff delta >= {cutoff}")
    backend = str(diagnostics.get("llm_backend") or "").strip()
    if backend:
        meta_bits.append(backend)
    if meta_bits:
        lines.extend([" · ".join(meta_bits), ""])
    return lines


def render_weekly_digest_md(digest: Dict[str, Any]) -> str:
    """Render one weekly digest dict (see ``scripts/generate_weekly_digest.py``).

    Expected shape::

        {
          "week_start": "YYYY-MM-DD", "week_end": "YYYY-MM-DD",
          "deep_read": [ {candidate_id, kind, title, url, date, delta_score,
                          question_id, field, one_line_reason, tiebreak}, ... ],
          "skim":      [ ...same... ],
          "diagnostics": { days_requested, days_found, days_missing,
                           paper_days_empty, paper_llm_tokens_seen,
                           paper_scanned_seen, paper_papers_seen,
                           candidates_scored, verdict_status_counts,
                           reader_questions_populated,
                           delta_score_cutoff, max_deep_read, max_skim,
                           llm_backend, llm_ledger }
        }

    Missing sub-dicts degrade to "unknown", never to a confident zero: a
    diagnostics block that lost ``reader_questions_populated`` renders the loud
    empty-reader-model message rather than the calm one-liner, because a digest
    that cannot prove it ran must not claim it ran.
    """
    diagnostics = _diagnostics(digest)
    deep_read = [item for item in (digest.get("deep_read") or []) if isinstance(item, Mapping)]
    skim = [item for item in (digest.get("skim") or []) if isinstance(item, Mapping)]

    lines = _header_lines(digest, diagnostics)

    outage = _outage_block(diagnostics)
    if outage is not None:
        lines.extend(outage)
        lines.append("")

    if not deep_read and not skim:
        if outage is None:
            # The one honest quiet week: a real reader model, scoring that ran to
            # completion on real archive days, and nothing over the cutoff.
            lines.append(QUIET_WEEK_LINE)
        return "\n".join(lines).rstrip() + "\n"

    lines.extend(_section(DEEP_READ_HEADING, deep_read))
    lines.extend(_section(SKIM_HEADING, skim))
    return "\n".join(lines).rstrip() + "\n"
