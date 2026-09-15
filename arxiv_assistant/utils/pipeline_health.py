"""Tell "nothing matched" apart from "the scorer never ran".

Why this module exists, concretely: between 2026-06-05 and 2026-09-09 this repo
published an archive in which EVERY ``out/json/*/<date>-output.json`` was the
two-byte literal ``{}``. The daily bundles recorded ``total_scanned_papers: 342``
next to ``total_relevant_papers: 0`` and ``prompt_tokens: 0`` -- the OpenAI key had
expired, every batch raised ``AuthenticationError``, and the bare ``except
Exception`` at ``filters/filter_gpt.py:182`` and ``:311`` printed one line and
continued. Every run exited 0. Actions was green for three months.

The signature of that outage is exact and cheap to test for: papers were scanned,
the LLM filter was configured to run, and yet ZERO tokens were consumed. That is
not a quiet day; it is an outage wearing a quiet day's clothes.

This module does not decide what to do about it. It classifies, and it returns a
message worth printing. Callers stamp the verdict into the archive so a later
reader (the weekly digest, a human, a CI assertion) can tell the two apart
without re-deriving anything.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

STATUS_OK = "ok"
STATUS_LLM_UNAVAILABLE = "llm_unavailable"
STATUS_SKIPPED = "skipped"
STATUS_NO_INPUT = "no_input"


@dataclass(frozen=True)
class FilterHealth:
    status: str
    scanned_papers: int
    selected_papers: int
    llm_tokens: int
    message: str

    @property
    def is_outage(self) -> bool:
        return self.status == STATUS_LLM_UNAVAILABLE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "scanned_papers": self.scanned_papers,
            "selected_papers": self.selected_papers,
            "llm_tokens": self.llm_tokens,
            "message": self.message,
        }


def assess_paper_filter_health(
    *,
    scanned_papers: int,
    selected_papers: int,
    prompt_tokens: int,
    completion_tokens: int,
    llm_filtering_enabled: bool,
    llm_calls_attempted: int | None = None,
    llm_calls_succeeded: int | None = None,
) -> FilterHealth:
    """Classify one paper-pipeline run.

    ``llm_filtering_enabled`` must be True only when the run actually intended to
    call a model (``[SELECTION] run_openai`` on, and at least one of the title /
    abstract filters on). When it is False a zero-token run is simply a run that
    did no scoring, which is a legitimate configuration and not an outage.

    TWO SIGNALS, AND THE CALL LEDGER WINS. Token counts only mean anything on a
    backend that reports tokens. Once this pipeline moved to the llmcall chain
    (codexg -> codex -> cc -> claude) there are no OpenAI tokens at all, so a
    token-based detector would report an outage on every healthy run forever --
    the exact inverse of the bug it was built to catch, and just as useless.
    So when ``llm_calls_attempted`` is supplied it is authoritative: N attempts
    with zero successes is an outage on any backend, and zero attempts while
    papers were waiting to be scored is also an outage, because the scoring never
    happened. Token counts remain the fallback for a legacy OpenAI-only run that
    reports no ledger.
    """
    tokens = int(prompt_tokens or 0) + int(completion_tokens or 0)

    if not llm_filtering_enabled:
        return FilterHealth(
            status=STATUS_SKIPPED,
            scanned_papers=scanned_papers,
            selected_papers=selected_papers,
            llm_tokens=tokens,
            message="LLM filtering is disabled by config; zero tokens is expected.",
        )

    if scanned_papers <= 0:
        return FilterHealth(
            status=STATUS_NO_INPUT,
            scanned_papers=scanned_papers,
            selected_papers=selected_papers,
            llm_tokens=tokens,
            message="No papers reached the filter (arXiv returned nothing for the configured categories).",
        )

    if llm_calls_attempted is not None:
        attempted = int(llm_calls_attempted)
        succeeded = int(llm_calls_succeeded or 0)
        if attempted == 0:
            return FilterHealth(
                status=STATUS_LLM_UNAVAILABLE,
                scanned_papers=scanned_papers,
                selected_papers=selected_papers,
                llm_tokens=tokens,
                message=(
                    f"{scanned_papers} papers were scanned but the pipeline made ZERO model "
                    "calls. Scoring did not run at all, so an empty result today carries no "
                    "information about relevance. Check the [LLM] backend setting and whether "
                    "the llmcall chain or the claude CLI is reachable."
                ),
            )
        if succeeded == 0:
            return FilterHealth(
                status=STATUS_LLM_UNAVAILABLE,
                scanned_papers=scanned_papers,
                selected_papers=selected_papers,
                llm_tokens=tokens,
                message=(
                    f"{scanned_papers} papers were scanned and {attempted} model calls were "
                    "attempted, but NONE succeeded. Every call failed and was degraded past. "
                    "An empty result today is an OUTAGE, not a quiet day. Check the llmcall "
                    "chain (codexg -> codex -> cc -> claude) and the claude CLI login."
                ),
            )
        return FilterHealth(
            status=STATUS_OK,
            scanned_papers=scanned_papers,
            selected_papers=selected_papers,
            llm_tokens=tokens,
            message=(
                f"Scored {scanned_papers} papers with {succeeded}/{attempted} successful model "
                f"calls; {selected_papers} selected."
            ),
        )

    if tokens <= 0:
        return FilterHealth(
            status=STATUS_LLM_UNAVAILABLE,
            scanned_papers=scanned_papers,
            selected_papers=selected_papers,
            llm_tokens=0,
            message=(
                f"{scanned_papers} papers were scanned but the LLM consumed ZERO tokens. "
                "Every scoring call failed and was swallowed by the per-batch except in "
                "filters/filter_gpt.py. An empty result today is an OUTAGE, not a quiet day. "
                "Check OPENAI_API_KEY (an expired key returns 401 on every batch) and the "
                "[SELECTION] model id."
            ),
        )

    return FilterHealth(
        status=STATUS_OK,
        scanned_papers=scanned_papers,
        selected_papers=selected_papers,
        llm_tokens=tokens,
        message=f"Scored {scanned_papers} papers with {tokens} tokens; {selected_papers} selected.",
    )


def format_banner(health: FilterHealth) -> str:
    """A banner loud enough to find in a CI log. Empty string when healthy."""
    if not health.is_outage:
        return ""
    rule = "!" * 78
    return "\n".join(
        [
            "",
            rule,
            "!! PAPER FILTER OUTAGE",
            f"!! {health.message}",
            "!! The archive for today will be written EMPTY. That empty file is not evidence",
            "!! that nothing was relevant; it is evidence that nothing was scored.",
            rule,
            "",
        ]
    )
