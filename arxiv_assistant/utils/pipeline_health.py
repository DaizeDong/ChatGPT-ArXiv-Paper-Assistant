# Tell "nothing matched" apart from "the scorer never ran". For three months the
# archive recorded scanned papers, zero selected and zero tokens, because an
# expired key made every batch raise into a bare except and the run still exited 0.
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

    The call ledger wins when present: the keyless backends report no tokens, so
    a token-based detector would call every healthy run an outage. Token counts
    stay as the fallback for a legacy OpenAI run that reports no ledger.
    """
    tokens = int(prompt_tokens or 0) + int(completion_tokens or 0)

    def verdict(status, message):
        return FilterHealth(status, scanned_papers, selected_papers, tokens, message)

    if not llm_filtering_enabled:
        return verdict(STATUS_SKIPPED, "LLM filtering is disabled by config; zero tokens is expected.")

    if scanned_papers <= 0:
        return verdict(STATUS_NO_INPUT,
                       "No papers reached the filter (arXiv returned nothing for the configured categories).")

    if llm_calls_attempted is not None:
        attempted = int(llm_calls_attempted)
        succeeded = int(llm_calls_succeeded or 0)
        if attempted == 0:
            return verdict(STATUS_LLM_UNAVAILABLE, (
                f"{scanned_papers} papers were scanned but the pipeline made ZERO model calls. "
                "Scoring did not run, so an empty result carries no information about relevance. "
                "Check the [LLM] backend and whether its transport is reachable."))
        if succeeded == 0:
            return verdict(STATUS_LLM_UNAVAILABLE, (
                f"{scanned_papers} papers were scanned and {attempted} model calls were attempted, "
                "but NONE succeeded. An empty result today is an OUTAGE, not a quiet day."))
        return verdict(STATUS_OK, (
            f"Scored {scanned_papers} papers with {succeeded}/{attempted} successful model calls; "
            f"{selected_papers} selected."))

    if tokens <= 0:
        return verdict(STATUS_LLM_UNAVAILABLE, (
            f"{scanned_papers} papers were scanned but the LLM consumed ZERO tokens. Every scoring "
            "call failed and was swallowed by the per-batch except in filters/filter_gpt.py. "
            "Check OPENAI_API_KEY (an expired key returns 401 on every batch) and the "
            "[SELECTION] model id."))

    return verdict(STATUS_OK,
                   f"Scored {scanned_papers} papers with {tokens} tokens; {selected_papers} selected.")


def format_banner(health: FilterHealth) -> str:
    """A banner loud enough to find in a CI log. Empty string when healthy."""
    if not health.is_outage:
        return ""
    rule = "!" * 78
    return "\n".join([
        "", rule,
        "!! PAPER FILTER OUTAGE",
        f"!! {health.message}",
        "!! The archive for today will be written EMPTY. That empty file is not evidence",
        "!! that nothing was relevant; it is evidence that nothing was scored.",
        rule, "",
    ])
