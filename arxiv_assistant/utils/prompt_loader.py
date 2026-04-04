from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

PROMPT_PATHS = {
    "paper.system_prompt": [
        REPO_ROOT / "prompts" / "paper" / "system_prompt.txt",
        REPO_ROOT / "prompts" / "system_prompt.txt",
    ],
    "paper.topics": [
        REPO_ROOT / "prompts" / "paper" / "paper_topics.txt",
        REPO_ROOT / "prompts" / "paper_topics.txt",
    ],
    "paper.score_criteria": [
        REPO_ROOT / "prompts" / "paper" / "score_criteria.txt",
        REPO_ROOT / "prompts" / "score_criteria.txt",
    ],
    "paper.postfix_title": [
        REPO_ROOT / "prompts" / "paper" / "postfix_prompt_title.txt",
        REPO_ROOT / "prompts" / "postfix_prompt_title.txt",
    ],
    "paper.postfix_abstract": [
        REPO_ROOT / "prompts" / "paper" / "postfix_prompt_abstract.txt",
        REPO_ROOT / "prompts" / "postfix_prompt_abstract.txt",
    ],
    "hotspot.system_prompt": [
        REPO_ROOT / "prompts" / "hotspot" / "system_prompt.txt",
        REPO_ROOT / "prompts" / "hotspot_system_prompt.txt",
    ],
    "hotspot.screening_criteria": [
        REPO_ROOT / "prompts" / "hotspot" / "screening_criteria.txt",
        REPO_ROOT / "prompts" / "hotspot_screening_criteria.txt",
    ],
    "hotspot.postfix_screening": [
        REPO_ROOT / "prompts" / "hotspot" / "postfix_prompt_screening.txt",
        REPO_ROOT / "prompts" / "postfix_prompt_hotspot_screening.txt",
    ],
    "hotspot.digest_writer": [
        REPO_ROOT / "prompts" / "hotspot" / "digest_writer.txt",
        REPO_ROOT / "prompts" / "hotspot_digest_writer.txt",
    ],
    "hotspot.enrich": [
        REPO_ROOT / "prompts" / "hotspot" / "enrich_prompt.txt",
    ],
    "monthly.system_prompt": [
        REPO_ROOT / "prompts" / "monthly" / "system_prompt.txt",
        REPO_ROOT / "prompts" / "monthly_summary_system_prompt.txt",
    ],
    "monthly.criteria": [
        REPO_ROOT / "prompts" / "monthly" / "criteria.txt",
        REPO_ROOT / "prompts" / "monthly_summary_criteria.txt",
    ],
    "monthly.postfix": [
        REPO_ROOT / "prompts" / "monthly" / "postfix_prompt.txt",
        REPO_ROOT / "prompts" / "postfix_prompt_monthly_summary.txt",
    ],
}


def resolve_prompt_path(prompt_key: str) -> Path:
    candidates = PROMPT_PATHS.get(prompt_key)
    if not candidates:
        raise KeyError(f"Unknown prompt key: {prompt_key}")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Unable to resolve prompt `{prompt_key}`. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def read_prompt(prompt_key: str) -> str:
    return resolve_prompt_path(prompt_key).read_text(encoding="utf-8")
