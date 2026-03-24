from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from arxiv_assistant.utils.hotspot.hotspot_schema import clean_text
from arxiv_assistant.utils.hotspot.x_authority_registry import DEFAULT_SEED_PATH, DEFAULT_SNAPSHOT_PATH, find_authority_record, load_x_authority_registry

X_HOSTS = {"x.com", "twitter.com", "mobile.twitter.com", "www.x.com", "www.twitter.com", "pic.x.com"}
GENERAL_DROP_PATTERNS = (
    re.compile(r"\bhiring\b", re.I),
    re.compile(r"\bjoin us\b", re.I),
    re.compile(r"\bsign up\b", re.I),
    re.compile(r"\bregister\b", re.I),
    re.compile(r"\bwaitlist\b", re.I),
    re.compile(r"\bworkshop\b", re.I),
    re.compile(r"\blivestream\b", re.I),
    re.compile(r"\bwebinar\b", re.I),
    re.compile(r"\bhow to\b", re.I),
    re.compile(r"\btutorial\b", re.I),
    re.compile(r"\btip:?s?\b", re.I),
    re.compile(r"\bdonut drop\b", re.I),
    re.compile(r"\bchallenge\b", re.I),
    re.compile(r"\bday \d+ of\b", re.I),
    re.compile(r"\blightning talk(s)?\b", re.I),
    re.compile(r"\btrivia\b", re.I),
    re.compile(r"\bbooth\b", re.I),
    re.compile(r"\bconference\b", re.I),
    re.compile(r"\bsummit\b", re.I),
    re.compile(r"\bgtc\b", re.I),
    re.compile(r"\bsee you at\b", re.I),
)
CONVERSATIONAL_PATTERNS = (
    re.compile(r"\bnobody told me\b", re.I),
    re.compile(r"\brandom thought\b", re.I),
    re.compile(r"\bwhat do you think\b", re.I),
    re.compile(r"\banyone else\b", re.I),
    re.compile(r"\bgood morning\b", re.I),
    re.compile(r"\bwtf\b", re.I),
    re.compile(r"\blol\b", re.I),
    re.compile(r"\blmao\b", re.I),
)
SELF_WORK_PATTERNS = (
    re.compile(r"\bmy new paper\b", re.I),
    re.compile(r"\bour new paper\b", re.I),
    re.compile(r"\bnew paper\b", re.I),
    re.compile(r"\bpreprint\b", re.I),
    re.compile(r"\barxiv\b", re.I),
    re.compile(r"\bwe released\b", re.I),
    re.compile(r"\bi released\b", re.I),
    re.compile(r"\bopen sourced?\b", re.I),
    re.compile(r"\bweights?\b", re.I),
    re.compile(r"\bcheckpoint(s)?\b", re.I),
    re.compile(r"\bcode for\b", re.I),
)
OFFICIAL_NEWS_PATTERNS = (
    re.compile(r"\bis available\b", re.I),
    re.compile(r"\bavailable today\b", re.I),
    re.compile(r"\bavailable now\b", re.I),
    re.compile(r"\bintroducing\b", re.I),
    re.compile(r"\bannounc(?:e|ing|ed)\b", re.I),
    re.compile(r"\blaunch(?:ed|ing|es)?\b", re.I),
    re.compile(r"\brollout\b", re.I),
    re.compile(r"\bpartnership\b", re.I),
    re.compile(r"\bacquisition\b", re.I),
    re.compile(r"\bpricing\b", re.I),
    re.compile(r"\breport\b", re.I),
    re.compile(r"\bstudy\b", re.I),
    re.compile(r"\bpolicy\b", re.I),
    re.compile(r"\bsafety\b", re.I),
    re.compile(r"\bapp store\b", re.I),
    re.compile(r"\bapi\b", re.I),
    re.compile(r"\bplatform\b", re.I),
)
OFFICIAL_SUBSTANCE_PATTERNS = (
    re.compile(r"\bapi\b", re.I),
    re.compile(r"\bmodel(s)?\b", re.I),
    re.compile(r"\bagent(s)?\b", re.I),
    re.compile(r"\bassistant\b", re.I),
    re.compile(r"\bcoding\b", re.I),
    re.compile(r"\breasoning\b", re.I),
    re.compile(r"\bresearch\b", re.I),
    re.compile(r"\bstudy\b", re.I),
    re.compile(r"\breport\b", re.I),
    re.compile(r"\bchatgpt\b", re.I),
    re.compile(r"\bgpt\b", re.I),
    re.compile(r"\bcodex\b", re.I),
    re.compile(r"\bclaude\b", re.I),
    re.compile(r"\bgemini\b", re.I),
    re.compile(r"\bcopilot\b", re.I),
    re.compile(r"\bcursor\b", re.I),
    re.compile(r"\bcomet\b", re.I),
    re.compile(r"\bstitch\b", re.I),
)
RESEARCHER_COMMENTARY_PATTERNS = (
    re.compile(r"\bresults?\b", re.I),
    re.compile(r"\beval(?:s|uation)?\b", re.I),
    re.compile(r"\bbenchmark\b", re.I),
    re.compile(r"\bpolicy\b", re.I),
    re.compile(r"\bsafety\b", re.I),
    re.compile(r"\binference\b", re.I),
    re.compile(r"\btraining\b", re.I),
    re.compile(r"\breasoning\b", re.I),
    re.compile(r"\bagent(s)?\b", re.I),
    re.compile(r"\bmodel(s)?\b", re.I),
    re.compile(r"\bresearch\b", re.I),
    re.compile(r"\brelease\b", re.I),
    re.compile(r"\blaunch\b", re.I),
    re.compile(r"\bcompany\b", re.I),
    re.compile(r"\blab\b", re.I),
)
AI_RELEVANCE_PATTERNS = (
    re.compile(r"\bai\b", re.I),
    re.compile(r"\bllm\b", re.I),
    re.compile(r"\bgpt\b", re.I),
    re.compile(r"\bclaude\b", re.I),
    re.compile(r"\bgemini\b", re.I),
    re.compile(r"\bopenai\b", re.I),
    re.compile(r"\banthropic\b", re.I),
    re.compile(r"\bdeepmind\b", re.I),
    re.compile(r"\bmistral\b", re.I),
    re.compile(r"\bllama\b", re.I),
    re.compile(r"\bqwen\b", re.I),
    re.compile(r"\bdeepseek\b", re.I),
    re.compile(r"\bcursor\b", re.I),
    re.compile(r"\bcopilot\b", re.I),
    re.compile(r"\bagent(s)?\b", re.I),
    re.compile(r"\breasoning\b", re.I),
    re.compile(r"\bmultimodal\b", re.I),
    re.compile(r"\bmodel(s)?\b", re.I),
    re.compile(r"\btraining\b", re.I),
    re.compile(r"\binference\b", re.I),
    re.compile(r"\brobotics\b", re.I),
    re.compile(r"\bworld model(s)?\b", re.I),
)


@lru_cache(maxsize=4)
def _cached_registry(seed_path: str, snapshot_path: str) -> dict[str, Any]:
    return load_x_authority_registry(seed_path=seed_path, snapshot_path=snapshot_path, max_age_hours=24)


def load_authority_registry(
    *,
    seed_path: str | Path | None = None,
    snapshot_path: str | Path | None = None,
) -> dict[str, Any]:
    resolved_seed = str(Path(seed_path) if seed_path else DEFAULT_SEED_PATH)
    resolved_snapshot = str(Path(snapshot_path) if snapshot_path else DEFAULT_SNAPSHOT_PATH)
    return _cached_registry(resolved_seed, resolved_snapshot)


def get_authority_record(
    author_handle: str | None,
    *,
    registry: dict[str, Any] | None = None,
    seed_path: str | Path | None = None,
    snapshot_path: str | Path | None = None,
) -> dict[str, Any] | None:
    registry_payload = registry or load_authority_registry(seed_path=seed_path, snapshot_path=snapshot_path)
    return find_authority_record(author_handle, registry_payload)


def is_authoritative_x_identity(
    author_handle: str | None,
    author_name: str | None = None,
    *,
    registry: dict[str, Any] | None = None,
    allowed_kinds: set[str] | None = None,
    seed_path: str | Path | None = None,
    snapshot_path: str | Path | None = None,
) -> bool:
    record = get_authority_record(
        author_handle,
        registry=registry,
        seed_path=seed_path,
        snapshot_path=snapshot_path,
    )
    if record is None or not record.get("active", False):
        return False
    if allowed_kinds and str(record.get("kind")) not in allowed_kinds:
        return False
    return True


def is_ai_relevant_x_text(text: str | None) -> bool:
    normalized = clean_text(text)
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in AI_RELEVANCE_PATTERNS)


def _has_external_link(urls: list[str] | None) -> bool:
    for url in urls or []:
        host = urlsplit(url).netloc.lower()
        if host and host not in X_HOSTS:
            return True
    return False


def is_newsworthy_x_text(
    text: str | None,
    *,
    authority_kind: str = "official",
    expanded_urls: list[str] | None = None,
    activity: int = 0,
) -> bool:
    normalized = clean_text(text).lower()
    if not normalized:
        return False
    if any(pattern.search(normalized) for pattern in GENERAL_DROP_PATTERNS):
        return False
    if any(pattern.search(normalized) for pattern in CONVERSATIONAL_PATTERNS):
        return False
    if not is_ai_relevant_x_text(normalized):
        return False

    has_external_link = _has_external_link(expanded_urls)
    if authority_kind == "researcher":
        if any(pattern.search(normalized) for pattern in SELF_WORK_PATTERNS):
            return False
        if not has_external_link:
            return False
        if activity < 40:
            return False
        return any(pattern.search(normalized) for pattern in (RESEARCHER_COMMENTARY_PATTERNS + OFFICIAL_NEWS_PATTERNS))

    if any(pattern.search(normalized) for pattern in SELF_WORK_PATTERNS):
        return False
    if not has_external_link and activity < 60:
        return False
    has_news_pattern = any(pattern.search(normalized) for pattern in OFFICIAL_NEWS_PATTERNS)
    has_substance = any(pattern.search(normalized) for pattern in OFFICIAL_SUBSTANCE_PATTERNS)
    if has_news_pattern:
        return True
    return has_external_link and has_substance and activity >= 120
