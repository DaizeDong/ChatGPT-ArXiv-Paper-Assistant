from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

TRACKING_QUERY_PREFIXES = ("utm_", "fbclid", "gclid", "mc_", "ref")


def clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()


def normalize_url(url: str | None, base_url: str | None = None) -> str:
    if not url:
        return ""
    if base_url:
        url = urljoin(base_url, url)

    split = urlsplit(url.strip())
    query_pairs = [
        (key, value)
        for key, value in parse_qsl(split.query, keep_blank_values=True)
        if not any(key.lower().startswith(prefix) for prefix in TRACKING_QUERY_PREFIXES)
    ]
    normalized_path = re.sub(r"/+", "/", split.path).rstrip("/")
    normalized = urlunsplit(
        (
            split.scheme.lower(),
            split.netloc.lower(),
            normalized_path,
            urlencode(query_pairs, doseq=True),
            "",
        )
    )
    return normalized.rstrip("/")


@dataclass
class HotspotItem:
    source_id: str
    source_name: str
    source_role: str
    source_type: str
    title: str
    summary: str
    url: str
    canonical_url: str
    published_at: str | None = None
    tags: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.title = clean_text(self.title)
        self.summary = clean_text(self.summary)
        self.url = normalize_url(self.url)
        self.canonical_url = normalize_url(self.canonical_url or self.url)
        self.tags = [clean_text(tag) for tag in self.tags if clean_text(tag)]
        self.authors = [clean_text(author) for author in self.authors if clean_text(author)]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class HotspotCluster:
    cluster_id: str
    title: str
    canonical_url: str
    summary: str
    items: list[dict[str, Any]]
    source_ids: list[str]
    source_names: list[str]
    source_roles: list[str]
    source_types: list[str]
    tags: list[str]
    published_at: str | None
    deterministic_score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
