from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "configs" / "paper_topics_schema.json"
HOTSPOT_PAPER_KIND_ORDER = ("new_frontier", "daily_hot")
HOTSPOT_PAPER_KIND_LABELS = {
    "new_frontier": "New Frontier Papers",
    "daily_hot": "Daily Hot Papers",
}
HOTSPOT_PAPER_KIND_DESCRIPTIONS = {
    "new_frontier": "Papers that appear to open a genuinely new direction, paradigm, or field.",
    "daily_hot": "Papers that feel broadly important to the day and belong in the hotspot paper feed.",
}
HOTSPOT_PAPER_KIND_ALIASES = {
    "daily hot": "daily_hot",
    "hot": "daily_hot",
    "trend": "daily_hot",
    "trending": "daily_hot",
    "new frontier": "new_frontier",
    "frontier": "new_frontier",
    "new field": "new_frontier",
    "new direction": "new_frontier",
    "new paradigm": "new_frontier",
}
LEGACY_GENERIC_HOTSPOT_COMMENTS = {
    "Heuristic fallback marked this as a likely new frontier direction based on very high novelty.",
    "Heuristic fallback marked this as a strong same-day hotspot paper based on high relevance and score.",
    "Heuristic fallback marked this as both a same-day hotspot paper and a likely new frontier direction.",
}


@dataclass(frozen=True)
class PaperTopic:
    id: str
    label: str
    order: int
    description: str
    aliases: tuple[str, ...]
    keywords: tuple[str, ...]
    negative_keywords: tuple[str, ...]


class TopicRegistry:
    def __init__(self, schema_version: int, default_topic_id: str, topics: Sequence[PaperTopic]):
        self.schema_version = int(schema_version)
        self.default_topic_id = str(default_topic_id)
        self.topics = tuple(sorted(topics, key=lambda topic: topic.order))
        self.topics_by_id = {topic.id: topic for topic in self.topics}
        self.topic_ids = tuple(topic.id for topic in self.topics)
        self.labels_by_id = {topic.id: topic.label for topic in self.topics}
        self.alias_to_id = self._build_alias_index()

        if self.default_topic_id not in self.topics_by_id:
            raise ValueError(f"Unknown default topic id: {self.default_topic_id}")

    def _build_alias_index(self) -> Dict[str, str]:
        alias_to_id: Dict[str, str] = {}
        for topic in self.topics:
            alias_values = {topic.id, topic.label, *topic.aliases}
            for alias in alias_values:
                normalized = normalize_topic_token(alias)
                if normalized:
                    alias_to_id[normalized] = topic.id
        return alias_to_id

    def get(self, topic_id: str) -> PaperTopic:
        return self.topics_by_id[topic_id]

    def normalize(self, value: object) -> str | None:
        if value is None:
            return None
        normalized = normalize_topic_token(str(value))
        if not normalized:
            return None
        return self.alias_to_id.get(normalized)


@dataclass(frozen=True)
class TopicAssignment:
    primary_topic_id: str
    matched_topic_ids: tuple[str, ...]
    topic_match_comment: str
    assignment_source: str


@dataclass(frozen=True)
class HotspotPaperAssignment:
    kinds: tuple[str, ...]
    comment: str
    assignment_source: str


@lru_cache(maxsize=1)
def get_topic_registry() -> TopicRegistry:
    payload = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    topics = [
        PaperTopic(
            id=topic["id"],
            label=topic["label"],
            order=int(topic["order"]),
            description=topic["description"],
            aliases=tuple(topic.get("aliases", [])),
            keywords=tuple(topic.get("keywords", [])),
            negative_keywords=tuple(topic.get("negative_keywords", [])),
        )
        for topic in payload["topics"]
    ]
    return TopicRegistry(
        schema_version=payload["schema_version"],
        default_topic_id=payload["default_topic_id"],
        topics=topics,
    )


def normalize_topic_token(value: str) -> str:
    value = re.sub(r"[_\-/]+", " ", value.strip().lower())
    value = re.sub(r"\s+", " ", value)
    return value


def build_topic_registry_prompt_block() -> str:
    registry = get_topic_registry()
    lines = [
        "## Topic Registry",
        "Use exactly one PRIMARY_TOPIC_ID chosen from the stable topic IDs below.",
    ]
    for topic in registry.topics:
        lines.append(f"- {topic.id}: {topic.label}")
        lines.append(f"  - {topic.description}")
    return "\n".join(lines)


def normalize_topic_ids(values: object, *, primary_topic_id: str | None = None) -> List[str]:
    registry = get_topic_registry()
    normalized_ids: List[str] = []
    if isinstance(values, str):
        candidate_values: Iterable[object] = [part.strip() for part in values.split(",") if part.strip()]
    elif isinstance(values, Iterable):
        candidate_values = values
    else:
        candidate_values = []

    for value in candidate_values:
        normalized = registry.normalize(value)
        if normalized is not None and normalized not in normalized_ids:
            normalized_ids.append(normalized)

    if primary_topic_id:
        normalized_primary = registry.normalize(primary_topic_id) or primary_topic_id
        if normalized_primary in normalized_ids:
            normalized_ids.remove(normalized_primary)
        normalized_ids.insert(0, normalized_primary)

    ordered_ids = [topic_id for topic_id in registry.topic_ids if topic_id in normalized_ids]
    return ordered_ids


def normalize_hotspot_paper_kind(value: object) -> str | None:
    if value is None:
        return None
    normalized = normalize_topic_token(str(value))
    if not normalized:
        return None
    if normalized in HOTSPOT_PAPER_KIND_ALIASES:
        return HOTSPOT_PAPER_KIND_ALIASES[normalized]
    candidate = normalized.replace(" ", "_")
    if candidate in HOTSPOT_PAPER_KIND_LABELS:
        return candidate
    return None


def normalize_hotspot_paper_kinds(values: object) -> List[str]:
    if isinstance(values, str):
        candidate_values: Iterable[object] = [part.strip() for part in values.split(",") if part.strip()]
    elif isinstance(values, Iterable):
        candidate_values = values
    else:
        candidate_values = []

    normalized_kinds: List[str] = []
    for value in candidate_values:
        normalized = normalize_hotspot_paper_kind(value)
        if normalized is not None and normalized not in normalized_kinds:
            normalized_kinds.append(normalized)

    return [kind for kind in HOTSPOT_PAPER_KIND_ORDER if kind in normalized_kinds]


def _field_text(paper_entry: Mapping[str, object], field_name: str) -> str:
    return str(paper_entry.get(field_name, "") or "").lower()


def heuristic_topic_assignment(paper_entry: Mapping[str, object]) -> TopicAssignment:
    registry = get_topic_registry()
    title_text = _field_text(paper_entry, "title")
    comment_text = _field_text(paper_entry, "COMMENT")
    abstract_text = _field_text(paper_entry, "abstract")

    topic_scores: Dict[str, int] = {}
    for topic in registry.topics:
        score = 0
        for keyword in topic.keywords:
            if keyword in title_text:
                score += 6
            if keyword in comment_text:
                score += 4
            if keyword in abstract_text:
                score += 2
        for keyword in topic.negative_keywords:
            if keyword in title_text:
                score -= 6
            if keyword in comment_text:
                score -= 4
            if keyword in abstract_text:
                score -= 2
        topic_scores[topic.id] = score

    ordered_scores = sorted(
        topic_scores.items(),
        key=lambda item: (item[1], -registry.get(item[0]).order),
        reverse=True,
    )
    matched_topic_ids = [topic_id for topic_id, score in ordered_scores if score > 0]

    if matched_topic_ids:
        primary_topic_id = matched_topic_ids[0]
        best_score = topic_scores[primary_topic_id]
        topic_match_comment = f"Heuristic topic match from title/abstract/comment signals (score={best_score})."
        assignment_source = "heuristic"
    else:
        primary_topic_id = registry.default_topic_id
        matched_topic_ids = [primary_topic_id]
        topic_match_comment = "Fell back to the default foundational topic because no stable topic signals were detected."
        assignment_source = "default"

    return TopicAssignment(
        primary_topic_id=primary_topic_id,
        matched_topic_ids=tuple(normalize_topic_ids(matched_topic_ids, primary_topic_id=primary_topic_id)),
        topic_match_comment=topic_match_comment,
        assignment_source=assignment_source,
    )


def heuristic_hotspot_paper_assignment(paper_entry: Mapping[str, object]) -> HotspotPaperAssignment:
    score = int(paper_entry.get("SCORE", 0) or 0)
    relevance = int(paper_entry.get("RELEVANCE", 0) or 0)
    novelty = int(paper_entry.get("NOVELTY", 0) or 0)

    kinds: List[str] = []
    if novelty >= 9 and score >= 17:
        kinds.append("new_frontier")
    if relevance >= 9 and score >= 17:
        kinds.append("daily_hot")

    if not kinds:
        return HotspotPaperAssignment(kinds=(), comment="", assignment_source="default")
    comment = build_hotspot_paper_comment(kinds, paper_entry)
    return HotspotPaperAssignment(kinds=tuple(kinds), comment=comment, assignment_source="heuristic")


def build_hotspot_paper_comment(kinds: Sequence[str], paper_entry: Mapping[str, object]) -> str:
    hotspot_reason = str(paper_entry.get("HOTSPOT_PAPER_COMMENT") or "").strip()
    if hotspot_reason in LEGACY_GENERIC_HOTSPOT_COMMENTS:
        hotspot_reason = ""
    if hotspot_reason:
        return hotspot_reason

    strongest_reason = str(paper_entry.get("COMMENT") or paper_entry.get("TOPIC_MATCH_COMMENT") or "").strip()
    if strongest_reason:
        if "new_frontier" in kinds and "daily_hot" in kinds:
            return f"Broadly important today and potentially frontier-opening: {strongest_reason}"
        if "new_frontier" in kinds:
            return f"Potentially frontier-opening: {strongest_reason}"
        if "daily_hot" in kinds:
            return f"Broadly important today: {strongest_reason}"

    if kinds == ["new_frontier"]:
        return "Likely frontier-opening based on unusually high novelty and foundational relevance."
    if kinds == ["daily_hot"]:
        return "Looks broadly important to the day based on high score and relevance."
    return "Looks broadly important today and potentially frontier-opening."


def ensure_topic_fields(paper_entry: Mapping[str, object], *, arxiv_id: str | None = None) -> Dict:
    registry = get_topic_registry()
    normalized_entry = dict(paper_entry)
    if arxiv_id is not None:
        normalized_entry.setdefault("arxiv_id", arxiv_id)
        normalized_entry.setdefault("ARXIVID", arxiv_id)

    raw_primary = (
        normalized_entry.get("PRIMARY_TOPIC_ID")
        or normalized_entry.get("PRIMARY_CATEGORY")
        or normalized_entry.get("TOPIC_ID")
    )
    existing_topic_assignment_source = str(normalized_entry.get("TOPIC_ASSIGNMENT_SOURCE") or "").strip()
    primary_topic_id = registry.normalize(raw_primary)
    matched_topic_ids = normalize_topic_ids(
        normalized_entry.get("MATCHED_TOPIC_IDS") or normalized_entry.get("SECONDARY_TOPIC_IDS") or [],
        primary_topic_id=primary_topic_id,
    )

    if primary_topic_id is None:
        assignment = heuristic_topic_assignment(normalized_entry)
        primary_topic_id = assignment.primary_topic_id
        matched_topic_ids = list(assignment.matched_topic_ids)
        topic_match_comment = normalized_entry.get("TOPIC_MATCH_COMMENT") or assignment.topic_match_comment
        source_suffix = "invalid_primary_topic" if raw_primary else "missing_primary_topic"
        assignment_source = f"{assignment.assignment_source}_{source_suffix}"
    else:
        if not matched_topic_ids:
            matched_topic_ids = [primary_topic_id]
        elif primary_topic_id not in matched_topic_ids:
            matched_topic_ids = normalize_topic_ids(matched_topic_ids, primary_topic_id=primary_topic_id)
        topic_match_comment = normalized_entry.get("TOPIC_MATCH_COMMENT") or ""
        if existing_topic_assignment_source and str(normalized_entry.get("PRIMARY_TOPIC_ID") or "") == primary_topic_id:
            assignment_source = existing_topic_assignment_source
        else:
            assignment_source = "llm_exact" if str(raw_primary) == primary_topic_id else "llm_normalized"

    primary_label = registry.labels_by_id[primary_topic_id]
    matched_labels = [registry.labels_by_id[topic_id] for topic_id in matched_topic_ids]
    raw_hotspot_kinds = normalized_entry.get("HOTSPOT_PAPER_TAGS") or normalized_entry.get("HOTSPOT_TAGS") or []
    existing_hotspot_assignment_source = str(normalized_entry.get("HOTSPOT_PAPER_ASSIGNMENT_SOURCE") or "").strip()
    hotspot_kinds = normalize_hotspot_paper_kinds(raw_hotspot_kinds)
    if hotspot_kinds:
        hotspot_comment = build_hotspot_paper_comment(hotspot_kinds, normalized_entry)
        hotspot_assignment_source = existing_hotspot_assignment_source or "llm_exact"
    else:
        hotspot_assignment = heuristic_hotspot_paper_assignment(normalized_entry)
        hotspot_kinds = list(hotspot_assignment.kinds)
        hotspot_comment = str(hotspot_assignment.comment).strip()
        if raw_hotspot_kinds:
            hotspot_assignment_source = f"{hotspot_assignment.assignment_source}_invalid_hotspot_tag"
        else:
            hotspot_assignment_source = hotspot_assignment.assignment_source
    hotspot_primary_kind = hotspot_kinds[0] if hotspot_kinds else None

    return {
        **normalized_entry,
        "PRIMARY_TOPIC_ID": primary_topic_id,
        "PRIMARY_TOPIC_LABEL": primary_label,
        "MATCHED_TOPIC_IDS": matched_topic_ids,
        "MATCHED_TOPIC_LABELS": matched_labels,
        "TOPIC_MATCH_COMMENT": topic_match_comment,
        "TOPIC_ASSIGNMENT_SOURCE": assignment_source,
        "HOTSPOT_PAPER_TAGS": hotspot_kinds,
        "HOTSPOT_PAPER_LABELS": [HOTSPOT_PAPER_KIND_LABELS[kind] for kind in hotspot_kinds],
        "HOTSPOT_PAPER_PRIMARY_KIND": hotspot_primary_kind,
        "HOTSPOT_PAPER_PRIMARY_LABEL": HOTSPOT_PAPER_KIND_LABELS.get(hotspot_primary_kind, ""),
        "HOTSPOT_PAPER_COMMENT": hotspot_comment,
        "HOTSPOT_PAPER_ASSIGNMENT_SOURCE": hotspot_assignment_source,
    }


def ensure_topic_fields_for_mapping(paper_mapping: Mapping[str, Mapping[str, object]]) -> Dict[str, Dict]:
    return {
        arxiv_id: ensure_topic_fields(paper_entry, arxiv_id=arxiv_id)
        for arxiv_id, paper_entry in paper_mapping.items()
    }


def daily_sort_key(paper_entry: Mapping[str, object]) -> tuple[int, int]:
    return (
        int(paper_entry.get("SCORE", 0)),
        int(paper_entry.get("RELEVANCE", 0)),
    )


def sort_paper_mapping_for_daily_display(paper_mapping: Mapping[str, Mapping[str, object]]) -> Dict[str, Dict]:
    enriched_mapping = {
        arxiv_id: dict(paper_entry)
        if "PRIMARY_TOPIC_ID" in paper_entry and "TOPIC_ASSIGNMENT_SOURCE" in paper_entry
        else ensure_topic_fields(paper_entry, arxiv_id=arxiv_id)
        for arxiv_id, paper_entry in paper_mapping.items()
    }
    return {
        arxiv_id: paper_entry
        for arxiv_id, paper_entry in sorted(
            enriched_mapping.items(),
            key=lambda item: daily_sort_key(item[1]),
            reverse=True,
        )
    }


def group_sorted_papers_by_topic(sorted_papers: Sequence[Mapping[str, object]]) -> List[Dict]:
    registry = get_topic_registry()
    grouped: List[Dict] = []
    for topic in registry.topics:
        papers = [dict(paper) for paper in sorted_papers if paper.get("PRIMARY_TOPIC_ID") == topic.id]
        if not papers:
            continue
        grouped.append(
            {
                "topic_id": topic.id,
                "topic_label": topic.label,
                "topic_order": topic.order,
                "paper_count": len(papers),
                "papers": papers,
            }
        )
    return grouped


def build_topic_diagnostics(paper_mapping: Mapping[str, Mapping[str, object]]) -> Dict:
    registry = get_topic_registry()
    sorted_mapping = sort_paper_mapping_for_daily_display(paper_mapping)
    papers = list(sorted_mapping.values())

    assignment_source_counts: Dict[str, int] = {}
    topic_counts: List[Dict] = []
    for topic in registry.topics:
        topic_papers = [paper for paper in papers if paper.get("PRIMARY_TOPIC_ID") == topic.id]
        topic_counts.append(
            {
                "topic_id": topic.id,
                "topic_label": topic.label,
                "paper_count": len(topic_papers),
            }
        )

    for paper in papers:
        assignment_source = str(paper.get("TOPIC_ASSIGNMENT_SOURCE", "unknown"))
        assignment_source_counts[assignment_source] = assignment_source_counts.get(assignment_source, 0) + 1

    return {
        "total_papers": len(papers),
        "topic_counts": topic_counts,
        "assignment_source_counts": assignment_source_counts,
        "llm_assignment_count": sum(
            count for source, count in assignment_source_counts.items() if source.startswith("llm_")
        ),
        "heuristic_assignment_count": sum(
            count for source, count in assignment_source_counts.items() if source.startswith("heuristic_")
        ),
        "default_assignment_count": sum(
            count for source, count in assignment_source_counts.items() if source.startswith("default_")
        ),
        "invalid_primary_topic_fallback_count": sum(
            count
            for source, count in assignment_source_counts.items()
            if source.endswith("invalid_primary_topic")
        ),
        "missing_primary_topic_fallback_count": sum(
            count
            for source, count in assignment_source_counts.items()
            if source.endswith("missing_primary_topic") and not source.startswith("llm_")
        ),
        "hotspot_paper_kind_counts": {
            kind: sum(1 for paper in papers if kind in paper.get("HOTSPOT_PAPER_TAGS", []))
            for kind in HOTSPOT_PAPER_KIND_ORDER
        },
        "hotspot_paper_assignment_source_counts": {
            source: count
            for source, count in _count_assignment_sources(papers, "HOTSPOT_PAPER_ASSIGNMENT_SOURCE").items()
        },
    }


def _count_assignment_sources(
    papers: Sequence[Mapping[str, object]],
    field_name: str,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for paper in papers:
        assignment_source = str(paper.get(field_name, "unknown"))
        counts[assignment_source] = counts.get(assignment_source, 0) + 1
    return counts


def build_daily_topic_bundle(
    date_key: tuple[int, int, int],
    paper_mapping: Mapping[str, Mapping[str, object]],
    *,
    usage: Mapping[str, object] | None = None,
) -> Dict:
    registry = get_topic_registry()
    sorted_mapping = sort_paper_mapping_for_daily_display(paper_mapping)
    sorted_papers = list(sorted_mapping.values())
    grouped_sections = group_sorted_papers_by_topic(sorted_papers)
    return {
        "schema_version": 2,
        "meta": {
            "date": f"{date_key[0]:04d}-{date_key[1]:02d}-{date_key[2]:02d}",
            "sort": ["score", "relevance"],
            "topic_schema_version": registry.schema_version,
            "total_papers": len(sorted_papers),
            "usage": dict(usage or {}),
        },
        "topic_order": list(registry.topic_ids),
        "diagnostics": build_topic_diagnostics(sorted_mapping),
        "papers": sorted_mapping,
        "topic_sections": [
            {
                "topic_id": section["topic_id"],
                "topic_label": section["topic_label"],
                "paper_ids": [paper["arxiv_id"] for paper in section["papers"]],
            }
            for section in grouped_sections
        ],
    }


def _hotspot_daily_hot_sort_key(paper_entry: Mapping[str, object]) -> tuple[int, int, int, str]:
    return (
        int(paper_entry.get("SCORE", 0)),
        int(paper_entry.get("RELEVANCE", 0)),
        int(paper_entry.get("NOVELTY", 0)),
        str(paper_entry.get("title", "")).lower(),
    )


def _hotspot_new_frontier_sort_key(paper_entry: Mapping[str, object]) -> tuple[int, int, int, str]:
    return (
        int(paper_entry.get("NOVELTY", 0)),
        int(paper_entry.get("SCORE", 0)),
        int(paper_entry.get("RELEVANCE", 0)),
        str(paper_entry.get("title", "")).lower(),
    )


def build_hotspot_paper_bundle(
    date_key: tuple[int, int, int],
    paper_mapping: Mapping[str, Mapping[str, object]],
    *,
    max_daily_hot: int = 6,
    max_new_frontier: int = 4,
    daily_hot_score_cutoff: int = 15,
    daily_hot_relevance_cutoff: int = 7,
    new_frontier_score_cutoff: int = 15,
    new_frontier_novelty_cutoff: int = 8,
) -> Dict:
    normalized_mapping = {
        arxiv_id: dict(paper_entry)
        if "PRIMARY_TOPIC_ID" in paper_entry and "TOPIC_ASSIGNMENT_SOURCE" in paper_entry and "HOTSPOT_PAPER_ASSIGNMENT_SOURCE" in paper_entry
        else ensure_topic_fields(paper_entry, arxiv_id=arxiv_id)
        for arxiv_id, paper_entry in paper_mapping.items()
    }
    sorted_mapping = {
        arxiv_id: paper_entry
        for arxiv_id, paper_entry in sorted(
            normalized_mapping.items(),
            key=lambda item: daily_sort_key(item[1]),
            reverse=True,
        )
    }
    spotlight_mapping: Dict[str, Dict] = {}
    sections: List[Dict] = []
    selected_ids: set[str] = set()

    new_frontier_candidates = [
        paper
        for paper in sorted_mapping.values()
        if "new_frontier" in paper.get("HOTSPOT_PAPER_TAGS", [])
        and int(paper.get("NOVELTY", 0)) >= new_frontier_novelty_cutoff
        and int(paper.get("SCORE", 0)) >= new_frontier_score_cutoff
    ]
    new_frontier_candidates = sorted(new_frontier_candidates, key=_hotspot_new_frontier_sort_key, reverse=True)
    chosen_new_frontier = []
    for paper in new_frontier_candidates:
        arxiv_id = str(paper.get("arxiv_id") or paper.get("ARXIVID") or "")
        if not arxiv_id or arxiv_id in selected_ids:
            continue
        chosen_new_frontier.append(dict(paper))
        spotlight_mapping[arxiv_id] = dict(paper)
        selected_ids.add(arxiv_id)
        if len(chosen_new_frontier) >= max_new_frontier:
            break
    if chosen_new_frontier:
        sections.append(
            {
                "kind": "new_frontier",
                "label": HOTSPOT_PAPER_KIND_LABELS["new_frontier"],
                "description": HOTSPOT_PAPER_KIND_DESCRIPTIONS["new_frontier"],
                "paper_ids": [paper["arxiv_id"] for paper in chosen_new_frontier],
            }
        )

    daily_hot_candidates = [
        paper
        for paper in sorted_mapping.values()
        if "daily_hot" in paper.get("HOTSPOT_PAPER_TAGS", [])
        and int(paper.get("RELEVANCE", 0)) >= daily_hot_relevance_cutoff
        and int(paper.get("SCORE", 0)) >= daily_hot_score_cutoff
    ]
    daily_hot_candidates = sorted(daily_hot_candidates, key=_hotspot_daily_hot_sort_key, reverse=True)
    chosen_daily_hot = []
    for paper in daily_hot_candidates:
        arxiv_id = str(paper.get("arxiv_id") or paper.get("ARXIVID") or "")
        if not arxiv_id or arxiv_id in selected_ids:
            continue
        chosen_daily_hot.append(dict(paper))
        spotlight_mapping[arxiv_id] = dict(paper)
        selected_ids.add(arxiv_id)
        if len(chosen_daily_hot) >= max_daily_hot:
            break
    if chosen_daily_hot:
        sections.append(
            {
                "kind": "daily_hot",
                "label": HOTSPOT_PAPER_KIND_LABELS["daily_hot"],
                "description": HOTSPOT_PAPER_KIND_DESCRIPTIONS["daily_hot"],
                "paper_ids": [paper["arxiv_id"] for paper in chosen_daily_hot],
            }
        )

    return {
        "schema_version": 1,
        "meta": {
            "date": f"{date_key[0]:04d}-{date_key[1]:02d}-{date_key[2]:02d}",
            "total_candidates": len(sorted_mapping),
            "total_spotlight_papers": len(spotlight_mapping),
            "thresholds": {
                "daily_hot_score_cutoff": daily_hot_score_cutoff,
                "daily_hot_relevance_cutoff": daily_hot_relevance_cutoff,
                "new_frontier_score_cutoff": new_frontier_score_cutoff,
                "new_frontier_novelty_cutoff": new_frontier_novelty_cutoff,
            },
            "caps": {
                "max_daily_hot": max_daily_hot,
                "max_new_frontier": max_new_frontier,
            },
        },
        "diagnostics": {
            "candidate_count_by_kind": {
                "new_frontier": len(new_frontier_candidates),
                "daily_hot": len(daily_hot_candidates),
            },
            "selected_count_by_kind": {
                section["kind"]: len(section["paper_ids"])
                for section in sections
            },
        },
        "papers": spotlight_mapping,
        "sections": sections,
    }


def paper_anchor_id(arxiv_id: str) -> str:
    return f"paper-{arxiv_id.replace('.', '-')}"


def topic_anchor_id(topic_id: str) -> str:
    return f"topic-{topic_id.replace('_', '-')}"
