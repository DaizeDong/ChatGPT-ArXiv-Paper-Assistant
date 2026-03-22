from __future__ import annotations

import json
import math
import re
from typing import Any

from openai import OpenAI

from arxiv_assistant.utils.hotspot_schema import HotspotCluster
from arxiv_assistant.utils.pricing_loader import get_model_pricing

ALLOWED_HOTSPOT_CATEGORIES = {
    "Research",
    "Product Release",
    "Tooling",
    "Industry Update",
    "Community Signal",
}

CATEGORY_KEYWORDS = {
    "Research": {"paper", "arxiv", "training", "reasoning", "representation", "quantization", "transformer", "moe", "benchmark", "theory"},
    "Product Release": {"launch", "release", "released", "announced", "introducing", "api", "preview", "available", "acquire", "acquisition", "rollout", "model"},
    "Tooling": {"tool", "sdk", "framework", "cli", "workflow", "inference", "serving", "editor", "platform"},
    "Industry Update": {"policy", "funding", "partnership", "infrastructure", "chip", "datacenter", "deployment"},
    "Community Signal": {"viral", "discussion", "thread", "debate", "reaction", "trend", "community"},
}

RESEARCH_TERMS = {
    "paper", "arxiv", "training", "reasoning", "representation", "quantization", "scaling",
    "transformer", "moe", "mixture of experts", "benchmark", "theory", "agent",
}
RELEASE_TERMS = {"launch", "launched", "release", "released", "announced", "announcement", "introducing", "preview", "api", "available", "acquire", "acquisition", "rollout", "model"}
TOOLING_TERMS = {"tool", "sdk", "framework", "platform", "editor", "cli", "workflow", "inference", "serving", "deployment"}


def calc_price(model: str, usage) -> tuple[float, float]:
    model_pricing = get_model_pricing()
    if model not in model_pricing:
        return 0.0, 0.0

    cached_tokens = usage.model_extra.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    prompt_tokens = usage.prompt_tokens - cached_tokens
    completion_tokens = usage.completion_tokens
    cache_pricing = model_pricing[model].get("cache", model_pricing[model]["prompt"])
    prompt_pricing = model_pricing[model]["prompt"]
    completion_pricing = model_pricing[model]["completion"]
    return (
        cache_pricing * cached_tokens / 1_000_000 + prompt_pricing * prompt_tokens / 1_000_000,
        completion_pricing * completion_tokens / 1_000_000,
    )


def _clamp(value: float, lower: float = 1.0, upper: float = 10.0) -> float:
    return max(lower, min(upper, value))


def _cluster_items(cluster: HotspotCluster) -> list[dict[str, Any]]:
    return [item for item in cluster.items if isinstance(item, dict)]


def _cluster_text(cluster: HotspotCluster) -> str:
    parts = [cluster.title, cluster.summary, " ".join(cluster.tags)]
    for item in _cluster_items(cluster)[:6]:
        parts.append(str(item.get("title", "")))
        parts.append(str(item.get("summary", "")))
    return re.sub(r"\s+", " ", " ".join(parts)).strip().lower()


def _max_metadata_int(cluster: HotspotCluster, *keys: str) -> int:
    values: list[int] = []
    for item in _cluster_items(cluster):
        metadata = item.get("metadata", {}) or {}
        for key in keys:
            raw_value = metadata.get(key)
            if raw_value is None:
                continue
            try:
                values.append(int(raw_value))
            except (TypeError, ValueError):
                continue
    return max(values, default=0)


def _sum_metadata_int(cluster: HotspotCluster, *keys: str) -> int:
    total = 0
    for item in _cluster_items(cluster):
        metadata = item.get("metadata", {}) or {}
        for key in keys:
            raw_value = metadata.get(key)
            if raw_value is None:
                continue
            try:
                total += int(raw_value)
                break
            except (TypeError, ValueError):
                continue
    return total


def _bool_signal(cluster: HotspotCluster, predicate) -> bool:
    return any(predicate(item) for item in _cluster_items(cluster))


def _cluster_signal_scores(cluster: HotspotCluster) -> dict[str, float]:
    items = _cluster_items(cluster)
    source_count = len(cluster.source_ids)
    item_count = len(items)
    source_type_count = len(cluster.source_types)

    has_paper = _bool_signal(cluster, lambda item: item.get("source_type") == "paper" or (item.get("metadata", {}) or {}).get("arxiv_id"))
    has_repo = _bool_signal(cluster, lambda item: (item.get("metadata", {}) or {}).get("github_url") or (item.get("metadata", {}) or {}).get("github_stars") or (item.get("metadata", {}) or {}).get("stars"))
    has_official = _bool_signal(cluster, lambda item: (item.get("metadata", {}) or {}).get("is_official")) or "official_news" in cluster.source_roles
    has_roundup = _bool_signal(cluster, lambda item: item.get("source_type") == "roundup")

    text = _cluster_text(cluster)
    has_research_terms = any(term in text for term in RESEARCH_TERMS)
    has_release_terms = any(term in text for term in RELEASE_TERMS)
    has_tooling_terms = any(term in text for term in TOOLING_TERMS)

    daily_score = _max_metadata_int(cluster, "daily_score", "score")
    upvotes = _max_metadata_int(cluster, "upvotes")
    github_stars = _max_metadata_int(cluster, "github_stars", "stars")
    hn_score = _max_metadata_int(cluster, "hn_score")
    community_activity = _sum_metadata_int(cluster, "activity")

    frontierness = 1.6 + (2.1 if has_paper else 0.0) + (1.4 if has_research_terms else 0.0)
    frontierness += 2.0 if has_official and has_release_terms else (1.1 if has_official else 0.0)
    frontierness += 1.0 if has_repo and has_tooling_terms else 0.0
    frontierness += min(1.5, daily_score / 12.0)
    frontierness += min(0.9, math.log1p(github_stars) / 3.2)
    frontierness = _clamp(frontierness)

    technical_depth = 1.8 + (1.9 if has_paper else 0.0) + (1.2 if has_research_terms else 0.0)
    technical_depth += 1.2 if has_tooling_terms else 0.0
    technical_depth += 0.9 if has_repo and has_tooling_terms else 0.0
    technical_depth += 1.0 if has_official and has_release_terms else (0.5 if has_official else 0.0)
    technical_depth += min(1.2, daily_score / 10.0)
    technical_depth += min(1.0, math.log1p(github_stars) / 3.8)
    technical_depth = _clamp(technical_depth)

    resonance = 1.6 + min(3.0, max(source_count - 1, 0) * 1.15) + min(1.6, max(source_type_count - 1, 0) * 0.8)
    resonance += min(1.5, math.log1p(upvotes) / 1.9)
    resonance += min(1.0, math.log1p(hn_score) / 2.2)
    resonance += min(1.6, math.log1p(community_activity) / 2.0)
    resonance += 0.8 if has_roundup else 0.0
    resonance += 0.5 if has_official else 0.0
    resonance = _clamp(resonance)

    importance = 2.0 + (2.3 if has_official else 0.0) + (1.3 if has_paper else 0.0) + (1.1 if has_repo else 0.0)
    importance += 2.2 if has_official and has_release_terms else (0.9 if has_release_terms else 0.0)
    importance += 0.8 if has_repo and community_activity >= 300 else 0.0
    importance += min(1.3, daily_score / 10.0)
    importance += min(1.0, math.log1p(github_stars) / 4.2)
    importance += min(0.8, math.log1p(community_activity) / 3.5)
    importance = _clamp(importance)

    evidence_strength = 2.0 + (1.7 if source_count > 1 else 0.0) + (1.1 if source_type_count > 1 else 0.0)
    evidence_strength += 1.4 if has_official else 0.0
    evidence_strength += 1.0 if has_official and has_release_terms else 0.0
    evidence_strength += 1.2 if has_paper else 0.0
    evidence_strength += 1.0 if has_repo else 0.0
    evidence_strength += min(1.4, math.log1p(community_activity) / 2.6) if has_roundup and (has_repo or has_release_terms or has_tooling_terms) else 0.0
    evidence_strength = _clamp(evidence_strength)

    actionability = 1.5 + (2.2 if has_repo else 0.0) + (1.5 if has_tooling_terms else 0.0)
    actionability += 1.2 if has_official else 0.0
    actionability += 1.0 if has_official and has_release_terms else (0.8 if has_release_terms else 0.0)
    actionability = _clamp(actionability)

    hype_penalty = 0.3
    if has_roundup and not (has_paper or has_official or has_repo):
        hype_penalty += 3.4
    if source_count == 1 and has_roundup and not has_repo and not has_official:
        hype_penalty += 1.6
    if source_count == 1 and has_paper and not has_repo and not has_official:
        hype_penalty += 0.8
    if not (has_research_terms or has_tooling_terms or has_release_terms) and has_roundup:
        hype_penalty += 1.4
    if item_count >= 4 and source_type_count == 1 and has_roundup and not has_official:
        hype_penalty += 1.2
    if has_repo or has_official:
        hype_penalty -= 0.7
    if community_activity >= 500 and (has_research_terms or has_release_terms or has_tooling_terms):
        hype_penalty -= 0.6
    hype_penalty = _clamp(hype_penalty, 0.0, 10.0)

    quality = round(_clamp(0.35 * frontierness + 0.27 * technical_depth + 0.18 * importance + 0.20 * evidence_strength))
    heat = round(_clamp(0.70 * resonance + 0.30 * min(10.0, 2.0 + item_count)))
    importance_score = round(_clamp(0.58 * importance + 0.22 * evidence_strength + 0.20 * frontierness))
    final_score = _clamp(
        0.28 * quality + 0.28 * heat + 0.24 * importance_score + 0.10 * actionability + 0.10 * evidence_strength - 0.05 * hype_penalty,
        0.0,
        10.0,
    )

    return {
        "QUALITY": int(quality),
        "HEAT": int(heat),
        "IMPORTANCE": int(importance_score),
        "FRONTIERNESS": round(frontierness, 3),
        "TECHNICAL_DEPTH": round(technical_depth, 3),
        "CROSS_SOURCE_RESONANCE": round(resonance, 3),
        "ACTIONABILITY": round(actionability, 3),
        "EVIDENCE_STRENGTH": round(evidence_strength, 3),
        "HYPE_PENALTY": round(hype_penalty, 3),
        "FINAL_SCORE": round(final_score, 3),
    }


def classify_category_heuristically(cluster: HotspotCluster) -> str:
    text = _cluster_text(cluster)
    scores = {
        category: sum(1 for token in CATEGORY_KEYWORDS[category] if token in text)
        for category in ALLOWED_HOTSPOT_CATEGORIES
    }
    if "official_news" in cluster.source_roles and (scores["Product Release"] > 0 or any(term in text for term in RELEASE_TERMS)):
        return "Product Release"
    if scores["Research"] and ("paper" in cluster.source_types or "paper_trending" in cluster.source_roles or "research_backbone" in cluster.source_roles):
        return "Research"
    if scores["Tooling"]:
        return "Tooling"
    if scores["Industry Update"] and "official_news" in cluster.source_roles:
        return "Industry Update"
    if scores["Community Signal"]:
        return "Community Signal"
    if "paper" in cluster.source_types:
        return "Research"
    if "official_news" in cluster.source_roles:
        return "Industry Update"
    return "Community Signal"


def _clean_json_text(raw_text: str) -> str:
    cleaned = re.sub(r"```jsonl?\s*", "", raw_text or "")
    cleaned = re.sub(r"```", "", cleaned)
    return cleaned.strip()


def _cluster_prompt_text(cluster: HotspotCluster) -> str:
    lines = [
        f"Cluster ID: {cluster.cluster_id}",
        f"Cluster Title: {cluster.title}",
        f"Deterministic Score: {cluster.deterministic_score}",
        f"Source Names: {', '.join(cluster.source_names)}",
        f"Source Roles: {', '.join(cluster.source_roles)}",
        f"Source Types: {', '.join(cluster.source_types)}",
        f"Tags: {', '.join(cluster.tags)}",
        "Representative items:",
    ]
    for item in _cluster_items(cluster)[:6]:
        metadata = item.get("metadata", {}) or {}
        metadata_bits = []
        for key in ("daily_score", "score", "upvotes", "github_stars", "stars", "hn_score"):
            if metadata.get(key) not in (None, "", 0):
                metadata_bits.append(f"{key}={metadata[key]}")
        lines.append(
            "- "
            + " | ".join(
                filter(
                    None,
                    [
                        str(item.get("source_name", "")),
                        str(item.get("source_role", "")),
                        str(item.get("title", "")),
                        str(item.get("summary", ""))[:320],
                        ", ".join(metadata_bits),
                    ],
                )
            )
        )
    return "\n".join(lines)


def build_screening_prompt(criteria_prompt: str, postfix_prompt: str, clusters: list[HotspotCluster]) -> str:
    return "\n\n".join([criteria_prompt.strip(), "## Candidate Clusters", "\n\n".join(_cluster_prompt_text(cluster) for cluster in clusters), postfix_prompt.strip()])


def parse_jsonl_response(raw_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in _clean_json_text(raw_text).splitlines():
        line = line.strip().rstrip(",")
        if line:
            rows.append(json.loads(line))
    return rows


def parse_json_object_response(raw_text: str) -> dict[str, Any]:
    return json.loads(_clean_json_text(raw_text))


def _default_summary(cluster: HotspotCluster) -> str:
    if cluster.summary:
        return cluster.summary
    first_item = _cluster_items(cluster)[0] if _cluster_items(cluster) else {}
    return str(first_item.get("summary", "")) or str(first_item.get("title", "")) or cluster.title


def _default_why(cluster: HotspotCluster, category: str) -> str:
    source_names = ", ".join(cluster.source_names[:3])
    if category == "Research":
        return f"This topic combines substantive research signal with visible attention across {source_names}."
    if category == "Product Release":
        return "This looks like a meaningful model or product release with credible supporting evidence."
    if category == "Tooling":
        return "This topic stands out as a practical tooling or workflow update with strong builder interest."
    if category == "Industry Update":
        return "This appears to be a substantive ecosystem update, not just chatter, based on the available evidence."
    return "This cluster surfaced repeatedly enough to warrant tracking, even if the evidence is still maturing."


def _build_topic(cluster: HotspotCluster, *, keep: bool, watchlist: bool, category: str, quality: int, heat: int, importance: int, summary: str, why_it_matters: str) -> dict[str, Any]:
    signals = _cluster_signal_scores(cluster)
    final_score = _clamp(
        0.34 * quality + 0.30 * heat + 0.24 * importance + 0.08 * signals["ACTIONABILITY"] + 0.04 * signals["EVIDENCE_STRENGTH"] - 0.08 * signals["HYPE_PENALTY"],
        0.0,
        10.0,
    )
    return {
        "TOPIC_ID": cluster.cluster_id,
        "cluster_id": cluster.cluster_id,
        "title": cluster.title,
        "summary": summary.strip() or _default_summary(cluster),
        "items": _cluster_items(cluster)[:4],
        "source_ids": cluster.source_ids,
        "source_names": cluster.source_names,
        "source_roles": cluster.source_roles,
        "source_types": cluster.source_types,
        "tags": cluster.tags,
        "PRIMARY_CATEGORY": category,
        "SECONDARY_CATEGORIES": [],
        "KEEP_IN_DAILY_HOTSPOTS": keep,
        "WATCHLIST": watchlist,
        "QUALITY": int(quality),
        "HEAT": int(heat),
        "IMPORTANCE": int(importance),
        "FRONTIERNESS": signals["FRONTIERNESS"],
        "TECHNICAL_DEPTH": signals["TECHNICAL_DEPTH"],
        "CROSS_SOURCE_RESONANCE": signals["CROSS_SOURCE_RESONANCE"],
        "ACTIONABILITY": signals["ACTIONABILITY"],
        "EVIDENCE_STRENGTH": signals["EVIDENCE_STRENGTH"],
        "HYPE_PENALTY": signals["HYPE_PENALTY"],
        "SHORT_COMMENT": summary.strip() or _default_summary(cluster),
        "WHY_IT_MATTERS": why_it_matters.strip() or _default_why(cluster, category),
        "FINAL_SCORE": round(final_score, 3),
        "published_at": cluster.published_at,
    }


def heuristic_screen_clusters(clusters: list[HotspotCluster], score_cutoff: float, watchlist_cutoff: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    watchlist: list[dict[str, Any]] = []
    for cluster in clusters:
        signals = _cluster_signal_scores(cluster)
        category = classify_category_heuristically(cluster)
        keep = signals["FINAL_SCORE"] >= score_cutoff and signals["QUALITY"] >= 5
        watch = not keep and signals["FINAL_SCORE"] >= watchlist_cutoff
        topic = _build_topic(
            cluster,
            keep=keep,
            watchlist=watch,
            category=category,
            quality=signals["QUALITY"],
            heat=signals["HEAT"],
            importance=signals["IMPORTANCE"],
            summary=_default_summary(cluster),
            why_it_matters=_default_why(cluster, category),
        )
        if keep:
            kept.append(topic)
        elif watch:
            watchlist.append(topic)
    kept.sort(
        key=lambda row: (
            row["FINAL_SCORE"],
            row["EVIDENCE_STRENGTH"],
            row["CROSS_SOURCE_RESONANCE"],
            row["QUALITY"],
            row["HEAT"],
            row["IMPORTANCE"],
        ),
        reverse=True,
    )
    watchlist.sort(
        key=lambda row: (
            row["FINAL_SCORE"],
            row["EVIDENCE_STRENGTH"],
            row["CROSS_SOURCE_RESONANCE"],
            row["QUALITY"],
            row["HEAT"],
            row["IMPORTANCE"],
        ),
        reverse=True,
    )
    return kept, watchlist


def _normalize_screening_row(cluster: HotspotCluster, row: dict[str, Any], score_cutoff: float, watchlist_cutoff: float) -> dict[str, Any]:
    category = row.get("CATEGORY", "").strip()
    if category not in ALLOWED_HOTSPOT_CATEGORIES:
        category = classify_category_heuristically(cluster)
    quality = int(_clamp(int(row.get("QUALITY", 0) or 0), 1, 10))
    heat = int(_clamp(int(row.get("HEAT", 0) or 0), 1, 10))
    importance = int(_clamp(int(row.get("IMPORTANCE", 0) or 0), 1, 10))
    topic = _build_topic(
        cluster,
        keep=bool(row.get("KEEP", False)),
        watchlist=bool(row.get("WATCHLIST", False)),
        category=category,
        quality=quality,
        heat=heat,
        importance=importance,
        summary=str(row.get("SUMMARY", "")).strip() or _default_summary(cluster),
        why_it_matters=str(row.get("WHY_IT_MATTERS", "")).strip() or _default_why(cluster, category),
    )
    if not topic["KEEP_IN_DAILY_HOTSPOTS"] and topic["FINAL_SCORE"] >= score_cutoff:
        topic["KEEP_IN_DAILY_HOTSPOTS"] = True
        topic["WATCHLIST"] = False
    if not topic["KEEP_IN_DAILY_HOTSPOTS"] and not topic["WATCHLIST"] and topic["FINAL_SCORE"] >= watchlist_cutoff:
        topic["WATCHLIST"] = True
    return topic


def screen_clusters_with_openai(
    clusters: list[HotspotCluster],
    system_prompt: str,
    criteria_prompt: str,
    postfix_prompt: str,
    model: str,
    batch_size: int,
    retry_count: int,
    score_cutoff: float,
    watchlist_cutoff: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float, float]:
    client = OpenAI()
    kept: list[dict[str, Any]] = []
    watchlist: list[dict[str, Any]] = []
    total_prompt_cost = 0.0
    total_completion_cost = 0.0

    for batch_start in range(0, len(clusters), batch_size):
        batch = clusters[batch_start: batch_start + batch_size]
        batch_prompt = build_screening_prompt(criteria_prompt, postfix_prompt, batch)
        parsed_rows: list[dict[str, Any]] | None = None
        last_exception: Exception | None = None
        for _ in range(max(retry_count, 1)):
            try:
                response = client.chat.completions.create(
                    model=model,
                    temperature=0.1,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": batch_prompt}],
                )
                parsed_rows = parse_jsonl_response(response.choices[0].message.content or "")
                prompt_cost, completion_cost = calc_price(model, response.usage)
                total_prompt_cost += prompt_cost
                total_completion_cost += completion_cost
                break
            except Exception as ex:
                last_exception = ex
        if parsed_rows is None:
            print(f"Warning: hotspot screening batch failed, falling back to heuristic: {last_exception}")
            fallback_kept, fallback_watchlist = heuristic_screen_clusters(batch, score_cutoff, watchlist_cutoff)
            kept.extend(fallback_kept)
            watchlist.extend(fallback_watchlist)
            continue
        rows_by_cluster = {str(row.get("CLUSTER_ID", "")).strip(): row for row in parsed_rows if str(row.get("CLUSTER_ID", "")).strip()}
        for cluster in batch:
            row = rows_by_cluster.get(cluster.cluster_id)
            if row is None:
                fallback_kept, fallback_watchlist = heuristic_screen_clusters([cluster], score_cutoff, watchlist_cutoff)
                kept.extend(fallback_kept)
                watchlist.extend(fallback_watchlist)
                continue
            topic = _normalize_screening_row(cluster, row, score_cutoff, watchlist_cutoff)
            if topic["KEEP_IN_DAILY_HOTSPOTS"]:
                kept.append(topic)
            elif topic["WATCHLIST"]:
                watchlist.append(topic)
    kept.sort(
        key=lambda row: (
            row["FINAL_SCORE"],
            row["EVIDENCE_STRENGTH"],
            row["CROSS_SOURCE_RESONANCE"],
            row["QUALITY"],
            row["HEAT"],
            row["IMPORTANCE"],
        ),
        reverse=True,
    )
    watchlist.sort(
        key=lambda row: (
            row["FINAL_SCORE"],
            row["EVIDENCE_STRENGTH"],
            row["CROSS_SOURCE_RESONANCE"],
            row["QUALITY"],
            row["HEAT"],
            row["IMPORTANCE"],
        ),
        reverse=True,
    )
    return kept, watchlist, total_prompt_cost, total_completion_cost


def _digest_prompt_payload(top_topics: list[dict[str, Any]], watchlist: list[dict[str, Any]]) -> str:
    payload = {
        "top_topics": [
            {
                "TOPIC_ID": topic["TOPIC_ID"],
                "TITLE": topic["title"],
                "CATEGORY": topic["PRIMARY_CATEGORY"],
                "QUALITY": topic["QUALITY"],
                "HEAT": topic["HEAT"],
                "IMPORTANCE": topic["IMPORTANCE"],
                "WHY_IT_MATTERS": topic["WHY_IT_MATTERS"],
                "SOURCE_NAMES": topic["source_names"],
                "EVIDENCE": [{"source_name": item.get("source_name"), "title": item.get("title"), "url": item.get("url")} for item in topic.get("items", [])],
            }
            for topic in top_topics
        ],
        "watchlist": [
            {"TOPIC_ID": topic["TOPIC_ID"], "TITLE": topic["title"], "CATEGORY": topic["PRIMARY_CATEGORY"], "WHY_IT_MATTERS": topic["WHY_IT_MATTERS"]}
            for topic in watchlist
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def synthesize_digest_with_openai(
    top_topics: list[dict[str, Any]],
    watchlist: list[dict[str, Any]],
    system_prompt: str,
    digest_prompt: str,
    model: str,
    retry_count: int,
) -> tuple[dict[str, Any], float, float]:
    client = OpenAI()
    last_exception: Exception | None = None
    user_prompt = "\n\n".join([digest_prompt.strip(), _digest_prompt_payload(top_topics, watchlist)])
    for _ in range(max(retry_count, 1)):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0.1,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            )
            payload = parse_json_object_response(response.choices[0].message.content or "{}")
            prompt_cost, completion_cost = calc_price(model, response.usage)
            return payload, prompt_cost, completion_cost
        except Exception as ex:
            last_exception = ex
    raise RuntimeError(f"Failed to synthesize hotspot digest with OpenAI: {last_exception}")
