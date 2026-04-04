from __future__ import annotations

import json
import math
import os
import re
from typing import Any

import requests

from arxiv_assistant.utils.hotspot.hotspot_cluster import title_similarity, title_overlap_boost
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotCluster
from arxiv_assistant.utils.local_env import load_local_env
from arxiv_assistant.utils.pricing_loader import get_model_pricing

ALLOWED_HOTSPOT_CATEGORIES = {
    "Research",
    "Product Release",
    "Tooling",
    "Industry Update",
    "Market Signal",
}

CATEGORY_KEYWORDS = {
    "Research": {"paper", "arxiv", "training", "reasoning", "representation", "quantization", "transformer", "moe", "benchmark", "theory"},
    "Product Release": {"launch", "launches", "release", "released", "announced", "introducing", "api", "preview", "available", "acquire", "acquisition", "rollout", "model", "revamps", "debuts", "limits"},
    "Tooling": {"tool", "sdk", "framework", "cli", "workflow", "inference", "serving", "editor", "platform", "memory", "retrieval", "ocr", "design", "app"},
    "Industry Update": {"policy", "partnership", "infrastructure", "chip", "datacenter", "deployment"},
    "Market Signal": {"funding", "raise", "raised", "series", "acquisition", "valuation", "ipo", "merger", "investment", "venture", "startup", "founded"},
}

RESEARCH_TERMS = {
    "paper", "arxiv", "training", "reasoning", "representation", "quantization", "scaling",
    "transformer", "moe", "mixture of experts", "benchmark", "theory", "agent",
}
RELEASE_TERMS = {"launch", "launches", "launched", "release", "released", "announced", "announcement", "introducing", "preview", "api", "available", "acquire", "acquisition", "rollout", "model", "revamps", "debuts", "limits", "doubles"}
TOOLING_TERMS = {"tool", "sdk", "framework", "platform", "editor", "cli", "workflow", "inference", "serving", "deployment", "memory", "retrieval", "ocr", "design", "app"}
VENDOR_TERMS = {"openai", "anthropic", "google", "deepmind", "meta", "nvidia", "amazon", "apple", "cursor", "claude", "gpt", "gemini", "qwen", "deepseek", "mistral", "llama"}
NEWS_ACTION_TERMS = {"launch", "launches", "released", "release", "introducing", "revamp", "revamps", "debuts", "doubles", "limits", "acquire", "acquisition", "built", "bets", "move", "moves", "testing", "tests", "tested", "unveils", "unveiled", "announces", "announced"}

# --- Per-item frontier relevance detection ---

ENGINEERING_DISCUSSION_PATTERNS = [
    re.compile(r"\b(?:how to|step by step|getting started)\b", re.I),
    re.compile(r"\b(?:my setup|my rig|on my|fitting on)\b", re.I),
    re.compile(r"\b(?:i tested|i tried|i benchmarked|i compared|i ran)\b", re.I),
    re.compile(r"\b(?:tutorial|walkthrough|guide to|cookbook)\b", re.I),
    re.compile(r"\b(?:fine-?tuning guide|quantiz\w+ trick|inference trick)\b", re.I),
    re.compile(r"\b(?:deploy locally|run locally|local(?:ly)? run)\b", re.I),
    re.compile(r"\b(?:source code (?:leak|analysis|reverse))\b", re.I),
    re.compile(r"\b(?:leak(?:ed|s)?\b|dug through|reverse engineer|decompil)\b", re.I),
    re.compile(r"\b(?:(?:are |is )very good|isn't just for)\b", re.I),
    re.compile(r"\b(?:vs |versus |comparison)\b", re.I),
    re.compile(r"\b(?:running on|run(?:s|ning)? (?:on |with )?(?:my |a )?(?:\d+gb|rtx|gpu|mac|laptop))\b", re.I),
    re.compile(r"\b(?:unhinged|wild|insane|crazy)\b", re.I),
    re.compile(r"\b(?:analyze its own|built .* from it)\b", re.I),
    re.compile(r"\banalyzing\b.*\bsource code\b", re.I),
    re.compile(r"\blocal\b.*\b(?:llm|embedding|model)\b.*\bno (?:api|cloud)\b", re.I),
    re.compile(r"\b(?:in big trouble|dead zone)\b", re.I),
    re.compile(r"\b(?:falls? (?:right )?into)\b", re.I),
    re.compile(r"\b(?:extracted its|I extracted)\b", re.I),
    re.compile(r"\b(?:cache|kv|attention)\s+trick\b", re.I),
    re.compile(r"\blands?\s+in\s+(?:llama|vllm|pytorch|transformers)", re.I),
]

FRONTIER_NEWS_PATTERNS = [
    re.compile(r"\b(?:launch(?:es|ed)?|releas(?:es|ed)?|announc(?:es|ed)?|introducing|unveil(?:s|ed)?)\b", re.I),
    re.compile(r"\b(?:breakthrough|state[- ]of[- ]the[- ]art|novel|first[- ]ever|outperforms)\b", re.I),
    re.compile(r"\b(?:open[- ]source(?:s|d)?)\s+(?:model|release|framework)\b", re.I),
    re.compile(r"\b(?:new model|new api|new platform|new architecture)\b", re.I),
]

# --- Artifact detection: concrete outputs that matter to investors/executives ---

ARTIFACT_PATTERNS = {
    "product_release": [
        re.compile(r"\b(?:launch(?:es|ed)?|releas(?:es|ed)?|ship(?:s|ped)?|roll(?:s|ed)?\s*out)\b", re.I),
        re.compile(r"\b(?:v\d+(?:\.\d+)*|version\s+\d+|beta|preview|GA|general\s+availability)\b", re.I),
        re.compile(r"\b(?:new\s+(?:model|api|platform|sdk|feature|tool|service))\b", re.I),
        re.compile(r"\b(?:open[- ]?sourc(?:es|ed|ing))\b", re.I),
    ],
    "funding_event": [
        re.compile(r"\$\s*\d+(?:\.\d+)?\s*(?:million|billion|[mb])\b", re.I),
        re.compile(r"\b(?:series\s+[a-f]|seed\s+round|funding|fundraise|raise[ds]?)\b", re.I),
        re.compile(r"\b(?:valuation|valued\s+at|worth)\b", re.I),
        re.compile(r"\b(?:acqui(?:re[ds]?|sition)|merg(?:er|es|ed)|buyout|IPO)\b", re.I),
    ],
    "startup_event": [
        re.compile(r"\b(?:co-?found(?:er|ed)|launch(?:es|ed)?\s+(?:a\s+)?(?:startup|company|venture))\b", re.I),
        re.compile(r"\b(?:pivot(?:s|ed)?|rebrand(?:s|ed)?|spin[- ]?off)\b", re.I),
        re.compile(r"\b(?:stealth\s+(?:mode|startup)|comes?\s+out\s+of\s+stealth)\b", re.I),
    ],
    "research_breakthrough": [
        re.compile(r"\b(?:state[- ]of[- ]the[- ]art|SOTA|new\s+(?:record|benchmark))\b", re.I),
        re.compile(r"\b(?:outperforms?|surpass(?:es|ed)?|exceed(?:s|ed)?)\b", re.I),
        re.compile(r"\b(?:novel\s+(?:architecture|method|approach|technique))\b", re.I),
        re.compile(r"\b(?:first[- ]ever|breakthrough|paradigm[- ]shift)\b", re.I),
    ],
}

# --- Substance detection: opinion/clickbait/promo patterns ---

OPINION_DISCUSSION_PATTERNS = [
    re.compile(r"\b(?:what do you think|thoughts on|hot take|unpopular opinion)\b", re.I),
    re.compile(r"\b(?:am i the only|anyone else|does anyone)\b", re.I),
    re.compile(r"\b(?:overrated|underrated|overhyped)\b", re.I),
    re.compile(r"\b(?:should i|which (?:is|should)|recommend(?:ation)?s?)\b", re.I),
    re.compile(r"\b(?:my experience|my take|my opinion|personally)\b", re.I),
]

CLICKBAIT_PATTERNS = [
    re.compile(r"\b(?:you won'?t believe|mind[- ]?blow(?:n|ing)|game[- ]?chang(?:er|ing))\b", re.I),
    re.compile(r"\b(?:is dead|is dying|killer|destroys?)\b", re.I),
    re.compile(r"\b(?:secret|shocking|insane|unhinged|wild|crazy)\b", re.I),
    re.compile(r"\b(?:just leaked|got leaked|leak(?:ed|s)?)\b", re.I),
    re.compile(r"\b(?:in big trouble|dead zone)\b", re.I),
]

NEWSLETTER_PROMO_PATTERNS = [
    re.compile(r"\b(?:subscribe|sign up|join (?:our|the))\b.*\bnewsletter\b", re.I),
    re.compile(r"\b(?:top \d+ (?:tools|apps|stories)|weekly (?:digest|roundup|wrap))\b", re.I),
]

ENGINEERING_EXEMPT_ROLES = {"official_news", "editorial_depth", "research_backbone"}

_ENGINEERING_ROLE_BASE = {
    "community_heat": 0.40,
    "hn_discussion": 0.40,
    "headline_consensus": 0.30,
    "builder_momentum": 0.35,
    "github_trend": 0.25,
    "paper_trending": 0.0,
}


def _item_engineering_score(title: str, summary: str, source_role: str) -> float:
    """Return 0.0 (pure frontier news) to 1.0 (pure engineering discussion)."""
    if source_role in ENGINEERING_EXEMPT_ROLES:
        return 0.0
    text = f"{title} {summary}".lower()
    eng_hits = sum(1 for p in ENGINEERING_DISCUSSION_PATTERNS if p.search(text))
    frontier_hits = sum(1 for p in FRONTIER_NEWS_PATTERNS if p.search(text))
    score = _ENGINEERING_ROLE_BASE.get(source_role, 0.20) + eng_hits * 0.15 - frontier_hits * 0.20
    return max(0.0, min(1.0, score))


def _item_substance_score(title: str, summary: str, source_role: str) -> float:
    """Return 0.0 (substantive) to 1.0 (pure opinion/clickbait/promo)."""
    if source_role in ENGINEERING_EXEMPT_ROLES:
        return 0.0
    text = f"{title} {summary}".lower()
    opinion_hits = sum(1 for p in OPINION_DISCUSSION_PATTERNS if p.search(text))
    clickbait_hits = sum(1 for p in CLICKBAIT_PATTERNS if p.search(text))
    promo_hits = sum(1 for p in NEWSLETTER_PROMO_PATTERNS if p.search(text))
    frontier_hits = sum(1 for p in FRONTIER_NEWS_PATTERNS if p.search(text))
    score = opinion_hits * 0.25 + clickbait_hits * 0.20 + promo_hits * 0.30 - frontier_hits * 0.15
    return max(0.0, min(1.0, score))


def _cluster_artifact_score(cluster: "HotspotCluster") -> dict[str, float]:
    """Return per-category artifact confidence and an overall score."""
    items = _cluster_items(cluster)
    text_parts = [cluster.title, cluster.summary]
    for item in items[:6]:
        text_parts.append(str(item.get("title", "")))
        text_parts.append(str(item.get("summary", "")))
    text = " ".join(text_parts)

    category_scores: dict[str, float] = {}
    for category, patterns in ARTIFACT_PATTERNS.items():
        hits = sum(1 for p in patterns if p.search(text))
        category_scores[category] = min(1.0, hits / max(len(patterns) * 0.5, 1.0))

    overall = max(category_scores.values()) if category_scores else 0.0
    return {**category_scores, "overall": overall}


def calc_price(model: str, usage) -> tuple[float, float]:
    model_pricing = get_model_pricing()
    if model not in model_pricing:
        return 0.0, 0.0

    # Support both OpenAI response objects (attribute access) and raw dicts
    if isinstance(usage, dict):
        details = usage.get("prompt_tokens_details", {}) or {}
        cached_tokens = details.get("cached_tokens", 0)
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0) - cached_tokens
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    else:
        cached_tokens = getattr(usage, "model_extra", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
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


def _avg_metadata_float(cluster: HotspotCluster, *keys: str) -> float:
    values: list[float] = []
    for item in _cluster_items(cluster):
        metadata = item.get("metadata", {}) or {}
        for key in keys:
            raw_value = metadata.get(key)
            if raw_value is None:
                continue
            try:
                values.append(float(raw_value))
                break
            except (TypeError, ValueError):
                continue
    if not values:
        return 0.0
    return sum(values) / len(values)


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
    has_editorial_depth = "editorial_depth" in cluster.source_roles or _bool_signal(cluster, lambda item: item.get("source_type") == "blog_analysis")

    text = _cluster_text(cluster)
    has_research_terms = any(term in text for term in RESEARCH_TERMS)
    has_release_terms = any(term in text for term in RELEASE_TERMS)
    has_tooling_terms = any(term in text for term in TOOLING_TERMS)
    has_vendor_terms = any(term in text for term in VENDOR_TERMS)
    has_news_action_terms = any(term in text for term in NEWS_ACTION_TERMS)
    has_product_news = has_vendor_terms and has_news_action_terms

    daily_score = _max_metadata_int(cluster, "daily_score", "score")
    upvotes = _max_metadata_int(cluster, "upvotes")
    github_stars = _max_metadata_int(cluster, "github_stars", "stars")
    hn_score = _max_metadata_int(cluster, "hn_score")
    community_activity = _sum_metadata_int(cluster, "activity")
    source_quality = _avg_metadata_float(cluster, "source_quality")

    has_research_source = bool({"research_backbone", "paper_trending"} & set(cluster.source_roles))
    research_term_boost = 1.4 if has_research_terms and (has_research_source or has_paper) else (0.5 if has_research_terms else 0.0)
    frontierness = 1.6 + (1.5 if has_paper else 0.0) + research_term_boost
    frontierness += 1.5 if has_editorial_depth else 0.0
    frontierness += 2.0 if has_official and has_release_terms else (1.1 if has_official else 0.0)
    frontierness += 1.0 if has_repo and has_tooling_terms else 0.0
    frontierness += 1.0 if has_product_news and has_roundup else 0.0
    frontierness += min(1.5, daily_score / 12.0)
    frontierness += min(0.9, math.log1p(github_stars) / 3.2)
    frontierness += min(0.8, source_quality * 0.55)
    # Cap FRONTIERNESS for community-only sources (no paper, no official, no editorial)
    is_community_only = not has_paper and not has_official and not has_editorial_depth and not has_research_source
    if is_community_only:
        frontierness = min(frontierness, 4.0)
    frontierness = _clamp(frontierness)

    technical_depth = 1.8 + (1.9 if has_paper else 0.0) + (1.2 if has_research_terms else 0.0)
    technical_depth += 1.2 if has_tooling_terms else 0.0
    technical_depth += 0.9 if has_repo and has_tooling_terms else 0.0
    technical_depth += 1.0 if has_official and has_release_terms else (0.5 if has_official else 0.0)
    technical_depth += 0.7 if has_product_news and has_tooling_terms else 0.0
    technical_depth += min(1.2, daily_score / 10.0)
    technical_depth += min(1.0, math.log1p(github_stars) / 3.8)
    technical_depth += min(0.7, source_quality * 0.45)
    technical_depth = _clamp(technical_depth)

    resonance = 1.6 + min(3.0, max(source_count - 1, 0) * 1.4) + min(2.0, max(source_type_count - 1, 0) * 1.0)
    resonance += min(1.5, math.log1p(upvotes) / 1.9)
    resonance += min(1.0, math.log1p(hn_score) / 2.2)
    resonance += min(1.6, math.log1p(community_activity) / 2.0)
    resonance += 0.8 if has_roundup else 0.0
    resonance += 0.5 if has_official else 0.0
    resonance += 0.6 if has_product_news and source_count > 1 else 0.0
    resonance += min(0.7, source_quality * 0.45)
    resonance = _clamp(resonance)

    importance = 2.0 + (3.0 if has_official else 0.0) + (1.0 if has_paper else 0.0) + (1.1 if has_repo else 0.0)
    importance += 0.8 if has_editorial_depth else 0.0
    importance += 2.2 if has_official and has_release_terms else (0.9 if has_release_terms else 0.0)
    importance += 1.2 if has_product_news and has_roundup else 0.0
    importance += 0.8 if has_repo and community_activity >= 300 else 0.0
    importance += min(1.3, daily_score / 10.0)
    importance += min(1.0, math.log1p(github_stars) / 4.2)
    importance += min(0.8, math.log1p(community_activity) / 3.5)
    importance += min(0.6, source_quality * 0.35)
    importance = _clamp(importance)

    evidence_strength = 2.0 + (1.7 if source_count > 1 else 0.0) + (1.1 if source_type_count > 1 else 0.0)
    evidence_strength += 1.4 if has_official else 0.0
    evidence_strength += 1.0 if has_official and has_release_terms else 0.0
    evidence_strength += 1.2 if has_paper else 0.0
    evidence_strength += 1.0 if has_repo else 0.0
    evidence_strength += 0.8 if has_product_news and source_count > 1 else 0.0
    evidence_strength += min(1.4, math.log1p(community_activity) / 2.6) if has_roundup and (has_repo or has_release_terms or has_tooling_terms) else 0.0
    evidence_strength += min(1.0, source_quality * 0.7)
    evidence_strength = _clamp(evidence_strength)

    actionability = 1.5 + (2.2 if has_repo else 0.0) + (1.5 if has_tooling_terms else 0.0)
    actionability += 1.2 if has_official else 0.0
    actionability += 1.0 if has_official and has_release_terms else (0.8 if has_release_terms else 0.0)
    actionability += 0.9 if has_product_news else 0.0
    actionability = _clamp(actionability)

    hype_penalty = 0.3
    if has_roundup and not (has_paper or has_official or has_repo):
        hype_penalty += 4.0
    if source_count == 1 and has_roundup and not has_repo and not has_official:
        hype_penalty += 2.0
    if source_count == 1 and has_paper and not has_repo and not has_official:
        hype_penalty += 0.8
    if not (has_research_terms or has_tooling_terms or has_release_terms) and has_roundup:
        hype_penalty += 1.4
    if item_count >= 4 and source_type_count == 1 and has_roundup and not has_official:
        hype_penalty += 1.2
    if has_product_news and source_count > 1:
        hype_penalty -= 1.0
    if has_repo or has_official:
        hype_penalty -= 0.7
    if community_activity >= 500 and (has_research_terms or has_release_terms or has_tooling_terms):
        hype_penalty -= 0.6
    hype_penalty = _clamp(hype_penalty, 0.0, 10.0)

    # Engineering discussion penalty (content-based, per-item)
    eng_scores = [_item_engineering_score(str(item.get("title", "")), str(item.get("summary", "")), str(item.get("source_role", ""))) for item in items]
    avg_eng_score = sum(eng_scores) / len(eng_scores) if eng_scores else 0.0
    engineering_penalty = avg_eng_score * 5.0
    if has_official or has_editorial_depth:
        engineering_penalty *= 0.2
    if has_paper and source_count >= 2:
        engineering_penalty *= 0.4
    engineering_penalty = _clamp(engineering_penalty, 0.0, 5.0)

    # Substance penalty (opinion/clickbait/promo content)
    sub_scores = [_item_substance_score(str(item.get("title", "")), str(item.get("summary", "")), str(item.get("source_role", ""))) for item in items]
    avg_sub_score = sum(sub_scores) / len(sub_scores) if sub_scores else 0.0
    substance_penalty = avg_sub_score * 5.0
    if has_official or has_editorial_depth:
        substance_penalty *= 0.2
    if has_paper and source_count >= 2:
        substance_penalty *= 0.3
    substance_penalty = _clamp(substance_penalty, 0.0, 5.0)

    # Artifact boost (concrete deliverables: product releases, funding, breakthroughs)
    artifact_scores = _cluster_artifact_score(cluster)
    artifact_boost = artifact_scores.get("overall", 0.0)
    # Funding/startup artifacts get extra frontierness
    if artifact_scores.get("funding_event", 0.0) >= 0.5 or artifact_scores.get("startup_event", 0.0) >= 0.5:
        frontierness = min(10.0, frontierness + 1.5)
    # Community-only cap lowered further without artifact
    if is_community_only and artifact_boost < 0.3:
        frontierness = min(frontierness, 3.0)
    frontierness = _clamp(frontierness)

    confidence = 1.8 + 0.38 * evidence_strength + 0.26 * resonance + 0.18 * importance
    confidence += 1.4 if source_count > 1 else 0.0
    confidence += 0.9 if source_type_count > 1 else 0.0
    confidence += 1.0 if has_official else 0.0
    confidence += 0.6 if has_repo else 0.0
    confidence += 0.4 if has_paper else 0.0
    confidence -= 0.45 * hype_penalty
    if source_count == 1 and has_roundup and not (has_official or has_repo or has_paper):
        confidence -= 0.9
    confidence = _clamp(confidence)

    quality = round(_clamp(0.35 * frontierness + 0.27 * technical_depth + 0.18 * importance + 0.20 * evidence_strength))
    heat = round(_clamp(0.70 * resonance + 0.30 * min(10.0, 2.0 + item_count)))
    importance_score = round(_clamp(0.58 * importance + 0.22 * evidence_strength + 0.20 * frontierness))
    final_score = _clamp(
        0.30 * importance_score
        + 0.25 * quality
        + 0.15 * heat
        + 0.12 * evidence_strength
        + 0.10 * actionability
        + 0.08 * artifact_boost * 10
        - 0.06 * hype_penalty
        - 0.18 * engineering_penalty
        - 0.15 * substance_penalty,
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
        "ENGINEERING_PENALTY": round(engineering_penalty, 3),
        "SUBSTANCE_PENALTY": round(substance_penalty, 3),
        "ARTIFACT_BOOST": round(artifact_boost, 3),
        "CONFIDENCE": round(confidence, 3),
        "FINAL_SCORE": round(final_score, 3),
    }


MARKET_SIGNAL_RE = re.compile(r"\$\s*\d+(?:\.\d+)?\s*(?:million|billion|[mb])\b", re.I)


def classify_category_heuristically(cluster: HotspotCluster) -> str:
    text = _cluster_text(cluster)
    title_text = cluster.title.lower()
    scores = {
        category: sum(1 for token in CATEGORY_KEYWORDS[category] if token in text)
        for category in ALLOWED_HOTSPOT_CATEGORIES
    }
    # Check vendor + action co-occurrence in title only (not full text)
    # to avoid false positives from user comments like "I was testing..."
    has_vendor_in_title = any(term in title_text for term in VENDOR_TERMS)
    has_action_in_title = any(term in title_text for term in NEWS_ACTION_TERMS)
    has_vendor_terms = has_vendor_in_title or any(term in text for term in VENDOR_TERMS)
    has_news_action_terms = has_action_in_title or any(term in text for term in NEWS_ACTION_TERMS)
    has_product_news = has_vendor_in_title and has_action_in_title
    # Market Signal: funding/M&A with dollar amounts or strong keyword match
    if scores["Market Signal"] >= 2 or (scores["Market Signal"] >= 1 and MARKET_SIGNAL_RE.search(text)):
        return "Market Signal"
    if "github_trend" in cluster.source_roles and "paper" not in cluster.source_types:
        return "Tooling"
    if "official_news" in cluster.source_roles and (scores["Product Release"] > 0 or any(term in text for term in RELEASE_TERMS)):
        return "Product Release"
    if has_product_news and ("headline_consensus" in cluster.source_roles or "community_heat" in cluster.source_roles) and "paper_trending" not in cluster.source_roles and "research_backbone" not in cluster.source_roles:
        return "Tooling" if scores["Tooling"] > scores["Product Release"] else "Product Release"
    if scores["Research"] and ("paper" in cluster.source_types or "paper_trending" in cluster.source_roles or "research_backbone" in cluster.source_roles):
        return "Research"
    if "paper" in cluster.source_types or "paper_trending" in cluster.source_roles or "research_backbone" in cluster.source_roles:
        return "Research"
    if scores["Tooling"]:
        return "Tooling"
    if scores["Industry Update"] and "official_news" in cluster.source_roles:
        return "Industry Update"
    if "official_news" in cluster.source_roles:
        return "Industry Update"
    return "Industry Update"


def _clean_json_text(raw_text: str) -> str:
    cleaned = re.sub(r"```jsonl?\s*", "", raw_text or "")
    cleaned = re.sub(r"```", "", cleaned)
    return cleaned.strip()


def _cluster_prompt_text(cluster: HotspotCluster) -> str:
    items = _cluster_items(cluster)
    lines = [
        f"Cluster ID: {cluster.cluster_id}",
        f"Cluster Title: {cluster.title}",
        f"Deterministic Score: {cluster.deterministic_score}",
        f"Independent Sources: {len(cluster.source_ids)} ({len(cluster.source_types)} distinct types)",
        f"Source Names: {', '.join(cluster.source_names)}",
        f"Source Roles: {', '.join(cluster.source_roles)}",
        f"Source Types: {', '.join(cluster.source_types)}",
        f"Tags: {', '.join(cluster.tags)}",
        f"Representative items ({len(items)} total):",
    ]
    for item in items[:4]:
        metadata = item.get("metadata", {}) or {}
        metadata_bits = []
        for key in ("daily_score", "score", "upvotes", "github_stars", "stars", "hn_score", "is_official"):
            val = metadata.get(key)
            if val not in (None, "", 0, False):
                metadata_bits.append(f"{key}={val}")
        lines.append(
            "* "
            + " | ".join(
                filter(
                    None,
                    [
                        str(item.get("source_name", "")),
                        str(item.get("source_role", "")),
                        str(item.get("title", "")),
                        str(item.get("summary", ""))[:200],
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
    if category == "Market Signal":
        return "This represents a notable funding, acquisition, or strategic market event in the AI landscape."
    return "This topic surfaced with enough authoritative evidence to warrant tracking."


def _build_topic(cluster: HotspotCluster, *, keep: bool, watchlist: bool, category: str, quality: int, heat: int, importance: int, summary: str, why_it_matters: str) -> dict[str, Any]:
    signals = _cluster_signal_scores(cluster)
    final_score = _clamp(
        0.30 * importance
        + 0.28 * quality
        + 0.15 * heat
        + 0.08 * signals["EVIDENCE_STRENGTH"]
        + 0.07 * signals["ACTIONABILITY"]
        + 0.06 * signals["ARTIFACT_BOOST"] * 10
        + 0.04 * signals["CONFIDENCE"]
        - 0.06 * signals["HYPE_PENALTY"]
        - 0.18 * signals["ENGINEERING_PENALTY"]
        - 0.12 * signals["SUBSTANCE_PENALTY"],
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
        "ENGINEERING_PENALTY": signals["ENGINEERING_PENALTY"],
        "SUBSTANCE_PENALTY": signals["SUBSTANCE_PENALTY"],
        "ARTIFACT_BOOST": signals["ARTIFACT_BOOST"],
        "CONFIDENCE": signals["CONFIDENCE"],
        "SHORT_COMMENT": summary.strip() or _default_summary(cluster),
        "WHY_IT_MATTERS": why_it_matters.strip() or _default_why(cluster, category),
        "KEY_TAKEAWAYS": [],
        "FINAL_SCORE": round(final_score, 3),
        "published_at": cluster.published_at,
    }


def build_candidate_topics(clusters: list[HotspotCluster]) -> list[dict[str, Any]]:
    topics: list[dict[str, Any]] = []
    for cluster in clusters:
        signals = _cluster_signal_scores(cluster)
        category = classify_category_heuristically(cluster)
        topics.append(
            _build_topic(
                cluster,
                keep=False,
                watchlist=False,
                category=category,
                quality=signals["QUALITY"],
                heat=signals["HEAT"],
                importance=signals["IMPORTANCE"],
                summary=_default_summary(cluster),
                why_it_matters=_default_why(cluster, category),
            )
        )
    topics.sort(
        key=lambda row: (
            row["FINAL_SCORE"],
            row["CONFIDENCE"],
            row["EVIDENCE_STRENGTH"],
            row["CROSS_SOURCE_RESONANCE"],
            row["QUALITY"],
            row["HEAT"],
            row["IMPORTANCE"],
        ),
        reverse=True,
    )
    # Post-clustering topic dedup: merge topics with high title similarity
    topics = _dedupe_topics(topics)
    return topics


def _dedupe_topics(topics: list[dict[str, Any]], threshold: float = 0.50) -> list[dict[str, Any]]:
    """Remove near-duplicate topics, keeping the higher-scored one."""
    if len(topics) <= 1:
        return topics
    kept: list[dict[str, Any]] = []
    for topic in topics:
        title = topic.get("title", "")
        is_dup = False
        for existing in kept:
            existing_title = existing.get("title", "")
            sim = max(title_similarity(title, existing_title), title_overlap_boost(title, existing_title))
            if sim >= threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(topic)
    return kept


def heuristic_screen_clusters(clusters: list[HotspotCluster], score_cutoff: float, watchlist_cutoff: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    watchlist: list[dict[str, Any]] = []
    for topic in build_candidate_topics(clusters):
        keep = topic["FINAL_SCORE"] >= score_cutoff and topic["QUALITY"] >= 5
        watch = not keep and topic["FINAL_SCORE"] >= watchlist_cutoff
        topic["KEEP_IN_DAILY_HOTSPOTS"] = keep
        topic["WATCHLIST"] = watch
        if keep:
            kept.append(topic)
        elif watch:
            watchlist.append(topic)
    return kept, watchlist


def _validate_key_takeaways(takeaways: list[str], title: str, summary: str) -> list[str]:
    if not takeaways:
        return []
    title_lower = title.lower().strip()
    summary_lower = summary.lower().strip()
    valid = []
    for ta in takeaways:
        ta_str = str(ta).strip()
        if len(ta_str) < 15:
            continue
        ta_lower = ta_str.lower()
        # Reject takeaways that merely restate the headline or summary
        if ta_lower == title_lower or ta_lower == summary_lower:
            continue
        # Reject if >80% token overlap with title (near-restatement)
        ta_tokens = set(ta_lower.split())
        title_tokens = set(title_lower.split())
        if title_tokens and len(ta_tokens & title_tokens) / max(len(ta_tokens), 1) > 0.7:
            continue
        valid.append(ta_str)
    return valid


def _normalize_screening_row(cluster: HotspotCluster, row: dict[str, Any], score_cutoff: float, watchlist_cutoff: float) -> dict[str, Any]:
    category = row.get("CATEGORY", "").strip()
    if category not in ALLOWED_HOTSPOT_CATEGORIES:
        category = classify_category_heuristically(cluster)
    quality = int(_clamp(int(row.get("QUALITY", 0) or 0), 1, 10))
    heat = int(_clamp(int(row.get("HEAT", 0) or 0), 1, 10))
    importance = int(_clamp(int(row.get("IMPORTANCE", 0) or 0), 1, 10))
    raw_title = str(row.get("TITLE", "")).strip() or cluster.title
    raw_summary = str(row.get("SUMMARY", "")).strip() or _default_summary(cluster)
    raw_takeaways = row.get("KEY_TAKEAWAYS", [])
    if not isinstance(raw_takeaways, list):
        raw_takeaways = []
    keep = bool(row.get("KEEP", False))
    watchlist_flag = bool(row.get("WATCHLIST", False))

    # KEY_TAKEAWAYS quality gate: KEEP requires at least 2 valid takeaways
    valid_takeaways = _validate_key_takeaways(raw_takeaways, raw_title, raw_summary)
    demoted_by_takeaway_gate = False
    if keep and len(valid_takeaways) < 2:
        keep = False
        watchlist_flag = True
        demoted_by_takeaway_gate = True

    topic = _build_topic(
        cluster,
        keep=keep,
        watchlist=watchlist_flag,
        category=category,
        quality=quality,
        heat=heat,
        importance=importance,
        summary=raw_summary,
        why_it_matters=str(row.get("WHY_IT_MATTERS", "")).strip() or _default_why(cluster, category),
    )
    topic["KEY_TAKEAWAYS"] = valid_takeaways
    # Score-based re-promotion, but not if explicitly demoted by takeaway quality gate
    if not demoted_by_takeaway_gate:
        if not topic["KEEP_IN_DAILY_HOTSPOTS"] and topic["FINAL_SCORE"] >= score_cutoff:
            topic["KEEP_IN_DAILY_HOTSPOTS"] = True
            topic["WATCHLIST"] = False
    if not topic["KEEP_IN_DAILY_HOTSPOTS"] and not topic["WATCHLIST"] and topic["FINAL_SCORE"] >= watchlist_cutoff:
        topic["WATCHLIST"] = True
    return topic


def _chat_completion(model: str, messages: list[dict[str, str]], temperature: float = 0.1) -> dict[str, Any]:
    """Call the chat completions API using requests.post (compatible with API proxy)."""
    load_local_env()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    resp = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "temperature": temperature, "messages": messages},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


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
 ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float, float, int, int, int]:
    kept: list[dict[str, Any]] = []
    watchlist: list[dict[str, Any]] = []
    total_prompt_cost = 0.0
    total_completion_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_requests = 0

    for batch_start in range(0, len(clusters), batch_size):
        batch = clusters[batch_start: batch_start + batch_size]
        batch_prompt = build_screening_prompt(criteria_prompt, postfix_prompt, batch)
        parsed_rows: list[dict[str, Any]] | None = None
        last_exception: Exception | None = None
        for _ in range(max(retry_count, 1)):
            try:
                data = _chat_completion(
                    model=model,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": batch_prompt}],
                )
                raw_content = data["choices"][0]["message"]["content"] or ""
                parsed_rows = parse_jsonl_response(raw_content)
                usage = data.get("usage", {})
                prompt_cost, completion_cost = calc_price(model, usage)
                total_prompt_cost += prompt_cost
                total_completion_cost += completion_cost
                total_prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
                total_completion_tokens += int(usage.get("completion_tokens", 0) or 0)
                total_requests += 1
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
    return kept, watchlist, total_prompt_cost, total_completion_cost, total_prompt_tokens, total_completion_tokens, total_requests


DIGEST_EVIDENCE_CAP = 3


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
                "KEY_TAKEAWAYS": topic.get("KEY_TAKEAWAYS", []),
                "SOURCE_NAMES": topic["source_names"],
                "EVIDENCE": [
                    {
                        "source_name": item.get("source_name"),
                        "title": item.get("title"),
                        **({"summary": item["summary"][:300]} if item.get("summary") else {}),
                    }
                    for item in topic.get("items", [])[:DIGEST_EVIDENCE_CAP]
                ],
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
) -> tuple[dict[str, Any], float, float, int, int, int]:
    last_exception: Exception | None = None
    user_prompt = "\n\n".join([digest_prompt.strip(), _digest_prompt_payload(top_topics, watchlist)])
    for _ in range(max(retry_count, 1)):
        try:
            data = _chat_completion(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            )
            raw_content = data["choices"][0]["message"]["content"] or "{}"
            payload = parse_json_object_response(raw_content)
            usage = data.get("usage", {})
            prompt_cost, completion_cost = calc_price(model, usage)
            return (
                payload,
                prompt_cost,
                completion_cost,
                int(usage.get("prompt_tokens", 0) or 0),
                int(usage.get("completion_tokens", 0) or 0),
                1,
            )
        except Exception as ex:
            last_exception = ex
    raise RuntimeError(f"Failed to synthesize hotspot digest with OpenAI: {last_exception}")
