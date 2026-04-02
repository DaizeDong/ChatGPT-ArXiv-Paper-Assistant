"""Source tier registry for hotspot quality gating.

Loads source tier definitions from configs/hotspot/source_tiers.json and
provides lookup functions used by filtering and scoring modules to enforce
lane-based eligibility and featured-topic constraints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = REPO_ROOT / "configs" / "hotspot" / "source_tiers.json"


@dataclass(frozen=True)
class SourceTier:
    source_id: str
    tier: str
    lane: str
    can_anchor_featured: bool
    freshness_required: bool = False
    max_age_days: int = 0

    @property
    def tier_rank(self) -> int:
        return TIER_HIERARCHY.get(self.tier, 0)


TIER_HIERARCHY = {
    "official": 5,
    "trusted_analysis": 4,
    "trusted_research": 3,
    "community": 2,
    "builder": 2,
    "low_trust_recap": 1,
}

# Tiers that can independently support a featured topic
FEATURED_ELIGIBLE_TIERS = {"official", "trusted_research"}

# Tiers that act as supporting evidence only
SUPPORTING_ONLY_TIERS = {"community", "builder", "low_trust_recap"}

# Role-based fallback tiers for source_ids not explicitly registered
_ROLE_FALLBACK = {
    "official_news": SourceTier(source_id="_role_official", tier="official", lane="authoritative_releases", can_anchor_featured=True),
    "research_backbone": SourceTier(source_id="_role_research", tier="trusted_research", lane="research_signals", can_anchor_featured=True),
    "paper_trending": SourceTier(source_id="_role_paper", tier="community", lane="research_signals", can_anchor_featured=False),
    "editorial_depth": SourceTier(source_id="_role_editorial", tier="trusted_analysis", lane="analysis_and_explainers", can_anchor_featured=False),
    "headline_consensus": SourceTier(source_id="_role_headline", tier="low_trust_recap", lane="community_heat", can_anchor_featured=False),
    "community_heat": SourceTier(source_id="_role_community", tier="community", lane="community_heat", can_anchor_featured=False),
    "builder_momentum": SourceTier(source_id="_role_builder", tier="builder", lane="builder_momentum", can_anchor_featured=False),
    "github_trend": SourceTier(source_id="_role_github", tier="builder", lane="builder_momentum", can_anchor_featured=False),
    "hn_discussion": SourceTier(source_id="_role_hn", tier="community", lane="community_heat", can_anchor_featured=False),
}

# Default fallback for unknown sources
_DEFAULT_TIER = SourceTier(
    source_id="unknown",
    tier="community",
    lane="community_heat",
    can_anchor_featured=False,
)


class SourceRegistry:
    """Registry of source tier definitions."""

    def __init__(self, registry_path: Path | None = None) -> None:
        self._tiers: dict[str, SourceTier] = {}
        self._roundup_overrides: dict[str, SourceTier] = {}
        path = registry_path or DEFAULT_REGISTRY_PATH
        if path.exists():
            self._load(path)

    def _load(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        for source_id, entry in data.get("sources", {}).items():
            self._tiers[source_id] = SourceTier(
                source_id=source_id,
                tier=entry.get("tier", "community"),
                lane=entry.get("lane", "community_heat"),
                can_anchor_featured=entry.get("can_anchor_featured", False),
                freshness_required=entry.get("freshness_required", False),
                max_age_days=entry.get("max_age_days", 0),
            )
        for site_id, entry in data.get("roundup_overrides", {}).items():
            self._roundup_overrides[site_id] = SourceTier(
                source_id=site_id,
                tier=entry.get("tier", "low_trust_recap"),
                lane=entry.get("lane", "community_heat"),
                can_anchor_featured=entry.get("can_anchor_featured", False),
            )

    def get(self, source_id: str, source_role: str | None = None) -> SourceTier:
        tier = self._tiers.get(source_id)
        if tier is not None:
            return tier
        if source_role:
            return _ROLE_FALLBACK.get(source_role, _DEFAULT_TIER)
        return _DEFAULT_TIER

    def get_roundup_override(self, site_id: str) -> SourceTier | None:
        return self._roundup_overrides.get(site_id)

    def can_anchor_featured(self, source_id: str, source_role: str | None = None) -> bool:
        return self.get(source_id, source_role).can_anchor_featured

    def tier_rank(self, source_id: str, source_role: str | None = None) -> int:
        return self.get(source_id, source_role).tier_rank

    def max_tier_rank_for_cluster(self, source_ids: list[str], source_roles: list[str] | None = None) -> int:
        if not source_ids:
            return 0
        roles = source_roles or []
        ranks = []
        for i, sid in enumerate(source_ids):
            role = roles[i] if i < len(roles) else None
            ranks.append(self.tier_rank(sid, role))
        return max(ranks) if ranks else 0

    def has_anchor_source(self, source_ids: list[str], source_roles: list[str] | None = None) -> bool:
        roles = source_roles or []
        for i, sid in enumerate(source_ids):
            role = roles[i] if i < len(roles) else None
            if self.can_anchor_featured(sid, role):
                return True
        return False

    def cluster_featured_eligible(self, source_ids: list[str], source_roles: list[str] | None = None) -> bool:
        """A cluster can be featured if it has an anchor-eligible source
        OR has 2+ distinct sources with at least one above community tier."""
        if self.has_anchor_source(source_ids, source_roles):
            return True
        if len(set(source_ids)) >= 2 and self.max_tier_rank_for_cluster(source_ids, source_roles) >= TIER_HIERARCHY["community"]:
            return True
        return False


# Singleton instance for convenience
_default_registry: SourceRegistry | None = None


def get_source_registry(path: Path | None = None) -> SourceRegistry:
    global _default_registry
    if _default_registry is None or path is not None:
        _default_registry = SourceRegistry(path)
    return _default_registry
