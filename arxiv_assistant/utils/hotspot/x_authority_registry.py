from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from arxiv_assistant.utils.hotspot.hotspot_sources import fetch_json, fetch_text


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SEED_PATH = REPO_ROOT / "configs" / "hotspot" / "x_authority_seeds.json"
DEFAULT_SNAPSHOT_PATH = REPO_ROOT / "configs" / "hotspot" / "x_authority_inventory.json"
FOLLOW_THE_AI_LEADERS_URL = "https://raw.githubusercontent.com/mattnigh/follow-the-ai-leaders/main/README.md"
PAPERPULSE_AUTHORS_URL = "https://www.paperpulse.ai/api/researcher-feed/authors"
X_USERS_BY_USERNAME_URL = "https://api.x.com/2/users/by/username/{username}"
X_USER_FOLLOWING_URL = "https://api.x.com/2/users/{user_id}/following"
BEARER_TOKEN_ENV_KEYS = ("X_BEARER_TOKEN", "X_API_BEARER_TOKEN", "TWITTER_BEARER_TOKEN")
TWITTER_URL_HANDLE_PATTERN = re.compile(r"https?://(?:x|twitter)\.com/([A-Za-z0-9_]{2,30})", re.I)
AI_PROFILE_PATTERNS = (
    re.compile(r"\bai\b", re.I),
    re.compile(r"\bml\b", re.I),
    re.compile(r"\bllm\b", re.I),
    re.compile(r"machine learning", re.I),
    re.compile(r"deep learning", re.I),
    re.compile(r"language model", re.I),
    re.compile(r"foundation model", re.I),
    re.compile(r"generative", re.I),
    re.compile(r"research scientist", re.I),
    re.compile(r"scientist", re.I),
    re.compile(r"professor", re.I),
    re.compile(r"postdoc", re.I),
    re.compile(r"phd", re.I),
    re.compile(r"robotics", re.I),
    re.compile(r"computer vision", re.I),
    re.compile(r"nlp", re.I),
    re.compile(r"alignment", re.I),
    re.compile(r"agents?", re.I),
    re.compile(r"reasoning", re.I),
    re.compile(r"inference", re.I),
    re.compile(r"training", re.I),
    re.compile(r"multimodal", re.I),
    re.compile(r"openai", re.I),
    re.compile(r"anthropic", re.I),
    re.compile(r"deepmind", re.I),
    re.compile(r"google ai", re.I),
    re.compile(r"google deepmind", re.I),
    re.compile(r"meta ai", re.I),
    re.compile(r"hugging face", re.I),
    re.compile(r"nvidia", re.I),
    re.compile(r"mistral", re.I),
    re.compile(r"cohere", re.I),
    re.compile(r"cursor", re.I),
    re.compile(r"perplexity", re.I),
)
COMPANY_PATTERNS = (
    re.compile(r"\bofficial\b", re.I),
    re.compile(r"\bcompany\b", re.I),
    re.compile(r"\bplatform\b", re.I),
    re.compile(r"\blab\b", re.I),
    re.compile(r"\bteam\b", re.I),
    re.compile(r"\bproduct\b", re.I),
)
RESEARCHER_PATTERNS = (
    re.compile(r"\bresearch scientist\b", re.I),
    re.compile(r"\bscientist\b", re.I),
    re.compile(r"\bengineer\b", re.I),
    re.compile(r"\bprofessor\b", re.I),
    re.compile(r"\bpostdoc\b", re.I),
    re.compile(r"\bphd\b", re.I),
    re.compile(r"\bfounder\b", re.I),
    re.compile(r"\bco-founder\b", re.I),
    re.compile(r"\bcto\b", re.I),
    re.compile(r"\bceo\b", re.I),
)
CORE_PRACTITIONER_PATTERNS = RESEARCHER_PATTERNS + (
    re.compile(r"\blab\b", re.I),
    re.compile(r"\bresearch\b", re.I),
    re.compile(r"\bmodel(s)?\b", re.I),
    re.compile(r"\bagent(s)?\b", re.I),
    re.compile(r"\binference\b", re.I),
    re.compile(r"\btraining\b", re.I),
    re.compile(r"\brobotics\b", re.I),
)
NON_AI_PATTERNS = (
    re.compile(r"\bcrypto\b", re.I),
    re.compile(r"\bweb3\b", re.I),
    re.compile(r"\bmarketing\b", re.I),
    re.compile(r"\breal estate\b", re.I),
    re.compile(r"\bfitness\b", re.I),
    re.compile(r"\bventure\b", re.I),
    re.compile(r"\bvc\b", re.I),
    re.compile(r"\binvestor\b", re.I),
    re.compile(r"\bnewsletter\b", re.I),
    re.compile(r"\bpodcast\b", re.I),
    re.compile(r"\bjournalist\b", re.I),
    re.compile(r"\breporter\b", re.I),
    re.compile(r"\bwriter\b", re.I),
)
HARD_EXCLUDE_PATTERNS = (
    re.compile(r"\bventure\b", re.I),
    re.compile(r"\bvc\b", re.I),
    re.compile(r"\binvestor\b", re.I),
    re.compile(r"\bpodcast\b", re.I),
    re.compile(r"\bnewsletter\b", re.I),
    re.compile(r"\byoutube\b", re.I),
    re.compile(r"\bcreator\b", re.I),
    re.compile(r"\bjournalist\b", re.I),
    re.compile(r"\breporter\b", re.I),
    re.compile(r"\bwriter\b", re.I),
)


def _normalize_handle(handle: str | None) -> str:
    return re.sub(r"[^a-z0-9_]+", "", (handle or "").strip().lower())


def _get_bearer_token() -> str | None:
    for key in BEARER_TOKEN_ENV_KEYS:
        value = (os.getenv(key) or "").strip()
        if value:
            return value
    return None


def _load_seed_payload(seed_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(seed_path) if seed_path else DEFAULT_SEED_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported X authority seed payload: {path}")
    return payload


def _parse_follow_handles(markdown_text: str) -> set[str]:
    return {
        _normalize_handle(match.group(1))
        for match in TWITTER_URL_HANDLE_PATTERN.finditer(markdown_text or "")
        if _normalize_handle(match.group(1))
    }


def _fetch_follow_the_ai_leaders_handles(url: str = FOLLOW_THE_AI_LEADERS_URL) -> set[str]:
    return _parse_follow_handles(fetch_text(url))


def _fetch_paperpulse_author_handles(url: str = PAPERPULSE_AUTHORS_URL) -> set[str]:
    payload = fetch_json(url)
    if isinstance(payload, dict):
        authors = payload.get("authors", [])
    else:
        authors = []
    return {
        _normalize_handle(author)
        for author in authors
        if _normalize_handle(author)
    }


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _is_registry_fresh(registry_path: Path, max_age_hours: int) -> bool:
    if not registry_path.exists():
        return False
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    generated_at = _parse_timestamp(str(payload.get("generated_at") or ""))
    if generated_at is None:
        return False
    return generated_at >= datetime.now(UTC) - timedelta(hours=max_age_hours)


def _registry_payload_for_compare(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "generated_at"}


def _seed_account_map(seed_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    account_map: dict[str, dict[str, Any]] = {}
    for row in seed_payload.get("accounts", []):
        handle = _normalize_handle(str(row.get("handle", "")))
        if not handle:
            continue
        source_refs = list(row.get("source_refs") or [])
        if "manual_seed" not in source_refs:
            source_refs.append("manual_seed")
        account_map[handle] = {
            "handle": handle,
            "name": str(row.get("name") or handle),
            "organization": str(row.get("organization") or ""),
            "kind": str(row.get("kind") or "researcher"),
            "tier": int(row.get("tier") or 1),
            "active": bool(row.get("active", True)),
            "source_refs": sorted(set(source_refs)),
        }
    return account_map


def _merge_seed_handles(
    account_map: dict[str, dict[str, Any]],
    handles: set[str],
    *,
    source_ref: str,
    kind: str = "researcher",
    active_tier: int = 2,
    watchlist_tier: int = 1,
    activate: bool = False,
) -> None:
    for handle in sorted(handles):
        record = account_map.get(handle)
        if record is None:
            record = {
                "handle": handle,
                "name": handle,
                "organization": "",
                "kind": kind,
                "tier": watchlist_tier,
                "active": False,
                "source_refs": [],
            }
            account_map[handle] = record
        if source_ref not in record["source_refs"]:
            record["source_refs"].append(source_ref)
        if record.get("kind") != "official":
            record["kind"] = kind
        if activate:
            record["active"] = True
            record["tier"] = max(int(record.get("tier") or 1), active_tier)
        else:
            record["tier"] = max(int(record.get("tier") or 1), watchlist_tier)


def _fetch_x_user(handle: str, *, bearer_token: str) -> dict[str, Any] | None:
    payload = fetch_json(
        X_USERS_BY_USERNAME_URL.format(username=handle),
        headers={"Authorization": f"Bearer {bearer_token}"},
        params={"user.fields": "description,public_metrics,verified,verified_type,location"},
    )
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        return payload["data"]
    return None


def _fetch_x_following(
    user_id: str,
    *,
    bearer_token: str,
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    next_token: str | None = None
    remaining = limit
    while remaining > 0:
        params: dict[str, Any] = {
            "max_results": min(100, remaining),
            "user.fields": "description,public_metrics,verified,verified_type,location",
        }
        if next_token:
            params["pagination_token"] = next_token
        payload = fetch_json(
            X_USER_FOLLOWING_URL.format(user_id=user_id),
            headers={"Authorization": f"Bearer {bearer_token}"},
            params=params,
        )
        batch = payload.get("data", []) or []
        rows.extend(batch)
        remaining = limit - len(rows)
        next_token = (payload.get("meta") or {}).get("next_token")
        if not next_token:
            break
    return rows


def _graph_config(seed_payload: dict[str, Any]) -> dict[str, Any]:
    return dict(seed_payload.get("following_graph") or {})


def _profile_text(user: dict[str, Any]) -> str:
    return " ".join(
        part.strip()
        for part in [
            str(user.get("name") or ""),
            str(user.get("username") or ""),
            str(user.get("description") or ""),
            str(user.get("location") or ""),
        ]
        if str(part or "").strip()
    )


def _is_ai_profile(user: dict[str, Any]) -> bool:
    text = _profile_text(user)
    if not text:
        return False
    if any(pattern.search(text) for pattern in HARD_EXCLUDE_PATTERNS):
        return False
    has_ai_terms = any(pattern.search(text) for pattern in AI_PROFILE_PATTERNS)
    has_core_terms = any(pattern.search(text) for pattern in CORE_PRACTITIONER_PATTERNS)
    if not has_ai_terms:
        return False
    if any(pattern.search(text) for pattern in NON_AI_PATTERNS) and not has_core_terms:
        return False
    return True


def _is_core_ai_actor(user: dict[str, Any]) -> bool:
    text = _profile_text(user)
    name = str(user.get("name") or "").strip()
    username = str(user.get("username") or "").strip().lower()
    verified_type = str(user.get("verified_type") or "")
    looks_like_person = " " in name and not any(char.isdigit() for char in name)
    handle_brand_like = any(token in username for token in ("ai", "labs", "lab", "app", "studio", "research", "models", "robotics"))
    if verified_type in {"business", "government"} and _is_ai_profile(user):
        return True
    if any(pattern.search(text) for pattern in CORE_PRACTITIONER_PATTERNS):
        return True
    if handle_brand_like and not looks_like_person and _is_ai_profile(user):
        return True
    return False


def _classify_graph_candidate(user: dict[str, Any]) -> str:
    text = _profile_text(user)
    name = str(user.get("name") or "").strip()
    username = str(user.get("username") or "").strip().lower()
    verified_type = str(user.get("verified_type") or "")
    if verified_type in {"business", "government"}:
        return "company"
    looks_like_person = " " in name and not any(char.isdigit() for char in name)
    handle_brand_like = any(token in username for token in ("ai", "labs", "lab", "app", "studio", "research", "models"))
    if any(pattern.search(text) for pattern in COMPANY_PATTERNS) and not looks_like_person and (handle_brand_like or not any(pattern.search(text) for pattern in RESEARCHER_PATTERNS)):
        return "company"
    return "researcher"


def _score_graph_candidate(row: dict[str, Any]) -> float:
    public_metrics = row.get("public_metrics") or {}
    followers = int(public_metrics.get("followers_count") or 0)
    listed = int(public_metrics.get("listed_count") or 0)
    support = len(row.get("seed_support", []))
    support_kind_count = len(row.get("seed_kinds", []))
    verified = bool(row.get("verified"))
    ai_profile = bool(row.get("ai_profile"))
    score = 0.0
    score += support * 3.2
    score += support_kind_count * 0.9
    score += min(2.5, listed / 150.0)
    score += min(2.0, followers / 25000.0)
    if verified:
        score += 0.8
    if ai_profile:
        score += 1.6
    return round(score, 3)


def _seed_handles_for_graph(seed_payload: dict[str, Any], account_map: dict[str, dict[str, Any]]) -> list[str]:
    graph_cfg = _graph_config(seed_payload)
    explicit = [_normalize_handle(handle) for handle in graph_cfg.get("seed_handles", []) if _normalize_handle(handle)]
    if explicit:
        return explicit[: int(graph_cfg.get("max_seeds", len(explicit)) or len(explicit))]
    fallback = [
        handle
        for handle, row in account_map.items()
        if bool(row.get("active")) and int(row.get("tier") or 0) >= 2
    ]
    return fallback[: int(graph_cfg.get("max_seeds", 12))]


def _expand_following_graph(
    seed_payload: dict[str, Any],
    account_map: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, str]]]:
    bearer_token = _get_bearer_token()
    if not bearer_token:
        return [], {"enabled": False, "reason": "missing_bearer_token"}, []

    graph_cfg = _graph_config(seed_payload)
    max_following_per_seed = int(graph_cfg.get("max_following_per_seed", 200) or 200)
    min_support_count = int(graph_cfg.get("min_support_count", 1) or 1)
    min_active_support_count = int(graph_cfg.get("min_active_support_count", 2) or 2)
    min_watchlist_score = float(graph_cfg.get("min_watchlist_score", 4.2) or 4.2)
    min_active_score = float(graph_cfg.get("min_active_score", 6.4) or 6.4)
    min_followers = int(graph_cfg.get("min_followers_count", 1000) or 1000)
    min_listed = int(graph_cfg.get("min_listed_count", 20) or 20)

    errors: list[dict[str, str]] = []
    aggregated: dict[str, dict[str, Any]] = {}
    seed_handles = _seed_handles_for_graph(seed_payload, account_map)
    resolved_seed_count = 0
    total_edges = 0

    for handle in seed_handles:
        seed_record = account_map.get(handle, {})
        try:
            user = _fetch_x_user(handle, bearer_token=bearer_token)
            if user is None:
                continue
            resolved_seed_count += 1
            following_rows = _fetch_x_following(str(user.get("id")), bearer_token=bearer_token, limit=max_following_per_seed)
        except Exception as ex:
            errors.append({"source": f"following:{handle}", "error": str(ex)})
            continue

        seed_kind = str(seed_record.get("kind") or "researcher")
        for candidate in following_rows:
            candidate_handle = _normalize_handle(str(candidate.get("username") or ""))
            if not candidate_handle or candidate_handle == handle:
                continue
            total_edges += 1
            bucket = aggregated.setdefault(
                candidate_handle,
                {
                    "handle": candidate_handle,
                    "name": str(candidate.get("name") or candidate_handle),
                    "description": str(candidate.get("description") or ""),
                    "verified": bool(candidate.get("verified")),
                    "verified_type": str(candidate.get("verified_type") or ""),
                    "public_metrics": dict(candidate.get("public_metrics") or {}),
                    "seed_support": set(),
                    "seed_kinds": set(),
                    "source_refs": set(),
                },
            )
            bucket["description"] = str(candidate.get("description") or bucket.get("description") or "")
            bucket["public_metrics"] = dict(candidate.get("public_metrics") or bucket.get("public_metrics") or {})
            bucket["verified"] = bool(candidate.get("verified") or bucket.get("verified"))
            bucket["verified_type"] = str(candidate.get("verified_type") or bucket.get("verified_type") or "")
            bucket["seed_support"].add(handle)
            bucket["seed_kinds"].add(seed_kind)
            bucket["source_refs"].add(f"following:{handle}")

    selected: list[dict[str, Any]] = []
    active_count = 0
    watchlist_count = 0
    for handle, row in aggregated.items():
        ai_profile = _is_ai_profile(row)
        row["ai_profile"] = ai_profile
        score = _score_graph_candidate(row)
        public_metrics = row.get("public_metrics") or {}
        followers = int(public_metrics.get("followers_count") or 0)
        listed = int(public_metrics.get("listed_count") or 0)
        support_count = len(row["seed_support"])
        if not ai_profile:
            continue
        if not _is_core_ai_actor(row):
            continue
        if support_count < min_support_count:
            continue
        strong_single_seed = (
            support_count == 1
            and score >= (min_watchlist_score + 0.4)
            and (
                row.get("verified")
                or followers >= max(min_followers * 3, 5000)
                or listed >= max(min_listed * 4, 80)
            )
        )
        if score < min_watchlist_score and followers < min_followers and listed < min_listed and not strong_single_seed:
            continue
        if (
            support_count == 1
            and not strong_single_seed
            and followers < max(min_followers, 2500)
            and listed < max(min_listed, 40)
            and not row.get("verified")
        ):
            continue
        active = (
            support_count >= min_active_support_count
            and score >= min_active_score
        ) or (
            support_count >= 1
            and row.get("verified")
            and listed >= max(min_listed * 2, 60)
            and followers >= max(min_followers, 4000)
            and score >= min_watchlist_score
        )
        tier = 2 if active else 1
        if active:
            active_count += 1
        else:
            watchlist_count += 1
        selected.append(
            {
                "handle": handle,
                "name": str(row.get("name") or handle),
                "organization": "",
                "kind": _classify_graph_candidate(row),
                "tier": tier,
                "active": active,
                "source_refs": sorted(row["source_refs"]),
                "graph_score": score,
                "graph_support": support_count,
                "graph_seed_kinds": sorted(row["seed_kinds"]),
            }
        )

    selected.sort(
        key=lambda candidate: (
            bool(candidate.get("active")),
            float(candidate.get("graph_score", 0.0)),
            int(candidate.get("graph_support", 0)),
            candidate.get("handle", ""),
        ),
        reverse=True,
    )
    stats = {
        "enabled": True,
        "seed_handles": seed_handles,
        "resolved_seeds": resolved_seed_count,
        "following_edges": total_edges,
        "raw_candidates": len(aggregated),
        "selected_candidates": len(selected),
        "active_candidates": active_count,
        "watchlist_candidates": watchlist_count,
    }
    return selected, stats, errors


def build_x_authority_registry(seed_path: str | Path | None = None) -> dict[str, Any]:
    seed_payload = _load_seed_payload(seed_path)
    account_map = _seed_account_map(seed_payload)
    external_sources = {
        entry.get("id"): entry.get("url")
        for entry in seed_payload.get("external_seed_sources", [])
        if entry.get("id") and entry.get("url")
    }

    follow_handles: set[str] = set()
    paperpulse_handles: set[str] = set()
    errors: list[dict[str, str]] = []

    try:
        follow_handles = _fetch_follow_the_ai_leaders_handles(
            external_sources.get("follow_the_ai_leaders", FOLLOW_THE_AI_LEADERS_URL)
        )
    except Exception as ex:
        errors.append({"source": "follow_the_ai_leaders", "error": str(ex)})
    try:
        paperpulse_handles = _fetch_paperpulse_author_handles(
            external_sources.get("paperpulse_authors", PAPERPULSE_AUTHORS_URL)
        )
    except Exception as ex:
        errors.append({"source": "paperpulse_authors", "error": str(ex)})

    _merge_seed_handles(account_map, follow_handles, source_ref="follow_the_ai_leaders", activate=False)
    _merge_seed_handles(account_map, paperpulse_handles, source_ref="paperpulse_authors", activate=True)

    overlap = follow_handles & paperpulse_handles
    for handle in overlap:
        record = account_map.get(handle)
        if record is None:
            continue
        record["active"] = True
        record["tier"] = max(int(record.get("tier") or 1), 2)

    graph_accounts, graph_stats, graph_errors = _expand_following_graph(seed_payload, account_map)
    errors.extend(graph_errors)
    for row in graph_accounts:
        handle = _normalize_handle(str(row.get("handle", "")))
        if not handle:
            continue
        existing = account_map.get(handle)
        if existing is None:
            account_map[handle] = {
                "handle": handle,
                "name": str(row.get("name") or handle),
                "organization": str(row.get("organization") or ""),
                "kind": str(row.get("kind") or "researcher"),
                "tier": int(row.get("tier") or 1),
                "active": bool(row.get("active", False)),
                "source_refs": sorted(set(row.get("source_refs") or [])),
                "graph_score": float(row.get("graph_score") or 0.0),
                "graph_support": int(row.get("graph_support") or 0),
                "graph_seed_kinds": sorted(set(row.get("graph_seed_kinds") or [])),
            }
            continue
        existing["tier"] = max(int(existing.get("tier") or 1), int(row.get("tier") or 1))
        existing["active"] = bool(existing.get("active", False) or row.get("active", False))
        if existing.get("kind") != "official" and row.get("kind"):
            existing["kind"] = str(row["kind"])
        existing["source_refs"] = sorted(set(existing.get("source_refs") or []) | set(row.get("source_refs") or []))
        existing["graph_score"] = max(float(existing.get("graph_score") or 0.0), float(row.get("graph_score") or 0.0))
        existing["graph_support"] = max(int(existing.get("graph_support") or 0), int(row.get("graph_support") or 0))
        existing["graph_seed_kinds"] = sorted(set(existing.get("graph_seed_kinds") or []) | set(row.get("graph_seed_kinds") or []))

    accounts = sorted(
        account_map.values(),
        key=lambda row: (
            {"official": 0, "company": 1, "researcher": 2}.get(str(row.get("kind")), 3),
            -int(row.get("tier") or 0),
            not bool(row.get("active", False)),
            row.get("handle", ""),
        ),
    )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "seed_path": str(Path(seed_path) if seed_path else DEFAULT_SEED_PATH),
        "seed_sources": {
            "follow_the_ai_leaders": len(follow_handles),
            "paperpulse_authors": len(paperpulse_handles),
            "overlap": len(overlap),
        },
        "graph_expansion": graph_stats,
        "errors": errors,
        "accounts": accounts,
    }


def refresh_x_authority_registry(
    *,
    seed_path: str | Path | None = None,
    snapshot_path: str | Path | None = None,
    max_age_hours: int = 24,
    force: bool = False,
) -> dict[str, Any]:
    resolved_snapshot = Path(snapshot_path) if snapshot_path else DEFAULT_SNAPSHOT_PATH
    existing_payload: dict[str, Any] | None = None
    if resolved_snapshot.exists():
        try:
            existing_payload = json.loads(resolved_snapshot.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_payload = None
    if not force and _is_registry_fresh(resolved_snapshot, max_age_hours):
        return existing_payload or json.loads(resolved_snapshot.read_text(encoding="utf-8"))

    payload = build_x_authority_registry(seed_path)
    if existing_payload and _registry_payload_for_compare(existing_payload) == _registry_payload_for_compare(payload):
        return existing_payload
    resolved_snapshot.parent.mkdir(parents=True, exist_ok=True)
    resolved_snapshot.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def load_x_authority_registry(
    *,
    seed_path: str | Path | None = None,
    snapshot_path: str | Path | None = None,
    max_age_hours: int = 24,
) -> dict[str, Any]:
    resolved_snapshot = Path(snapshot_path) if snapshot_path else DEFAULT_SNAPSHOT_PATH
    if resolved_snapshot.exists():
        try:
            return json.loads(resolved_snapshot.read_text(encoding="utf-8"))
        except Exception:
            pass
    return refresh_x_authority_registry(
        seed_path=seed_path,
        snapshot_path=resolved_snapshot,
        max_age_hours=max_age_hours,
        force=False,
    )


def authority_lookup(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        _normalize_handle(str(row.get("handle", ""))): row
        for row in registry.get("accounts", [])
        if _normalize_handle(str(row.get("handle", "")))
    }


def find_authority_record(handle: str | None, registry: dict[str, Any]) -> dict[str, Any] | None:
    normalized = _normalize_handle(handle)
    if not normalized:
        return None
    return authority_lookup(registry).get(normalized)


def iter_active_authority_accounts(
    registry: dict[str, Any],
    *,
    kinds: set[str] | None = None,
    min_tier: int = 1,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in registry.get("accounts", []):
        if not row.get("active", False):
            continue
        if int(row.get("tier") or 0) < min_tier:
            continue
        if kinds and str(row.get("kind")) not in kinds:
            continue
        rows.append(row)
    return rows
