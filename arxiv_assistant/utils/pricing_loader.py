import copy
import json
import pprint
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.request import urlopen

from arxiv_assistant.utils.pricing import MODEL_PRICING as STATIC_MODEL_PRICING

PRICING_SOURCE_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
PRICING_COMMIT_URL = "https://api.github.com/repos/BerriAI/litellm/commits/main"
DEFAULT_CACHE_PATH = Path(__file__).resolve().parents[2] / ".cache" / "model_pricing_cache.json"
DEFAULT_FALLBACK_MODULE_PATH = Path(__file__).resolve().parent / "pricing.py"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_AGE_HOURS = 24

_MODEL_PRICING_CACHE: Optional[Dict[str, Dict[str, float]]] = None


def clear_cached_model_pricing():
    global _MODEL_PRICING_CACHE
    _MODEL_PRICING_CACHE = None


def _fetch_json(url: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> Dict[str, Any]:
    with urlopen(url, timeout=timeout) as response:
        return json.load(response)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _to_per_million(value: Any) -> Optional[float]:
    if not _is_number(value):
        return None
    return round(float(value) * 1_000_000, 12)


def _merge_with_static(pricing_table: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    merged = copy.deepcopy(STATIC_MODEL_PRICING)
    merged.update(pricing_table)
    return merged


def normalize_model_pricing(raw_table: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    normalized = {}

    for model_name, entry in raw_table.items():
        if model_name == "sample_spec":
            continue

        if not isinstance(entry, dict):
            continue

        prompt_cost = _to_per_million(entry.get("input_cost_per_token"))
        completion_cost = _to_per_million(entry.get("output_cost_per_token"))

        if prompt_cost is None or completion_cost is None:
            continue

        normalized_entry = {
            "prompt": prompt_cost,
            "completion": completion_cost,
        }

        cache_cost = _to_per_million(entry.get("cache_read_input_token_cost"))
        if cache_cost is not None:
            normalized_entry["cache"] = cache_cost

        normalized[model_name] = normalized_entry

    return normalized


def fetch_latest_pricing_commit_sha(timeout: int = DEFAULT_TIMEOUT_SECONDS) -> Optional[str]:
    try:
        response = _fetch_json(PRICING_COMMIT_URL, timeout=timeout)
    except Exception:
        return None
    return response.get("sha")


def fetch_remote_model_pricing(timeout: int = DEFAULT_TIMEOUT_SECONDS) -> Dict[str, Dict[str, float]]:
    raw_table = _fetch_json(PRICING_SOURCE_URL, timeout=timeout)
    return normalize_model_pricing(raw_table)


def load_pricing_cache(cache_path: Path = DEFAULT_CACHE_PATH) -> Optional[Dict[str, Any]]:
    if not cache_path.exists():
        return None

    with cache_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        return None

    pricing_table = payload.get("pricing_table")
    if not isinstance(pricing_table, dict):
        return None

    return payload


def save_pricing_cache(payload: Dict[str, Any], cache_path: Path = DEFAULT_CACHE_PATH):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _cache_is_fresh(payload: Dict[str, Any], max_age_hours: int) -> bool:
    fetched_at = payload.get("fetched_at")
    if not isinstance(fetched_at, str):
        return False

    try:
        fetched_time = datetime.fromisoformat(fetched_at)
    except ValueError:
        return False

    now = datetime.now(UTC)
    return now - fetched_time <= timedelta(hours=max_age_hours)


def refresh_model_pricing(
    cache_path: Path = DEFAULT_CACHE_PATH,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    force_download: bool = False,
) -> Dict[str, Any]:
    commit_sha = fetch_latest_pricing_commit_sha(timeout=timeout)
    cached_payload = load_pricing_cache(cache_path)

    if (
        not force_download and
        cached_payload is not None and
        commit_sha and
        cached_payload.get("commit_sha") == commit_sha
    ):
        return cached_payload

    remote_pricing = fetch_remote_model_pricing(timeout=timeout)
    merged_pricing = _merge_with_static(remote_pricing)

    payload = {
        "fetched_at": datetime.now(UTC).isoformat(),
        "source_url": PRICING_SOURCE_URL,
        "commit_sha": commit_sha,
        "remote_model_count": len(remote_pricing),
        "pricing_table": merged_pricing,
    }
    save_pricing_cache(payload, cache_path)
    return payload


def get_model_pricing(
    force_refresh: bool = False,
    max_age_hours: int = DEFAULT_MAX_AGE_HOURS,
    cache_path: Path = DEFAULT_CACHE_PATH,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Dict[str, float]]:
    global _MODEL_PRICING_CACHE

    if _MODEL_PRICING_CACHE is not None and not force_refresh:
        return _MODEL_PRICING_CACHE

    cached_payload = load_pricing_cache(cache_path)
    if not force_refresh and cached_payload is not None and _cache_is_fresh(cached_payload, max_age_hours):
        _MODEL_PRICING_CACHE = cached_payload["pricing_table"]
        return _MODEL_PRICING_CACHE

    try:
        refreshed_payload = refresh_model_pricing(
            cache_path=cache_path,
            timeout=timeout,
            force_download=force_refresh,
        )
        _MODEL_PRICING_CACHE = refreshed_payload["pricing_table"]
        return _MODEL_PRICING_CACHE
    except Exception:
        if cached_payload is not None:
            _MODEL_PRICING_CACHE = cached_payload["pricing_table"]
            return _MODEL_PRICING_CACHE

    _MODEL_PRICING_CACHE = copy.deepcopy(STATIC_MODEL_PRICING)
    return _MODEL_PRICING_CACHE


def write_pricing_fallback_module(
    pricing_table: Dict[str, Dict[str, float]],
    output_path: Path = DEFAULT_FALLBACK_MODULE_PATH,
    fetched_at: Optional[str] = None,
    commit_sha: Optional[str] = None,
):
    header_lines = [
        "# Auto-generated fallback snapshot for model pricing.",
        f"# Source: {PRICING_SOURCE_URL}",
    ]
    if fetched_at:
        header_lines.append(f"# Fetched at: {fetched_at}")
    if commit_sha:
        header_lines.append(f"# LiteLLM main commit: {commit_sha}")
    header_lines.append("")

    rendered_table = pprint.pformat(pricing_table, sort_dicts=True, width=120)
    output = "\n".join(header_lines) + "\nMODEL_PRICING = " + rendered_table + "\n"
    output_path.write_text(output, encoding="utf-8")
