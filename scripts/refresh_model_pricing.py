import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arxiv_assistant.utils.pricing_loader import (
    DEFAULT_CACHE_PATH,
    DEFAULT_FALLBACK_MODULE_PATH,
    get_model_pricing,
    load_pricing_cache,
    refresh_model_pricing,
    write_pricing_fallback_module,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Refresh LiteLLM-backed model pricing cache.")
    parser.add_argument("--force", action="store_true", help="Force refresh even if the cache is still fresh.")
    parser.add_argument(
        "--max-age-hours",
        type=int,
        default=24,
        help="Maximum cache age before a runtime refresh is attempted.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="Path to the pricing cache JSON file.",
    )
    parser.add_argument(
        "--write-fallback-py",
        action="store_true",
        help="Write the merged pricing table to the local fallback pricing.py module.",
    )
    parser.add_argument(
        "--fallback-path",
        type=Path,
        default=DEFAULT_FALLBACK_MODULE_PATH,
        help="Path to the fallback pricing.py module.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.force:
        payload = refresh_model_pricing(cache_path=args.cache_path, force_download=True)
    else:
        pricing_table = get_model_pricing(max_age_hours=args.max_age_hours, cache_path=args.cache_path)
        payload = load_pricing_cache(args.cache_path)
        if payload is None:
            raise RuntimeError("Expected pricing cache to exist after loading pricing table")
        print(f"Loaded pricing table with {len(pricing_table)} models")

    print(f"Pricing cache path: {args.cache_path}")
    print(f"Fetched at: {payload.get('fetched_at')}")
    print(f"Commit SHA: {payload.get('commit_sha')}")
    print(f"Remote models normalized: {payload.get('remote_model_count')}")
    print(f"Models available to the app: {len(payload.get('pricing_table', {}))}")

    if args.write_fallback_py:
        write_pricing_fallback_module(
            payload["pricing_table"],
            output_path=args.fallback_path,
            fetched_at=payload.get("fetched_at"),
            commit_sha=payload.get("commit_sha"),
        )
        print(f"Wrote fallback pricing module to: {args.fallback_path}")


if __name__ == "__main__":
    main()
