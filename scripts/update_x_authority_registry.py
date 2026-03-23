from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arxiv_assistant.utils.local_env import load_local_env
from arxiv_assistant.utils.hotspot.x_authority_registry import refresh_x_authority_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh the local X authority registry from curated seed sources.")
    parser.add_argument(
        "--seed-path",
        default="configs/hotspot/x_authority_seeds.json",
        help="Path to the curated X authority seed file.",
    )
    parser.add_argument(
        "--snapshot-path",
        default="configs/hotspot/x_authority_inventory.json",
        help="Path to the tracked X authority inventory snapshot.",
    )
    parser.add_argument(
        "--max-age-hours",
        type=int,
        default=24,
        help="Maximum tracked inventory age before rebuilding.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refresh even if the tracked inventory is still fresh.",
    )
    return parser.parse_args()


def main() -> None:
    load_local_env()
    args = parse_args()
    payload = refresh_x_authority_registry(
        seed_path=Path(args.seed_path),
        snapshot_path=Path(args.snapshot_path),
        max_age_hours=args.max_age_hours,
        force=args.force,
    )
    print(Path(args.snapshot_path))
    print(f"accounts={len(payload.get('accounts', []))}")
    print(f"seed_sources={payload.get('seed_sources', {})}")
    if payload.get("graph_expansion"):
        print(f"graph_expansion={payload['graph_expansion']}")
    if payload.get("errors"):
        print(f"errors={payload['errors']}")


if __name__ == "__main__":
    main()
