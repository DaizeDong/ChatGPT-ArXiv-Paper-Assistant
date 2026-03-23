from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arxiv_assistant.hotspots.pipeline import (
    generate_daily_hotspot_report,
    parse_target_datetime,
)
from arxiv_assistant.utils.hotspot.hotspot_config import load_repo_config, repo_root
from arxiv_assistant.utils.local_env import load_local_env

load_local_env()

REPO_ROOT = repo_root()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate daily AI hotspots from multiple signals.")
    parser.add_argument("--output-root", default="out", help="Output root directory.")
    parser.add_argument("--target-date", "--date", dest="target_date", default=None, help="Target date in YYYY-MM-DD format.")
    parser.add_argument("--mode", choices=["auto", "openai", "heuristic"], default="auto", help="Override hotspot mode.")
    parser.add_argument("--force", action="store_true", help="Regenerate even when cached raw items exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_repo_config(REPO_ROOT / "configs" / "config.ini")
    report = generate_daily_hotspot_report(
        output_root=args.output_root,
        target_date=parse_target_datetime(args.target_date, Path(args.output_root)),
        config=config,
        mode_override=args.mode,
        force=args.force,
    )
    if report is None:
        print("Hotspots disabled.")
        return
    print(Path(args.output_root) / "hot" / "reports" / f"{report['date']}.json")


if __name__ == "__main__":
    main()
