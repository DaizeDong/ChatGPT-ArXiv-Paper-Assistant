from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arxiv_assistant.utils.hotspot_schema import HotspotItem
from arxiv_assistant.utils.hotspot_web_data import write_hotspot_web_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild hotspot web_data payloads from saved normalized items and reports.")
    parser.add_argument("--output-root", default="out", help="Root output directory that contains hot/ and web_data/.")
    return parser.parse_args()


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_raw_items(path: Path) -> list[HotspotItem]:
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of hotspot items in {path}")
    return [HotspotItem(**row) for row in payload]


def rebuild_hotspot_web_data(output_root: str | Path) -> list[str]:
    output_root = Path(output_root)
    hot_root = output_root / "hot"
    reports_root = hot_root / "reports"
    normalized_root = hot_root / "normalized"
    web_root = output_root / "web_data" / "hot"

    if not reports_root.exists():
        return []

    shutil.rmtree(web_root, ignore_errors=True)

    rebuilt_dates: list[str] = []
    for report_path in sorted(reports_root.glob("*.json")):
        report = _load_json(report_path)
        if not isinstance(report, dict):
            raise ValueError(f"Expected report object in {report_path}")
        date = str(report.get("date") or report_path.stem).strip()
        if not date:
            continue
        normalized_path = normalized_root / f"{date}.json"
        if not normalized_path.exists():
            raise FileNotFoundError(f"Missing normalized hotspot items for {date}: {normalized_path}")
        raw_items = _load_raw_items(normalized_path)
        write_hotspot_web_data(output_root, report, raw_items)
        rebuilt_dates.append(date)

    return rebuilt_dates


def main() -> None:
    args = parse_args()
    rebuilt_dates = rebuild_hotspot_web_data(args.output_root)
    print(f"Rebuilt hotspot web_data for {len(rebuilt_dates)} day(s).")


if __name__ == "__main__":
    main()
