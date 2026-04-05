from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arxiv_assistant.utils.hotspot.hotspot_dates import is_supported_hotspot_date
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem
from arxiv_assistant.utils.hotspot.hotspot_web_data import write_hotspot_web_data

DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}\.json$")
_ZH_MATCH_KEYS = ("topic_id", "id", "slug", "kind", "category")


def _match_key(item: dict) -> str | None:
    """Return a stable identifier for matching items across rebuilds."""
    for k in _ZH_MATCH_KEYS:
        v = item.get(k)
        if v:
            return f"{k}\x00{v}"
    return None


def _merge_zh(target: Any, source: Any) -> None:
    """Recursively copy *_zh fields from *source* into *target* in-place."""
    if isinstance(target, dict) and isinstance(source, dict):
        for k, v in source.items():
            if k.endswith("_zh") and k not in target:
                target[k] = v
        for k in target:
            if k in source and isinstance(target[k], (dict, list)):
                _merge_zh(target[k], source[k])
    elif isinstance(target, list) and isinstance(source, list):
        if not target or not isinstance(target[0], dict):
            return
        # Try key-based matching first
        src_map: dict[str, dict] = {}
        has_keys = False
        for item in source:
            if isinstance(item, dict):
                mk = _match_key(item)
                if mk:
                    src_map[mk] = item
                    has_keys = True
        if has_keys:
            for item in target:
                if isinstance(item, dict):
                    mk = _match_key(item)
                    if mk and mk in src_map:
                        _merge_zh(item, src_map[mk])
        else:
            # Fallback: index-based matching for keyless lists (e.g. evidence)
            for i in range(min(len(target), len(source))):
                if isinstance(target[i], dict) and isinstance(source[i], dict):
                    _merge_zh(target[i], source[i])


def _cache_zh_payloads(hot_dir: Path) -> dict[str, dict]:
    """Load existing daily payloads before rebuild so _zh fields can be restored."""
    cache: dict[str, dict] = {}
    if not hot_dir.exists():
        return cache
    for fp in sorted(hot_dir.glob("*.json")):
        if fp.name == "index.json" or not DATE_PATTERN.fullmatch(fp.name):
            continue
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            date = str(data.get("meta", {}).get("date", "")).strip() or fp.stem
            cache[date] = data
        except Exception:
            continue
    return cache


def _restore_zh_to_rebuilt(hot_dir: Path, zh_cache: dict[str, dict]) -> int:
    """Merge cached _zh fields back into rebuilt daily payloads. Returns count of files updated."""
    updated = 0
    for fp in sorted(hot_dir.glob("*.json")):
        if fp.name == "index.json" or not DATE_PATTERN.fullmatch(fp.name):
            continue
        date = fp.stem
        old = zh_cache.get(date)
        if not old:
            continue
        new_data = json.loads(fp.read_text(encoding="utf-8"))
        _merge_zh(new_data, old)
        fp.write_text(json.dumps(new_data, indent=2, ensure_ascii=False), encoding="utf-8")
        updated += 1
    return updated


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

    # Cache existing _zh translations before wiping
    zh_cache = _cache_zh_payloads(web_root)

    shutil.rmtree(web_root, ignore_errors=True)
    web_root.mkdir(parents=True, exist_ok=True)

    rebuilt_dates: list[str] = []
    for report_path in sorted(reports_root.glob("*.json")):
        report = _load_json(report_path)
        if not isinstance(report, dict):
            raise ValueError(f"Expected report object in {report_path}")
        date = str(report.get("date") or report_path.stem).strip()
        if not date or not is_supported_hotspot_date(date):
            continue
        normalized_path = normalized_root / f"{date}.json"
        if not normalized_path.exists():
            raise FileNotFoundError(f"Missing normalized hotspot items for {date}: {normalized_path}")
        raw_items = _load_raw_items(normalized_path)
        write_hotspot_web_data(output_root, report, raw_items)
        rebuilt_dates.append(date)

    if not rebuilt_dates:
        (web_root / "index.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "latest_date": None,
                    "dates": [],
                    "months": [],
                    "years": [],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # Restore _zh translations into rebuilt payloads
    if zh_cache:
        restored = _restore_zh_to_rebuilt(web_root, zh_cache)
        if restored:
            print(f"Restored _zh translations for {restored} day(s).")

    return rebuilt_dates


def main() -> None:
    args = parse_args()
    rebuilt_dates = rebuild_hotspot_web_data(args.output_root)
    print(f"Rebuilt hotspot web_data for {len(rebuilt_dates)} day(s).")


if __name__ == "__main__":
    main()
