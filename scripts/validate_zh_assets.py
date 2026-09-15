"""Validate _zh translation assets in the hotspot web_data directory.

Fails with exit code 1 if any daily payload that contains featured_topics or
resurgence sections is missing its _zh companion, or if the _zh file is
byte-identical to the source (silent _merge_zh failure).

Usage:
    python scripts/validate_zh_assets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    web_root = REPO_ROOT / "out" / "hot" / "web_data"
    missing: list[str] = []
    empty: list[str] = []

    for daily in sorted(web_root.rglob("*.json")):
        if daily.name.endswith("_zh.json"):
            continue
        zh = daily.with_name(daily.stem + "_zh.json")
        try:
            data = json.loads(daily.read_text(encoding="utf-8"))
        except Exception:
            continue
        # Only daily payloads carry these sections; skip index/aggregate files.
        if not any(k in data for k in ("featured_topics", "resurgence")):
            continue
        if not zh.exists():
            missing.append(str(daily))
            continue
        zh_data = json.loads(zh.read_text(encoding="utf-8"))
        # Silent _merge_zh failure => _zh file exists but is byte-identical (no translation).
        if zh_data == data:
            empty.append(str(zh))

    if missing or empty:
        print("Missing _zh companions:", *missing, sep="\n  ")
        print("Untranslated (identical) _zh files:", *empty, sep="\n  ")
        sys.exit(1)
    print("All _zh translation assets present and non-trivial.")


if __name__ == "__main__":
    main()
