from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge the hotspot web app build into the published dist tree.")
    parser.add_argument("--web-dist", default="web/dist", help="Built hotspot web application directory.")
    parser.add_argument("--site-dist", default="dist", help="Final dist directory that also contains Personalized Daily Arxiv Paper pages.")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_tree(source: Path, target: Path) -> None:
    if not source.exists():
        return
    shutil.copytree(source, target, dirs_exist_ok=True)


def _route_targets(web_dist: Path) -> list[str]:
    web_data_root = web_dist / "web_data" / "hot"
    root_index = _load_json(web_data_root / "index.json")
    routes = {"hot"}
    for month_entry in root_index.get("months", []):
        month = str(month_entry.get("month", "")).strip()
        if month:
            routes.add(f"hot/{month}")
    for year_entry in root_index.get("years", []):
        year = str(year_entry.get("year", "")).strip()
        if year:
            routes.add(f"hot/{year}")
    for day_entry in root_index.get("dates", []):
        date = str(day_entry.get("date", "")).strip()
        if not date:
            continue
        routes.add(f"hot/{date}")
    return sorted(routes)


def merge_hotspot_web_dist(web_dist: Path, site_dist: Path) -> None:
    web_dist = web_dist.resolve()
    site_dist = site_dist.resolve()
    site_dist.mkdir(parents=True, exist_ok=True)

    # Remove markdown-rendered hotspot pages before overlaying the React app routes.
    shutil.rmtree(site_dist / "hot", ignore_errors=True)

    _copy_tree(web_dist / "assets", site_dist / "assets")
    _copy_tree(web_dist / "web_data", site_dist / "web_data")

    app_index = (web_dist / "index.html").read_text(encoding="utf-8")
    for route in _route_targets(web_dist):
        target_dir = site_dist / route
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "index.html").write_text(app_index, encoding="utf-8")

    not_found_path = web_dist / "404.html"
    if not_found_path.exists():
        (site_dist / "hot" / "404.html").write_text(not_found_path.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    args = parse_args()
    merge_hotspot_web_dist(Path(args.web_dist), Path(args.site_dist))


if __name__ == "__main__":
    main()
