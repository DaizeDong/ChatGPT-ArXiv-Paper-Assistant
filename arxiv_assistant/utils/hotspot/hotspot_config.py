import configparser
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass
class HotspotPaths:
    raw_root: Path
    normalized_path: Path
    clusters_path: Path
    report_path: Path
    markdown_path: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_repo_config(config_path: Path | None = None) -> configparser.ConfigParser:
    path = config_path or (repo_root() / "configs" / "config.ini")
    config = configparser.ConfigParser()
    config.read(path, encoding="utf-8")
    return config

def build_hotspot_paths(output_root: Path, target_date: date) -> HotspotPaths:
    date_string = target_date.isoformat()
    month_string = target_date.strftime("%Y-%m")
    return HotspotPaths(
        raw_root=output_root / "hot" / "raw" / date_string,
        normalized_path=output_root / "hot" / "normalized" / f"{date_string}.json",
        clusters_path=output_root / "hot" / "clusters" / f"{date_string}.json",
        report_path=output_root / "hot" / "reports" / f"{date_string}.json",
        markdown_path=output_root / "hot" / "md" / month_string / f"{date_string}-hotspots.md",
    )


def ensure_parent_dirs(paths: HotspotPaths):
    paths.raw_root.mkdir(parents=True, exist_ok=True)
    paths.normalized_path.parent.mkdir(parents=True, exist_ok=True)
    paths.clusters_path.parent.mkdir(parents=True, exist_ok=True)
    paths.report_path.parent.mkdir(parents=True, exist_ok=True)
    paths.markdown_path.parent.mkdir(parents=True, exist_ok=True)
