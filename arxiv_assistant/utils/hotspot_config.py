import configparser
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List


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


def load_hotspot_registry(config: configparser.ConfigParser, root: Path | None = None) -> List[Dict]:
    repo = root or repo_root()
    registry_path = repo / config["HOTSPOTS"]["source_registry_path"]
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        sources = payload
    elif isinstance(payload, dict):
        sources = payload.get("sources") or payload.get("sites") or []
    else:
        raise ValueError(f"Unsupported hotspot registry format: {registry_path}")

    enabled_map = {
        key: config["HOTSPOT_SOURCES"].getboolean(key, fallback=True)
        for key in config["HOTSPOT_SOURCES"]
    }
    filtered = []
    for source in sources:
        enabled = bool(source.get("enabled", True))
        source_key = source.get("id") or source.get("site_id")
        if source_key in enabled_map:
            enabled = enabled and enabled_map[source_key]
        if source.get("adapter") == "official_blogs" or source_key == "official_blogs":
            enabled = enabled and config["HOTSPOT_SOURCES"].getboolean("official_blogs", fallback=True)
        if enabled:
            filtered.append(source)
    return filtered


def read_prompt(prompt_name: str, root: Path | None = None) -> str:
    repo = root or repo_root()
    return (repo / "prompts" / prompt_name).read_text(encoding="utf-8")


def parse_target_date(raw_date: str | None) -> date:
    if not raw_date:
        return datetime.now().date()
    return datetime.strptime(raw_date, "%Y-%m-%d").date()


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
