"""The one place configuration is read.

There were three loaders before this, and each was wrong in its own way:
environment.py read a RELATIVE path, so it silently returned an empty config
whenever the process was not started from the repository root;
generate_monthly_summaries.py passed no encoding; hotspot_config.py was the only
correct one. Three loaders also meant a config file could be split only by
finding and fixing all three, which is how a reader ends up quietly falling back
to defaults for a section it never saw.

Configuration is split by SUBSYSTEM: the paper pipeline in config.ini, the
hotspot feed in hotspot.ini. Half of the single file was one subsystem's
settings, which is what made the paper pipeline's own settings hard to find.
Both files are merged into one ConfigParser, so nothing downstream changes.

A missing hotspot.ini is an ERROR, not a default. Its sections carry cutoffs and
source switches, and configparser answers a missing section by falling back to
whatever the caller passed -- so losing the file would not break anything
loudly, it would just quietly change what the feed collects.
"""

from __future__ import annotations

import configparser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"

#: Read in order; later files win on a key collision.
CONFIG_FILES = ("config.ini", "hotspot.ini")


def config_paths(config_dir: Path | None = None) -> list[Path]:
    base = config_dir or CONFIG_DIR
    return [base / name for name in CONFIG_FILES]


def load_repo_config(config_path: Path | None = None) -> configparser.ConfigParser:
    """Load the repository configuration.

    ``config_path`` names the primary file; its siblings are read alongside it,
    which keeps the old single-path call shape working.
    """
    primary = Path(config_path) if config_path else (CONFIG_DIR / CONFIG_FILES[0])
    config = configparser.ConfigParser()
    read = config.read(config_paths(primary.parent), encoding="utf-8")
    if not read:
        raise FileNotFoundError(
            "no configuration found in %s; expected %s"
            % (primary.parent, ", ".join(CONFIG_FILES))
        )
    missing = [p.name for p in config_paths(primary.parent) if str(p) not in read]
    if missing:
        raise FileNotFoundError(
            "configuration is incomplete: %s missing from %s. A missing file is "
            "not an empty one -- the sections it carries would silently fall "
            "back to caller defaults." % (", ".join(missing), primary.parent)
        )
    return config
