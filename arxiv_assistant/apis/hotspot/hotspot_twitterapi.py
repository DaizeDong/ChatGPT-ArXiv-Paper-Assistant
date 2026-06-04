from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from arxiv_assistant.apis.hotspot.hotspot_x_common import (
    get_authority_record,
    is_authoritative_x_identity,
    is_newsworthy_x_text,
)
from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem, clean_text
from arxiv_assistant.utils.hotspot.hotspot_sources import clip_text, is_fresh, record_api_usage
from arxiv_assistant.utils.hotspot.x_authority_registry import (
    iter_active_authority_accounts,
    load_x_authority_registry,
)

SOURCE_ID = "x_twitterapi"
PROVENANCE = "native:x_twitterapi"
TWITTERAPI_IO_URL = "https://api.twitterapi.io/twitter/user/last_tweets"
TWITTERAPI_IO_ENV_KEYS = ("TWITTERAPI_IO_KEY", "TWITTERAPI_KEY")
X_HOSTS = {"x.com", "twitter.com", "mobile.twitter.com", "www.x.com", "www.twitter.com", "pic.x.com"}
_RATE_LIMIT_MAX_RETRIES = 2
_RATE_LIMIT_WAIT_SECONDS = 16


def _get_twitterapi_key() -> str | None:
    for key in TWITTERAPI_IO_ENV_KEYS:
        value = clean_text(os.getenv(key))
        if value:
            return value
    return None


def fetch_hotspot_items(
    target_date: datetime,
    freshness_hours: int,
    seed_path: str | Path,
    *,
    result_limit: int = 80,
    snapshot_path: str | Path | None = None,
    max_age_hours: int = 24,
    official_account_limit: int = 24,
    researcher_account_limit: int = 18,
    tweets_per_user: int = 10,
) -> list[HotspotItem]:
    api_key = _get_twitterapi_key()
    if not api_key:
        print(
            "Warning: no twitterapi.io key configured. Set one of "
            f"{TWITTERAPI_IO_ENV_KEYS} to enable the x_twitterapi source. Skipping."
        )
        return []
    return []
