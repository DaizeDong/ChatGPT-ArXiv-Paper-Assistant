# Stage 5 — X Coverage via twitterapi.io Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a dedicated `hotspot_twitterapi.py` harvest adapter that pulls live tweets from the 310 active accounts in `x_authority_inventory.json` through twitterapi.io (REST or the `twitterapi-mcp` tools), normalizes them to `HotspotItem(source_id="x_twitterapi", provenance="native:x_twitterapi")`, and wires it into the pipeline fan-out — fixing the X≈0 coverage gap whose root cause is the data channel (official X Recent Search needs Pro $5,000/mo; PaperPulse is dead), not the filters.

**Architecture:** A thin, single-responsibility adapter that (1) loads the existing X authority registry, (2) selects active official + researcher accounts, (3) fetches each account's recent timeline through twitterapi.io's `user/last_tweets` REST endpoint, (4) normalizes provider JSON to the canonical X-API-v2 dict shape, (5) reuses the *already partly-fixed* `is_newsworthy_x_text` filter (official accounts are exempt from `SELF_WORK_PATTERNS`), and (6) emits `HotspotItem`s. The adapter degrades cleanly (empty list, never raises) on missing key, empty response, or rate-limit. A new `use_twitterapi` config flag supersedes `use_x_official`/`use_x_paperpulse`.

**Tech Stack:** Python 3, `requests` (REST to `https://api.twitterapi.io/twitter/user/last_tweets`, header `X-API-Key`), `pytest` (`unittest.TestCase` style, `@patch` for network), existing `configparser` config, existing `x_authority_registry` + `hotspot_x_common` infrastructure.

---

## Context the implementer must know

**Why this stage exists (from `docs/investigation_x_coverage.md`):** Across 14 tracked days, `x_official` returned 0 items every day (its Recent Search endpoint needs X API Pro at $5,000/mo) and `x_paperpulse` returned 0 every day (its upstream feed is frozen/stale). Only `x_ainews_twitter` produced anything, sporadically (~2.6 items/day, weekdays only). **The bottleneck is the data channel, not the newsworthiness filter** — `is_newsworthy_x_text` already exempts official/company accounts from `SELF_WORK_PATTERNS`, so an official tweet like "We released GPT-5 today" already passes. twitterapi.io is the way out: ~$0.15/1k tweets, no X developer account, usable by new accounts. The registry inventory (`configs/hotspot/x_authority_inventory.json`, 546 accounts / 310 active) is already built and refreshed monthly.

**Locked contract (from `…-00-overview.md`):**
- §1 module layout: this stage creates `arxiv_assistant/apis/hotspot/hotspot_twitterapi.py`.
- §2.1: `HotspotItem` (in `arxiv_assistant/utils/hotspot/hotspot_schema.py`) gains two optional Stage-0 fields — `verified_first_date: str | None = None` and `provenance: str = ""`. **Stage 5 must not assume Stage 0 has landed**: it sets `provenance` defensively (only if the dataclass field exists) and **always** mirrors `provenance` and `source_id` into `metadata` so downstream code works either way.
- §3: add `[HOTSPOT_SOURCES] use_twitterapi = true`; it supersedes `use_x_official`/`use_x_paperpulse` (those two are retired — see Task 6 migration note). `TWITTERAPI_IO_KEY` env already exists in `.env.example` and `README.md`.
- §4 test conventions: `unittest.TestCase` subclasses, `test_` methods, `tempfile.TemporaryDirectory()` for fs, `@patch` for network. **Never hit the network in tests.**

**Existing real code this stage reuses (do not re-implement):**
- `arxiv_assistant/apis/hotspot/hotspot_x_common.py`
  - `is_newsworthy_x_text(text, *, authority_kind="official", expanded_urls=None, activity=0) -> bool` — the filter. Official/company accounts already pass `SELF_WORK_PATTERNS` content ("we released", "new paper", "arxiv"); only `researcher`-kind accounts are blocked by those patterns.
  - `is_authoritative_x_identity(author_handle, ..., *, registry=None) -> bool`
  - `get_authority_record(author_handle, *, registry=None) -> dict | None`
- `arxiv_assistant/utils/hotspot/x_authority_registry.py`
  - `load_x_authority_registry(*, seed_path=None, snapshot_path=None, max_age_hours=24) -> dict`
  - `iter_active_authority_accounts(registry, *, kinds=None, min_tier=1) -> list[dict]` (each row has `handle`, `name`, `kind`, `tier`, `active`, optionally `x_user_id`, `organization`)
- `arxiv_assistant/utils/hotspot/hotspot_sources.py`
  - `is_fresh(published_at, target_date, freshness_hours) -> bool` (window = `[target-freshness_hours, target+6h]`)
  - `clip_text(text, limit=500) -> str`
  - `record_api_usage(*, requests=1, estimated_cost=0.0, source_id=None, provider=None) -> None`
- `arxiv_assistant/utils/hotspot/hotspot_schema.py`
  - `HotspotItem(source_id, source_name, source_role, source_type, title, summary, url, canonical_url, published_at=None, tags=[], authors=[], metadata={})`, plus `clean_text(value) -> str`.

**twitterapi.io REST contract (confirmed against the existing fallback path in `hotspot_x_official.py`):**
- URL: `GET https://api.twitterapi.io/twitter/user/last_tweets`
- Header: `X-API-Key: <TWITTERAPI_IO_KEY>`
- Params: `userName=<handle>` (or `userId=<id>` if the registry cached one — more stable/faster), `includeReplies=false`.
- Response: `{"tweets": [...]}` OR `{"data": {"tweets": [...]}}` (tolerate both). Each tweet uses camelCase keys: `id`, `text`, `createdAt` (Twitter classic format `"Tue Dec 10 07:00:00 +0000 2024"`), `lang`, `inReplyToUserId`, `retweeted_tweet`, `entities`, `likeCount`, `replyCount`, `retweetCount`, `quoteCount`, `viewCount`, `bookmarkCount`, and nested `author: {id, name, userName, isBlueVerified}`.
- 429 → degrade (return what we have / empty), never raise out of the adapter.

**`twitterapi-mcp` note (spec §A.3):** The `twitterapi-mcp` tools (`get_user_last_tweets`, `search_tweets`, `get_user_info`) are the *same* twitterapi.io backend exposed as MCP tools for the agent-native runtime (Stage 6). Stage 5 implements the deterministic Python REST path (testable, no network in CI). The function is structured so a future MCP-backed fetcher can replace `_fetch_last_tweets_rest` without touching `fetch_hotspot_items` — keep that seam clean.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `arxiv_assistant/apis/hotspot/hotspot_twitterapi.py` | twitterapi.io harvest adapter: registry → REST fetch → normalize → filter → `HotspotItem` | **Create** (~230 lines) |
| `arxiv_assistant/hotspots/pipeline.py` | source fan-out: register `x_twitterapi` spec behind `use_twitterapi` | **Modify** (import + spec block) |
| `configs/config.ini` | enable `use_twitterapi`; retire `use_x_official`/`use_x_paperpulse` | **Modify** |
| `configs/templates/config.template.ini` | same, with migration comment | **Modify** |
| `tests/test_hotspot_x_sources.py` | append `TestHotspotTwitterapiSource` with replay fixtures | **Modify (append)** |

---

## Task 1: Create `hotspot_twitterapi.py` skeleton + provider-key gate (degrade when no key)

**Files:**
- Create: `arxiv_assistant/apis/hotspot/hotspot_twitterapi.py`
- Test: `tests/test_hotspot_x_sources.py`

- [ ] **Step 1: Write the failing test** (append this class near the end of `tests/test_hotspot_x_sources.py`, before the `if __name__ == "__main__":` block)

```python
class TestHotspotTwitterapiSource(unittest.TestCase):
    def _seed_file(self, tmp_dir: str) -> Path:
        seed_path = Path(tmp_dir) / "x_seeds.json"
        seed_path.write_text(
            json.dumps(
                {
                    "accounts": [
                        {"handle": "openai", "name": "OpenAI", "kind": "official", "tier": 3, "active": True},
                        {"handle": "demishassabis", "name": "Demis Hassabis", "kind": "researcher", "tier": 3, "active": True},
                    ]
                }
            ),
            encoding="utf-8",
        )
        return seed_path

    def test_returns_empty_when_no_twitterapi_key_configured(self) -> None:
        from arxiv_assistant.apis.hotspot.hotspot_twitterapi import fetch_hotspot_items

        with tempfile.TemporaryDirectory() as tmp_dir, patch.dict(os.environ, {}, clear=True):
            seed_path = self._seed_file(tmp_dir)
            items = fetch_hotspot_items(
                datetime(2026, 3, 21, tzinfo=UTC),
                36,
                seed_path,
                snapshot_path=Path(tmp_dir) / "x_authority_inventory.json",
            )
        self.assertEqual(items, [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hotspot_x_sources.py::TestHotspotTwitterapiSource::test_returns_empty_when_no_twitterapi_key_configured -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'arxiv_assistant.apis.hotspot.hotspot_twitterapi'`

- [ ] **Step 3: Write minimal implementation** (create the file with imports, constants, key gate, and a stub `fetch_hotspot_items`)

```python
from __future__ import annotations

import os
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_hotspot_x_sources.py::TestHotspotTwitterapiSource::test_returns_empty_when_no_twitterapi_key_configured -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add arxiv_assistant/apis/hotspot/hotspot_twitterapi.py tests/test_hotspot_x_sources.py
git commit -m "feat(hotspot): scaffold hotspot_twitterapi adapter with key gate"
```

---

## Task 2: REST fetch + JSON normalization helpers

This task adds the provider-JSON helpers: classic-timestamp → ISO, tweet → canonical dict, the REST call with 429 degrade, and a per-account timeline iterator with a client-side freshness window. We test normalization against a **real twitterapi.io sample payload** with the network `@patch`ed.

**Files:**
- Modify: `arxiv_assistant/apis/hotspot/hotspot_twitterapi.py`
- Test: `tests/test_hotspot_x_sources.py`

- [ ] **Step 1: Write the failing test** (append to `TestHotspotTwitterapiSource`)

```python
    # Real-shape twitterapi.io payload (camelCase, Twitter classic createdAt).
    _OPENAI_PAYLOAD = {
        "tweets": [
            {
                "id": "2035012260008272007",
                "text": "We released GPT-5.4 mini today in ChatGPT, Codex, and the API. https://t.co/abc123",
                "createdAt": "Sat Mar 21 10:00:00 +0000 2026",
                "lang": "en",
                "inReplyToUserId": None,
                "retweeted_tweet": None,
                "entities": {"urls": [{"expanded_url": "https://openai.com/index/gpt-5-4-mini"}]},
                "likeCount": 1200,
                "replyCount": 90,
                "retweetCount": 150,
                "quoteCount": 40,
                "viewCount": 530000,
                "bookmarkCount": 100,
                "author": {"id": "1", "name": "OpenAI", "userName": "OpenAI", "isBlueVerified": True},
            }
        ]
    }

    def test_map_twitterapi_tweet_normalizes_camelcase_and_timestamp(self) -> None:
        from arxiv_assistant.apis.hotspot.hotspot_twitterapi import _map_twitterapi_tweet

        mapped = _map_twitterapi_tweet(self._OPENAI_PAYLOAD["tweets"][0], handle="openai", user_id="1")
        self.assertEqual(mapped["id"], "2035012260008272007")
        self.assertEqual(mapped["created_at"], "2026-03-21T10:00:00Z")
        self.assertEqual(mapped["public_metrics"]["like_count"], 1200)
        self.assertEqual(mapped["public_metrics"]["impression_count"], 530000)
        self.assertEqual(mapped["author"]["username"], "openai")
        self.assertTrue(mapped["author"]["verified"])

    def test_fetch_last_tweets_rest_returns_empty_on_429(self) -> None:
        from arxiv_assistant.apis.hotspot import hotspot_twitterapi as mod

        class _Resp:
            status_code = 429

            def raise_for_status(self) -> None:  # pragma: no cover - not reached on 429
                raise AssertionError("should not raise_for_status on 429")

            def json(self) -> dict:  # pragma: no cover - not reached on 429
                return {}

        with patch.object(mod.requests, "get", return_value=_Resp()), \
                patch.object(mod.time, "sleep", return_value=None):
            rows = mod._fetch_last_tweets_rest(
                user_id=None,
                handle="openai",
                api_key="k",
                since=datetime(2026, 3, 20, tzinfo=UTC),
                max_results=10,
            )
        self.assertEqual(rows, [])

    def test_fetch_last_tweets_rest_filters_by_since_window(self) -> None:
        from arxiv_assistant.apis.hotspot import hotspot_twitterapi as mod

        payload = {
            "tweets": [
                dict(self._OPENAI_PAYLOAD["tweets"][0]),  # 2026-03-21 (in window)
                {
                    **self._OPENAI_PAYLOAD["tweets"][0],
                    "id": "999",
                    "createdAt": "Mon Jan 05 10:00:00 +0000 2026",  # old, out of window
                },
            ]
        }

        class _Resp:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return payload

        with patch.object(mod.requests, "get", return_value=_Resp()):
            rows = mod._fetch_last_tweets_rest(
                user_id="1",
                handle="openai",
                api_key="k",
                since=datetime(2026, 3, 20, tzinfo=UTC),
                max_results=10,
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "2035012260008272007")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hotspot_x_sources.py::TestHotspotTwitterapiSource -v -k "map_twitterapi or rest"`
Expected: FAIL — `AttributeError: module ... has no attribute '_map_twitterapi_tweet'` (and `_fetch_last_tweets_rest`, `time`).

- [ ] **Step 3: Write minimal implementation** (add `import time` at top with the other stdlib imports, then add these helpers above `fetch_hotspot_items`)

Add to the import block (top of file, after `import os`):

```python
import time
```

Add these functions (place them after `_get_twitterapi_key`):

```python
def _classic_created_at_to_iso(value: Any) -> str:
    """twitterapi.io returns Twitter's classic format ('Tue Dec 10 07:00:00 +0000 2024').
    Normalize to ISO 8601 'Z' so is_fresh()/published_at behave exactly as with the official
    X API v2 (which already returns ISO). Falls back to the raw string if unparseable."""
    raw = clean_text(value)
    if not raw:
        return ""
    try:
        return datetime.strptime(raw, "%a %b %d %H:%M:%S %z %Y").astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, TypeError):
        return raw


def _map_twitterapi_tweet(tw: dict[str, Any], *, handle: str, user_id: str | None) -> dict[str, Any]:
    """Map a twitterapi.io tweet to the canonical X-API-v2 dict shape the rest of this module
    consumes (text, id, created_at, public_metrics, entities, in_reply_to_user_id, author)."""
    author = tw.get("author") or {}
    return {
        "id": clean_text(tw.get("id")),
        "text": tw.get("text"),
        "created_at": _classic_created_at_to_iso(tw.get("createdAt")),
        "lang": tw.get("lang"),
        "in_reply_to_user_id": clean_text(tw.get("inReplyToUserId")) or None,
        "referenced_tweets": ([{"type": "retweeted"}] if tw.get("retweeted_tweet") else []),
        "entities": tw.get("entities", {}) or {},
        "public_metrics": {
            "like_count": int(tw.get("likeCount", 0) or 0),
            "reply_count": int(tw.get("replyCount", 0) or 0),
            "retweet_count": int(tw.get("retweetCount", 0) or 0),
            "quote_count": int(tw.get("quoteCount", 0) or 0),
            "impression_count": int(tw.get("viewCount", 0) or 0),
            "bookmark_count": int(tw.get("bookmarkCount", 0) or 0),
        },
        "author": {
            "username": clean_text(author.get("userName") or handle).lower() or clean_text(handle).lower(),
            "id": user_id or clean_text(author.get("id")),
            "name": clean_text(author.get("name")),
            "verified": bool(author.get("isBlueVerified") or author.get("verified")),
        },
    }


def _twitterapi_get(params: dict[str, Any], *, api_key: str) -> dict[str, Any]:
    headers = {"X-API-Key": api_key}
    for attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
        response = requests.get(TWITTERAPI_IO_URL, headers=headers, params=params, timeout=30)
        if response.status_code == 429:
            if attempt < _RATE_LIMIT_MAX_RETRIES:
                wait = _RATE_LIMIT_WAIT_SECONDS * (attempt + 1)
                print(f"twitterapi.io rate limited (429). Waiting {wait}s before retry {attempt + 1}...")
                time.sleep(wait)
                continue
            print("twitterapi.io rate limit exceeded after retries. Stopping.")
            return {}
        response.raise_for_status()
        record_api_usage(source_id=SOURCE_ID, provider="twitterapi.io")
        return response.json()
    return {}


def _fetch_last_tweets_rest(
    *,
    user_id: str | None,
    handle: str,
    api_key: str,
    since: datetime,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    # last_tweets resolves by userName, so a missing user_id needs no extra lookup call.
    # Prefer userId when the registry cached one (provider notes it is more stable/faster).
    params: dict[str, Any] = {"includeReplies": "false"}
    if user_id:
        params["userId"] = user_id
    else:
        params["userName"] = handle
    try:
        payload = _twitterapi_get(params, api_key=api_key)
    except Exception as ex:  # network/HTTP errors degrade to empty, never crash the pipeline
        print(f"Warning: twitterapi.io timeline fetch failed for @{handle}: {ex}")
        return []
    # tolerate both flat {tweets:[...]} and wrapped {data:{tweets:[...]}} shapes
    container = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    raw_tweets = (container or {}).get("tweets", []) or []
    rows: list[dict[str, Any]] = []
    for tw in raw_tweets:
        mapped = _map_twitterapi_tweet(tw, handle=handle, user_id=user_id)
        created = mapped.get("created_at")
        if created:
            try:
                if datetime.strptime(created, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC) < since:
                    continue  # client-side window filter (last_tweets has no start_time param)
            except ValueError:
                pass
        rows.append(mapped)
        if len(rows) >= max_results:
            break
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_hotspot_x_sources.py::TestHotspotTwitterapiSource -v -k "map_twitterapi or rest"`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add arxiv_assistant/apis/hotspot/hotspot_twitterapi.py tests/test_hotspot_x_sources.py
git commit -m "feat(hotspot): twitterapi.io REST fetch + JSON normalization helpers"
```

---

## Task 3: Tweet → `HotspotItem` normalization (provenance, source_id, fields)

This task adds the per-tweet helpers and the `_tweet_to_item` builder, and asserts the normalized `HotspotItem` carries `source_id="x_twitterapi"`, `provenance` mirrored in metadata, correct `published_at`, and `x.com` permalink.

**Files:**
- Modify: `arxiv_assistant/apis/hotspot/hotspot_twitterapi.py`
- Test: `tests/test_hotspot_x_sources.py`

- [ ] **Step 1: Write the failing test** (append to `TestHotspotTwitterapiSource`)

```python
    def test_tweet_to_item_sets_provenance_and_canonical_fields(self) -> None:
        from arxiv_assistant.apis.hotspot import hotspot_twitterapi as mod

        row = mod._map_twitterapi_tweet(self._OPENAI_PAYLOAD["tweets"][0], handle="openai", user_id="1")
        authority = {"handle": "openai", "name": "OpenAI", "kind": "official", "tier": 3, "organization": "OpenAI"}
        item = mod._tweet_to_item(row, authority=authority)

        self.assertIsNotNone(item)
        self.assertEqual(item.source_id, "x_twitterapi")
        self.assertEqual(item.source_role, "official_news")
        self.assertEqual(item.source_type, "tweet")
        self.assertEqual(item.url, "https://x.com/OpenAI/status/2035012260008272007")
        self.assertEqual(item.published_at, "2026-03-21T10:00:00Z")
        self.assertEqual(item.metadata["provenance"], "native:x_twitterapi")
        self.assertEqual(item.metadata["source_id"], "x_twitterapi")
        self.assertEqual(item.metadata["authority_kind"], "official")
        self.assertEqual(item.metadata["author_handle"], "OpenAI")
        self.assertGreater(item.metadata["activity"], 500)

    def test_tweet_to_item_drops_replies_and_retweets(self) -> None:
        from arxiv_assistant.apis.hotspot import hotspot_twitterapi as mod

        reply = {**self._OPENAI_PAYLOAD["tweets"][0], "in_reply_to_user_id": "42"}
        authority = {"handle": "openai", "name": "OpenAI", "kind": "official", "tier": 3}
        # _map already ran on the raw payload above; here pass an already-canonical reply dict.
        canonical_reply = mod._map_twitterapi_tweet(reply, handle="openai", user_id="1")
        canonical_reply["in_reply_to_user_id"] = "42"
        self.assertIsNone(mod._tweet_to_item(canonical_reply, authority=authority))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hotspot_x_sources.py::TestHotspotTwitterapiSource -v -k "tweet_to_item"`
Expected: FAIL — `AttributeError: module ... has no attribute '_tweet_to_item'`

- [ ] **Step 3: Write minimal implementation** (add these helpers after `_fetch_last_tweets_rest`)

```python
def _derive_title(text: str) -> str:
    text = clean_text(text)
    if len(text) <= 180:
        return text
    for marker in (". ", "; ", ": "):
        if marker in text[:180]:
            return clean_text(text.split(marker, 1)[0])
    return clip_text(text, 180)


def _compute_activity(metrics: dict[str, Any] | None) -> int:
    metrics = metrics or {}
    likes = int(metrics.get("like_count", 0) or 0)
    replies = int(metrics.get("reply_count", 0) or 0)
    reposts = int(metrics.get("retweet_count", 0) or 0)
    quotes = int(metrics.get("quote_count", 0) or 0)
    impressions = int(metrics.get("impression_count", 0) or 0)
    bookmarks = int(metrics.get("bookmark_count", 0) or 0)
    return likes + bookmarks + replies * 3 + reposts * 2 + quotes * 4 + impressions // 1000


def _is_reply_or_retweet(tweet: dict[str, Any]) -> bool:
    text = clean_text(tweet.get("text")).lower()
    if text.startswith("rt @") or text.startswith("@"):
        return True
    if tweet.get("in_reply_to_user_id"):
        return True
    for ref in tweet.get("referenced_tweets", []) or []:
        if ref.get("type") in {"retweeted", "replied_to"}:
            return True
    return False


def _expanded_urls(tweet: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    entities = tweet.get("entities", {}) or {}
    for row in entities.get("urls", []) or []:
        candidate = clean_text(row.get("expanded_url") or row.get("unwound_url") or row.get("url"))
        if candidate:
            urls.append(candidate)
    return urls


def _authority_source_role(record: dict[str, Any]) -> str:
    if str(record.get("kind")) in {"official", "company"}:
        return "official_news"
    return "community_heat"


def _authority_source_quality(record: dict[str, Any]) -> float:
    tier = int(record.get("tier") or 1)
    kind = str(record.get("kind") or "researcher")
    base = 1.05 if kind == "researcher" else 1.2
    return round(base + tier * 0.12, 2)


def _set_provenance(item: HotspotItem) -> HotspotItem:
    """Set the Stage-0 `provenance` field if the dataclass exposes it, and ALWAYS mirror
    provenance + source_id into metadata so downstream works whether or not Stage 0 landed."""
    if hasattr(item, "provenance"):
        item.provenance = PROVENANCE
    item.metadata.setdefault("provenance", PROVENANCE)
    item.metadata.setdefault("source_id", SOURCE_ID)
    return item


def _tweet_to_item(row: dict[str, Any], *, authority: dict[str, Any]) -> HotspotItem | None:
    if _is_reply_or_retweet(row):
        return None

    author = row.get("author", {}) or {}
    author_handle = clean_text(authority.get("handle") or author.get("username"))
    text = clean_text(row.get("text"))
    tweet_id = clean_text(row.get("id"))
    if not author_handle or not text or not tweet_id:
        return None

    metrics = row.get("public_metrics", {}) or {}
    activity = _compute_activity(metrics)
    expanded_urls = _expanded_urls(row)
    if not is_newsworthy_x_text(
        text,
        authority_kind=str(authority.get("kind") or "official"),
        expanded_urls=expanded_urls,
        activity=activity,
    ):
        return None

    url = f"https://x.com/{author_handle}/status/{tweet_id}"
    non_x_urls = [entry for entry in expanded_urls if urlsplit(entry).netloc.lower() not in X_HOSTS]
    created_at = row.get("created_at")

    item = HotspotItem(
        source_id=SOURCE_ID,
        source_name=clean_text(authority.get("name") or author.get("name") or author_handle),
        source_role=_authority_source_role(authority),
        source_type="tweet",
        title=_derive_title(text),
        summary=clip_text(text, 420),
        url=url,
        canonical_url=url,
        published_at=created_at,
        tags=["x-twitterapi", str(authority.get("kind") or "official")],
        authors=[author_handle],
        metadata={
            "tweet_id": tweet_id,
            "author_handle": author_handle,
            "author_name": clean_text(author.get("name")),
            "verified": bool(author.get("verified")),
            "public_metrics": metrics,
            "activity": activity,
            "source_quality": _authority_source_quality(authority),
            "signal_tier": "x_twitterapi_timeline",
            "authority_kind": str(authority.get("kind") or "official"),
            "authority_tier": int(authority.get("tier") or 1),
            "organization": clean_text(authority.get("organization")),
            "expanded_urls": expanded_urls,
            "non_x_urls": non_x_urls,
            "has_external_link": bool(non_x_urls),
            "host": "x.com",
        },
    )
    return _set_provenance(item)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_hotspot_x_sources.py::TestHotspotTwitterapiSource -v -k "tweet_to_item"`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add arxiv_assistant/apis/hotspot/hotspot_twitterapi.py tests/test_hotspot_x_sources.py
git commit -m "feat(hotspot): twitterapi tweet to HotspotItem with native provenance"
```

---

## Task 4: Wire `fetch_hotspot_items` end-to-end (registry → fetch → filter → items)

This replaces the Task-1 stub body. Asserts the full path: an official "We released X" tweet survives (not killed by `SELF_WORK_PATTERNS`), counts meet the threshold, freshness filtering drops out-of-window tweets, and unknown (non-authority) handles are dropped.

**Files:**
- Modify: `arxiv_assistant/apis/hotspot/hotspot_twitterapi.py`
- Test: `tests/test_hotspot_x_sources.py`

- [ ] **Step 1: Write the failing test** (append to `TestHotspotTwitterapiSource`)

```python
    # Per-handle payloads keyed by the userName param twitterapi.io receives.
    _RESEARCHER_PAYLOAD = {
        "tweets": [
            {
                "id": "2035012260008272010",
                "text": "Strong new results on agent benchmarks and reasoning evals. https://t.co/paper",
                "createdAt": "Sat Mar 21 10:05:00 +0000 2026",
                "lang": "en",
                "inReplyToUserId": None,
                "retweeted_tweet": None,
                "entities": {"urls": [{"expanded_url": "https://arxiv.org/abs/2603.12345"}]},
                "likeCount": 400,
                "replyCount": 40,
                "retweetCount": 80,
                "quoteCount": 8,
                "viewCount": 90000,
                "bookmarkCount": 20,
                "author": {"id": "3", "name": "Demis Hassabis", "userName": "demishassabis", "isBlueVerified": True},
            }
        ]
    }

    def _fake_twitterapi_get(self, params, *, api_key):
        handle = (params.get("userName") or "").lower()
        if handle == "openai" or params.get("userId") == "openai":
            return self._OPENAI_PAYLOAD
        if handle == "demishassabis":
            return self._RESEARCHER_PAYLOAD
        return {"tweets": []}

    def test_fetch_hotspot_items_official_release_survives_filter(self) -> None:
        from arxiv_assistant.apis.hotspot import hotspot_twitterapi as mod

        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.dict(os.environ, {"TWITTERAPI_IO_KEY": "test-key"}, clear=True), \
                patch.object(mod, "_twitterapi_get", side_effect=self._fake_twitterapi_get):
            seed_path = self._seed_file(tmp_dir)
            items = mod.fetch_hotspot_items(
                datetime(2026, 3, 21, tzinfo=UTC),
                36,
                seed_path,
                result_limit=80,
                snapshot_path=Path(tmp_dir) / "x_authority_inventory.json",
            )

        # The official "We released GPT-5.4 mini today" tweet must NOT be killed by SELF_WORK_PATTERNS.
        self.assertGreaterEqual(len(items), 1)
        urls = {item.url for item in items}
        self.assertIn("https://x.com/OpenAI/status/2035012260008272007", urls)
        official = next(i for i in items if i.url.endswith("2035012260008272007"))
        self.assertEqual(official.source_id, "x_twitterapi")
        self.assertEqual(official.metadata["provenance"], "native:x_twitterapi")
        self.assertEqual(official.source_role, "official_news")

    def test_fetch_hotspot_items_drops_out_of_window_tweets(self) -> None:
        from arxiv_assistant.apis.hotspot import hotspot_twitterapi as mod

        stale_payload = {
            "tweets": [
                {
                    **self._OPENAI_PAYLOAD["tweets"][0],
                    "createdAt": "Mon Jan 05 10:00:00 +0000 2026",  # ~11 weeks old
                }
            ]
        }

        def _stale_get(params, *, api_key):
            if (params.get("userName") or "").lower() == "openai":
                return stale_payload
            return {"tweets": []}

        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.dict(os.environ, {"TWITTERAPI_IO_KEY": "test-key"}, clear=True), \
                patch.object(mod, "_twitterapi_get", side_effect=_stale_get):
            seed_path = self._seed_file(tmp_dir)
            items = mod.fetch_hotspot_items(
                datetime(2026, 3, 21, tzinfo=UTC),
                36,
                seed_path,
                snapshot_path=Path(tmp_dir) / "x_authority_inventory.json",
            )
        self.assertEqual(items, [])

    def test_fetch_hotspot_items_empty_response_degrades_cleanly(self) -> None:
        from arxiv_assistant.apis.hotspot import hotspot_twitterapi as mod

        with tempfile.TemporaryDirectory() as tmp_dir, \
                patch.dict(os.environ, {"TWITTERAPI_IO_KEY": "test-key"}, clear=True), \
                patch.object(mod, "_twitterapi_get", return_value={"tweets": []}):
            seed_path = self._seed_file(tmp_dir)
            items = mod.fetch_hotspot_items(
                datetime(2026, 3, 21, tzinfo=UTC),
                36,
                seed_path,
                snapshot_path=Path(tmp_dir) / "x_authority_inventory.json",
            )
        self.assertEqual(items, [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hotspot_x_sources.py::TestHotspotTwitterapiSource -v -k "fetch_hotspot_items"`
Expected: FAIL — `test_..._release_survives_filter` returns `[]` (stub body), so `assertGreaterEqual(len(items), 1)` fails.

- [ ] **Step 3: Write minimal implementation** (replace the Task-1 stub `fetch_hotspot_items` body — keep the key gate — with the full implementation)

```python
def _collect_timelines(
    accounts: list[dict[str, Any]],
    *,
    api_key: str,
    since: datetime,
    tweets_per_user: int,
    result_limit: int,
    registry: dict[str, Any],
    seen_urls: set[str],
) -> list[HotspotItem]:
    items: list[HotspotItem] = []
    for account in accounts:
        if len(items) >= result_limit:
            break
        handle = clean_text(account.get("handle"))
        if not handle:
            continue
        user_id = clean_text(account.get("x_user_id")) or None
        rows = _fetch_last_tweets_rest(
            user_id=user_id,
            handle=handle,
            api_key=api_key,
            since=since,
            max_results=tweets_per_user,
        )
        for row in rows:
            author = row.get("author", {}) or {}
            author_handle = clean_text(author.get("username")) or handle
            if not is_authoritative_x_identity(author_handle, registry=registry):
                continue
            authority = get_authority_record(author_handle, registry=registry) or account
            item = _tweet_to_item(row, authority=authority)
            if item is None:
                continue
            if item.url in seen_urls:
                continue
            seen_urls.add(item.url)
            items.append(item)
            if len(items) >= result_limit:
                break
    return items


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

    if target_date.tzinfo is None:
        since = target_date.replace(tzinfo=UTC) - timedelta(hours=freshness_hours)
    else:
        since = target_date - timedelta(hours=freshness_hours)

    registry = load_x_authority_registry(
        seed_path=seed_path, snapshot_path=snapshot_path, max_age_hours=max_age_hours
    )
    official_accounts = iter_active_authority_accounts(registry, kinds={"official", "company"}, min_tier=2)
    researcher_accounts = iter_active_authority_accounts(registry, kinds={"researcher"}, min_tier=2)
    official_accounts = official_accounts[:official_account_limit]
    researcher_accounts = researcher_accounts[:researcher_account_limit]

    seen_urls: set[str] = set()
    items = _collect_timelines(
        official_accounts,
        api_key=api_key,
        since=since,
        tweets_per_user=tweets_per_user,
        result_limit=result_limit,
        registry=registry,
        seen_urls=seen_urls,
    )
    items.extend(
        _collect_timelines(
            researcher_accounts,
            api_key=api_key,
            since=since,
            tweets_per_user=tweets_per_user,
            result_limit=max(0, result_limit // 3),
            registry=registry,
            seen_urls=seen_urls,
        )
    )

    # Server-side window is approximate; enforce the canonical is_fresh() gate too.
    fresh_items = [
        item
        for item in items
        if item.published_at is None or is_fresh(item.published_at, target_date, freshness_hours)
    ]
    return fresh_items[:result_limit]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_hotspot_x_sources.py::TestHotspotTwitterapiSource -v -k "fetch_hotspot_items"`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the whole adapter test class**

Run: `pytest tests/test_hotspot_x_sources.py::TestHotspotTwitterapiSource -v`
Expected: PASS (all 9 tests in the class)

- [ ] **Step 6: Commit**

```bash
git add arxiv_assistant/apis/hotspot/hotspot_twitterapi.py tests/test_hotspot_x_sources.py
git commit -m "feat(hotspot): wire twitterapi fetch_hotspot_items end-to-end"
```

---

## Task 5: Pipeline fan-out — register `x_twitterapi` behind `use_twitterapi`

Add the import and a source spec so the adapter participates in the daily harvest. The spec passes the X registry paths already resolved earlier in `generate_daily_hotspot_report` (`x_seed_path`, `x_registry_snapshot_path`, `x_registry_max_age_hours`).

**Files:**
- Modify: `arxiv_assistant/hotspots/pipeline.py` (import near line 24; spec block near line 945)
- Test: `tests/test_hotspot_x_sources.py`

- [ ] **Step 1: Write the failing test** (append to `TestHotspotTwitterapiSource`)

```python
    def test_pipeline_exports_twitterapi_fetcher_symbol(self) -> None:
        # The pipeline must import the adapter under a stable alias for the fan-out spec.
        from arxiv_assistant.hotspots import pipeline

        self.assertTrue(hasattr(pipeline, "fetch_x_twitterapi_items"))
        self.assertIs(
            pipeline.fetch_x_twitterapi_items,
            __import__(
                "arxiv_assistant.apis.hotspot.hotspot_twitterapi",
                fromlist=["fetch_hotspot_items"],
            ).fetch_hotspot_items,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hotspot_x_sources.py::TestHotspotTwitterapiSource::test_pipeline_exports_twitterapi_fetcher_symbol -v`
Expected: FAIL — `AttributeError: module 'arxiv_assistant.hotspots.pipeline' has no attribute 'fetch_x_twitterapi_items'`

- [ ] **Step 3a: Add the import** (in `arxiv_assistant/hotspots/pipeline.py`, immediately after the existing line 25 `from arxiv_assistant.apis.hotspot.hotspot_x_paperpulse import fetch_hotspot_items as fetch_x_paperpulse_items`)

```python
from arxiv_assistant.apis.hotspot.hotspot_twitterapi import fetch_hotspot_items as fetch_x_twitterapi_items
```

- [ ] **Step 3b: Register the source spec** (in `generate_daily_hotspot_report`, insert this block immediately BEFORE the existing `if hotspot_sources.getboolean("use_x_official", fallback=False):` block near line 946)

```python
    if hotspot_sources.getboolean("use_twitterapi", fallback=True):
        specs.append(
            (
                "x_twitterapi",
                lambda: fetch_x_twitterapi_items(
                    target_date,
                    freshness_hours,
                    x_seed_path,
                    result_limit=int(x_config.get("list_result_limit", 80)),
                    snapshot_path=x_registry_snapshot_path,
                    max_age_hours=x_registry_max_age_hours,
                ),
            )
        )
```

- [ ] **Step 3c: Retire the superseded specs** — change the two old gates so they default OFF and only run if a config explicitly opts back in. Replace `fallback=True`/`fallback=False` on the `use_x_paperpulse` and `use_x_official` gates with `fallback=False`:

In the `use_x_paperpulse` block (near line 935), change:
```python
    if hotspot_sources.getboolean("use_x_paperpulse", fallback=True):
```
to:
```python
    if hotspot_sources.getboolean("use_x_paperpulse", fallback=False):  # superseded by use_twitterapi (Stage 5)
```

The `use_x_official` gate is already `fallback=False`; append the same migration comment:
```python
    if hotspot_sources.getboolean("use_x_official", fallback=False):  # superseded by use_twitterapi (Stage 5)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_hotspot_x_sources.py::TestHotspotTwitterapiSource::test_pipeline_exports_twitterapi_fetcher_symbol -v`
Expected: PASS

- [ ] **Step 5: Sanity-check the pipeline still imports**

Run: `python -c "import arxiv_assistant.hotspots.pipeline as p; print('x_twitterapi' if hasattr(p, 'fetch_x_twitterapi_items') else 'MISSING')"`
Expected: prints `x_twitterapi`

- [ ] **Step 6: Commit**

```bash
git add arxiv_assistant/hotspots/pipeline.py tests/test_hotspot_x_sources.py
git commit -m "feat(hotspot): register x_twitterapi source behind use_twitterapi, retire x_official/x_paperpulse"
```

---

## Task 6: Config — enable `use_twitterapi`, retire legacy X flags, document migration

**Files:**
- Modify: `configs/config.ini` (`[HOTSPOT_SOURCES]`, lines 109-111)
- Modify: `configs/templates/config.template.ini` (`[HOTSPOT_SOURCES]`, lines 57-59)

- [ ] **Step 1: Edit `configs/config.ini`** — replace the three legacy X-source lines (109-111):

Replace:
```ini
use_x_ainews_twitter = true
use_x_paperpulse = true
use_x_official = true
```
with:
```ini
use_x_ainews_twitter = true
# X/Twitter direct channel: twitterapi.io (managed, ~$0.15/1k tweets, set TWITTERAPI_IO_KEY).
# Supersedes use_x_official (X API Pro $5000/mo) and use_x_paperpulse (upstream feed dead).
use_twitterapi = true
use_x_paperpulse = false
use_x_official = false
```

- [ ] **Step 2: Edit `configs/templates/config.template.ini`** — replace the three legacy X-source lines (57-59):

Replace:
```ini
use_x_ainews_twitter = true
use_x_paperpulse = true
use_x_official = false
```
with:
```ini
use_x_ainews_twitter = true
# Direct X/Twitter channel via twitterapi.io. Requires the TWITTERAPI_IO_KEY environment
# variable (see .env.example / README). Supersedes the retired use_x_official and
# use_x_paperpulse sources, which always returned zero items (X API Pro $5000/mo;
# PaperPulse feed dead). Leave the two legacy flags false.
use_twitterapi = true
use_x_paperpulse = false
use_x_official = false
```

- [ ] **Step 3: Verify both configs parse**

Run: `python -c "import configparser; c=configparser.ConfigParser(); c.read(['configs/config.ini','configs/templates/config.template.ini']); s=c['HOTSPOT_SOURCES']; print('use_twitterapi=', s.getboolean('use_twitterapi'), 'x_official=', s.getboolean('use_x_official'), 'x_paperpulse=', s.getboolean('use_x_paperpulse'))"`
Expected: prints `use_twitterapi= True x_official= False x_paperpulse= False`

- [ ] **Step 4: Commit**

```bash
git add configs/config.ini configs/templates/config.template.ini
git commit -m "config(hotspot): enable use_twitterapi, retire use_x_official/use_x_paperpulse"
```

---

## Task 7: Full-suite regression + threshold/volume sanity

Confirm the new adapter does not break the existing X-source tests and that the volume threshold (≥1 newsworthy item from the two-account fixture, and the official release surviving) holds.

**Files:**
- Test: `tests/test_hotspot_x_sources.py` (no new code; run the suite)

- [ ] **Step 1: Run the full X-sources test file**

Run: `pytest tests/test_hotspot_x_sources.py -v`
Expected: All `TestHotspotTwitterapiSource` tests PASS.

Note: the pre-existing `TestHotspotXSources` class imports `_query_accounts` and `_iter_recent_search` from `hotspot_x_official`, which the current timeline-based `hotspot_x_official.py` no longer defines. Those are **pre-existing failures unrelated to Stage 5** — do not "fix" them in this stage. If they error at collection time and block your class from running, run only the new class explicitly:

Run: `pytest "tests/test_hotspot_x_sources.py::TestHotspotTwitterapiSource" -v`
Expected: PASS (10 tests).

- [ ] **Step 2: Run the broader hotspot pipeline test to confirm the import wiring is intact**

Run: `pytest tests/test_hotspot_pipeline.py -v`
Expected: PASS (or unchanged from the pre-Stage-5 baseline — record any pre-existing failures and confirm Stage 5 added none).

- [ ] **Step 3: Commit (if any test-only adjustments were needed; otherwise skip)**

```bash
git add tests/test_hotspot_x_sources.py
git commit -m "test(hotspot): confirm x_twitterapi adapter passes full X-source suite"
```

---

## §G invariant acceptance checks (stage exit criteria)

This stage is **not done** until all are green:

- **Channel restored (the whole point):** `test_fetch_hotspot_items_official_release_survives_filter` proves an official "We released GPT-5.4 mini today" tweet flows through twitterapi.io → normalize → `is_newsworthy_x_text` (official exemption) → `HotspotItem`. This is the X≈0 root-cause fix (channel, not filter).
- **Provenance/normalization (contract §2.1):** every emitted item has `source_id == "x_twitterapi"`, `metadata["provenance"] == "native:x_twitterapi"`, `metadata["source_id"] == "x_twitterapi"`, and the Stage-0 `provenance` field is set when present (`_set_provenance`). Covered by `test_tweet_to_item_sets_provenance_and_canonical_fields`.
- **Freshness gate (INV2 spirit):** out-of-window tweets are dropped by both the client-side `since` filter and the canonical `is_fresh()` gate. Covered by `test_fetch_hotspot_items_drops_out_of_window_tweets` and `test_fetch_last_tweets_rest_filters_by_since_window`.
- **Degrade-not-crash (spec §E reliability):** missing key, empty response, and HTTP 429 each return `[]` (or partial) without raising. Covered by `test_returns_empty_when_no_twitterapi_key_configured`, `test_fetch_hotspot_items_empty_response_degrades_cleanly`, `test_fetch_last_tweets_rest_returns_empty_on_429`.
- **No network in tests (§4):** every test `@patch`es `_twitterapi_get`/`requests.get`; none hits the wire.
- **Config migration (§3):** `use_twitterapi=true` present in both configs; `use_x_official`/`use_x_paperpulse` retired to `false` with migration comments; pipeline gates default-safe.

---

## Notes for the implementer

- **Do not modify `hotspot_x_common.py`.** The official-account exemption from `SELF_WORK_PATTERNS` is already in place (`is_newsworthy_x_text`: only `authority_kind == "researcher"` is blocked by `SELF_WORK_PATTERNS`; official/company accounts return `True` on `has_self_work`). Stage 5 relies on that existing behavior — adding the exemption is out of scope.
- **Do not touch `hotspot_x_official.py`.** It keeps its own twitterapi.io fallback for the legacy `use_x_official` path; Stage 5 is a separate, dedicated adapter with the canonical `source_id="x_twitterapi"`/`provenance="native:x_twitterapi"` identity required by the contract. The two can coexist; only `use_twitterapi` is enabled by default.
- **MCP seam (Stage 6 handoff):** `_fetch_last_tweets_rest` is the single network boundary. A Stage-6 MCP-backed variant (using `mcp__twitterapi-mcp__get_user_last_tweets`) can substitute for it behind the same signature `(*, user_id, handle, api_key, since, max_results) -> list[dict]` returning canonical X-API-v2 dicts — `_tweet_to_item` and `fetch_hotspot_items` stay unchanged.
- **`x_user_id` cache:** `iter_active_authority_accounts` rows may carry `x_user_id`; when present the adapter sends `userId` (more stable/faster per the provider) instead of `userName`. Both paths are exercised: official fixture is matched by `userName`, the `since`-window test passes `user_id="1"`.
