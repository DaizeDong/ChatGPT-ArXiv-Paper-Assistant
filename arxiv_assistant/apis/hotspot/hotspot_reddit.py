"""Fetch AI-related hot posts from Reddit subreddits via public JSON API."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem, clean_text
from arxiv_assistant.utils.hotspot.hotspot_sources import clip_text, fetch_json

SUBREDDITS = [
    {"name": "MachineLearning", "min_score": 50},
    {"name": "artificial", "min_score": 30},
    {"name": "LocalLLaMA", "min_score": 40},
]

REDDIT_JSON_SUFFIX = "/hot.json?limit=25"


def fetch_hotspot_items(
    target_date: datetime,
    freshness_hours: int,
    *,
    result_limit: int = 15,
) -> list[HotspotItem]:
    items: list[HotspotItem] = []
    seen_urls: set[str] = set()
    cutoff = target_date - timedelta(hours=freshness_hours)

    for sub_config in SUBREDDITS:
        subreddit = sub_config["name"]
        min_score = sub_config["min_score"]
        url = f"https://www.reddit.com/r/{subreddit}{REDDIT_JSON_SUFFIX}"

        try:
            payload = fetch_json(url, timeout=15)
        except Exception as ex:
            print(f"Warning: failed to fetch Reddit r/{subreddit}: {ex}")
            continue

        posts = payload.get("data", {}).get("children", [])
        for post_wrapper in posts:
            post = post_wrapper.get("data", {})
            if not post:
                continue

            score = int(post.get("score", 0) or 0)
            if score < min_score:
                continue

            created_utc = post.get("created_utc")
            if created_utc:
                post_dt = datetime.fromtimestamp(float(created_utc), tz=UTC)
                if post_dt < cutoff:
                    continue
            else:
                continue

            title = clean_text(post.get("title", ""))
            if not title or len(title) < 15:
                continue

            permalink = post.get("permalink", "")
            post_url = f"https://www.reddit.com{permalink}" if permalink else ""
            if not post_url or post_url in seen_urls:
                continue
            seen_urls.add(post_url)

            # Prefer external URL if available
            external_url = post.get("url", "")
            is_self = post.get("is_self", True)
            canonical = external_url if external_url and not is_self else post_url

            selftext = clean_text(post.get("selftext", ""))
            num_comments = int(post.get("num_comments", 0) or 0)

            items.append(
                HotspotItem(
                    source_id="reddit",
                    source_name=f"Reddit r/{subreddit}",
                    source_role="community_heat",
                    source_type="discussion",
                    title=title,
                    summary=clip_text(selftext, 400) if selftext else "",
                    url=post_url,
                    canonical_url=canonical,
                    published_at=post_dt.isoformat() if created_utc else None,
                    tags=["reddit", subreddit.lower()],
                    authors=[clean_text(post.get("author", ""))] if post.get("author") else [],
                    metadata={
                        "reddit_score": score,
                        "num_comments": num_comments,
                        "subreddit": subreddit,
                        "activity": score + num_comments * 3,
                        "source_quality": 1.0,
                        "is_self": is_self,
                        "external_url": external_url if not is_self else "",
                        "host": "reddit.com",
                    },
                )
            )

            if len(items) >= result_limit:
                return items

    return items
