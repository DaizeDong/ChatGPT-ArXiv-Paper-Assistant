from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

import feedparser
from bs4 import BeautifulSoup

from arxiv_assistant.utils.hotspot.hotspot_schema import HotspotItem, clean_text
from arxiv_assistant.utils.hotspot.hotspot_sources import clip_text, fetch_text, is_fresh, normalize_url, parse_datetime

FALLBACK_OFFICIAL_SOURCES = [
    {
        "source_id": "openai_news",
        "source_name": "OpenAI News",
        "url": "https://openai.com/news/rss.xml",
        "mode": "rss",
    },
    {
        "source_id": "anthropic_news",
        "source_name": "Anthropic News",
        "url": "https://www.anthropic.com/news",
        "mode": "anthropic_html",
    },
    {
        "source_id": "google_ai_blog",
        "source_name": "Google AI Blog",
        "url": "https://blog.google/innovation-and-ai/technology/ai/rss/",
        "mode": "rss",
    },
    {
        "source_id": "meta_ai_blog",
        "source_name": "Meta AI Blog",
        "url": "https://ai.meta.com/blog/",
        "mode": "meta_html",
    },
]


def _load_official_blog_registry(registry_path: str | None = None) -> list[dict[str, Any]]:
    if registry_path:
        from pathlib import Path
        path = Path(registry_path)
        if path.exists():
            import json
            sources = json.loads(path.read_text(encoding="utf-8"))
            return [s for s in sources if s.get("enabled", True)]
    return FALLBACK_OFFICIAL_SOURCES
GENERIC_TITLES = {"news", "newsroom", "ai", "blog", "home", "featured", "learn more"}

# Navigation / CTA links that generic HTML extractors should skip
_NAV_CTA_RE = re.compile(
    r"^(?:contact\s*(?:us|sales)?|get\s*(?:started|a\s*demo)|"
    r"try\s+[\w\s]+|request\s+(?:a\s+)?demo|sign\s*(?:up|in)|log\s*in|"
    r"enterprise\s*sales|schedule\s+(?:a\s+)?(?:demo|call)|"
    r"(?:commercial|business)\s+(?:license|partnership|inquiry)|"
    r"media\s+inquiry|playground|documentation|about\s+\w+|"
    r"join\s+(?:us|our)|careers?|press|pricing|"
    r"news\s*(?:&|and)\s*updates|models?\s*(?:&|and)\s*pricing|"
    r"newsletters?(?:\s*signup)?|(?:website\s+)?terms\s+of\s+(?:use|service)|"
    r"privacy\s+policy|cookie\s+policy|(?:other\s+)?terms\s*(?:&|and)\s*policies|"
    r"service\s+status|build\s+on\s+[\w\s]+|talk\s+to\s+[\w\s]+|"
    r"trust\s+center|change\s*log|this\s+documentation|"
    r"our\s+approach|infrastructure|overview|faq|support|"
    r"in\s+the\s+news|customer\s+spotlight|publications?|whitepapers?|"
    r"explore\s+[\w\s]+|get\s+[\w\s]+|about\s+[\w\s]+|"
    r"[\w.-]+@[\w.-]+|"  # email addresses
    r"(?:中文|english)[\s（(].*|"  # language switchers
    r"read\s+more|learn\s+more|see\s+(?:all|more)|view\s+all|"
    r"models?\s*(?:&|and)\s*pricing)$",
    re.IGNORECASE,
)

# Elements that are definitely navigation, not content
_NAV_TAGS = {"nav", "header", "footer"}
DATE_PATTERN = re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}\b", re.I)
ANTHROPIC_PREFIX_PATTERN = re.compile(
    r"^(?:(?:Announcements|Product|Policy|Research|Safety)\s+)?"
    r"(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}\s+)?"
    r"(?:(?:Announcements|Product|Policy|Research|Safety)\s+)?",
    re.I,
)


def _normalize_official_title(title: str) -> str:
    normalized = clean_text(ANTHROPIC_PREFIX_PATTERN.sub("", title or ""))
    normalized = clean_text(DATE_PATTERN.sub("", normalized))
    for marker in (" We’re ", " We're ", " Learn how ", " Read more "):
        if marker in normalized:
            normalized = normalized.split(marker, 1)[0].strip()
    if len(normalized) > 140 and ". " in normalized:
        normalized = normalized.split(". ", 1)[0].strip()
    return clean_text(normalized)


def _looks_generic_title(title: str, url: str, base_url: str) -> bool:
    normalized = clean_text(title).lower()
    if normalized in GENERIC_TITLES:
        return True
    return normalize_url(url) == normalize_url(base_url)


def _extract_date_from_text(text: str) -> str | None:
    match = DATE_PATTERN.search(text)
    if match is None:
        return None
    parsed = parse_datetime(match.group(0))
    return parsed.astimezone(UTC).isoformat() if parsed else None


def _build_item(source: dict[str, str], title: str, summary: str, url: str, published_at: str | None) -> HotspotItem:
    return HotspotItem(
        source_id=source["source_id"],
        source_name=source["source_name"],
        source_role="official_news",
        source_type="official_blog",
        title=title,
        summary=clip_text(summary, 650),
        url=url,
        canonical_url=url,
        published_at=published_at,
        tags=[],
        authors=[],
        metadata={"is_official": True, "publisher": source["source_name"]},
    )


def _fetch_rss_source(source: dict[str, str], target_date: datetime, freshness_hours: int) -> list[HotspotItem]:
    feed = feedparser.parse(fetch_text(source["url"]))
    items: list[HotspotItem] = []
    for entry in feed.entries:
        published_at = entry.get("published") or entry.get("updated")
        if not is_fresh(published_at, target_date, freshness_hours):
            continue
        items.append(
            _build_item(
                source,
                title=entry.get("title", source["source_name"]),
                summary=entry.get("summary", "") or entry.get("description", ""),
                url=entry.get("link", source["url"]),
                published_at=published_at,
            )
        )
    return items[:8]


def _extract_anthropic_rows(page_html: str, base_url: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(page_html, "html.parser")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        href = normalize_url(anchor["href"], base_url)
        if "/news/" not in href or href in seen:
            continue
        title_node = anchor.find(["h1", "h2", "h3", "h4"])
        summary_node = anchor.find("p")
        time_node = anchor.find("time")

        published_at = None
        if time_node:
            published_at = time_node.get("datetime") or clean_text(time_node.get_text(" ", strip=True))

        title = clean_text(title_node.get_text(" ", strip=True)) if title_node else clean_text(anchor.get_text(" ", strip=True))
        summary = clean_text(summary_node.get_text(" ", strip=True)) if summary_node else ""
        combined_text = clean_text(anchor.get_text(" ", strip=True))

        title = _normalize_official_title(title)
        if not title or _looks_generic_title(title, href, base_url):
            continue
        if not published_at:
            published_at = _extract_date_from_text(combined_text)
        if not published_at:
            continue
        if not summary:
            summary = clean_text(ANTHROPIC_PREFIX_PATTERN.sub("", combined_text))
            if summary.startswith(title):
                summary = clean_text(summary[len(title):])

        seen.add(href)
        rows.append({"title": title, "summary": summary, "url": href, "published_at": published_at})
        if len(rows) >= 8:
            break
    return rows


def _extract_meta_rows(page_html: str, base_url: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(page_html, "html.parser")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        href = normalize_url(anchor["href"], base_url)
        title = clean_text(anchor.get_text(" ", strip=True))
        if "/blog/" not in href or href == normalize_url(base_url) or href in seen:
            continue
        title = _normalize_official_title(title)
        if not title or _looks_generic_title(title, href, base_url):
            continue

        container = anchor.parent
        parent_text = clean_text(container.get_text(" ", strip=True)) if container else title
        published_at = _extract_date_from_text(parent_text)
        if not published_at:
            continue

        summary = parent_text.replace(title, "", 1)
        summary = DATE_PATTERN.sub("", summary)
        summary = clean_text(summary)

        seen.add(href)
        rows.append({"title": title, "summary": summary, "url": href, "published_at": published_at})
        if len(rows) >= 8:
            break
    return rows


def _find_article_container(element) -> Any:
    """Walk up the DOM to find the nearest article/section/card container."""
    for _ in range(5):
        parent = element.parent
        if parent is None:
            break
        tag = getattr(parent, "name", None)
        if tag in ("article", "section", "li"):
            return parent
        css_class = " ".join(parent.get("class", []))
        if any(kw in css_class.lower() for kw in ("card", "post", "item", "entry", "blog")):
            return parent
        element = parent
    return element.parent


def _extract_generic_blog_rows(page_html: str, base_url: str) -> list[dict[str, Any]]:
    """Generic HTML blog extractor for blogs without RSS.

    Strategy: try multiple approaches in order:
    1. JSON-LD structured data (most reliable when available)
    2. <article> tags with internal links
    3. Anchor tags with heading children (original approach)
    4. Anchors inside card-like containers
    """
    soup = BeautifulSoup(page_html, "html.parser")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    base_domain = re.sub(r"^www\.", "", (base_url.split("//", 1)[-1]).split("/", 1)[0])
    base_normalized = normalize_url(base_url)

    def _is_internal(href: str) -> bool:
        return base_domain in href and href != base_normalized

    def _in_nav_area(el) -> bool:
        """Check if element is inside nav/header/footer."""
        for parent in el.parents:
            if getattr(parent, "name", None) in _NAV_TAGS:
                return True
        return False

    def _try_add(title: str, summary: str, href: str, published_at: str | None) -> bool:
        if href in seen:
            return False
        title = _normalize_official_title(title)
        if not title or _looks_generic_title(title, href, base_url) or len(title) < 10:
            return False
        if _NAV_CTA_RE.match(title.strip()):
            return False
        seen.add(href)
        rows.append({"title": title, "summary": summary, "url": href, "published_at": published_at})
        return True

    # --- Strategy 1: JSON-LD structured data ---
    import json as _json
    for script_tag in soup.find_all("script", type="application/ld+json"):
        try:
            ld = _json.loads(script_tag.string or "")
            ld_items = ld if isinstance(ld, list) else ld.get("itemListElement", [ld])
            for item in ld_items:
                obj = item.get("item", item) if isinstance(item, dict) else item
                if not isinstance(obj, dict):
                    continue
                ld_type = obj.get("@type", "")
                if ld_type not in ("BlogPosting", "NewsArticle", "Article", "WebPage"):
                    continue
                href = normalize_url(obj.get("url", ""), base_url)
                if not _is_internal(href):
                    continue
                title = clean_text(obj.get("headline", obj.get("name", "")))
                summary = clean_text(obj.get("description", ""))
                pub = obj.get("datePublished") or obj.get("dateCreated")
                if pub:
                    parsed = parse_datetime(pub)
                    pub = parsed.astimezone(UTC).isoformat() if parsed else None
                _try_add(title, summary, href, pub)
                if len(rows) >= 8:
                    return rows
        except Exception:
            continue

    # --- Strategy 2: <article> tags ---
    for article in soup.find_all("article"):
        anchor = article.find("a", href=True)
        if not anchor:
            continue
        href = normalize_url(anchor["href"], base_url)
        if not _is_internal(href):
            continue
        title_node = article.find(["h1", "h2", "h3", "h4", "h5"])
        title = clean_text(title_node.get_text(" ", strip=True)) if title_node else clean_text(anchor.get_text(" ", strip=True))
        summary_node = article.find("p")
        summary = clean_text(summary_node.get_text(" ", strip=True)) if summary_node else ""
        time_node = article.find("time")
        published_at = None
        if time_node:
            published_at = time_node.get("datetime") or clean_text(time_node.get_text(" ", strip=True))
            if published_at:
                parsed = parse_datetime(published_at)
                published_at = parsed.astimezone(UTC).isoformat() if parsed else None
        if not published_at:
            published_at = _extract_date_from_text(clean_text(article.get_text(" ", strip=True)))
        _try_add(title, summary, href, published_at)
        if len(rows) >= 8:
            return rows

    # --- Strategy 3: Anchor tags (original approach, broadened) ---
    for anchor in soup.find_all("a", href=True):
        href = normalize_url(anchor["href"], base_url)
        if not _is_internal(href) or href in seen:
            continue
        if _in_nav_area(anchor):
            continue

        # Check for title in heading child or anchor text
        title_node = anchor.find(["h1", "h2", "h3", "h4", "h5"])
        title = clean_text(title_node.get_text(" ", strip=True)) if title_node else ""

        # Also try: anchor has a strong/span with substantial text
        if not title:
            for tag in anchor.find_all(["strong", "span", "div"]):
                candidate = clean_text(tag.get_text(" ", strip=True))
                if len(candidate) >= 15:
                    title = candidate
                    break

        if not title:
            title = clean_text(anchor.get_text(" ", strip=True))

        title = _normalize_official_title(title)
        if not title or _looks_generic_title(title, href, base_url) or len(title) < 10:
            continue

        # Walk up to find the card/article container for richer metadata
        container = _find_article_container(anchor)
        container_text = clean_text(container.get_text(" ", strip=True)) if container else ""

        published_at = _extract_date_from_text(container_text)
        time_node = anchor.find("time") or (container.find("time") if container else None)
        if not published_at and time_node:
            published_at = time_node.get("datetime") or clean_text(time_node.get_text(" ", strip=True))
            if published_at:
                parsed = parse_datetime(published_at)
                published_at = parsed.astimezone(UTC).isoformat() if parsed else None

        # Look for date in data-* attributes on container
        if not published_at and container:
            for attr_name, attr_val in (container.attrs or {}).items():
                if "date" in attr_name.lower() and isinstance(attr_val, str):
                    parsed = parse_datetime(attr_val)
                    if parsed:
                        published_at = parsed.astimezone(UTC).isoformat()
                        break

        summary_node = anchor.find("p") or (container.find("p") if container else None)
        summary = clean_text(summary_node.get_text(" ", strip=True)) if summary_node else ""

        _try_add(title, summary, href, published_at)
        if len(rows) >= 8:
            break
    return rows


def _fetch_sitemap_source(source: dict[str, Any], target_date: datetime, freshness_hours: int) -> list[HotspotItem]:
    """Fetch blog posts by discovering URLs from sitemap.xml, then scraping individual SSR pages."""
    from xml.etree import ElementTree

    sitemap_url = source.get("sitemap_url", "")
    if not sitemap_url:
        return []
    url_contains = source.get("sitemap_url_contains", "/blog/")

    try:
        xml_text = fetch_text(sitemap_url)
        root = ElementTree.fromstring(xml_text)
    except Exception as ex:
        print(f"Warning: failed to fetch sitemap {sitemap_url}: {ex}")
        return []

    ns = {"ns": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    url_entries: list[tuple[str, str | None]] = []
    for url_el in root.findall(".//ns:url", ns):
        loc_el = url_el.find("ns:loc", ns)
        lastmod_el = url_el.find("ns:lastmod", ns)
        if loc_el is None or not loc_el.text:
            continue
        loc = loc_el.text.strip()
        if url_contains and url_contains not in loc:
            continue
        lastmod = lastmod_el.text.strip() if lastmod_el is not None and lastmod_el.text else None
        # Filter by freshness if lastmod is available
        if lastmod and not is_fresh(lastmod, target_date, freshness_hours):
            continue
        url_entries.append((loc, lastmod))

    # Sort by lastmod (most recent first), then take top entries
    url_entries.sort(key=lambda x: x[1] or "", reverse=True)
    url_entries = url_entries[:8]

    items: list[HotspotItem] = []
    for page_url, lastmod in url_entries:
        try:
            page_html = fetch_text(page_url)
        except Exception:
            continue
        soup = BeautifulSoup(page_html, "html.parser")

        # Extract title from <title>, <h1>, or og:title
        title = ""
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = clean_text(og_title["content"])
        if not title:
            h1 = soup.find("h1")
            title = clean_text(h1.get_text(" ", strip=True)) if h1 else ""
        if not title:
            title_tag = soup.find("title")
            title = clean_text(title_tag.get_text(" ", strip=True)) if title_tag else ""
        title = _normalize_official_title(title)
        if not title or len(title) < 10:
            continue

        # Extract summary from og:description or first <p>
        summary = ""
        og_desc = soup.find("meta", property="og:description") or soup.find("meta", attrs={"name": "description"})
        if og_desc and og_desc.get("content"):
            summary = clean_text(og_desc["content"])
        if not summary:
            # Find first substantial paragraph
            for p in soup.find_all("p"):
                p_text = clean_text(p.get_text(" ", strip=True))
                if len(p_text) > 60:
                    summary = p_text
                    break

        # Extract date from page if sitemap didn't provide one
        published_at = None
        if lastmod:
            parsed = parse_datetime(lastmod)
            published_at = parsed.astimezone(UTC).isoformat() if parsed else None
        if not published_at:
            # Try JSON-LD
            import json as _json
            for script_tag in soup.find_all("script", type="application/ld+json"):
                try:
                    ld = _json.loads(script_tag.string or "")
                    if isinstance(ld, dict):
                        pub = ld.get("datePublished") or ld.get("dateCreated")
                        if pub:
                            parsed = parse_datetime(pub)
                            published_at = parsed.astimezone(UTC).isoformat() if parsed else None
                            break
                except Exception:
                    continue
        if not published_at:
            time_node = soup.find("time")
            if time_node:
                pub_str = time_node.get("datetime") or clean_text(time_node.get_text(" ", strip=True))
                if pub_str:
                    parsed = parse_datetime(pub_str)
                    published_at = parsed.astimezone(UTC).isoformat() if parsed else None
        if not published_at:
            published_at = _extract_date_from_text(clean_text(soup.get_text(" ", strip=True)[:2000]))

        # Skip stale items if we now have a date
        if published_at and not is_fresh(published_at, target_date, freshness_hours):
            continue

        items.append(
            _build_item(
                source,
                title=title,
                summary=clip_text(summary, 650),
                url=page_url,
                published_at=published_at,
            )
        )
    return items


_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def _render_with_playwright(url: str, wait_ms: int = 5000) -> str:
    """Render a page with headless Chromium and return the full HTML after JS execution."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Warning: playwright not installed; falling back to static fetch")
        return fetch_text(url)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent=_BROWSER_UA)
            page.goto(url, timeout=30000, wait_until="domcontentloaded")
            page.wait_for_timeout(wait_ms)
            html = page.content()
            page.close()
            browser.close()
            return html
    except Exception as ex:
        print(f"Warning: playwright render failed for {url}: {ex}")
        return fetch_text(url)


def _fetch_playwright_source(source: dict[str, Any], target_date: datetime, freshness_hours: int) -> list[HotspotItem]:
    """Fetch from SPA sites using headless Chromium to render JavaScript."""
    page_html = _render_with_playwright(source["url"])
    rows = _extract_generic_blog_rows(page_html, source["url"])

    items: list[HotspotItem] = []
    for row in rows:
        published_at = row.get("published_at")
        if published_at and not is_fresh(published_at, target_date, freshness_hours):
            continue
        # Standard HTML extraction may return old blog posts without dates;
        # only skip if require_date is true (default).
        if not published_at and source.get("require_date", True):
            continue
        items.append(
            _build_item(
                source,
                title=row["title"],
                summary=row.get("summary", ""),
                url=row["url"],
                published_at=published_at,
            )
        )
    return items[:8]


_LLM_EXTRACT_PROMPT = """\
You are extracting product announcements and news from an AI company's website.
The company is: {company_name}
The page URL is: {page_url}

Below is the visible text from the rendered page. Extract ALL product releases, \
model announcements, partnerships, funding events, and other newsworthy items.

Rules:
- Only extract items that represent real product/company NEWS (model launches, API releases, partnerships, funding)
- Do NOT extract: navigation items, login/download prompts, footer text, legal notices, \
app download links, pricing plans, platform descriptions, or UI elements
- Each item must be a specific named product, model, or event — not a generic description
- For each item, provide a concise English title and a 1-2 sentence English summary
- If a link URL is visible in the text context, include it; otherwise use the page URL
- Return ONLY a JSON array. No markdown, no explanation.

Output format:
[{{"title": "...", "summary": "...", "url": "..."}}]

If there are no newsworthy items, return: []

---
PAGE TEXT:
{page_text}
"""


def _extract_with_llm(page_text: str, source: dict[str, Any]) -> list[dict[str, Any]]:
    """Use LLM to extract structured news items from raw page text."""
    import json as _json
    import os

    import requests

    from arxiv_assistant.utils.local_env import load_local_env

    if len(page_text.strip()) < 50:
        return []

    load_local_env()

    prompt = _LLM_EXTRACT_PROMPT.format(
        company_name=source.get("source_name", "Unknown"),
        page_url=source.get("url", ""),
        page_text=page_text[:4000],
    )

    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = source.get("llm_model", "gpt-5.4")

    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "temperature": 0.1, "messages": [{"role": "user", "content": prompt}]},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data["choices"][0]["message"]["content"] or "[]"
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        items = _json.loads(raw)
        return items if isinstance(items, list) else []
    except Exception as ex:
        print(f"Warning: LLM extraction failed for {source.get('source_id')}: {ex}")
        return []


def _fetch_playwright_llm_source(source: dict[str, Any], target_date: datetime, freshness_hours: int) -> list[HotspotItem]:
    """Fetch from SPA sites using Playwright + LLM extraction for non-standard pages."""
    page_html = _render_with_playwright(source["url"])

    # First try standard extraction
    rows = _extract_generic_blog_rows(page_html, source["url"])
    if len(rows) >= 2:
        # Standard extraction worked; use it.
        # Don't assign target_date to dateless items here — generic HTML
        # extraction may return old blog posts mixed with current ones.
        items: list[HotspotItem] = []
        for row in rows:
            published_at = row.get("published_at")
            if published_at and not is_fresh(published_at, target_date, freshness_hours):
                continue
            if not published_at and source.get("require_date", True):
                continue
            items.append(_build_item(source, row["title"], row.get("summary", ""), row["url"], published_at))
        return items[:8]

    # Fallback: LLM extraction from rendered text
    soup = BeautifulSoup(page_html, "html.parser")
    page_text = clean_text(soup.get_text(" ", strip=True))
    llm_items = _extract_with_llm(page_text, source)

    # Post-filter patterns for LLM-extracted noise
    _llm_noise_re = re.compile(
        r"(?:mac|windows|android|ios|iphone|ipad)\s+(?:适用|下载|download|available)",
        re.IGNORECASE,
    )

    items = []
    for item in llm_items:
        title = clean_text(item.get("title", ""))
        if not title or len(title) < 10:
            continue
        if _NAV_CTA_RE.match(title.strip()):
            continue
        if _llm_noise_re.search(title):
            continue
        summary = clean_text(item.get("summary", ""))
        url = item.get("url", source["url"])
        items.append(_build_item(source, title, summary, url, published_at=target_date.isoformat()))
    return items[:8]


def _fetch_html_source(source: dict[str, str], target_date: datetime, freshness_hours: int) -> list[HotspotItem]:
    page_html = fetch_text(source["url"])
    if source["mode"] == "anthropic_html":
        rows = _extract_anthropic_rows(page_html, source["url"])
    elif source["mode"] == "meta_html":
        rows = _extract_meta_rows(page_html, source["url"])
    elif source["mode"] == "generic_html":
        rows = _extract_generic_blog_rows(page_html, source["url"])
    else:
        rows = []

    items: list[HotspotItem] = []
    for row in rows:
        published_at = row.get("published_at")
        if published_at and not is_fresh(published_at, target_date, freshness_hours):
            continue
        # Skip items without dates for HTML sources (can't verify freshness)
        # unless require_date is explicitly set to false in config
        if not published_at and source.get("require_date", True):
            continue
        items.append(
            _build_item(
                source,
                title=row["title"],
                summary=row.get("summary", ""),
                url=row["url"],
                published_at=published_at,
            )
        )
    return items[:8]


def fetch_hotspot_items(target_date: datetime, freshness_hours: int, registry_path: str | None = None) -> list[HotspotItem]:
    sources = _load_official_blog_registry(registry_path)
    items: list[HotspotItem] = []
    for source in sources:
        source_id = source.get("source_id", source.get("name", "unknown"))
        try:
            if source["mode"] == "rss":
                items.extend(_fetch_rss_source(source, target_date, freshness_hours))
            elif source["mode"] == "sitemap":
                items.extend(_fetch_sitemap_source(source, target_date, freshness_hours))
            elif source["mode"] == "playwright":
                items.extend(_fetch_playwright_source(source, target_date, freshness_hours))
            elif source["mode"] == "playwright_llm":
                items.extend(_fetch_playwright_llm_source(source, target_date, freshness_hours))
            else:
                items.extend(_fetch_html_source(source, target_date, freshness_hours))
        except Exception as ex:
            print(f"Warning: failed to fetch official blog source {source_id}: {ex}")
    return items
