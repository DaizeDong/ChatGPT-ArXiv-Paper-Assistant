from __future__ import annotations

from typing import Any

CATEGORY_SUBTITLES = {
    "Research": "Fresh papers with frontier impact",
    "Product Release": "Official launches and model updates",
    "Tooling": "Developer tools, frameworks, and infrastructure",
    "Industry Update": "Policy, partnerships, and ecosystem shifts",
    "Community Signal": "Discussion trends and emerging narratives",
}


def _source_tier_badge(topic: dict[str, Any]) -> str:
    source_roles = set(topic.get("source_roles", []))
    source_count = len(topic.get("source_names", []))
    badges = []
    if "official_news" in source_roles:
        badges.append("[Official]")
    if source_count >= 2:
        badges.append(f"[{source_count} Sources]")
    if "editorial_depth" in source_roles:
        badges.append("[Analysis]")
    if "research_backbone" in source_roles or "paper_trending" in source_roles:
        if "paper" in set(topic.get("source_types", [])):
            badges.append("[Research]")
    if "github_trend" in source_roles:
        badges.append("[GitHub]")
    return " ".join(badges)


def _render_source_list(items: list[dict[str, Any]], limit: int = 4) -> str:
    lines = []
    # Sort: official sources first, then by source_role weight
    role_priority = {"official_news": 0, "research_backbone": 1, "paper_trending": 2, "editorial_depth": 3, "community_heat": 4, "headline_consensus": 5, "github_trend": 6, "builder_momentum": 7, "hn_discussion": 8}
    sorted_items = sorted(items[:limit * 2], key=lambda item: role_priority.get(item.get("source_role", ""), 9))[:limit]
    for item in sorted_items:
        title = item.get("title", "Untitled")
        url = item.get("url", "")
        source_name = item.get("source_name", item.get("source_id", "source"))
        source_role = item.get("source_role", "")
        role_label = ""
        if source_role == "official_news":
            role_label = " [Primary]"
        elif source_role in ("headline_consensus", "community_heat", "hn_discussion"):
            role_label = " [Corroboration]"
        if url:
            lines.append(f"- [{title}]({url}) ({source_name}{role_label})")
        else:
            lines.append(f"- {title} ({source_name}{role_label})")
    return "\n".join(lines)


def _compact_topic_line(topic: dict[str, Any]) -> str:
    headline = topic.get("HEADLINE") or topic.get("title") or "Untitled Topic"
    badge = _source_tier_badge(topic)
    if badge:
        return f"- **{headline}** {badge}"
    source_count = len(topic.get("source_names", []))
    return f"- **{headline}** ({source_count} sources)"


def _radar_excerpt(topic: dict[str, Any], max_length: int = 160) -> str:
    text = (
        topic.get("SHORT_COMMENT", "").strip()
        or topic.get("summary", "").strip()
        or topic.get("WHY_IT_MATTERS", "").strip()
    )
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def _render_paper_spotlight_item(item: dict[str, Any]) -> list[str]:
    title = item.get("title", "Untitled paper")
    url = item.get("url", "")
    arxiv_id = item.get("arxiv_id", "")
    primary_topic_label = item.get("primary_topic_label", "")
    spotlight_comment = (item.get("spotlight_comment") or "").strip()
    summary = (item.get("summary") or "").strip()
    if url:
        lines = [f"- [{title}]({url})"]
    else:
        lines = [f"- {title}"]
    meta_bits = []
    if arxiv_id:
        meta_bits.append(f"arXiv {arxiv_id}")
    if primary_topic_label:
        meta_bits.append(primary_topic_label)
    if meta_bits:
        lines.append(f"  - {' | '.join(meta_bits)}")
    if spotlight_comment:
        lines.append(f"  - {spotlight_comment}")
    elif summary:
        lines.append(f"  - {summary}")
    return lines


def render_hot_daily_md(report: dict[str, Any]) -> str:
    featured_topics = report.get("featured_topics") or report.get("top_topics") or []
    category_sections = report.get("category_sections") or []
    long_tail_sections = report.get("long_tail_sections") or []
    paper_spotlight = report.get("paper_spotlight") or []
    x_buzz = report.get("x_buzz") or []
    watchlist = report.get("watchlist") or []

    source_stats = report.get("source_stats") or {}
    active_sources = sum(1 for count in source_stats.values() if count > 0)
    total_sources = len(source_stats)
    inactive_sources = [name for name, count in sorted(source_stats.items()) if count == 0]

    lines = [
        f'# Daily AI Hotspots {report["date"]}',
        "",
        report.get("summary", "").strip() or "No summary available.",
        "",
        "## Coverage Snapshot",
        "",
        f'- Featured topics: {len(featured_topics)}',
        f'- Category radar topics: {sum(len(section.get("topics", [])) for section in category_sections)}',
        f'- Long-tail signals: {sum(len(section.get("topics", [])) for section in long_tail_sections)}',
        f'- Paper spotlight items: {sum(len(section.get("items", [])) for section in paper_spotlight)}',
        f'- Watchlist topics: {len(watchlist)}',
        f'- Active sources: {active_sources}/{total_sources}',
        f'- Raw items scanned: {report.get("totals", {}).get("raw_items", 0)}',
        "",
    ]

    if inactive_sources:
        lines.append(f'*Inactive sources: {", ".join(inactive_sources)}*')
        lines.append("")

    lines.extend(["## Source Stats", ""])
    for source_name, count in sorted(source_stats.items()):
        if count > 0:
            lines.append(f"- `{source_name}`: {count}")
        else:
            lines.append(f"- `{source_name}`: 0 (inactive)")

    if paper_spotlight:
        lines.extend(["", "## Paper Spotlight", ""])
        for section in paper_spotlight:
            label = section.get("label", "Paper Spotlight")
            description = (section.get("description") or "").strip()
            items = section.get("items", [])
            lines.extend([f"### {label} ({len(items)})", ""])
            if description:
                lines.extend([description, ""])
            for item in items:
                lines.extend(_render_paper_spotlight_item(item))
            lines.append("")

    if featured_topics:
        lines.extend(["", "## Featured Topics", ""])
        for idx, topic in enumerate(featured_topics, start=1):
            headline = topic.get("HEADLINE") or topic.get("title") or f"Topic {idx}"
            badge = _source_tier_badge(topic)
            category = topic.get("PRIMARY_CATEGORY", "Other Frontier AI")
            source_names = ", ".join(topic.get("source_names", [])) or "Unknown"
            header_line = f"### {idx}. {headline}"
            if badge:
                header_line += f"  {badge}"
            lines.extend(
                [
                    header_line,
                    "",
                    f"- Category: {category}",
                    f"- Sources: {source_names}",
                    "",
                    topic.get("WHY_IT_MATTERS", "").strip() or topic.get("SHORT_COMMENT", "").strip(),
                    "",
                ]
            )
            takeaways = [takeaway for takeaway in topic.get("KEY_TAKEAWAYS", []) if takeaway]
            if takeaways:
                lines.append("**Key takeaways:**")
                lines.extend(f"- {takeaway}" for takeaway in takeaways)
                lines.append("")
            lines.append("**Sources:**")
            rendered_sources = _render_source_list(topic.get("items", []))
            if rendered_sources:
                lines.append(rendered_sources)
            else:
                lines.append("- No representative sources available.")
            lines.append("")

    non_empty_category_sections = [section for section in category_sections if section.get("topics")]
    if non_empty_category_sections:
        lines.extend(["## Topic Radar By Category", "", "Broader same-day coverage beyond the featured list.", ""])
        for section in non_empty_category_sections:
            category = section.get("category", "Other")
            displayed = len(section.get("topics", []))
            total_candidates = section.get("total_candidates", displayed)
            subtitle = CATEGORY_SUBTITLES.get(category, "")
            header = f"### {category} ({displayed} shown / {total_candidates} candidates)"
            if subtitle:
                header += f" — {subtitle}"
            lines.extend([header, ""])
            for topic in section.get("topics", []):
                lines.append(_compact_topic_line(topic))
                short_text = _radar_excerpt(topic)
                if short_text:
                    lines.append(f"  - {short_text}")
                evidence = topic.get("items", [])[:1]
                if evidence:
                    evidence_bits = []
                    for item in evidence:
                        title = item.get("title", "Untitled")
                        url = item.get("url", "")
                        if url:
                            evidence_bits.append(f"[{title}]({url})")
                        else:
                            evidence_bits.append(title)
                    lines.append(f"  - Evidence: {' | '.join(evidence_bits)}")
            lines.append("")

    non_empty_long_tail = [section for section in long_tail_sections if section.get("topics")]
    if non_empty_long_tail:
        lines.extend(["## Long-tail Signals", "", "Lower-priority same-day candidates worth tracking.", ""])
        for section in non_empty_long_tail:
            category = section.get("category", "Other")
            displayed = len(section.get("topics", []))
            total_candidates = section.get("total_candidates", displayed)
            lines.extend([f"### {category} ({displayed} shown / {total_candidates} candidates)", ""])
            for topic in section.get("topics", []):
                lines.append(_compact_topic_line(topic))
            lines.append("")

    if x_buzz:
        lines.extend(["## X Buzz", "", "Proxy social signal collected from roundup and community sources.", ""])
        for item in x_buzz:
            title = item.get("title", "Untitled item")
            source = item.get("source_name", item.get("source_id", "source"))
            url = item.get("url", "")
            linked_topic = item.get("linked_topic", "")
            summary = item.get("summary", "").strip()
            if url:
                lines.append(f"- [{title}]({url}) ({source})")
            else:
                lines.append(f"- {title} ({source})")
            if linked_topic:
                lines.append(f"  - Linked topic: {linked_topic}")
            if summary:
                lines.append(f"  - {summary}")
        lines.append("")

    leftover_watchlist = [
        topic
        for topic in watchlist
        if all(topic.get("TOPIC_ID") != candidate.get("TOPIC_ID") for section in category_sections for candidate in section.get("topics", []))
    ]
    if leftover_watchlist:
        lines.extend(["## Watchlist", ""])
        for topic in leftover_watchlist:
            lines.extend(
                [
                    _compact_topic_line(topic),
                    f'  - {topic.get("SHORT_COMMENT", "").strip() or topic.get("summary", "").strip()}',
                ]
            )
        lines.append("")

    return "\n".join(lines).strip() + "\n"
