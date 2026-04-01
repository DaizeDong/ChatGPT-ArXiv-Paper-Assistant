from __future__ import annotations

from typing import Any


def _render_source_list(items: list[dict[str, Any]], limit: int = 4) -> str:
    lines = []
    for item in items[:limit]:
        title = item.get("title", "Untitled")
        url = item.get("url", "")
        source_name = item.get("source_name", item.get("source_id", "source"))
        if url:
            lines.append(f"- [{title}]({url}) ({source_name})")
        else:
            lines.append(f"- {title} ({source_name})")
    return "\n".join(lines)


def _compact_topic_line(topic: dict[str, Any]) -> str:
    headline = topic.get("HEADLINE") or topic.get("title") or "Untitled Topic"
    final_score = topic.get("FINAL_SCORE", 0)
    heat = topic.get("HEAT", 0)
    occurrence = topic.get("OCCURRENCE_SCORE", len(topic.get("source_names", [])) or len(topic.get("items", [])))
    source_count = len(topic.get("source_names", []))
    status = topic.get("LLM_STATUS", "")
    suffix = f" | final={final_score} | heat={heat} | occurrence={occurrence} | sources={source_count}"
    if status == "watchlist":
        suffix += " | watchlist"
    elif status == "featured":
        suffix += " | featured"
    return f"- **{headline}**{suffix}"


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
    daily_score = item.get("daily_score", 0)
    relevance = item.get("relevance", 0)
    novelty = item.get("novelty", 0)
    if url:
        lines = [f"- [{title}]({url})"]
    else:
        lines = [f"- {title}"]
    meta_bits = []
    if arxiv_id:
        meta_bits.append(f"arXiv {arxiv_id}")
    if primary_topic_label:
        meta_bits.append(primary_topic_label)
    meta_bits.append(f"score={daily_score}")
    meta_bits.append(f"rel={relevance}")
    meta_bits.append(f"nov={novelty}")
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
        f'- X Buzz items: {len(x_buzz)}',
        f'- Watchlist topics: {len(watchlist)}',
        f'- Raw items scanned: {report.get("totals", {}).get("raw_items", 0)}',
        f'- Clusters formed: {report.get("totals", {}).get("clusters", 0)}',
        f'- Radar clusters considered: {report.get("totals", {}).get("radar_clusters", 0)}',
        "",
        "## Source Stats",
        "",
    ]

    for source_name, count in sorted((report.get("source_stats") or {}).items()):
        lines.append(f"- `{source_name}`: {count}")

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

    lines.extend(["", "## Featured Topics", ""])
    if not featured_topics:
        lines.append("- No topic cleared the featured threshold.")
    else:
        for idx, topic in enumerate(featured_topics, start=1):
            headline = topic.get("HEADLINE") or topic.get("title") or f"Topic {idx}"
            lines.extend(
                [
                    f"### {idx}. {headline}",
                    "",
                    f'- Category: {topic.get("PRIMARY_CATEGORY", "Other Frontier AI")}',
                    f'- Scores: final={topic.get("FINAL_SCORE", 0)} quality={topic.get("QUALITY", 0)} heat={topic.get("HEAT", 0)} importance={topic.get("IMPORTANCE", 0)}',
                    f'- Sources: {", ".join(topic.get("source_names", [])) or "Unknown"}',
                    "",
                    topic.get("WHY_IT_MATTERS", "").strip() or topic.get("SHORT_COMMENT", "").strip(),
                    "",
                ]
            )
            takeaways = [takeaway for takeaway in topic.get("KEY_TAKEAWAYS", []) if takeaway]
            if takeaways:
                lines.append("Key takeaways:")
                lines.extend(f"- {takeaway}" for takeaway in takeaways)
                lines.append("")
            lines.append("Representative sources:")
            rendered_sources = _render_source_list(topic.get("items", []))
            if rendered_sources:
                lines.append(rendered_sources)
            else:
                lines.append("- No representative sources available.")
            lines.append("")

    if category_sections:
        lines.extend(["## Topic Radar By Category", "", "Broader same-day coverage beyond the featured list. Entries stay intentionally short so the page can cover more of the day's signal surface.", ""])
        for section in category_sections:
            category = section.get("category", "Other")
            displayed = len(section.get("topics", []))
            total_candidates = section.get("total_candidates", displayed)
            lines.extend([f"### {category} ({displayed} shown / {total_candidates} candidates)", ""])
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

    if long_tail_sections:
        lines.extend(["## Long-tail Signals", "", "Lower-priority but still relevant same-day candidates, compressed to one line each for breadth.", ""])
        for section in long_tail_sections:
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
