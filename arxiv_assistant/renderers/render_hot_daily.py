from __future__ import annotations

from typing import Any


def _render_source_list(items: list[dict[str, Any]], limit: int = 4) -> str:
    lines = []
    for item in items[:limit]:
        lines.append(f'- [{item["title"]}]({item["url"]}) ({item["source_name"]})')
    return "\n".join(lines)


def render_hot_daily_md(report: dict[str, Any]) -> str:
    lines = [
        f'# Daily AI Hotspots {report["date"]}',
        "",
        report.get("summary", "").strip() or "No summary available.",
        "",
        f'- Top topics: {len(report.get("top_topics", []))}',
        f'- Watchlist: {len(report.get("watchlist", []))}',
        f'- Raw items scanned: {report.get("totals", {}).get("raw_items", 0)}',
        f'- Clusters formed: {report.get("totals", {}).get("clusters", 0)}',
        "",
    ]

    for idx, topic in enumerate(report.get("top_topics", []), start=1):
        headline = topic.get("HEADLINE") or topic.get("title") or f"Topic {idx}"
        lines.extend(
            [
                f"## {idx}. {headline}",
                "",
                f'- Category: {topic.get("PRIMARY_CATEGORY", "Other Frontier AI")}',
                f'- Score: {topic.get("FINAL_SCORE", 0)}',
                f'- Source roles: {", ".join(topic.get("source_roles", []))}',
                "",
                topic.get("WHY_IT_MATTERS", "").strip() or topic.get("SHORT_COMMENT", "").strip(),
                "",
                "Representative sources:",
                _render_source_list(topic.get("items", [])),
                "",
            ]
        )
        takeaways = topic.get("KEY_TAKEAWAYS", [])
        if takeaways:
            lines.append("Key takeaways:")
            lines.extend(f"- {takeaway}" for takeaway in takeaways if takeaway)
            lines.append("")

    x_buzz = report.get("x_buzz") or []
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

    if report.get("watchlist"):
        lines.extend(["## Watchlist", ""])
        for topic in report["watchlist"]:
            lines.extend(
                [
                    f'- {topic.get("title", "Untitled topic")} ({topic.get("PRIMARY_CATEGORY", "Other Frontier AI")}, score {topic.get("FINAL_SCORE", 0)})',
                    f'  - {topic.get("SHORT_COMMENT", "").strip() or topic.get("summary", "").strip()}',
                ]
            )
        lines.append("")

    return "\n".join(lines).strip() + "\n"
