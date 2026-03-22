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
