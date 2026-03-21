from html import escape
from pathlib import Path

LINK_COLOR = "#0969da"
TEXT_FONT_SIZE = 16
ARROW_FONT_SIZE = 16
EDGE_BUTTON_WIDTH = 250
CENTER_BUTTON_WIDTH = 180
BUTTON_HEIGHT = 54
EDGE_GUTTER = 28


def build_nav_button_svg(title: str, subtitle: str, arrow: str | None = None, arrow_side: str | None = None) -> str:
    title = escape(title)
    subtitle = escape(subtitle)
    width = EDGE_BUTTON_WIDTH if arrow_side in {"left", "right"} else CENTER_BUTTON_WIDTH
    text_center_x = width / 2

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{BUTTON_HEIGHT}" '
        f'viewBox="0 0 {width} {BUTTON_HEIGHT}" fill="none" role="img" aria-label="{title} {subtitle}">',
        f'<rect width="{width}" height="{BUTTON_HEIGHT}" fill="transparent"/>',
    ]

    if arrow and arrow_side == "left":
        text_center_x = EDGE_GUTTER + (width - EDGE_GUTTER) / 2
        parts.append(
            f'<text x="12" y="20" text-anchor="start" font-family="Arial, sans-serif" '
            f'font-size="{ARROW_FONT_SIZE}" fill="{LINK_COLOR}">{escape(arrow)}</text>'
        )
    elif arrow and arrow_side == "right":
        text_center_x = (width - EDGE_GUTTER) / 2
        parts.append(
            f'<text x="{width - 12}" y="20" text-anchor="end" font-family="Arial, sans-serif" '
            f'font-size="{ARROW_FONT_SIZE}" fill="{LINK_COLOR}">{escape(arrow)}</text>'
        )

    parts.append(
        f'<text x="{text_center_x}" y="20" text-anchor="middle" '
        f'font-family="Arial, sans-serif" font-size="{TEXT_FONT_SIZE}" fill="{LINK_COLOR}">{title}</text>'
    )
    parts.append(
        f'<text x="{text_center_x}" y="41" text-anchor="middle" '
        f'font-family="Arial, sans-serif" font-size="{TEXT_FONT_SIZE}" fill="{LINK_COLOR}">{subtitle}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def write_nav_button_svg(
    target_path: Path,
    title: str,
    subtitle: str,
    arrow: str | None = None,
    arrow_side: str | None = None,
) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        build_nav_button_svg(title, subtitle, arrow=arrow, arrow_side=arrow_side),
        encoding="utf-8",
    )
