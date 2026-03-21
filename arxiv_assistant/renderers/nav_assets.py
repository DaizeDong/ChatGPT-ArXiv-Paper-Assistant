from html import escape
from pathlib import Path

LINK_COLOR = "#0969da"
TITLE_FONT_SIZE = 17
SUBTITLE_FONT_SIZE = 15
BUTTON_WIDTH = 220
BUTTON_HEIGHT = 54


def build_nav_button_svg(title: str, subtitle: str) -> str:
    title = escape(title)
    subtitle = escape(subtitle)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{BUTTON_WIDTH}" height="{BUTTON_HEIGHT}" '
        f'viewBox="0 0 {BUTTON_WIDTH} {BUTTON_HEIGHT}" fill="none" role="img" aria-label="{title} {subtitle}">'
        f'<rect width="{BUTTON_WIDTH}" height="{BUTTON_HEIGHT}" fill="transparent"/>'
        f'<text x="{BUTTON_WIDTH / 2}" y="20" text-anchor="middle" '
        f'font-family="Arial, sans-serif" font-size="{TITLE_FONT_SIZE}" fill="{LINK_COLOR}">{title}</text>'
        f'<text x="{BUTTON_WIDTH / 2}" y="41" text-anchor="middle" '
        f'font-family="Arial, sans-serif" font-size="{SUBTITLE_FONT_SIZE}" fill="{LINK_COLOR}">{subtitle}</text>'
        "</svg>"
    )


def write_nav_button_svg(target_path: Path, title: str, subtitle: str) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(build_nav_button_svg(title, subtitle), encoding="utf-8")
