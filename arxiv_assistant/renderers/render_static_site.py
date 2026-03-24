import argparse
import posixpath
import re
import shutil
from urllib.parse import quote
from pathlib import Path

import markdown

SITE_DESCRIPTION = (
    "LLM-based personalized arXiv paper assistant bot for automatic paper filtering. "
    "Powerful, free, and easy-to-use."
)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" data-color-mode="light">
<head>
  <meta charset="utf-8">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="{description}">
  <meta name="color-scheme" content="light dark">
  <title>{title}</title>
  <link rel="icon" type="image/svg+xml" href="{favicon_href}">
  <link rel="stylesheet" href="{css_href}">
</head>
<body>
  <button class="theme-toggle" type="button" aria-label="Toggle dark mode" title="Toggle dark mode">
    <span class="theme-toggle-track"></span>
    <span class="theme-toggle-thumb">&#9680;</span>
  </button>
  <main class="site-shell">
    <article class="markdown-body">
{content}
    </article>
  </main>
  <script>
    (function() {{
      var MODE_KEY = 'arxiv_site_color_mode';
      var root = document.documentElement;
      var toggle = document.querySelector('.theme-toggle');
      function applyMode(mode) {{
        root.setAttribute('data-color-mode', mode);
        localStorage.setItem(MODE_KEY, mode);
      }}
      function nextMode() {{
        return root.getAttribute('data-color-mode') === 'dark' ? 'light' : 'dark';
      }}
      var savedMode = localStorage.getItem(MODE_KEY);
      if (!savedMode) {{
        savedMode = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
      }}
      applyMode(savedMode);
      toggle.addEventListener('click', function() {{
        applyMode(nextMode());
      }});
    }})();
  </script>
</body>
</html>
"""

TITLE_PATTERN = re.compile(r"^#\s+(.+)$", re.MULTILINE)
USER_CONTENT_PREFIX_PATTERN = re.compile(r'((?:href|id)=")#user-content-')


def _build_favicon_href() -> str:
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
        "<rect width='64' height='64' rx='12' fill='%230969da'/>"
        "<text x='32' y='42' text-anchor='middle' font-family='Arial, sans-serif' "
        "font-size='34' fill='white'>A</text>"
        "</svg>"
    )
    return f"data:image/svg+xml,{quote(svg)}"


def _extract_title(markdown_text: str, fallback: str) -> str:
    match = TITLE_PATTERN.search(markdown_text)
    if match is None:
        return fallback
    return match.group(1).strip()


def _normalize_html(rendered_html: str) -> str:
    return USER_CONTENT_PREFIX_PATTERN.sub(r"\1#", rendered_html)


def _indent_html(rendered_html: str, indent: str = "      ") -> str:
    return "\n".join(f"{indent}{line}" if line else "" for line in rendered_html.splitlines())


def _render_markdown(markdown_text: str) -> str:
    md = markdown.Markdown(
        extensions=[
            "extra",
            "sane_lists",
            "toc",
            "md_in_html",
        ]
    )
    return _normalize_html(md.convert(markdown_text))


def _build_page_html(markdown_text: str, title: str, css_href: str) -> str:
    return HTML_TEMPLATE.format(
        title=title,
        description=SITE_DESCRIPTION,
        favicon_href=_build_favicon_href(),
        css_href=css_href,
        content=_indent_html(_render_markdown(markdown_text)),
    )


def render_static_site(site_root: str | Path, dist_root: str | Path, css_source: str | Path) -> Path:
    site_root = Path(site_root)
    dist_root = Path(dist_root)
    css_source = Path(css_source)

    if not site_root.exists():
        raise FileNotFoundError(f"Site root does not exist: {site_root}")

    if dist_root.exists():
        shutil.rmtree(dist_root)
    dist_root.mkdir(parents=True, exist_ok=True)

    shutil.copy2(css_source, dist_root / "site.css")

    for source_path in sorted(site_root.rglob("*")):
        if source_path.is_dir():
            continue

        relative_path = source_path.relative_to(site_root)
        if source_path.suffix.lower() == ".md":
            output_relative_path = relative_path.with_suffix(".html")
            output_path = dist_root / output_relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            markdown_text = source_path.read_text(encoding="utf-8")
            fallback_title = posixpath.join("ChatGPT-ArXiv-Paper-Assistant", output_relative_path.as_posix())
            title = _extract_title(markdown_text, fallback_title)
            css_href = posixpath.relpath("site.css", output_relative_path.parent.as_posix() or ".")

            output_path.write_text(
                _build_page_html(markdown_text, title, css_href),
                encoding="utf-8",
            )
            continue

        output_path = dist_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, output_path)

    return dist_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-root", default="out/site")
    parser.add_argument("--dist-root", default="dist")
    parser.add_argument("--css-source", default="site.css")
    args = parser.parse_args()

    dist_path = render_static_site(args.site_root, args.dist_root, args.css_source)
    print(dist_path)


if __name__ == "__main__":
    main()
