"""Make stdout survive the content this repo actually prints."""
from __future__ import annotations

import sys


def force_utf8_stdio() -> None:
    """Reconfigure stdin/stdout/stderr to UTF-8 with replacement, if possible.

    Guarded on every stream: a stream that has been replaced by something without
    ``reconfigure`` (a pytest capture object, a pipe wrapper) is left alone rather
    than crashing the program that was trying to make printing safer.
    """
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # Already detached, or not a real text stream. Not worth failing over.
            continue
