"""Make stdout survive the content this repo actually prints.

On a zh-CN Windows console Python defaults stdout to GBK (cp936). The archive is
full of Chinese headlines and the occasional euro sign, so printing a retrieval
hit or a digest line raises ``UnicodeEncodeError`` and takes the whole command
down at the LAST step, after all the work is done. The same bug bit llmcall
(a model answer beginning with an emoji killed the call) and was fixed the same
way there.

Callers used to work around it by exporting ``PYTHONIOENCODING=utf-8``, which
only helps the person who remembers. Scripts call :func:`force_utf8_stdio` in
``main()`` instead, so the fix travels with the code.
"""
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
