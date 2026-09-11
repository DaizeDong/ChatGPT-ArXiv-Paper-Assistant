"""Weekly delta digest renderer.

Deliberately separate from ``renderers/paper`` and ``renderers/hotspot``: the
daily pages are an archive (source-first tables, category expansions, every item
of the day) and their markdown is byte-compared by tests. The weekly digest is a
*decision* surface -- a short ranked list of the things that moved a question
document -- so it shares no code path with them and cannot regress them.
"""
