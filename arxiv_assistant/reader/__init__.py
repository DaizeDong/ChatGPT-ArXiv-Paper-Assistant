"""Reader model: score new material by whether it would CHANGE a research question.

`questions` loads the human-authored documents under configs/reader/questions/.
`delta` scores candidates against them behind a deterministic verifier.
"""
