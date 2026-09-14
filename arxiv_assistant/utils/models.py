"""The model names this pipeline defaults to, in one place.

WHY THIS MODULE EXISTS. The same two model ids were hardcoded in nine different
modules. Model generations turn over every few months, so nine literals meant
nine chances to leave a retired name behind -- and the pipeline kept naming
`gpt-5.4` in published digests long after it had stopped calling OpenAI at all.
A default that appears once can be moved once.

These are DEFAULTS, not policy. A caller that passes an explicit model wins, and
the llmcall chain resolves its own per-provider models from the fleet config;
these names apply to the direct-agent paths that ask for a model by name.

Prices below are per million tokens, read from the refreshed LiteLLM table on
2026-09-14 (scripts/refresh_model_pricing.py), and are recorded here only as the
reason for the choice -- nothing computes costs from these comments.
"""

#: General workhorse: summarising, enriching, fetching through an agent.
#:
#: claude-sonnet-5 replaced claude-sonnet-4-6 here. It is the current
#: generation AND cheaper: $2.00/$10.00 against $3.00/$15.00. There is no
#: trade-off to weigh, which is the whole reason the old name was worth hunting
#: down in nine places.
DEFAULT_AGENT_MODEL = "claude-sonnet-5"

#: Reserved for judgements worth paying more for, such as date verification,
#: where a wrong answer silently corrupts the archive.
#:
#: claude-opus-5 replaced claude-opus-4-8 at identical cost ($5.00/$25.00).
DEFAULT_DEEP_MODEL = "claude-opus-5"
