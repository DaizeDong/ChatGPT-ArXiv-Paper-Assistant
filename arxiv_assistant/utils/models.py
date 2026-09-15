# Default model ids for the direct-agent paths, in one place so a generation
# turnover is one edit. The llmcall chain resolves its own models and ignores these.

# Workhorse: summarising, enriching, agent fetches.
DEFAULT_AGENT_MODEL = "claude-sonnet-5"

# Judgements worth paying more for, such as date verification.
DEFAULT_DEEP_MODEL = "claude-opus-5"
