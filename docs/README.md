# Documentation map

Start with the [repository README](../README.md). It covers what the pipelines do,
how to run them, and how to configure them. The files here are the longer pieces
that do not belong in it.

## Current

| File | What it is |
|---|---|
| [DAILY_AI_HOTSPOTS.md](DAILY_AI_HOTSPOTS.md) | How the hotspot digest picks and ranks stories: sources, scoring, dedup, the verifier gates. |
| [UPGRADE-agent-native-hotspot.md](UPGRADE-agent-native-hotspot.md) | Migrating an older checkout to the agent-native hotspot pipeline. |

## History

Everything under [history/](history/) records a decision that was made, not how
the code works today. It is kept because the reasoning is often the only place a
measurement survives, and deleted documents take their evidence with them. Read
it as dated: none of it is maintained.

- `history/plans-2026-06/` -- the staged plan and design specs for the June 2026
  agent-native rewrite.
- `history/investigation_*.md` -- one-off investigations into staleness,
  cross-day dedup and X coverage.
- `history/competitive_landscape_2026-06.md` -- how comparable feeds looked at
  that time.

## Where the rest of the explanation lives

- **Config**: `configs/config.ini` is commented section by section, and
  `configs/templates/config.template.ini` is the annotated copy to start from.
- **Prompts**: `prompts/` holds the paper and hotspot prompts. They are the
  behaviour of the filter more than the code is.
- **Reader model**: `configs/reader/questions/README.md` explains the five
  question documents that drive the weekly digest.
- **Hotspot sources**: `configs/hotspot/README.md` covers the source registries.
