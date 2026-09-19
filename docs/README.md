# Documentation map

Start with the [repository README](../README.md). It says what the two pipelines are
and how to start them. Everything longer lives here.

## Current

| File | What it is |
|---|---|
| [SETUP.md](SETUP.md) | Run modes, quickstart, backfilling missed days, the weekly digest, how paper filtering works. |
| [GUIDE_GITHUB_API.md](GUIDE_GITHUB_API.md) | A free OpenAI-compatible endpoint, for the legacy opt-in path. |
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
  `configs/templates/config.template.ini` and `hotspot.template.ini` are the annotated copies to start from; copy both.
- **Prompts**: `prompts/` holds the paper and hotspot prompts. They are the
  behaviour of the filter more than the code is.
- **Reader model**: `configs/reader/questions/README.md` explains the five
  question documents that drive the weekly digest.
- **Hotspot sources**: `configs/hotspot/README.md` covers the source registries.
