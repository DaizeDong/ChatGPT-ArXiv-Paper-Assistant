#!/usr/bin/env bash
set -euo pipefail

REPO="/opt/ChatGPT-ArXiv-Paper-Assistant"
cd "$REPO"

TODAY="$(date -u +%F)"
OUT="$REPO/out"

# No-key agent-native flow (when configs/config.ini is the agent-native profile):
#   - Papers: `python main.py` filters via [PAPER_FILTER] mode = agent_only -> claude -p,
#     so NO OpenAI key is needed for the digest.
#   - Hotspots: the Kernel runs with use_agent_scout = true (WebSearch/WebFetch) and
#     use_twitterapi = false, so NO twitterapi.io key is needed either.
#   Both stages run on the Claude subscription. The only secret this unit needs is the
#   git push token. (If you run a key-based profile instead, main.py uses OpenAI as usual.)

# 0. Agent-native paper digest. With the agent-native profile this filters papers through
#    claude -p (no OpenAI). It must NOT abort the whole run if it fails -- the hotspot
#    generation below is independent and should still proceed.
python main.py || echo "paper digest failed; continuing"

# 1. Drive the fixed-DAG Kernel headlessly. Claude Code subagents (DateVerify/Synthesize)
#    are dispatched from inside the Kernel; this single invocation is idempotent and
#    resumes from the last incomplete (date,stage) checkpoint.
claude -p "Run the daily hotspot pipeline for ${TODAY} via the Kernel. \
Execute: python scripts/generate_daily_hotspots.py --output-root out --mode auto --date ${TODAY}. \
Do not improvise stage order; the Kernel owns the topology." \
  --dangerously-skip-permissions \
  || python scripts/generate_daily_hotspots.py --output-root out --mode auto --date "${TODAY}"

# 2. Dump the Store text snapshot (MUST include date_verdicts) for audit + reproducibility.
python - <<'PY'
from pathlib import Path
from arxiv_assistant.hotspots.store import StoryStore
db = Path("out/hot/state/story_store.sqlite")
if db.exists():
    store = StoryStore(db)
    out = store.dump_text_snapshot(Path("out/hot/state/snapshot"))
    print(f"snapshot written: {out}")
else:
    print("no story_store.sqlite yet; skipping snapshot")
PY

# 3. Push the text snapshot (incl. date_verdicts) to the audit branch. The binary SQLite
#    is NOT committed; only the schema-ized text snapshot travels.
AUDIT_BRANCH="${AUDIT_BRANCH:-hotspot-audit}"
if [ -d "out/hot/state/snapshot" ]; then
  git fetch origin "${AUDIT_BRANCH}:${AUDIT_BRANCH}" 2>/dev/null || git branch -f "${AUDIT_BRANCH}"
  git worktree add --force /tmp/hotspot-audit "${AUDIT_BRANCH}" 2>/dev/null || true
  rm -rf /tmp/hotspot-audit/snapshot
  cp -R out/hot/state/snapshot /tmp/hotspot-audit/snapshot
  ( cd /tmp/hotspot-audit
    git add snapshot
    git commit -m "audit: date_verdicts + story snapshot ${TODAY}" || echo "no snapshot changes"
    git push origin "${AUDIT_BRANCH}" )
  git worktree remove --force /tmp/hotspot-audit || true
fi

# 4. Publish generated web_data to the auto_update branch consumed by the Actions Publisher.
git add -A out
git commit -m "data: hotspot run ${TODAY}" || echo "no data changes"
git push origin HEAD:auto_update || echo "push to auto_update failed (will retry next run)"
