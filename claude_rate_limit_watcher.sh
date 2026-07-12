#!/usr/bin/env bash
#
# claude_rate_limit_watcher.sh
#
# Watches a tmux pane running Claude Code. When a rate-limit prompt is
# detected, sleeps until the rolling window resets, then sends option "1"
# (Continue) to resume.
#
# USAGE:
#   chmod +x claude_rate_limit_watcher.sh
#   ./claude_rate_limit_watcher.sh <tmux-session-name> [wait-seconds]
#
# Run this in its OWN tmux window so it survives independently of the
# Claude Code session it is watching.
#
# VERIFY before relying on this: let Claude Code hit a rate limit, then run
#   tmux capture-pane -t "<session-name>" -p
# and confirm the exact prompt text matches RATE_LIMIT_PATTERN and that
# option "1" selects Continue.

set -euo pipefail

SESSION="${1:-}"
WAIT_SECONDS="${2:-3600}"    # 1 hour — clears any rolling rate-limit window
POLL_INTERVAL=30             # seconds between pane checks
LOG_DIR="$HOME/claude-rate-limit-watcher"
LOG_FILE="$LOG_DIR/watcher.log"

if [ -z "$SESSION" ]; then
  echo "Usage: $0 <tmux-session-name> [wait-seconds]"
  exit 1
fi

mkdir -p "$LOG_DIR"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Matches Claude Code's rate-limit prompt variations.
# Adjust if the real wording differs from what you see in the pane capture.
RATE_LIMIT_PATTERN="limit reached|resets|rate limit|stop and wait"

log "Watcher active for tmux session '$SESSION' (poll every ${POLL_INTERVAL}s, wait ${WAIT_SECONDS}s on hit)"

while true; do
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    log "Session '$SESSION' not found. Exiting."
    exit 1
  fi

  PANE_CONTENT=$(tmux capture-pane -t "$SESSION" -p -S -50)

  if echo "$PANE_CONTENT" | grep -qiE "$RATE_LIMIT_PATTERN"; then
    log "Rate-limit prompt detected. Sleeping for ${WAIT_SECONDS}s (~$((WAIT_SECONDS / 3600))h)."

    sleep "$WAIT_SECONDS"

    log "Wait complete. Sending resume keystroke (option 1 = Continue)."
    tmux send-keys -t "$SESSION" "1" Enter

    log "Keystroke sent. Resuming watch loop."

    # Brief pause so the screen refreshes before the next poll catches the
    # same text and fires again.
    sleep 10
  fi

  sleep "$POLL_INTERVAL"
done
