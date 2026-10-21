#!/usr/bin/env bash
# Periodic git commit + push so work survives SSH drops on the GPU server.
#
# Usage (script only — start it yourself when ready):
#   chmod +x run_scripts/auto_git_push.sh
#
#   # one-shot
#   bash run_scripts/auto_git_push.sh once
#
#   # every 2 hours in background (valid through 2026-11-10)
#   mkdir -p logs
#   nohup bash run_scripts/auto_git_push.sh loop 7200 \
#     >> logs/auto_git_push.log 2>&1 &
#
#   # or under tmux (survives SSH better than bare nohup):
#   tmux new -s gitpush
#   bash run_scripts/auto_git_push.sh loop
#
# Env overrides:
#   AUTO_GIT_INTERVAL_SEC   default 7200 = 2h (loop only)
#   AUTO_GIT_EXPIRES         default 2026-11-10 (inclusive; YYYY-MM-DD)
#   AUTO_GIT_REMOTE          default origin
#   AUTO_GIT_BRANCH          default current branch
#   AUTO_GIT_PATHS           space-separated paths to add
#                           default: exp_oct_2026 analysis run_scripts
#   AUTO_GIT_COMMIT_MSG      default includes hostname + timestamp
#   AUTO_GIT_DRY_RUN=1       print actions only, no commit/push
#   AUTO_GIT_PUSH_ONLY=1     never commit; only push if ahead of remote
#
# Safety: never --force, never amend, never rewrite history.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SCRIPT="$ROOT/run_scripts/auto_git_push.sh"

MODE="${1:-once}"          # once | loop
INTERVAL="${2:-${AUTO_GIT_INTERVAL_SEC:-7200}}"
REMOTE="${AUTO_GIT_REMOTE:-origin}"
BRANCH="${AUTO_GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"
DRY_RUN="${AUTO_GIT_DRY_RUN:-0}"
PUSH_ONLY="${AUTO_GIT_PUSH_ONLY:-0}"
# Inclusive calendar day; script refuses to run after this date.
EXPIRES="${AUTO_GIT_EXPIRES:-2026-11-10}"
# shellcheck disable=SC2206
PATHS=(${AUTO_GIT_PATHS:-exp_oct_2026 analysis run_scripts})
LOCK_FILE="${ROOT}/.auto_git_push.lock"

LOG_TS() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(LOG_TS)] $*"; }

check_expiry() {
  local today
  today="$(date '+%Y-%m-%d')"
  if [[ "$today" > "$EXPIRES" ]]; then
    log "EXPIRED: today=$today > expires=$EXPIRES; auto_git_push disabled"
    exit 0
  fi
}

need_git() {
  command -v git >/dev/null || { log "ERROR: git not found"; exit 1; }
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || { log "ERROR: not a git repo: $ROOT"; exit 1; }
}

has_changes() {
  local p
  for p in "${PATHS[@]}"; do
    [[ -e "$p" ]] || continue
    if ! git diff --quiet -- "$p" 2>/dev/null; then return 0; fi
    if ! git diff --cached --quiet -- "$p" 2>/dev/null; then return 0; fi
    if [[ -n "$(git ls-files --others --exclude-standard -- "$p")" ]]; then return 0; fi
  done
  return 1
}

ahead_of_remote() {
  git fetch --quiet "$REMOTE" "$BRANCH" 2>/dev/null || true
  local local_sha remote_sha
  local_sha="$(git rev-parse HEAD)"
  if ! remote_sha="$(git rev-parse "$REMOTE/$BRANCH" 2>/dev/null)"; then
    log "WARN: $REMOTE/$BRANCH missing; will try push -u"
    return 0
  fi
  [[ "$local_sha" != "$remote_sha" ]] \
    && git merge-base --is-ancestor "$remote_sha" "$local_sha"
}

do_commit() {
  local msg="${AUTO_GIT_COMMIT_MSG:-auto: checkpoint $(hostname -s) $(LOG_TS)}"
  log "git add: ${PATHS[*]}"
  if [[ "$DRY_RUN" == "1" ]]; then
    git status --short -- "${PATHS[@]}" || true
    return 0
  fi
  git add -- "${PATHS[@]}"
  if git diff --cached --quiet; then
    log "nothing staged; skip commit"
    return 0
  fi
  git commit -m "$msg"
  log "committed: $msg"
}

do_push() {
  log "git push $REMOTE $BRANCH"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY_RUN: skip push"
    return 0
  fi
  if git rev-parse --abbrev-ref --symbolic-full-name '@{u}' >/dev/null 2>&1; then
    git push "$REMOTE" "$BRANCH"
  else
    git push -u "$REMOTE" "$BRANCH"
  fi
  log "push ok"
}

run_once() {
  check_expiry
  need_git
  log "repo=$ROOT branch=$BRANCH remote=$REMOTE push_only=$PUSH_ONLY dry_run=$DRY_RUN expires=$EXPIRES"

  if [[ "$PUSH_ONLY" != "1" ]]; then
    if has_changes; then
      do_commit
    else
      log "no local changes under PATHS"
    fi
  fi

  if ahead_of_remote; then
    do_push || log "WARN: push failed (network/auth?); will retry next cycle"
  else
    log "already up to date with $REMOTE/$BRANCH"
  fi
}

run_loop() {
  check_expiry
  log "starting loop every ${INTERVAL}s (~$(( INTERVAL / 3600 ))h), valid through $EXPIRES (Ctrl-C to stop)"
  while true; do
    check_expiry
    if command -v flock >/dev/null 2>&1; then
      flock -n "$LOCK_FILE" "$SCRIPT" once \
        || log "skip: another auto_git_push holds the lock"
    else
      run_once || log "WARN: run_once failed (exit $?); will retry later"
    fi
    sleep "$INTERVAL"
  done
}

case "$MODE" in
  once) run_once ;;
  loop) run_loop ;;
  *)
    echo "Usage: $0 once|loop [interval_sec]"
    exit 2
    ;;
esac
