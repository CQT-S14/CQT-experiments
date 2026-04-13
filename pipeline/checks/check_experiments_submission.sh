#!/bin/bash
# check_experiments_submission.sh
#
# Monitors that benchmarking pipeline part1 worked (or not)
# Checks two conditions:
#   1. Did the most recent Slurm job produce errors? (.err file > threshold)
#   2. Did any job run today at all?
#
# Intended to run via cron at 23:50 daily on cqt-paul-server:
#   50 23 * * * /path/to/check_experiment_submission.sh
#
# Requires: SLACK_NOTIFICATIONS_WEBHOOK set in ~/.env_user (KEY=VALUE format)

ENV_FILE="$(eval echo ~)/.env_user"
if [ ! -f "$ENV_FILE" ]; then
    echo "~/.env_user not found."
    exit 1
fi

# Source ~/.env_user (KEY=VALUE lines, comments and blanks ignored by shell)
set -a
source "$ENV_FILE"
set +a

LOGDIR="$HOME/logs"
ERR_THRESHOLD_BYTES=10

if [ -z "$SLACK_NOTIFICATIONS_WEBHOOK" ]; then
    echo "SLACK_NOTIFICATIONS_WEBHOOK not set in ~/.env_user"
    exit 1
fi

TODAY=$(date +"%Y-%m-%d")
ALERT_MESSAGES=()

# --- Check 1: Did any job run today? ---
TODAY_DISPLAY=$(date +"%b %_d")  # e.g. "Apr 13" to match ls output
LATEST_OUT=$(ls -t "$LOGDIR"/slurm_sinq20_dev_*.out 2>/dev/null | head -1)

if [ -z "$LATEST_OUT" ]; then
    ALERT_MESSAGES+=("The nightly benchmarking experiments did not run today ($TODAY). No Slurm job output was found, which means the cron scheduler may not have fired or the job was never submitted.\n\n_Trace:_ No files matching \`slurm_sinq20_dev_*.out\` in \`$LOGDIR\`.")
else
    LATEST_OUT_DATE=$(date -r "$LATEST_OUT" +"%Y-%m-%d" 2>/dev/null)
    if [ "$LATEST_OUT_DATE" != "$TODAY" ]; then
        ALERT_MESSAGES+=("The nightly benchmarking experiments did not run today ($TODAY). The Slurm job was not submitted or failed before producing output.\n\n_Trace:_ Most recent output is \`$(basename "$LATEST_OUT")\` from $LATEST_OUT_DATE.")
    fi
fi

# --- Check 2: Did the most recent .err file contain errors? ---
LATEST_ERR=$(ls -t "$LOGDIR"/slurm_sinq20_dev_*.err 2>/dev/null | head -1)

if [ -n "$LATEST_ERR" ]; then
    ERR_SIZE=$(stat --format=%s "$LATEST_ERR" 2>/dev/null || stat -f%z "$LATEST_ERR" 2>/dev/null)

    if [ "$ERR_SIZE" -gt "$ERR_THRESHOLD_BYTES" ]; then
        ERR_FILENAME=$(basename "$LATEST_ERR")
        ERR_CONTENT=$(head -20 "$LATEST_ERR")
        ALERT_MESSAGES+=("The nightly benchmarking experiments were submitted but crashed during execution. The Slurm job produced errors, which typically means an experiment script failed (e.g. hardware unreachable, or a code error). No results were uploaded.\n\n_Trace:_ Error file \`$ERR_FILENAME\` ($ERR_SIZE bytes):\n\`\`\`\n${ERR_CONTENT}\n\`\`\`")
    fi
fi

# --- Send notification ---
if [ ${#ALERT_MESSAGES[@]} -eq 0 ]; then
    # All checks passed
    BODY=":white_check_mark: *Benchmarking experiments ran successfully on $TODAY.* Results available."
    BODY+="\n\n_Run at $(date '+%Y-%m-%d %H:%M') on $(hostname)_"

    curl -s -X POST "$SLACK_NOTIFICATIONS_WEBHOOK" \
        -H 'Content-type: application/json' \
        -d "$(printf '{"text": "%s"}' "$BODY")" > /dev/null
    exit 0
fi

BODY=":warning: *Benchmarking pipeline — experiment submission failed*\n\n"
for msg in "${ALERT_MESSAGES[@]}"; do
    BODY+="$msg\n\n"
done
BODY+="_Run at $(date '+%Y-%m-%d %H:%M') on $(hostname)_"

curl -s -X POST "$SLACK_NOTIFICATIONS_WEBHOOK" \
    -H 'Content-type: application/json' \
    -d "$(printf '{"text": "%s"}' "$BODY")" > /dev/null

exit 0
