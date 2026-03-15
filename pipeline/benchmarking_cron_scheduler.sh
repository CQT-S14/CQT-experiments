#!/bin/bash
# Cron-driven job scheduler for benchmarking runs.
# Crontab fires this script every minute; the script submits a Slurm job
# only when the current time matches SCHEDULED_TIME.
#
# To change the schedule: edit SCHEDULED_TIME below, commit, and pull on the server.
# No crontab changes needed.

SCHEDULED_TIME="21:03"   # 24h format HH:MM

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="$REPO_ROOT/logs/benchmarking_cron_scheduler.log"

current_time=$(date +"%H:%M")

if [ "$current_time" != "$SCHEDULED_TIME" ]; then
    exit 0
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Scheduled time reached — submitting job" >> "$LOG_FILE"
cd "$REPO_ROOT"
sbatch pipeline/run_sinq20_dev.sh >> "$LOG_FILE" 2>&1
