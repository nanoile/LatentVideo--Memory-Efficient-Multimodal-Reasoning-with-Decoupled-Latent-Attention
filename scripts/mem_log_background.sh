#!/usr/bin/env bash
# Usage: bash scripts/mem_log_background.sh [mem_log.txt]
# Logs `free -m` every second until you Ctrl+C (run in a second terminal while training).
LOG="${1:-mem_log.txt}"
echo "Appending to $LOG — Ctrl+C to stop"
while true; do
  date -Iseconds >> "$LOG"
  free -m >> "$LOG"
  echo "" >> "$LOG"
  sleep 1
done
