#!/usr/bin/env bash
# Copy results to Google Drive. Configure the rclone remote once with `rclone config` (name: gdrive).
source "$(dirname "$0")/_common.sh"

remote="${1:-gdrive:ICECC2027}"
rclone copy results "$remote/results" --exclude "**/last.pt" --progress
