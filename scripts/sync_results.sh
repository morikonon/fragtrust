#!/usr/bin/env bash
# Copy results (and the paper assets, once built) to Google Drive. Configure the rclone remote once
# with `rclone config` (name: gdrive).
source "$(dirname "$0")/_common.sh"

remote="${1:-gdrive:ICECC2027}"
rclone copy results "$remote/results" --exclude "**/last.pt" --progress
if [[ -d paper ]]; then
    rclone copy paper "$remote/paper" --progress
fi
