#!/usr/bin/env bash
# Usage: scripts/download_fft75.sh "<signed IEEE DataPort URL of 4k_1.tar.gz>"
# Signed links expire: if the transfer stops, copy a fresh link and rerun; wget -c resumes.
source "$(dirname "$0")/_common.sh"

url="${1:?usage: scripts/download_fft75.sh <signed URL of 4k_1.tar.gz>}"
archive=data/raw/4k_1.tar.gz
mkdir -p data/raw
wget -c --tries=20 --retry-connrefused --progress=dot:giga -O "$archive" "$url"
sha256sum "$archive" | tee "$archive.sha256"
tar -xzf "$archive" -C data/raw
ls -la data/raw/4k_1
