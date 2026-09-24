#!/usr/bin/env bash
# Helper for 05_download_replica.sh: download ONE split part of the Replica
# release with aria2 (multi-connection), resolving the signed CDN redirect.
# Args: <part_name> <data_dir>
set -e

BASE="https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0"
part="replica_v1_0.tar.gz.$1"   # $1 is the part suffix, e.g. "partaa"
data_dir="$2"
cd "$data_dir" || exit 1

# a leftover $part.aria2 control file marks an incomplete download;
# aria2c --continue resumes it automatically
if [ -f "$part" ] && [ ! -f "$part.aria2" ]; then
    echo "[skip] $part already downloaded"
    exit 0
fi

# github.com itself is often unreachable from this network; resolve the
# signed CDN URL through the ghfast / gh-proxy mirrors first, direct last.
resolve_url() {
    local url
    for candidate in \
        "https://ghfast.top/$BASE/$part" \
        "https://gh-proxy.com/$BASE/$part" \
        "$BASE/$part"; do
        url="$(curl -sI --max-time 15 "$candidate" 2>/dev/null | grep -i '^location:' | tr -d '\r' | awk '{print $2}')"
        if [ -n "$url" ]; then
            echo "$url"
            return 0
        fi
    done
    return 1
}

echo "[download] $part"
url="$(resolve_url)"
if [ -z "$url" ]; then
    echo "[FAIL] cannot resolve $part"
    exit 1
fi
# 6 connections per part; re-resolve and retry if the signed link expired
aria2c -q -x 6 -s 6 -k 1M --continue --max-tries 3 --retry-wait 5 -o "$part" "$url" || {
    echo "[retry] re-resolving $part"
    url="$(resolve_url)"
    aria2c -q -x 6 -s 6 -k 1M --continue --max-tries 3 --retry-wait 5 -o "$part" "$url"
}
echo "[done] $part"
