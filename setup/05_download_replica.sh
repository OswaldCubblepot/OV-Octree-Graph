#!/usr/bin/env bash
# Download and organize the Replica dataset (public).
# - meshes + semantic annotations: GitHub release v1.0 (17 split parts).
#   GitHub is slow in China, so we use aria2 with multiple connections
#   directly against the release-assets CDN (resolve each part's signed
#   redirect first; re-resolve on expiry and resume).
# - RGB-D scans: ETH Zurich CDN (nice-slam's Replica.zip).
# Run inside WSL/Linux:
#   bash setup/05_download_replica.sh
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$REPO_ROOT/data/replica"
mkdir -p "$DATA"
cd "$DATA"

# ------------------------------------------------------------------ #
# 1. Replica meshes (aria2, multi-connection, 4 parts in parallel)
# ------------------------------------------------------------------ #
printf "parta%c\n" {a..q} | xargs -P 4 -I{} bash "$REPO_ROOT/setup/_replica_part_dl.sh" {} "$DATA"

if [ ! -d room0/habitat ]; then
    echo "[extract] replica_v1_0.tar.gz"
    cat replica_v1_0.tar.gz.part?? | unpigz -p 8 | tar -xvC "$DATA"
fi

# additional habitat configs (contains extra scene metadata)
if [ ! -f assets/additional_habitat_configs.zip ]; then
    echo "[download] additional_habitat_configs.zip"
    mkdir -p assets
    wget -q --show-progress http://dl.fbaipublicfiles.com/habitat/Replica/additional_habitat_configs.zip -P assets/
    unzip -qn assets/additional_habitat_configs.zip -d "$DATA"
fi

# the pipeline reads {scene}_mesh.ply at the scene root; the release ships
# it as mesh.ply under some layouts, so make the link if needed
for scene in room0 room1 room2 office0 office1 office2 office3 office4; do
    if [ ! -f "$scene/${scene}_mesh.ply" ] && [ -f "$scene/mesh.ply" ]; then
        echo "[fix] $scene/mesh.ply -> ${scene}_mesh.ply"
        cp "$scene/mesh.ply" "$scene/${scene}_mesh.ply"
    fi
done

# ------------------------------------------------------------------ #
# 2. RGB-D scans (nice-slam zip from ETH Zurich; fallback: caiyun cloud)
# ------------------------------------------------------------------ #
if [ ! -d room0/results ]; then
    echo "[download] nice-slam Replica.zip (ETH CDN)"
    if ! wget -q --continue --show-progress https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip; then
        echo "[WARN] ETH CDN failed. Download Replica.zip manually from"
        echo "       https://caiyun.139.com/m/i?1A5Ch5C3abNiL (password: v3fY)"
        echo "       and place it at $DATA/Replica.zip, then re-run this script."
        exit 1
    fi
    unzip -q -n Replica.zip
fi

# merge the nice-slam layout (Replica/<scene>/results) into <scene>/results
if [ -d Replica ]; then
    echo "[merge] Replica/* into scene folders"
    for scene in Replica/*/; do
        name="$(basename "$scene")"
        mkdir -p "$name"
        cp -rn "$scene". "$name"/
    done
    rm -rf Replica Replica.zip
fi

echo "REPLICA_DONE"
du -sh "$DATA"
ls -d "$DATA"/*/
