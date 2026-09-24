#!/usr/bin/env bash
# Map the official Replica release layout to what the pipeline expects.
# Release:  <data_root>/room_0/{mesh.ply, habitat/}     (underscore)
# Pipeline: <data_root>/room0/{room0_mesh.ply, habitat/}
# Run from the repo root AFTER 05_download_replica.sh finishes.
set -e
DATA="$HOME/OV-Octree-Graph/data/replica"
cd "$DATA"

for pair in \
    room_0:room0 room_1:room1 room_2:room2 \
    office_0:office0 office_1:office1 office_2:office2 office_3:office3 office_4:office4; do
    src="${pair%:*}"
    dst="${pair#*:}"
    [ -d "$src" ] || { echo "[warn] $src missing"; continue; }
    mkdir -p "$dst"
    [ -d "$dst/habitat" ] || cp -rn "$src/habitat" "$dst/"
    [ -f "$dst/mesh.ply" ] || cp "$src/mesh.ply" "$dst/mesh.ply"
    if [ ! -f "$dst/${dst}_mesh.ply" ]; then
        cp "$src/mesh.ply" "$dst/${dst}_mesh.ply"
        echo "[mapped] $src -> $dst (${dst}_mesh.ply)"
    else
        echo "[skip] $dst already mapped"
    fi
done

echo "--- final layout ---"
for d in room0 room1 room2 office0 office4; do
    echo "$d: $(ls "$d" 2>/dev/null | tr '\n' ' ')"
    echo "   results frames: $(ls "$d/results" 2>/dev/null | wc -l)"
done
