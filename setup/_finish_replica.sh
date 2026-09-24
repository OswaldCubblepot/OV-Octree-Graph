#!/usr/bin/env bash
# Post-download steps of 05_download_replica.sh (the wrapper was killed by the
# host memory watchdog, but the Replica.zip wget survived as an orphan).
# Steps: unzip RGB-D scans, merge nice-slam layout into scene dirs,
# map room_0 -> room0 (mesh + habitat), verify final layout.
set -e
DATA="$HOME/OV-Octree-Graph/data/replica"
cd "$DATA"

echo "[1/4] wait for Replica.zip download to finish..."
while pgrep -f "wget.*Replica.zip" > /dev/null; do sleep 20; done
ls -la Replica.zip

echo "[2/4] unzip RGB-D scans"
if [ ! -d room0/results ]; then
    unzip -q -n Replica.zip
fi

echo "[3/4] merge nice-slam layout (Replica/<scene>/results) into scene folders"
if [ -d Replica ]; then
    for scene in Replica/*/; do
        name="$(basename "$scene")"
        mkdir -p "$name"
        cp -rn "$scene". "$name"/
    done
    rm -rf Replica
fi

echo "[4/4] map room_0/office_0 release layout onto pipeline layout"
for pair in \
    room_0:room0 room_1:room1 room_2:room2 \
    office_0:office0 office_1:office1 office_2:office2 office_3:office3 office_4:office4; do
    src="${pair%:*}"
    dst="${pair#*:}"
    [ -d "$src" ] || { echo "[warn] $src missing"; continue; }
    mkdir -p "$dst"
    [ -d "$dst/habitat" ] || cp -rn "$src/habitat" "$dst/"
    [ -f "$dst/mesh.ply" ] || cp "$src/mesh.ply" "$dst/mesh.ply"
    [ -f "$dst/${dst}_mesh.ply" ] || cp "$src/mesh.ply" "$dst/${dst}_mesh.ply"
    echo "[mapped] $src -> $dst"
done

echo "--- final layout ---"
for d in room0 room1 room2 office0 office4; do
    frames=$(ls "$d/results"/frame*.jpg 2>/dev/null | wc -l)
    depths=$(ls "$d/results"/depth*.png 2>/dev/null | wc -l)
    echo "$d: frames=$frames depths=$depths mesh=$([ -f "$d/${d}_mesh.ply" ] && echo yes || echo NO) habitat=$([ -d "$d/habitat" ] && echo yes || echo NO) traj=$([ -f "$d/traj.txt" ] && echo yes || echo NO)"
done
