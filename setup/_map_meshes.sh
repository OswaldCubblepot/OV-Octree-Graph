#!/usr/bin/env bash
# Map official Replica layout (room_0, office_0, ...) to the pipeline layout
# (room0, office0, ...) and create {scene}_mesh.ply. Run AFTER mesh extraction.
# Safe to re-run (uses cp -f to overwrite).
#
# Key correctness point: scene_replica.py builds gt_xyz from {scene}_mesh.ply and
# semantic_gt/instance_gt (per-VERTEX, propagated from per-face object_id) from
# habitat/mesh_semantic.ply. Those two must share vertex order, so BOTH are taken
# from mesh_semantic.ply.
set -u
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MESH_DIR="$REPO_ROOT/data/replica"
cd "$MESH_DIR" || exit 1

for pair in room_0:room0 room_1:room1 room_2:room2 \
            office_0:office0 office_1:office1 office_2:office2 office_3:office3 office_4:office4; do
  src="${pair%:*}"; dst="${pair#*:}"
  [ -d "$src" ] || { echo "[warn] $src missing"; continue; }
  mkdir -p "$dst/habitat"
  # semantic mesh + semantic class map (per-scene)
  cp -f "$src/habitat/mesh_semantic.ply" "$dst/habitat/mesh_semantic.ply" 2>/dev/null \
    && echo "[cp] $dst/habitat/mesh_semantic.ply"
  cp -f "$src/habitat/info_semantic.json" "$dst/habitat/info_semantic.json" 2>/dev/null \
    && echo "[cp] $dst/habitat/info_semantic.json"
  # {scene}_mesh.ply = semantic mesh (vertex-order aligned with the labels)
  cp -f "$src/habitat/mesh_semantic.ply" "$dst/${dst}_mesh.ply" \
    && echo "[cp] $dst/${dst}_mesh.ply (from mesh_semantic.ply)"
done

echo "=== VERIFY ==="
for d in room0 room1 room2 office0 office1 office2 office3 office4; do
  fr=$(ls "$d/results"/frame*.jpg 2>/dev/null | wc -l)
  dp=$(ls "$d/results"/depth*.png 2>/dev/null | wc -l)
  mesh=$([ -f "$d/${d}_mesh.ply" ] && echo mesh || echo NOmesh)
  sem=$([ -f "$d/habitat/mesh_semantic.ply" ] && echo sem || echo NOsem)
  info=$([ -f "$d/habitat/info_semantic.json" ] && echo info || echo NOinfo)
  tj=$([ -f "$d/traj.txt" ] && echo traj || echo NOtraj)
  echo "$d: frames=$fr depths=$dp $mesh $sem $info $tj"
done
