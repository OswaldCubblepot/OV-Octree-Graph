#!/usr/bin/env bash
# Master download: Replica RGB-D + meshes + OVSeg/TAP weights.
# Run inside WSL:  bash setup/_dl_all.sh
# Logs to setup/_dl_all.log ; safe to re-run (resumes via aria2 -c / wget -c).
set -u
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$REPO_ROOT/setup/_dl_all.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== _dl_all start $(date) ==="

mkdir -p "$REPO_ROOT/data" "$REPO_ROOT/pretrained_weights"
cd "$REPO_ROOT"

# ------------------------------------------------------------------ #
# 1. Replica RGB-D (GS3LAM mirror of nice-slam Replica.zip, 12.8 GB)
# ------------------------------------------------------------------ #
RGBD_ZIP="$REPO_ROOT/data/GS3LAM-Replica.zip"
if [ ! -d "$REPO_ROOT/data/replica/room0/results" ]; then
  if [ ! -f "$RGBD_ZIP" ]; then
    echo "[1/3] RGB-D download (12.8 GB) via hf-mirror"
    aria2c -x 8 -s 8 -k 4M --continue --file-allocation=none \
      -d "$REPO_ROOT/data" -o GS3LAM-Replica.zip \
      "https://hf-mirror.com/datasets/3David14/GS3LAM-Replica/resolve/main/GS3LAM-Replica.zip"
  fi
  echo "[1/3] unzip RGB-D"
  mkdir -p "$REPO_ROOT/data/replica"
  unzip -q -n "$RGBD_ZIP" -d "$REPO_ROOT/data/replica_rgbd_tmp"
  # nice-slam layout: Replica/<scene>/{results,taj.txt}
  if [ -d "$REPO_ROOT/data/replica_rgbd_tmp/Replica" ]; then
    for d in "$REPO_ROOT"/data/replica_rgbd_tmp/Replica/*/; do
      name="$(basename "$d")"
      mkdir -p "$REPO_ROOT/data/replica/$name"
      cp -rn "$d". "$REPO_ROOT/data/replica/$name"/
    done
    rm -rf "$REPO_ROOT/data/replica_rgbd_tmp"
  fi
else
  echo "[1/3] RGB-D already present"
fi

# ------------------------------------------------------------------ #
# 2. Replica meshes (official release v1.0, 17 parts ~34 GB total)
# ------------------------------------------------------------------ #
MESH_DIR="$REPO_ROOT/data/replica"
if [ ! -d "$MESH_DIR/room0/habitat" ] && [ ! -d "$MESH_DIR/room_0/habitat" ]; then
  echo "[2/3] mesh parts download (34 GB) via gh-proxy (range-supported, 7 MB/s)"
  cd "$MESH_DIR"
  BASE="https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0"
  for p in a b c d e f g h i j k l m n o p q; do
    part="replica_v1_0.tar.gz.part$p"
    if [ -f "$part" ] && [ ! -f "$part.aria2" ]; then
      echo "  [skip] $part"; continue
    fi
    # gh-proxy streams the release asset directly (HTTP 200 + ranges)
    aria2c -x 6 -s 6 -k 1M --continue --file-allocation=none -o "$part" \
      "https://gh-proxy.com/$BASE/$part" \
      || aria2c -x 6 -s 6 -k 1M --continue --file-allocation=none -o "$part" \
           "https://ghfast.top/$BASE/$part"
    echo "  [done] $part"
  done
  echo "[2/3] extract meshes"
  cat replica_v1_0.tar.gz.part?? | unpigz -p 8 | tar -xC "$MESH_DIR"
  cd "$REPO_ROOT"
else
  echo "[2/3] meshes already present"
fi

# additional habitat configs (info_semantic.json lives in the release already,
# but fetch the extra configs too for completeness)
if [ ! -f "$MESH_DIR/assets/additional_habitat_configs.zip" ]; then
  echo "[2b] additional_habitat_configs.zip"
  mkdir -p "$MESH_DIR/assets"
  wget -q --continue "http://dl.fbaipublicfiles.com/habitat/Replica/additional_habitat_configs.zip" \
    -P "$MESH_DIR/assets/" || echo "  [WARN] additional configs download failed (non-fatal)"
  unzip -qo -n "$MESH_DIR/assets/additional_habitat_configs.zip" -d "$MESH_DIR" 2>/dev/null || true
fi

# map official layout (room_0) -> pipeline layout (room0), mesh -> {scene}_mesh.ply
cd "$MESH_DIR"
for pair in room_0:room0 room_1:room1 room_2:room2 office_0:office0 office_1:office1 office_2:office2 office_3:office3 office_4:office4; do
  src="${pair%:*}"; dst="${pair#*:}"
  [ -d "$src" ] || { echo "[warn] $src missing"; continue; }
  mkdir -p "$dst"
  [ -d "$dst/habitat" ] || cp -rn "$src/habitat" "$dst/" 2>/dev/null || true
  # geometry for gt_xyz: use the semantic mesh (guaranteed vertex-aligned with
  # the per-face object_id labels) so evaluation is consistent.
  if [ ! -f "$dst/${dst}_mesh.ply" ]; then
    if [ -f "$dst/habitat/mesh_semantic.ply" ]; then
      cp "$dst/habitat/mesh_semantic.ply" "$dst/${dst}_mesh.ply"
      echo "[mapped] $src -> $dst (${dst}_mesh.ply = mesh_semantic.ply)"
    elif [ -f "$src/mesh.ply" ]; then
      cp "$src/mesh.ply" "$dst/${dst}_mesh.ply"
      echo "[mapped] $src -> $dst (${dst}_mesh.ply = mesh.ply)"
    fi
  fi
done

# ------------------------------------------------------------------ #
# 3. Weights: OVSeg + TAP (hf-mirror)
# ------------------------------------------------------------------ #
cd "$REPO_ROOT/pretrained_weights"
echo "[3/3] weights"
[ -f ovseg_swinbase_vitL14_ft_mpt.pth ] || aria2c -x 8 -s 8 -k 4M --continue --file-allocation=none \
  -o ovseg_swinbase_vitL14_ft_mpt.pth \
  "https://hf-mirror.com/camenduru/ovseg/resolve/main/ovseg_swinbase_vitL14_ft_mpt.pth"
[ -f openai_vitl14_from_ovseg.pt ] || aria2c -x 8 -s 8 -k 4M --continue --file-allocation=none \
  -o openai_vitl14_from_ovseg.pt \
  "https://hf-mirror.com/camenduru/ovseg/resolve/main/ovseg_clip_l_9a1909.pth"
[ -f tap_vit_h_v1_1.pkl ] || wget -q --continue \
  "https://hf-mirror.com/BAAI/tokenize-anything/resolve/main/models/tap_vit_h_v1_1.pkl" -O tap_vit_h_v1_1.pkl
[ -f merged_2560.pkl ] || wget -q --continue \
  "https://hf-mirror.com/BAAI/tokenize-anything/resolve/main/concepts/merged_2560.pkl" -O merged_2560.pkl

echo "=== _dl_all done $(date) ==="
ls -lh "$REPO_ROOT/pretrained_weights/"
for d in room0 room1 room2 office0 office1 office2 office3 office4; do
  f=$(ls "$MESH_DIR/$d/results"/frame*.jpg 2>/dev/null | wc -l)
  dp=$(ls "$MESH_DIR/$d/results"/depth*.png 2>/dev/null | wc -l)
  echo "$d: frames=$f depths=$dp mesh=$([ -f "$MESH_DIR/$d/${d}_mesh.ply" ] && echo yes || echo NO) habitat=$([ -d "$MESH_DIR/$d/habitat" ] && echo yes || echo NO) traj=$([ -f "$MESH_DIR/$d/traj.txt" ] && echo yes || echo NO)"
done
