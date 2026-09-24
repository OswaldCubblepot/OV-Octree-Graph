#!/usr/bin/env bash
# Replica meshes: official release v1.0, 17 parts partaa..partaq (33.7 GB).
# ghfast.top is confirmed range-capable (206); gh-proxy.com returned 403.
set -u
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MESH_DIR="$REPO_ROOT/data/replica"
mkdir -p "$MESH_DIR"
cd "$MESH_DIR"
BASE="https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0"
declare -A SZ
SZ[aa]=2000000000; SZ[ab]=2000000000; SZ[ac]=2000000000; SZ[ad]=2000000000
SZ[ae]=2000000000; SZ[af]=2000000000; SZ[ag]=2000000000; SZ[ah]=2000000000
SZ[ai]=2000000000; SZ[aj]=2000000000; SZ[ak]=2000000000; SZ[al]=2000000000
SZ[am]=2000000000; SZ[an]=2000000000; SZ[ao]=2000000000; SZ[ap]=2000000000
SZ[aq]=1859047808
for p in aa ab ac ad ae af ag ah ai aj ak al am an ao ap aq; do
  part="replica_v1_0.tar.gz.part$p"
  if [ -f "$part" ] && [ "$(stat -c %s "$part" 2>/dev/null)" = "${SZ[$p]}" ]; then
    echo "[skip] $part"; continue
  fi
  echo "[dl] $part ($((${SZ[$p]}/1000000)) MB)"
  aria2c -x 6 -s 6 -k 4M --continue --file-allocation=none \
    --retry-wait 8 --max-tries 50 --timeout 20 -o "$part" \
    "https://ghfast.top/$BASE/$part" \
    || aria2c -x 6 -s 6 -k 4M --continue --file-allocation=none \
    --retry-wait 8 --max-tries 50 --timeout 20 -o "$part" \
    "https://gh.ddlc.top/$BASE/$part"
  sz=$(stat -c %s "$part" 2>/dev/null || echo 0)
  echo "  [got] $part = $sz bytes (want ${SZ[$p]})"
done
echo "=== extract ==="
cat replica_v1_0.tar.gz.part?? | unpigz -p 8 | tar -xC "$MESH_DIR"
echo "=== MESH DONE ==="
