#!/usr/bin/env bash
# Fast torch cu118 install via SJTU mirror (torch wheels) + tuna PyPI (deps).
set -u
export PATH=$HOME/miniconda3/bin:$PATH
PIP=$HOME/miniconda3/envs/ovgraph/bin/pip
W=$HOME/wheels
mkdir -p "$W"
cd "$W"

# Wheel URLs (SJTU pytorch-wheels mirror; fallback aliyun, then official)
TORCH="https://mirror.sjtu.edu.cn/pytorch-wheels/cu118/torch-2.1.2%2Bcu118-cp310-cp310-linux_x86_64.whl"
TORCHV="https://mirror.sjtu.edu.cn/pytorch-wheels/cu118/torchvision-0.16.2%2Bcu118-cp310-cp310-linux_x86_64.whl"
TORCHA="https://mirror.sjtu.edu.cn/pytorch-wheels/cu118/torchaudio-2.1.2%2Bcu118-cp310-cp310-linux_x86_64.whl"
ALI_B="https://mirrors.aliyun.com/pytorch-wheels/cu118/"

dl() { # dl <name> <sjtu-url>
  local name="$1" url="$2"
  [ -f "$name" ] && { echo "[skip] $name present"; return 0; }
  echo "[dl] $name"
  aria2c -x 6 -s 6 -k 4M --continue --file-allocation=none -o "$name" "$url" \
    || aria2c -x 6 -s 6 -k 4M --continue --file-allocation=none -o "$name" "${ALI_B}${name}" \
    || echo "[FAIL] $name"
}

dl "torch-2.1.2+cu118-cp310-cp310-linux_x86_64.whl" "$TORCH"
dl "torchvision-0.16.2+cu118-cp310-cp310-linux_x86_64.whl" "$TORCHV"
dl "torchaudio-2.1.2+cu118-cp310-cp310-linux_x86_64.whl" "$TORCHA"

echo "=== wheels ==="; ls -lh "$W"

echo "=== pip install local wheels (deps from tuna) ==="
$PIP install \
  "$W/torch-2.1.2+cu118-cp310-cp310-linux_x86_64.whl" \
  "$W/torchvision-0.16.2+cu118-cp310-cp310-linux_x86_64.whl" \
  "$W/torchaudio-2.1.2+cu118-cp310-cp310-linux_x86_64.whl" \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | tail -20

echo "=== verify torch ==="
$HOME/miniconda3/envs/ovgraph/bin/python -c "import torch, torchvision, torchaudio; print(\"torch\", torch.__version__, \"cuda_avail\", torch.cuda.is_available(), \"cuda_ver\", torch.version.cuda)"
echo "=== torch install DONE ==="
