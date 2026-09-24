#!/usr/bin/env bash
# Install python dependencies into the `ovgraph` conda env.
# Prerequisite: setup/00 conda env + torch installed (see below).
# Run inside WSL/Linux:
#   bash setup/02_install_python_deps.sh
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ovgraph

cd "$(dirname "$0")/.." || exit 1

# TUNA pip mirror (fast in China)
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# legacy requirements.txt needs an old build toolchain
echo "[pip] pin setuptools + Cython for legacy builds"
pip install -q "setuptools<70" wheel Cython==0.29.37
pip install -q PyYAML==5.4.1 --no-build-isolation

echo "[pip] install requirements.txt"
pip install -q -r requirements.txt

echo "[pip] open3d (commented out in requirements.txt but required)"
# pin 0.18.0: compatible with numpy<2 (torch 2.1.2 needs numpy 1.x)
pip install -q "open3d==0.18.0"

echo "[pip] open_clip_torch"
pip install -q open_clip_torch

echo "[pip] mmcv 2.1.0 lite (for CropFormer) — tuna PyPI"
# CropFormer only needs mmcv.list_from_file (pure-python); mmcv-lite is enough.
# openmim/mim would hit the slow openmmlab CDN, so install mmcv 2.1.0 directly.
pip install -q "mmcv==2.1.0"

echo "[pip] ovclip (OVSeg's forked CLIP, vendored)"
pip install -q -e 3rdparty/ovseg/third_party/CLIP

echo "[pip] tokenize-anything (TAP)"
pip install -q -e 3rdparty/tokenize-anything

echo "[pip] openai CLIP (for vocab features / octree demo retrieval)"
pip install -q -e 3rdparty/CLIP

echo "PYTHON_DEPS_DONE"
python -c "import torch, open3d, cv2, open_clip, clip, ovclip, mmcv, tokenize_anything; print('imports OK')"
