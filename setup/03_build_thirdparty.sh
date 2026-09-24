#!/usr/bin/env bash
# Build the third-party packages that need compilation (CUDA extensions).
# Prerequisite: 02_install_python_deps.sh + CUDA toolkit (nvcc 11.8, installed by 00).
# Run inside WSL/Linux:
#   bash setup/03_build_thirdparty.sh
set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ovgraph

# CUDA 11.8 toolchain + gcc-11 (see ~/ovgraph_env.sh): CUDA_HOME, CC/CXX, arch list
source ~/ovgraph_env.sh

# RTX 4060 Laptop = sm_89 (Ada); tell every CUDA build about it
export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS=8

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT/3rdparty" || exit 1

# --no-build-isolation: torch/CUDA_HOME/gcc-11 are only present in the env, and the
# isolated build env would install setuptools>=81 (no pkg_resources) / no torch.

echo "[build] chamferdist"
pip install -q -e chamferdist --no-build-isolation

echo "[build] gradslam (conceptfusion branch)"
pip install -q -e gradslam --no-build-isolation

echo "[build] detectron2 v0.6"
pip install -q -e detectron2 --no-build-isolation

echo "[build] CropFormer entity_api (CUDA op)"
make -j8 -C Entity/Entityv2/CropFormer/entity_api/PythonAPI

echo "[build] CropFormer mask2former ops (MSDeformAttn)"
# make.sh runs `python setup.py build install` from its own dir, not from 3rdparty/
( cd Entity/Entityv2/CropFormer/mask2former/modeling/pixel_decoder/ops && sh make.sh )

echo "[build] our code"
cd "$REPO_ROOT"
pip install -q -e . --no-build-isolation

echo "BUILD_DONE"
python -c "import chamferdist, gradslam, detectron2, ovgraph; print('builds import OK')"
