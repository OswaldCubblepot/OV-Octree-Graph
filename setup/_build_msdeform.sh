#!/usr/bin/env bash
# Build the mask2former MSDeformAttn CUDA op (make.sh was silent in build3).
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ovgraph
export PATH=/usr/local/cuda-11.8/bin:${PATH}
export CUDA_HOME=/usr/local/cuda-11.8
export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS=8
cd ~/OV-Octree-Graph/3rdparty/Entity/Entityv2/CropFormer/mask2former/modeling/pixel_decoder/ops
ls
echo "=== make.sh ==="
sh make.sh 2>&1 | tail -25
echo "=== result ==="
ls -la | grep -iE '\.so|build' | head -10
