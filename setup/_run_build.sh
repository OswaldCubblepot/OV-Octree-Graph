#!/usr/bin/env bash
# Drive dep install + third-party CUDA builds (run inside WSL).
# Kept as a file because inline $VARS get mangled by the outer shell.
set -u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ovgraph
export PATH=/usr/local/cuda-11.8/bin:${PATH}
export CUDA_HOME=/usr/local/cuda-11.8
export TORCH_CUDA_ARCH_LIST="8.9"
export MAX_JOBS=8
cd ~/OV-Octree-Graph
{ bash setup/02_install_python_deps.sh; echo "02_EXIT=$?"; } > /tmp/02_deps.log 2>&1
{ bash setup/03_build_thirdparty.sh; echo "03_EXIT=$?"; } > /tmp/03_build.log 2>&1
echo "=== 02 tail ==="
tail -3 /tmp/02_deps.log
echo "=== 03 tail ==="
tail -5 /tmp/03_build.log
